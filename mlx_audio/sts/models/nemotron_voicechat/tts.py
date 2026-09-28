from __future__ import annotations

import math
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.lm.models import gemma3_text
from mlx_audio.lm.models.cache import KVCache, RotatingKVCache
from mlx_audio.lm.sample_utils import apply_top_p

from . import t5gemma
from .config import VoiceChatTTSConfig



class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1.0e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, inputs: mx.array) -> mx.array:
        return mx.fast.rms_norm(inputs, 1.0 + self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.down_proj(
            nn.gelu_approx(self.gate_proj(inputs)) * self.up_proj(inputs)
        )


class MLPLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.post_norm = RMSNorm(hidden_size)

    def __call__(self, inputs: mx.array) -> mx.array:
        return inputs + self.post_norm(self.mlp(self.pre_norm(inputs)))


class MoGHead(nn.Module):
    def __init__(self, config: VoiceChatTTSConfig):
        super().__init__()
        self.low_rank = config.mog_low_rank
        self.num_predictions = config.mog_num_predictions
        self.min_log_std = config.mog_min_log_std
        self.mlp_stack = [
            MLPLayer(config.hidden_size, config.mog_intermediate_size)
            for _ in range(config.mog_num_layers)
        ] + [RMSNorm(config.hidden_size)]
        self.proj_logits = nn.Linear(
            config.hidden_size, config.mog_num_predictions, bias=False
        )
        self.proj_mus = nn.Linear(
            config.hidden_size,
            config.mog_num_predictions * config.mog_low_rank,
            bias=False,
        )
        self.proj_logs = nn.Linear(config.hidden_size, 1, bias=False)
        self.proj_else = nn.Linear(config.hidden_size, config.latent_size, bias=False)
        self.low_mat = mx.zeros(
            (
                config.mog_num_predictions,
                config.latent_size,
                config.mog_low_rank,
            )
        )

    def _selected_means(self, inputs: mx.array, mixture_indices: mx.array) -> mx.array:
        """proj_mus(inputs) at the sampled mixture only: the low_rank rows of
        that mixture instead of all num_predictions x low_rank outputs (a
        1152 x 65536 mat-vec, 150 MB of bf16 read 8 times per frame, to keep
        one mixture). Same values as projecting everything and gathering."""
        batch, length, hidden = inputs.shape
        flat = mixture_indices.reshape(-1)
        proj = self.proj_mus
        if isinstance(proj, nn.QuantizedLinear):
            rows = self.num_predictions
            w = proj.weight.reshape(rows, self.low_rank, -1)[flat]
            s = proj.scales.reshape(rows, self.low_rank, -1)[flat]
            b = proj.biases.reshape(rows, self.low_rank, -1)[flat]
            w = mx.dequantize(
                w.reshape(-1, w.shape[-1]),
                s.reshape(-1, s.shape[-1]),
                b.reshape(-1, b.shape[-1]),
                group_size=proj.group_size,
                bits=proj.bits,
            ).reshape(batch * length, self.low_rank, hidden)
        else:
            w = proj.weight.reshape(self.num_predictions, self.low_rank, hidden)[flat]
        # The dtype the full projection would compute in: the inputs' and the
        # (dequantized) weights' promoted type, not the weights' alone
        # (a zero-size sum: mx.result_type needs MLX >= 0.32).
        dtype = (mx.zeros((0,), inputs.dtype) + mx.zeros((0,), w.dtype)).dtype
        x = inputs.reshape(batch * length, hidden, 1).astype(dtype)
        out = (w.astype(dtype) @ x).reshape(batch, length, self.low_rank)
        if "bias" in proj:
            out = out + proj["bias"].reshape(self.num_predictions, self.low_rank)[flat].reshape(
                batch, length, self.low_rank
            ).astype(dtype)
        return out

    def infer(
        self,
        inputs: mx.array,
        *,
        guidance_scale: float,
        top_p: float,
    ) -> tuple[mx.array, mx.array]:
        for layer in self.mlp_stack:
            inputs = layer(inputs)

        if guidance_scale > 0:
            conditional, unconditional = mx.split(inputs, 2, axis=0)
            inputs = conditional + guidance_scale * (conditional - unconditional)

        logits = self.proj_logits(inputs)
        log_probabilities = mx.log(mx.softmax(logits, axis=-1))
        if 0 < top_p < 1:
            log_probabilities = apply_top_p(log_probabilities, top_p)
        mixture_indices = mx.random.categorical(log_probabilities)

        selected_means = self._selected_means(inputs, mixture_indices)
        selected_projection = self.low_mat[mixture_indices]
        means = mx.einsum("btol,btl->bto", selected_projection, selected_means)
        residual = self.proj_else(inputs)
        log_stds = mx.maximum(self.proj_logs(inputs), self.min_log_std)
        return means * mx.exp(log_stds) + residual, log_stds


class SubwordFlagEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.is_continuation = mx.zeros((vocab_size + 1,), dtype=mx.int64)
        self.pad_tensor = mx.array(vocab_size, dtype=mx.int64)
        self.cont_emb = nn.Embedding(2, hidden_size)

    def __call__(self, inputs: mx.array, token_ids: mx.array) -> mx.array:
        safe_ids = mx.where(
            token_ids >= self.is_continuation.shape[0] - 1,
            self.pad_tensor,
            token_ids,
        )
        return inputs + self.cont_emb(self.is_continuation[safe_ids])


class BOSEOSEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.special_flags = mx.zeros((vocab_size,), dtype=mx.int64)
        self.pad_tensor = mx.array(vocab_size - 1, dtype=mx.int64)
        self.special_emb = nn.Embedding(3, hidden_size)

    def __call__(self, inputs: mx.array, token_ids: mx.array) -> mx.array:
        safe_ids = mx.where(
            token_ids >= self.special_flags.shape[0], self.pad_tensor, token_ids
        )
        return inputs + self.special_emb(self.special_flags[safe_ids])


class CharAwareSubwordEncoder(nn.Module):
    def __init__(self, config: VoiceChatTTSConfig):
        super().__init__()
        self.backbone = t5gemma.Model(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )
        self.embed_tokens = nn.Embedding(config.char_vocab_size + 1, config.hidden_size)
        self.proj_embedding = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.subword_flag_emb = SubwordFlagEmbedding(
            config.text_vocab_size, config.hidden_size
        )
        self.bos_eos_emb = BOSEOSEmbedding(config.text_vocab_size, config.hidden_size)
        self.char_padding_idx = config.char_vocab_size
        self.subword_id_to_char_ids: dict[int, tuple[int, ...]] = {}

    def set_tokenizer(self, tokenizer) -> None:
        vocabulary = tokenizer.get_vocab()
        single_characters = {
            token: token_id for token, token_id in vocabulary.items() if len(token) == 1
        }
        characters = sorted(single_characters, key=single_characters.get)
        char_vocabulary = {
            character: index for index, character in enumerate(characters)
        }
        if len(char_vocabulary) != self.char_padding_idx:
            raise ValueError(
                "Tokenizer character vocabulary does not match VoiceChat weights: "
                f"expected {self.char_padding_idx}, got {len(char_vocabulary)}"
            )
        self.subword_id_to_char_ids = {
            token_id: tuple(
                char_vocabulary[character]
                for character in token
                if character in char_vocabulary
            )
            for token, token_id in vocabulary.items()
        }
        self.subword_id_to_char_ids = {
            token_id: character_ids
            for token_id, character_ids in self.subword_id_to_char_ids.items()
            if character_ids
        }
        tokens = {token_id: token for token, token_id in vocabulary.items()}
        continuation_flags = [0] * self.subword_flag_emb.is_continuation.shape[0]
        for token_id in range(len(continuation_flags) - 1):
            token = tokens.get(token_id, "")
            continuation_flags[token_id] = int(
                bool(token) and not token.startswith(("Ġ", "▁", "<"))
            )
        self.subword_flag_emb.is_continuation = mx.array(
            continuation_flags, dtype=mx.int64
        )

        special_flags = [0] * self.bos_eos_emb.special_flags.shape[0]
        bos_id = getattr(tokenizer, "bos_token_id", vocabulary.get("<s>"))
        eos_id = getattr(tokenizer, "eos_token_id", vocabulary.get("</s>"))
        if bos_id is not None and bos_id < len(special_flags):
            special_flags[bos_id] = 1
        if eos_id is not None and eos_id < len(special_flags):
            special_flags[eos_id] = 2
        self.bos_eos_emb.special_flags = mx.array(special_flags, dtype=mx.int64)

    def __call__(
        self, token_ids: mx.array, subword_mask: mx.array | None = None
    ) -> mx.array:
        if not self.subword_id_to_char_ids:
            raise RuntimeError("VoiceChat tokenizer has not been initialized")
        if subword_mask is None:
            subword_mask = mx.ones(token_ids.shape, dtype=mx.bool_)

        positions: list[tuple[int, int]] = []
        character_sequences: list[tuple[int, ...]] = []
        mask_values = subword_mask.tolist()
        token_values = token_ids.tolist()
        for batch_index, row in enumerate(mask_values):
            for time_index, enabled in enumerate(row):
                if enabled:
                    positions.append((batch_index, time_index))
                    character_sequences.append(
                        self.subword_id_to_char_ids.get(
                            int(token_values[batch_index][time_index]), ()
                        )
                    )

        output = mx.zeros(
            token_ids.shape + (self.proj_embedding.weight.shape[0],),
            dtype=self.proj_embedding.weight.dtype,
        )
        if positions:
            max_length = max(max(map(len, character_sequences)), 1)
            character_ids = mx.full(
                (len(positions), max_length),
                self.char_padding_idx,
                dtype=mx.int32,
            )
            character_mask = mx.zeros((len(positions), max_length), dtype=mx.bool_)
            for index, sequence in enumerate(character_sequences):
                if sequence:
                    character_ids[index, : len(sequence)] = mx.array(sequence)
                    character_mask[index, : len(sequence)] = True

            hidden = self.backbone(
                self.embed_tokens(character_ids), attention_mask=character_mask
            )
            lengths = mx.maximum(character_mask.sum(axis=1, keepdims=True), 1)
            pooled = (hidden * character_mask[..., None]).sum(axis=1) / lengths
            encoded = self.proj_embedding(pooled)
            for index, (batch_index, time_index) in enumerate(positions):
                output[batch_index, time_index] = encoded[index]

        output = self.subword_flag_emb(output, token_ids)
        return self.bos_eos_emb(output, token_ids)


class GatedProjectedSumRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, num_quantizers: int):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.audio_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.gate = mx.zeros((hidden_size,), dtype=mx.float32)
        self.residual_scale = mx.array(0.5, dtype=mx.float32)
        self.final_norm = RMSNorm(hidden_size)

    def __call__(self, audio: mx.array, text: mx.array) -> mx.array:
        audio = self.audio_proj(audio / self.num_quantizers)
        text = self.text_proj(text)
        gate = mx.sigmoid(self.gate).astype(audio.dtype)
        residual_scale = mx.sigmoid(self.residual_scale).astype(audio.dtype)
        return self.final_norm(residual_scale * (gate * audio + (1 - gate) * text))


class _BufferLayerCache:
    """One layer's view of a StaticBackboneCache inside the compiled step:
    what gemma3_text's Attention asks of a cache (``offset`` for RoPE,
    ``update_and_fetch``), writing the frame's K/V at ``offset`` into the
    fixed buffers and handing back all of them (masked by the caller)."""

    def __init__(self, keys: mx.array, values: mx.array, offset: mx.array):
        self.keys, self.values, self.offset = keys, values, offset

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        start = mx.reshape(self.offset, (1,))
        self.keys = mx.slice_update(self.keys, keys.astype(self.keys.dtype), start, axes=(2,))
        self.values = mx.slice_update(self.values, values.astype(self.values.dtype), start, axes=(2,))
        return self.keys, self.values


class StaticBackboneCache:
    """The TTS backbone's K/V in buffers of a fixed length (``capacity``,
    grown by ``chunk`` frames when full), so one frame's step has fixed
    shapes and runs as one compiled graph instead of ~28 layers of small
    kernels. Entries past ``offset`` are masked out; the sliding layers
    also mask what their RotatingKVCache would have dropped, so the step is
    exact at any length (the buffers keep growing, as the global layers'
    KVCache does anyway)."""

    def __init__(self, keys: list, values: list, offset: int, chunk: int):
        self.keys, self.values, self.offset, self.chunk = keys, values, offset, chunk

    @property
    def capacity(self) -> int:
        return self.keys[0].shape[2]

    @classmethod
    def from_caches(cls, caches: list, chunk: int) -> "StaticBackboneCache | None":
        """From the prefilled KVCache/RotatingKVCache list, while every
        layer still holds its whole history in order (None otherwise)."""
        offset = caches[0].offset
        keys, values = [], []
        for cache in caches:
            if cache.offset != offset or cache.keys is None or cache.keys.shape[2] < offset:
                return None
            if isinstance(cache, RotatingKVCache) and offset >= cache.max_size:
                return None
            keys.append(cache.keys[..., :offset, :])
            values.append(cache.values[..., :offset, :])
        static = cls(keys, values, offset, chunk)
        static._grow(offset + 1)
        return static

    def _grow(self, needed: int) -> None:
        if needed <= self.capacity:
            return
        capacity = -(-needed // self.chunk) * self.chunk
        pad = ((0, 0), (0, 0), (0, capacity - self.capacity), (0, 0))
        self.keys = [mx.pad(k, pad) for k in self.keys]
        self.values = [mx.pad(v, pad) for v in self.values]


class EARTTSModel(nn.Module):
    # Frames the backbone's static K/V buffers grow by (one compiled graph
    # per capacity): 512 frames = 41 s of speech, at most ~0.3 ms of masked
    # attention a step. 0: the eager backbone with growing caches.
    backbone_buffer_chunk = 512

    def __init__(self, config: VoiceChatTTSConfig):
        super().__init__()
        self.config = config
        backbone_args = gemma3_text.ModelArgs(
            model_type="gemma3_text",
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            sliding_window=config.sliding_window,
            sliding_window_pattern=min(6, config.num_hidden_layers),
            max_position_embeddings=131_072,
            vocab_size=1,
        )
        self.backbone = gemma3_text.Gemma3Model(backbone_args)
        self.backbone.pop("embed_tokens")
        self.bos_emb = mx.zeros((config.hidden_size,))
        self.null_emb = mx.zeros((config.hidden_size,))
        self.embed_code = nn.Linear(config.latent_size, config.hidden_size, bias=False)
        self.embed_subword = CharAwareSubwordEncoder(config)
        self.gated_fusion_audio_text = GatedProjectedSumRMSNorm(
            config.hidden_size, config.num_quantizers
        )
        self.mog_head = MoGHead(config)
        self.rvq_embs = mx.zeros(
            (config.num_quantizers, config.codebook_size, config.latent_size)
        )

    def make_cache(self):
        caches = []
        pattern = self.backbone.sliding_window_pattern
        for index in range(self.config.num_hidden_layers):
            if index % pattern == pattern - 1:
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.config.sliding_window))
        return caches

    def depthsum_embedding(self, codes: mx.array) -> mx.array:
        embeddings = mx.pad(self.rvq_embs, ((0, 0), (0, 1), (0, 0)))
        output = mx.zeros(
            codes.shape[:2] + (self.config.latent_size,),
            dtype=embeddings.dtype,
        )
        for index in range(codes.shape[-1]):
            output = output + embeddings[index][codes[..., index]]
        return output

    def _conditioning(
        self,
        subword_ids: mx.array,
        subword_mask: mx.array,
        guidance_enabled: bool,
    ) -> mx.array:
        conditioning = self.embed_subword(subword_ids, subword_mask)
        if guidance_enabled:
            unconditional = mx.broadcast_to(self.null_emb, conditioning.shape)
            conditioning = mx.concatenate([conditioning, unconditional], axis=0)
        return conditioning

    def warmup(
        self,
        codes: mx.array,
        audio_mask: mx.array,
        subword_ids: mx.array,
        subword_mask: mx.array,
        audio_prompt_latent: mx.array,
        *,
        guidance_enabled: bool,
    ):
        shifted_codes = mx.pad(codes[:, :-1], ((0, 0), (1, 0), (0, 0)))
        code_embeddings = self.embed_code(self.depthsum_embedding(shifted_codes))
        bos_mask = audio_mask & ~mx.pad(audio_mask[:, :-1], ((0, 0), (1, 0)))
        pre_bos_mask = mx.cumsum(bos_mask, axis=1) == 0
        code_embeddings = mx.where(
            pre_bos_mask[..., None], audio_prompt_latent, code_embeddings
        )
        code_embeddings = code_embeddings + bos_mask[..., None] * self.bos_emb
        conditioning = self._conditioning(subword_ids, subword_mask, guidance_enabled)
        if guidance_enabled:
            code_embeddings = mx.concatenate([code_embeddings] * 2, axis=0)
        inputs = self.gated_fusion_audio_text(code_embeddings, conditioning)
        cache = self.make_cache()
        self.backbone(None, cache=cache, input_embeddings=inputs)
        return codes[:, -1:], cache

    def generate_step(
        self,
        current_subword_id: mx.array,
        previous_codes: mx.array,
        cache,
        *,
        text_eos_id: int,
        silence_codes: mx.array,
        guidance_enabled: bool,
    ) -> tuple[mx.array, list]:
        previous_codes = mx.where(
            (current_subword_id == text_eos_id)[..., None],
            mx.broadcast_to(silence_codes, previous_codes.shape),
            previous_codes,
        )
        code_embeddings = self.embed_code(self.depthsum_embedding(previous_codes))
        subword_mask = mx.ones(current_subword_id.shape, dtype=mx.bool_)
        conditioning = self._conditioning(
            current_subword_id, subword_mask, guidance_enabled
        )
        if guidance_enabled:
            code_embeddings = mx.concatenate([code_embeddings] * 2, axis=0)
        inputs = self.gated_fusion_audio_text(code_embeddings, conditioning)
        hidden, cache = self._backbone_step(inputs, cache)
        codes = self._generate_codes(hidden, guidance_enabled=guidance_enabled)
        return codes, cache

    def _backbone_step(self, inputs: mx.array, cache):
        """One frame through the backbone: compiled over static K/V buffers
        (from the first step after warmup), or eager with the growing caches
        when ``backbone_buffer_chunk`` is 0 or the prefill already rotated."""
        if self.backbone_buffer_chunk > 0 and isinstance(cache, list):
            cache = StaticBackboneCache.from_caches(cache, self.backbone_buffer_chunk) or cache
        if not isinstance(cache, StaticBackboneCache):
            return self.backbone(None, cache=cache, input_embeddings=inputs), cache
        cache._grow(cache.offset + 1)
        fn = self._compiled_backbone(cache.capacity, inputs.shape, inputs.dtype, cache.keys[0].dtype)
        hidden, *state = fn(inputs, mx.array(cache.offset, dtype=mx.int32), *cache.keys, *cache.values)
        layers = len(cache.keys)
        cache.keys, cache.values = list(state[:layers]), list(state[layers:])
        cache.offset += 1
        return hidden, cache

    def _compiled_backbone(self, capacity: int, shape, dtype, kv_dtype):
        compiled = self.__dict__.setdefault("_compiled_backbone_fns", SimpleNamespace(fns={}))
        key = (capacity, tuple(shape), dtype, kv_dtype)
        fn = compiled.fns.get(key)
        if fn is None:
            fn = mx.compile(self._backbone_step_static)
            compiled.fns[key] = fn
        return fn

    def _backbone_step_static(self, inputs: mx.array, offset: mx.array, *buffers):
        backbone = self.backbone
        layers = len(backbone.layers)
        keys, values = buffers[:layers], buffers[layers:]
        positions = mx.arange(keys[0].shape[2])
        seen = positions <= offset
        global_mask = seen[None, None, None, :]
        local_mask = (seen & (positions > offset - self.config.sliding_window))[None, None, None, :]
        pattern = backbone.sliding_window_pattern
        h = inputs
        new_keys, new_values = [], []
        for index, layer in enumerate(backbone.layers):
            view = _BufferLayerCache(keys[index], values[index], offset)
            mask = global_mask if index % pattern == pattern - 1 else local_mask
            h = layer(h, mask, view)
            new_keys.append(view.keys)
            new_values.append(view.values)
        return (backbone.norm(h), *new_keys, *new_values)

    def _generate_codes(self, hidden: mx.array, *, guidance_enabled: bool) -> mx.array:
        """One frame's codes (num_iterations MoG steps over the RVQ levels:
        hundreds of small kernels) as one compiled graph per guidance mode,
        the random state threaded through so the sampled codes are the same."""
        # Held by the model itself (a plain attribute, outside the parameter
        # tree), so the graphs go when the model does.
        compiled = self.__dict__.setdefault("_compiled_codes", SimpleNamespace(fns={}))
        fn = compiled.fns.get(guidance_enabled)
        if fn is None:
            fn = mx.compile(
                lambda h: self._generate_codes_eager(h, guidance_enabled=guidance_enabled),
                inputs=[mx.random.state],
                outputs=[mx.random.state],
            )
            compiled.fns[guidance_enabled] = fn
        return fn(hidden)

    def _generate_codes_eager(self, hidden: mx.array, *, guidance_enabled: bool) -> mx.array:
        if guidance_enabled:
            conditional, _ = mx.split(hidden, 2, axis=0)
        else:
            conditional = hidden
        batch, length, _ = conditional.shape
        codes = mx.full(
            (batch, length, self.config.num_quantizers),
            self.config.codebook_size,
            dtype=mx.int32,
        )
        rates = [
            index / self.config.num_iterations
            for index in range(self.config.num_iterations)
        ]
        masked = [
            math.ceil(
                ((1 - rate**self.config.exponent) ** (1 / self.config.exponent))
                * self.config.num_quantizers
            )
            for rate in rates
        ] + [0]

        completed = 0
        for current, following in zip(masked, masked[1:]):
            count = current - following
            if count <= 0:
                continue
            embedded = self.embed_code(self.depthsum_embedding(codes))
            if guidance_enabled:
                conditional_hidden, unconditional_hidden = mx.split(hidden, 2, axis=0)
                mog_inputs = mx.concatenate(
                    [
                        embedded + conditional_hidden,
                        embedded + unconditional_hidden,
                    ],
                    axis=0,
                )
            else:
                mog_inputs = embedded + hidden
            means, log_stds = self.mog_head.infer(
                mog_inputs,
                guidance_scale=(
                    self.config.guidance_scale if guidance_enabled else 0.0
                ),
                top_p=self.config.top_p,
            )
            residual = (
                means
                + mx.exp(log_stds)
                * mx.random.normal(means.shape)
                * self.config.noise_scale
            )
            for quantizer in range(completed, completed + count):
                codebook = self.rvq_embs[quantizer]
                distances = mx.sum(codebook * codebook, axis=-1)[None, None, :] - 2 * (
                    residual @ codebook.T
                )
                indices = mx.argmin(distances, axis=-1)
                codes[..., quantizer] = indices
                residual = residual - codebook[indices]
            completed += count
        return codes
