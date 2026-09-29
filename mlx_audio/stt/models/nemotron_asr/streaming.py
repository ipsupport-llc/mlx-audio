"""Cache-aware streaming for the Nemotron FastConformer encoder.

Each conformer layer keeps an attention cache (last ``left_context`` attention-input
frames) and a causal-conv cache (last ``conv_kernel-1`` GLU-output frames);
subsampling is incremental with a small mel cache. With the window sized to the
allowed left context, no attention mask is needed, so the streamed encoder output is
numerically close to the offline ``chunked_limited`` encoder at the native chunk
size (``right_context + 1``), and produces the same greedy tokens in parity tests.
This yields the model's native O(n), no-recompute streaming.
"""

import mlx.core as mx
import mlx.nn as nn

_PRE_ENCODE_MEL_CACHE = 16  # >= causal receptive field of the 8x dw-striding stack


def _stream_block(
    block, x, pos_enc, attn_cache, conv_cache, left_cache, conv_left, pos_proj=None
):
    # half-step FFN 1
    residual = x + 0.5 * block.feed_forward1(block.norm_feed_forward1(x))

    # cache-aware self-attention: Q = chunk, K/V = [cache ++ chunk]
    xn = block.norm_self_att(residual)
    kv = xn if attn_cache is None else mx.concatenate([attn_cache, xn], axis=1)
    pos_emb = None if pos_proj is not None else pos_enc.pos_emb_for(kv.shape[1], x.dtype)
    residual = residual + block.self_attn.stream(xn, kv, pos_emb, pos_proj=pos_proj)
    attn_next = kv[:, -left_cache:] if left_cache > 0 else kv[:, :0]

    # cache-aware causal conv: prepend conv cache instead of zero-padding
    xc = block.norm_conv(residual)
    g = nn.glu(block.conv.pointwise_conv1(xc), axis=-1)  # (B, c, d)
    if conv_cache is None:
        conv_cache = mx.zeros((g.shape[0], conv_left, g.shape[2]), dtype=g.dtype)
    din = mx.concatenate([conv_cache, g], axis=1)
    dw = block.conv.depthwise_conv(din)  # valid conv -> (B, c, d)
    # -0 would keep the whole history: a kernel of 1 has no conv state.
    conv_next = din[:, -conv_left:] if conv_left > 0 else din[:, :0]
    y = block.conv.batch_norm(dw)
    y = block.conv.activation(y)
    residual = residual + block.conv.pointwise_conv2(y)

    # half-step FFN 2 + final norm
    residual = residual + 0.5 * block.feed_forward2(block.norm_feed_forward2(residual))
    return block.norm_out(residual), attn_next, conv_next


class ConformerStreamingState:
    """Reusable incremental state for a causal Nemotron FastConformer.

    The state owns the subsampling, attention, and causal-convolution caches. A
    caller may push arbitrary mel chunk sizes; complete native encoder chunks are
    returned as a list of ``(B, T, D)`` arrays.
    """

    def __init__(self, encoder, *, chunk_frames=None, att_context_size=None, compile_steady=True):
        self.encoder = encoder
        # mx.compile of the steady-state layer stack (see _encode_mel_chunk).
        self.compile_steady = compile_steady
        self._compiled_layers = None
        acs = att_context_size or encoder.args.att_context_size[0]
        self.left_cache = int(acs[0])
        self.right_context = int(acs[1])
        self.chunk_frames = chunk_frames or (self.right_context + 1)
        if self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        self.subsampling_factor = encoder.args.subsampling_factor
        self.chunk_mel = self.chunk_frames * self.subsampling_factor
        self.conv_left = encoder.args.conv_kernel_size - 1

        n = len(encoder.layers)
        # Run the encoder in its weight dtype: the mel frontend yields float32, and
        # float32 activations against bf16 weights promote every matmul to float32
        # (3.4x slower per frame on an M5, same tokens).
        out = encoder.pre_encode.out
        self.dtype = out.weight.dtype
        if not mx.issubdtype(self.dtype, mx.floating):
            # quantized: the scales carry the float type the layer computes in
            self.dtype = out.scales.dtype if hasattr(out, "scales") else mx.bfloat16
        # linear_pos(pos_emb) per layer, keyed by window length (steady state: one key)
        self._pos_proj: dict[int, list[mx.array]] = {}
        self.attn_cache = [None] * n
        self.conv_cache = [None] * n
        self.mel_cache = None
        self.emitted = 0
        self.consumed = 0
        self.pending = None
        self.closed = False

    def _append_pending(self, chunk):
        if chunk.ndim == 2:
            chunk = mx.expand_dims(chunk, 0)
        if chunk.shape[1] == 0:
            return
        self.pending = (
            chunk
            if self.pending is None
            else mx.concatenate([self.pending, chunk], axis=1)
        )

    def _encode_mel_chunk(self, m, include_boundary):
        cache_len = 0 if self.mel_cache is None else self.mel_cache.shape[1]
        win = (
            m if self.mel_cache is None else mx.concatenate([self.mel_cache, m], axis=1)
        )
        win_len = win.shape[1]
        sub = self.encoder.pre_encode(win, mx.array([win_len], dtype=mx.int32))[0]

        end = self.consumed + m.shape[1]
        base = (self.consumed - cache_len) // self.subsampling_factor
        lo = self.emitted - base
        hi = (
            sub.shape[1]
            if include_boundary
            else (end // self.subsampling_factor - base)
        )
        self.consumed = end
        self.mel_cache = win[:, -_PRE_ENCODE_MEL_CACHE:]

        if hi <= lo:
            self.emitted = base + max(lo, hi)
            return None
        self.emitted = base + hi
        h = sub[:, lo:hi]
        window = h.shape[1] + (0 if self.attn_cache[0] is None else self.attn_cache[0].shape[1])
        pos_proj = self._pos_projections(window, h.dtype)
        # Steady state (full caches, the same chunk size every step): every
        # shape repeats, so the layer stack runs as one compiled graph --
        # hundreds of small kernels per frame fused into far fewer launches.
        steady = (
            self.compile_steady
            and self.left_cache > 0
            and all(c is not None and c.shape[1] == self.left_cache for c in self.attn_cache)
            and all(c is not None for c in self.conv_cache)
        )
        if steady:
            if self._compiled_layers is None:
                self._compiled_layers = mx.compile(self._layers)
            h, attn, conv = self._compiled_layers(h, self.attn_cache, self.conv_cache, pos_proj)
            self.attn_cache, self.conv_cache = list(attn), list(conv)
            return h
        h, self.attn_cache, self.conv_cache = self._layers(h, self.attn_cache, self.conv_cache, pos_proj)
        return h

    def _layers(self, h, attn_cache, conv_cache, pos_proj):
        attn_next, conv_next = [], []
        for li, block in enumerate(self.encoder.layers):
            h, a, c = _stream_block(
                block,
                h,
                self.encoder.pos_enc,
                attn_cache[li],
                conv_cache[li],
                self.left_cache,
                self.conv_left,
                pos_proj[li],
            )
            attn_next.append(a)
            conv_next.append(c)
        return h, attn_next, conv_next

    def _pos_projections(self, window: int, dtype) -> list[mx.array]:
        cached = self._pos_proj.get(window)
        if cached is None or cached[0].dtype != dtype:
            pos_emb = self.encoder.pos_enc.pos_emb_for(window, dtype)
            cached = [block.self_attn.linear_pos(pos_emb) for block in self.encoder.layers]
            mx.eval(cached)
            # keep only the steady-state window (the largest seen) plus this one
            keep = max(self._pos_proj, default=window)
            self._pos_proj = {k: v for k, v in self._pos_proj.items() if k == keep}
            self._pos_proj[window] = cached
        return cached

    def push(self, mel, *, final=False, emit_partial=False):
        """Push mel frames and return newly encoded chunks.

        ``emit_partial`` is for protocols whose input boundary is already known to
        align with a valid causal encoder frame. It emits the preencoder's current
        right-boundary output without closing the reusable state.
        """

        if self.closed:
            raise RuntimeError("conformer streaming state is closed")
        self._append_pending(mel)
        outputs = []
        while self.pending is not None and self.pending.shape[1] > 0:
            if self.pending.shape[1] < self.chunk_mel and not (final or emit_partial):
                break

            take = min(self.chunk_mel, self.pending.shape[1])
            if (final or emit_partial) and self.pending.shape[1] <= self.chunk_mel:
                take = self.pending.shape[1]

            m = self.pending[:, :take].astype(self.dtype)
            self.pending = self.pending[:, take:]
            include_boundary = (final or emit_partial) and self.pending.shape[1] == 0
            encoded = self._encode_mel_chunk(m, include_boundary)
            if encoded is not None:
                outputs.append(encoded)
        if final:
            self.closed = True
        return outputs

    def materialize(self, *arrays):
        """Synchronize outputs and cache slices at an online iteration boundary."""

        state = [self.mel_cache]
        state.extend(value for value in self.attn_cache if value is not None)
        state.extend(value for value in self.conv_cache if value is not None)
        mx.eval(*arrays, *(value for value in state if value is not None))


def stream_encode_chunks(
    model, mel_chunks, language, chunk_frames=None, att_context_size=None
):
    """Yield post-prompt encoder frames from one or more mel chunks.

    The encoder/conv/subsampling caches persist across input mel chunks, so callers
    can keep STFT memory bounded without resetting model context at chunk boundaries.
    """
    state = ConformerStreamingState(
        model.encoder,
        chunk_frames=chunk_frames,
        att_context_size=att_context_size or model.default_att_context_size,
    )

    iterator = iter(mel_chunks)
    try:
        current = next(iterator)
    except StopIteration:
        return

    for next_chunk in iterator:
        for encoded in state.push(current):
            yield model.apply_prompt(encoded, language)
        current = next_chunk

    for encoded in state.push(current, final=True):
        yield model.apply_prompt(encoded, language)


def stream_encode(model, mel, language, chunk_frames=None, att_context_size=None):
    """Yield post-prompt encoder frames (1, c, d) per chunk, cache-aware.

    Token-equivalent to ``encoder(...)`` + ``apply_prompt(...)`` at the native
    chunk size (right_context + 1), within normal floating-point kernel drift.
    """
    yield from stream_encode_chunks(
        model,
        [mel],
        language,
        chunk_frames=chunk_frames,
        att_context_size=att_context_size,
    )
