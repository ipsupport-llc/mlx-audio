import mlx.core as mx

from mlx_audio.lm.models import gemma3_text
from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.sts.models.nemotron_voicechat import Model, ModelConfig
from mlx_audio.sts.models.nemotron_voicechat.config import _llm_config
from mlx_audio.sts.models.nemotron_voicechat.convert import _quantize
from mlx_audio.sts.utils import infer_model_type_from_config


def mini_config():
    llm = {
        "model_type": "nemotron_h",
        "vocab_size": 64,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 3,
        "max_position_embeddings": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "attention_bias": False,
        "mamba_num_heads": 2,
        "mamba_head_dim": 4,
        "mamba_proj_bias": False,
        "ssm_state_size": 8,
        "conv_kernel": 4,
        "n_groups": 1,
        "mlp_bias": False,
        "layer_norm_epsilon": 1e-5,
        "use_bias": False,
        "use_conv_bias": True,
        "hybrid_override_pattern": ["M", "*", "-"],
    }
    return {
        "data": {
            "source_sample_rate": 16_000,
            "target_sample_rate": 22_050,
            "frame_length": 0.001,
        },
        "_rnnt_merge_info": {
            "decoder_config": {
                "vocab_size": 8,
                "blank_as_pad": True,
                "prednet": {"pred_hidden": 8, "pred_rnn_layers": 1},
            },
            "joint_config": {
                "num_classes": 8,
                "vocabulary": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "jointnet": {
                    "joint_hidden": 8,
                    "activation": "relu",
                    "encoder_hidden": 8,
                    "pred_hidden": 8,
                },
            },
        },
        "mlx_audio": {"llm_config": llm, "char_vocab_size": 4},
        "model": {
            "inference_speaker_name": "Aria",
            "stt": {
                "model": {
                    "pretrained_llm": "local",
                    "perception": {
                        "output_dim": 8,
                        "preprocessor": {
                            "features": 8,
                            "n_fft": 16,
                            "window_size": 0.001,
                            "window_stride": 0.0005,
                        },
                        "encoder": {
                            "feat_in": 8,
                            "n_layers": 1,
                            "d_model": 8,
                            "n_heads": 2,
                            "ff_expansion_factor": 2,
                            "subsampling_factor": 2,
                            "subsampling_conv_channels": 4,
                            "conv_kernel_size": 3,
                            "att_context_size": [4, 0],
                        },
                    },
                }
            },
            "speech_generation": {
                "data": {
                    "audio_prompt_duration": 0.002,
                    "frame_length": 0.001,
                },
                "model": {
                    "codec_config": {
                        "base_hidden_size": 8,
                        "channel_mult": [1],
                        "rates": [2],
                        "num_blocks": 1,
                        "kernel_size": 3,
                        "latent_size": 4,
                        "n_fft": 4,
                        "hop_length": 2,
                        "num_quantizers": 2,
                        "codebook_size": 8,
                    },
                    "tts_config": {
                        "backbone_config": {
                            "hidden_size": 8,
                            "intermediate_size": 16,
                            "num_hidden_layers": 2,
                            "num_attention_heads": 2,
                            "num_key_value_heads": 2,
                            "head_dim": 4,
                            "sliding_window": 16,
                        },
                        "latent_size": 4,
                        "codebook_size": 8,
                        "num_quantizers": 2,
                        "mog_head_config": {
                            "intermediate_size": 16,
                            "num_layers": 1,
                            "num_predictions": 8,
                            "low_rank": 2,
                        },
                    },
                },
            },
        },
    }


class MiniTokenizer:
    bos_token_id = 5
    eos_token_id = 6
    pad_token_id = 12

    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return [0]

    def decode(self, tokens, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(ord("a") + int(token) % 26) for token in tokens)

    def get_vocab(self):
        return {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "ab": 4,
            "<s>": 5,
            "</s>": 6,
            "<SPECIAL_11>": 11,
            "<SPECIAL_12>": 12,
        }


def gemma_args(num_hidden_layers=1):
    return gemma3_text.ModelArgs(
        model_type="gemma3_text",
        hidden_size=8,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=8,
        sliding_window=16,
        sliding_window_pattern=1,
    )


def test_gemma3_preserves_external_embedding_scale():
    model = gemma3_text.Gemma3Model(gemma_args())
    embeddings = mx.random.normal((2, 3, 8))

    output = model(None, input_embeddings=embeddings)
    expected = embeddings
    mask = create_attention_mask(expected)
    for layer in model.layers:
        expected = layer(expected, mask)
    expected = model.norm(expected)

    assert mx.allclose(output, expected).item()


def test_gemma3_batched_cache_matches_full_forward():
    model = gemma3_text.Gemma3Model(gemma_args())
    embeddings = mx.random.normal((2, 4, 8))
    full_output = model(None, input_embeddings=embeddings)
    cache = [KVCache()]
    prefix_output = model(None, cache=cache, input_embeddings=embeddings[:, :3])
    mx.eval(prefix_output)
    cached_output = model(None, cache=cache, input_embeddings=embeddings[:, 3:])
    mx.eval(full_output, cached_output)

    assert mx.allclose(cached_output, full_output[:, 3:], atol=1e-2, rtol=1e-2).item()


def test_official_config_is_detected():
    assert (
        infer_model_type_from_config({"model": {"stt": {}, "speech_generation": {}}})
        == "nemotron_voicechat"
    )


def test_llm_config_does_not_execute_remote_code(monkeypatch):
    expected = {"model_type": "nemotron_h", "hidden_size": 8}

    monkeypatch.setattr(
        "transformers.PretrainedConfig.get_config_dict",
        lambda model_name: (expected, {"model_name": model_name}),
    )

    assert _llm_config({}, "nvidia/test-model") == expected


def test_streaming_session_yields_aligned_outputs():
    model = Model(ModelConfig.from_dict(mini_config()))
    model.tokenizer = MiniTokenizer()
    stream = model.create_duplex_session(
        system_prompt="",
        use_perception_cache=True,
    )
    stream._rnnt.step = lambda _encoded: ("hello", "hello")
    events = stream.push_audio(
        mx.zeros((stream.frame_samples,), dtype=mx.float32),
        sample_rate=16_000,
    )
    audio_event = next(event for event in events if event.kind == "audio")
    mx.eval(audio_event.samples)

    assert audio_event.audio_codes.shape == (2,)
    assert audio_event.samples.shape == (4,)
    assert audio_event.sample_rate == 22_050
    assert any(
        event.kind == "user_transcript_delta" and event.delta == "hello"
        for event in events
    )
    assert len(stream._text_tokens) == 1
    assert len(stream._function_tokens) == 1
    assert stream.flush()[-1].kind == "done"
    assert (
        model.tts_model.tts_model.embed_subword.subword_flag_emb.is_continuation[
            4
        ].item()
        == 1
    )
    assert (
        model.tts_model.tts_model.embed_subword.bos_eos_emb.special_flags[5].item() == 1
    )
    assert (
        model.tts_model.tts_model.embed_subword.bos_eos_emb.special_flags[6].item() == 2
    )


def test_streaming_session_buffers_partial_frames_and_cancels():
    model = Model(ModelConfig.from_dict(mini_config()))
    model.tokenizer = MiniTokenizer()
    stream = model.create_duplex_session(
        system_prompt="",
        use_perception_cache=False,
    )
    half = stream.frame_samples // 2

    assert stream.push_audio(mx.zeros((half,)), sample_rate=16_000) == []
    events = stream.push_audio(
        mx.zeros((stream.frame_samples - half,)), sample_rate=16_000
    )

    assert any(event.kind == "audio" for event in events)
    assert stream.cancel()[-1].kind == "cancelled"
    assert stream.closed


def test_sanitize_convolution_layouts():
    model = Model(ModelConfig.from_dict(mini_config()))
    weights = {
        "stt_model.perception.encoder.pre_encode.conv.0.weight": mx.zeros((4, 1, 3, 3)),
        "stt_model.llm.layers.0.mixer.conv1d.weight": mx.zeros((12, 1, 4)),
        "tts_model.tts_model.rvq_embs": mx.zeros((2, 8, 4)),
        "tts_model.tts_model.audio_prompt_projection_W": mx.zeros((8, 8)),
        "stt_model.rnnt_decoder.prediction.embed.weight": mx.zeros((8, 4)),
        "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.weight_ih_l0": mx.zeros(
            (32, 8)
        ),
        "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.weight_hh_l0": mx.zeros(
            (32, 8)
        ),
        "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.bias_ih_l0": mx.ones((32,)),
        "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.bias_hh_l0": mx.ones((32,)),
    }
    converted = model.sanitize(weights)

    assert converted["stt_model.perception.encoder.pre_encode.conv.0.weight"].shape == (
        4,
        3,
        3,
        1,
    )
    assert converted["stt_model.llm.layers.0.mixer.conv1d.weight"].shape == (12, 4, 1)
    assert converted["tts_model.tts_model.rvq_embs"].shape == (2, 8, 4)
    assert "tts_model.tts_model.audio_prompt_projection_W" not in converted
    assert "stt_model.rnnt_decoder.prediction.embed.weight" in converted
    assert (
        converted["stt_model.rnnt_decoder.prediction.dec_rnn.lstm.0.bias"].tolist()
        == [2.0] * 32
    )


def test_quantize_only_supported_linear_weights():
    weights = {
        "stt_model.llm.linear.weight": mx.zeros((8, 64)),
        "stt_model.llm.linear.bias": mx.zeros((8,)),
        "stt_model.perception.linear.weight": mx.zeros((8, 64)),
        "stt_model.llm.conv.weight": mx.zeros((8, 3, 4)),
    }
    quantized = _quantize(weights, group_size=64, bits=4)

    assert "stt_model.llm.linear.scales" in quantized
    assert "stt_model.llm.linear.biases" in quantized
    assert quantized["stt_model.llm.linear.bias"].shape == (8,)
    assert quantized["stt_model.perception.linear.weight"].shape == (8, 64)
    assert quantized["stt_model.llm.conv.weight"].shape == (8, 3, 4)


def mini_runtime_config():
    """The mini model in the mlx-vlm (``mlx_runtime_config_version: 2``) layout
    used by the mlx-community checkpoints."""
    nemo = mini_config()
    stt = nemo["model"]["stt"]["model"]["perception"]
    speech = nemo["model"]["speech_generation"]["model"]
    backbone = speech["tts_config"]["backbone_config"]
    mog = speech["tts_config"]["mog_head_config"]
    codec = speech["codec_config"]
    rnnt = nemo["_rnnt_merge_info"]
    return {
        "mlx_runtime_config_version": 2,
        "model_type": "nemotron_voicechat",
        "architectures": ["NemotronVoiceChatForConditionalGeneration"],
        "text_config": {
            **nemo["mlx_audio"]["llm_config"],
            "hybrid_override_pattern": "M*-",
        },
        "audio_config": {
            "preprocessor": {"sample_rate": 16_000, **stt["preprocessor"]},
            "encoder": {
                **stt["encoder"],
                "att_context_size": [stt["encoder"]["att_context_size"]],
            },
            "decoder": {
                **rnnt["decoder_config"]["prednet"],
                "vocab_size": 8,
                "blank_as_pad": True,
            },
            "joint": {**rnnt["joint_config"]["jointnet"], "num_classes": 8},
            "output_dim": 8,
            "max_symbols": 10,
        },
        "tts_config": {
            **backbone,
            "latent_size": 4,
            "codebook_size": 8,
            "num_quantizers": 2,
            "exponent": 3.0,
            "char_vocab_size": 4,
            "mog_head": {**mog, "min_log_std": -4.0},
            "guidance_scale": 0.2,
            "top_p": 0.95,
            "noise_scale": 0.001,
            "audio_prompt_duration": 0.002,
        },
        "codec_config": {
            "sample_rate": 22_050,
            "base_channels": codec["base_hidden_size"],
            "channel_multipliers": codec["channel_mult"],
            "downsample_rates": codec["rates"],
            "blocks_per_stage": codec["num_blocks"],
            "block_kernel_size": codec["kernel_size"],
            "latent_dim": codec["latent_size"],
            "n_fft": codec["n_fft"],
            "hop_length": codec["hop_length"],
            "num_quantizers": codec["num_quantizers"],
            "codebook_size": codec["codebook_size"],
        },
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 12,
        "silence_token_id": 11,
        "rnnt_blank_id": 8,
        "input_sample_rate": 16_000,
        "output_sample_rate": 22_050,
        "frame_duration": 0.001,
        "function_channel_weight": 2.0,
        "speaker": "Aria",
        "rnnt_vocabulary": rnnt["joint_config"]["vocabulary"],
        "quantization": {"group_size": 64, "bits": 4},
    }


def test_runtime_config_is_detected():
    from mlx_audio.sts.models.nemotron_voicechat.config import is_runtime_config

    runtime = mini_runtime_config()
    assert is_runtime_config(runtime)
    assert not is_runtime_config(mini_config())
    assert infer_model_type_from_config(runtime) == "nemotron_voicechat"


def test_runtime_config_matches_nemo_config():
    import dataclasses

    nemo = ModelConfig.from_dict(mini_config()).config
    runtime = ModelConfig.from_dict(mini_runtime_config()).config

    for field in dataclasses.fields(runtime):
        if field.name in {"pretrained_llm", "llm"}:
            continue
        assert getattr(runtime, field.name) == getattr(nemo, field.name), field.name
    assert list(runtime.llm.hybrid_override_pattern) == nemo.llm.hybrid_override_pattern
    assert runtime.llm.num_hidden_layers == nemo.llm.num_hidden_layers
    assert runtime.llm.hidden_size == nemo.llm.hidden_size
    assert runtime.encoder.att_context_size == [[4, 0]]
    assert runtime.audio_prompt_frames == 2
    assert runtime.rnnt_blank_id == 8
    assert runtime.prepared_weights is False


def test_runtime_config_never_fetches_remote_llm_config(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("runtime configs must not fetch the base LLM config")

    monkeypatch.setattr("transformers.PretrainedConfig.get_config_dict", fail)
    config = ModelConfig.from_dict(mini_runtime_config()).config
    assert config.llm.vocab_size == 64


def test_runtime_config_real_layout_values():
    # Values as published in mlx-community/NemotronLabs-VoiceChat-11B-4bit.
    runtime = mini_runtime_config()
    runtime["audio_config"]["encoder"]["att_context_size"] = [[70, 0]]
    runtime["frame_duration"] = 0.08
    runtime["tts_config"]["audio_prompt_duration"] = 3.0
    runtime["tts_config"].pop("char_vocab_size")
    config = ModelConfig.from_dict(runtime).config

    assert config.encoder.att_context_size == [[70, 0]]
    assert config.audio_prompt_frames == 37
    assert config.tts.char_vocab_size == 256
    assert config.tts.num_iterations == 8
    assert config.codec.sample_rate == 22_050


def test_runtime_config_model_builds_and_streams():
    model = Model(ModelConfig.from_dict(mini_runtime_config()))
    model.tokenizer = MiniTokenizer()
    stream = model.create_duplex_session(system_prompt="")
    events = stream.push_audio(
        mx.zeros((stream.frame_samples,), dtype=mx.float32), sample_rate=16_000
    )
    assert any(event.kind == "audio" for event in events)


def test_tokenizer_source_prefers_local_files(tmp_path):
    from mlx_audio.sts.models.nemotron_voicechat.model import _tokenizer_source

    assert _tokenizer_source(tmp_path, "nvidia/base") == "nvidia/base"
    assert _tokenizer_source(None, "nvidia/base") == "nvidia/base"
    (tmp_path / "tokenizer.json").write_text("{}")
    assert _tokenizer_source(tmp_path, "nvidia/base") == str(tmp_path)


def test_post_load_hook_loads_tokenizer_from_model_folder(tmp_path, monkeypatch):
    (tmp_path / "tokenizer.json").write_text("{}")
    calls = []

    class FakeTokenizer(MiniTokenizer):
        pass

    def from_pretrained(source, *args, **kwargs):
        calls.append(source)
        return FakeTokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", from_pretrained)
    model = Model(ModelConfig.from_dict(mini_runtime_config()))
    Model.post_load_hook(model, tmp_path)

    assert calls == [str(tmp_path)]
    assert isinstance(model.tokenizer, FakeTokenizer)


def _full_selected_means(proj, inputs, idx, num_predictions, low_rank):
    b, l, _ = inputs.shape
    full = proj(inputs).reshape(b, l, num_predictions, low_rank)
    return mx.take_along_axis(full, idx[..., None, None], axis=2).squeeze(2)


def test_selected_mog_means_match_the_full_projection():
    from types import SimpleNamespace

    import mlx.nn as nn

    from mlx_audio.sts.models.nemotron_voicechat.tts import MoGHead

    # CPU: exact fp32 GEMM (Metal's fp32 matmul rounds like tf32), so any
    # indexing/layout mistake shows up instead of hiding in the tolerance.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        _check_selected_means(MoGHead, SimpleNamespace, nn)
    finally:
        mx.set_default_device(previous)


def _check_selected_means(MoGHead, SimpleNamespace, nn):
    mx.random.seed(3)
    hidden, predictions, rank = 64, 16, 8
    for bias in (False, True):
        dense = nn.Linear(hidden, predictions * rank, bias=bias)
        quantized = nn.QuantizedLinear.from_linear(dense, group_size=32, bits=4)
        for proj in (dense, quantized):
            for dtype in (mx.float32, mx.bfloat16):
                inputs = mx.random.normal((2, 3, hidden)).astype(dtype)
                idx = mx.random.randint(0, predictions, (2, 3))
                head = SimpleNamespace(proj_mus=proj, num_predictions=predictions, low_rank=rank)
                got = MoGHead._selected_means(head, inputs, idx)
                want = _full_selected_means(proj, inputs, idx, predictions, rank)
                assert got.dtype == want.dtype, (bias, type(proj).__name__, dtype)
                tol = 1e-5 if want.dtype == mx.float32 else 5e-2
                assert mx.allclose(got, want, atol=tol, rtol=tol).item(), (bias, type(proj).__name__, dtype)


def test_compiled_conformer_steady_state_matches_eager():
    from mlx_audio.stt.models.nemotron_asr.streaming import ConformerStreamingState

    model = Model(ModelConfig.from_dict(mini_config()))
    encoder = model.stt_model.perception.encoder
    mx.random.seed(5)
    features = model.config.preprocessor.features if hasattr(model.config, "preprocessor") else 8
    mels = [mx.random.normal((1, 8, features)) for _ in range(40)]  # far past the 4-frame context

    def run(compiled):
        state = ConformerStreamingState(encoder, chunk_frames=1, att_context_size=[4, 0], compile_steady=compiled)
        out = []
        for m in mels:
            out += state.push(m, emit_partial=True)
        return mx.concatenate(out, axis=1)

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)  # exact fp32; Metal's fp32 matmul rounds like tf32
    try:
        eager, compiled = run(False), run(True)
    finally:
        mx.set_default_device(previous)
    assert eager.shape == compiled.shape
    assert mx.abs(eager - compiled).max().item() < 1e-5


def test_compiled_codes_live_on_the_model():
    import gc
    import weakref

    model = Model(ModelConfig.from_dict(mini_config()))
    model.tokenizer = MiniTokenizer()
    stream = model.create_duplex_session(system_prompt="")
    stream._rnnt.step = lambda _encoded: ("", "")
    mx.eval([e.samples for e in stream.push_audio(mx.zeros((stream.frame_samples,)), sample_rate=16_000) if e.kind == "audio"])
    tts = model.tts_model.tts_model
    assert "_compiled_codes" not in tts  # not in the parameter tree
    ref = weakref.ref(model)
    del stream, model, tts
    gc.collect()
    assert ref() is None


def test_compiled_codec_decode_step_matches_eager_decode():
    from mlx_audio.codec.models.nemotron_voicechat import CausalConv1dCache

    model = Model(ModelConfig.from_dict(mini_config()))
    codec = model.tts_model.audio_codec
    cfg = codec.config
    mx.random.seed(7)
    steps = [mx.random.randint(0, cfg.codebook_size, (1, cfg.num_quantizers, 1)) for _ in range(12)]

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)  # exact fp32; Metal's fp32 matmul rounds like tf32
    try:
        eager_cache, step_cache = CausalConv1dCache(), CausalConv1dCache()
        eager = mx.concatenate([codec.decode(c, cache=eager_cache) for c in steps], axis=-1)
        stepped = mx.concatenate([codec.decode_step(c, step_cache) for c in steps], axis=-1)
    finally:
        mx.set_default_device(previous)
    assert "_compiled_decode" in codec.__dict__ and codec.__dict__["_compiled_decode"].fns
    assert eager.shape == stepped.shape
    assert mx.abs(eager - stepped).max().item() < 1e-5


def test_tts_pauses_while_quiet_and_resumes_on_a_token():
    from types import SimpleNamespace

    model = Model(ModelConfig.from_dict(mini_config()))
    model.tokenizer = MiniTokenizer()
    # rms threshold above anything the random mini codec makes: every
    # pad frame counts as quiet.
    stream = model.create_duplex_session(system_prompt="", tts_idle_frames=3, tts_idle_rms=1e9)
    stream._rnnt.step = lambda _encoded: None
    pad = stream.config.pad_token_id
    script = [pad] * 6 + [4] + [pad] * 2
    language_step = stream._language_step

    def forced(inputs):
        out = language_step(inputs)
        wanted = script[min(len(stream._text_tokens), len(script) - 1)]
        text = mx.full(out.text_logits.shape, -1e9).at[..., wanted].add(2e9)
        function = mx.full(out.function_logits.shape, -1e9).at[..., pad].add(2e9)
        return SimpleNamespace(text_logits=text, function_logits=function)

    stream._language_step = forced
    tts = model.tts_model.tts_model
    offsets, audio = [], []
    for _ in script:
        events = stream.push_audio(mx.zeros((stream.frame_samples,)), sample_rate=16_000)
        audio += [e for e in events if e.kind == "audio"]
        offsets.append(stream._tts_cache[0].offset if isinstance(stream._tts_cache, list) else stream._tts_cache.offset)

    assert stream._text_tokens == script
    # Frames 0-2 are quiet (frame 0 decodes the warm-up silence), 3-5
    # pause, the token at frame 6 resumes the TTS, and frames 7-8 are only
    # 2 quiet in a row: running again.
    assert stream.tts_idle_skipped == 3
    assert offsets[2] == offsets[3] == offsets[4] == offsets[5]
    assert offsets[6] == offsets[5] + 1 and offsets[8] == offsets[6] + 2
    assert len(audio) == len(script)
    assert all(mx.all(e.samples == 0).item() for e in audio[3:6])
    assert audio[6].audio_codes is not None
