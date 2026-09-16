from types import SimpleNamespace

from freetoken.models.qwen4_exp.gguf import parse_gguf_config


def _shim():
    compress = [0, 0, 0, 4] * 12
    metadata = {
        "qwen4exp.block_count": 48,
        "qwen4exp.context_length": 262144,
        "qwen4exp.embedding_length": 2560,
        "qwen4exp.attention.head_count": 24,
        "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.attention.key_length": 256,
        "qwen4exp.attention.value_length": 256,
        "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen4exp.rope.dimension_count": 64,
        "qwen4exp.rope.freq_base": 10_000_000.0,
        "qwen4exp.expert_count": 512,
        "qwen4exp.expert_used_count": 10,
        "qwen4exp.expert_feed_forward_length": 640,
        "qwen4exp.expert_shared_feed_forward_length": 640,
        "qwen4exp.attention.compress_ratios": compress,
        "qwen4exp.attention.indexer.head_count": 4,
        "qwen4exp.attention.indexer.key_length": 128,
        "qwen4exp.attention.indexer.top_k": 2048,
        "qwen4exp.ssm.conv_kernel": 4,
        "qwen4exp.ssm.state_size": 128,
        "qwen4exp.ssm.group_count": 16,
        "qwen4exp.ssm.time_step_rank": 48,
        "qwen4exp.hyper_connection.count": 4,
        "qwen4exp.hyper_connection.low_rank": 320,
        "qwen4exp.ple.layers": [1],
        "qwen4exp.ple.ngram_size": 3,
        "qwen4exp.ple.heads_per_ngram": 8,
        "qwen4exp.ple.conv_kernel": 4,
        "qwen4exp.ple.eos_token_id": 248044,
        "qwen4exp.ple.image_token_id": 248056,
        "qwen4exp.embedding_length_per_layer_input": 160,
    }
    return SimpleNamespace(
        metadata=metadata,
        vocab_size=248320,
        tie_word_embeddings=False,
        architectures=["Qwen4ExpGGUFForCausalLM"],
    )


def test_qwen4exp_gguf_config_matches_published_geometry():
    cfg = parse_gguf_config(_shim())

    assert cfg.num_layers == 48
    assert cfg.hidden_size == 2560
    assert cfg.num_qo_heads == 24
    assert cfg.num_kv_heads == 2
    assert cfg.head_dim == 256
    assert cfg.rotary_config.rotary_dim == 64
    assert cfg.num_experts == 512
    assert cfg.num_experts_per_tok == 10
    assert cfg.moe_intermediate_size == 640
    assert cfg.shared_expert_intermediate_size == 640
    assert cfg.expert_quant == "iq2_xxs_q4_0"
    assert cfg.moe_weight_format == "iq2_xxs_q4_0"
    assert cfg.qwen4_args.ple_layer_ids == (1,)
    assert cfg.qwen4_args.ple_embed_dim == 2560
    assert cfg.qwen4_args.ngram_head_dim == 160
    assert cfg.qwen4_args.index_n_heads == 4
    assert cfg.qwen4_args.index_head_dim == 128
    assert cfg.qwen4_args.index_budget == 2048
    assert cfg.qwen4_args.index_ratio == 4

    full = cfg.full_attention_group()
    linear = cfg.linear_attention_group()
    assert full.layer_ids == tuple(range(3, 48, 4))
    assert linear.num_key_heads == 16
    assert linear.num_value_heads == 48
    assert linear.key_head_dim == 128
    assert linear.value_head_dim == 128
    assert linear.conv_kernel_dim == 4
    assert linear.output_gate == "sigmoid"


def test_qwen4exp_gguf_config_fails_closed_on_geometry_drift():
    import pytest

    shim = _shim()
    shim.metadata["qwen4exp.expert_count"] = 256
    with pytest.raises(ValueError, match="geometry drift"):
        parse_gguf_config(shim)
