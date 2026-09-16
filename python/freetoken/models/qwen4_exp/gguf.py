"""Qwen3.8-Flash-Next GGUF config adapter.

Build the existing FreeToken Qwen4Exp ``ModelConfig`` from llama.cpp GGUF
metadata. This file is intentionally model-specific: only the published Qwen3.8
geometry is admitted, and the mixed expert weight tag remains a research-only
backend until the physical G4 gate is green.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.qwen4_exp.config import Qwen4ExpArgs, ple_slot_states

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


_MIXED_EXPERT_FORMAT = "iq2_xxs_q4_0"


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    def g(key: str):
        full = f"qwen4exp.{key}"
        val = m.get(full)
        if val is None:
            raise KeyError(f"missing GGUF metadata key {full}")
        return val

    num_layers = int(g("block_count"))
    hidden = int(g("embedding_length"))
    head_dim = int(g("attention.key_length"))
    value_dim = int(g("attention.value_length"))
    num_q = int(g("attention.head_count"))
    num_kv = int(g("attention.head_count_kv"))
    num_experts = int(g("expert_count"))
    top_k = int(g("expert_used_count"))
    moe_i = int(g("expert_feed_forward_length"))
    shared_i = int(g("expert_shared_feed_forward_length"))

    # Fail closed on the exact family this adapter was written for. A future
    # Qwen4Exp variant should add evidence, not silently inherit these constants.
    expected = {
        "block_count": (num_layers, 48),
        "embedding_length": (hidden, 2560),
        "attention.key_length": (head_dim, 256),
        "attention.value_length": (value_dim, 256),
        "attention.head_count": (num_q, 24),
        "attention.head_count_kv": (num_kv, 2),
        "expert_count": (num_experts, 512),
        "expert_used_count": (top_k, 10),
        "expert_feed_forward_length": (moe_i, 640),
        "expert_shared_feed_forward_length": (shared_i, 640),
    }
    drift = {name: pair for name, pair in expected.items() if pair[0] != pair[1]}
    if drift:
        raise ValueError(f"unsupported Qwen4Exp GGUF geometry drift: {drift}")

    compress = tuple(int(v) for v in g("attention.compress_ratios"))
    if len(compress) != num_layers:
        raise ValueError("attention.compress_ratios length != block_count")
    full_ids = tuple(i for i, ratio in enumerate(compress) if ratio > 0)
    linear_ids = tuple(i for i, ratio in enumerate(compress) if ratio == 0)
    if not full_ids or not linear_ids:
        raise ValueError("Qwen3.8 requires both full and linear attention layers")
    ratios = {compress[i] for i in full_ids}
    if len(ratios) != 1:
        raise ValueError(f"full-attention compression ratio drift: {sorted(ratios)}")
    index_ratio = ratios.pop()

    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=int(g("rope.dimension_count")),
        max_position=int(g("context_length")),
        base=float(g("rope.freq_base")),
        scaling=None,
    )

    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        rotary_config=rotary,
        index_head_dim=int(g("attention.indexer.key_length")),
        num_index_layers=len(full_ids),
        index_ratio=index_ratio,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=int(g("ssm.group_count")),
        num_value_heads=int(g("ssm.time_step_rank")),
        key_head_dim=int(g("ssm.state_size")),
        value_head_dim=int(g("ssm.state_size")),
        conv_kernel_dim=int(g("ssm.conv_kernel")),
        output_gate="sigmoid",
    )

    ngram_size = int(g("ple.ngram_size"))
    heads_per_ngram = int(g("ple.heads_per_ngram"))
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    ngram_head_dim = int(g("embedding_length_per_layer_input"))
    ple_embed_dim = ngram_heads * ngram_head_dim
    if ple_embed_dim != hidden:
        raise ValueError(
            f"PLE width {ngram_heads}*{ngram_head_dim}={ple_embed_dim} != hidden {hidden}"
        )
    ple_layers = tuple(int(v) for v in g("ple.layers"))
    if any(layer not in linear_ids for layer in ple_layers):
        raise ValueError("PLE layer must be a linear-attention layer")

    qwen4_args = Qwen4ExpArgs(
        hidden_size=hidden,
        hc_count=int(g("hyper_connection.count")),
        hc_lowrank=int(g("hyper_connection.low_rank")),
        ple_layer_ids=ple_layers,
        ple_embed_dim=ple_embed_dim,
        ple_conv_kernel_size=int(g("ple.conv_kernel")),
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        # These three are model constants from the official Qwen3.8 config. The
        # GGUF carries the already-derived multipliers/sizes/offsets; the GGUF
        # weight adapter will validate and load those exact arrays separately.
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
        ngram_boundary_token_id=int(g("ple.eos_token_id")),
        index_n_heads=int(g("attention.indexer.head_count")),
        index_kv_heads=1,
        index_head_dim=int(g("attention.indexer.key_length")),
        index_budget=int(g("attention.indexer.top_k")),
        index_ratio=index_ratio,
        image_token_id=int(g("ple.image_token_id")),
    )

    groups = tuple(sorted((full_group, linear_group), key=lambda group: group.layer_ids[0]))
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_q,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=0,
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=rotary,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_i,
        shared_expert_intermediate_size=shared_i,
        norm_topk_prob=True,
        moe_enabled=True,
        use_qk_norm=True,
        model_type="qwen4_exp",
        architectures=list(shim.architectures),
        expert_quant=_MIXED_EXPERT_FORMAT,
        moe_weight_format=_MIXED_EXPERT_FORMAT,
        qwen4_args=qwen4_args,
        attention_groups=groups,
        slot_states=ple_slot_states(qwen4_args),
    )


__all__ = ["parse_gguf_config"]
