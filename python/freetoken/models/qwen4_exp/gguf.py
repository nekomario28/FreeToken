"""Native GGUF adapter for Qwen3.8-Flash-Next (``qwen4exp``).

The adapter deliberately keeps three ownership domains separate:

* dense / shared weights stay in their native GGUF block layout and are executed by
  FreeToken's existing GGUF kernels;
* routed experts are *not* materialized here (the cold-source/offload path owns them);
* the huge PLE table is *not* a state-dict tensor (``ple_gguf`` reads selected IQ4_NL rows).

Projection fusion is header-driven. Parts with one packed quant type are concatenated
as packed output rows and use one ``GGUFLinear``. Mixed packed/dense parts use a small
correctness-first composite op, so the adapter never assumes PeasantSmith's exact
per-tensor quant recipe beyond the routed-expert contract.

One semantic conversion is mandatory: llama.cpp stores Qwen3.5/3.8 GDN value heads in
its tiled broadcast order, while FreeToken follows the HF grouped-by-key-head order.
Row-oriented tensors are inverse-permuted at load without dequantizing packed rows.
The output projection keeps its GGUF column layout and instead permutes its activation
from grouped -> tiled immediately before the native GGUF GEMM; this avoids any
requantization or block-layout surgery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Sequence

import torch
import torch.nn.functional as F

from freetoken.layers.base import BaseOP
from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear, fused_mul_mat_gguf
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.gguf.dequant import (
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_Q4_0,
    GGML_Q4_K,
    GGML_Q6_K,
    GGML_Q8_0,
    dequantize,
    row_bytes,
)
from freetoken.models.qwen4_exp.config import Qwen4ExpArgs, ple_slot_states

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


_MIXED_EXPERT_FORMAT = "iq2_xxs_q4_0"
_UNQUANTIZED = frozenset({GGML_F32, GGML_F16, GGML_BF16})
_PACKED_DENSE = frozenset({GGML_Q4_0, GGML_Q8_0, GGML_Q4_K, GGML_Q6_K})
_PLE_TABLE = "per_layer_token_embd.weight"
_ROUTED_EXPERT_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)


@dataclass(frozen=True)
class Qwen4ExpGgufArgs(Qwen4ExpArgs):
    """Qwen4Exp geometry plus the immutable GGUF source used to build native ops."""

    gguf_model_path: str = ""


@dataclass(frozen=True)
class _TensorSpec:
    name: str
    shape: tuple[int, ...]  # torch order [out, in] for matrices
    ggml_type: int


def _v_grouped_to_tiled_perm(
    num_k_heads: int, num_v_heads: int, head_dim: int
) -> torch.Tensor:
    """Output-position -> grouped-input-position permutation used by llama.cpp.

    HF grouped order is ``[K0:v0..vr, K1:v0..vr, ...]``. llama.cpp transposes the
    logical ``[K, r, D]`` head grid to ``[r, K, D]`` before flattening.
    """
    if num_v_heads % num_k_heads:
        raise ValueError(f"num_v_heads {num_v_heads} not divisible by num_k_heads {num_k_heads}")
    r = num_v_heads // num_k_heads
    return (
        torch.arange(num_v_heads * head_dim, dtype=torch.long)
        .reshape(num_k_heads, r, head_dim)
        .permute(1, 0, 2)
        .reshape(-1)
    )


def _v_tiled_to_grouped_perm(
    num_k_heads: int, num_v_heads: int, head_dim: int
) -> torch.Tensor:
    return torch.argsort(_v_grouped_to_tiled_perm(num_k_heads, num_v_heads, head_dim))


def _grouped_to_tiled_last(
    x: torch.Tensor, num_k_heads: int, num_v_heads: int, head_dim: int
) -> torch.Tensor:
    """Runtime activation permutation matching llama.cpp's output-projection columns."""
    if x.shape[-1] != num_v_heads * head_dim:
        raise ValueError(
            f"GDN output width {x.shape[-1]} != {num_v_heads}*{head_dim}"
        )
    r = num_v_heads // num_k_heads
    shape = x.shape
    return (
        x.reshape(*shape[:-1], num_k_heads, r, head_dim)
        .transpose(-3, -2)
        .reshape(*shape)
        .contiguous()
    )


def _gdn_inverse_row_indices(
    group: LinearGatedDeltaGroupConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Indices restoring llama.cpp's tiled GDN rows to HF/FreeToken grouped order."""
    inv_v = _v_tiled_to_grouped_perm(
        group.num_key_heads, group.num_value_heads, group.value_head_dim
    )
    inv_h = _v_tiled_to_grouped_perm(group.num_key_heads, group.num_value_heads, 1)
    key_dim = group.num_key_heads * group.key_head_dim
    qk = torch.arange(2 * key_dim, dtype=torch.long)
    qkv = torch.cat((qk, 2 * key_dim + inv_v))
    return qkv, inv_v, inv_h


class _GGUFMixedLinear(BaseOP):
    """Fused logical Linear backed by independently-typed GGUF projection parts.

    Quantized parts stay packed; F32/F16/BF16 parts are loaded as normal model-dtype
    tensors. The outputs are concatenated in checkpoint order. ``pad`` is used only by
    Qwen4Exp's HC down+inject projection (the runtime buffer is padded to 16 rows).
    """

    def __init__(self, in_features: int, parts: Sequence[_TensorSpec], *, pad: int = 0):
        self.in_features = int(in_features)
        self._part_types = tuple(int(p.ggml_type) for p in parts)
        self._part_out = tuple(int(p.shape[0]) for p in parts)
        self._pad = int(pad)
        self.out_features = sum(self._part_out) + self._pad
        for i, (qtype, out) in enumerate(zip(self._part_types, self._part_out)):
            if qtype in _PACKED_DENSE:
                setattr(
                    self,
                    f"qweight_{i}",
                    torch.empty(out, row_bytes(self.in_features, qtype), dtype=torch.uint8),
                )
            elif qtype in _UNQUANTIZED:
                setattr(self, f"weight_{i}", torch.empty(out, self.in_features))
            else:
                raise NotImplementedError(f"Qwen4Exp mixed GGUF Linear type {qtype} is unsupported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i, qtype in enumerate(self._part_types):
            if qtype in _PACKED_DENSE:
                outs.append(fused_mul_mat_gguf(x, getattr(self, f"qweight_{i}"), qtype))
            else:
                w = getattr(self, f"weight_{i}")
                if w.dtype != x.dtype:
                    w = w.to(x.dtype)
                outs.append(F.linear(x, w))
        if self._pad:
            outs.append(x.new_zeros((x.shape[0], self._pad)))
        return torch.cat(outs, dim=-1)


class _GGUFVOutputLinear(BaseOP):
    """GDN output projection with llama.cpp tiled GGUF columns.

    FreeToken's GDN produces HF grouped value-head order. The stored GGUF projection
    had its columns permuted grouped->tiled by llama.cpp, so apply the same permutation
    to the activation and leave the packed weight byte-for-byte unchanged.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: int,
        *,
        num_k_heads: int,
        num_v_heads: int,
        head_dim: int,
    ) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self._quant_type = int(quant_type)
        self._num_k_heads = int(num_k_heads)
        self._num_v_heads = int(num_v_heads)
        self._head_dim = int(head_dim)
        if self._quant_type in _PACKED_DENSE:
            self.qweight = torch.empty(
                self.out_features,
                row_bytes(self.in_features, self._quant_type),
                dtype=torch.uint8,
            )
        elif self._quant_type in _UNQUANTIZED:
            self.weight = torch.empty(self.out_features, self.in_features)
        else:
            raise NotImplementedError(f"Qwen4Exp GDN output GGUF type {quant_type} unsupported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _grouped_to_tiled_last(
            x, self._num_k_heads, self._num_v_heads, self._head_dim
        )
        if self._quant_type in _PACKED_DENSE:
            return fused_mul_mat_gguf(x, self.qweight, self._quant_type)
        w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        return F.linear(x, w)


class _GGUFUntiedLMHead(BaseOP):
    """TP=1 native-GGUF LM head preserving ParallelLMHead's prefill row selection."""

    def __init__(self, vocab_size: int, hidden_size: int, quant_type: int):
        self.num_embeddings = int(vocab_size)
        self.in_features = int(hidden_size)
        self.out_features = int(vocab_size)
        self._quant_type = int(quant_type)
        self.qweight = torch.empty(
            self.out_features, row_bytes(self.in_features, self._quant_type), dtype=torch.uint8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            idx = batch.attn_metadata.get_last_indices(batch.size)
            x = x[idx].contiguous()
        return fused_mul_mat_gguf(x, self.qweight, self._quant_type)


class _GGUFTiedLMHead(BaseOP):
    """TP=1 tied head over a packed GGUF embedding table; owns no weight itself."""

    def __init__(self, embedding: GGUFEmbedding, quant_type: int):
        self._embedding = embedding
        self._quant_type = int(quant_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            idx = batch.attn_metadata.get_last_indices(batch.size)
            x = x[idx].contiguous()
        return fused_mul_mat_gguf(x, self._embedding.qweight, self._quant_type)


def _require_tp1(what: str) -> None:
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(f"Qwen4Exp GGUF {what} currently supports TP=1 only")


def _header_specs(model_path: str) -> dict[str, _TensorSpec]:
    import gguf

    reader = gguf.GGUFReader(model_path)
    return {
        t.name: _TensorSpec(
            t.name,
            tuple(reversed(tuple(int(v) for v in t.shape))),
            int(t.tensor_type),
        )
        for t in reader.tensors
    }


def _required(by_name, name: str):
    try:
        return by_name[name]
    except KeyError as exc:
        raise ValueError(f"missing Qwen4Exp GGUF tensor {name!r}") from exc


def _projection_mode(parts: Sequence[_TensorSpec], *, pad: int = 0) -> str:
    qtypes = tuple(p.ggml_type for p in parts)
    bad = [q for q in qtypes if q not in _UNQUANTIZED and q not in _PACKED_DENSE]
    if bad:
        raise NotImplementedError(f"unsupported Qwen4Exp dense GGUF quant types {sorted(set(bad))}")
    if all(q in _UNQUANTIZED for q in qtypes):
        return "dense"
    if not pad and len(set(qtypes)) == 1 and qtypes[0] in _PACKED_DENSE:
        return "packed"
    return "mixed"


def _validate_projection(parts: Sequence[_TensorSpec], *, expected_in: int) -> None:
    if not parts:
        raise ValueError("GGUF projection has no parts")
    for part in parts:
        if len(part.shape) != 2:
            raise ValueError(f"{part.name}: expected matrix, got shape {part.shape}")
        if part.shape[1] != expected_in:
            raise ValueError(
                f"{part.name}: input width {part.shape[1]} != module width {expected_in}"
            )


def _swap_projection(owner, attr: str, parts: Sequence[_TensorSpec], *, pad: int = 0) -> None:
    old = getattr(owner, attr)
    in_features = int(old.in_features)
    _validate_projection(parts, expected_in=in_features)
    expected_out = sum(int(p.shape[0]) for p in parts) + pad
    if int(old.out_features) != expected_out:
        raise ValueError(
            f"{attr}: GGUF output rows {expected_out} != module rows {int(old.out_features)}"
        )
    mode = _projection_mode(parts, pad=pad)
    if mode == "dense":
        return  # existing bf16/F32 Linear is the right owner; the loader fuses its rows.
    if mode == "packed":
        setattr(owner, attr, GGUFLinear(in_features, expected_out, parts[0].ggml_type))
        return
    setattr(owner, attr, _GGUFMixedLinear(in_features, parts, pad=pad))


def _swap_gdn_output(
    owner,
    part: _TensorSpec,
    group: LinearGatedDeltaGroupConfig,
) -> None:
    old = owner.out_proj
    if part.shape != (int(old.out_features), int(old.in_features)):
        raise ValueError(
            f"{part.name}: shape {part.shape} != GDN out_proj "
            f"{(int(old.out_features), int(old.in_features))}"
        )
    owner.out_proj = _GGUFVOutputLinear(
        int(old.in_features),
        int(old.out_features),
        part.ggml_type,
        num_k_heads=group.num_key_heads,
        num_v_heads=group.num_value_heads,
        head_dim=group.value_head_dim,
    )


def _dense_payload(t, *, out_dtype: torch.dtype = torch.bfloat16, zero_centered: bool = False):
    if int(t.ggml_type) not in _UNQUANTIZED:
        raise ValueError(f"{t.name}: expected unquantized GGUF tensor, got type {t.ggml_type}")
    value = dequantize(t.packed().reshape(-1), int(t.ggml_type), out_dtype).reshape(t.shape)
    if zero_centered:
        value = value - value.new_tensor(1.0)
    return value


def _row_payload(t, spec: _TensorSpec, rows: torch.Tensor | None):
    if spec.ggml_type in _PACKED_DENSE:
        value = t.packed()
    else:
        value = _dense_payload(t)
    if rows is not None:
        value = value.index_select(0, rows.to(value.device))
    return value


def _emit_projection(
    target: str,
    parts,
    *,
    pad: int = 0,
    row_indices: Sequence[torch.Tensor | None] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    specs = tuple(_TensorSpec(t.name, t.shape, int(t.ggml_type)) for t in parts)
    _validate_projection(specs, expected_in=int(specs[0].shape[1]))
    rows = tuple(row_indices or (None,) * len(parts))
    if len(rows) != len(parts):
        raise ValueError("row_indices length must match projection parts")
    mode = _projection_mode(specs, pad=pad)
    payloads = tuple(_row_payload(t, spec, row) for t, spec, row in zip(parts, specs, rows))
    if mode == "dense":
        dense_rows = list(payloads)
        if pad:
            dense_rows.append(
                torch.zeros(pad, specs[0].shape[1], dtype=dense_rows[0].dtype)
            )
        yield f"{target}.weight", torch.cat(dense_rows, dim=0)
        return
    if mode == "packed":
        yield f"{target}.qweight", torch.cat(payloads, dim=0)
        return
    for i, (payload, spec) in enumerate(zip(payloads, specs)):
        if spec.ggml_type in _PACKED_DENSE:
            yield f"{target}.qweight_{i}", payload
        else:
            yield f"{target}.weight_{i}", payload


def _emit_gdn_output(target: str, tensor) -> Iterator[tuple[str, torch.Tensor]]:
    spec = _TensorSpec(tensor.name, tensor.shape, int(tensor.ggml_type))
    if spec.ggml_type in _PACKED_DENSE:
        yield f"{target}.qweight", tensor.packed()
    elif spec.ggml_type in _UNQUANTIZED:
        yield f"{target}.weight", _dense_payload(tensor)
    else:
        raise NotImplementedError(f"{tensor.name}: unsupported GDN output type {spec.ggml_type}")


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

    image_token = m.get("qwen4exp.ple.image_token_id")
    qwen4_args = Qwen4ExpGgufArgs(
        hidden_size=hidden,
        hc_count=int(g("hyper_connection.count")),
        hc_lowrank=int(g("hyper_connection.low_rank")),
        ple_layer_ids=ple_layers,
        ple_embed_dim=ple_embed_dim,
        ple_conv_kernel_size=int(g("ple.conv_kernel")),
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
        ngram_boundary_token_id=int(g("ple.eos_token_id")),
        index_n_heads=int(g("attention.indexer.head_count")),
        index_kv_heads=1,
        index_head_dim=int(g("attention.indexer.key_length")),
        index_budget=int(g("attention.indexer.top_k")),
        index_ratio=index_ratio,
        image_token_id=None if image_token is None else int(image_token),
        gguf_model_path=shim.model_path,
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


def is_gguf_model(config: ModelConfig) -> bool:
    args = getattr(config, "qwen4_args", None)
    return isinstance(args, Qwen4ExpGgufArgs) and bool(args.gguf_model_path)


def convert_qwen4exp_to_gguf(model, config: ModelConfig) -> None:
    """Replace Qwen4Exp dense ops in-place with native-GGUF owners from the real header."""
    _require_tp1("model construction")
    if not is_gguf_model(config):
        return
    path = config.qwen4_args.gguf_model_path
    h = _header_specs(path)

    def specs(*names: str) -> tuple[_TensorSpec, ...]:
        return tuple(_required(h, name) for name in names)

    token = _required(h, "token_embd.weight")
    if token.ggml_type in _PACKED_DENSE:
        model.model.embed_tokens = GGUFEmbedding(config.vocab_size, config.hidden_size, token.ggml_type)
    elif token.ggml_type not in _UNQUANTIZED:
        raise NotImplementedError(f"token_embd.weight GGUF type {token.ggml_type} unsupported")

    # Final hyper-connection mixer.
    top = model.model.hyper_connection_mixer
    _swap_projection(top, "input_mix_weight_down", specs("output_hc_down.weight"))
    _swap_projection(top, "input_mix_weight_up", specs("output_hc_up.weight"))

    hc_pad = (-(config.qwen4_args.hc_lowrank + config.qwen4_args.hc_count)) % 16
    linear_group = config.linear_attention_group()
    assert linear_group is not None
    for lid, layer in enumerate(model.model.layers.op_list):
        for stem, owner in (
            ("hc_attn", layer.attn_hyper_connection),
            ("hc_ffn", layer.mlp_hyper_connection),
        ):
            _swap_projection(
                owner,
                "input_mix_weight_down_block_inject",
                specs(f"blk.{lid}.{stem}_down.weight", f"blk.{lid}.{stem}_inject.weight"),
                pad=hc_pad,
            )
            _swap_projection(owner, "input_mix_weight_up", specs(f"blk.{lid}.{stem}_up.weight"))

        if config.is_linear_layer(lid):
            la = layer.linear_attn
            _swap_projection(
                la,
                "in_proj",
                specs(
                    f"blk.{lid}.attn_qkv.weight",
                    f"blk.{lid}.attn_gate.weight",
                    f"blk.{lid}.ssm_beta.weight",
                    f"blk.{lid}.ssm_alpha.weight",
                ),
            )
            _swap_gdn_output(la, specs(f"blk.{lid}.ssm_out.weight")[0], linear_group)
        else:
            attn = layer.self_attn
            _swap_projection(
                attn,
                "qkv_proj",
                specs(
                    f"blk.{lid}.attn_q.weight",
                    f"blk.{lid}.attn_k.weight",
                    f"blk.{lid}.attn_v.weight",
                ),
            )
            _swap_projection(attn, "o_proj", specs(f"blk.{lid}.attn_output.weight"))
            _swap_projection(
                attn.indexer,
                "index_qk_proj",
                specs(f"blk.{lid}.indexer.q_proj.weight", f"blk.{lid}.indexer.k_proj.weight"),
            )

        if layer.ple is not None:
            _swap_projection(layer.ple, "key_proj", specs(f"blk.{lid}.ple_key.weight"))
            _swap_projection(layer.ple, "value_proj", specs(f"blk.{lid}.ple_value.weight"))

        mlp = layer.mlp
        _swap_projection(mlp, "gate", specs(f"blk.{lid}.ffn_gate_inp.weight"))
        _swap_projection(
            mlp.shared_expert,
            "gate_up_proj",
            specs(f"blk.{lid}.ffn_gate_shexp.weight", f"blk.{lid}.ffn_up_shexp.weight"),
        )
        _swap_projection(
            mlp.shared_expert, "down_proj", specs(f"blk.{lid}.ffn_down_shexp.weight")
        )
        _swap_projection(
            mlp, "shared_expert_gate", specs(f"blk.{lid}.ffn_gate_inp_shexp.weight")
        )

    output = h.get("output.weight")
    if output is None:
        if not config.tie_word_embeddings:
            raise ValueError("Qwen4Exp GGUF has no output.weight but config is not tied")
        if isinstance(model.model.embed_tokens, GGUFEmbedding):
            model.lm_head = _GGUFTiedLMHead(model.model.embed_tokens, token.ggml_type)
    elif output.ggml_type in _PACKED_DENSE:
        if output.shape != (config.vocab_size, config.hidden_size):
            raise ValueError(f"output.weight shape {output.shape} is not vocab x hidden")
        model.lm_head = _GGUFUntiedLMHead(config.vocab_size, config.hidden_size, output.ggml_type)
    elif output.ggml_type not in _UNQUANTIZED:
        raise NotImplementedError(f"output.weight GGUF type {output.ggml_type} unsupported")


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the Qwen4Exp text state dict while routed experts and PLE table stay external."""
    del device
    _require_tp1("weight loading")
    if include_moe_experts:
        raise ValueError(
            "Qwen4Exp GGUF routed experts must use the offload/cold-source backend; "
            "they are not dense state-dict weights"
        )
    if not include_non_moe:
        return

    from freetoken.models.gguf.reader import iter_gguf_tensors, load_gguf_metadata
    from freetoken.utils import cached_load_hf_config

    shim = cached_load_hf_config(model_path)
    config = parse_gguf_config(shim)
    by = {t.name: t for t in iter_gguf_tensors(model_path)}
    used: set[str] = set()

    def take(name: str):
        t = _required(by, name)
        used.add(name)
        return t

    def emit(
        target: str,
        *names: str,
        pad: int = 0,
        row_indices: Sequence[torch.Tensor | None] | None = None,
    ):
        parts = tuple(take(name) for name in names)
        yield from _emit_projection(
            target, parts, pad=pad, row_indices=row_indices
        )

    def dense(
        target: str,
        name: str,
        *,
        zero_centered: bool = False,
        dtype=torch.bfloat16,
        reshape=None,
        rows: torch.Tensor | None = None,
    ):
        t = take(name)
        value = _dense_payload(t, out_dtype=dtype, zero_centered=zero_centered)
        if rows is not None:
            value = value.index_select(0, rows.to(value.device))
        if reshape is not None:
            value = value.reshape(*reshape)
        yield target, value

    # Embedding / final HC / output head.
    yield from emit("model.embed_tokens", "token_embd.weight")
    yield from dense(
        "model.hyper_connection_mixer.hc_norm.weight",
        "output_hc_norm.weight",
        zero_centered=True,
        reshape=(-1,),
    )
    yield from emit("model.hyper_connection_mixer.input_mix_weight_down", "output_hc_down.weight")
    yield from emit("model.hyper_connection_mixer.input_mix_weight_up", "output_hc_up.weight")
    if "output.weight" in by:
        yield from emit("lm_head", "output.weight")

    hc_pad = (-(config.qwen4_args.hc_lowrank + config.qwen4_args.hc_count)) % 16
    linear_group = config.linear_attention_group()
    assert linear_group is not None
    qkv_rows, v_rows, head_rows = _gdn_inverse_row_indices(linear_group)
    for lid in range(config.num_layers):
        base = f"model.layers.{lid}"
        for stem, target in (
            ("hc_attn", "attn_hyper_connection"),
            ("hc_ffn", "mlp_hyper_connection"),
        ):
            yield from dense(
                f"{base}.{target}.hc_norm.weight",
                f"blk.{lid}.{stem}_norm.weight",
                zero_centered=True,
                reshape=(-1,),
            )
            yield from emit(
                f"{base}.{target}.input_mix_weight_down_block_inject",
                f"blk.{lid}.{stem}_down.weight",
                f"blk.{lid}.{stem}_inject.weight",
                pad=hc_pad,
            )
            yield from emit(
                f"{base}.{target}.input_mix_weight_up", f"blk.{lid}.{stem}_up.weight"
            )

        if config.is_linear_layer(lid):
            p = f"{base}.linear_attn"
            yield from emit(
                f"{p}.in_proj",
                f"blk.{lid}.attn_qkv.weight",
                f"blk.{lid}.attn_gate.weight",
                f"blk.{lid}.ssm_beta.weight",
                f"blk.{lid}.ssm_alpha.weight",
                row_indices=(qkv_rows, v_rows, head_rows, head_rows),
            )
            conv_dim = (
                linear_group.num_key_heads * linear_group.key_head_dim * 2
                + linear_group.num_value_heads * linear_group.value_head_dim
            )
            yield from dense(
                f"{p}.conv1d.weight",
                f"blk.{lid}.ssm_conv1d.weight",
                rows=qkv_rows,
                reshape=(conv_dim, 1, linear_group.conv_kernel_dim),
            )
            yield from dense(
                f"{p}.dt_bias",
                f"blk.{lid}.ssm_dt.bias",
                dtype=torch.float32,
                rows=head_rows,
                reshape=(-1,),
            )
            yield from dense(
                f"{p}.A_log",
                f"blk.{lid}.ssm_a",
                dtype=torch.float32,
                rows=head_rows,
                reshape=(-1,),
            )
            yield from dense(f"{p}.norm.weight", f"blk.{lid}.ssm_norm.weight", reshape=(-1,))
            out = take(f"blk.{lid}.ssm_out.weight")
            yield from _emit_gdn_output(f"{p}.out_proj", out)
        else:
            p = f"{base}.self_attn"
            yield from emit(
                f"{p}.qkv_proj",
                f"blk.{lid}.attn_q.weight",
                f"blk.{lid}.attn_k.weight",
                f"blk.{lid}.attn_v.weight",
            )
            yield from emit(f"{p}.o_proj", f"blk.{lid}.attn_output.weight")
            yield from dense(
                f"{p}.q_norm.weight",
                f"blk.{lid}.attn_q_norm.weight",
                zero_centered=True,
                reshape=(-1,),
            )
            yield from dense(
                f"{p}.k_norm.weight",
                f"blk.{lid}.attn_k_norm.weight",
                zero_centered=True,
                reshape=(-1,),
            )
            yield from emit(
                f"{p}.indexer.index_qk_proj",
                f"blk.{lid}.indexer.q_proj.weight",
                f"blk.{lid}.indexer.k_proj.weight",
            )
            yield from dense(
                f"{p}.indexer.q_layernorm.weight",
                f"blk.{lid}.indexer.q_norm.weight",
                zero_centered=True,
                reshape=(-1,),
            )
            yield from dense(
                f"{p}.indexer.k_layernorm.weight",
                f"blk.{lid}.indexer.k_norm.weight",
                zero_centered=True,
                reshape=(-1,),
            )

        if lid in config.qwen4_args.ple_layer_ids:
            p = f"{base}.ple"
            yield from emit(f"{p}.key_proj", f"blk.{lid}.ple_key.weight")
            yield from emit(f"{p}.value_proj", f"blk.{lid}.ple_value.weight")
            width = config.qwen4_args.ple_state_width
            for leaf, gguf_leaf in (
                ("norm_key", "ple_norm_key"),
                ("norm_query", "ple_norm_query"),
                ("norm_conv", "ple_norm_conv"),
            ):
                yield from dense(
                    f"{p}.{leaf}.weight",
                    f"blk.{lid}.{gguf_leaf}.weight",
                    zero_centered=True,
                    reshape=(-1,),
                )
            yield from dense(
                f"{p}.conv1d.weight",
                f"blk.{lid}.ple_conv1d.weight",
                reshape=(width, 1, config.qwen4_args.ple_conv_kernel_size),
            )

        p = f"{base}.mlp"
        yield from emit(f"{p}.gate", f"blk.{lid}.ffn_gate_inp.weight")
        yield from emit(
            f"{p}.shared_expert.gate_up_proj",
            f"blk.{lid}.ffn_gate_shexp.weight",
            f"blk.{lid}.ffn_up_shexp.weight",
        )
        yield from emit(f"{p}.shared_expert.down_proj", f"blk.{lid}.ffn_down_shexp.weight")
        yield from emit(f"{p}.shared_expert_gate", f"blk.{lid}.ffn_gate_inp_shexp.weight")

    # The GGUF converter stores PLE hash constants as exact KV metadata, not tensors.
    md = load_gguf_metadata(model_path)
    for lid in config.qwen4_args.ple_layer_ids:
        p = f"model.layers.{lid}.ple.ple_embedding"
        constants = (
            ("layer_multipliers", "qwen4exp.ple.layer_multipliers"),
            ("ngram_heads_offsets", "qwen4exp.ple.head_offsets"),
            ("ngram_heads_vocab_sizes", "qwen4exp.ple.head_vocab_sizes"),
        )
        for leaf, key in constants:
            if key not in md:
                raise KeyError(f"missing GGUF metadata key {key}")
            yield f"{p}.{leaf}", torch.tensor(md[key], dtype=torch.int64)

    # Explicitly external payloads: PLE table and routed experts.
    if _PLE_TABLE in by:
        used.add(_PLE_TABLE)
    for name in by:
        if any(name.endswith(sfx) for sfx in _ROUTED_EXPERT_SUFFIXES):
            used.add(name)

    # Fail closed if a future qwen4exp converter adds a text tensor this adapter ignores.
    unmapped = sorted(set(by) - used)
    if unmapped:
        raise ValueError(f"unmapped Qwen4Exp GGUF tensors ({len(unmapped)}): {unmapped[:12]}")


class Qwen4ExpGGUFForCausalLM:
    """Marker replaced below after importing the normal model class (avoids import cycles)."""


def _make_gguf_model_class():
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    class _Qwen4ExpGGUFForCausalLM(Qwen4ExpForCausalLM):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__(config)
            convert_qwen4exp_to_gguf(self, config)

        def load_host_tables(self, engine_config) -> int:
            if getattr(engine_config, "use_dummy_weight", False):
                return super().load_host_tables(engine_config)
            if not is_gguf_model(self._config):
                return super().load_host_tables(engine_config)
            ple_layers = self.model.ple_layers
            if not ple_layers:
                return 0
            assert len(ple_layers) == 1, "Qwen4Exp GGUF currently expects exactly one PLE layer"
            from .ple_gguf import GgufIq4NlPleTable, compile_iq4_nl_ple_index

            index = compile_iq4_nl_ple_index(
                self._config.qwen4_args.gguf_model_path,
                expected_head_dim=self._config.qwen4_args.ngram_head_dim,
            )
            table = GgufIq4NlPleTable(index)
            self._ple_table = table
            for ple in ple_layers:
                ple.ple_embedding.attach_table(table)
            return 0

    _Qwen4ExpGGUFForCausalLM.__name__ = "Qwen4ExpGGUFForCausalLM"
    _Qwen4ExpGGUFForCausalLM.__qualname__ = "Qwen4ExpGGUFForCausalLM"
    return _Qwen4ExpGGUFForCausalLM


Qwen4ExpGGUFForCausalLM = _make_gguf_model_class()


__all__ = [
    "Qwen4ExpGgufArgs",
    "Qwen4ExpGGUFForCausalLM",
    "convert_qwen4exp_to_gguf",
    "is_gguf_model",
    "iter_gguf_weights",
    "parse_gguf_config",
]
