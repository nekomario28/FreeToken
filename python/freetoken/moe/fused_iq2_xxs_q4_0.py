"""Mixed packed routed experts for PeasantSmith Qwen3.8-Flash-Next.

Gate/up stay IQ2_XXS (GGML type 16); down stays Q4_0 (type 2). Both
operations reuse FreeToken's existing borrowed GGML routed-MoE vector kernel.
No dequantized expert copy is materialized.
"""
from __future__ import annotations

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

GGML_Q4_0 = 2
GGML_IQ2_XXS = 16

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def fused_experts_gguf_iq2_xxs_q4_0(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [slots, 2I, packed_IQ2_row_bytes]
    down_q: torch.Tensor,  # [slots, H, packed_Q4_0_row_bytes]
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")
    if gate_up_q.dtype is not torch.uint8 or down_q.dtype is not torch.uint8:
        raise TypeError("mixed GGUF expert banks must stay packed uint8")
    if gate_up_q.ndim != 3 or down_q.ndim != 3:
        raise ValueError("mixed GGUF expert banks must be rank-3 [slots, rows, packed_bytes]")
    if gate_up_q.shape[0] != down_q.shape[0]:
        raise ValueError("gate/up and down slot counts differ")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]
    hidden = down_q.shape[1]
    top_k = topk_ids.shape[1]

    gate_up = ggml_moe_a8_vec(
        hidden_states,
        gate_up_q,
        topk_ids,
        top_k,
        GGML_IQ2_XXS,
        n2,
        num_tokens,
    )
    inter = act_fn(gate_up)
    out = ggml_moe_a8_vec(
        inter,
        down_q,
        topk_ids,
        1,
        GGML_Q4_0,
        hidden,
        num_tokens * top_k,
    )
    out = out.reshape(num_tokens, top_k, hidden) * topk_weights.reshape(
        num_tokens, top_k, 1
    ).to(out.dtype)
    return out.sum(dim=1)


__all__ = [
    "GGML_Q4_0",
    "GGML_IQ2_XXS",
    "fused_experts_gguf_iq2_xxs_q4_0",
]
