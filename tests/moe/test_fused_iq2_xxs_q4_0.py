import torch


def test_mixed_iq2_q4_expert_wrapper_uses_existing_kernel_with_two_qtypes(monkeypatch):
    from freetoken.moe.fused_iq2_xxs_q4_0 import (
        GGML_IQ2_XXS,
        GGML_Q4_0,
        fused_experts_gguf_iq2_xxs_q4_0,
    )

    calls = []

    def fake_moe(x, weight, ids, top_k, quant_type, row, tokens):
        calls.append((quant_type, tuple(weight.shape), top_k, row, tokens))
        if quant_type == GGML_IQ2_XXS:
            # two routed rows, 2I=4 -> activation halves to I=2
            return torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
        assert quant_type == GGML_Q4_0
        # input has num_tokens*top_k=2 rows, output hidden=3
        return torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    monkeypatch.setattr("freetoken.kernel.gguf.ggml_moe_a8_vec", fake_moe)
    monkeypatch.setattr(
        "freetoken.moe.fused_iq2_xxs_q4_0._ACT",
        {"identity": lambda x: x[:, : x.shape[1] // 2]},
    )

    hidden = torch.zeros((1, 3))
    gate_up = torch.zeros((8, 4, 7), dtype=torch.uint8)
    down = torch.zeros((8, 3, 5), dtype=torch.uint8)
    weights = torch.tensor([[0.25, 0.75]])
    ids = torch.tensor([[5, 2]], dtype=torch.int32)

    out = fused_experts_gguf_iq2_xxs_q4_0(
        hidden, gate_up, down, weights, ids, "identity"
    )

    assert calls == [
        (GGML_IQ2_XXS, (8, 4, 7), 2, 4, 1),
        (GGML_Q4_0, (8, 3, 5), 1, 3, 2),
    ]
    assert torch.allclose(out, torch.tensor([[7.75, 15.5, 23.25]]))
