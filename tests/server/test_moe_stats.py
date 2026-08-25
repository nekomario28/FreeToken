from __future__ import annotations

import asyncio
from types import SimpleNamespace

import torch

from freetoken.control_cli import _format_stats
from freetoken.message import CacheRebuildReply, UserReply
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.server import api_server
from freetoken.server.stats import StatsTracker, build_stats


def _state(stats: StatsTracker, *, bytes_per_expert: int = 1024):
    model_config = SimpleNamespace(
        is_moe=True,
        has_linear_attention=False,
        has_swa_attention=False,
    )
    config = SimpleNamespace(
        model_config=model_config,
        served_model_name="moe-test",
        max_seq_len=4096,
        page_size=1,
    )
    return SimpleNamespace(
        stats=stats,
        config=config,
        ready_at=None,
        instance_id="test-instance",
        unit_bytes={"moe_bytes_per_expert": bytes_per_expert},
    )


class _RebuildState:
    def __init__(self, *, reply_status: str | None) -> None:
        self.rebuild_futures = {}
        self.moe_stats_reset_rebuilds: set[str] = set()
        self.maintenance_state = "serving"
        self.last_rebuild = None
        self.fatal_error = None
        self.stats = StatsTracker()
        self.stats.moe_layer_calls = 80
        self.stats.moe_active_experts = 640
        self.stats.moe_missing_experts = 160
        self.stats.moe_fetched_experts = 120
        self.reply_status = reply_status

    async def send_one(self, msg) -> None:
        if self.reply_status is None:
            return
        reply = CacheRebuildReply(
            request_id=msg.request_id,
            status=self.reply_status,
            moe_cache_size=msg.moe_cache_size or 0,
            num_pages=msg.num_pages or 0,
            mamba_slots=msg.num_mamba_slots or 0,
            num_swa_pages=msg.num_swa_pages or 0,
        )
        asyncio.get_running_loop().call_soon(
            api_server.FrontendManager._resolve_rebuild, self, reply
        )


def _run_rebuild(state: _RebuildState, *, moe_cache_size: int | None, num_pages: int | None = None):
    return asyncio.run(
        api_server.dispatch_rebuild(
            state,
            moe_cache_size=moe_cache_size,
            num_pages=num_pages,
            num_mamba_slots=None,
            num_swa_pages=None,
            mode="if_idle",
            timeout=0.01,
        )
    )


def test_moe_snapshot_derives_hit_split_and_logical_h2d_payload():
    stats = StatsTracker()
    stats.observe(
        UserReply(
            uid=1,
            incremental_output="x",
            finished=False,
            moe_layer_calls=80,
            moe_active_experts=640,
            moe_missing_experts=160,
            moe_fetched_experts=120,
        ),
        now=1.0,
    )

    moe = build_stats(_state(stats), p95_ms=0, ttft_mean_ms=0)["moe"]

    assert moe == {
        "scope": "decode_since_start_or_cache_rebuild",
        "layer_calls": 80,
        "active_experts": 640,
        "cache_hits": 480,
        "cache_misses": 160,
        "cache_hit_rate": 0.75,
        "fetched_experts": 120,
        "host_computed_experts": 40,
        "bytes_per_expert": 1024,
        "h2d_payload_bytes": 120 * 1024,
    }


def test_missing_moe_snapshot_is_null_and_does_not_replace_last_known_value():
    stats = StatsTracker()
    assert build_stats(_state(stats), p95_ms=0, ttft_mean_ms=0)["moe"] is None

    stats.observe(
        SimpleNamespace(
            uid=1,
            finished=False,
            moe_layer_calls=2,
            moe_active_experts=10,
            moe_missing_experts=4,
            moe_fetched_experts=4,
        ),
        now=1.0,
    )
    stats.observe(UserReply(uid=1, incremental_output="x", finished=False), now=2.0)

    moe = build_stats(_state(stats, bytes_per_expert=0), p95_ms=0, ttft_mean_ms=0)["moe"]
    assert moe["layer_calls"] == 2
    assert moe["cache_hit_rate"] == 0.6
    assert moe["h2d_payload_bytes"] == 0


def test_successful_moe_rebuild_starts_a_new_frontend_stats_epoch():
    state = _RebuildState(reply_status="ok")

    result = _run_rebuild(state, moe_cache_size=256)

    assert result["status"] == "ok"
    assert state.maintenance_state == "serving"
    assert state.moe_stats_reset_rebuilds == set()
    assert (
        state.stats.moe_layer_calls,
        state.stats.moe_active_experts,
        state.stats.moe_missing_experts,
        state.stats.moe_fetched_experts,
    ) == (0, 0, 0, 0)


def test_failed_moe_rebuild_preserves_the_last_valid_snapshot():
    state = _RebuildState(reply_status="failed")

    result = _run_rebuild(state, moe_cache_size=256)

    assert result["status"] == "failed"
    assert state.maintenance_state == "failed"
    assert state.moe_stats_reset_rebuilds == set()
    assert state.stats.moe_layer_calls == 80


def test_successful_non_moe_rebuild_does_not_reset_moe_stats():
    state = _RebuildState(reply_status="ok")

    result = _run_rebuild(state, moe_cache_size=None, num_pages=8192)

    assert result["status"] == "ok"
    assert state.moe_stats_reset_rebuilds == set()
    assert state.stats.moe_layer_calls == 80


def test_timed_out_moe_rebuild_keeps_epoch_intent_for_eventual_reply():
    state = _RebuildState(reply_status=None)

    result = _run_rebuild(state, moe_cache_size=256)

    assert result["status"] == "timeout"
    assert state.maintenance_state == "rebuilding"
    assert state.rebuild_futures == {}
    assert state.moe_stats_reset_rebuilds == {result["request_id"]}


def test_decode_counter_snapshot_uses_misses_as_fetches_only_for_regular_lru():
    common = {
        "prefill_hit_rows": 0,
        "prefill_total_rows": 0,
        "stat_fetched": torch.tensor(3, dtype=torch.int64),
    }
    regular = SimpleNamespace(
        decode_target="gpu",
        lru_stats=torch.tensor([[8, 3, 1], [7, 2, 1]], dtype=torch.int64),
        **common,
    )
    hybrid = SimpleNamespace(
        decode_target="hybrid",
        stat_active=torch.tensor(15, dtype=torch.int64),
        stat_missing=torch.tensor(5, dtype=torch.int64),
        stat_calls=torch.tensor(2, dtype=torch.int64),
        lru_stats=torch.zeros((2, 3), dtype=torch.int64),
        **common,
    )

    regular_stats = OffloadMoeCache.decode_miss_stats(regular)
    hybrid_stats = OffloadMoeCache.decode_miss_stats(hybrid)

    assert regular_stats["fetched_experts"] == regular_stats["missing_experts"] == 5
    assert hybrid_stats["missing_experts"] == 5
    assert hybrid_stats["fetched_experts"] == 3
    assert hybrid_stats["cpu_per_layer"] == 1.0


def test_control_cli_labels_h2d_as_payload_not_bandwidth():
    doc = {
        "model": {"id": "m", "ctx": 1, "attn": "mha", "moe": True},
        "throughput": {},
        "requests": {},
        "moe": {
            "cache_hit_rate": 0.75,
            "cache_hits": 12,
            "cache_misses": 4,
            "fetched_experts": 3,
            "host_computed_experts": 1,
            "h2d_payload_bytes": 3072,
            "bytes_per_expert": 1024,
            "scope": "decode_since_start_or_cache_rebuild",
        },
    }

    rendered = _format_stats(doc)

    assert "hit_rate=75.00%" in rendered
    assert "moe_h2d payload_bytes=3072" in rendered
    assert "bandwidth" not in rendered
