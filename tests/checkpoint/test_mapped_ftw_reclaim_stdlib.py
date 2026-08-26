from __future__ import annotations

import importlib.util
import json
import mmap
import os
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python" / "freetoken" / "checkpoint" / "mapped_ftw_core.py"
)
SPEC = importlib.util.spec_from_file_location("mapped_ftw_core_reclaim", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


def _vma_rss_bytes(path: Path) -> int:
    smaps = Path("/proc/self/smaps")
    if not smaps.is_file():
        raise unittest.SkipTest("/proc/self/smaps unavailable")
    target = str(path.resolve())
    total_kib = 0
    active = False
    for line in smaps.read_text(encoding="utf-8", errors="replace").splitlines():
        if line and "-" in line.split(maxsplit=1)[0]:
            active = target in line
            continue
        if active and line.startswith("Rss:"):
            total_kib += int(line.split()[1])
    return total_kib * 1024


class MappedFTWReclaimTests(unittest.TestCase):
    def test_clean_file_backed_pages_can_be_dropped_after_faulting(self):
        if not hasattr(mmap.mmap, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
            self.skipTest("mmap MADV_DONTNEED unavailable")
        if os.name != "posix":
            self.skipTest("Linux/POSIX residency probe only")

        size = 64 * 1024 * 1024
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shard = root / "freetoken-00000.ftw"
            with shard.open("wb") as handle:
                handle.truncate(size)
            (root / CORE.INDEX_NAME).write_text(json.dumps({
                "format": "freetoken_weight",
                "version": 1,
                "tensors": [{
                    "name": "gate_up_packed#L00000",
                    "kind": "experts_bank",
                    "dtype": "uint8",
                    "shape": [size],
                    "global_off": 0,
                    "nbytes": size,
                }],
                "shards": [{"file": shard.name, "global_off": 0, "nbytes": size}],
            }), encoding="utf-8")

            owner = CORE.map_ftw_range(root, "gate_up_packed#L00000")
            rss_mapped = _vma_rss_bytes(shard)
            # Mapping is virtual-address reservation; it must not eagerly materialize 64 MiB.
            self.assertLess(rss_mapped, 8 * 1024 * 1024)

            checksum = 0
            page = mmap.PAGESIZE
            for offset in range(owner.data_offset, owner.data_offset + size, page):
                checksum ^= owner.mapping[offset]
            self.assertEqual(checksum, 0)
            rss_faulted = _vma_rss_bytes(shard)
            self.assertGreater(rss_faulted, 32 * 1024 * 1024)

            owner.mapping.madvise(mmap.MADV_DONTNEED)
            rss_dropped = _vma_rss_bytes(shard)
            self.assertLess(rss_dropped, 8 * 1024 * 1024)
            self.assertLess(rss_dropped, rss_faulted // 4)


if __name__ == "__main__":
    unittest.main()
