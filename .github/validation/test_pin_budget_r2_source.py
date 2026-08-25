from __future__ import annotations

import ast
import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

ENGINE = Path("python/freetoken/engine/engine.py")


def load_pin_budget(fake_os):
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"), filename=str(ENGINE))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pin_budget_bytes")
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"os": fake_os}
    exec(compile(module, str(ENGINE), "exec"), ns)
    return ns["_pin_budget_bytes"]


GIB = 2**30

# Explicit user authority must win before platform probing.
override_os = SimpleNamespace(name="nt", environ={"FREETOKEN_PIN_BUDGET_GB": "3.5"})
assert load_pin_budget(override_os)() == int(3.5 * GIB)
print("PIN_BUDGET_EXPLICIT_OVERRIDE_PASS")

# WSL and plain-Linux behavior must remain unchanged.
values = {"SC_PHYS_PAGES": 16 * 1024 * 1024, "SC_PAGE_SIZE": 4096}
wsl_os = SimpleNamespace(
    name="posix",
    environ={},
    uname=lambda: SimpleNamespace(release="6.8.0-microsoft-standard-WSL2"),
    sysconf=values.__getitem__,
)
assert load_pin_budget(wsl_os)() == int(values["SC_PHYS_PAGES"] * 4096 * 0.4)
print("PIN_BUDGET_WSL_PASS")

linux_os = SimpleNamespace(
    name="posix",
    environ={},
    uname=lambda: SimpleNamespace(release="6.8.0-64-generic"),
)
assert load_pin_budget(linux_os)() is None
print("PIN_BUDGET_PLAIN_LINUX_PASS")

# On a real Windows runner, exercise the exact clean function against the real Win32 API
# before replacing ctypes.windll for the synthetic ABI/failure checks below.
if os.name == "nt":
    real_budget = load_pin_budget(os)()
    assert isinstance(real_budget, int) and real_budget > GIB
    print(f"NATIVE_WINDOWS_GLOBAL_MEMORY_STATUS_PASS budget={real_budget}")


class FakeGlobalMemoryStatusEx:
    argtypes = None
    restype = None

    def __init__(self, *, total_phys: int, succeeds: bool = True):
        self.total_phys = total_phys
        self.succeeds = succeeds
        self.seen_length = None

    def __call__(self, ptr):
        status = ptr._obj
        self.seen_length = int(status.dwLength)
        if not self.succeeds or self.seen_length != 64:
            return 0
        status.ullTotalPhys = self.total_phys
        return 1


orig_windll = getattr(ctypes, "windll", None)
try:
    query = FakeGlobalMemoryStatusEx(total_phys=64 * GIB)
    ctypes.windll = SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=query))
    fake_windows = SimpleNamespace(name="nt", environ={})
    assert load_pin_budget(fake_windows)() == int(64 * GIB * 0.4)
    assert query.seen_length == 64
    print("PIN_BUDGET_MEMORYSTATUSEX_ABI_PASS")

    failing = FakeGlobalMemoryStatusEx(total_phys=64 * GIB, succeeds=False)
    ctypes.windll = SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=failing))
    try:
        load_pin_budget(fake_windows)()
    except OSError as exc:
        assert "GlobalMemoryStatusEx failed" in str(exc)
    else:
        raise AssertionError("failed Win32 capacity query was silently treated as usable")
    print("PIN_BUDGET_WIN32_FAILURE_FAILS_CLOSED_PASS")
finally:
    if orig_windll is None:
        try:
            del ctypes.windll
        except AttributeError:
            pass
    else:
        ctypes.windll = orig_windll
