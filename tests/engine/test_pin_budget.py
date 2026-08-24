from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

import freetoken.engine.engine as engine


_GIB = 2**30


class _FakeGlobalMemoryStatusEx:
    """Small Win32 stand-in that also checks MEMORYSTATUSEX's ABI size.

    DWORD is 32-bit on Windows even when the test itself runs on an LP64 Linux host,
    so a portable ctypes declaration should still produce the documented 64-byte
    structure here.
    """

    argtypes = None
    restype = None

    def __init__(self, *, total_phys: int, avail_phys: int = 0, succeeds: bool = True):
        self.total_phys = total_phys
        self.avail_phys = avail_phys
        self.succeeds = succeeds
        self.seen_length: int | None = None

    def __call__(self, ptr) -> int:
        status = ptr._obj
        self.seen_length = int(status.dwLength)
        if not self.succeeds or self.seen_length != 64:
            return 0
        status.ullTotalPhys = self.total_phys
        status.ullAvailPhys = self.avail_phys
        return 1


def _set_platform(monkeypatch: pytest.MonkeyPatch, *, name: str, release: str | None) -> None:
    monkeypatch.setattr(engine.os, "name", name)
    if release is None:
        monkeypatch.delattr(engine.os, "uname", raising=False)
    else:
        monkeypatch.setattr(
            engine.os,
            "uname",
            lambda: SimpleNamespace(release=release),
            raising=False,
        )


def test_pin_budget_env_override_precedes_platform_detection(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "3.5")
    _set_platform(monkeypatch, name="nt", release=None)

    assert engine._pin_budget_bytes() == int(3.5 * _GIB)


def test_pin_budget_native_windows_uses_total_physical_memory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    _set_platform(monkeypatch, name="nt", release=None)

    query = _FakeGlobalMemoryStatusEx(total_phys=64 * _GIB, avail_phys=7 * _GIB)
    monkeypatch.setattr(
        ctypes,
        "windll",
        SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=query)),
        raising=False,
    )

    assert engine._pin_budget_bytes() == int(64 * _GIB * 0.4)
    assert query.seen_length == 64


def test_pin_budget_wsl_keeps_sysconf_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    _set_platform(monkeypatch, name="posix", release="6.8.0-microsoft-standard-WSL2")

    values = {
        "SC_PHYS_PAGES": 16 * 1024 * 1024,
        "SC_PAGE_SIZE": 4096,
    }
    monkeypatch.setattr(engine.os, "sysconf", values.__getitem__)

    assert engine._pin_budget_bytes() == int(values["SC_PHYS_PAGES"] * 4096 * 0.4)


def test_pin_budget_plain_linux_remains_uncapped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    _set_platform(monkeypatch, name="posix", release="6.8.0-64-generic")

    assert engine._pin_budget_bytes() is None


def test_pin_budget_native_windows_query_failure_is_not_treated_as_uncapped(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    _set_platform(monkeypatch, name="nt", release=None)

    query = _FakeGlobalMemoryStatusEx(total_phys=64 * _GIB, succeeds=False)
    monkeypatch.setattr(
        ctypes,
        "windll",
        SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=query)),
        raising=False,
    )
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    monkeypatch.setattr(
        ctypes,
        "WinError",
        lambda code=5: OSError(code, "GlobalMemoryStatusEx failed"),
        raising=False,
    )

    with pytest.raises(OSError):
        engine._pin_budget_bytes()
