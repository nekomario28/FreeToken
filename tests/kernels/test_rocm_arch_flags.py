from freetoken.kernel import utils


def test_hip_cflags_expand_default_arches(monkeypatch):
    monkeypatch.delenv("FREETOKEN_ROCM_ARCH", raising=False)

    flags = utils._hip_cflags([])

    assert flags[-4:] == [
        "--offload-arch=gfx1100",
        "--offload-arch=gfx1101",
        "--offload-arch=gfx1102",
        "--offload-arch=gfx1103",
    ]
    assert all(";" not in flag for flag in flags)


def test_hip_cflags_accept_common_arch_separators(monkeypatch):
    monkeypatch.setenv(
        "FREETOKEN_ROCM_ARCH",
        "gfx1100; gfx1101,gfx1102 gfx1103",
    )

    flags = utils._hip_cflags(["-DFOO=1"])

    assert flags == [
        "-std=c++20",
        "-O3",
        "-DFOO=1",
        "--offload-arch=gfx1100",
        "--offload-arch=gfx1101",
        "--offload-arch=gfx1102",
        "--offload-arch=gfx1103",
    ]
