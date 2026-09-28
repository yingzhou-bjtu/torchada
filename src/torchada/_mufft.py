"""Graph-capturable MUSA FFT helpers backed by the platform muFFT library."""

from __future__ import annotations

import os
import os.path as osp
from typing import Any

_module: Any | None = None


def load_mufft_ops(force_reload: bool = False) -> Any:
    """Build and load the small muFFT extension used by graph-captured audio."""
    global _module
    if _module is not None and not force_reload:
        return _module

    import torch_musa
    from torch.utils.cpp_extension import load

    package_dir = osp.dirname(__file__)
    source = osp.join(package_dir, "_mufft_src", "musa_irfft.cpp")
    torch_musa_dir = osp.dirname(torch_musa.__file__)
    torch_musa_parent = osp.dirname(torch_musa_dir)
    musa_home = os.environ.get("MUSA_HOME", "/usr/local/musa")
    torch_musa_lib = osp.join(torch_musa_dir, "lib")

    _module = load(
        name="torchada_mufft_ops",
        sources=[source],
        extra_include_paths=[
            torch_musa_parent,
            osp.join(torch_musa_dir, "csrc"),
            osp.join(musa_home, "include"),
        ],
        extra_cflags=["-O2"],
        extra_ldflags=[
            "-L" + osp.join(musa_home, "lib"),
            "-lmufft",
            "-L" + torch_musa_lib,
            "-lmusa_python",
            "-Wl,-rpath," + osp.join(musa_home, "lib"),
            "-Wl,-rpath," + torch_musa_lib,
        ],
    )
    return _module


def musa_irfft_graph(spectrum: Any, n: int) -> Any:
    """Run inverse real FFT through the graph-capturable muFFT implementation."""
    return load_mufft_ops().musa_irfft_graph(spectrum, int(n))
