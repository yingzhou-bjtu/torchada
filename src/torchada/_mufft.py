"""Graph-capturable MUSA FFT helpers backed by the platform muFFT library."""

from __future__ import annotations

import os
import os.path as osp
from types import ModuleType
from typing import Callable, cast

import torch

_MufftIrfft = Callable[[torch.Tensor, int], torch.Tensor]
_mufft_irfft: _MufftIrfft | None = None


def load_mufft_ops() -> _MufftIrfft:
    """Build and load the small muFFT extension used by graph-captured audio."""
    global _mufft_irfft
    if _mufft_irfft is not None:
        return _mufft_irfft

    import torch_musa
    from torch.utils.cpp_extension import load

    package_dir = osp.dirname(__file__)
    source = osp.join(package_dir, "_mufft_src", "musa_irfft.cpp")
    torch_musa_dir = osp.dirname(torch_musa.__file__)
    torch_musa_parent = osp.dirname(torch_musa_dir)
    musa_home = os.environ.get("MUSA_HOME", "/usr/local/musa")
    torch_musa_lib = osp.join(torch_musa_dir, "lib")

    module: ModuleType = load(
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
    _mufft_irfft = cast(_MufftIrfft, module.musa_irfft_graph)
    return _mufft_irfft


def musa_irfft_graph(spectrum: torch.Tensor, n: int) -> torch.Tensor:
    """Run inverse real FFT through the graph-capturable muFFT implementation."""
    return load_mufft_ops()(spectrum, int(n))
