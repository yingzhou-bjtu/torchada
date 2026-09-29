"""Graph-capturable MUSA FFT helpers backed by the platform muFFT library."""

from __future__ import annotations

import atexit
import os
import os.path as osp
import threading
from types import ModuleType
from typing import Callable, cast

import torch

MufftIrfft = Callable[[torch.Tensor, int], torch.Tensor]
mufft_irfft: MufftIrfft | None = None
mufft_clear_plans: Callable[[], None] | None = None
mufft_load_lock = threading.Lock()


def load_mufft_ops() -> MufftIrfft:
    """Build and load the small muFFT extension used by graph-captured audio."""
    global mufft_clear_plans, mufft_irfft
    if mufft_irfft is None:
        with mufft_load_lock:
            if mufft_irfft is None:
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
                mufft_irfft = cast(MufftIrfft, module.musa_irfft_graph)
                mufft_clear_plans = cast(Callable[[], None], module.clear_mufft_plans)
    assert mufft_irfft is not None
    return mufft_irfft


def clear_mufft_plans() -> None:
    if mufft_clear_plans is not None:
        mufft_clear_plans()


def musa_irfft_graph(spectrum: torch.Tensor, n: int) -> torch.Tensor:
    """Run inverse real FFT through the graph-capturable muFFT implementation."""
    return load_mufft_ops()(spectrum, int(n))


atexit.register(clear_mufft_plans)
