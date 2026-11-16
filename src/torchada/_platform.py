"""
Platform detection module for torchada.

Detects whether the current environment supports CUDA (NVIDIA) or MUSA (Moore Threads).
"""

import os
from enum import Enum
from functools import lru_cache


class Platform(Enum):
    """Supported GPU platforms."""

    CUDA = "cuda"
    MUSA = "musa"
    CPU = "cpu"


@lru_cache(maxsize=1)
def detect_platform() -> Platform:
    """
    Detect the current GPU platform.

    Priority:
    1. TORCHADA_PLATFORM environment variable (force specific platform)
    2. MUSA availability (Moore Threads GPU)
    3. CUDA availability (NVIDIA GPU)
    4. CPU fallback

    Returns:
        Platform: The detected or configured platform.
    """
    # Check for forced platform via environment variable
    forced_platform = os.environ.get("TORCHADA_PLATFORM", "").lower()
    if forced_platform == "cuda":
        return Platform.CUDA
    elif forced_platform == "musa":
        return Platform.MUSA
    elif forced_platform == "cpu":
        return Platform.CPU

    # Auto-detect platform
    # Check MUSA first (Moore Threads)
    if _is_musa_available():
        return Platform.MUSA

    # Check CUDA (NVIDIA)
    if _is_cuda_available():
        return Platform.CUDA

    # Fallback to CPU
    return Platform.CPU


def _is_musa_available() -> bool:
    """
    Check if this is a MUSA (Moore Threads) platform.

    Detects the MUSA platform by checking if torch_musa is installed,
    rather than requiring a GPU to be present. This allows torchada to
    work correctly in environments where torch_musa is installed but
    no Moore Threads GPU card is available (e.g., build-only environments,
    CI/CD, CPU-only testing).

    Detection signals (in order):
    1. torch.version.musa is set (torch was built with MUSA support)
    2. torch_musa is importable
    """
    try:
        import torch

        # Primary signal: torch.version.musa is set by torch_musa at build time.
        # This is the most reliable indicator that we're on a MUSA platform,
        # regardless of whether a GPU card is present.
        if hasattr(torch.version, "musa") and torch.version.musa is not None:
            return True

        # Secondary signal: torch_musa is importable
        try:
            import torch_musa  # noqa: F401

            return True
        except ImportError:
            pass

        return False
    except ImportError:
        return False


def _is_cuda_available() -> bool:
    """Check if CUDA (NVIDIA) is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except (ImportError, AttributeError):
        return False


def is_musa_platform() -> bool:
    """Check if we're on MUSA platform."""
    return detect_platform() == Platform.MUSA


def is_cuda_platform() -> bool:
    """Check if we're on CUDA platform."""
    return detect_platform() == Platform.CUDA


def is_cpu_platform() -> bool:
    """Check if we're on CPU-only platform."""
    return detect_platform() == Platform.CPU


def get_device_name() -> str:
    """Get the device name string ('cuda', 'musa', or 'cpu')."""
    return detect_platform().value


def get_torch_device_module():
    """
    Get the appropriate torch device module (torch.cuda or torch.musa).

    Returns:
        The torch.cuda or torch.musa module.

    Raises:
        RuntimeError: If no GPU platform is available.
    """
    platform = detect_platform()

    if platform == Platform.MUSA:
        import torch

        return torch.musa
    elif platform == Platform.CUDA:
        import torch

        return torch.cuda
    else:
        raise RuntimeError("No GPU platform available. Running on CPU only.")


def is_gpu_device(device) -> bool:
    """
    Check if a device is a GPU device (either CUDA or MUSA).

    This is a helper function for code that needs to check if a device
    is a GPU device. On MUSA platform, device.type == "cuda" comparisons
    fail because the device type is "musa", not "cuda".

    Use this function instead of `device.type == "cuda"` for portable code.

    Args:
        device: A torch.device object, a device string, or an object exposing
            the tensor-style ``is_cuda`` compatibility attribute

    Returns:
        True if the device is cuda or musa, False otherwise

    Example:
        # Instead of:
        #   if tensor.device.type == "cuda":
        # Use:
        #   if torchada.is_gpu_device(tensor):
        #       ...
    """
    import torch

    # Preserve torchada's tensor is_cuda compatibility contract. In particular,
    # MUSA tensors report is_cuda=True while their physical device type is musa.
    is_cuda = getattr(device, "is_cuda", None)
    if isinstance(is_cuda, bool):
        return is_cuda

    # Handle modules and other objects that expose a device attribute.
    if hasattr(device, "device"):
        device = device.device

    # Handle torch.device objects
    if isinstance(device, torch.device):
        return device.type in ("cuda", "musa")

    # Handle string device specs
    if isinstance(device, str):
        return (
            device == "cuda"
            or device.startswith("cuda:")
            or device == "musa"
            or device.startswith("musa:")
        )

    return False


def is_cuda_like_device(device) -> bool:
    """Alias for is_gpu_device() for clarity."""
    return is_gpu_device(device)
