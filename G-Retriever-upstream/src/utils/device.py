"""Device-aware memory budgets and precision for Hugging Face models."""

from contextlib import contextmanager

import torch


def cuda_max_memory(limits):
    """Use automatic detection by default; restrict explicit budgets to visible GPUs.

    CUDA numbers visible devices from zero, including when CUDA_VISIBLE_DEVICES
    selects a subset of the physical GPUs. Extra budgets from older two-GPU
    configurations must not introduce nonexistent devices into the device map.
    """
    if limits is None:
        return None
    return {
        device: f"{size}GiB"
        for device, size in enumerate(limits[:torch.cuda.device_count()])
    } or None


@contextmanager
def model_autocast(device, dtype=torch.bfloat16):
    """Use CUDA autocast with float16 fallback on GPUs without bfloat16 support."""
    device = torch.device(device)
    if device.type != "cuda":
        yield
        return

    # Capability checks and autocast construction refer to the current GPU.
    with torch.cuda.device(device):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            dtype = torch.float16
        with torch.cuda.amp.autocast(dtype=dtype):
            yield
