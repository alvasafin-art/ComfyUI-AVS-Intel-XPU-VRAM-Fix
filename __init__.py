from __future__ import annotations

import logging
from typing import Callable

import torch
import comfy.model_management as model_management


# This is a startup patch, not a visual node.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


LOG_PREFIX = "[XPU Global VRAM V2]"

MIB = 1024**2

# A tolerance is required because allocator statistics and global
# memory telemetry may not be updated at exactly the same time.
VALIDATION_TOLERANCE = 512 * MIB

_original_get_free_memory = model_management.get_free_memory

_last_backend: str | None = None
_shown_warnings: set[str] = set()


def _warn_once(key: str, message: str, *args: object) -> None:
    """Show each fallback/error warning only once."""

    if key in _shown_warnings:
        return

    _shown_warnings.add(key)
    logging.warning("%s " + message, LOG_PREFIX, *args)


def _set_backend(name: str) -> None:
    """Log only when the active telemetry backend changes."""

    global _last_backend

    if _last_backend == name:
        return

    _last_backend = name
    logging.warning(
        "%s Active memory backend: %s",
        LOG_PREFIX,
        name,
    )


def _get_mem_get_info_function() -> Callable:
    """
    Find torch.xpu.mem_get_info across PyTorch API layouts.

    Depending on the PyTorch build, it may be exported directly
    through torch.xpu or through torch.xpu.memory.
    """

    direct_function = getattr(torch.xpu, "mem_get_info", None)

    if callable(direct_function):
        return direct_function

    memory_module = getattr(torch.xpu, "memory", None)
    memory_function = getattr(
        memory_module,
        "mem_get_info",
        None,
    )

    if callable(memory_function):
        return memory_function

    raise AttributeError(
        "torch.xpu.mem_get_info() is unavailable"
    )


def _get_allocator_memory(
    device: torch.device,
) -> tuple[int, int, int]:
    """
    Return:
        active memory,
        reserved memory,
        reusable reserved cache.
    """

    stats = torch.xpu.memory_stats(device)

    active = int(
        stats.get("active_bytes.all.current", 0)
    )
    reserved = int(
        stats.get("reserved_bytes.all.current", 0)
    )

    # Defensive protection against inconsistent transient statistics.
    if reserved < active:
        reserved = active

    reusable = reserved - active

    return active, reserved, reusable


def _validate_mem_get_info(
    *,
    global_free: int,
    global_total: int,
    active: int,
    allocatable_total: int,
) -> None:
    """Reject impossible or obviously stale driver values."""

    if global_total <= 0:
        raise RuntimeError(
            f"invalid total memory: {global_total}"
        )

    if global_free < 0:
        raise RuntimeError(
            f"negative free memory: {global_free}"
        )

    if global_free > global_total:
        raise RuntimeError(
            "free memory is larger than total memory: "
            f"{global_free} > {global_total}"
        )

    # Physical total and allocatable total may differ slightly,
    # but a very large mismatch indicates incompatible telemetry.
    minimum_expected_total = int(
        allocatable_total * 0.75
    )
    maximum_expected_total = int(
        allocatable_total * 1.25
    )

    if not (
        minimum_expected_total
        <= global_total
        <= maximum_expected_total
    ):
        raise RuntimeError(
            "global total differs too much from allocatable total: "
            f"{global_total} vs {allocatable_total}"
        )

    global_used = global_total - global_free

    # Active PyTorch tensors must be represented inside globally
    # occupied device memory. If they are not, mem_get_info is
    # likely returning stale or incorrect values.
    if active > global_used + VALIDATION_TOLERANCE:
        raise RuntimeError(
            "global telemetry does not include current PyTorch "
            "allocations: "
            f"active={active}, global_used={global_used}"
        )


def _calculate_with_mem_get_info(
    device: torch.device,
    active: int,
    reusable: int,
    allocatable_total: int,
) -> int:
    """
    Preferred path.

    This mirrors the CUDA calculation used by ComfyUI:
        global free + reusable PyTorch cache.
    """

    mem_get_info = _get_mem_get_info_function()

    global_free, global_total = mem_get_info(device)

    global_free = int(global_free)
    global_total = int(global_total)

    _validate_mem_get_info(
        global_free=global_free,
        global_total=global_total,
        active=active,
        allocatable_total=allocatable_total,
    )

    available = global_free + reusable

    # ComfyUI should never be told that more memory is usable than
    # PyTorch reports as allocatable for this device.
    available = max(
        0,
        min(allocatable_total, available),
    )

    _set_backend("torch.xpu.mem_get_info")

    return available


def _patched_get_free_memory(
    dev=None,
    torch_free_too: bool = False,
):
    """
    Use device-wide XPU memory telemetry for Intel GPUs.

    Other device types continue using the original ComfyUI function.
    """

    if dev is None:
        dev = model_management.get_torch_device()

    if not model_management.is_device_xpu(dev):
        return _original_get_free_memory(
            dev,
            torch_free_too,
        )

    try:
        active, reserved, reusable = (
            _get_allocator_memory(dev)
        )

        allocatable_total = int(
            torch.xpu.get_device_properties(
                dev
            ).total_memory
        )

    except Exception as error:
        _warn_once(
            "allocator_stats_failed",
            "Cannot read XPU allocator statistics. "
            "Using original ComfyUI calculation: %s",
            error,
        )

        _set_backend("original ComfyUI fallback")

        return _original_get_free_memory(
            dev,
            torch_free_too,
        )

    try:
        available = _calculate_with_mem_get_info(
            device=dev,
            active=active,
            reusable=reusable,
            allocatable_total=allocatable_total,
        )

    except Exception as mem_get_info_error:
        _warn_once(
            "mem_get_info_failed",
            "mem_get_info is unavailable or invalid: %s. "
            "Using original ComfyUI calculation.",
            mem_get_info_error,
        )

        _set_backend("original ComfyUI fallback")

        return _original_get_free_memory(
            dev,
            torch_free_too,
        )

    if torch_free_too:
        return available, reusable

    return available


def _install_patch() -> None:
    """Install the patch once during ComfyUI startup."""

    current_function = model_management.get_free_memory

    if getattr(
        current_function,
        "_xpu_global_vram_v2",
        False,
    ):
        logging.warning(
            "%s Patch is already installed.",
            LOG_PREFIX,
        )
        return

    if (
        not hasattr(torch, "xpu")
        or not torch.xpu.is_available()
    ):
        logging.warning(
            "%s Intel XPU is unavailable.",
            LOG_PREFIX,
        )
        return

    device = model_management.get_torch_device()

    if not model_management.is_device_xpu(device):
        logging.warning(
            "%s Current ComfyUI device is not XPU: %s",
            LOG_PREFIX,
            device,
        )
        return

    setattr(
        _patched_get_free_memory,
        "_xpu_global_vram_v2",
        True,
    )

    model_management.get_free_memory = (
        _patched_get_free_memory
    )

    try:
        available, reusable = (
            _patched_get_free_memory(
                device,
                torch_free_too=True,
            )
        )

        allocatable_total = int(
            torch.xpu.get_device_properties(
                device
            ).total_memory
        )

        logging.warning(
            "%s Patch installed. "
            "Current usable free: %.2f GiB; "
            "reusable PyTorch cache: %.2f GiB; "
            "allocatable total: %.2f GiB.",
            LOG_PREFIX,
            available / 1024**3,
            reusable / 1024**3,
            allocatable_total / 1024**3,
        )

    except Exception:
        # The patched function already has safe fallbacks.
        logging.exception(
            "%s Initial diagnostic query failed.",
            LOG_PREFIX,
        )


_install_patch()