from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from aiohttp import web
import folder_paths
import torch

import comfy.model_management as model_management
from server import PromptServer


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web/js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]


PATCH_VERSION = "2.0.1"
LOG_PREFIX = "[AVS XPU VRAM Fix]"
CONFIG_FILENAME = "config.json"
CONFIG_KEY = "vram_fix_enabled"

# Capture the functions that were present before AVS installs its wrappers.
# When AVS is disabled, calls are delegated to these functions immediately.
_original_get_total_memory = model_management.get_total_memory
_original_get_free_memory = model_management.get_free_memory

_patch_installed = False
_enabled = True
_last_backend: str | None = None
_shown_warnings: set[str] = set()


def _warn_once(key: str, message: str, *args: object) -> None:
    if key in _shown_warnings:
        return
    _shown_warnings.add(key)
    logging.warning("%s " + message, LOG_PREFIX, *args)


def _set_backend(name: str) -> None:
    global _last_backend
    if _last_backend == name:
        return
    _last_backend = name
    logging.info("%s Active memory backend: %s", LOG_PREFIX, name)


def _config_path() -> Path:
    """Store the runtime toggle outside the custom-node repository."""
    try:
        user_dir = Path(folder_paths.get_user_directory())
    except Exception:
        # Compatibility fallback for older/non-standard ComfyUI layouts.
        user_dir = Path(__file__).resolve().parent
    return user_dir / "avs_xpu_vram_fix" / CONFIG_FILENAME


def _load_enabled_state() -> bool:
    path = _config_path()
    try:
        if not path.is_file():
            return True
        data = json.loads(path.read_text(encoding="utf-8"))
        value = data.get(CONFIG_KEY, True)
        return value if isinstance(value, bool) else True
    except Exception as error:
        _warn_once(
            "config_read_failed",
            "Could not read saved setting from %s: %s. Defaulting to enabled.",
            path,
            error,
        )
        return True


def _save_enabled_state(enabled: bool) -> None:
    path = _config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps({CONFIG_KEY: enabled}, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    except Exception as error:
        _warn_once(
            "config_write_failed",
            "Could not persist setting to %s: %s.",
            path,
            error,
        )


def _set_enabled(enabled: bool, *, persist: bool = True) -> None:
    global _enabled

    enabled = bool(enabled)
    changed = _enabled != enabled
    _enabled = enabled

    if persist:
        _save_enabled_state(enabled)

    if changed:
        if enabled:
            logging.info("%s VRAM fix enabled.", LOG_PREFIX)
        else:
            logging.info(
                "%s VRAM fix disabled. Original ComfyUI memory accounting is active.",
                LOG_PREFIX,
            )


def _is_xpu_device(dev: Any) -> bool:
    checker = getattr(model_management, "is_device_xpu", None)
    if callable(checker):
        try:
            return bool(checker(dev))
        except Exception:
            pass
    return getattr(dev, "type", None) == "xpu"


def _xpu_get_memory_info(dev, mem_reserved):
    """PR #15487-compatible guarded Intel XPU memory query."""
    try:
        mem_free_xpu, mem_total_xpu = torch.xpu.mem_get_info(dev)
        if (
            mem_total_xpu <= 0
            or mem_free_xpu < 0
            or mem_free_xpu > mem_total_xpu
            or mem_total_xpu - mem_free_xpu < mem_reserved
        ):
            raise RuntimeError("Inconsistent XPU memory information")
        _set_backend("torch.xpu.mem_get_info")
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        # This is intentionally the same fallback principle as PR #15487.
        mem_total_xpu = torch.xpu.get_device_properties(dev).total_memory
        mem_free_xpu = mem_total_xpu - mem_reserved
        _set_backend("ComfyUI XPU fallback")
        _warn_once(
            "mem_get_info_fallback",
            "torch.xpu.mem_get_info() is unavailable or inconsistent (%s). "
            "Using the ComfyUI XPU fallback.",
            error,
        )

    return mem_free_xpu, mem_total_xpu


def _patched_get_total_memory(dev=None, torch_total_too: bool = False):
    if dev is None:
        dev = model_management.get_torch_device()

    if not _enabled or not _is_xpu_device(dev):
        return _original_get_total_memory(dev, torch_total_too)

    try:
        stats = torch.xpu.memory_stats(dev)
        mem_reserved = stats["reserved_bytes.all.current"]
        _, mem_total_xpu = _xpu_get_memory_info(dev, mem_reserved)

        if torch_total_too:
            return mem_total_xpu, mem_reserved
        return mem_total_xpu
    except Exception as error:
        # The standalone patch must never break ComfyUI memory queries if the
        # runtime/API shape changes outside the conditions handled by the PR.
        _warn_once(
            "total_memory_wrapper_failed",
            "Patched get_total_memory() failed (%s). Using the original function.",
            error,
        )
        return _original_get_total_memory(dev, torch_total_too)


def _patched_get_free_memory(dev=None, torch_free_too: bool = False):
    if dev is None:
        dev = model_management.get_torch_device()

    if not _enabled or not _is_xpu_device(dev):
        return _original_get_free_memory(dev, torch_free_too)

    try:
        stats = torch.xpu.memory_stats(dev)
        mem_active = stats["active_bytes.all.current"]
        mem_reserved = stats["reserved_bytes.all.current"]

        mem_free_xpu, mem_total_xpu = _xpu_get_memory_info(dev, mem_reserved)
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = min(mem_free_xpu + mem_free_torch, mem_total_xpu)

        if torch_free_too:
            return mem_free_total, mem_free_torch
        return mem_free_total
    except Exception as error:
        _warn_once(
            "free_memory_wrapper_failed",
            "Patched get_free_memory() failed (%s). Using the original function.",
            error,
        )
        return _original_get_free_memory(dev, torch_free_too)


def _install_patch() -> None:
    global _patch_installed

    if _patch_installed:
        return

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        logging.info("%s Intel XPU is unavailable; patch not installed.", LOG_PREFIX)
        return

    device = model_management.get_torch_device()
    if not _is_xpu_device(device):
        logging.info("%s Current ComfyUI device is not XPU: %s", LOG_PREFIX, device)
        return

    current_free = model_management.get_free_memory
    current_total = model_management.get_total_memory

    already_free = getattr(current_free, "_avs_xpu_vram_fix", False)
    already_total = getattr(current_total, "_avs_xpu_vram_fix", False)
    if already_free and already_total:
        logging.info("%s Patch is already installed.", LOG_PREFIX)
        _patch_installed = True
        return

    setattr(_patched_get_free_memory, "_avs_xpu_vram_fix", True)
    setattr(_patched_get_total_memory, "_avs_xpu_vram_fix", True)

    model_management.get_free_memory = _patched_get_free_memory
    model_management.get_total_memory = _patched_get_total_memory
    _patch_installed = True

    if _enabled:
        try:
            available, reusable = _patched_get_free_memory(device, torch_free_too=True)
            total = _patched_get_total_memory(device)
            logging.info(
                "%s v%s installed and enabled. Current usable free: %.2f GiB; "
                "reusable PyTorch cache: %.2f GiB; total: %.2f GiB.",
                LOG_PREFIX,
                PATCH_VERSION,
                available / 1024**3,
                reusable / 1024**3,
                total / 1024**3,
            )
        except Exception:
            logging.exception("%s Initial diagnostic query failed.", LOG_PREFIX)
    else:
        logging.info(
            "%s v%s installed but disabled by saved setting. "
            "Original ComfyUI memory accounting is active.",
            LOG_PREFIX,
            PATCH_VERSION,
        )


def _status_payload() -> dict[str, Any]:
    return {
        "enabled": _enabled,
        "installed": _patch_installed,
        "version": PATCH_VERSION,
        "backend": _last_backend if _enabled else "original ComfyUI (fix disabled)",
        "config_path": str(_config_path()),
    }


@PromptServer.instance.routes.get("/avs-xpu-vram-fix/status")
async def avs_xpu_vram_fix_status(_request):
    return web.json_response(_status_payload())


@PromptServer.instance.routes.post("/avs-xpu-vram-fix/enabled")
async def avs_xpu_vram_fix_set_enabled(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        return web.json_response(
            {"error": "'enabled' must be a boolean"},
            status=400,
        )

    _set_enabled(enabled, persist=True)
    return web.json_response(_status_payload())


_enabled = _load_enabled_state()
_install_patch()
