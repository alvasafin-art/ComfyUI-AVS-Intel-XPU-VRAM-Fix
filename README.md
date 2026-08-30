# ComfyUI-AVS-Intel-XPU-VRAM-Fix

**Intel XPU / Intel Arc VRAM accounting fix for ComfyUI, aligned with ComfyUI PR #15487.**

This custom node is a startup/runtime patch for Intel XPU. It does not add workflow nodes. It changes how ComfyUI reads available XPU VRAM so model loading/offloading decisions can use device-wide free-memory information when `torch.xpu.mem_get_info()` is trustworthy.

The patch was originally created after reproducible Intel Arc B580 failures including KSampler staying at 0%, freezes during heavy model switching, VRAM overflow, and crashes under memory pressure.

## What changed in v2.0.1

The standalone patch follows the same guarded logic and operating principle as [ComfyUI PR #15487](https://github.com/Comfy-Org/ComfyUI/pull/15487):

- uses one shared guarded XPU memory helper;
- calls `torch.xpu.mem_get_info(dev)` when available;
- rejects the result when total memory is non-positive, free memory is negative, free exceeds total, or globally used VRAM is smaller than PyTorch's reserved allocator memory;
- falls back to the existing ComfyUI XPU calculation when `mem_get_info()` is unavailable or inconsistent;
- uses the same helper for both `get_total_memory()` and `get_free_memory()`;
- calculates reusable PyTorch cache as `reserved - active`;
- caps usable free memory at the XPU total with `min(free + reusable_cache, total)`.

The older standalone-only validation heuristics were removed so the core memory logic stays close to the PR.

## New AVS XPU setting

Open:

**ComfyUI → Settings → Application settings → AVS XPU**

Use **Enable Intel XPU VRAM fix** to turn the fix on or off.

- **On** — guarded `torch.xpu.mem_get_info()` path is used, with the PR-compatible fallback.
- **Off** — the extension immediately delegates to the original ComfyUI `get_total_memory()` and `get_free_memory()` functions. You do not need to delete or uninstall the custom node.
- The state is persisted under the ComfyUI `user/avs_xpu_vram_fix/` directory and survives restart.

The frontend setting is implemented with ComfyUI's extension settings API and appears under the dedicated **AVS XPU** category.

In v2.0.1 the settings callback reads the new boolean value from the correct callback argument. This fixes the v2.0.0 bug that could invert an Off action or send `false` during frontend initialization. Both ComfyUI's persisted UI setting and the backend `user/avs_xpu_vram_fix/config.json` state therefore stay aligned after the setting is changed.

## Why the fix exists

The stock Intel XPU path in affected ComfyUI versions estimates free VRAM mainly from PyTorch allocator statistics. That can miss memory already occupied outside PyTorch's allocator by the driver, desktop, runtime, other applications, or other GPU allocations.

The preferred patched calculation is:

```text
global free XPU VRAM
+ reusable PyTorch reserved cache
= usable free VRAM reported to ComfyUI
```

If the global XPU telemetry cannot be trusted, the patch uses the original ComfyUI calculation instead.

## What it does not change

It does not modify model weights, image quality, sampling, KSampler logic, attention implementations, quantization, model architecture, or the normal CUDA/AMD/CPU paths. The wrappers act only on XPU devices.

## Installation

No additional Python packages are required.

Clone into `ComfyUI/custom_nodes/`:

```bat
cd /d "PATH\TO\ComfyUI\custom_nodes"
git clone https://github.com/alvasafin-art/ComfyUI-AVS-Intel-XPU-VRAM-Fix.git
```

Restart ComfyUI and hard-refresh the browser once so the frontend extension is loaded.

Folder layout:

```text
ComfyUI/custom_nodes/ComfyUI-AVS-Intel-XPU-VRAM-Fix/
    __init__.py
    README.md
    CHANGELOG.md
    LICENSE
    web/
        js/
            avs_xpu_settings.js
```

## Updating

```bat
cd /d "PATH\TO\ComfyUI\custom_nodes\ComfyUI-AVS-Intel-XPU-VRAM-Fix"
git pull
```

Restart ComfyUI after updating Python code. A hard browser refresh may be needed when frontend JavaScript changes.

## Startup log

When enabled, expect messages similar to:

```text
[AVS XPU VRAM Fix] Active memory backend: torch.xpu.mem_get_info
[AVS XPU VRAM Fix] v2.0.1 installed and enabled. Current usable free: ...
```

If `mem_get_info()` is unsupported or inconsistent, the log reports the fallback once and continues with stock ComfyUI XPU accounting.

When disabled:

```text
[AVS XPU VRAM Fix] v2.0.1 installed but disabled by saved setting. Original ComfyUI memory accounting is active.
```

## Tested configuration

The original issue and patch were tested primarily on:

| Component | Configuration |
| --- | --- |
| GPU | Intel Arc B580 12 GB |
| OS | Windows 11 |
| ComfyUI | 0.31.1 during the original validation cycle |
| PyTorch | 2.13.0+xpu during the original validation cycle |

Additional Intel XPU hardware and newer PyTorch/ComfyUI versions should still be tested independently.

## Compatibility notes

- Intended for Intel XPU / Intel Arc.
- `torch.xpu.mem_get_info()` support varies across Intel devices and PyTorch stacks; the guarded fallback exists for this reason.
- If ComfyUI merges PR #15487 or equivalent XPU VRAM logic upstream, disable this custom fix first and test stock ComfyUI. The custom node can then be removed if no longer needed.
- Another custom node that replaces the same `comfy.model_management` functions can still conflict with this runtime patch.

## Technical references

- ComfyUI PR #15487: https://github.com/Comfy-Org/ComfyUI/pull/15487
- ComfyUI `model_management.py`: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_management.py
- ComfyUI frontend settings system: https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/docs/SETTINGS.md
- PyTorch XPU `mem_get_info`: https://docs.pytorch.org/docs/stable/generated/torch.xpu.memory.mem_get_info.html

## Credits

Special thanks to Simon Lui for his help with the XPU VRAM implementation, additional testing, and valuable feedback during the work on ComfyUI PR #15487.

His input helped improve the reliability and alignment of this patch with ComfyUI's XPU memory-management logic.

Development, debugging, testing analysis, and documentation were assisted by OpenAI ChatGPT. The Intel Arc B580 testing and upstream PR validation were performed by the repository author.

## License

GNU General Public License v3.0 (GPL-3.0).
