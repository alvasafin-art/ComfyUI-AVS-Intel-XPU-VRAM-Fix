# ComfyUI-AVS-Intel-XPU-VRAM-Fix

**Fix ComfyUI Intel Arc VRAM issues — Intel XPU free-VRAM detection, model-switch freezes, KSampler 0% hangs and crashes**

For Intel Arc / Intel XPU users experiencing KSampler stuck at 0%, VRAM overflow, crashes, or freezes when switching between heavy models.

This is a small startup patch for **ComfyUI running on Intel XPU GPUs**. It does not add a visual node to the workflow. It only changes how ComfyUI answers one important question:

> **How much VRAM is actually safe to use right now?**

On my Intel Arc B580, this patch helped eliminate several stability problems in heavy workflows: KSampler getting stuck at 0%, freezes while switching large models, and ComfyUI crashes during VRAM pressure.

## The problem in simple terms

ComfyUI needs to know how much VRAM is free before it decides whether to load more model data, keep a model on the GPU, or move something back to system RAM.

In ComfyUI 0.31.1, the Intel XPU path estimates free VRAM mainly from **PyTorch's own allocator statistics**. That tells ComfyUI how much memory PyTorch is actively using and caching, but it can miss VRAM that is already occupied outside that allocator by the driver, runtime, desktop, other applications, or other GPU allocations.

This can make ComfyUI believe that more VRAM is available than is really safe to use.

When a large model is loaded or models are switched quickly, ComfyUI may then try to keep or load too much data before offloading memory. On my system this showed up as:

- KSampler stuck at **0%**;
- freezes when switching large models;
- unstable model unload/load behavior;
- ComfyUI crashes under heavy VRAM pressure.

## What the patch changes

The patch makes Intel XPU use a memory calculation closer to the one ComfyUI already uses for CUDA:

1. Ask the XPU runtime for **global free GPU memory** using `torch.xpu.mem_get_info()`.
2. Check how much of PyTorch's reserved VRAM cache can be reused.
3. Add only that reusable cache to the globally free VRAM.
4. Validate the result before giving it to ComfyUI.
5. If XPU global memory reporting is unavailable or looks invalid, safely fall back to the original ComfyUI calculation.

In simple terms:

```text
Stock XPU path:
PyTorch allocator estimate -> ComfyUI decides how much VRAM it can use

Patched XPU path:
Actual global free VRAM + reusable PyTorch cache -> ComfyUI gets a safer VRAM estimate
```

This helps ComfyUI make better load/offload decisions instead of discovering too late that the GPU is already under memory pressure.

## What it does NOT change

The patch does not modify:

- model weights;
- image quality;
- sampling or KSampler logic;
- attention implementations;
- Triton kernels;
- model architecture;
- quantization;
- the normal behavior of CUDA, AMD, CPU, or other non-XPU devices.

It only patches `comfy.model_management.get_free_memory()` **for Intel XPU devices**. Other backends continue using the original ComfyUI function.

## Tested system

This patch has been personally tested on one main system:

| Component | Tested configuration |
| --- | --- |
| GPU | Intel Arc B580 12 GB |
| System RAM | 32 GB |
| OS | Windows 11 |
| ComfyUI | 0.31.1 |
| PyTorch | 2.13.0+xpu |
| comfy-kitchen | 0.2.28 |
| comfy-aimdo | 0.4.13 |
| Python | 3.13.12 |

I also received positive feedback from another user who tested the patch successfully.

**Important:** this is still a small real-world test base. Different Intel GPUs, PyTorch versions, drivers, operating systems, and ComfyUI versions may behave differently.

## Recommended ComfyUI 0.31.1 XPU launch arguments

These arguments are **not required by this patch**. They are the most stable ComfyUI 0.31.1 XPU configuration I found on my own Intel Arc B580 after extensive heavy model-switching tests:

```text
--cache-classic --disable-async-offload --oneapi-device-selector level_zero:gpu
```

### What these arguments mean

| Argument | Why I use it |
| --- | --- |
| `--cache-classic` | Strongly recommended on my ComfyUI 0.31.1 setup. The default RAM-pressure cache was less stable during heavy multi-model switching. This may be unnecessary on older or future ComfyUI versions. |
| `--disable-async-offload` | XPU does not enable async offload by default in ComfyUI 0.31.1, but this forces it off if it was enabled by your launcher or previous arguments. `--async-offload 2` was faster in some tests but noticeably less stable on ComfyUI 0.31.1. |
| `--oneapi-device-selector level_zero:gpu` | Explicitly selects the Intel Level Zero GPU device. Intel/XPU-specific and unrelated to the VRAM patch itself. |

These recommendations are specifically based on **ComfyUI 0.31.1** and my hardware. Future ComfyUI or PyTorch releases may not need them.

## Installation

No additional Python packages are required.

### Option 1 — Download ZIP

1. Download the repository ZIP from GitHub.
2. Fully close ComfyUI.
3. Extract the repository folder into:

```text
ComfyUI/custom_nodes/
```

You should have:

```text
ComfyUI/custom_nodes/ComfyUI-AVS-Intel-XPU-VRAM-Fix/
    __init__.py
    README.md
    LICENSE
```

4. Start ComfyUI.

### Option 2 — Install with Git

Open Command Prompt in your ComfyUI `custom_nodes` folder:

```bat
cd /d "PATH\TO\ComfyUI\custom_nodes"
git clone https://github.com/alvasafin-art/ComfyUI-AVS-Intel-XPU-VRAM-Fix.git
```

Restart ComfyUI.

### Update with Git

```bat
cd /d "PATH\TO\ComfyUI\custom_nodes\ComfyUI-AVS-Intel-XPU-VRAM-Fix"
git pull
```

Restart ComfyUI.

### Uninstall

Close ComfyUI and delete:

```text
ComfyUI/custom_nodes/ComfyUI-AVS-Intel-XPU-VRAM-Fix
```

## How to check that the patch is active

At startup, look for messages similar to:

```text
[XPU Global VRAM V2] Active memory backend: torch.xpu.mem_get_info
[XPU Global VRAM V2] Patch installed. Current usable free: ...
```

If `torch.xpu.mem_get_info()` is unavailable or returns values that fail the patch's sanity checks, the patch logs a warning and automatically falls back to the original ComfyUI calculation.

## Compatibility and limitations

- Intended only for **Intel XPU / Intel Arc** devices.
- Personally tested on Intel Arc B580 with ComfyUI 0.31.1 and PyTorch 2.13.0+xpu.
- One additional user has reported positive results.
- The code itself does not depend on Windows-specific APIs, but other OS / Intel GPU combinations have not been personally validated yet.
- `torch.xpu.mem_get_info()` support and behavior can differ between PyTorch / Intel driver stacks. The patch includes validation and a safe fallback for this reason.
- A future ComfyUI release may change or fix XPU VRAM accounting upstream. If that happens, test stock ComfyUI first before keeping this patch enabled.
- Other custom nodes that also replace `comfy.model_management.get_free_memory()` could conflict with this startup patch.

## Reporting results

If you test this patch, useful information includes:

- Intel GPU model;
- VRAM and system RAM;
- OS;
- ComfyUI version;
- PyTorch version;
- Intel graphics driver version;
- launch arguments;
- model / quantization used;
- whether you previously saw KSampler 0% hangs, model-switch freezes, OOM errors, or crashes;
- whether the problem changed after installing the patch.

## Technical references

- ComfyUI 0.31.1 `get_free_memory()` implementation: https://github.com/Comfy-Org/ComfyUI/blob/v0.31.1/comfy/model_management.py
- PyTorch XPU global memory API: https://docs.pytorch.org/docs/stable/generated/torch.xpu.memory.mem_get_info.html

## Credits

Development, debugging, testing analysis, and documentation were assisted by **OpenAI ChatGPT**.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

You are free to use, modify, redistribute, and commercially use this software under the terms of the GPL-3.0 license. Distributed modified versions must remain open source under the GPL-3.0 license.
