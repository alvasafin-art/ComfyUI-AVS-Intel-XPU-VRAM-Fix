# Changelog

## v1.0.0 — Initial public release

- Adds device-wide Intel XPU free-VRAM reporting for ComfyUI.
- Uses `torch.xpu.mem_get_info()` plus reusable PyTorch allocator cache.
- Validates global XPU telemetry before using it.
- Automatically falls back to the original ComfyUI memory calculation if telemetry is unavailable or invalid.
- Leaves all non-XPU devices unchanged.
- Tested primarily on Intel Arc B580 12 GB with ComfyUI 0.31.1 and PyTorch 2.13.0+xpu.
