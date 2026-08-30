# Changelog

## v2.0.1 — Settings state fix and PR parity review

- Fixed the **AVS XPU** toggle callback so the new boolean value is read from the correct argument.
- Fixed the observed reversed behavior where switching the toggle off could send the previous `true` state to the backend.
- Fixed startup synchronization where the old callback could convert an absent second argument to `false` and disable the patch after restart/page load.
- Added compatibility handling for both declarative-extension and store-style settings callback signatures.
- Added frontend verification that the backend accepted the exact requested state before logging enabled/disabled.
- Re-reviewed the memory helper against the current PR #15487 implementation.
- Removed `int()` coercion of `torch.xpu.mem_get_info()` values so malformed values follow the PR's `TypeError`/`ValueError` fallback behavior instead of being silently normalized.
- Kept the PR checks for invalid total/free values and `global_used < mem_reserved`.
- Kept the same PR fallback and `min(free + reusable_cache, total)` cap.
- Kept an additional outer runtime safety fallback to the original ComfyUI functions if the standalone wrapper itself encounters an unexpected error.
- Status now reports the original ComfyUI path explicitly while the fix is disabled.

## v2.0.0 — PR #15487 alignment and settings toggle

- Reworked XPU VRAM accounting to match the guarded logic and principle of ComfyUI PR #15487.
- Removed the old standalone-only 25% total-memory comparison and 512 MiB telemetry tolerance checks.
- Added the PR consistency checks: valid total/free values and global used memory must not be smaller than PyTorch reserved memory.
- Added PR-style fallback to `get_device_properties(...).total_memory - mem_reserved` for unsupported or inconsistent `torch.xpu.mem_get_info()` results.
- Patches both `get_total_memory()` and `get_free_memory()` for XPU, using the same guarded helper.
- Caps reported usable free XPU memory to total XPU memory.
- Added **Settings → Application settings → AVS XPU → Enable Intel XPU VRAM fix**.
- The fix can now be enabled/disabled at runtime without deleting the custom node.
- The enabled state persists under the ComfyUI user directory and survives restart.
- Non-XPU devices continue to use the original ComfyUI functions.

## v1.0.0 — Initial public release

- Adds device-wide Intel XPU free-VRAM reporting for ComfyUI.
- Uses `torch.xpu.mem_get_info()` plus reusable PyTorch allocator cache.
- Validates global XPU telemetry before using it.
- Automatically falls back to the original ComfyUI memory calculation if telemetry is unavailable or invalid.
- Leaves all non-XPU devices unchanged.
- Tested primarily on Intel Arc B580 12 GB with ComfyUI 0.31.1 and PyTorch 2.13.0+xpu.
