import { app } from "../../scripts/app.js";

const SETTING_ID = "AVS.XPU.VRAMFix.Enabled";
const API_URL = "/avs-xpu-vram-fix/enabled";

function getBooleanValue(args) {
    // Declarative extension settings currently pass the new value first.
    // Keep a compatibility fallback for store-style callbacks that pass
    // (settingDefinition, newValue, oldValue).
    if (typeof args[0] === "boolean") {
        return args[0];
    }
    if (typeof args[1] === "boolean") {
        return args[1];
    }
    return null;
}

async function syncBackend(enabled) {
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ enabled }),
        });

        if (!response.ok) {
            throw new Error(`${response.status} ${response.statusText}`);
        }

        const status = await response.json();
        if (status.enabled !== enabled) {
            throw new Error(
                `backend returned enabled=${String(status.enabled)} for requested enabled=${String(enabled)}`,
            );
        }

        console.info(
            `[AVS XPU VRAM Fix] VRAM fix ${enabled ? "enabled" : "disabled"}.`,
        );
    } catch (error) {
        console.error("[AVS XPU VRAM Fix] Failed to update backend setting:", error);
    }
}

app.registerExtension({
    name: "AVS.XPU.VRAMFix",
    settings: [
        {
            id: SETTING_ID,
            category: ["AVS XPU", "VRAM", "Fix"],
            name: "Enable Intel XPU VRAM fix",
            tooltip:
                "Use the guarded Intel XPU VRAM calculation from ComfyUI PR #15487. " +
                "When disabled, the extension immediately delegates to the original " +
                "ComfyUI memory functions; the custom node does not need to be removed.",
            type: "boolean",
            defaultValue: true,
            onChange: (...args) => {
                const enabled = getBooleanValue(args);
                if (enabled === null) {
                    console.error(
                        "[AVS XPU VRAM Fix] Ignoring settings callback with no boolean value:",
                        args,
                    );
                    return;
                }
                void syncBackend(enabled);
            },
        },
    ],
});
