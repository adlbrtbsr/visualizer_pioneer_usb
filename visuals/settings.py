import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VisualIntensitySettings:
    master: float = 1.0
    exposure: float = 1.0
    glow_gain: float = 1.0
    trap_mix_gain: float = 1.0
    motion_gain: float = 1.0
    iteration_gain: float = 1.0
    # Extended customization
    scale: float = 2.4
    iterations_base: float = 150.0
    bailout_radius: float = 8.0
    morph_gain: float = 1.0
    ship_gain: float = 1.0
    trap_radius_scale: float = 1.0
    contrast: float = 1.0
    palette_id: int = 0
    hue_offset: float = 0.0
    palette_saturation: float = 0.9
    fractal_type: int = 0

    @classmethod
    def from_yaml(cls, path: Path):
        try:
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                node = {}
                if isinstance(data, dict):
                    node = data.get("live_fractal_intensity") or data.get("intensity") or {}
                if isinstance(node, dict):
                    return cls(
                        master=float(node.get("master", 1.0)),
                        exposure=float(node.get("exposure", 1.0)),
                        glow_gain=float(node.get("glow_gain", 1.0)),
                        trap_mix_gain=float(node.get("trap_mix_gain", 1.0)),
                        motion_gain=float(node.get("motion_gain", 1.0)),
                        iteration_gain=float(node.get("iteration_gain", 1.0)),
                        scale=float(node.get("scale", 2.4)),
                        iterations_base=float(node.get("iterations_base", 150.0)),
                        bailout_radius=float(node.get("bailout_radius", 8.0)),
                        morph_gain=float(node.get("morph_gain", 1.0)),
                        ship_gain=float(node.get("ship_gain", 1.0)),
                        trap_radius_scale=float(node.get("trap_radius_scale", 1.0)),
                        contrast=float(node.get("contrast", 1.0)),
                        palette_id=int(node.get("palette_id", 0)),
                        hue_offset=float(node.get("hue_offset", 0.0)),
                        palette_saturation=float(node.get("palette_saturation", 0.9)),
                        fractal_type=int(node.get("fractal_type", 0)),
                    )
        except Exception:
            pass
        return cls()


def save_visual_intensity_yaml(settings: VisualIntensitySettings, path: Path) -> None:
    try:
        data = {}
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                if isinstance(loaded, dict):
                    data = loaded
        block = {
            "master": float(settings.master),
            "exposure": float(settings.exposure),
            "glow_gain": float(settings.glow_gain),
            "trap_mix_gain": float(settings.trap_mix_gain),
            "motion_gain": float(settings.motion_gain),
            "iteration_gain": float(settings.iteration_gain),
            "scale": float(settings.scale),
            "iterations_base": float(settings.iterations_base),
            "bailout_radius": float(settings.bailout_radius),
            "morph_gain": float(settings.morph_gain),
            "ship_gain": float(settings.ship_gain),
            "trap_radius_scale": float(settings.trap_radius_scale),
            "contrast": float(settings.contrast),
            "palette_id": int(settings.palette_id),
            "hue_offset": float(settings.hue_offset),
            "palette_saturation": float(settings.palette_saturation),
            "fractal_type": int(settings.fractal_type),
        }
        data["live_fractal_intensity"] = block
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    except Exception:
        pass


