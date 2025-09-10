from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from .settings import VisualIntensitySettings


@dataclass
class MappingConfig:
    # Which mel bins to average for proxies (indices 0..mel_bands-1)
    mel_low_idx: List[int]
    mel_mid_idx: List[int]
    mel_high_idx: List[int]
    # Scale mapping from spectral centroid
    exp_scale_k: float = 3000.0
    scale_min: float = 0.6
    scale_max: float = 3.2
    # Iterations mapping
    iterations_base: float = 180.0
    iterations_from_bass: float = 220.0
    # Bailout mapping
    bailout_base: float = 12.0
    bailout_mid_gain: float = 8.0
    # Warp/bend from flux
    warp_from_flux: float = 0.45
    bend_from_flux: float = 0.85
    # Motion/morph gains
    motion_from_mid: float = 1.4
    morph_from_mid: float = 1.0
    ship_from_high: float = 1.4
    # Trap mix from energy
    trap_mix_from_energy: float = 0.9
    trap_radius_from_bass: float = 0.9
    # Palette selection
    palette_from_chroma: bool = True
    hue_from_chroma_weighted: bool = True
    # Fractal type switching
    fractal_on_beat_types: List[int] = (0, 1, 2, 3)
    beat_hold_seconds: float = 0.35


def _mean_indices(vec: np.ndarray, idx: List[int]) -> float:
    if len(idx) == 0:
        return 0.0
    idx_arr = np.array([i for i in idx if 0 <= int(i) < vec.shape[0]], dtype=np.int32)
    if idx_arr.size == 0:
        return 0.0
    return float(np.mean(vec[idx_arr]))


class MappingEngine:
    def __init__(self, cfg: MappingConfig) -> None:
        self.cfg = cfg
        self._last_beat_time: float = -1e9
        self._last_fractal_idx: int = 0

    @staticmethod
    def from_yaml(path: Path, mel_bands: int) -> "MappingEngine":
        # Provide sensible defaults for low/mid/high splitting of mel bins
        def default_idx() -> Tuple[List[int], List[int], List[int]]:
            if mel_bands < 6:
                a = list(range(max(mel_bands, 1)))
                return a, a, a
            third = mel_bands // 3
            low = list(range(0, third))
            mid = list(range(third, 2 * third))
            high = list(range(2 * third, mel_bands))
            return low, mid, high

        data: Dict[str, object] = {}
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception:
                data = {}
        low_d, mid_d, high_d = default_idx()
        block = data.get("mapping") if isinstance(data, dict) else None
        if not isinstance(block, dict):
            block = {}
        cfg = MappingConfig(
            mel_low_idx=list(block.get("mel_low_idx", low_d)),
            mel_mid_idx=list(block.get("mel_mid_idx", mid_d)),
            mel_high_idx=list(block.get("mel_high_idx", high_d)),
            exp_scale_k=float(block.get("exp_scale_k", 3000.0)),
            scale_min=float(block.get("scale_min", 0.6)),
            scale_max=float(block.get("scale_max", 3.2)),
            iterations_base=float(block.get("iterations_base", 180.0)),
            iterations_from_bass=float(block.get("iterations_from_bass", 220.0)),
            bailout_base=float(block.get("bailout_base", 12.0)),
            bailout_mid_gain=float(block.get("bailout_mid_gain", 8.0)),
            warp_from_flux=float(block.get("warp_from_flux", 0.45)),
            bend_from_flux=float(block.get("bend_from_flux", 0.85)),
            motion_from_mid=float(block.get("motion_from_mid", 1.4)),
            morph_from_mid=float(block.get("morph_from_mid", 1.0)),
            ship_from_high=float(block.get("ship_from_high", 1.4)),
            trap_mix_from_energy=float(block.get("trap_mix_from_energy", 0.9)),
            trap_radius_from_bass=float(block.get("trap_radius_from_bass", 0.9)),
            palette_from_chroma=bool(block.get("palette_from_chroma", True)),
            hue_from_chroma_weighted=bool(block.get("hue_from_chroma_weighted", True)),
            fractal_on_beat_types=list(block.get("fractal_on_beat_types", (0, 1, 2, 3))),
            beat_hold_seconds=float(block.get("beat_hold_seconds", 0.35)),
        )
        return MappingEngine(cfg)

    def map(self, features: Dict[str, object], settings: VisualIntensitySettings, time_sec: float) -> Tuple[float, float, float, VisualIntensitySettings]:
        mel = np.asarray(features.get("mel", np.zeros((24,), dtype=np.float32)), dtype=np.float32)
        chroma = np.asarray(features.get("chroma", np.zeros((12,), dtype=np.float32)), dtype=np.float32)
        flux = float(features.get("flux", 0.0))
        rms = float(features.get("rms", 0.0))
        centroid_hz = float(features.get("centroid_hz", 0.0))
        beat = bool(features.get("beat", False))

        # Proxies
        bass = _mean_indices(mel, self.cfg.mel_low_idx)
        mid = _mean_indices(mel, self.cfg.mel_mid_idx)
        high = _mean_indices(mel, self.cfg.mel_high_idx)

        # Working copy of settings
        out = VisualIntensitySettings(
            master=settings.master,
            exposure=settings.exposure,
            glow_gain=settings.glow_gain,
            trap_mix_gain=settings.trap_mix_gain,
            motion_gain=settings.motion_gain,
            iteration_gain=settings.iteration_gain,
            scale=settings.scale,
            iterations_base=settings.iterations_base,
            bailout_radius=settings.bailout_radius,
            morph_gain=settings.morph_gain,
            ship_gain=settings.ship_gain,
            trap_radius_scale=settings.trap_radius_scale,
            contrast=settings.contrast,
            palette_id=settings.palette_id,
            hue_offset=settings.hue_offset,
            palette_saturation=settings.palette_saturation,
            fractal_type=settings.fractal_type,
            bend_gain=getattr(settings, "bend_gain", 1.0),
            view_angle_deg=settings.view_angle_deg,
            view_center_x=settings.view_center_x,
            view_center_y=settings.view_center_y,
        )

        # Structural/continuous mappings
        # Scale from centroid: exp(-centroid/k) scaled into [scale_min, scale_max]
        scale_factor = math.exp(-centroid_hz / max(self.cfg.exp_scale_k, 1.0))
        out.scale = float(np.clip(self.cfg.scale_min + (self.cfg.scale_max - self.cfg.scale_min) * scale_factor, self.cfg.scale_min, self.cfg.scale_max))

        # Iterations base + bass contribution
        out.iterations_base = float(np.clip(self.cfg.iterations_base + self.cfg.iterations_from_bass * bass, 60.0, 420.0))

        # Bailout from mid
        out.bailout_radius = float(np.clip(self.cfg.bailout_base + self.cfg.bailout_mid_gain * mid, 4.0, 32.0))

        # Motion/morph/ship gains influenced by bands
        out.motion_gain = float(np.clip(0.6 + self.cfg.motion_from_mid * mid, 0.0, 3.0))
        out.morph_gain = float(np.clip(0.4 + self.cfg.morph_from_mid * mid, 0.0, 2.0))
        out.ship_gain = float(np.clip(self.cfg.ship_from_high * max(high - 0.25, 0.0), 0.0, 2.0))

        # Trap mix and radius
        energy = float(np.clip((bass + mid + high) / 3.0, 0.0, 1.0))
        out.trap_mix_gain = float(np.clip(0.6 + self.cfg.trap_mix_from_energy * energy, 0.2, 2.0))
        out.trap_radius_scale = float(np.clip(0.9 + self.cfg.trap_radius_from_bass * bass, 0.5, 2.5))

        # Bend via flux -> use bend_gain to scale shader's bend param indirectly
        out.bend_gain = float(np.clip(0.6 + self.cfg.bend_from_flux * flux, 0.2, 3.0))

        # Glow/exposure subtly from RMS
        out.glow_gain = float(np.clip(0.8 + 0.8 * rms, 0.4, 2.0))
        out.exposure = float(np.clip(0.95 + 0.2 * rms, 0.6, 1.4))

        # Palette mapping from chroma
        if self.cfg.palette_from_chroma and chroma.shape[0] == 12:
            pid = int(np.argmax(chroma)) % 4
            out.palette_id = pid
            if self.cfg.hue_from_chroma_weighted:
                # Weighted average hue: chroma bins mapped around circle
                idx = np.arange(12, dtype=np.float32)
                w = chroma + 1e-6
                hue = float(((idx @ w) / float(np.sum(w))) / 12.0)
                out.hue_offset = float(np.mod(hue, 1.0))

        # Fractal type switching on beat (momentary hold)
        if beat:
            self._last_beat_time = time_sec
            self._last_fractal_idx = (self._last_fractal_idx + 1) % max(len(self.cfg.fractal_on_beat_types), 1)
        if (time_sec - self._last_beat_time) <= self.cfg.beat_hold_seconds:
            out.fractal_type = int(self.cfg.fractal_on_beat_types[self._last_fractal_idx % len(self.cfg.fractal_on_beat_types)])

        return bass, mid, high, out


