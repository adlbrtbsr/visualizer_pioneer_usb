from __future__ import annotations

import random
import time
from typing import Optional, Sequence

import numpy as np
from manim import VGroup, Circle, RegularPolygon, Square, config as manim_config, GREEN

from .utils import interpolate_color


def default_shapes_cfg() -> dict:
	return {
		"enabled": True,
		"low_bands_count": 4,
		"spawn_threshold": 0.6,
		"cooldown_ms": 150,
		"max_active": 25,
		"lifespan_s": 1.2,
		"drift_y": 1.5,
		"size_range": [0.2, 0.8],
		"shape_types": ["circle", "triangle", "square"],
		"colors": None,
		"opacity_range": [0.9, 0.0],
	}


def sanitize_shapes_cfg(cfg: dict) -> dict:
	out = dict(cfg)
	out["low_bands_count"] = max(1, int(out.get("low_bands_count", 4)))
	out["spawn_threshold"] = float(np.clip(float(out.get("spawn_threshold", 0.6)), 0.0, 1.0))
	out["cooldown_ms"] = max(0, int(out.get("cooldown_ms", 150)))
	out["max_active"] = max(0, int(out.get("max_active", 25)))
	out["lifespan_s"] = max(0.05, float(out.get("lifespan_s", 1.2)))
	out["drift_y"] = float(out.get("drift_y", 1.5))
	sr = out.get("size_range", [0.2, 0.8])
	if not isinstance(sr, (list, tuple)) or len(sr) != 2:
		sr = [0.2, 0.8]
	out["size_range"] = [float(max(0.01, sr[0])), float(max(sr[0], sr[1]))]
	orng = out.get("opacity_range", [0.9, 0.0])
	if not isinstance(orng, (list, tuple)) or len(orng) != 2:
		orng = [0.9, 0.0]
	out["opacity_range"] = [float(np.clip(orng[0], 0.0, 1.0)), float(np.clip(orng[1], 0.0, 1.0))]
	st = out.get("shape_types", ["circle", "triangle", "square"]) or ["circle"]
	out["shape_types"] = [str(s).lower() for s in st]
	return out


class ShapesLayer:
	"""Manages transient shapes that spawn and fade based on low-frequency energy."""

	def __init__(self, rng: Optional[random.Random], colors: Sequence, cfg: dict, scene_scale: float):
		self._rng = rng or random.Random()
		self._colors = list(colors)
		self._cfg = cfg
		self._scene_scale = float(scene_scale)
		self._group: VGroup = VGroup()
		self._active: list[dict] = []
		self._last_spawn_time: float = 0.0

	def get_group(self) -> VGroup:
		return self._group

	def _choose_shape_mobject(self, kind: str, size: float):
		k = kind.lower()
		if k == "circle":
			return Circle(radius=size / 2.0)
		if k == "square":
			return Square(side_length=size)
		if k == "triangle":
			return RegularPolygon(n=3, radius=size / 2.0)
		return Circle(radius=size / 2.0)

	def spawn(self, energy: float) -> None:
		cfg = self._cfg
		size_min, size_max = cfg["size_range"]
		size = float(size_min + (size_max - size_min) * float(np.clip(energy, 0.0, 1.0)))
		shape_kind = self._rng.choice(cfg["shape_types"]) if cfg.get("shape_types") else "circle"
		mobj = self._choose_shape_mobject(shape_kind, size)
		shape_colors = cfg.get("colors") or self._colors
		color = interpolate_color(float(np.clip(energy, 0.0, 1.0)), shape_colors) if shape_colors else GREEN
		mobj.set_fill(color=color, opacity=float(cfg["opacity_range"][0]))
		mobj.set_stroke(width=0)
		x_radius = float(manim_config.frame_width) / 2.0
		y_radius = float(manim_config.frame_height) / 2.0
		x_min = -x_radius + size / 2.0
		x_max = x_radius - size / 2.0
		y_min = -y_radius + size / 2.0
		y_max = y_radius - size / 2.0
		x = self._rng.uniform(x_min, x_max)
		y = self._rng.uniform(y_min, y_max)
		mobj.move_to([x, y, 0.0])
		self._group.add(mobj)
		self._active.append({
			"mobj": mobj,
			"birth": time.time(),
			"lifespan": float(cfg["lifespan_s"]),
			"start_opacity": float(cfg["opacity_range"][0]),
			"end_opacity": float(cfg["opacity_range"][1]),
			"drift_y": float(cfg["drift_y"]) * self._scene_scale,
		})

	def maybe_spawn(self, low_energy: float, now: float) -> None:
		cooldown_s = self._cfg["cooldown_ms"] / 1000.0
		if (low_energy > self._cfg["spawn_threshold"]) and ((now - self._last_spawn_time) >= cooldown_s):
			if len(self._active) < int(self._cfg["max_active"]):
				self.spawn(low_energy)
				self._last_spawn_time = now

	def update(self, now: float) -> None:
		alive: list[dict] = []
		for s in self._active:
			mobj = s["mobj"]
			birth = s["birth"]
			life = s["lifespan"]
			t = (now - birth) / max(1e-6, life)
			if t >= 1.0:
				try:
					mobj.set_opacity(0.0)
					self._group.remove(mobj)
				except Exception:
					pass
				continue
			start_o = s["start_opacity"]
			end_o = s["end_opacity"]
			opacity = float(start_o + (end_o - start_o) * t)
			try:
				mobj.set_opacity(np.clip(opacity, 0.0, 1.0))
			except Exception:
				pass
			if s["drift_y"] != 0.0:
				try:
					pos = mobj.get_center()
					dy = s["drift_y"] * (1.0 / 60.0)
					mobj.move_to([pos[0], pos[1] + dy, pos[2]])
				except Exception:
					pass
			alive.append(s)
		self._active = alive


