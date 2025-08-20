from __future__ import annotations

import threading
from dataclasses import dataclass
import time
from typing import Optional, Sequence

import numpy as np
from manim import (
	Scene,
	VGroup,
	Rectangle,
	ValueTracker,
	always_redraw,
	config as manim_config,
	BLUE, RED, GREEN,
	Circle, RegularPolygon, Square,
)
import random


@dataclass
class SharedBands:
	"""Thread-safe container for the latest band magnitudes in [0,1]."""
	values: Optional[np.ndarray] = None
	lock: threading.Lock = threading.Lock()

	def set(self, arr: np.ndarray) -> None:
		with self.lock:
			self.values = np.asarray(arr, dtype=np.float32).copy()

	def get(self) -> Optional[np.ndarray]:
		with self.lock:
			if self.values is None:
				return None
			return self.values.copy()


def interpolate_color(v: float, colors: Sequence) -> any:
	"""Map v in [0,1] to a color along a simple gradient list."""
	v = float(np.clip(v, 0.0, 1.0))
	if not colors:
		return GREEN
	if len(colors) == 1:
		return colors[0]
	step = 1.0 / (len(colors) - 1)
	idx = min(int(v // step), len(colors) - 2)
	local_t = (v - idx * step) / step
	return colors[idx].interpolate(colors[idx + 1], local_t)


class SpectrumBarsScene(Scene):
	"""Horizontal bar spectrum driven by SharedBands.

	Use with the OpenGL renderer for responsiveness:
	  manim -p -ql --renderer=opengl scripts/run_visual.py SpectrumBarsScene
	"""

	def __init__(self, renderer=None, shared: Optional[SharedBands] = None, num_bands: int = 32, min_height: float = 0.05, color_scheme: Optional[Sequence] = None, shapes_config: Optional[dict] = None, scene_scale: float = 1.0, baseline_y: float = -3.0, scene_width: float = 12.0, bar_opacity: float = 0.9, **kwargs):
		# Accept Manim's renderer as positional arg and pass to super
		super().__init__(renderer, **kwargs)
		self.shared = shared or SharedBands()
		self.num_bands = int(num_bands)
		self.min_height = float(min_height)
		self._bar_opacity = float(np.clip(bar_opacity, 0.0, 1.0))
		self.scene_scale = float(scene_scale)
		self.baseline_y = float(baseline_y)
		self.scene_width = float(scene_width)
		self._trackers: list[ValueTracker] = []
		self._bars: VGroup | None = None
		self._colors = self._resolve_colors(color_scheme) if color_scheme is not None else [BLUE, GREEN, RED]
		# Shapes state
		self._shapes_group: VGroup | None = None
		self._active_shapes: list[dict] = []
		self._last_spawn_time: float = 0.0
		self._rng = random.Random()
		self._shapes_cfg = self._default_shapes_cfg()
		if isinstance(shapes_config, dict):
			self._shapes_cfg.update(self._sanitize_shapes_cfg(shapes_config))
		if self._shapes_cfg.get("colors"):
			self._shapes_cfg["colors"] = self._resolve_colors(self._shapes_cfg["colors"]) or self._colors

	def _resolve_colors(self, colors: Optional[Sequence]) -> list:
		if not colors:
			return []
		resolved = []
		for c in colors:
			if isinstance(c, str):
				try:
					from manim import __dict__ as manim_dict
					resolved.append(manim_dict[c])
				except Exception:
					continue
			else:
				resolved.append(c)
		return resolved

	def _default_shapes_cfg(self) -> dict:
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

	def _sanitize_shapes_cfg(self, cfg: dict) -> dict:
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

	def construct(self):
		# Layout bars centered; apply vertical scale only to heights
		width = self.scene_width
		gap = 0.05
		bar_width = max((width - gap * (self.num_bands - 1)) / self.num_bands, 0.05)
		height_base = 0.2

		bars = VGroup()
		self._trackers = []
		self._rects: list[Rectangle] = []
		for i in range(self.num_bands):
			tracker = ValueTracker(self.min_height)
			self._trackers.append(tracker)
			h0 = max(tracker.get_value(), self.min_height) * self.scene_scale
			rect = Rectangle(width=bar_width, height=h0)
			rect.set_fill(color=interpolate_color(h0, self._colors), opacity=self._bar_opacity)
			rect.set_stroke(width=0)
			x = -width / 2.0 + (bar_width + gap) * i + bar_width / 2.0
			rect.move_to([x, self.baseline_y + h0 / 2.0, 0.0])
			self._rects.append(rect)
			bars.add(rect)

		self._bars = bars
		self.add(bars)

		# Layer for shapes above bars
		self._shapes_group = VGroup()
		self.add(self._shapes_group)

		# Updater: poll shared bands once per frame
		self._last_log_time = 0.0

		def update_trackers(_dt):
			arr = self.shared.get()
			now = time.time()
			if arr is None:
				# Log at most once per second when nothing received yet
				if now - self._last_log_time > 1.0:
					try:
						print("visuals: no bands yet")
					except Exception:
						pass
					self._last_log_time = now
				return
			if arr.shape[0] != self.num_bands:
				# Simple resize by clipping or padding
				if arr.shape[0] > self.num_bands:
					arr = arr[: self.num_bands]
				else:
					arr = np.pad(arr, (0, self.num_bands - arr.shape[0]), mode="edge")
			# Clamp to [0,1] and floor
			arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
			arr = np.maximum(arr, self.min_height)
			if now - self._last_log_time > 1.0:
				try:
					print(f"visuals: update bands min={float(np.min(arr)):.3f} max={float(np.max(arr)):.3f} len={arr.shape[0]}")
				except Exception:
					pass
				self._last_log_time = now
			for i, tracker in enumerate(self._trackers):
				val = float(arr[i])
				tracker.set_value(val)
				# Update corresponding rectangle once per frame
				rect = self._rects[i]
				rect.set_fill(color=interpolate_color(val, self._colors), opacity=self._bar_opacity)
				rect.set_stroke(width=0)
				rect.stretch_to_fit_height(val * self.scene_scale)
				# Recompute position to keep bottom baseline stable
				x = rect.get_center()[0]
				rect.move_to([x, self.baseline_y + (val * self.scene_scale) / 2.0, 0.0])

			# Spawn shapes based on low frequencies
			if self._shapes_cfg.get("enabled", True):
				low_n = min(int(self._shapes_cfg["low_bands_count"]), arr.shape[0])
				low_energy = float(np.mean(arr[:low_n])) if low_n > 0 else 0.0
				cooldown_s = self._shapes_cfg["cooldown_ms"] / 1000.0
				if (low_energy > self._shapes_cfg["spawn_threshold"]) and ((now - self._last_spawn_time) >= cooldown_s):
					if len(self._active_shapes) < int(self._shapes_cfg["max_active"]):
						self._spawn_shape(low_energy, width)
						self._last_spawn_time = now

			# Animate and clean shapes
			self._update_shapes(now)

		# Attach updater to the bars group (ensures it's called each frame)
		self._bars.add_updater(lambda _m, dt: update_trackers(dt))
		self.wait(60.0)  # Keep scene running for interactive session

	def _choose_shape_mobject(self, kind: str, size: float):
		k = kind.lower()
		if k == "circle":
			return Circle(radius=size / 2.0)
		if k == "square":
			return Square(side_length=size)
		if k == "triangle":
			return RegularPolygon(n=3, radius=size / 2.0)
		return Circle(radius=size / 2.0)

	def _spawn_shape(self, energy: float, total_width: float) -> None:
		if self._shapes_group is None:
			return
		cfg = self._shapes_cfg
		size_min, size_max = cfg["size_range"]
		size = float(size_min + (size_max - size_min) * float(np.clip(energy, 0.0, 1.0)))
		shape_kind = self._rng.choice(cfg["shape_types"]) if cfg.get("shape_types") else "circle"
		mobj = self._choose_shape_mobject(shape_kind, size)
		shape_colors = cfg.get("colors") or self._colors
		color = interpolate_color(float(np.clip(energy, 0.0, 1.0)), shape_colors) if shape_colors else GREEN
		mobj.set_fill(color=color, opacity=float(cfg["opacity_range"][0]))
		mobj.set_stroke(width=0)
		# Random position within current frame bounds, keeping shape fully visible
		x_radius = float(manim_config.frame_width) / 2.0
		y_radius = float(manim_config.frame_height) / 2.0
		x_min = -x_radius + size / 2.0
		x_max = x_radius - size / 2.0
		y_min = -y_radius + size / 2.0
		y_max = y_radius - size / 2.0
		x = self._rng.uniform(x_min, x_max)
		y = self._rng.uniform(y_min, y_max)
		mobj.move_to([x, y, 0.0])
		self._shapes_group.add(mobj)
		self.add(mobj)
		self._active_shapes.append({
			"mobj": mobj,
			"birth": time.time(),
			"lifespan": float(cfg["lifespan_s"]),
			"start_opacity": float(cfg["opacity_range"][0]),
			"end_opacity": float(cfg["opacity_range"][1]),
			"drift_y": float(cfg["drift_y"]) * self.scene_scale,
		})

	def _update_shapes(self, now: float) -> None:
		alive: list[dict] = []
		for s in self._active_shapes:
			mobj = s["mobj"]
			birth = s["birth"]
			life = s["lifespan"]
			t = (now - birth) / max(1e-6, life)
			if t >= 1.0:
				try:
					mobj.set_opacity(0.0)
					self.remove(mobj)
					if self._shapes_group is not None:
						self._shapes_group.remove(mobj)
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
		self._active_shapes = alive


