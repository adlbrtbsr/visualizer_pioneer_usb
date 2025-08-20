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
	BLUE, RED, GREEN, WHITE,
	Circle, RegularPolygon, Square, ImageMobject,
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

	def __init__(self, renderer=None, shared: Optional[SharedBands] = None, num_bands: int = 32, min_height: float = 0.05, color_scheme: Optional[Sequence] = None, shapes_config: Optional[dict] = None, scene_scale: float = 1.0, baseline_y: float = -3.0, scene_width: float = 12.0, bar_opacity: float = 0.9, fractal_config: Optional[dict] = None, **kwargs):
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
		# Fractal background state
		self._fractal_cfg = self._default_fractal_cfg()
		if isinstance(fractal_config, dict):
			self._fractal_cfg.update(self._sanitize_fractal_cfg(fractal_config))
		self._fractal_bg: _FractalBackground | None = None

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

	def _default_fractal_cfg(self) -> dict:
		return {
			"enabled": False,
			"resolution": [256, 144],
			"max_iter": 120,
			"hf_bands_count": 8,
			"band_range": None,
			"update_fps": 18,
			"alpha": 0.65,
			"zoom_range": [0.95, 1.6],
			"julia_strength": 0.7,
			"color_speed": 0.35,
			"color_gamma": 1.0,
			"color_scale": 1.0,
			"color_offset": 0.0,
			"interior_palette": False,
			"palette": None,
			"seed": None,
			"overlay": False,
			"extent_x": 1.5,
			"extent_y": 1.5,
			"debug_force_seconds": 0.0,
			"debug_constant": None,
			"debug_outline": False,
			"debug_solid_panel": False,
			"debug_fullscreen_panel": False,
			"mode": "image",  # "image" or "tiles"
			"tiles_max_cols": 64,
			"tiles_max_rows": 36,
			"only": False,
		}

	def _sanitize_fractal_cfg(self, cfg: dict) -> dict:
		out = dict(cfg)
		res = out.get("resolution", [256, 144])
		if not isinstance(res, (list, tuple)) or len(res) != 2:
			res = [256, 144]
		out["resolution"] = [max(32, int(res[0])), max(32, int(res[1]))]
		out["max_iter"] = max(10, int(out.get("max_iter", 120)))
		out["hf_bands_count"] = max(1, int(out.get("hf_bands_count", 8)))
		# Optional explicit band range [start, end) for energy; overrides hf_bands_count if present
		br = out.get("band_range")
		if isinstance(br, (list, tuple)) and len(br) == 2:
			try:
				start = int(br[0])
				end = int(br[1])
				if end < start:
					start, end = end, start
				out["band_range"] = [max(0, start), max(0, end)]
			except Exception:
				out["band_range"] = None
		else:
			out["band_range"] = None
		out["update_fps"] = max(1, int(out.get("update_fps", 18)))
		out["alpha"] = float(np.clip(float(out.get("alpha", 0.35)), 0.0, 1.0))
		zr = out.get("zoom_range", [0.95, 1.6])
		if not isinstance(zr, (list, tuple)) or len(zr) != 2:
			zr = [0.95, 1.6]
		out["zoom_range"] = [float(max(0.1, zr[0])), float(max(zr[0], zr[1]))]
		out["julia_strength"] = float(np.clip(float(out.get("julia_strength", 0.7)), 0.0, 2.0))
		out["color_speed"] = float(max(0.0, float(out.get("color_speed", 0.35))))
		# Color mapping controls
		try:
			out["color_gamma"] = float(max(0.01, float(out.get("color_gamma", 1.0))))
		except Exception:
			out["color_gamma"] = 1.0
		try:
			out["color_scale"] = float(max(0.01, float(out.get("color_scale", 1.0))))
		except Exception:
			out["color_scale"] = 1.0
		try:
			out["color_offset"] = float(out.get("color_offset", 0.0))
		except Exception:
			out["color_offset"] = 0.0
		out["interior_palette"] = bool(out.get("interior_palette", False))
		out["overlay"] = bool(out.get("overlay", False))
		# Domain extents (controls how far the fractal stretches in view)
		ext = out.get("extent", None)
		try:
			default_ext = float(1.5)
		except Exception:
			default_ext = 1.5
		ext_x = out.get("extent_x", ext if ext is not None else default_ext)
		ext_y = out.get("extent_y", ext if ext is not None else default_ext)
		try:
			out["extent_x"] = float(max(0.1, float(ext_x)))
		except Exception:
			out["extent_x"] = default_ext
		try:
			out["extent_y"] = float(max(0.1, float(ext_y)))
		except Exception:
			out["extent_y"] = default_ext
		out["debug_force_seconds"] = float(max(0.0, float(out.get("debug_force_seconds", 0.0))))
		dbg_c = out.get("debug_constant", None)
		out["debug_constant"] = None if dbg_c is None else float(np.clip(float(dbg_c), 0.0, 1.0))
		out["debug_outline"] = bool(out.get("debug_outline", False))
		out["debug_solid_panel"] = bool(out.get("debug_solid_panel", False))
		out["debug_fullscreen_panel"] = bool(out.get("debug_fullscreen_panel", False))
		mode = str(out.get("mode", "image")).lower()
		out["mode"] = mode if mode in ("image", "tiles") else "image"
		out["tiles_max_cols"] = max(8, int(out.get("tiles_max_cols", 64)))
		out["tiles_max_rows"] = max(6, int(out.get("tiles_max_rows", 36)))
		out["only"] = bool(out.get("only", False))
		pal = out.get("palette")
		if pal:
			out["palette"] = self._resolve_colors(pal)
		else:
			out["palette"] = None
		seed = out.get("seed")
		out["seed"] = int(seed) if seed is not None else None
		return out

	def construct(self):
		# Early debug panel to verify rendering pipeline unmistakably
		if bool(self._fractal_cfg.get("debug_fullscreen_panel", False)):
			try:
				dbg = Rectangle(width=float(manim_config.frame_width), height=float(manim_config.frame_height))
				dbg.set_stroke(width=0)
				dbg.set_fill(color=WHITE, opacity=1.0)
				dbg.set_z_index(10_000)
				self.add(dbg)
				print("visuals: initial fullscreen panel added; pausing 1s")
				self.wait(1.0)
			except Exception:
				pass
		# Optional fractal background behind everything
		if self._fractal_cfg.get("enabled", False):
			self._fractal_bg = _FractalBackground(self._fractal_cfg, fallback_colors=self._colors)
			bg_mobj = self._fractal_bg.get_mobject()
			# Place behind or in front depending on overlay flag
			try:
				bg_mobj.set_z_index(10 if self._fractal_cfg.get("overlay", False) else -10)
			except Exception:
				pass
			if bg_mobj is not None:
				try:
					w = float(manim_config.frame_width)
					h = float(manim_config.frame_height)
					if isinstance(bg_mobj, VGroup):
						# Tiles mode already sized to frame
						pass
					else:
						bg_mobj.scale_to_fit_height(h)
					bg_mobj.move_to([0.0, 0.0, 0.0])
				except Exception:
					pass
				# Always add tiles or image; if tiles, force foreground to ensure visibility
				is_tiles = isinstance(bg_mobj, VGroup)
				if self._fractal_cfg.get("overlay", False) or is_tiles:
					try:
						self.add_foreground_mobject(bg_mobj)
					except Exception:
						self.add(bg_mobj)
				else:
					try:
						self.add_to_back(bg_mobj)
					except Exception:
						self.add(bg_mobj)
				# Optional visible outline to confirm placement
				if bool(self._fractal_cfg.get("debug_outline", False)):
					try:
						rect = Rectangle(width=float(manim_config.frame_width), height=float(manim_config.frame_height))
						rect.set_stroke(color=WHITE, width=2.0)
						rect.set_fill(opacity=0.0)
						self.add_foreground_mobject(rect)
					except Exception:
						pass
				# Optional fullscreen solid panel to debug layering
				if bool(self._fractal_cfg.get("debug_fullscreen_panel", False)):
					try:
						panel = Rectangle(width=float(manim_config.frame_width), height=float(manim_config.frame_height))
						panel.set_stroke(width=0)
						panel.set_fill(color=WHITE, opacity=1.0)
						self.add_foreground_mobject(panel)
						print("visuals: added fullscreen debug panel")
					except Exception:
						pass
			try:
				print("visuals: fractal background enabled")
			except Exception:
				pass
			# Populate initial texture to avoid a blank frame
			try:
				self._fractal_bg.refresh(0.0, time.time())
			except Exception:
				pass

		# Optionally render fractal-only as the scene background layer to rule out layering issues
		if self._fractal_cfg.get("enabled", False) and bool(self._fractal_cfg.get("only", False)):
			print("visuals: fractal-only mode is active")
			# Create a full-screen solid background to verify draw
			try:
				bg_rect = Rectangle(width=float(manim_config.frame_width), height=float(manim_config.frame_height))
				bg_rect.set_stroke(width=0)
				bg_rect.set_fill(color=BLUE, opacity=0.3)
				bg_rect.set_z_index(-1000)
				bg_rect.set_depth_test(False)
				self.add(bg_rect)
			except Exception:
				pass
			# Keep only the fractal background running with periodic refresh via updater
			def bg_updater(_dt):
				arr = self.shared.get()
				if arr is None:
					arr = np.zeros((self.num_bands,), dtype=np.float32)
				now = time.time()
				hn = min(int(self._fractal_cfg.get("hf_bands_count", 8)), arr.shape[0])
				hf_energy = float(np.mean(arr[-hn:])) if hn > 0 else float(np.mean(arr))
				self._fractal_bg.refresh(hf_energy, now)
			if self._fractal_bg and self._fractal_bg.get_mobject() is not None:
				self.add(self._fractal_bg.get_mobject())
				self._fractal_bg.get_mobject().add_updater(lambda _m, dt: bg_updater(dt))
			self.wait(60.0)
			return

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

			# Update fractal background from selected band range (with debug overrides)
			if self._fractal_bg is not None:
				band_range = self._fractal_cfg.get("band_range")
				if isinstance(band_range, (list, tuple)) and len(band_range) == 2:
					start = max(0, int(band_range[0]))
					end = max(start + 1, int(band_range[1]))
					end = min(end, arr.shape[0])
					seg = arr[start:end]
					energy = float(np.mean(seg)) if seg.size > 0 else float(np.mean(arr))
				else:
					hn = min(int(self._fractal_cfg["hf_bands_count"]), arr.shape[0])
					energy = float(np.mean(arr[-hn:])) if hn > 0 else float(np.mean(arr))
				# Debug overrides
				force_s = float(self._fractal_cfg.get("debug_force_seconds", 0.0))
				dbg_c = self._fractal_cfg.get("debug_constant", None)
				start_time = getattr(self, "_fractal_debug_start", None)
				if start_time is None:
					self._fractal_debug_start = now
				if force_s > 0.0 and (now - self._fractal_debug_start) <= force_s:
					energy = float(1.0 if dbg_c is None else dbg_c)
				# Update underlying fractal mobject in-place
				self._fractal_bg.refresh(energy, now)

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



class _FractalBackground:
	"""Julia-set background rendered to an ImageMobject and driven by HF energy."""

	def __init__(self, cfg: dict, fallback_colors: list):
		self.cfg = cfg
		self.width = int(cfg["resolution"][0])
		self.height = int(cfg["resolution"][1])
		self.max_iter_base = int(cfg["max_iter"])
		self.julia_strength = float(cfg["julia_strength"])
		self.zoom_min, self.zoom_max = float(cfg["zoom_range"][0]), float(cfg["zoom_range"][1])
		self.extent_x = float(cfg.get("extent_x", 1.5))
		self.extent_y = float(cfg.get("extent_y", 1.5))
		self.update_interval = 1.0 / float(max(1, int(cfg["update_fps"])) )
		self.alpha = float(cfg["alpha"])
		self.color_speed = float(cfg["color_speed"]) 
		self.color_gamma = float(cfg.get("color_gamma", 1.0))
		self.color_scale = float(cfg.get("color_scale", 1.0))
		self.color_offset = float(cfg.get("color_offset", 0.0))
		self.interior_palette = bool(cfg.get("interior_palette", False))
		self._rng = random.Random(int(cfg["seed"])) if cfg.get("seed") is not None else random.Random()
		self._last_update_time = -1e9
		self._palette = self._build_palette(cfg.get("palette") or fallback_colors)
		self._phase = self._rng.random()
		print("visuals: fractal init", {
			"res": (self.width, self.height),
			"alpha": self.alpha,
			"opengl": True,
		})
		self._current_mobject: ImageMobject | None = None
		self._is_opengl: bool = False
		# Choose rendering mode: single image texture or tiling with Rectangles (guaranteed visible)
		mode = str(cfg.get("mode", "image")).lower()
		self._mode = mode
		if mode == "image":
			try:
				from PIL import Image as PILImage
				from manim.mobject.opengl.opengl_image_mobject import OpenGLImageMobject  # type: ignore
				alpha_byte = int(np.clip(self.alpha, 0.0, 1.0) * 255.0)
				pil = PILImage.new("RGBA", (2, 2), (0, 0, 0, alpha_byte))
				mobj = OpenGLImageMobject(pil)
				try:
					mobj.set_z_index(-10)
				except Exception:
					pass
				try:
					mobj.scale_to_fit_height(float(manim_config.frame_height))
				except Exception:
					pass
				try:
					mobj.set_depth_test(False)
				except Exception:
					pass
				self._current_mobject = mobj
				self._is_opengl = True
			except Exception:
				self._current_mobject = None
		else:
			# Tiles mode: build a VGroup of rectangles colored from the fractal image
			self._is_opengl = False
			self._tile_group = VGroup()
			cols = int(cfg.get("tiles_max_cols", 64))
			rows = int(cfg.get("tiles_max_rows", 36))
			fw = float(manim_config.frame_width)
			fh = float(manim_config.frame_height)
			tile_w = fw / float(cols)
			tile_h = fh / float(rows)
			for r in range(rows):
				for c in range(cols):
					rect = Rectangle(width=tile_w, height=tile_h)
					rect.set_stroke(width=0)
					rect.set_fill(WHITE, opacity=self.alpha)
					try:
						rect.set_depth_test(False)
					except Exception:
						pass
					x = -fw / 2.0 + (c + 0.5) * tile_w
					y = -fh / 2.0 + (r + 0.5) * tile_h
					rect.move_to([x, y, 0.0])
					self._tile_group.add(rect)
			try:
				self._tile_group.set_z_index(999)
				self._tile_group.set_depth_test(False)
			except Exception:
				pass
			self._current_mobject = self._tile_group  # type: ignore

	def _color_to_uint8(self, color_obj) -> np.ndarray:
		try:
			r, g, b = color_obj.get_rgb()
		except Exception:
			try:
				r, g, b = color_obj.to_rgb()
			except Exception:
				r, g, b = (0.0, 0.0, 0.0)
		arr = np.array([r, g, b], dtype=np.float32)
		return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)

	def _build_palette(self, colors: list) -> np.ndarray:
		lut = np.zeros((256, 3), dtype=np.uint8)
		if not colors:
			start = np.array([30, 60, 200], dtype=np.uint8)
			mid = np.array([120, 20, 160], dtype=np.uint8)
			end = np.array([220, 40, 60], dtype=np.uint8)
			for i in range(256):
				t = i / 255.0
				if t < 0.5:
					local = t / 0.5
					lut[i] = (start * (1 - local) + mid * local).astype(np.uint8)
				else:
					local = (t - 0.5) / 0.5
					lut[i] = (mid * (1 - local) + end * local).astype(np.uint8)
			return lut
		segments = max(1, len(colors) - 1)
		for i in range(256):
			t = i / 255.0
			seg = min(int(t * segments), segments - 1)
			if segments == 1:
				seg = 0
				local_t = t
			else:
				seg_width = 1.0 / segments
				local_t = (t - seg * seg_width) / seg_width
			c0 = self._color_to_uint8(colors[seg])
			c1 = self._color_to_uint8(colors[min(seg + 1, len(colors) - 1)])
			lut[i] = np.clip(np.round(c0 * (1 - local_t) + c1 * local_t), 0, 255).astype(np.uint8)
		return lut

	def _render_frame(self, hf_energy: float, now: float) -> np.ndarray:
		hf = float(np.clip(hf_energy, 0.0, 1.0))
		zoom = self.zoom_min + (self.zoom_max - self.zoom_min) * hf
		max_iter = int(self.max_iter_base * (0.6 + 0.8 * hf))
		radius = 0.7885 * (0.3 + 0.7 * hf)
		angle = (self._phase + now * self.color_speed * 0.8) * (2.0 * np.pi)
		c = radius * np.exp(1j * angle) * self.julia_strength
		w, h = self.width, self.height
		aspect = float(w) / float(h)
		x = np.linspace(-self.extent_x * aspect, self.extent_x * aspect, w, dtype=np.float32) / zoom
		y = np.linspace(-self.extent_y, self.extent_y, h, dtype=np.float32) / zoom
		X, Y = np.meshgrid(x, y)
		Z = X + 1j * Y
		Z_iter = Z.copy()
		counts = np.zeros(Z.shape, dtype=np.int32)
		mask = np.ones(Z.shape, dtype=bool)
		for i in range(max_iter):
			Z_iter[mask] = Z_iter[mask] * Z_iter[mask] + c
			escaped = np.abs(Z_iter) > 2.0
			newly = escaped & mask
			counts[newly] = i
			mask &= (~escaped)
			if not mask.any():
				break
		counts = counts.astype(np.float32)
		if max_iter > 0:
			counts = counts / float(max_iter)
		# Apply color mapping adjustments to spread palette usage
		mapped = counts
		try:
			if self.color_gamma != 1.0:
				mapped = np.power(np.clip(mapped, 0.0, 1.0), 1.0 / max(1e-6, self.color_gamma))
		except Exception:
			pass
		try:
			if self.color_scale != 1.0 or self.color_offset != 0.0:
				mapped = np.clip(mapped * self.color_scale + self.color_offset, 0.0, 1.0)
		except Exception:
			pass
		idx = np.clip((mapped * 255.0).astype(np.int32), 0, 255)
		img = self._palette[idx]
		# Color interior (non-escaped) points
		if self.interior_palette:
			try:
				img[mask] = self._palette[0]
			except Exception:
				pass
		else:
			try:
				img[mask] = np.array([30, 80, 160], dtype=np.uint8)
			except Exception:
				img[mask] = (img[mask] * 0.5).astype(np.uint8)
		# Log a checksum of the frame to verify updates
		try:
			cs = int(np.sum(img))
			print(f"visuals: fractal frame checksum={cs}")
		except Exception:
			pass
		return img

	def _to_rgba(self, img_rgb: np.ndarray) -> np.ndarray:
		if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
			alpha_byte = int(np.clip(self.alpha, 0.0, 1.0) * 255.0)
			a = np.full((img_rgb.shape[0], img_rgb.shape[1], 1), alpha_byte, dtype=np.uint8)
			return np.concatenate([img_rgb, a], axis=2)
		return img_rgb

	def _build_mobject(self, img_rgb: np.ndarray) -> ImageMobject:
		# Deprecated; OpenGL-only flow.
		return self._current_mobject if self._current_mobject is not None else ImageMobject(img_rgb)

	def _build_mobject_cpu(self, img_rgb: np.ndarray) -> ImageMobject:
		# Not used under OpenGL renderer; kept for compatibility.
		return ImageMobject(img_rgb)

	def get_mobject(self) -> ImageMobject:
		return self._current_mobject  # type: ignore

	def refresh(self, hf_energy: float, now: float) -> ImageMobject | None:
		if (now - self._last_update_time) < self.update_interval:
			return None
		self._last_update_time = now
		if bool(self.cfg.get("debug_solid_panel", False)):
			img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
			img[:, :, 0] = 255
			img[:, :, 1] = 64
			img[:, :, 2] = 64
		else:
			img = self._render_frame(hf_energy=hf_energy, now=now)
		# Update existing mobject's texture if possible; else replace
		if self._mode == "image" and self._is_opengl and (self._current_mobject is not None):
			try:
				rgba = self._to_rgba(img)
				rgba_f = (rgba.astype(np.float32) / 255.0)
				self._current_mobject.set_data(rgba_f)  # type: ignore[attr-defined]
				# Force a re-upload/redraw by nudging opacity very slightly
				try:
					current_opacity = float(self.alpha)
					self._current_mobject.set_opacity(min(1.0, max(0.0, current_opacity)))
				except Exception:
					pass
				return None
			except Exception as e:
				print(f"visuals: fractal OpenGL update failed: {e}")
				self._is_opengl = False
		if self._mode == "tiles" and hasattr(self, "_tile_group"):
			# Update colors of tiles based on downsampled image
			cols = int(self.cfg.get("tiles_max_cols", 64))
			rows = int(self.cfg.get("tiles_max_rows", 36))
			# Downsample image to tiles grid
			r_idx = np.linspace(0, img.shape[0] - 1, rows).astype(int)
			c_idx = np.linspace(0, img.shape[1] - 1, cols).astype(int)
			down = img[np.ix_(r_idx, c_idx)]  # rows x cols x 3
			k = 0
			for r in range(rows):
				for c in range(cols):
					color = down[r, c, :]
					# Convert to hex string accepted by Manim's set_fill
					rf, gf, bf = (float(color[0]) / 255.0, float(color[1]) / 255.0, float(color[2]) / 255.0)
					r8 = int(round(rf * 255.0))
					g8 = int(round(gf * 255.0))
					b8 = int(round(bf * 255.0))
					hex_color = f"#{r8:02x}{g8:02x}{b8:02x}"
					self._tile_group[k].set_fill(color=hex_color, opacity=self.alpha)
					self._tile_group[k].set_stroke(width=0)
					try:
						self._tile_group[k].invalidate_shader_data()  # force GL uniform refresh
					except Exception:
						pass
					k += 1
			# Invalidate group once per refresh as well
			try:
				self._tile_group.invalidate_shader_data()
			except Exception:
				pass
			return None
		return None
