from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
from manim import (
	Scene,
	VGroup,
	Rectangle,
	config as manim_config,
	BLUE, RED, GREEN, WHITE,
)
import random

from .shared import SharedBands
from .utils import resolve_colors
from .bars import BarsLayer
from .shapes import ShapesLayer, default_shapes_cfg, sanitize_shapes_cfg
from .fractals import FractalBackground, default_fractal_cfg, sanitize_fractal_cfg


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
		self._rng = random.Random()

		# Colors
		self._colors = resolve_colors(color_scheme) if color_scheme is not None else [BLUE, GREEN, RED]

		# Shapes state/config
		self._shapes_cfg = default_shapes_cfg()
		if isinstance(shapes_config, dict):
			self._shapes_cfg.update(sanitize_shapes_cfg(shapes_config))
		if self._shapes_cfg.get("colors"):
			self._shapes_cfg["colors"] = resolve_colors(self._shapes_cfg["colors"]) or self._colors
		self._shapes_layer: ShapesLayer | None = None

		# Fractal background state/config
		self._fractal_cfg = default_fractal_cfg()
		if isinstance(fractal_config, dict):
			self._fractal_cfg.update(sanitize_fractal_cfg(fractal_config, resolve_colors))
		self._fractal_bg: FractalBackground | None = None

		# Bars layer
		self._bars_layer: BarsLayer | None = None
		self._bars: VGroup | None = None

		# Logging cadence
		self._last_log_time = 0.0

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
			self._fractal_bg = FractalBackground(self._fractal_cfg, fallback_colors=self._colors)
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

		# Bars layer
		self._bars_layer = BarsLayer(
			num_bands=self.num_bands,
			min_height=self.min_height,
			scene_scale=self.scene_scale,
			baseline_y=self.baseline_y,
			scene_width=self.scene_width,
			colors=self._colors,
			bar_opacity=self._bar_opacity,
		)
		self._bars = self._bars_layer.group
		self.add(self._bars)

		# Shapes layer above bars
		self._shapes_layer = ShapesLayer(
			rng=self._rng,
			colors=self._shapes_cfg.get("colors") or self._colors,
			cfg=self._shapes_cfg,
			scene_scale=self.scene_scale,
		)
		self._shapes_group = self._shapes_layer.get_group()
		self.add(self._shapes_group)

		# Updater: poll shared bands once per frame
		self._last_log_time = 0.0

		def update_frame(_dt):
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

			# Update bars
			if self._bars_layer is not None:
				self._bars_layer.update(arr)

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
				self._fractal_bg.refresh(energy, now)

			# Spawn shapes based on low frequencies
			if self._shapes_layer is not None and self._shapes_cfg.get("enabled", True):
				low_n = min(int(self._shapes_cfg["low_bands_count"]), arr.shape[0])
				low_energy = float(np.mean(arr[:low_n])) if low_n > 0 else 0.0
				self._shapes_layer.maybe_spawn(low_energy, now)
				self._shapes_layer.update(now)

		# Attach updater to the bars group (ensures it's called each frame)
		self._bars.add_updater(lambda _m, dt: update_frame(dt))
		self.wait(60.0)  # Keep scene running for interactive session
