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
)


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

	def __init__(self, renderer=None, shared: Optional[SharedBands] = None, num_bands: int = 32, min_height: float = 0.05, **kwargs):
		# Accept Manim's renderer as positional arg and pass to super
		super().__init__(renderer, **kwargs)
		self.shared = shared or SharedBands()
		self.num_bands = int(num_bands)
		self.min_height = float(min_height)
		self._trackers: list[ValueTracker] = []
		self._bars: VGroup | None = None
		self._colors = [BLUE, GREEN, RED]

	def construct(self):
		# Layout bars centered
		width = 12.0
		gap = 0.05
		bar_width = max((width - gap * (self.num_bands - 1)) / self.num_bands, 0.05)
		height_base = 0.2

		bars = VGroup()
		self._trackers = []
		self._rects: list[Rectangle] = []
		for i in range(self.num_bands):
			tracker = ValueTracker(self.min_height)
			self._trackers.append(tracker)
			h0 = max(tracker.get_value(), self.min_height)
			rect = Rectangle(width=bar_width, height=h0)
			rect.set_fill(color=interpolate_color(h0, self._colors), opacity=0.9)
			rect.set_stroke(width=0)
			x = -width / 2.0 + (bar_width + gap) * i + bar_width / 2.0
			rect.move_to([x, -3.0 + h0 / 2.0, 0.0])
			self._rects.append(rect)
			bars.add(rect)

		self._bars = bars
		self.add(bars)

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
				rect.set_fill(color=interpolate_color(val, self._colors), opacity=0.9)
				rect.set_stroke(width=0)
				rect.stretch_to_fit_height(val)
				# Recompute position to keep bottom baseline stable
				x = rect.get_center()[0]
				rect.move_to([x, -3.0 + val / 2.0, 0.0])

		# Attach updater to the bars group (ensures it's called each frame)
		self._bars.add_updater(lambda _m, dt: update_trackers(dt))
		self.wait(60.0)  # Keep scene running for interactive session


