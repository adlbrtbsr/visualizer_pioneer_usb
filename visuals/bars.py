from __future__ import annotations

from typing import Sequence, List, Tuple

import numpy as np
from manim import VGroup, Rectangle, ValueTracker

from .utils import interpolate_color


class BarsLayer:
	"""Creates and updates a horizontal bar spectrum."""

	def __init__(self, num_bands: int, min_height: float, scene_scale: float, baseline_y: float, scene_width: float, colors: Sequence, bar_opacity: float):
		self.num_bands = int(num_bands)
		self.min_height = float(min_height)
		self.scene_scale = float(scene_scale)
		self.baseline_y = float(baseline_y)
		self.scene_width = float(scene_width)
		self._colors = list(colors)
		self._bar_opacity = float(np.clip(bar_opacity, 0.0, 1.0))
		self.group: VGroup = VGroup()
		self.trackers: list[ValueTracker] = []
		self.rects: list[Rectangle] = []
		self._build()

	def _build(self) -> None:
		width = self.scene_width
		gap = 0.05
		bar_width = max((width - gap * (self.num_bands - 1)) / self.num_bands, 0.05)
		for i in range(self.num_bands):
			tracker = ValueTracker(self.min_height)
			self.trackers.append(tracker)
			h0 = max(tracker.get_value(), self.min_height) * self.scene_scale
			rect = Rectangle(width=bar_width, height=h0)
			rect.set_fill(color=interpolate_color(h0, self._colors), opacity=self._bar_opacity)
			rect.set_stroke(width=0)
			x = -width / 2.0 + (bar_width + gap) * i + bar_width / 2.0
			rect.move_to([x, self.baseline_y + h0 / 2.0, 0.0])
			self.rects.append(rect)
			self.group.add(rect)

	def update(self, arr: np.ndarray) -> None:
		arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
		arr = np.maximum(arr, self.min_height)
		for i, tracker in enumerate(self.trackers):
			val = float(arr[i])
			tracker.set_value(val)
			rect = self.rects[i]
			rect.set_fill(color=interpolate_color(val, self._colors), opacity=self._bar_opacity)
			rect.set_stroke(width=0)
			rect.stretch_to_fit_height(val * self.scene_scale)
			x = rect.get_center()[0]
			rect.move_to([x, self.baseline_y + (val * self.scene_scale) / 2.0, 0.0])


