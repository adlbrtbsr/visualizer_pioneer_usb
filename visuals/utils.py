from __future__ import annotations

from typing import Optional, Sequence, List

import numpy as np
from manim import GREEN


def resolve_colors(colors: Optional[Sequence]) -> List:
	"""Resolve a list of color names/objects to Manim Color objects.

	Accepts strings referencing manim color names or already-instantiated
	color-like objects. Invalid entries are skipped.
	"""
	if not colors:
		return []
	resolved: list = []
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


def interpolate_color(v: float, colors: Sequence) -> any:
	"""Map v in [0,1] to a color along a simple gradient list.

	Returns GREEN if colors are missing. Performs piecewise linear
	interpolation across the provided color sequence.
	"""
	v = float(np.clip(v, 0.0, 1.0))
	if not colors:
		return GREEN
	if len(colors) == 1:
		return colors[0]
	step = 1.0 / (len(colors) - 1)
	idx = min(int(v // step), len(colors) - 2)
	local_t = (v - idx * step) / step
	return colors[idx].interpolate(colors[idx + 1], local_t)


