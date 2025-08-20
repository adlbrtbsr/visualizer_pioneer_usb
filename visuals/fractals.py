from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np
from manim import VGroup, Rectangle, ImageMobject, config as manim_config, WHITE

# Optional acceleration via numba (falls back to NumPy if unavailable)
try:
    from numba import njit, prange  # type: ignore
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _numba_julia_counts(zr: np.ndarray, zi: np.ndarray, zoom: float, max_iter: int, cr: float, ci: float):
        h, w = zr.shape
        counts = np.zeros((h, w), np.int32)
        mask = np.ones((h, w), np.bool_)
        for y in prange(h):
            for x in range(w):
                zr0 = zr[y, x] / zoom
                zi0 = zi[y, x] / zoom
                it = 0
                while it < max_iter:
                    zr2 = zr0 * zr0 - zi0 * zi0 + cr
                    zi0 = 2.0 * zr0 * zi0 + ci
                    zr0 = zr2
                    if zr0 * zr0 + zi0 * zi0 > 4.0:
                        counts[y, x] = it
                        mask[y, x] = False
                        break
                    it += 1
        return counts, mask


def default_fractal_cfg() -> dict:
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
		"mode": "image",
		"tiles_max_cols": 64,
		"tiles_max_rows": 36,
		"only": False,
		# Performance options
		"accel": "auto",  # 'auto' | 'numba' | 'numpy'
		"tiles_compute_at_grid": True,
		# Incremental compute
		"reuse_across_frames": True,
		"iters_per_frame": 8,
		"param_reset_threshold": 0.03,
	}


def sanitize_fractal_cfg(cfg: dict, resolve_colors_fn) -> dict:
	out = dict(cfg)
	res = out.get("resolution", [256, 144])
	if not isinstance(res, (list, tuple)) or len(res) != 2:
		res = [256, 144]
	out["resolution"] = [max(32, int(res[0])), max(32, int(res[1]))]
	out["max_iter"] = max(10, int(out.get("max_iter", 120)))
	out["hf_bands_count"] = max(1, int(out.get("hf_bands_count", 8)))
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
	try:
		default_ext = float(1.5)
	except Exception:
		default_ext = 1.5
	ext = out.get("extent", None)
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
	accel = str(out.get("accel", "auto")).lower()
	if accel not in ("auto", "numba", "numpy"):
		accel = "auto"
	out["accel"] = accel
	out["tiles_compute_at_grid"] = bool(out.get("tiles_compute_at_grid", True))
	out["reuse_across_frames"] = bool(out.get("reuse_across_frames", True))
	out["iters_per_frame"] = max(1, int(out.get("iters_per_frame", 8)))
	out["param_reset_threshold"] = float(max(0.0, float(out.get("param_reset_threshold", 0.03))))
	pal = out.get("palette")
	if pal:
		out["palette"] = resolve_colors_fn(pal)
	else:
		out["palette"] = None
	seed = out.get("seed")
	out["seed"] = int(seed) if seed is not None else None
	return out


class FractalBackground:
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
		self._aspect = float(self.width) / float(self.height)
		x_1d = np.linspace(-self.extent_x * self._aspect, self.extent_x * self._aspect, self.width, dtype=np.float32)
		y_1d = np.linspace(-self.extent_y, self.extent_y, self.height, dtype=np.float32)
		X, Y = np.meshgrid(x_1d, y_1d)
		self._Z_template = (X + 1j * Y).astype(np.complex64)
		self._Z_iter = np.empty_like(self._Z_template)
		self._counts = np.zeros((self.height, self.width), dtype=np.uint16)
		self._mask = np.ones((self.height, self.width), dtype=bool)
		print("visuals: fractal init", {
			"res": (self.width, self.height),
			"alpha": self.alpha,
			"opengl": True,
		})
		self._current_mobject: ImageMobject | None = None
		self._is_opengl: bool = False
		mode = str(cfg.get("mode", "image")).lower()
		self._mode = mode
		# Incremental iteration state
		self._reuse = bool(cfg.get("reuse_across_frames", True))
		self._iters_per_frame = int(cfg.get("iters_per_frame", 8))
		self._param_reset_threshold = float(cfg.get("param_reset_threshold", 0.03))
		self._last_zoom = None
		self._last_c = None
		self._accum_max_iter = 0
		if mode == "image":
			try:
				from PIL import Image as PILImage
				from manim.mobject.opengl.opengl_image_mobject import OpenGLImageMobject  # type: ignore
				alpha_byte = int(np.clip(self.alpha, 0.0, 1.0) * 255.0)
				pil = PILImage.new("RGBA", (self.width, self.height), (0, 0, 0, alpha_byte))
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
				try:
					mobj.set_opacity(self.alpha)
				except Exception:
					pass
				self._current_mobject = mobj
				self._is_opengl = True
			except Exception:
				self._is_opengl = False
				self._mode = "tiles"
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
				self._tiles_cols = cols
				self._tiles_rows = rows
				# Optionally compute fractal exactly at tile grid resolution
				if bool(self.cfg.get("tiles_compute_at_grid", True)):
					self.width = cols
					self.height = rows
					x_1d = np.linspace(-self.extent_x * self._aspect, self.extent_x * self._aspect, self.width, dtype=np.float32)
					y_1d = np.linspace(-self.extent_y, self.extent_y, self.height, dtype=np.float32)
					X, Y = np.meshgrid(x_1d, y_1d)
					self._Z_template = (X + 1j * Y).astype(np.complex64)
					self._Z_iter = np.empty_like(self._Z_template)
					self._counts = np.zeros((self.height, self.width), dtype=np.uint16)
					self._mask = np.ones((self.height, self.width), dtype=bool)
				self._r_idx = np.linspace(0, self.height - 1, rows).astype(int)
				self._c_idx = np.linspace(0, self.width - 1, cols).astype(int)
				self._current_mobject = self._tile_group  # type: ignore
		else:
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
			self._tiles_cols = cols
			self._tiles_rows = rows
			# Optionally compute fractal exactly at tile grid resolution
			if bool(self.cfg.get("tiles_compute_at_grid", True)):
				self.width = cols
				self.height = rows
				x_1d = np.linspace(-self.extent_x * self._aspect, self.extent_x * self._aspect, self.width, dtype=np.float32)
				y_1d = np.linspace(-self.extent_y, self.extent_y, self.height, dtype=np.float32)
				X, Y = np.meshgrid(x_1d, y_1d)
				self._Z_template = (X + 1j * Y).astype(np.complex64)
				self._Z_iter = np.empty_like(self._Z_template)
				self._counts = np.zeros((self.height, self.width), dtype=np.uint16)
				self._mask = np.ones((self.height, self.width), dtype=bool)
			self._r_idx = np.linspace(0, self.height - 1, rows).astype(int)
			self._c_idx = np.linspace(0, self.width - 1, cols).astype(int)
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
									# noqa: E101 (tabs kept intentionally)
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
		max_iter_target = int(self.max_iter_base * (0.6 + 0.8 * hf))
		radius = 0.7885 * (0.3 + 0.7 * hf)
		angle = (self._phase + now * self.color_speed * 0.8) * (2.0 * np.pi)
		c = radius * np.exp(1j * angle) * self.julia_strength
		# Reset on parameter jump
		need_reset = True
		if self._reuse and (self._last_zoom is not None) and (self._last_c is not None):
			if abs(zoom - self._last_zoom) <= self._param_reset_threshold and abs(c - self._last_c) <= self._param_reset_threshold:
				need_reset = False
		if need_reset:
			np.copyto(self._Z_iter, self._Z_template)
			self._Z_iter /= np.float32(zoom)
			self._counts.fill(0)
			self._mask.fill(True)
			self._accum_max_iter = 0
		# Incremental step budget
		step_iters = int(self._iters_per_frame)
		target = min(max_iter_target, self._accum_max_iter + step_iters)
		# Choose backend
		use_numba = (_HAS_NUMBA and str(self.cfg.get("accel", "auto")).lower() in ("auto", "numba"))
		if use_numba:
			try:
				zr = self._Z_iter.real.astype(np.float32)
				zi = self._Z_iter.imag.astype(np.float32)
				counts_i, mask = _numba_julia_counts(zr, zi, 1.0, int(target - self._accum_max_iter), float(c.real), float(c.imag))  # type: ignore
				# Merge results: advance Z_iter by running explicit loop to sync state
				# Fallback: run numpy for the state advance to keep code simple
				use_numba = False
			except Exception:
				use_numba = False
		if not use_numba:
			for i in range(self._accum_max_iter, target):
				self._Z_iter *= self._Z_iter
				self._Z_iter += c
				mod2 = (self._Z_iter.real * self._Z_iter.real) + (self._Z_iter.imag * self._Z_iter.imag)
				escaped = mod2 > 4.0
				newly = escaped & self._mask
				self._counts[newly] = i
				self._mask &= (~escaped)
				if not self._mask.any():
					break
			self._accum_max_iter = target
		# Normalize counts by current target
		counts = self._counts.astype(np.float32)
		if target > 0:
			counts = counts / float(target)
		mask = self._mask
		self._last_zoom = zoom
		self._last_c = c
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
		return img

	def _to_rgba(self, img_rgb: np.ndarray) -> np.ndarray:
		if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
			alpha_byte = int(np.clip(self.alpha, 0.0, 1.0) * 255.0)
			a = np.full((img_rgb.shape[0], img_rgb.shape[1], 1), alpha_byte, dtype=np.uint8)
			return np.concatenate([img_rgb, a], axis=2)
		return img_rgb

	def _build_mobject_cpu(self, img_rgb: np.ndarray) -> ImageMobject:
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
		if self._mode == "image" and (self._current_mobject is not None):
			try:
				from PIL import Image as PILImage
				rgba = self._to_rgba(img)
				pil = PILImage.fromarray(rgba, mode="RGBA")
				if self._is_opengl:
					try:
						from manim.mobject.opengl.opengl_image_mobject import OpenGLImageMobject  # type: ignore
						new_im = OpenGLImageMobject(pil)
					except Exception:
						new_im = self._build_mobject_cpu(img)
				else:
					new_im = self._build_mobject_cpu(img)
				try:
					new_im.set_opacity(self.alpha)
				except Exception:
					pass
				try:
					new_im.scale_to_fit_height(float(manim_config.frame_height))
				except Exception:
					pass
				self._current_mobject.become(new_im)
				return None
			except Exception:
				pass
		if self._mode == "image" and (not self._is_opengl) and (self._current_mobject is not None):
			try:
				imobj = self._build_mobject_cpu(img)
				try:
					imobj.set_opacity(self.alpha)
				except Exception:
					pass
				try:
					imobj.scale_to_fit_height(float(manim_config.frame_height))
				except Exception:
					pass
				self._current_mobject.become(imobj)
				return None
			except Exception:
				pass
		if self._mode == "tiles" and hasattr(self, "_tile_group"):
			cols = int(self.cfg.get("tiles_max_cols", 64))
			rows = int(self.cfg.get("tiles_max_rows", 36))
			down = img[np.ix_(self._r_idx, self._c_idx)]
			k = 0
			for r in range(rows):
				for c in range(cols):
					color = down[r, c, :]
					rf, gf, bf = (float(color[0]) / 255.0, float(color[1]) / 255.0, float(color[2]) / 255.0)
					r8 = int(round(rf * 255.0))
					g8 = int(round(gf * 255.0))
					b8 = int(round(bf * 255.0))
					hex_color = f"#{r8:02x}{g8:02x}{b8:02x}"
					self._tile_group[k].set_fill(color=hex_color, opacity=self.alpha)
					self._tile_group[k].set_stroke(width=0)
					k += 1
			try:
				self._tile_group.invalidate_shader_data()
			except Exception:
				pass
			return None
		return None


