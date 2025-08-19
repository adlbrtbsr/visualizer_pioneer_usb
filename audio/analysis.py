from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Iterable[float]]


def _as_mono(frames: np.ndarray) -> np.ndarray:

	if frames.ndim == 1:
		return frames.astype(np.float32, copy=False)
	if frames.ndim == 2:
		# Average channels to mono for spectrum analysis by default
		return frames.astype(np.float32, copy=False).mean(axis=1)
	raise ValueError("frames must be 1D (mono) or 2D (frames, channels)")


def _get_window(window: str, n_fft: int) -> np.ndarray:
	name = (window or "hann").lower()
	if name in ("hann", "hanning"):
		w = np.hanning(n_fft).astype(np.float32)
	elif name in ("hamming",):
		w = np.hamming(n_fft).astype(np.float32)
	elif name in ("blackman",):
		w = np.blackman(n_fft).astype(np.float32)
	elif name in ("rect", "rectangular", "boxcar"):
		w = np.ones(n_fft, dtype=np.float32)
	else:
		raise ValueError(f"Unsupported window type: {window}")
	return w


def _frame_signal(signal: np.ndarray, n_fft: int, hop_size: int) -> np.ndarray:

	if hop_size <= 0:
		raise ValueError("hop_size must be > 0")
	if n_fft <= 0:
		raise ValueError("n_fft must be > 0")

	num_samples = signal.shape[0]
	if num_samples < n_fft:
		pad = n_fft - num_samples
		signal = np.pad(signal, (0, pad), mode="constant")
		num_samples = signal.shape[0]

	# Number of frames with padding on the tail to include last partial frame
	num_frames = 1 + int(math.ceil((num_samples - n_fft) / float(hop_size)))
	pad_needed = (n_fft + (num_frames - 1) * hop_size) - num_samples
	if pad_needed > 0:
		signal = np.pad(signal, (0, pad_needed), mode="constant")

	# Build strided view for efficiency
	stride = signal.strides[0]
	frames = np.lib.stride_tricks.as_strided(
		signal,
		shape=(num_frames, n_fft),
		strides=(hop_size * stride, stride),
		writeable=False,
	)
	return frames


def compute_spectrum(frames: np.ndarray, window: str, n_fft: int, hop_size: int) -> np.ndarray:
	"""
	Compute magnitude spectrum (rFFT) over time for a mono-ized signal.

	Parameters
	- frames: np.ndarray of shape (num_samples,) or (num_samples, channels)
	- window: window name ("hann", "hamming", "blackman", "rect")
	- n_fft: FFT size
	- hop_size: hop between adjacent frames

	Returns
	- magnitude spectrum as np.ndarray with shape (num_bins, num_frames),
	  where num_bins = n_fft//2 + 1
	"""
	mono = _as_mono(frames)
	win = _get_window(window, n_fft)
	framed = _frame_signal(mono, n_fft=n_fft, hop_size=hop_size)
	windowed = framed * win[np.newaxis, :]
	# FFT along last axis of each frame, then take absolute value
	stft = np.fft.rfft(windowed, n=n_fft, axis=1)
	mag = np.abs(stft).astype(np.float32)
	# Return as (freq_bins, time_frames)
	return mag.T


def _freq_edges_for_log_spacing(
	num_bands: int,
	sample_rate: int,
	min_freq: float,
	max_freq: Optional[float],
) -> np.ndarray:
	if max_freq is None:
		max_freq = sample_rate / 2.0
	if min_freq <= 0.0:
		min_freq = 1.0
	if min_freq >= max_freq:
		raise ValueError("min_freq must be < max_freq")
	return np.geomspace(min_freq, max_freq, num_bands + 1)


def _bin_edges_from_freq_edges(
	freq_edges: np.ndarray,
	n_fft: int,
	sample_rate: int,
) -> np.ndarray:
	bin_freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
	bin_indices = np.searchsorted(bin_freqs, freq_edges, side="left")
	bin_indices = np.clip(bin_indices, 0, bin_freqs.shape[0] - 1)
	# Ensure strictly increasing and cover full range
	bin_indices[0] = 0
	bin_indices[-1] = bin_freqs.shape[0] - 1
	for i in range(1, bin_indices.shape[0]):
		if bin_indices[i] <= bin_indices[i - 1]:
			bin_indices[i] = min(bin_indices[i - 1] + 1, bin_freqs.shape[0] - 1)
	return bin_indices


def aggregate_bands(
	mag_spectrum: np.ndarray,
	scheme: Union[str, dict],
) -> np.ndarray:
	"""
	Aggregate per-FFT-bin magnitudes into a fixed number of bands.

	Parameters
	- mag_spectrum: ndarray, shape (num_bins, num_frames)
	- scheme: either a string like "linear:64" / "log:64" or a dict with keys:
	  {"mode": "linear"|"log", "num_bands": int, "sample_rate": int, "n_fft": int, "min_freq": float, "max_freq": float}

	Returns
	- band_energies: ndarray, shape (num_bands, num_frames), linear scale in [0, +inf)
	"""
	if isinstance(scheme, str):
		mode, num_str = scheme.split(":", 1)
		mode = mode.strip().lower()
		num_bands = int(num_str)
		params = {"mode": mode, "num_bands": num_bands}
	else:
		params = dict(scheme)
		params["mode"] = params.get("mode", "linear").lower()

	mode = params["mode"]
	num_bands = int(params.get("num_bands", 32))
	if num_bands <= 0:
		raise ValueError("num_bands must be > 0")
	bins, frames = mag_spectrum.shape

	if mode == "linear":
		# Evenly divide bins across bands
		edges = np.linspace(0, bins - 1, num_bands + 1, dtype=np.int32)
		# Guarantee strictly increasing
		edges[0] = 0
		edges[-1] = bins - 1
	else:
		# Log spacing requires sample_rate and n_fft
		sample_rate = int(params.get("sample_rate"))
		n_fft = int(params.get("n_fft"))
		if not sample_rate or not n_fft:
			raise ValueError("log scheme requires 'sample_rate' and 'n_fft'")
		# Handle optional min/max when YAML provides null
		min_raw = params.get("min_freq", 20.0)
		max_raw = params.get("max_freq", None)
		min_freq = float(20.0 if min_raw is None else min_raw)
		max_freq = float((sample_rate / 2.0) if max_raw is None else max_raw)
		freq_edges = _freq_edges_for_log_spacing(num_bands, sample_rate, min_freq, max_freq)
		edges = _bin_edges_from_freq_edges(freq_edges, n_fft=n_fft, sample_rate=sample_rate)

	band_means = np.empty((num_bands, frames), dtype=np.float32)
	for i in range(num_bands):
		start = edges[i]
		end = edges[i + 1]
		if end <= start:
			end = min(start + 1, bins - 1)
		# Inclusive end index: use +1 to include 'end'
		segment = mag_spectrum[start : end + 1, :]
		band_means[i, :] = segment.mean(axis=0)
	return band_means


@dataclass
class Smoother:
	"""Per-sample or per-vector attack-release smoother using EMA.

	attack: alpha when input is rising (0..1). 1.0 => no smoothing on rise
	release: alpha when input is falling (0..1). 1.0 => no smoothing on fall
	"""
	attack: float = 0.5
	release: float = 0.1
	_state: Optional[np.ndarray] = None

	def reset(self, value: Optional[ArrayLike] = None) -> None:
		if value is None:
			self._state = None
			return
		arr = np.asarray(value, dtype=np.float32)
		self._state = arr.copy()

	def update(self, value: ArrayLike) -> np.ndarray:
		x = np.asarray(value, dtype=np.float32)
		if self._state is None:
			self._state = x.copy()
			return self._state
		alpha_rise = float(self.attack)
		alpha_fall = float(self.release)
		# Broadcast-safe update
		mask = x >= self._state
		self._state[mask] = (1.0 - alpha_rise) * self._state[mask] + alpha_rise * x[mask]
		self._state[~mask] = (1.0 - alpha_fall) * self._state[~mask] + alpha_fall * x[~mask]
		return self._state


@dataclass
class Normalizer:
	"""Rolling normalizer to map values into [0, 1].

	mode: 'peak' or 'percentile'
	window: number of recent frames to consider (for percentile)
	decay: peak hold decay per update (0..1). 0 => infinite hold, 1 => immediate
	floor: minimum denominator to avoid divide-by-zero
	"""
	mode: str = "peak"
	window: int = 60
	decay: float = 0.01
	floor: float = 1e-6
	_peak: Optional[np.ndarray] = None
	_buffer: Optional[deque] = None

	def reset(self) -> None:
		self._peak = None
		self._buffer = None

	def _ensure_shapes(self, value: np.ndarray) -> None:
		if self.mode == "peak":
			if self._peak is None:
				self._peak = np.maximum(value.astype(np.float32), self.floor)
		else:
			if self._buffer is None:
				self._buffer = deque(maxlen=int(self.window))
			self._buffer.append(value.astype(np.float32))

	def update(self, value: ArrayLike) -> np.ndarray:
		x = np.asarray(value, dtype=np.float32)
		self._ensure_shapes(x)
		if self.mode == "peak":
			# Peak hold with decay
			self._peak = np.maximum(x, (1.0 - float(self.decay)) * self._peak)
			den = np.maximum(self._peak, self.floor)
			return np.clip(x / den, 0.0, 1.0)
		# Percentile mode
		self._buffer.append(x)
		stack = np.stack(list(self._buffer), axis=0)
		# 95th percentile by default of recent window to limit outliers
		p = np.percentile(stack, 95, axis=0)
		den = np.maximum(p.astype(np.float32), self.floor)
		return np.clip(x / den, 0.0, 1.0)


@dataclass
class BeatDetector:
	"""Simple energy-based beat detector with refractory period.

	threshold: multiplier above long-term average to trigger (e.g., 1.5)
	short_window: recent frames for short-term energy
	long_window: recent frames for long-term baseline
	refractory_frames: minimum frames between triggers
	"""
	threshold: float = 1.5
	short_window: int = 5
	long_window: int = 43
	refractory_frames: int = 10
	_energy_short: deque | None = None
	_energy_long: deque | None = None
	_frames_since: int = 1_000_000

	def reset(self) -> None:
		self._energy_short = None
		self._energy_long = None
		self._frames_since = 1_000_000

	def update(self, band_energies: ArrayLike) -> bool:
		# band_energies expected shape (num_bands,) at a single time step
		arr = np.asarray(band_energies, dtype=np.float32)
		energy = float(np.mean(arr * arr))
		if self._energy_short is None:
			self._energy_short = deque([energy] * self.short_window, maxlen=self.short_window)
			self._energy_long = deque([energy] * self.long_window, maxlen=self.long_window)
			self._frames_since = self.refractory_frames + 1
			return False
		self._energy_short.append(energy)
		self._energy_long.append(energy)
		self._frames_since += 1
		avg_short = float(np.mean(self._energy_short))
		avg_long = float(np.mean(self._energy_long))
		trigger = avg_short > (self.threshold * max(avg_long, 1e-12))
		if trigger and self._frames_since > self.refractory_frames:
			self._frames_since = 0
			return True
		return False


def magnitude_to_dbfs(magnitude: np.ndarray, ref: float = 1.0, min_db: float = -80.0) -> np.ndarray:

	mag = np.asarray(magnitude, dtype=np.float32)
	mag = np.maximum(mag, 1e-12)
	db = 20.0 * np.log10(mag / float(ref))
	return np.maximum(db, float(min_db)).astype(np.float32)


