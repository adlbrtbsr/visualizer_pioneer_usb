from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
import argparse
import os
from typing import Optional

import numpy as np
import yaml

# Ensure project root importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))

from audio import AudioCapture, AudioConfig, list_devices
from audio.analysis import compute_spectrum, aggregate_bands, Smoother, Normalizer
from visuals import SpectrumBarsScene as BaseSpectrumBarsScene, SharedBands
from manim import config as manim_config
try:
	import soundcard as sc  # optional fallback
except Exception:
	sc = None
import sys
import ctypes
import warnings


def _ensure_ffmpeg_path() -> None:
	"""Set manim_config.ffmpeg_executable if ffmpeg.exe is found in common locations."""
	candidates: list[str] = []
	localapp = os.environ.get("LOCALAPPDATA")
	if localapp:
		candidates.append(str(Path(localapp) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"))
	# Common installs
	candidates.append("C\\Program Files\\ffmpeg\\bin\\ffmpeg.exe")
	candidates.append("C\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe")
	for c in candidates:
		try:
			if Path(c).is_file():
				manim_config.ffmpeg_executable = c
				break
		except Exception:
			pass


_ensure_ffmpeg_path()


def load_yaml(path: Path) -> dict:
	if not path.is_file():
		return {}
	with path.open("r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


class SilenceError(Exception):
	pass


def analysis_worker(shared: SharedBands, audio_cfg_path: Path, analysis_cfg_path: Path, stop_event: threading.Event) -> None:
	# Initialize COM on this worker thread for soundcard/MediaFoundation
	if sys.platform.startswith("win"):
		try:
			ctypes.windll.ole32.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED
		except Exception:
			pass
	# Suppress noisy soundcard discontinuity warnings
	try:
		from soundcard import SoundcardRuntimeWarning  # type: ignore
		warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
	except Exception:
		pass
	audio_cfg_data = load_yaml(audio_cfg_path)
	analysis_cfg = load_yaml(analysis_cfg_path)

	# Environment variable overrides for one-off runs
	def _as_bool(s: str) -> bool:
		return s.strip().lower() in ("1", "true", "yes", "on")
	env_overrides: list[tuple[str, str, object]] = [
		("AUDIO_DEVICE_INDEX", "device_index", int),
		("AUDIO_DEVICE", "device_substring", str),
		("AUDIO_HOST_API", "host_api_name", str),
		("AUDIO_SAMPLE_RATE", "sample_rate", int),
		("AUDIO_BLOCK_SIZE", "block_size", int),
		("AUDIO_CHANNELS", "channels", int),
		("AUDIO_LATENCY", "latency", float),
		("AUDIO_LOOPBACK", "loopback", _as_bool),
		("AUDIO_EXCLUSIVE", "exclusive", _as_bool),
	]
	for env_name, key, caster in env_overrides:
		val = os.environ.get(env_name)
		if val is not None and val != "":
			try:
				audio_cfg_data[key] = caster(val)
			except Exception:
				pass

	cfg = AudioConfig(
		device_substring=audio_cfg_data.get("device_substring"),
		device_index=audio_cfg_data.get("device_index"),
		host_api_name=audio_cfg_data.get("host_api_name", "Windows WASAPI"),
		sample_rate=int(audio_cfg_data.get("sample_rate", 44100)),
		block_size=int(audio_cfg_data.get("block_size", 512)),
		channels=int(audio_cfg_data.get("channels", 2)),
		latency=float(audio_cfg_data.get("latency", 0.02)),
		loopback=bool(audio_cfg_data.get("loopback", False)),
		exclusive=bool(audio_cfg_data.get("exclusive", False)),
		dtype=str(audio_cfg_data.get("dtype", "float32")),
		ringbuffer_blocks=int(audio_cfg_data.get("ringbuffer_blocks", 64)),
	)

	n_fft = int(analysis_cfg.get("n_fft", 1024))
	hop_size = int(analysis_cfg.get("hop_size", n_fft // 2))
	window = str(analysis_cfg.get("window", "hann"))
	bands_cfg = analysis_cfg.get("bands", {"mode": "log", "num_bands": 32})
	smoother_cfg = analysis_cfg.get("smoother", {"attack": 0.6, "release": 0.15})
	norm_cfg = analysis_cfg.get("normalizer", {"mode": "peak", "decay": 0.02, "window": 60, "floor": 1.0e-6})

	# Enrich bands config for log mode
	if isinstance(bands_cfg, dict) and bands_cfg.get("mode", "linear").lower() == "log":
		bands_cfg = dict(bands_cfg)
		bands_cfg.setdefault("sample_rate", cfg.sample_rate)
		bands_cfg.setdefault("n_fft", n_fft)

	smoother = Smoother(attack=float(smoother_cfg.get("attack", 0.6)), release=float(smoother_cfg.get("release", 0.15)))
	normalizer = Normalizer(
		mode=str(norm_cfg.get("mode", "peak")),
		window=int(norm_cfg.get("window", 60)),
		decay=float(norm_cfg.get("decay", 0.02)),
		floor=float(norm_cfg.get("floor", 1.0e-6)),
	)

	def process_block_stream(block_iter):
		buffer: Optional[np.ndarray] = None
		last_time = time.time()
		last_log = 0.0
		silence_frames = 0
		SILENCE_THRESH = 1e-6
		SILENCE_LIMIT_FRAMES = 40  # trigger fallback faster on continuous silence
		for block in block_iter:
			if stop_event.is_set():
				break
			if block is None:
				continue
			# Detect capture silence at the source
			try:
				block_abs_max = float(np.max(np.abs(block)))
			except Exception:
				block_abs_max = 0.0
			if block_abs_max < SILENCE_THRESH:
				silence_frames += 1
				if silence_frames >= SILENCE_LIMIT_FRAMES:
					print("analysis: detected continuous silence in input blocks")
					raise SilenceError("continuous silence from capture")
				# Keep accumulating until threshold reached
				pass
			else:
				silence_frames = 0
			if buffer is None:
				buffer = block
			else:
				if buffer.shape[1] == block.shape[1]:
					buffer = np.concatenate([buffer, block], axis=0)
				else:
					buffer = block
			if buffer.shape[0] >= n_fft:
				mag = compute_spectrum(buffer, window=window, n_fft=n_fft, hop_size=hop_size)
				bands = aggregate_bands(mag, scheme=bands_cfg)
				last = bands[:, -1]
				last = smoother.update(last)
				last = normalizer.update(last)
				shared.set(last)
				# Lightweight debug log once per second
				now = time.time()
				if now - last_log >= 1.0:
					try:
						print(f"analysis: bands min={float(np.min(last)):.3f} max={float(np.max(last)):.3f} len={last.shape[0]}")
					except Exception:
						pass
					last_log = now
				buffer = buffer[-hop_size:, :]
			now = time.time()
			if now - last_time < 0.01:
				time.sleep(0.005)
				last_time = now

	def yield_from_sounddevice():
		with AudioCapture(cfg) as cap:
			while not stop_event.is_set():
				yield cap.read(timeout=0.1)

	def yield_from_soundcard():
		if sc is None:
			return
		# Choose loopback mic: prefer device_substring match; else DDJ; else Realtek; else default speaker loopback
		mic = None
		dev_hint = (cfg.device_substring or "").lower() if cfg.device_substring else None
		candidates = list(sc.all_microphones(include_loopback=True))
		# Prefer loopback-only
		candidates = [m for m in candidates if getattr(m, "isloopback", False)] or candidates
		for m in candidates:
			if dev_hint and dev_hint in m.name.lower():
				mic = m
				break
		if mic is None:
			for m in candidates:
				if "DDJ-FLX4" in m.name:
					mic = m
					break
		if mic is None:
			for s in sc.all_speakers():
				if "Realtek" in s.name:
					mic = sc.get_microphone(s.name, include_loopback=True)
					break
		if mic is None:
			mic = sc.get_microphone(sc.default_speaker().name, include_loopback=True)

		# Candidate samplerates and channels
		sr_candidates = []
		for s in [cfg.sample_rate, 48000, 44100]:
			if s and s not in sr_candidates:
				sr_candidates.append(int(s))
		ch_candidates = []
		mic_ch = getattr(mic, "channels", None)
		if isinstance(mic_ch, int) and mic_ch > 0:
			ch_candidates.append(mic_ch)
		for c in [2, 4, 1]:
			if c not in ch_candidates:
				ch_candidates.append(c)
		blocksize_candidates = [cfg.block_size, max(cfg.block_size, 2048), 1024, 4096]
		# Try combinations until one produces non-silent data
		for sr in sr_candidates:
			for ch in ch_candidates:
				for bs in blocksize_candidates:
					try:
						exclusive_mode = ("DDJ-FLX4" in mic.name)
						with mic.recorder(samplerate=sr, channels=ch, blocksize=bs, exclusive_mode=exclusive_mode) as rec:
							while not stop_event.is_set():
								block = rec.record(numframes=bs)
								block = block.astype(np.float32, copy=False)
								yield block
					except Exception:
						continue
		# If all attempts fail, return
		return

	# Keep trying sources until stop
	while not stop_event.is_set():
		try:
			process_block_stream(yield_from_sounddevice())
			break
		except SilenceError:
			print("analysis: sustained silence on sounddevice; falling back to soundcard...")
			try:
				process_block_stream(yield_from_soundcard())
				break
			except SilenceError:
				print("analysis: sustained silence on soundcard; retrying in 1s...")
				time.sleep(1.0)
				continue
			except Exception as e2:
				print(f"analysis: soundcard error {e2}; retrying in 1s...")
				time.sleep(1.0)
				continue
		except Exception as e:
			print(f"analysis: sounddevice error {e}; trying soundcard...")
			try:
				process_block_stream(yield_from_soundcard())
				break
			except Exception as e2:
				print(f"analysis: soundcard error {e2}; retrying in 1s...")
				time.sleep(1.0)
				continue


# Globals for CLI scene
configs_dir = _PROJECT_ROOT / "configs"
audio_cfg_path = configs_dir / "audio.yaml"
analysis_cfg_path = configs_dir / "analysis.yaml"
visuals_cfg_path = configs_dir / "visuals.yaml"
_shared = SharedBands()
_stop_event = threading.Event()


class SpectrumBarsScene(BaseSpectrumBarsScene):
	"""CLI-exposed scene name that Manim discovers in this file.

	Starts the analysis worker and then runs the base SpectrumBarsScene.
	"""

	def __init__(self, renderer=None, **kwargs):
		# Load visuals configuration if available and pass to base
		viz_cfg = load_yaml(visuals_cfg_path)
		num_bands = int(viz_cfg.get("num_bands", 32))
		min_height = float(viz_cfg.get("min_height", 0.05))
		color_scheme = viz_cfg.get("color_scheme")
		shapes_cfg = viz_cfg.get("shapes")
		scene_scale = float(viz_cfg.get("scene_scale", 1.0))
		baseline_y = float(viz_cfg.get("baseline_y", -3.0))
		scene_width = float(viz_cfg.get("scene_width", 12.0))
		bar_opacity = float(viz_cfg.get("bar_opacity", 0.9))
		# Wire global shared into the base scene with visuals config
		super().__init__(renderer=renderer, shared=_shared, num_bands=num_bands, min_height=min_height, color_scheme=color_scheme, shapes_config=shapes_cfg, scene_scale=scene_scale, baseline_y=baseline_y, scene_width=scene_width, bar_opacity=bar_opacity, fractal_config=viz_cfg.get("fractal"), **kwargs)
		self._worker_started = False

	def construct(self):
		# Start worker once
		if not self._worker_started:
			self._worker_started = True
			self._worker = threading.Thread(target=analysis_worker, args=(_shared, audio_cfg_path, analysis_cfg_path, _stop_event), daemon=True)
			self._worker.start()
		# Proceed with base construct to build bars/updaters
		super().construct()


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Run realtime spectrum bars visual with audio capture")
	p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
	p.add_argument("--host-api", default=None, help="Filter devices by host API name (e.g., Windows WASAPI)")
	p.add_argument("--device-index", type=int, default=None, help="Device index to use for capture")
	p.add_argument("--device", default=None, help="Substring match for device name (e.g., Głośniki or DDJ-FLX4)")
	p.add_argument("--loopback", action="store_true", help="Capture output loopback")
	p.add_argument("--no-loopback", action="store_true", help="Disable loopback even if config enables it")
	p.add_argument("--sample-rate", type=int, default=None)
	p.add_argument("--channels", type=int, default=None)
	p.add_argument("--block-size", type=int, default=None)
	return p


def main(argv: list[str] | None = None):
	args = build_parser().parse_args(argv)
	if args.list_devices:
		cfg_data = load_yaml(audio_cfg_path)
		host_api = args.host_api or cfg_data.get("host_api_name", "Windows WASAPI")
		devs = list_devices(host_api_name=host_api)
		for d in devs:
			print(f"[{d['index']:>3}] {d['name']} | {d['hostapi']} | in:{d['max_input_channels']} out:{d['max_output_channels']} | default_sr:{d['default_samplerate']}")
		return
	# Apply one-off overrides via environment variables consumed by analysis_worker
	if args.host_api:
		os.environ["AUDIO_HOST_API"] = str(args.host_api)
	if args.device_index is not None:
		os.environ["AUDIO_DEVICE_INDEX"] = str(args.device_index)
	if args.device is not None:
		os.environ["AUDIO_DEVICE"] = str(args.device)
	if args.loopback and not args.no_loopback:
		os.environ["AUDIO_LOOPBACK"] = "1"
	if args.no_loopback:
		os.environ["AUDIO_LOOPBACK"] = "0"
	if args.sample_rate is not None:
		os.environ["AUDIO_SAMPLE_RATE"] = str(args.sample_rate)
	if args.channels is not None:
		os.environ["AUDIO_CHANNELS"] = str(args.channels)
	if args.block_size is not None:
		os.environ["AUDIO_BLOCK_SIZE"] = str(args.block_size)

	# Launch worker and render once
	thread = threading.Thread(target=analysis_worker, args=(_shared, audio_cfg_path, analysis_cfg_path, _stop_event), daemon=True)
	thread.start()
	try:
		# Configure interactive preview and disable movie writing
		try:
			manim_config.write_to_movie = False
			manim_config.preview = True
			manim_config.disable_caching = True
		except Exception:
			pass
		# Load visuals config for direct invocation too
		viz_cfg = load_yaml(visuals_cfg_path)
		# Honor renderer from visuals config if provided
		try:
			renderer_choice = str(viz_cfg.get("renderer", "opengl")).lower()
			if renderer_choice in ("opengl", "cairo"):
				manim_config.renderer = renderer_choice
				print(f"visuals: using renderer={renderer_choice}")
		except Exception:
			pass
		num_bands = int(viz_cfg.get("num_bands", 32))
		min_height = float(viz_cfg.get("min_height", 0.05))
		color_scheme = viz_cfg.get("color_scheme")
		shapes_cfg = viz_cfg.get("shapes")
		scene_scale = float(viz_cfg.get("scene_scale", 1.0))
		baseline_y = float(viz_cfg.get("baseline_y", -3.0))
		scene_width = float(viz_cfg.get("scene_width", 12.0))
		bar_opacity = float(viz_cfg.get("bar_opacity", 0.9))
		scene = BaseSpectrumBarsScene(shared=_shared, num_bands=num_bands, min_height=min_height, color_scheme=color_scheme, shapes_config=shapes_cfg, scene_scale=scene_scale, baseline_y=baseline_y, scene_width=scene_width, bar_opacity=bar_opacity, fractal_config=viz_cfg.get("fractal"))
		scene.render()
	except SystemExit:
		pass
	finally:
		_stop_event.set()
		thread.join(timeout=2.0)


if __name__ == "__main__":
	main()


