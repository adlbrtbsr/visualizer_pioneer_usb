import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import wave
import yaml
import contextlib

try:
	import soundcard as sc
except Exception:
	sc = None

# Ensure project root is importable when running this script directly
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))

from audio import AudioCapture, AudioConfig, list_devices


def load_defaults(config_path: Path) -> dict:
	if config_path.is_file():
		with config_path.open("r", encoding="utf-8") as f:
			data = yaml.safe_load(f) or {}
			return data
	return {}


def do_list(host_api: str) -> int:
	devices = list_devices(host_api_name=host_api)
	for d in devices:
		print(f"[{d['index']:>3}] {d['name']} | {d['hostapi']} | in:{d['max_input_channels']} out:{d['max_output_channels']} | default_sr:{d['default_samplerate']}")
	return 0


def do_probe(cfg: AudioConfig, seconds: float, wav: str | None) -> int:
	write_wav_path = Path(wav) if wav else None
	start_time = time.time()
	frames_collected = 0
	last_meter_time = 0.0
	meters_interval = 0.1
	wav_writer = None

	if write_wav_path is not None:
		wav_writer = wave.open(str(write_wav_path), "wb")
		wav_writer.setnchannels(cfg.channels)
		wav_writer.setsampwidth(2)
		wav_writer.setframerate(cfg.sample_rate)

	def write_wav(block_float32: np.ndarray):
		int16 = np.clip(block_float32, -1.0, 1.0)
		int16 = (int16 * 32767.0).astype(np.int16)
		wav_writer.writeframes(int16.tobytes())

	try:
		with AudioCapture(cfg) as cap:
			for block in cap.frames():
				frames_collected += block.shape[0]
				if wav_writer is not None:
					write_wav(block)
				now = time.time()
				if now - last_meter_time >= meters_interval:
					rms, peak = AudioCapture.compute_rms_and_peak(block)
					meter = " ".join(
						f"ch{ch+1}: RMS {rms[ch]:.3f} PEAK {peak[ch]:.3f}" for ch in range(block.shape[1])
					)
					print(f"t={now - start_time:5.1f}s  {meter}  queue={cap.metrics['queue_size']} overruns={cap.metrics['overruns']} late={cap.metrics['callback_late']}")
					last_meter_time = now
				if seconds and (now - start_time) >= seconds:
					break
		print(f"Done. Frames captured: {frames_collected}")
		return 0
	except Exception as e:
		print(f"Error: {e}", file=sys.stderr)
		if cfg.loopback and sc is not None:
			print("Falling back to soundcard loopback...")
			return do_probe_soundcard(seconds, wav)
		elif (not cfg.loopback) and sc is not None:
			print("Falling back to soundcard input mic...")
			return do_probe_soundcard_input(device_hint=cfg.device_substring, seconds=seconds, wav=wav, samplerate=cfg.sample_rate, channels=cfg.channels)
		return 2
	finally:
		if wav_writer is not None:
			wav_writer.close()


def do_probe_soundcard(seconds: float, wav: str | None) -> int:
	if sc is None:
		print("soundcard not available", file=sys.stderr)
		return 2
	# Try DDJ-FLX4 loopback first; if it fails to open, fall back to default speaker loopback
	ddj_mic = None
	for m in sc.all_microphones(include_loopback=True):
		if "DDJ-FLX4" in m.name and m.isloopback:
			ddj_mic = m
			break
	# Prepare specific Realtek loopback if available
	realtek_mic = None
	for s in sc.all_speakers():
		if "Realtek" in s.name:
			realtek_mic = sc.get_microphone(s.name, include_loopback=True)
			break
	def_mic = sc.get_microphone(sc.default_speaker().name, include_loopback=True)

	def try_record(microphone):
		print(f"soundcard: capturing loopback from: {microphone.name}")
		blocksize = 1024
		# Try common samplerates
		samplerate_candidates = [48000, 44100]
		# Try device channel count first if available, then DDJ 4ch, then stereo
		mic_ch = getattr(microphone, "channels", None)
		channels_candidates = []
		if isinstance(mic_ch, int) and mic_ch > 0:
			channels_candidates.append(mic_ch)
		if "DDJ-FLX4" in microphone.name:
			channels_candidates.extend([4, 2])
		else:
			channels_candidates.append(2)
		# Deduplicate while preserving order
		seen = set()
		channels_candidates = [c for c in channels_candidates if not (c in seen or seen.add(c))]
		last_err = None
		for sr in samplerate_candidates:
			for channels in channels_candidates:
				start_time = time.time()
				frames_collected = 0
				last_meter_time = 0.0
				meters_interval = 0.1
				wav_writer = None
				try:
					if wav:
						wav_writer = wave.open(str(wav), "wb")
						wav_writer.setnchannels(channels)
						wav_writer.setsampwidth(2)
						wav_writer.setframerate(sr)
					with microphone.recorder(samplerate=sr, channels=channels, blocksize=blocksize, exclusive_mode=("DDJ-FLX4" in microphone.name)) as rec:
						for _ in range(int(seconds * sr / blocksize) + 1):
							block = rec.record(numframes=blocksize)
							block = block.astype(np.float32)
							frames_collected += block.shape[0]
							if wav_writer is not None:
								int16 = np.clip(block, -1.0, 1.0)
								int16 = (int16 * 32767.0).astype(np.int16)
								wav_writer.writeframes(int16.tobytes())
							now = time.time()
							if now - last_meter_time >= meters_interval:
								rms, peak = AudioCapture.compute_rms_and_peak(block)
								meter = " ".join(
									f"ch{ch+1}: RMS {rms[ch]:.3f} PEAK {peak[ch]:.3f}" for ch in range(block.shape[1])
								)
								print(f"t={now - start_time:5.1f}s  {meter}")
								last_meter_time = now
					if wav_writer is not None:
						wav_writer.close()
					print(f"Done (soundcard). Frames captured: {frames_collected}")
					return 0
				except Exception as e:
					last_err = e
					if wav_writer is not None:
						with contextlib.suppress(Exception):
							wav_writer.close()
					continue
		# If all attempts failed, re-raise last error for outer handler
		if last_err:
			raise last_err

	try:
		if ddj_mic is not None:
			return try_record(ddj_mic)
		elif realtek_mic is not None:
			return try_record(realtek_mic)
		else:
			return try_record(def_mic)
	except Exception as e:
		print(f"soundcard DDJ/default loopback failed: {e}")
		try:
			if realtek_mic is not None:
				return try_record(realtek_mic)
			return try_record(def_mic)
		except Exception as e2:
			print(f"soundcard default loopback failed: {e2}", file=sys.stderr)
			return 2


def do_probe_soundcard_input(device_hint: str | None, seconds: float, wav: str | None, samplerate: int, channels: int) -> int:
	if sc is None:
		print("soundcard not available", file=sys.stderr)
		return 2
	# Find non-loopback microphone matching hint (e.g., DDJ-FLX4)
	mic = None
	for m in sc.all_microphones(include_loopback=False):
		if device_hint and device_hint.lower() in m.name.lower():
			mic = m
			break
	if mic is None:
		for m in sc.all_microphones(include_loopback=False):
			if "DDJ-FLX4" in m.name:
				mic = m
				break
	if mic is None:
		print("No matching input microphone found for hint; aborting.", file=sys.stderr)
		return 2
	print(f"soundcard: capturing input from mic: {mic.name}")
	blocksize = 1024
	start_time = time.time()
	frames_collected = 0
	last_meter_time = 0.0
	meters_interval = 0.1
	wav_writer = None
	if wav:
		wav_writer = wave.open(str(wav), "wb")
		wav_writer.setnchannels(channels)
		wav_writer.setsampwidth(2)
		wav_writer.setframerate(samplerate)
	with mic.recorder(samplerate=samplerate, channels=channels, blocksize=blocksize) as rec:
		for _ in range(int(seconds * samplerate / blocksize) + 1):
			block = rec.record(numframes=blocksize)
			block = block.astype(np.float32)
			frames_collected += block.shape[0]
			if wav_writer is not None:
				int16 = np.clip(block, -1.0, 1.0)
				int16 = (int16 * 32767.0).astype(np.int16)
				wav_writer.writeframes(int16.tobytes())
			now = time.time()
			if now - last_meter_time >= meters_interval:
				rms, peak = AudioCapture.compute_rms_and_peak(block)
				meter = " ".join(
					f"ch{ch+1}: RMS {rms[ch]:.3f} PEAK {peak[ch]:.3f}" for ch in range(block.shape[1])
				)
				print(f"t={now - start_time:5.1f}s  {meter}")
				last_meter_time = now
	if wav_writer is not None:
		wav_writer.close()
	print(f"Done (soundcard input). Frames captured: {frames_collected}")
	return 0


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Probe audio capture devices and levels")
	parser.add_argument("--list", action="store_true", help="List audio devices and exit")
	parser.add_argument("--config", default="configs/audio.yaml")
	parser.add_argument("--host-api", default=None)
	parser.add_argument("--device", default=None)
	parser.add_argument("--device-index", type=int, default=None)
	parser.add_argument("--sample-rate", type=int, default=None)
	parser.add_argument("--block-size", type=int, default=None)
	parser.add_argument("--channels", type=int, default=None)
	parser.add_argument("--latency", type=float, default=None)
	parser.add_argument("--loopback", action="store_true")
	parser.add_argument("--exclusive", action="store_true")
	parser.add_argument("--seconds", type=float, default=5.0)
	parser.add_argument("--wav", default=None)
	return parser


def main(argv=None) -> int:
	args = build_parser().parse_args(argv)
	config_path = Path(args.config)
	defaults = load_defaults(config_path)
	# Host API for listing
	host_api = args.host_api or defaults.get("host_api_name", "Windows WASAPI")
	if args.list:
		return do_list(host_api)

	cfg = AudioConfig(
		device_substring=args.device if args.device is not None else defaults.get("device_substring"),
		device_index=args.device_index if args.device_index is not None else None,
		host_api_name=host_api,
		sample_rate=args.sample_rate if args.sample_rate is not None else int(defaults.get("sample_rate", 44100)),
		block_size=args.block_size if args.block_size is not None else int(defaults.get("block_size", 512)),
		channels=args.channels if args.channels is not None else int(defaults.get("channels", 2)),
		latency=args.latency if args.latency is not None else float(defaults.get("latency", 0.02)),
		loopback=bool(args.loopback) if args.loopback else bool(defaults.get("loopback", False)),
		exclusive=bool(args.exclusive) if args.exclusive else bool(defaults.get("exclusive", False)),
		dtype=str(defaults.get("dtype", "float32")),
		ringbuffer_blocks=int(defaults.get("ringbuffer_blocks", 64)),
	)

	return do_probe(cfg, seconds=args.seconds, wav=args.wav)


if __name__ == "__main__":
	sys.exit(main())


