from __future__ import annotations

import asyncio
import sys
import threading
import time
from dataclasses import dataclass
from queue import Queue, Full, Empty
from typing import Generator, Iterable, List, Optional, Tuple

import numpy as np
import sounddevice as sd


@dataclass
class AudioConfig:
    device_substring: Optional[str] = None
    device_index: Optional[int] = None
    host_api_name: str = "Windows WASAPI"
    sample_rate: int = 44100
    block_size: int = 512
    channels: int = 2
    latency: float = 0.02
    loopback: bool = False
    exclusive: bool = False
    dtype: str = "float32"
    ringbuffer_blocks: int = 64


def _get_hostapi_index_by_name(name: str) -> Optional[int]:
    hostapis = sd.query_hostapis()
    for idx, info in enumerate(hostapis):
        if info.get("name", "").lower() == name.lower():
            return idx
    # Fallback: try containment match
    for idx, info in enumerate(hostapis):
        if name.lower() in info.get("name", "").lower():
            return idx
    return None


def list_devices(host_api_name: Optional[str] = None) -> List[dict]:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    results: List[dict] = []
    for idx, dev in enumerate(devices):
        ha = hostapis[dev["hostapi"]]["name"]
        if host_api_name and host_api_name.lower() not in ha.lower():
            continue
        results.append(
            {
                "index": idx,
                "name": dev["name"],
                "hostapi": ha,
                "max_input_channels": dev.get("max_input_channels", 0),
                "max_output_channels": dev.get("max_output_channels", 0),
                "default_samplerate": int(dev.get("default_samplerate", 0) or 0),
            }
        )
    return results


def find_device_index(
    cfg: AudioConfig,
) -> Tuple[Optional[int], Optional[int]]:
    hostapi_index = _get_hostapi_index_by_name(cfg.host_api_name) if cfg.host_api_name else None
    best_device_index: Optional[int] = None

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    # If explicit index provided, validate and return it
    if cfg.device_index is not None:
        try:
            dev = devices[cfg.device_index]
            ha_idx = dev["hostapi"]
            if hostapi_index is not None and ha_idx != hostapi_index:
                raise ValueError("Device host API does not match requested host API")
            if cfg.loopback and dev.get("max_output_channels", 0) <= 0:
                raise ValueError("Selected device has no output channels for loopback")
            if not cfg.loopback and dev.get("max_input_channels", 0) <= 0:
                raise ValueError("Selected device has no input channels")
            return hostapi_index, cfg.device_index
        except Exception:
            pass

    for idx, dev in enumerate(devices):
        ha_idx = dev["hostapi"]
        ha_name = hostapis[ha_idx]["name"]
        if hostapi_index is not None and ha_idx != hostapi_index:
            continue

        # For loopback capture we need an output-capable device; otherwise input-capable
        if cfg.loopback:
            if dev.get("max_output_channels", 0) <= 0:
                continue
        else:
            if dev.get("max_input_channels", 0) <= 0:
                continue

        if cfg.device_substring:
            if cfg.device_substring.lower() not in dev["name"].lower():
                continue

        best_device_index = idx
        break

    return hostapi_index, best_device_index


class AudioCapture:
    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._queue: Queue[np.ndarray] = Queue(maxsize=config.ringbuffer_blocks)
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._lock = threading.Lock()
        self._overruns = 0
        self._callback_late = 0
        self._frames_captured = 0
        self._device_info: Optional[dict] = None
        self._selected_device_index: Optional[int] = None
        self._selected_channels: Optional[int] = None
        self._selected_samplerate: Optional[int] = None

    @property
    def metrics(self) -> dict:
        return {
            "queue_size": self._queue.qsize(),
            "overruns": self._overruns,
            "callback_late": self._callback_late,
            "frames_captured": self._frames_captured,
        }

    @staticmethod
    def compute_rms_and_peak(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if block.size == 0:
            return np.zeros((block.shape[1],), dtype=np.float32), np.zeros((block.shape[1],), dtype=np.float32)
        # Expect shape (frames, channels)
        # RMS per channel
        rms = np.sqrt(np.mean(block.astype(np.float32) ** 2, axis=0))
        peak = np.max(np.abs(block.astype(np.float32)), axis=0)
        return rms, peak

    def _callback(self, indata, frames, time_info, status):
        if status.input_underflow or status.input_overflow or status:
            # status may include other flags; record metrics
            self._callback_late += 1
        # Ensure correct dtype and shape
        block = np.asarray(indata, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, np.newaxis]
        try:
            self._queue.put_nowait(block.copy())
        except Full:
            self._overruns += 1
        self._frames_captured += frames

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            _, device_index = find_device_index(self.config)
            if device_index is None and self.config.device_substring:
                # Fall back to default device if substring not found
                print(
                    (
                        "Audio: device containing '"
                        + str(self.config.device_substring)
                        + "' not found for host API '"
                        + str(self.config.host_api_name)
                        + "'. Falling back to default device."
                    ),
                    file=sys.stderr,
                )
            if device_index is None:
                # Fallback to system defaults
                device_index = None

            extra = None
            if self.config.loopback:
                # Configure WASAPI-specific settings only for loopback
                try:
                    ws = sd.WasapiSettings()
                    if hasattr(ws, "loopback"):
                        setattr(ws, "loopback", bool(self.config.loopback))
                    if hasattr(ws, "exclusive"):
                        setattr(ws, "exclusive", bool(self.config.exclusive))
                    extra = ws
                except Exception:
                    extra = None

            # If loopback, prefer device default samplerate; for channels, probe candidates instead of forcing
            samplerate = self.config.sample_rate
            requested_channels = self.config.channels
            sel_info = sd.query_devices(device_index) if device_index is not None else None
            available_out_channels = 0
            available_in_channels = 0
            if sel_info is not None:
                if isinstance(sel_info.get("default_samplerate"), (int, float)) and sel_info["default_samplerate"]:
                    samplerate = int(sel_info["default_samplerate"]) or samplerate
                available_out_channels = int(sel_info.get("max_output_channels", 0) or 0)
                available_in_channels = int(sel_info.get("max_input_channels", 0) or 0)
            # For non-loopback inputs, clamp to device capability
            if not self.config.loopback and available_in_channels > 0:
                requested_channels = min(requested_channels, available_in_channels)

            # Try to open stream; for loopback, probe a few likely channel counts
            def _open_stream(ch: int):
                # For WASAPI loopback, use the OUTPUT device index directly with WasapiSettings(loopback=True)
                return sd.InputStream(
                    device=device_index,
                    samplerate=samplerate,
                    channels=ch,
                    dtype=self.config.dtype,
                    blocksize=self.config.block_size,
                    latency=self.config.latency,
                    extra_settings=extra,
                    callback=self._callback,
                )

            open_ok = False
            last_exc: Optional[Exception] = None
            # Build channel candidates
            if self.config.loopback:
                # For WASAPI loopback, commonly stereo; try stereo then mono regardless of reported caps
                channel_candidates = []
                for c in [2, 1]:
                    if c not in channel_candidates:
                        channel_candidates.append(c)
            else:
                channel_candidates = []
                if requested_channels and requested_channels not in channel_candidates:
                    channel_candidates.append(requested_channels)
                if available_in_channels > 0 and available_in_channels not in channel_candidates:
                    channel_candidates.append(available_in_channels)
                # Common practical options
                for c in [2, 1, 4, 6, 8]:
                    if c not in channel_candidates:
                        channel_candidates.append(c)
            for ch in channel_candidates:
                try:
                    stream = _open_stream(ch)
                    stream.start()
                    # Success
                    self._stream = stream
                    requested_channels = ch
                    open_ok = True
                    break
                except Exception as e:
                    last_exc = e
                    # If host API specific info is incompatible, retry without extra_settings
                    if isinstance(e, Exception) and "Incompatible host API specific stream info" in str(e):
                        try:
                            stream = sd.InputStream(
                                device=device_index,
                                samplerate=samplerate,
                                channels=ch,
                                dtype=self.config.dtype,
                                blocksize=self.config.block_size,
                                latency=self.config.latency,
                                callback=self._callback,
                            )
                            stream.start()
                            self._stream = stream
                            requested_channels = ch
                            open_ok = True
                            break
                        except Exception as e2:
                            last_exc = e2
                            continue
                    continue
            if not open_ok:
                self._stream = None
                if last_exc:
                    raise last_exc
                raise RuntimeError("Failed to open audio input stream for loopback.")
            self._running = True
            # Set default device to help some backends
            if device_index is not None:
                try:
                    sd.default.device = (device_index, sd.default.device[1])
                except Exception:
                    pass
            self._selected_device_index = device_index
            self._selected_channels = requested_channels
            self._selected_samplerate = samplerate
            if device_index is not None:
                self._device_info = sd.query_devices(device_index)
            else:
                # None means default device; get default input device index
                default_in = sd.default.device[0]
                if default_in is not None:
                    self._device_info = sd.query_devices(default_in)

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            finally:
                self._stream = None
                self._running = False
                # Drain queue to unblock any consumers
                try:
                    while True:
                        self._queue.get_nowait()
                except Empty:
                    pass

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def frames(self) -> Iterable[np.ndarray]:
        while self._running:
            block = self.read(timeout=0.5)
            if block is None:
                continue
            yield block

    async def aiter_frames(self) -> Iterable[np.ndarray]:
        loop = asyncio.get_running_loop()
        while self._running:
            block = await loop.run_in_executor(None, self.read, 0.5)
            if block is None:
                continue
            yield block

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


