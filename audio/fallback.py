from __future__ import annotations

import time
import threading
from queue import Queue, Full
from typing import Optional

import numpy as np
import warnings

try:
    import soundcard as sc
except Exception:
    sc = None
else:
    # Suppress benign discontinuity warnings from Media Foundation backend
    try:
        warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)
    except Exception:
        warnings.filterwarnings("ignore", message="data discontinuity in recording")


class SoundcardLoopbackCapture:
    def __init__(self, samplerate: int, channels: int, blocksize: int, device_name_hint: Optional[str] = None) -> None:
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.blocksize = int(blocksize)
        self.device_name_hint = device_name_hint
        self._queue: Queue[np.ndarray] = Queue(maxsize=64)
        self._running = False
        self._recorder = None
        self._selected_sr: Optional[int] = None
        self._selected_ch: Optional[int] = None
        self._frames = 0
        self._overruns = 0

    @property
    def metrics(self) -> dict:
        return {
            "queue_size": self._queue.qsize(),
            "overruns": self._overruns,
            "callback_late": 0,
            "frames_captured": self._frames,
        }

    def start(self) -> None:
        if sc is None:
            raise RuntimeError("soundcard module not available for fallback loopback")

        mic = None
        try:
            # Prefer explicit device hint if provided
            if self.device_name_hint:
                # Try exact/substring match on speaker names, then fallback to default
                for s in sc.all_speakers():
                    if self.device_name_hint.lower() in s.name.lower():
                        mic = sc.get_microphone(s.name, include_loopback=True)
                        break
            # Otherwise try known stable vendors like Realtek
            if mic is None:
                for s in sc.all_speakers():
                    if "Realtek" in s.name:
                        mic = sc.get_microphone(s.name, include_loopback=True)
                        break
        except Exception:
            mic = None
        if mic is None:
            mic = sc.get_microphone(sc.default_speaker().name, include_loopback=True)

        sr_candidates = []
        for sr in [self.samplerate, 48000, 44100]:
            if sr and int(sr) not in sr_candidates:
                sr_candidates.append(int(sr))
        ch_candidates = []
        mic_ch = getattr(mic, "channels", None)
        for ch in [self.channels, mic_ch, 2, 1]:
            if isinstance(ch, int) and ch > 0 and ch not in ch_candidates:
                ch_candidates.append(int(ch))

        last_err = None
        for sr in sr_candidates:
            for ch in ch_candidates:
                try:
                    rec = mic.recorder(
                        samplerate=sr,
                        channels=ch,
                        blocksize=self.blocksize,
                        exclusive_mode=("DDJ-FLX4" in mic.name),
                    )
                    rec.__enter__()
                    _ = rec.record(numframes=self.blocksize)
                    self._recorder = rec
                    self._selected_sr = int(sr)
                    self._selected_ch = int(ch)
                    self._running = True

                    def _reader_loop():
                        while self._running and self._recorder is not None:
                            try:
                                block = self._recorder.record(numframes=self.blocksize)
                                block = block.astype(np.float32)
                                if block.ndim == 1:
                                    block = block[:, np.newaxis]
                                try:
                                    self._queue.put_nowait(block)
                                except Full:
                                    try:
                                        _ = self._queue.get_nowait()
                                    except Exception:
                                        pass
                                    try:
                                        self._queue.put_nowait(block)
                                    except Exception:
                                        self._overruns += 1
                                        pass
                                self._frames += block.shape[0]
                            except Exception:
                                time.sleep(0.002)
                                continue

                    t = threading.Thread(target=_reader_loop, daemon=True)
                    t.start()
                    return
                except Exception as e:
                    last_err = e
                    try:
                        if rec is not None:
                            rec.__exit__(None, None, None)
                    except Exception:
                        pass
                    self._recorder = None
                    continue

        raise last_err if last_err else RuntimeError("Failed to start soundcard loopback")

    def stop(self) -> None:
        self._running = False
        try:
            if self._recorder is not None:
                self._recorder.__exit__(None, None, None)
        except Exception:
            pass
        self._recorder = None
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass

    def read(self, timeout: float | None = None):
        if timeout is None:
            try:
                return self._queue.get_nowait()
            except Exception:
                return None
        if timeout <= 0.0:
            try:
                return self._queue.get_nowait()
            except Exception:
                return None
        try:
            return self._queue.get(timeout=timeout)
        except Exception:
            return None


