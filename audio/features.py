from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .analysis import (
    Smoother,
    Normalizer,
    BeatDetector,
    compute_spectrum,
    aggregate_bands,
    _freq_edges_for_log_spacing,
    _bin_edges_from_freq_edges,
)


try:
    import librosa  # type: ignore
except Exception:
    librosa = None  # graceful fallback to numpy-based features


@dataclass
class FeatureExtractorConfig:
    sample_rate: int
    n_fft: int = 1024
    hop_size: int = 512
    # Feature set sizes
    mel_bands: int = 24
    use_chroma: bool = True
    # Smoothing and normalization
    mel_smoother_attack: float = 0.6
    mel_smoother_release: float = 0.35
    chroma_smoother_attack: float = 0.6
    chroma_smoother_release: float = 0.35
    scalar_smoother_attack: float = 0.6
    scalar_smoother_release: float = 0.35
    normalizer_mode: str = "percentile"  # 'peak' | 'percentile'
    normalizer_window: int = 120
    normalizer_decay: float = 0.05
    normalizer_floor: float = 1.0e-3
    beat_threshold: float = 1.4
    beat_short_window: int = 5
    beat_long_window: int = 43
    beat_refractory_frames: int = 12


class FeatureExtractor:
    """Realtime feature extraction for audio-reactive visuals.

    Produces per-block features as a dict with keys:
      - 'mel': np.ndarray (mel_bands,)
      - 'mel_phase': np.ndarray (mel_bands,) instantaneous phases in radians (-pi..pi)
      - 'chroma': np.ndarray (12,) if enabled, else zeros
      - 'flux': float in [0,1]
      - 'rms': float in [0,1]
      - 'centroid_hz': float (not normalized)
      - 'beat': bool
    Values except centroid are smoothed and normalized to [0,1].
    """

    def __init__(self, cfg: FeatureExtractorConfig) -> None:
        self.cfg = cfg

        # Vector smoothers
        self._mel_smoother = Smoother(
            attack=cfg.mel_smoother_attack,
            release=cfg.mel_smoother_release,
        )
        self._chroma_smoother = Smoother(
            attack=cfg.chroma_smoother_attack,
            release=cfg.chroma_smoother_release,
        )
        # Scalar smoother for flux / rms
        self._scalar_smoother = Smoother(
            attack=cfg.scalar_smoother_attack,
            release=cfg.scalar_smoother_release,
        )

        # Normalizers
        self._mel_norm = Normalizer(
            mode=cfg.normalizer_mode,
            window=cfg.normalizer_window,
            decay=cfg.normalizer_decay,
            floor=cfg.normalizer_floor,
        )
        self._chroma_norm = Normalizer(
            mode=cfg.normalizer_mode,
            window=cfg.normalizer_window,
            decay=cfg.normalizer_decay,
            floor=cfg.normalizer_floor,
        )
        self._scalar_norm = Normalizer(
            mode=cfg.normalizer_mode,
            window=cfg.normalizer_window,
            decay=cfg.normalizer_decay,
            floor=cfg.normalizer_floor,
        )

        self._beat = BeatDetector(
            threshold=cfg.beat_threshold,
            short_window=cfg.beat_short_window,
            long_window=cfg.beat_long_window,
            refractory_frames=cfg.beat_refractory_frames,
        )

        # Previous vectors for flux
        self._prev_mel: Optional[np.ndarray] = None

        # Precompute librosa filters if available
        self._mel_filter: Optional[np.ndarray] = None
        self._chroma_filterbank_ready = False
        if librosa is not None:
            try:
                self._mel_filter = librosa.filters.mel(
                    sr=cfg.sample_rate,
                    n_fft=cfg.n_fft,
                    n_mels=cfg.mel_bands,
                    fmin=20.0,
                    fmax=cfg.sample_rate / 2.0,
                ).astype(np.float32)
                self._chroma_filterbank_ready = True
            except Exception:
                self._mel_filter = None
                self._chroma_filterbank_ready = False

    def _compute_basic_spectrum(self, mono: np.ndarray) -> np.ndarray:
        mag = compute_spectrum(mono, window="hann", n_fft=self.cfg.n_fft, hop_size=self.cfg.n_fft)
        return mag  # (bins, frames)

    def _mel_from_mag(self, mag: np.ndarray) -> np.ndarray:
        # Take last frame
        last = mag[:, -1]
        if self._mel_filter is not None:
            mel = self._mel_filter @ last
        else:
            # Fallback: log-spaced aggregation approximating mel spacing
            mel = aggregate_bands(
                mag,
                {
                    "mode": "log",
                    "num_bands": int(self.cfg.mel_bands),
                    "sample_rate": int(self.cfg.sample_rate),
                    "n_fft": int(self.cfg.n_fft),
                    "min_freq": 20.0,
                    "max_freq": float(self.cfg.sample_rate / 2.0),
                },
            )[:, -1]
        mel = np.asarray(mel, dtype=np.float32)
        mel = np.maximum(mel, 0.0)
        return mel

    def _chroma_from_block(self, mono: np.ndarray) -> np.ndarray:
        if not self.cfg.use_chroma:
            return np.zeros((12,), dtype=np.float32)
        if librosa is None:
            return np.zeros((12,), dtype=np.float32)
        try:
            chroma = librosa.feature.chroma_stft(
                y=mono.astype(np.float32, copy=False),
                sr=self.cfg.sample_rate,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.n_fft,
                center=False,
            )  # shape (12, frames)
            vec = chroma[:, -1].astype(np.float32)
            vec = np.maximum(vec, 0.0)
            return vec
        except Exception:
            return np.zeros((12,), dtype=np.float32)

    def _mel_phase_from_last_frame(self, mono: np.ndarray) -> np.ndarray:
        n = int(self.cfg.n_fft)
        if mono.shape[0] < n:
            pad = n - mono.shape[0]
            mono = np.pad(mono, (pad, 0), mode="constant")
        frame = mono[-n:].astype(np.float32, copy=False)
        win = np.hanning(n).astype(np.float32)
        frame_w = frame * win
        X = np.fft.rfft(frame_w, n=n)
        # Project complex bins into mel bands
        if self._mel_filter is not None:
            mel_complex = (self._mel_filter.astype(np.float32) @ X).astype(np.complex64)
        else:
            # Fallback: log-spaced bin aggregation
            freq_edges = _freq_edges_for_log_spacing(int(self.cfg.mel_bands), int(self.cfg.sample_rate), 20.0, float(self.cfg.sample_rate) / 2.0)
            edges = _bin_edges_from_freq_edges(freq_edges, n_fft=int(self.cfg.n_fft), sample_rate=int(self.cfg.sample_rate))
            mel_complex = np.zeros((int(self.cfg.mel_bands),), dtype=np.complex64)
            for i in range(int(self.cfg.mel_bands)):
                start = int(edges[i])
                end = int(edges[i + 1])
                if end <= start:
                    end = min(start + 1, (n // 2))
                seg = X[start : end + 1]
                mel_complex[i] = np.sum(seg).astype(np.complex64)
        # Avoid zeros to keep angle stable
        eps = 1.0e-12
        real = np.where(np.abs(mel_complex.real) < eps, eps, mel_complex.real)
        imag = np.where(np.abs(mel_complex.imag) < eps, eps, mel_complex.imag)
        mel_complex = real + 1j * imag
        phases = np.angle(mel_complex).astype(np.float32)
        return phases

    def _centroid_from_mag(self, mag: np.ndarray) -> float:
        # Last frame
        spec = mag[:, -1]
        freqs = np.fft.rfftfreq(self.cfg.n_fft, d=1.0 / float(self.cfg.sample_rate))
        denom = float(np.sum(spec))
        if denom <= 1.0e-12:
            return 0.0
        return float(np.sum(freqs * spec) / denom)

    def update(self, block: np.ndarray) -> Dict[str, object]:
        # Ensure mono float32
        mono = block.astype(np.float32, copy=False)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)

        mag = self._compute_basic_spectrum(mono)
        mel_raw = self._mel_from_mag(mag)
        chroma_raw = self._chroma_from_block(mono)
        centroid_hz = self._centroid_from_mag(mag)
        mel_phase = self._mel_phase_from_last_frame(mono)

        # Normalize and smooth vectors
        mel_norm = self._mel_norm.update(mel_raw)
        mel_sm = self._mel_smoother.update(mel_norm)

        if chroma_raw.shape[0] == 12:
            chroma_norm = self._chroma_norm.update(chroma_raw)
            chroma_sm = self._chroma_smoother.update(chroma_norm)
        else:
            chroma_sm = np.zeros((12,), dtype=np.float32)

        # Flux from mel change
        if self._prev_mel is None:
            flux_val = 0.0
        else:
            diff = np.maximum(mel_sm - self._prev_mel, 0.0)
            flux_val = float(np.sqrt(np.sum(diff * diff)) / math.sqrt(max(len(diff), 1)))
        self._prev_mel = mel_sm.copy()
        flux_norm = self._scalar_norm.update(np.array([flux_val], dtype=np.float32))[0]
        flux_sm = float(self._scalar_smoother.update(flux_norm))

        # RMS
        rms_val = float(np.sqrt(np.mean(mono * mono)))
        rms_norm = self._scalar_norm.update(np.array([rms_val], dtype=np.float32))[0]
        rms_sm = float(self._scalar_smoother.update(rms_norm))

        # Beat from short mel energy
        beat_flag = bool(self._beat.update(mel_sm))

        return {
            "mel": mel_sm.astype(np.float32),
            "mel_phase": mel_phase.astype(np.float32),
            "chroma": chroma_sm.astype(np.float32),
            "flux": float(np.clip(flux_sm, 0.0, 1.0)),
            "rms": float(np.clip(rms_sm, 0.0, 1.0)),
            "centroid_hz": float(centroid_hz),
            "beat": beat_flag,
        }


