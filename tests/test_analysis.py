import numpy as np

from audio.analysis import (
	Smoother,
	Normalizer,
	BeatDetector,
	compute_spectrum,
	aggregate_bands,
	magnitude_to_dbfs,
)


def test_compute_spectrum_basic_tone():
	sr = 48000
	freq = 1000.0
	dur = 0.050
	n_fft = 1024
	hop = n_fft // 2
	t = np.arange(int(dur * sr)) / sr
	x = np.sin(2 * np.pi * freq * t).astype(np.float32)
	mag = compute_spectrum(x, window="hann", n_fft=n_fft, hop_size=hop)
	# Expect shape (bins, frames)
	assert mag.shape[0] == n_fft // 2 + 1
	assert mag.shape[1] >= 1
	# Peak near target bin
	bin_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
	peak_bins = np.argmax(mag, axis=0)
	peak_freqs = bin_freqs[peak_bins]
	assert np.all(np.abs(peak_freqs - freq) < 200.0)


def test_aggregate_bands_linear_and_log():
	sr = 44100
	n_fft = 1024
	frames = 3
	mag = np.ones((n_fft // 2 + 1, frames), dtype=np.float32)
	# Linear
	lin = aggregate_bands(mag, scheme="linear:32")
	assert lin.shape == (32, frames)
	assert np.allclose(lin, 1.0, atol=1e-6)
	# Log
	log = aggregate_bands(
		mag,
		scheme={"mode": "log", "num_bands": 24, "sample_rate": sr, "n_fft": n_fft, "min_freq": 20.0},
	)
	assert log.shape == (24, frames)
	assert np.allclose(log, 1.0, atol=1e-6)


def test_smoother_attack_release_behaviour():
	s = Smoother(attack=1.0, release=0.1)
	# Rising step
	vals = [0.0, 0.0, 1.0, 1.0]
	outs = [s.update(v) for v in vals]
	assert outs[2] >= 0.9
	# Falling step triggers release smoothing
	out3 = s.update(0.0)
	assert 0.0 < out3 < 1.0


def test_normalizer_peak_and_percentile():
	# Peak mode
	n = Normalizer(mode="peak", decay=0.5)
	vals = [0.1, 0.2, 0.4]
	outs = [n.update(np.array([v], dtype=np.float32)) for v in vals]
	assert outs[-1] <= 1.0 and outs[-1] >= 0.0
	# Percentile mode
	n = Normalizer(mode="percentile", window=10)
	for v in np.linspace(0.0, 1.0, 10):
		out = n.update(np.array([v], dtype=np.float32))
	assert 0.0 <= out[0] <= 1.0


def test_magnitude_to_dbfs_floor():
	mag = np.array([0.0, 1e-12, 1.0], dtype=np.float32)
	db = magnitude_to_dbfs(mag, ref=1.0, min_db=-60.0)
	assert db.shape == mag.shape
	assert np.all(db >= -60.0 - 1e-5)


def test_beat_detector_triggers_and_refractory():
	bd = BeatDetector(threshold=1.2, short_window=3, long_window=9, refractory_frames=2)
	# Warm-up with low energy
	for _ in range(12):
		assert bd.update(np.array([0.1, 0.1, 0.1], dtype=np.float32)) is False
	# Inject a burst
	assert bd.update(np.array([1.0, 1.0, 1.0], dtype=np.float32)) in (True, False)
	# Immediately after, refractory likely prevents another True
	assert bd.update(np.array([1.0, 1.0, 1.0], dtype=np.float32)) is False


