### Milestone 2 — Real‑time FFT and Feature Extraction

#### Goals
- Transform captured audio into frequency-domain features in real time.
- Provide stable band energies (linear and log-spaced), RMS, and simple beat/peak indicators.
- Control smoothing and normalization for visually pleasant dynamics.

#### Deliverables
- `audio/analysis.py`:
  - `compute_spectrum(frames, window, n_fft, hop_size)` → magnitude spectrum.
  - `aggregate_bands(mag_spectrum, scheme)` → N bands (e.g., 32/64), linear or log.
  - `Smoother` for EMA/attack-release; `Normalizer` with rolling stats.
  - Optional `BeatDetector` (energy-based or spectral flux baseline).
- `tests/test_analysis.py`: Unit tests for windowing, band aggregation, and smoothing edge cases.
- `configs/analysis.yaml`: FFT size, hop, window type, banding, smoothing, normalization.

#### Tasks
- Choose parameters and implement DSP primitives
  - Start with 1024–2048 FFT, hop 50% (window: Hann), sample rate from capture.
  - Compute magnitude in dBFS and linear; add floor to avoid log singularities.
- Band aggregation
  - Linear buckets and psychoacoustic-ish log buckets (e.g., 20 Hz to Nyquist with ~1/6th octave spacing).
  - Per-band smoothing with separate attack/release factors.
- Normalization
  - Rolling percentile or peak hold with decay; clamp to [0,1] for visuals.
- Beat/peak detection (simple)
  - Short-term energy vs moving average; flag boolean pulse with refractory period.
- Throughput testing
  - Benchmark end-to-end pipeline at 60 FPS target; ensure < 2 ms per audio block at typical settings.

#### Acceptance criteria
- Processing keeps up with live capture at default FFT settings on the target Windows machine.
- Stable band outputs (no flicker) with tunable smoothing; values in [0,1].
- Unit tests pass and cover windowing and aggregation correctness.

#### Risks and mitigations
- Parameter sensitivity (FFT size vs latency)
  - Provide presets and document tradeoffs; expose via config/UI later.
- Spectral leakage and DC bias
  - Always window; high-pass or DC removal if needed.
- Performance regressions
  - Use vectorized NumPy; avoid Python loops in hot path.

#### References
- NumPy FFT: [numpy.fft](https://numpy.org/doc/stable/reference/routines.fft.html)
- Spectral basics: [Spectrum analysis (SciPy docs)](https://docs.scipy.org/doc/scipy/tutorial/fft.html)
