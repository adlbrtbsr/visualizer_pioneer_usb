### Milestone 4 — Synchronization and Latency Calibration

#### Goals
- Minimize and stabilize visual latency versus audible output.
- Provide a user-adjustable latency offset and an optional automated calibration routine.

#### Deliverables
- `sync/clock.py`: Helpers for timestamping audio blocks and estimating end-to-end latency.
- Calibration tool `scripts/calibrate_latency.py` producing a recommended `latency_ms` stored in `configs/audio.yaml`.
- Integration: visuals apply a global offset so bars align with transients.

#### Tasks
- Timebase and timestamps
  - Use `sounddevice` stream timestamps to estimate capture time of each block (WASAPI timing).
  - Maintain a monotonic clock for render frames; compute expected alignment.
- Measure and compensate (Windows)
  - Option A: WASAPI loopback calibration — play a synthetic click to the output device and capture via loopback; detect peaks and measure offset.
  - Option B: Manual calibration — hotkeys to nudge visual offset until alignment feels correct; persist value.
  - Expose `latency_ms` to shift band values when read by visuals.
- Stability and jitter handling
  - Smoothing on timestamp jitter; clamp extreme outliers; basic logging of effective latency.
- UX hooks
  - CLI to increment/decrement latency live and persist to config.

#### Acceptance criteria
- With default settings, measured steady-state latency is within ±25 ms of target after calibration.
- Offset remains stable (<10 ms jitter RMS) during a 2-minute run.
- Users can adjust latency at runtime and persist preference.

#### Risks and mitigations
- System scheduling variations
  - Favor larger audio blocks or fewer FFT updates if jitter is high; document tradeoffs.
- Inability to run loopback calibration on some setups
  - Provide manual tap-tempo style alignment tool as fallback.

#### References
- PortAudio timing: [SoundDevice timing](https://python-sounddevice.readthedocs.io/en/0.4.6/api/streams.html#sounddevice.Stream.time)
- Audio latency concepts: [Latency and buffering](https://www.cockos.com/reaper/userguide/reaper4605b.pdf)
