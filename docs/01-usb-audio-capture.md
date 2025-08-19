### Milestone 1 — USB Audio Capture (Pioneer DDJ‑FLX4)

#### Goals
- Capture stereo PCM frames from the DDJ‑FLX4 (or any USB audio device) on Windows 10/11 using WASAPI (shared/exclusive) or WASAPI loopback with low latency. Note: ASIO support is optional and can be added later if required.
- Provide device discovery and selection by name, sample rate, and channel count.
- Expose a non-blocking stream API yielding NumPy arrays for downstream processing.
- Offer a simple CLI to verify capture (levels, sample rate, glitches).

#### Deliverables
- `audio/capture.py`: High-level capture module using `sounddevice` (PortAudio) with callbacks or generator API.
- `scripts/probe_audio.py`: CLI to list devices and print live RMS/peak for the selected device.
- `configs/audio.yaml`: Defaults (device name pattern, sample rate, block size, channels, latency).

#### Tasks
- Device reconnaissance
  - Enumerate devices using `sounddevice.query_devices()` and `sounddevice.query_hostapis()`; identify the "Windows WASAPI" host API and available input/endpoints.
  - Check Windows Sound Settings to confirm the DDJ‑FLX4 exposes a capture endpoint; otherwise plan to use WASAPI loopback of the controller's playback endpoint or system output.
  - Confirm supported sample rates (44.1/48 kHz) and channel mappings.
- Library choice and setup
  - Use `sounddevice` (PortAudio) with WASAPI; for loopback capture, use `sd.WasapiSettings(loopback=True)`.
  - Create environment (Windows): `py -m venv .venv && .venv\Scripts\activate && pip install sounddevice numpy pyyaml`.
- Implement capture API
  - Non-blocking callback -> ring buffer -> async iterator yielding `float32` frames shaped `(frames, channels)`.
  - Configurable: device name substring, host API (WASAPI), sample rate, block size (e.g., 256/512/1024), channels (2), latency hints, loopback mode.
  - Thread-safe shutdown and overrun/underrun logging.
- CLI and smoke tests
  - `scripts/probe_audio.py --list` shows devices and metadata (including host API and loopback capability).
  - `scripts/probe_audio.py --device "Pioneer" --seconds 5 --wav out.wav` records and saves WAV.
  - Print live RMS/peak per channel at ~10 Hz to verify signal.
- Observability and performance
  - Basic metrics: callback late count, buffer underruns/overruns, effective throughput.
  - Document minimal block size achieving stable capture on target machine.

#### Acceptance criteria
- Can select DDJ‑FLX4 by name and capture stable stereo audio at 44.1 kHz with <20 ms total buffer latency (configurable).
- 60-second capture produces no more than 1 buffer underrun/overrun at default settings.
- CLI lists devices and records a WAV playable in standard players.
- Code handles device not found / in-use errors with clear messages and exit codes.

#### Risks and mitigations
- Device is ASIO-only or grabbed by DJ software (exclusive mode)
  - Use WASAPI loopback on the controller's playback endpoint or system output; allow selecting an alternate loopback device.
  - Optionally investigate ASIO support later via PortAudio ASIO builds or third-party bridges (e.g., ASIO4ALL/FlexASIO) if direct capture is required.
- Channel mapping ambiguity (multiple endpoints)
  - Provide device/channel preview and persist last working configuration.
- Sample rate mismatch (44.1 vs 48 kHz)
  - Query supported rates; resample if necessary (future milestone) or fail clearly.
- CPU spikes causing underruns
  - Keep small, fixed work in callback; move heavy work to consumer thread with ring buffer backpressure.

#### References
- `sounddevice` docs (WASAPI and loopback): [Python SoundDevice](https://python-sounddevice.readthedocs.io/)
- PortAudio WASAPI notes: [PortAudio Wiki - WASAPI](http://portaudio.com/docs/v19-doxydocs/)
- Microsoft WASAPI overview: [Windows Audio Session API](https://learn.microsoft.com/windows/win32/coreaudio/core-audio-apis-in-windows-vista)
- Optional: ASIO drivers and bridges (ASIO4ALL, FlexASIO)
