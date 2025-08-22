### Visualizer Pioneer USB — Audio‑Reactive Fractal Visualizer (Windows)

An OpenGL fractal visualizer that reacts to live system audio. Audio is captured via Windows WASAPI loopback (primary) or `soundcard` (fallback), analyzed into bass/mid/high bands, and mapped to fractal parameters in real time. UI controls are provided via Dear ImGui (overlay) or a Tkinter panel.

### Contents
- Requirements and environment setup
- Visual Studio Build Tools note (for `imgui` wheels)
- How audio capture and analysis work
- Configuration files (`configs/*.yaml`)
- Running the tools and visualizer scripts
- Troubleshooting

---

## Requirements and environment setup

Tested on Windows 10/11 with Python 3.10–3.12.

1) Create and activate a virtual environment (PowerShell):

```powershell
cd C:\Projects\visualizer_pioneer_usb
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

This installs: `numpy`, `sounddevice`, `PyYAML`, `soundcard` (fallback), `moderngl`, `glfw`, `imgui`.

3) GPU/OpenGL requirement:

- Needs OpenGL 3.3+ (provided by your GPU drivers). Keep graphics drivers up to date.

### Visual Studio Build Tools (for imgui if no wheel is available)

Most users will get prebuilt wheels for `imgui` on Windows. If pip attempts to build from source, install Microsoft Build Tools:

1) Download and install “Visual Studio Build Tools 2022”. During installation select:
- Desktop development with C++
- MSVC v143 build tools
- Windows 10/11 SDK

2) After installation, reopen your terminal and retry:

```powershell
pip install imgui>=2.0.0
```

If you still see build errors, ensure you’re in the activated venv and have `pip`, `setuptools`, `wheel` updated.

---

## How audio capture and analysis work

- Primary capture (`audio/capture.py`): Uses `sounddevice` with Windows WASAPI. In loopback mode it records the system’s currently playing audio from a selected output device.
- Device selection: You can target a device by substring (e.g., “Realtek”, “DDJ‑FLX4”) or explicit index. If not found, it falls back to the default device.
- Fallback (`audio/fallback.py`): If WASAPI capture fails, the app can switch to `soundcard` loopback and probe common samplerates/channel counts.

Processing pipeline (in `scripts/live_fractal.py` and `audio/analysis.py`):
- Blocks of float32 PCM frames are read from a ring buffer.
- Frames are converted to mono by averaging channels.
- A Hann window and rFFT produce a magnitude spectrum (`n_fft=1024`).
- Magnitudes are aggregated into log‑spaced bands. The visualizer uses 3 bands: bass / mid / high.
- Values are normalized (percentile window) and smoothed (attack/release EMA).
- The final 3 values drive fractal parameters (iterations, hue, trap mix, rotation, motion, bend/warp, etc.).

Controls at runtime:
- F1: Toggle ImGui overlay (if available).
- Ctrl+S: Save current visual intensity settings to `configs/visuals.yaml`.
- If ImGui is unavailable, a Tkinter control panel is launched with similar sliders.

---

## Configuration files

All YAMLs live in `configs/`. You can edit them before running, or adjust settings live and save from the UI.

### `configs/audio.yaml`
Keys map to `AudioConfig` in `audio/capture.py`:
- `host_api_name`: Typically "Windows WASAPI" on Windows.
- `device_substring`: Case‑insensitive substring to select a device by name.
- `device_index`: Explicit device index (overrides substring if set).
- `sample_rate`: Preferred sample rate (actual may follow device default).
- `block_size`: Audio block size in frames.
- `channels`: Desired channel count (clamped to device capability).
- `latency`: Target latency in seconds.
- `loopback`: true to capture system output (what you hear).
- `exclusive`: Try exclusive mode for WASAPI loopback.
- `dtype`: Sample dtype, e.g., `float32`.
- `ringbuffer_blocks`: Queue capacity in blocks.

Example (default repository values):

```yaml
host_api_name: "Windows WASAPI"
device_substring: "Głośniki"
device_index: null
sample_rate: 48000
block_size: 512
channels: 2
latency: 0.02
loopback: true
exclusive: false
dtype: "float32"
ringbuffer_blocks: 64
```

### `configs/analysis.yaml`
Reference parameters for spectrum and features (used by tooling and as documentation of the analysis model):
- `sample_rate`, `n_fft`, `hop_size`, `window`: STFT parameters.
- `bands`: Aggregation scheme — `mode` (linear/log), `num_bands`, `min_freq`, `max_freq`.
- `smoother`: Attack/release for EMA smoothing.
- `normalizer`: Peak/percentile mode with window and decay.
- `beat`: Energy‑based beat detector parameters.

Note: `live_fractal.py` currently uses fixed STFT settings (`n_fft=1024`, hop equal to FFT size for a per‑block spectrum) and a 3‑band log layout tuned for visuals.

### `configs/visuals.yaml`
Holds rendering and UI‑tuned intensity parameters. The visualizer reads the `live_fractal_intensity` block into `VisualIntensitySettings`:
- `master`, `exposure`, `contrast`: Global tone mapping controls.
- `glow_gain`, `trap_mix_gain`, `motion_gain`, `iteration_gain`: Audio‑reactive gains.
- `scale`, `iterations_base`, `bailout_radius`: Fractal shape/depth controls.
- `palette_id`, `hue_offset`, `palette_saturation`: Color controls.
- `fractal_type`: Switch among shader fractal variants.
 - `bend_gain`: Scales audio‑reactive bend/warp driven by overall energy (smoothed).
 - `view_angle_deg`: Static view rotation in degrees (added to audio‑driven rotation).
 - `view_center_x`, `view_center_y`: View UV center position (normalized 0..1).

Other keys in this file (e.g., `renderer`, `fps`, `shapes`, `color_scheme`) are for broader visualization contexts and may not be consumed by the fractal script.

---

## Running the tools and visualizer

All commands assume the venv is activated in the project root.

### 1) Probe audio devices and levels

List devices (filters by host API if provided):

```powershell
python scripts\probe_audio.py --list --host-api "Windows WASAPI"
```

Record meters (and optionally a WAV) using your config defaults:

```powershell
python scripts\probe_audio.py --config configs\audio.yaml --seconds 5 --loopback
# Optional WAV:
python scripts\probe_audio.py --config configs\audio.yaml --seconds 10 --loopback --wav out.wav
```

Override selections at the CLI:

```powershell
python scripts\probe_audio.py --loopback --device "Realtek" --sample-rate 48000 --channels 2 --seconds 5
```

If WASAPI fails, the tool can fall back to `soundcard` loopback and try common samplerates.

### 2) Run the live fractal visualizer

```powershell
python scripts\live_fractal.py
```

Tips:
- Start music playback first so loopback has a signal.
- Press F1 to show/hide the ImGui overlay. Adjust sliders live.
- Press Ctrl+S to write the current settings into `configs/visuals.yaml`.

---

## Troubleshooting

- No audio captured / silence:
  - Ensure `loopback: true` and `host_api_name: "Windows WASAPI"` in `configs/audio.yaml`.
  - Use `--list` to confirm the output device name, set `device_substring` accordingly (e.g., "Realtek", "Headphones").
  - Some devices only expose stereo loopback; keep `channels: 2`.

- Stream open errors on `imgui`, `glfw`, or `moderngl`:
  - Update GPU drivers; ensure OpenGL 3.3+.
  - If `imgui` wheel isn’t available, install Visual Studio Build Tools (see above) and reinstall.

- Buffer overruns or stutter:
  - Increase `latency` or `block_size`; reduce `ringbuffer_blocks` pressure.
  - Close other audio apps or disable `exclusive`.</n+
- Fallback captures instead of WASAPI:
  - The visualizer attempts WASAPI first; if it fails, `SoundcardLoopbackCapture` is used.

---

## Project structure

- `scripts/live_fractal.py`: Main visualizer entry point.
- `scripts/probe_audio.py`: Device listing, meters, optional WAV dump.
- `audio/capture.py`: WASAPI‑based capture via `sounddevice` with ring buffer.
- `audio/fallback.py`: `soundcard` loopback fallback.
- `audio/analysis.py`: STFT, band aggregation, smoothing, normalization, beat detection.
- `visuals/renderer.py`: OpenGL fractal shader and audio‑driven parameter mapping.
- `visuals/imgui_overlay.py`: Dear ImGui overlay integration (GLFW).
- `visuals/ui.py`: Tkinter control panel fallback.
- `configs/*.yaml`: Audio, analysis, and visual settings.

---

## License

MIT. Contributions welcome.


