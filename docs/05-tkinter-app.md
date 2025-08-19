### Milestone 5 — Tkinter App (MVP Packaging)

#### Goals
- Provide a simple desktop UI to start/stop capture, select device, choose visual preset, and tweak latency/smoothing.
- Package the app with reproducible dependencies; distribute a single-command run for Windows.

#### Deliverables
- `app/main.py`: Tkinter UI with controls and status indicators.
- `app/controller.py`: Orchestration layer (start/stop audio, analysis, visuals subprocess/window).
- `requirements.txt` and `Makefile` or `Invoke` tasks for `dev`, `run`, and `build` (Windows-friendly).
- PyInstaller one-folder build for Windows with icon and version metadata.
- Basic logging to console and rotating file in `logs/`.

#### Tasks
- UI layout
  - Device dropdown, start/stop button, latency slider, smoothing slider, scene preset dropdown.
  - Status: device name, sample rate, buffer stats, FPS.
- Process model
  - For MVP, launch Manim visual in a separate process/window; communicate over shared memory or local socket for band data.
  - Graceful shutdown and cleanup on exit.
- Configuration
  - Load/save YAML configs; persist last selections.
- Packaging (Windows)
  - `requirements.txt` with pinned versions; tasks for setup and run with `py -m venv .venv` and `.venv\\Scripts\\activate`.
  - PyInstaller spec for a single-folder Windows build; include `--add-data` for configs and icon.
- Testing
  - Manual smoke test instructions; headless check for import errors in CI (future).

#### Acceptance criteria
- App launches, lists audio devices, and starts a visual that reacts to audio.
- Start/stop works repeatedly without orphaned processes; close exits cleanly.
- Basic controls (latency, smoothing, preset) affect visuals immediately.

#### Risks and mitigations
- Embedding Manim in Tkinter is complex
  - Keep separate window for MVP; evaluate embedding later if required.
- Packaging with OpenGL dependencies on Windows GPUs
  - Document GPU driver requirements; test on a clean Windows VM and a real machine.

#### References
- Tkinter patterns: [Tkinter docs](https://docs.python.org/3/library/tkinter.html)
- PyInstaller: [PyInstaller docs](https://pyinstaller.org/en/stable/)
- Windows app packaging tips: [MS Docs - Desktop apps](https://learn.microsoft.com/windows/apps/)
