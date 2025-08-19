### Milestone 3 — Minimal Manim Visuals

#### Goals
- Render a simple but responsive spectrum visual with Manim Community Edition using the OpenGL renderer.
- Drive mobject properties (height, color) from real-time band energies.
- Keep the scene structure extensible for future visuals.

#### Deliverables
- `visuals/scenes.py`:
  - `SpectrumBarsScene`: Horizontal/vertical bars mapped to N bands with `ValueTracker`s and updaters.
  - Utility to color-map values (e.g., gradient by magnitude).
- `scripts/run_visual.py`: Launch the live scene connected to the analysis pipeline.
- `configs/visuals.yaml`: Band count, color scheme, FPS, renderer options.

#### Tasks
- Environment (Windows)
  - Install ManimCE with OpenGL: `pip install manim` and enable `--renderer=opengl`.
  - Ensure required Microsoft Visual C++ Redistributable and GPU drivers are installed.
  - Verify a basic scene renders at 60 FPS windowed.
- Data flow and threading
  - Start audio capture + analysis in a worker thread; share latest band vector via a thread-safe structure.
  - In Manim scene, use updaters to poll the shared state each frame and update bar heights/colors.
- Visual implementation
  - Pre-create N rectangles; scale Y (or X) by band value in [0,1].
  - Optional: simple color gradient; floor to minimum height to avoid disappearing bars.
- Controls and stability
  - Command-line flags for device, bands, FPS.
  - Graceful shutdown on ESC/close; ensure worker thread stops.

#### Acceptance criteria
- Scene window opens and reacts to music from the DDJ‑FLX4 with visible, stable bars.
- Frame rate ≥ 30 FPS sustained for 2 minutes without visual stutter or memory growth.
- No cross-thread exceptions; clean shutdown.

#### Risks and mitigations
- Manim not primarily designed for interactive real-time rendering
  - Use OpenGL renderer and minimal mobject count; avoid heavy animations.
- Threading and GIL contention
  - Keep audio/FFT in separate thread; share compact numpy array; avoid locks in the render loop by using atomic swap or copy-on-read.
- Embedding in GUI later
  - For MVP, run Manim in its own window; embed/compose with Tkinter in a later milestone.

#### References
- ManimCE docs: [Community Edition Guide](https://docs.manim.community/)
- Updaters and ValueTrackers: [Updaters](https://docs.manim.community/en/stable/reference/manim.mobject.updating.html)
