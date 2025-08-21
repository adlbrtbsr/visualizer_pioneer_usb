### Live GPU Fractal Visual (Real-time)

This visualizer uses `moderngl` + `glfw` to render a Mandelbrot fractal in a fragment shader, driven by live audio captured via `sounddevice`. FFT band features (bass/mid/treble) are computed with NumPy and mapped to zoom, pan velocity, and iteration depth.

#### Install

```bash
pip install -r requirements.txt
```

This includes `moderngl`, `glfw`, and `sounddevice`. Ensure your GPU supports OpenGL 3.3+ and your audio input device is available.

#### Run

```bash
python scripts/live_fractal.py
```

#### Tuning

- Lower `BLOCK` in `scripts/live_fractal.py` (e.g., 512) for lower audio latency (higher CPU).
- The fragment shader iteration count (`u_max_iter`) is driven by audio energy; raise `iter_base` or its scaling for more detail if your GPU has headroom.
- Extend the mapping: add more bands or a beat detector and map them to new uniforms (palette switches, rotation, Julia morphs, etc.).

#### Notes

- The fractal iterations run fully on the GPU for high frame rates. The CPU performs small FFTs and simple reductions each frame.
- Windows users may need the latest graphics drivers for OpenGL 3.3 support.


