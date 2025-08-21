import sys
import time
from pathlib import Path

import numpy as np
import yaml
import glfw
import moderngl

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.capture import AudioCapture, AudioConfig
from audio.fallback import SoundcardLoopbackCapture
from audio.analysis import compute_spectrum, aggregate_bands, Smoother, Normalizer
from visuals.settings import VisualIntensitySettings, save_visual_intensity_yaml
from visuals.renderer import FractalRenderer
from visuals.ui import TkControlPanel
from visuals.imgui_overlay import ImGuiOverlay


SAMPLE_RATE = 48000
BLOCK = 512
NFFT = 1024


def main():
    cfg_path = PROJECT_ROOT / "configs" / "audio.yaml"
    cfg_data = {}
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}

    cap_cfg = AudioConfig(
        device_substring=cfg_data.get("device_substring"),
        device_index=cfg_data.get("device_index"),
        host_api_name=cfg_data.get("host_api_name", "Windows WASAPI"),
        sample_rate=int(cfg_data.get("sample_rate", SAMPLE_RATE)),
        block_size=int(cfg_data.get("block_size", BLOCK)),
        channels=int(cfg_data.get("channels", 2)),
        latency=float(cfg_data.get("latency", 0.02)),
        loopback=bool(cfg_data.get("loopback", True)),
        exclusive=bool(cfg_data.get("exclusive", False)),
        dtype=str(cfg_data.get("dtype", "float32")),
        ringbuffer_blocks=int(cfg_data.get("ringbuffer_blocks", 64)),
    )

    cap: object
    analysis_sample_rate = int(cap_cfg.sample_rate)
    try:
        cap = AudioCapture(cap_cfg)
        cap.start()
        analysis_sample_rate = int(cap._selected_samplerate or cap_cfg.sample_rate)
    except Exception as e:
        print(f"AudioCapture failed ({e}); falling back to soundcard loopback...", file=sys.stderr)
        fallback = SoundcardLoopbackCapture(
            samplerate=int(cap_cfg.sample_rate),
            channels=int(cap_cfg.channels),
            blocksize=int(cap_cfg.block_size),
        )
        fallback.start()
        analysis_sample_rate = int(getattr(fallback, "_selected_sr", cap_cfg.sample_rate))
        cap = fallback

    W, H = 1280, 720
    if not glfw.init():
        raise SystemExit("Failed to init GLFW")
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(W, H, "Live Audio Fractal", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("Cannot open window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    ctx = moderngl.create_context()

    vis_settings = VisualIntensitySettings.from_yaml(PROJECT_ROOT / "configs" / "visuals.yaml")
    overlay = ImGuiOverlay(window, W, H)
    tk_panel = None
    if not overlay.available:
        try:
            tk_panel = TkControlPanel(
                vis_settings,
                save_callback=lambda s: save_visual_intensity_yaml(s, PROJECT_ROOT / "configs" / "visuals.yaml"),
            )
            tk_panel.start()
            print("[UI] Tkinter control panel launched.")
        except Exception:
            tk_panel = None
            print("[UI] Tkinter control panel failed to launch.")

    renderer = FractalRenderer(ctx, W, H)
    band_normalizer = Normalizer(mode="percentile", window=90, decay=0.02)
    band_smoother = Smoother(attack=0.6, release=0.3)

    last_time = time.time()
    prev_f1 = glfw.RELEASE
    prev_save_combo = False

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()

            fb_w, fb_h = glfw.get_framebuffer_size(window)
            if fb_w > 0 and fb_h > 0:
                renderer.resize(fb_w, fb_h)
                overlay.set_size(fb_w, fb_h)

            f1_state = glfw.get_key(window, glfw.KEY_F1)
            if f1_state == glfw.PRESS and prev_f1 != glfw.PRESS:
                try:
                    overlay.toggle_visibility()
                except Exception:
                    pass
            prev_f1 = f1_state

            ctrl_down = (glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS) or (glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS)
            s_down = glfw.get_key(window, glfw.KEY_S) == glfw.PRESS
            pending_save = bool(ctrl_down and s_down and not prev_save_combo)
            prev_save_combo = bool(ctrl_down and s_down)

            last_block = None
            while True:
                blk = cap.read(timeout=0.0)
                if blk is None:
                    break
                last_block = blk
            if last_block is not None:
                mono = last_block.astype(np.float32, copy=False)
                if mono.ndim == 2:
                    mono = mono.mean(axis=1)
                mag = compute_spectrum(mono, window="hann", n_fft=NFFT, hop_size=NFFT)
                bands_mat = aggregate_bands(
                    mag,
                    {
                        "mode": "log",
                        "num_bands": 3,
                        "sample_rate": int(analysis_sample_rate),
                        "n_fft": NFFT,
                        "min_freq": 20.0,
                        "max_freq": 8000.0,
                    },
                )
                bands_vec = bands_mat[:, -1]
                bands_norm = band_normalizer.update(bands_vec)
                bands_sm = band_smoother.update(bands_norm)
                b, m, h = float(bands_sm[0]), float(bands_sm[1]), float(bands_sm[2])
                renderer.update_audio(b, m, h)

            now = time.time()
            dt = now - last_time
            last_time = now
            renderer.step(vis_settings, dt)
            renderer.draw()

            overlay.draw(
                vis_settings,
                pending_save,
                save_cb=lambda s: save_visual_intensity_yaml(s, PROJECT_ROOT / "configs" / "visuals.yaml"),
            )
            glfw.swap_buffers(window)
    finally:
        try:
            cap.stop()
        except Exception:
            pass
        try:
            overlay.shutdown()
        except Exception:
            pass
        try:
            if tk_panel is not None:
                tk_panel.close()
        except Exception:
            pass
        glfw.terminate()


if __name__ == '__main__':
    main()


