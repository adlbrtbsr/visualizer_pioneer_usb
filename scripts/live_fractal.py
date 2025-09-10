import sys
import time
from pathlib import Path

import numpy as np
import yaml
import glfw
import moderngl

# Resolve project root for both source and frozen executables
IS_FROZEN = bool(getattr(sys, 'frozen', False))
if IS_FROZEN:
    # When packaged (PyInstaller), use the executable directory as the root
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    # Make imports work when running directly from source
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from audio.capture import AudioCapture, AudioConfig
from audio.fallback import SoundcardLoopbackCapture
from audio.analysis import Smoother, Normalizer
from audio.features import FeatureExtractor, FeatureExtractorConfig
from visuals.mapping import MappingEngine
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

    # Load analysis features config
    analysis_cfg_path = PROJECT_ROOT / "configs" / "analysis.yaml"
    feature_cfg = None
    if analysis_cfg_path.is_file():
        with analysis_cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            fnode = (data.get("features") or {}) if isinstance(data, dict) else {}
            mel_bands = int(fnode.get("mel_bands", 24))
            use_chroma = bool(fnode.get("chroma", True))
            ms = fnode.get("mel_smoother", {})
            cs = fnode.get("chroma_smoother", {})
            ss = fnode.get("scalar_smoother", {})
            nn = fnode.get("normalizer", {})
            bb = fnode.get("beat", {})
            feature_cfg = FeatureExtractorConfig(
                sample_rate=int(analysis_sample_rate),
                n_fft=int(data.get("n_fft", 1024)),
                hop_size=int(data.get("hop_size", 512)),
                mel_bands=mel_bands,
                use_chroma=use_chroma,
                mel_smoother_attack=float(ms.get("attack", 0.6)),
                mel_smoother_release=float(ms.get("release", 0.35)),
                chroma_smoother_attack=float(cs.get("attack", 0.6)),
                chroma_smoother_release=float(cs.get("release", 0.35)),
                scalar_smoother_attack=float(ss.get("attack", 0.6)),
                scalar_smoother_release=float(ss.get("release", 0.35)),
                normalizer_mode=str(nn.get("mode", "percentile")),
                normalizer_window=int(nn.get("window", 120)),
                normalizer_decay=float(nn.get("decay", 0.05)),
                normalizer_floor=float(nn.get("floor", 1.0e-3)),
                beat_threshold=float(bb.get("threshold", 1.4)),
                beat_short_window=int(bb.get("short_window", 5)),
                beat_long_window=int(bb.get("long_window", 43)),
                beat_refractory_frames=int(bb.get("refractory_frames", 12)),
            )
    if feature_cfg is None:
        feature_cfg = FeatureExtractorConfig(sample_rate=int(analysis_sample_rate))
    extractor = FeatureExtractor(feature_cfg)

    # Mapping engine from YAML
    mapping_path = PROJECT_ROOT / "configs" / "mapping.yaml"
    mapping = MappingEngine.from_yaml(mapping_path, mel_bands=feature_cfg.mel_bands)

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
                feats = extractor.update(last_block)
                now_t = time.time()
                b, m, h, vis_out = mapping.map(feats, vis_settings, time_sec=now_t)
                # Update renderer with proxy bands; step() uses VisualIntensitySettings
                renderer.update_audio(float(b), float(m), float(h))
                # Replace settings for this frame
                vis_settings = vis_out

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


