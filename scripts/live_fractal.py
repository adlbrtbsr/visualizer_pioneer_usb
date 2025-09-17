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

from audio.capture import AudioCapture, AudioConfig, list_devices
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
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
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

    # Runtime audio device switching via ImGui
    pending_switch_to_device_index = None
    is_switching_audio = False
    loopback_enabled = bool(cap_cfg.loopback)
    current_backend = "sounddevice" if isinstance(cap, AudioCapture) else "soundcard"

    def _current_device_index() -> int | None:
        try:
            if isinstance(cap, AudioCapture):
                return getattr(cap, "_selected_device_index", None)
        except Exception:
            pass
        return None

    def _switch_device(new_device_index: int | None) -> None:
        nonlocal cap, analysis_sample_rate, feature_cfg, extractor, is_switching_audio
        is_switching_audio = True
        try:
            # Stop current capture (both primary and fallback paths)
            try:
                cap.stop()
            except Exception:
                pass
            # Small grace to let backend finish callbacks
            time.sleep(0.05)
            # Ensure PortAudio releases resources fully before reopening
            try:
                import sounddevice as _sd
                _sd.stop()
                try:
                    # Clear default device tuple to avoid stale device handles
                    _sd.default.device = (None, None)
                except Exception:
                    pass
            except Exception:
                pass
            time.sleep(0.15)
            # Try primary AudioCapture with requested device
            try:
                new_cfg = AudioConfig(
                    device_substring=None,
                    device_index=new_device_index,
                    host_api_name=cap_cfg.host_api_name,
                    sample_rate=cap_cfg.sample_rate,
                    block_size=cap_cfg.block_size,
                    channels=cap_cfg.channels,
                    latency=cap_cfg.latency,
                    loopback=loopback_enabled,
                    exclusive=cap_cfg.exclusive,
                    dtype=cap_cfg.dtype,
                    ringbuffer_blocks=cap_cfg.ringbuffer_blocks,
                )
                new_cap = AudioCapture(new_cfg)
                new_cap.start()
                cap = new_cap
                analysis_sample_rate = int(getattr(new_cap, "_selected_samplerate", new_cfg.sample_rate) or new_cfg.sample_rate)
                current_backend = "sounddevice"
            except Exception:
                # Fallback to soundcard (loopback or input depending on flag)
                try:
                    name_hint = None
                    # Try to pass the selected device name as hint when possible
                    try:
                        devs = list_devices(host_api_name=cap_cfg.host_api_name)
                        for d in devs:
                            if d.get("index") == new_device_index:
                                name_hint = d.get("name")
                                break
                    except Exception:
                        pass
                    fallback = SoundcardLoopbackCapture(
                        samplerate=int(cap_cfg.sample_rate),
                        channels=int(cap_cfg.channels),
                        blocksize=int(cap_cfg.block_size),
                        device_name_hint=name_hint,
                    )
                    fallback.start()
                    cap = fallback
                    analysis_sample_rate = int(getattr(fallback, "_selected_sr", cap_cfg.sample_rate))
                    current_backend = "soundcard"
                except Exception:
                    # If even fallback fails, try to restart previous cap
                    try:
                        new_cap = AudioCapture(cap_cfg)
                        new_cap.start()
                        cap = new_cap
                        analysis_sample_rate = int(getattr(new_cap, "_selected_samplerate", cap_cfg.sample_rate) or cap_cfg.sample_rate)
                        current_backend = "sounddevice"
                    except Exception:
                        pass
            # Small grace period after (re)starting to avoid backend race conditions
            time.sleep(0.05)

            # Rebuild feature extractor with updated sample rate
            try:
                feature_cfg = FeatureExtractorConfig(
                    sample_rate=int(analysis_sample_rate),
                    n_fft=int(feature_cfg.n_fft),
                    hop_size=int(feature_cfg.hop_size),
                    mel_bands=int(feature_cfg.mel_bands),
                    use_chroma=bool(feature_cfg.use_chroma),
                    mel_smoother_attack=float(feature_cfg.mel_smoother_attack),
                    mel_smoother_release=float(feature_cfg.mel_smoother_release),
                    chroma_smoother_attack=float(feature_cfg.chroma_smoother_attack),
                    chroma_smoother_release=float(feature_cfg.chroma_smoother_release),
                    scalar_smoother_attack=float(feature_cfg.scalar_smoother_attack),
                    scalar_smoother_release=float(feature_cfg.scalar_smoother_release),
                    normalizer_mode=str(feature_cfg.normalizer_mode),
                    normalizer_window=int(feature_cfg.normalizer_window),
                    normalizer_decay=float(feature_cfg.normalizer_decay),
                    normalizer_floor=float(feature_cfg.normalizer_floor),
                    beat_threshold=float(feature_cfg.beat_threshold),
                    beat_short_window=int(feature_cfg.beat_short_window),
                    beat_long_window=int(feature_cfg.beat_long_window),
                    beat_refractory_frames=int(feature_cfg.beat_refractory_frames),
                )
                extractor = FeatureExtractor(feature_cfg)
            except Exception:
                pass
        finally:
            is_switching_audio = False

    def _draw_audio_panel() -> None:
        nonlocal pending_switch_to_device_index, loopback_enabled
        try:
            import imgui  # local import to keep module optional
        except Exception:
            return
        # Build device list filtered by host API and capability
        devices = list_devices(host_api_name=cap_cfg.host_api_name)
        # Filter by input capability for normal capture, output for loopback
        items = ["Default (System)"]
        idx_map: list[int | None] = [None]
        for d in devices:
            if cap_cfg.loopback:
                if int(d.get("max_output_channels", 0) or 0) <= 0:
                    continue
            else:
                if int(d.get("max_input_channels", 0) or 0) <= 0:
                    continue
            label = f"[{d['index']}] {d['name']}"
            items.append(label)
            idx_map.append(int(d["index"]))

        current_sel = 0
        cur_dev = _current_device_index()
        if cur_dev is not None:
            for i, val in enumerate(idx_map):
                if val == cur_dev:
                    current_sel = i
                    break

        if imgui.begin("Audio Input", True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text(f"Backend: {current_backend}")
            cur = 1 if loopback_enabled else 0
            changed_lb, cur = imgui.combo("Mode", cur, ["Input (mic)", "Loopback (speaker)"])
            if changed_lb:
                loopback_enabled = bool(cur == 1)
            changed, new_sel = imgui.combo("Device", current_sel, items)
            if changed and 0 <= new_sel < len(idx_map):
                pending_switch_to_device_index = idx_map[new_sel]
        imgui.end()

    # Mapping engine from YAML
    mapping_path = PROJECT_ROOT / "configs" / "mapping.yaml"
    mapping = MappingEngine.from_yaml(mapping_path, mel_bands=feature_cfg.mel_bands)

    last_time = time.time()
    prev_f1 = glfw.RELEASE
    prev_f11 = glfw.RELEASE
    prev_save_combo = False

    # Fullscreen toggle state
    is_fullscreen = False
    try:
        windowed_size = glfw.get_window_size(window)
        windowed_pos = glfw.get_window_pos(window)
    except Exception:
        windowed_size = (W, H)
        windowed_pos = (100, 100)

    def _toggle_fullscreen() -> None:
        nonlocal is_fullscreen, windowed_size, windowed_pos
        try:
            if not is_fullscreen:
                # Save current windowed position and size
                try:
                    windowed_pos = glfw.get_window_pos(window)
                    windowed_size = glfw.get_window_size(window)
                except Exception:
                    pass
                monitor = glfw.get_window_monitor(window) or glfw.get_primary_monitor()
                if not monitor:
                    return
                mode = glfw.get_video_mode(monitor)
                if mode is None:
                    return
                try:
                    width = int(mode.size.width)
                    height = int(mode.size.height)
                    refresh = int(getattr(mode, "refresh_rate", 0) or 0)
                except Exception:
                    width, height, refresh = 1920, 1080, 0
                glfw.set_window_monitor(window, monitor, 0, 0, width, height, refresh)
                is_fullscreen = True
            else:
                x, y = int(windowed_pos[0]), int(windowed_pos[1])
                w, h = int(windowed_size[0]), int(windowed_size[1])
                if w <= 0 or h <= 0:
                    w, h = 1280, 720
                glfw.set_window_monitor(window, None, x, y, w, h, 0)
                is_fullscreen = False
        except Exception:
            # Fail-safe: ignore toggle errors
            pass

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

            # F11: Toggle fullscreen
            f11_state = glfw.get_key(window, glfw.KEY_F11)
            if f11_state == glfw.PRESS and prev_f11 != glfw.PRESS:
                _toggle_fullscreen()
            prev_f11 = f11_state

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
                extra_draw=_draw_audio_panel,
            )

            # Apply pending device switch after UI frame to avoid re-entrancy
            if (not is_switching_audio) and (pending_switch_to_device_index is not None):
                _switch_device(pending_switch_to_device_index)
                pending_switch_to_device_index = None
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


