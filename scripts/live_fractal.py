import math
import warnings
import time
import sys
from pathlib import Path
import numpy as np
import yaml
from queue import Queue, Full, Empty
import threading

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.capture import AudioCapture, AudioConfig, list_devices
import sounddevice as sd
from audio.analysis import compute_spectrum, aggregate_bands, Smoother, Normalizer
import glfw
import moderngl
try:
    import soundcard as sc
except Exception:
    sc = None
else:
    # Suppress benign discontinuity warnings from Media Foundation backend
    try:
        warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)
    except Exception:
        warnings.filterwarnings("ignore", message="data discontinuity in recording")

# ---------------------- audio ----------------------
SAMPLE_RATE = 48000
BLOCK = 512             # smaller -> lower latency, higher CPU
NFFT = 1024             # zero-pad above BLOCK for cleaner bands


def main():
    # Load audio capture configuration similar to probe_audio
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

    # Try primary capture (sounddevice WASAPI loopback). If it fails, fall back to soundcard loopback.
    cap: object
    analysis_sample_rate = int(cap_cfg.sample_rate)

    class SoundcardLoopbackCapture:
        def __init__(self, samplerate: int, channels: int, blocksize: int) -> None:
            self.samplerate = int(samplerate)
            self.channels = int(channels)
            self.blocksize = int(blocksize)
            self._queue: Queue = Queue(maxsize=64)
            self._running = False
            self._recorder = None
            self._selected_sr = None
            self._selected_ch = None
            self._frames = 0
            self._overruns = 0

        @property
        def metrics(self) -> dict:
            return {
                "queue_size": self._queue.qsize(),
                "overruns": self._overruns,
                "callback_late": 0,
                "frames_captured": self._frames,
            }

        def start(self) -> None:
            if sc is None:
                raise RuntimeError("soundcard module not available for fallback loopback")
            # Prefer Realtek default speaker loopback
            mic = None
            try:
                for s in sc.all_speakers():
                    if "Realtek" in s.name:
                        mic = sc.get_microphone(s.name, include_loopback=True)
                        break
            except Exception:
                mic = None
            if mic is None:
                mic = sc.get_microphone(sc.default_speaker().name, include_loopback=True)

            # Build candidate lists
            sr_candidates = []
            for sr in [self.samplerate, 48000, 44100]:
                if sr and int(sr) not in sr_candidates:
                    sr_candidates.append(int(sr))
            ch_candidates = []
            mic_ch = getattr(mic, "channels", None)
            for ch in [self.channels, mic_ch, 2, 1]:
                if isinstance(ch, int) and ch > 0 and ch not in ch_candidates:
                    ch_candidates.append(int(ch))

            last_err = None
            for sr in sr_candidates:
                for ch in ch_candidates:
                    try:
                        rec = mic.recorder(samplerate=sr, channels=ch, blocksize=self.blocksize,
                                           exclusive_mode=("DDJ-FLX4" in mic.name))
                        # Start the recorder context explicitly
                        rec.__enter__()
                        # Test a short record to validate
                        _ = rec.record(numframes=self.blocksize)
                        self._recorder = rec
                        self._selected_sr = int(sr)
                        self._selected_ch = int(ch)
                        self._running = True
                        # Start background reader thread to keep UI non-blocking
                        def _reader_loop():
                            while self._running and self._recorder is not None:
                                try:
                                    block = self._recorder.record(numframes=self.blocksize)
                                    block = block.astype(np.float32)
                                    if block.ndim == 1:
                                        block = block[:, np.newaxis]
                                    try:
                                        self._queue.put_nowait(block)
                                    except Full:
                                        # Drop oldest to make room
                                        try:
                                            _ = self._queue.get_nowait()
                                        except Exception:
                                            pass
                                        try:
                                            self._queue.put_nowait(block)
                                        except Exception:
                                            self._overruns += 1
                                            pass
                                    self._frames += block.shape[0]
                                except Exception:
                                    # brief backoff to avoid tight error loop
                                    time.sleep(0.002)
                                    continue
                        t = threading.Thread(target=_reader_loop, daemon=True)
                        t.start()
                        return
                    except Exception as e:
                        last_err = e
                        # Ensure cleanup if __enter__ succeeded
                        try:
                            if rec is not None:
                                rec.__exit__(None, None, None)
                        except Exception:
                            pass
                        self._recorder = None
                        continue
            raise last_err if last_err else RuntimeError("Failed to start soundcard loopback")

        def stop(self) -> None:
            self._running = False
            try:
                if self._recorder is not None:
                    self._recorder.__exit__(None, None, None)
            except Exception:
                pass
            self._recorder = None
            try:
                while True:
                    self._queue.get_nowait()
            except Exception:
                pass

        def read(self, timeout: float | None = None):
            if timeout is None:
                try:
                    return self._queue.get_nowait()
                except Exception:
                    return None
            if timeout <= 0.0:
                try:
                    return self._queue.get_nowait()
                except Exception:
                    return None
            try:
                return self._queue.get(timeout=timeout)
            except Exception:
                return None

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

    # ---------------------- window + GL ----------------------
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
    # Stable timing and no special mouse modes
    glfw.swap_interval(1)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    ctx = moderngl.create_context()

    # Fullscreen quad
    quad = ctx.buffer(np.array([
        -1, -1,  0, 0,
         1, -1,  1, 0,
        -1,  1,  0, 1,
         1,  1,  1, 1,
    ], dtype='f4').tobytes())
    vao = ctx.simple_vertex_array(
        ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_position;
                in vec2 in_uv;
                out vec2 v_uv;
                void main(){
                    v_uv = in_uv;
                    gl_Position = vec4(in_position, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_uv;
                out vec4 f_color;

                // uniforms (driven by audio)
                uniform float u_time;
                uniform vec2  u_res;
                uniform vec2  u_center;   // complex plane center
                uniform float u_scale;    // zoom scale (smaller -> zoom in)
                uniform int   u_max_iter; // fractal iterations
                uniform float u_bass;
                uniform float u_mid;
                uniform float u_high;
                uniform vec2  u_c;        // Julia parameter (audio-driven)
                uniform float u_power;    // Multibrot power (audio-driven)
                uniform float u_rot;      // pre-rotation angle (audio-driven)
                uniform float u_pre_scale;// subtle pre-scale (audio-driven)
                uniform float u_warp;     // quadratic domain warp strength
                uniform float u_energy;   // overall energy for gentle exposure
                uniform float u_morph;    // 0: Mandelbrot-like, 1: Julia
                uniform float u_ship;     // 0: normal, 1: Burning Ship blend
                uniform float u_swirl;    // radial swirl strength
                uniform float u_shear;    // shear factor
                uniform vec2  u_view_uv_center; // screen-space center in UV
                // Orbit-trap parameters for interior coloring
                uniform float u_trap_r;      // circle trap radius
                uniform float u_trap_mix;    // mix factor 0..1
                uniform float u_trap_rot;    // rotation for cross trap
                uniform float u_bail;        // bailout radius for escape and smoothing

                // High-contrast HSV palette
                vec3 hsv2rgb(vec3 c) {
                    vec3 p = abs(fract(c.xxx + vec3(0.0, 0.6666667, 0.3333333)) * 6.0 - 3.0);
                    vec3 rgb = c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
                    return rgb;
                }

                void main(){
                    // keep aspect ratio and apply audio-driven pre-transform (rotation, slight scale, warp)
                    float aspect = u_res.x / u_res.y;
                    vec2 p0 = (v_uv - u_view_uv_center) * vec2(aspect, 1.0) * u_scale;
                    float cs = cos(u_rot), sn = sin(u_rot);
                    mat2 R = mat2(cs, -sn, sn, cs);
                    mat2 Sh = mat2(1.0, u_shear, u_shear, 1.0);
                    vec2 p = R * (Sh * (p0 * u_pre_scale));
                    // swirl disabled: keep p unchanged after rotation/shear
                    // small quadratic domain warp (stays centered)
                    vec2 p2 = vec2(p.x*p.x - p.y*p.y, 2.0*p.x*p.y);
                    p += u_warp * p2;
                    p += u_center;

                    // Audio-driven hybrid fractal
                    // External parameter: c_p blends between pixel coord and Julia c
                    vec2 c_p = mix(p, u_c, clamp(u_morph, 0.0, 1.0));
                    vec2 z = p;
                    int i;
                    // Orbit-trap accumulators
                    float trap_min_circ = 1e9;
                    float trap_min_cross = 1e9;
                    float cst = cos(u_trap_rot), snt = sin(u_trap_rot);
                    mat2 RT = mat2(cst, -snt, snt, cst);
                    for(i=0; i<u_max_iter && dot(z,z) <= (u_bail * u_bail); i++){
                        // Burning Ship blend (abs before power)
                        vec2 zb = mix(z, vec2(abs(z.x), abs(z.y)), clamp(u_ship, 0.0, 1.0));
                        float r = length(zb);
                        float theta = atan(zb.y, zb.x);
                        float rp = pow(r, u_power);
                        float ang = u_power * theta;
                        vec2 zp = rp * vec2(cos(ang), sin(ang));
                        z = zp + c_p;

                        // Orbit-trap measurements (on unblended z for variety)
                        vec2 zt = RT * z;
                        float r_zt = length(zt);
                        trap_min_circ = min(trap_min_circ, abs(r_zt - u_trap_r));
                        trap_min_cross = min(trap_min_cross, min(abs(zt.x), abs(zt.y)));
                    }

                    float mu = float(i);
                    if (i < u_max_iter) {
                        // Smooth-ish coloring for general power
                        float r2 = dot(z,z);
                        float log_zn = 0.5 * log(r2);
                        float nu = log(log_zn / log(max(u_bail, 1.01))) / max(log(u_power), 1e-4);
                        mu = float(i) + 1.0 - nu;
                    }

                    // Add very small jitter to mu to break iso-contour banding during zoom
                    float mu_jitter = (fract(sin(dot(gl_FragCoord.xy + vec2(mu), vec2(127.1, 311.7))) * 43758.5453) - 0.5) * 0.15;
                    mu += mu_jitter;
                    // Map to color purely from low-frequency (bass) energy
                    float t = mu / float(u_max_iter);
                    float hue = fract(0.02 + 0.96 * clamp(u_bass, 0.0, 1.0));
                    float sat = 0.9;
                    // Keep structural contrast via iteration-based brightness only
                    float val = mix(0.6, 1.0, clamp(t, 0.0, 1.0));
                    vec3 col = hsv2rgb(vec3(hue, sat, val));

                    // Orbit-trap interior/exterior enhancement
                    float trap_c = exp(-8.0 * trap_min_circ);
                    float trap_x = exp(-8.0 * trap_min_cross);
                    float trap = clamp(max(trap_c, trap_x), 0.0, 1.0);
                    float trap_hue = fract(0.1 + 0.35 * trap + 0.2 * u_bass);
                    vec3 trap_col = hsv2rgb(vec3(trap_hue, 0.85, clamp(0.6 + 0.4 * trap, 0.6, 1.0)));
                    col = mix(col, trap_col, clamp(u_trap_mix, 0.0, 1.0));

                    // Bass adds warm glow (reduced)
                    float glow = clamp(u_bass * 0.3, 0.0, 0.35);
                    vec3 hdr = col + glow * vec3(0.12, 0.07, 0.02);

                    // Gentle exposure from energy and tone mapping to avoid blowouts
                    float expo = mix(0.85, 1.15, clamp(u_energy, 0.0, 1.0));
                    hdr *= expo;
                    vec3 mapped = hdr / (hdr + vec3(1.0)); // Reinhard
                    mapped = pow(mapped, vec3(1.0/2.2));   // gamma
                    // Add subtle screen-space dithering to hide gradient banding, more visible during zoom
                    float dither = (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453123) - 0.5) * (2.0/255.0);
                    mapped = clamp(mapped + vec3(dither), 0.0, 1.0);
                    f_color = vec4(mapped, 1.0);
                }
            """
        ),
        quad, 'in_position', 'in_uv'
    )

    prog = vao.program
    prog['u_res'].value = (W, H)
    set_uniform_if_present = None  # placeholder to satisfy linter in case of reordering

    def set_uniform_if_present(program, name, value):
        if name in program:
            program[name].value = value

    # Initial fractal params
    EDGE_CENTER = np.array([-0.745, 0.115], dtype=np.float32)
    center = EDGE_CENTER.copy()
    scale = 2.4
    iter_base = 150
    t0 = time.time()
    bass_s, mid_s, high_s = 0.0, 0.0, 0.0  # smoothed bands
    band_normalizer = Normalizer(mode="percentile", window=90, decay=0.02)
    band_smoother = Smoother(attack=0.6, release=0.3)

    # Audio responsiveness helpers
    agc_level = 1.0  # automatic gain control running level (EMA of 90th percentile)
    prev_spec = np.zeros(NFFT//2 + 1, dtype=np.float32)

    def smooth(prev: float, new: float, a: float = 0.6) -> float:
        return (1 - a) * prev + a * new

    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def move_towards(current: float, target: float, max_delta: float) -> float:
        if abs(target - current) <= max_delta:
            return target
        return current + math.copysign(max_delta, target - current)

    last_time = t0
    angle_accum = 0.0  # purely audio-driven phase accumulator
    omega = 0.0        # smoothed angular speed (rad/s)
    c_param = np.array([0.0, 0.0], dtype=np.float32)  # smoothed Julia parameter
    power_param = 2.0  # smoothed power
    rot_param = 0.0    # smoothed rotation
    prescale_param = 1.0
    warp_param = 0.0
    low_energy_accum = 0.0
    morph_param = 0.5  # 0..1 Mandelbrot->Julia
    ship_param = 0.0   # 0..1 Burning Ship blend
    # Audio activation gating
    noise_level = 0.0
    is_active = False
    silence_accum = 0.0
    activity_accum = 0.0
    # Visual drift params (continuous, subtle motion)
    swirl_param = 0.0
    shear_param = 0.0
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            # Keep resolution uniform in sync with framebuffer size
            fb_w, fb_h = glfw.get_framebuffer_size(window)
            if fb_w > 0 and fb_h > 0:
                set_uniform_if_present(prog, 'u_res', (fb_w, fb_h))

            # Read and analyze audio using project analysis utilities
            last_block = None
            while True:
                blk = cap.read(timeout=0.0)
                if blk is None:
                    break
                last_block = blk
            if last_block is not None:
                # Convert to mono
                if last_block.ndim == 2:
                    mono = last_block.astype(np.float32, copy=False).mean(axis=1)
                else:
                    mono = last_block.astype(np.float32, copy=False)
                # Compute spectrum frame and aggregate into 3 log-spaced bands
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
                # Smooth into fractal params
                bass_s = smooth(bass_s, b)
                mid_s  = smooth(mid_s,  m)
                high_s = smooth(high_s, h)

            # Drive fractal params
            now = time.time()
            dt = now - last_time
            last_time = now

            # Zoom in/out with bass; pan with mids; add iteration with overall energy
            energy = (bass_s + mid_s + high_s) / 3.0

            # Track sustained low energy to auto-reset parameters to safe defaults
            if energy < 0.35:
                low_energy_accum = min(low_energy_accum + dt, 1.0)
            else:
                low_energy_accum = max(low_energy_accum - 2.0 * dt, 0.0)

            # Active/inactive gating with hysteresis
            if energy > 0.28:
                activity_accum = min(activity_accum + dt, 1.0)
            else:
                activity_accum = max(activity_accum - dt, 0.0)
            if energy < 0.18:
                silence_accum = min(silence_accum + dt, 1.0)
            else:
                silence_accum = max(silence_accum - dt, 0.0)
            if (not is_active) and activity_accum > 0.2:
                is_active = True
            if is_active and silence_accum > 0.6:
                is_active = False

            # No time-based drift: all motion driven by audio only

            # Fixed zoom level (no audio-driven zoom)
            scale = 2.4

            # Audio-driven Julia parameter c (smoothed, narrow range) and power (smoothed)
            # No baseline rotation: only audio drives omega
            target_omega = (0.13 * float(np.clip(mid_s, 0.0, 1.0)) if is_active else 0.0)
            if is_active:
                omega = lerp(omega, target_omega, min(1.5 * dt, 0.25))
                angle_accum += omega * dt
            else:
                omega = 0.0
            r_c = float(np.clip(0.08 + 0.16 * float(np.clip(bass_s, 0.0, 1.0)), 0.08, 0.24))
            c_target = np.array([r_c * math.cos(angle_accum), r_c * math.sin(angle_accum)], dtype=np.float32)
            # If inactive or sustained low energy, steer c back to a gentle default radius
            if (not is_active) or (low_energy_accum > 0.4):
                c_target = np.array([0.12 * math.cos(angle_accum), 0.12 * math.sin(angle_accum)], dtype=np.float32)
            c_param = (1.0 - min(3.0 * dt, 0.25)) * c_param + min(3.0 * dt, 0.25) * c_target

            # Morph parameters driven only by audio
            morph_target = float(np.clip(0.2 + 1.8 * (float(mid_s) - 0.25), 0.0, 1.0))
            ship_target = float(np.clip(2.2 * (float(high_s) - 0.4), 0.0, 1.0))
            if (not is_active) or (low_energy_accum > 0.4):
                morph_target = 0.35
                ship_target = 0.0
            morph_param = lerp(morph_param, morph_target, min(2.5 * dt, 0.5))
            ship_param = lerp(ship_param, ship_target, min(2.0 * dt, 0.4))

            # Audio-driven pre-transform parameters
            rot_target = 0.0 + 1.0 * float(np.clip(mid_s - 0.25, 0.0, 1.0))
            if is_active:
                rot_param = lerp(rot_param, rot_target, min(2.5 * dt, 0.4))
            else:
                rot_param = 0.0
            # Disable pre-scale zoom
            prescale_param = 1.0
            # Disable outward quadratic warp to avoid radial stretching
            warp_param = 0.0

            # Shear (from mids) to bend arms; swirl disabled; no drift
            swirl_target = 0.0
            shear_target = 0.10 * float(np.clip(mid_s - 0.2, 0.0, 1.0))
            if is_active:
                shear_param = lerp(shear_param, shear_target, min(3.0 * dt, 0.5))
            else:
                shear_param = 0.0
            swirl_param = 0.0

            # Keep the view anchored on a known interesting edge region
            center += (EDGE_CENTER - center) * min(0.45 * dt, 0.45)
            # Constrain center to a small radius around the edge center
            delta_c = center - EDGE_CENTER
            rad = float(np.linalg.norm(delta_c))
            max_rad = 0.18
            if rad > max_rad and rad > 1e-6:
                center = EDGE_CENTER + (delta_c * (max_rad / rad))

            # Iterations based only on energy (reverted for performance)
            u_iters = int(iter_base + 120 * np.clip(energy, 0.0, 1.5))
            u_iters = int(np.clip(u_iters, 100, 360))

            # Upload uniforms (guarded in case of optimization removing unused uniforms)
            set_uniform_if_present(prog, 'u_time', now - t0)
            set_uniform_if_present(prog, 'u_center', tuple(center.tolist()))
            set_uniform_if_present(prog, 'u_scale', float(scale))
            set_uniform_if_present(prog, 'u_max_iter', int(u_iters))
            set_uniform_if_present(prog, 'u_bass', float(bass_s))
            set_uniform_if_present(prog, 'u_mid', float(mid_s))
            set_uniform_if_present(prog, 'u_high', float(high_s))
            set_uniform_if_present(prog, 'u_energy', float(np.clip(energy, 0.0, 1.5)))
            set_uniform_if_present(prog, 'u_view_uv_center', (0.5, 0.5))
            # Bailout radius adjusted for larger visual window
            set_uniform_if_present(prog, 'u_bail', float(8.0))
            # Julia parameters (smoothed)
            set_uniform_if_present(prog, 'u_c', (float(c_param[0]), float(c_param[1])))
            target_power = float(np.clip(2.0 + 0.3 * (float(np.clip(high_s, 0.0, 1.0)) - 0.3), 1.8, 2.3))
            power_param = lerp(power_param, target_power, min(2.0 * dt, 0.3))
            set_uniform_if_present(prog, 'u_power', float(power_param))
            # Pre-transform uniforms
            set_uniform_if_present(prog, 'u_rot', float(rot_param))
            set_uniform_if_present(prog, 'u_pre_scale', float(prescale_param))
            set_uniform_if_present(prog, 'u_warp', float(warp_param))
            # Morph uniforms
            set_uniform_if_present(prog, 'u_morph', float(morph_param))
            set_uniform_if_present(prog, 'u_ship', float(ship_param))
            set_uniform_if_present(prog, 'u_swirl', float(swirl_param))
            set_uniform_if_present(prog, 'u_shear', float(shear_param))
            # Orbit-trap uniforms (audio-driven)
            trap_r = float(np.clip(0.32 + 0.25 * (float(bass_s) - 0.4), 0.1, 0.7))
            trap_mix = float(np.clip(0.15 + 0.6 * float(energy), 0.1, 1))
            trap_rot = float((0.5 * angle_accum) + 1.2 * float(np.clip(high_s - 0.3, 0.0, 1.0)))
            if not is_active:
                trap_mix *= 0.4
            set_uniform_if_present(prog, 'u_trap_r', trap_r)
            set_uniform_if_present(prog, 'u_trap_mix', trap_mix)
            set_uniform_if_present(prog, 'u_trap_rot', trap_rot)

            # Draw
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            vao.render(moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(window)
    finally:
        # Cleanup
        try:
            cap.stop()
        except Exception:
            pass
        glfw.terminate()


if __name__ == '__main__':
    main()


