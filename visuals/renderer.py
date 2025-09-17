from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import moderngl

from .settings import VisualIntensitySettings


@dataclass
class FractalState:
    center: np.ndarray
    scale: float
    iter_base: int
    bass_s: float = 0.0
    mid_s: float = 0.0
    high_s: float = 0.0
    t0: float = 0.0
    angle_accum: float = 0.0
    omega: float = 0.0
    c_param: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=np.float32))
    power_param: float = 2.0
    rot_param: float = 0.0
    prescale_param: float = 1.0
    warp_param: float = 0.0
    morph_param: float = 0.5
    ship_param: float = 0.0
    swirl_param: float = 0.0
    shear_param: float = 0.0
    bend_param: float = 0.0
    is_active: bool = False
    low_energy_accum: float = 0.0
    silence_accum: float = 0.0
    activity_accum: float = 0.0


class FractalRenderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        self.ctx = ctx
        self.width = width
        self.height = height
        # Ensure viewport matches initial framebuffer size
        try:
            self.ctx.viewport = (0, 0, int(self.width), int(self.height))
        except Exception:
            pass
        self.quad = ctx.buffer(np.array([
            -1, -1,  0, 0,
             1, -1,  1, 0,
            -1,  1,  0, 1,
             1,  1,  1, 1,
        ], dtype='f4').tobytes())
        self.program = ctx.program(vertex_shader=self._vs_src(), fragment_shader=self._fs_src())
        self.vao = ctx.simple_vertex_array(self.program, self.quad, 'in_position', 'in_uv')
        self._edge_center = np.array([-0.745, 0.115], dtype=np.float32)
        self.state = FractalState(center=self._edge_center.copy(), scale=2.4, iter_base=150, t0=time.time())
        if 'u_res' in self.program:
            self.program['u_res'].value = (self.width, self.height)

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        # Update viewport and resolution uniform
        try:
            self.ctx.viewport = (0, 0, int(self.width), int(self.height))
        except Exception:
            pass
        if 'u_res' in self.program:
            self.program['u_res'].value = (self.width, self.height)

    def update_audio(self, bass: float, mid: float, high: float) -> None:
        s = self.state
        s.bass_s = float(bass)
        s.mid_s = float(mid)
        s.high_s = float(high)

    def step(self, settings: VisualIntensitySettings, dt: float) -> None:
        s = self.state
        energy = (s.bass_s + s.mid_s + s.high_s) / 3.0

        if energy < 0.35:
            s.low_energy_accum = min(s.low_energy_accum + dt, 1.0)
        else:
            s.low_energy_accum = max(s.low_energy_accum - 2.0 * dt, 0.0)

        if energy > 0.28:
            s.activity_accum = min(s.activity_accum + dt, 1.0)
        else:
            s.activity_accum = max(s.activity_accum - dt, 0.0)
        if energy < 0.18:
            s.silence_accum = min(s.silence_accum + dt, 1.0)
        else:
            s.silence_accum = max(s.silence_accum - dt, 0.0)
        if (not s.is_active) and s.activity_accum > 0.2:
            s.is_active = True
        if s.is_active and s.silence_accum > 0.6:
            s.is_active = False

        s.scale = float(getattr(settings, 'scale', 2.4))

        target_omega = (0.13 * float(np.clip(s.mid_s, 0.0, 1.0)) if s.is_active else 0.0)
        target_omega *= float(settings.motion_gain) * float(settings.master)
        if s.is_active:
            s.omega = self._lerp(s.omega, target_omega, min(1.5 * dt, 0.25))
            s.angle_accum += s.omega * dt
        else:
            s.omega = 0.0

        r_c = float(np.clip(0.08 + 0.16 * float(np.clip(s.bass_s, 0.0, 1.0)), 0.08, 0.24))
        c_target = np.array([r_c * math.cos(s.angle_accum), r_c * math.sin(s.angle_accum)], dtype=np.float32)
        if (not s.is_active) or (s.low_energy_accum > 0.4):
            c_target = np.array([0.12 * math.cos(s.angle_accum), 0.12 * math.sin(s.angle_accum)], dtype=np.float32)
        s.c_param = (1.0 - min(3.0 * dt, 0.25)) * s.c_param + min(3.0 * dt, 0.25) * c_target

        morph_target = float(np.clip(0.2 + 1.8 * (float(s.mid_s) - 0.25), 0.0, 1.0))
        ship_target = float(np.clip(2.2 * (float(s.high_s) - 0.4), 0.0, 1.0))
        morph_target *= float(getattr(settings, 'morph_gain', 1.0)) * float(getattr(settings, 'master', 1.0))
        ship_target *= float(getattr(settings, 'ship_gain', 1.0)) * float(getattr(settings, 'master', 1.0))
        if (not s.is_active) or (s.low_energy_accum > 0.4):
            morph_target = 0.35
            ship_target = 0.0
        s.morph_param = self._lerp(s.morph_param, morph_target, min(2.5 * dt, 0.5))
        s.ship_param = self._lerp(s.ship_param, ship_target, min(2.0 * dt, 0.4))

        rot_target = 0.0 + 1.0 * float(np.clip(s.mid_s - 0.25, 0.0, 1.0))
        rot_target *= float(settings.motion_gain) * float(settings.master)
        if s.is_active:
            s.rot_param = self._lerp(s.rot_param, rot_target, min(2.5 * dt, 0.4))
        else:
            s.rot_param = 0.0
        s.prescale_param = 1.0
        # Map overall energy to bend/warp; smooth for stability
        bend_target = float(np.clip((energy - 0.15) * 1.8, 0.0, 1.0))
        bend_target *= float(getattr(settings, 'bend_gain', 1.0)) * float(getattr(settings, 'master', 1.0))
        s.bend_param = self._lerp(s.bend_param, bend_target, min(2.0 * dt, 0.35))
        s.warp_param = float(np.clip(s.bend_param * 0.65, 0.0, 1.0))

        shear_target = 0.10 * float(np.clip(s.mid_s - 0.2, 0.0, 1.0))
        shear_target *= float(settings.motion_gain) * float(settings.master)
        if s.is_active:
            s.shear_param = self._lerp(s.shear_param, shear_target, min(3.0 * dt, 0.5))
        else:
            s.shear_param = 0.0
        s.swirl_param = 0.0

        s.center += (self._edge_center - s.center) * min(0.45 * dt, 0.45)
        delta_c = s.center - self._edge_center
        rad = float(np.linalg.norm(delta_c))
        max_rad = 0.18
        if rad > max_rad and rad > 1e-6:
            s.center = self._edge_center + (delta_c * (max_rad / rad))

        base_iters = float(getattr(settings, 'iterations_base', s.iter_base))
        u_iters = int(base_iters + 60 * np.clip(energy, 0.0, 1.5))
        u_iters = int(u_iters * float(settings.iteration_gain) * float(settings.master))
        u_iters = int(np.clip(u_iters, 5, 420))
        self._upload_uniforms(settings, u_iters)

    def draw(self) -> None:
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def _upload_uniforms(self, settings: VisualIntensitySettings, u_iters: int) -> None:
        s = self.state
        prog = self.program
        now = time.time()
        self._set_uniform(prog, 'u_time', now - s.t0)
        self._set_uniform(prog, 'u_center', tuple(s.center.tolist()))
        self._set_uniform(prog, 'u_scale', float(s.scale))
        self._set_uniform(prog, 'u_max_iter', int(u_iters))
        bass_scaled = float(s.bass_s) * float(settings.glow_gain) * float(settings.master)
        self._set_uniform(prog, 'u_bass', float(bass_scaled))
        self._set_uniform(prog, 'u_mid', float(s.mid_s))
        self._set_uniform(prog, 'u_high', float(s.high_s))
        energy = float(np.clip(((s.bass_s + s.mid_s + s.high_s) / 3.0) * float(settings.exposure) * float(settings.master), 0.0, 1.5))
        self._set_uniform(prog, 'u_energy', energy)
        cx = float(np.clip(getattr(settings, 'view_center_x', 0.5), 0.0, 1.0))
        cy = float(np.clip(getattr(settings, 'view_center_y', 0.5), 0.0, 1.0))
        self._set_uniform(prog, 'u_view_uv_center', (cx, cy))
        self._set_uniform(prog, 'u_bail', float(getattr(settings, 'bailout_radius', 8.0)))
        self._set_uniform(prog, 'u_contrast', float(getattr(settings, 'contrast', 1.0)))
        self._set_uniform(prog, 'u_palette_id', int(getattr(settings, 'palette_id', 0)))
        self._set_uniform(prog, 'u_hue_offset', float(getattr(settings, 'hue_offset', 0.0)))
        self._set_uniform(prog, 'u_palette_sat', float(getattr(settings, 'palette_saturation', 0.9)))
        self._set_uniform(prog, 'u_fractal_type', int(getattr(settings, 'fractal_type', 0)))
        self._set_uniform(prog, 'u_c', (float(s.c_param[0]), float(s.c_param[1])))
        target_power = float(np.clip(2.0 + 0.3 * (float(np.clip(s.high_s, 0.0, 1.0)) - 0.3), 1.8, 2.3))
        s.power_param = self._lerp(s.power_param, target_power, 0.3)
        self._set_uniform(prog, 'u_power', float(s.power_param))
        rot_user = float(np.deg2rad(float(getattr(settings, 'view_angle_deg', 0.0))))
        self._set_uniform(prog, 'u_rot', float(s.rot_param + rot_user))
        self._set_uniform(prog, 'u_pre_scale', float(s.prescale_param))
        self._set_uniform(prog, 'u_warp', float(s.warp_param))
        # Provide direct bend uniform if future shader wants separate control
        self._set_uniform(prog, 'u_bend', float(s.bend_param))
        self._set_uniform(prog, 'u_morph', float(s.morph_param))
        self._set_uniform(prog, 'u_ship', float(s.ship_param))
        self._set_uniform(prog, 'u_swirl', float(s.swirl_param))
        self._set_uniform(prog, 'u_shear', float(s.shear_param))

        trap_r = float(np.clip(0.32 + 0.25 * (float(s.bass_s) - 0.4), 0.1, 0.7))
        trap_r *= float(getattr(settings, 'trap_radius_scale', 1.0))
        trap_mix = float(np.clip(0.15 + 0.6 * float(((s.bass_s + s.mid_s + s.high_s) / 3.0)), 0.1, 1))
        trap_rot = float((0.5 * s.angle_accum) + 1.2 * float(np.clip(s.high_s - 0.3, 0.0, 1.0)))
        if not s.is_active:
            trap_mix *= 0.4
        trap_mix = float(np.clip(trap_mix * float(settings.trap_mix_gain) * float(settings.master), 0.0, 1.0))
        self._set_uniform(prog, 'u_trap_r', trap_r)
        self._set_uniform(prog, 'u_trap_mix', trap_mix)
        self._set_uniform(prog, 'u_trap_rot', trap_rot)

    def _set_uniform(self, program, name, value):
        if name in program:
            program[name].value = value

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    @staticmethod
    def _vs_src() -> str:
        return (
            """
            #version 330
            in vec2 in_position;
            in vec2 in_uv;
            out vec2 v_uv;
            void main(){
                v_uv = in_uv;
                gl_Position = vec4(in_position, 0.0, 1.0);
            }
            """
        )

    @staticmethod
    def _fs_src() -> str:
        return (
            """
            #version 330
            in vec2 v_uv;
            out vec4 f_color;

            uniform float u_time;
            uniform vec2  u_res;
            uniform vec2  u_center;
            uniform float u_scale;
            uniform int   u_max_iter;
            uniform float u_bass;
            uniform float u_mid;
            uniform float u_high;
            uniform vec2  u_c;
            uniform float u_power;
            uniform float u_rot;
            uniform float u_pre_scale;
            uniform float u_warp;
            uniform float u_bend;
            uniform float u_energy;
            uniform float u_morph;
            uniform float u_ship;
            uniform float u_swirl;
            uniform float u_shear;
            uniform vec2  u_view_uv_center;
            uniform float u_trap_r;
            uniform float u_trap_mix;
            uniform float u_trap_rot;
            uniform float u_bail;
            uniform float u_contrast;
            uniform int   u_palette_id;
            uniform float u_hue_offset;
            uniform float u_palette_sat;
            uniform int   u_fractal_type;

            vec3 hsv2rgb(vec3 c) {
                vec3 p = abs(fract(c.xxx + vec3(0.0, 0.6666667, 0.3333333)) * 6.0 - 3.0);
                vec3 rgb = c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
                return rgb;
            }

            void main(){
                float aspect = u_res.x / u_res.y;
                vec2 p0 = (v_uv - u_view_uv_center) * vec2(aspect, 1.0) * u_scale;
                float cs = cos(u_rot), sn = sin(u_rot);
                mat2 R = mat2(cs, -sn, sn, cs);
                mat2 Sh = mat2(1.0, u_shear, u_shear, 1.0);
                // Bend is expressed via warp quadratic term and shear mixing
                vec2 p = R * (Sh * (p0 * u_pre_scale));
                vec2 p2 = vec2(p.x*p.x - p.y*p.y, 2.0*p.x*p.y);
                p += (u_warp + 0.4 * u_bend) * p2;
                p += u_center;

                vec2 c_p = mix(p, u_c, clamp(u_morph, 0.0, 1.0));
                vec2 z = p;
                int i;
                float trap_min_circ = 1e9;
                float trap_min_cross = 1e9;
                float cst = cos(u_trap_rot), snt = sin(u_trap_rot);
                mat2 RT = mat2(cst, -snt, snt, cst);
                for(i=0; i<u_max_iter && dot(z,z) <= (u_bail * u_bail); i++){
                    vec2 z_iter = z;
                    if (u_fractal_type == 0) {
                        vec2 zb = mix(z, vec2(abs(z.x), abs(z.y)), clamp(u_ship, 0.0, 1.0));
                        float r = length(zb);
                        float theta = atan(zb.y, zb.x);
                        float rp = pow(r, u_power);
                        float ang = u_power * theta;
                        z_iter = rp * vec2(cos(ang), sin(ang));
                        z = z_iter + c_p;
                    } else if (u_fractal_type == 1) {
                        float r = length(z);
                        float theta = atan(z.y, z.x);
                        float rp = pow(r, u_power);
                        float ang = u_power * theta;
                        z = rp * vec2(cos(ang), sin(ang)) + p;
                    } else if (u_fractal_type == 2) {
                        float r = length(z);
                        float theta = atan(z.y, z.x);
                        float rp = pow(r, u_power);
                        float ang = u_power * theta;
                        z = rp * vec2(cos(ang), sin(ang)) + u_c;
                    } else if (u_fractal_type == 3) {
                        vec2 zb = vec2(abs(z.x), abs(z.y));
                        float r = length(zb);
                        float theta = atan(zb.y, zb.x);
                        float rp = pow(r, u_power);
                        float ang = u_power * theta;
                        z = rp * vec2(cos(ang), sin(ang)) + p;
                    } else {
                        vec2 zc = vec2(z.x, -z.y);
                        float r = length(zc);
                        float theta = atan(zc.y, zc.x);
                        float rp = pow(r, u_power);
                        float ang = u_power * theta;
                        z = rp * vec2(cos(ang), sin(ang)) + p;
                    }

                    vec2 zt = RT * z;
                    float r_zt = length(zt);
                    trap_min_circ = min(trap_min_circ, abs(r_zt - u_trap_r));
                    trap_min_cross = min(trap_min_cross, min(abs(zt.x), abs(zt.y)));
                }

                float mu = float(i);
                if (i < u_max_iter) {
                    float r2 = dot(z,z);
                    float log_zn = 0.5 * log(r2);
                    float nu = log(log_zn / log(max(u_bail, 1.01))) / max(log(u_power), 1e-4);
                    mu = float(i) + 1.0 - nu;
                }

                float mu_jitter = (fract(sin(dot(gl_FragCoord.xy + vec2(mu), vec2(127.1, 311.7))) * 43758.5453) - 0.5) * 0.15;
                mu += mu_jitter;
                float t = mu / float(u_max_iter);
                float hue = fract(u_hue_offset + 0.02 + 0.96 * clamp(u_bass, 0.0, 1.0));
                float sat = clamp(u_palette_sat, 0.0, 1.0);
                float val = mix(0.6, 1.0, clamp(t, 0.0, 1.0));
                vec3 col0 = hsv2rgb(vec3(hue, sat, val));
                vec3 col1 = mix(vec3(0.9, 0.4, 0.05), vec3(1.0, 0.1, 0.0), pow(clamp(t,0.0,1.0), 0.5));
                vec3 col2 = mix(vec3(0.1, 0.8, 0.9), vec3(0.0, 0.2, 0.8), clamp(t,0.0,1.0));
                vec3 col3 = hsv2rgb(vec3(fract(hue + 0.25 * clamp(u_high, 0.0, 1.0)), 1.0, mix(0.7, 1.1, clamp(u_mid, 0.0, 1.0))));
                vec3 col = (u_palette_id == 0) ? col0 : (u_palette_id == 1 ? col1 : (u_palette_id == 2 ? col2 : col3));

                float trap_c = exp(-8.0 * trap_min_circ);
                float trap_x = exp(-8.0 * trap_min_cross);
                float trap = clamp(max(trap_c, trap_x), 0.0, 1.0);
                float trap_hue = fract(u_hue_offset + 0.1 + 0.35 * trap + 0.2 * u_bass);
                vec3 trap_col = hsv2rgb(vec3(trap_hue, clamp(0.6 + 0.25 * u_palette_sat, 0.0, 1.0), clamp(0.6 + 0.4 * trap, 0.6, 1.0)));
                col = mix(col, trap_col, clamp(u_trap_mix, 0.0, 1.0));

                float glow = clamp(u_bass * 0.3, 0.0, 0.35);
                vec3 hdr = col + glow * vec3(0.12, 0.07, 0.02);

                float expo = mix(0.85, 1.15, clamp(u_energy, 0.0, 1.0));
                hdr *= expo;
                vec3 mapped = hdr / (hdr + vec3(1.0));
                mapped = pow(mapped, vec3(1.0/2.2));
                float c = clamp(u_contrast, 0.2, 3.0);
                mapped = clamp((mapped - vec3(0.5)) * c + vec3(0.5), 0.0, 1.0);
                float dither = (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453123) - 0.5) * (2.0/255.0);
                mapped = clamp(mapped + vec3(dither), 0.0, 1.0);
                f_color = vec4(mapped, 1.0);
            }
            """
        )


