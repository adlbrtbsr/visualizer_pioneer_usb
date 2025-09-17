from __future__ import annotations

from typing import Optional, Callable

try:
    import imgui
    from imgui.integrations.glfw import GlfwRenderer
except Exception:
    imgui = None
    GlfwRenderer = None

from .settings import VisualIntensitySettings


class ImGuiOverlay:
    def __init__(self, window, width: int, height: int) -> None:
        self._impl: Optional[GlfwRenderer] = None
        self._visible: bool = True
        if imgui is None or GlfwRenderer is None:
            return
        try:
            imgui.create_context()
            self._impl = GlfwRenderer(window)
            io = imgui.get_io()
            io.display_size = (width, height)
        except Exception:
            self._impl = None

    @property
    def available(self) -> bool:
        return self._impl is not None and imgui is not None

    def set_size(self, width: int, height: int) -> None:
        if not self.available:
            return
        try:
            io = imgui.get_io()
            io.display_size = (width, height)
        except Exception:
            pass

    def toggle_visibility(self) -> None:
        if not self.available:
            return
        self._visible = not self._visible

    def draw(self, settings: VisualIntensitySettings, pending_save: bool, save_cb: Callable[[VisualIntensitySettings], None]) -> None:
        if not self.available or not self._visible:
            return
        try:
            self._impl.process_inputs()
            imgui.new_frame()
            if imgui.begin("Visual Intensity", True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
                changed = False
                _c, settings.master = imgui.slider_float("Master", float(settings.master), 0.0, 2.0); changed = changed or _c
                _c, settings.exposure = imgui.slider_float("Exposure", float(settings.exposure), 0.6, 1.6); changed = changed or _c
                _c, settings.contrast = imgui.slider_float("Contrast", float(settings.contrast), 0.5, 2.0); changed = changed or _c
                _c, settings.palette_id = imgui.slider_int("Palette", int(settings.palette_id), 0, 3); changed = changed or _c
                _c, settings.hue_offset = imgui.slider_float("Hue Offset", float(settings.hue_offset), 0.0, 1.0); changed = changed or _c
                _c, settings.palette_saturation = imgui.slider_float("Palette Saturation", float(settings.palette_saturation), 0.0, 1.0); changed = changed or _c
                _c, settings.fractal_type = imgui.slider_int("Fractal Type", int(settings.fractal_type), 0, 4); changed = changed or _c
                _c, settings.motion_gain = imgui.slider_float("Motion Gain", float(settings.motion_gain), 0.0, 3.0); changed = changed or _c
                _c, settings.iteration_gain = imgui.slider_float("Iteration Gain", float(settings.iteration_gain), 0.5, 2.0); changed = changed or _c
                _c, settings.trap_mix_gain = imgui.slider_float("Trap Mix Gain", float(settings.trap_mix_gain), 0.0, 2.0); changed = changed or _c
                _c, settings.glow_gain = imgui.slider_float("Glow Gain", float(settings.glow_gain), 0.0, 2.0); changed = changed or _c
                _c, settings.scale = imgui.slider_float("Zoom (Scale)", float(settings.scale), 0.4, 4.0); changed = changed or _c
                _c, settings.iterations_base = imgui.slider_float("Iterations Base", float(settings.iterations_base), 5.0, 420.0); changed = changed or _c
                _c, settings.bailout_radius = imgui.slider_float("Bailout Radius", float(settings.bailout_radius), 2.0, 16.0); changed = changed or _c
                _c, settings.morph_gain = imgui.slider_float("Morph Gain", float(settings.morph_gain), 0.0, 2.0); changed = changed or _c
                _c, settings.ship_gain = imgui.slider_float("Ship Gain", float(settings.ship_gain), 0.0, 2.0); changed = changed or _c
                _c, settings.trap_radius_scale = imgui.slider_float("Trap Radius Scale", float(settings.trap_radius_scale), 0.5, 2.0); changed = changed or _c
                _c, settings.bend_gain = imgui.slider_float("Bend Gain", float(settings.bend_gain), 0.0, 3.0); changed = changed or _c
                _c, settings.view_angle_deg = imgui.slider_float("View Angle (deg)", float(settings.view_angle_deg), -180.0, 180.0); changed = changed or _c
                _c, settings.view_center_x = imgui.slider_float("View Center X", float(settings.view_center_x), 0.0, 1.0); changed = changed or _c
                _c, settings.view_center_y = imgui.slider_float("View Center Y", float(settings.view_center_y), 0.0, 1.0); changed = changed or _c

                if imgui.button("Reset"):
                    settings.master = 1.0
                    settings.exposure = 1.0
                    settings.contrast = 1.0
                    settings.motion_gain = 1.0
                    settings.iteration_gain = 1.0
                    settings.trap_mix_gain = 1.0
                    settings.glow_gain = 1.0
                    settings.palette_id = 0
                    settings.hue_offset = 0.0
                    settings.palette_saturation = 0.9
                    settings.fractal_type = 0
                    settings.scale = 2.4
                    settings.iterations_base = 150.0
                    settings.bailout_radius = 8.0
                    settings.morph_gain = 1.0
                    settings.ship_gain = 1.0
                    settings.trap_radius_scale = 1.0
                    settings.bend_gain = 1.0
                    settings.view_angle_deg = 0.0
                    settings.view_center_x = 0.5
                    settings.view_center_y = 0.5
                imgui.same_line()
                if imgui.button("Save (Ctrl+S)") or pending_save:
                    try:
                        save_cb(settings)
                    except Exception:
                        pass
            imgui.end()
            imgui.render()
            self._impl.render(imgui.get_draw_data())
        except Exception:
            pass

    def shutdown(self) -> None:
        try:
            if self._impl is not None:
                self._impl.shutdown()
        except Exception:
            pass


