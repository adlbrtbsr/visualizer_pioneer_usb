import threading
from typing import Callable

try:
    import tkinter as tk
except Exception:
    tk = None

from .settings import VisualIntensitySettings


class TkControlPanel:
    def __init__(self, settings: VisualIntensitySettings, save_callback: Callable[[VisualIntensitySettings], None]) -> None:
        self.settings = settings
        self.save_callback = save_callback
        self._thread = None
        self._root = None
        self._vars = {}
        self._running = False

    def start(self) -> None:
        if tk is None:
            return
        if self._thread is not None:
            return
        self._running = True

        def _run():
            try:
                self._root = tk.Tk()
                self._root.title("Fractal Controls")
                self._root.geometry("360x280")
                self._root.protocol("WM_DELETE_WINDOW", self._on_close)

                def add_slider(row, label, key, from_, to_, resolution=0.01):
                    tk.Label(self._root, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
                    var = tk.DoubleVar(value=float(getattr(self.settings, key)))
                    self._vars[key] = var

                    def on_change(val=None, k=key, v=var):
                        try:
                            setattr(self.settings, k, float(v.get()))
                        except Exception:
                            pass

                    scale = tk.Scale(
                        self._root,
                        from_=from_,
                        to=to_,
                        resolution=resolution,
                        orient=tk.HORIZONTAL,
                        showvalue=True,
                        length=220,
                        command=lambda _=None: on_change(),
                    )
                    scale.set(float(getattr(self.settings, key)))
                    scale.grid(row=row, column=1, sticky="ew", padx=6)
                    self._vars[key + "_scale"] = scale

                add_slider(0, "Master", "master", 0.0, 2.0, 0.01)
                add_slider(1, "Exposure", "exposure", 0.6, 1.6, 0.01)
                add_slider(1, "Contrast", "contrast", 0.5, 2.0, 0.01)
                add_slider(2, "Palette (0..3)", "palette_id", 0, 3, 1.0)
                add_slider(3, "Hue Offset", "hue_offset", 0.0, 1.0, 0.01)
                add_slider(4, "Palette Saturation", "palette_saturation", 0.0, 1.0, 0.01)
                add_slider(5, "Fractal Type (0..4)", "fractal_type", 0, 4, 1.0)
                add_slider(6, "Motion Gain", "motion_gain", 0.0, 3.0, 0.01)
                add_slider(7, "Iteration Gain", "iteration_gain", 0.5, 2.0, 0.01)
                add_slider(8, "Trap Mix Gain", "trap_mix_gain", 0.0, 2.0, 0.01)
                add_slider(9, "Glow Gain", "glow_gain", 0.0, 2.0, 0.01)
                add_slider(10, "Zoom (Scale)", "scale", 0.4, 4.0, 0.01)
                add_slider(11, "Iterations Base", "iterations_base", 60.0, 300.0, 1.0)
                add_slider(12, "Bailout Radius", "bailout_radius", 2.0, 16.0, 0.1)
                add_slider(13, "Morph Gain", "morph_gain", 0.0, 2.0, 0.01)
                add_slider(14, "Ship Gain", "ship_gain", 0.0, 2.0, 0.01)
                add_slider(15, "Trap Radius Scale", "trap_radius_scale", 0.5, 2.0, 0.01)
                add_slider(16, "Bend Gain", "bend_gain", 0.0, 3.0, 0.01)
                add_slider(17, "View Angle (deg)", "view_angle_deg", -180.0, 180.0, 0.5)
                add_slider(18, "View Center X", "view_center_x", 0.0, 1.0, 0.005)
                add_slider(19, "View Center Y", "view_center_y", 0.0, 1.0, 0.005)

                btn_frame = tk.Frame(self._root)
                btn_frame.grid(row=20, column=0, columnspan=2, pady=8)

                def do_reset():
                    self.settings.master = 1.0
                    self.settings.exposure = 1.0
                    self.settings.contrast = 1.0
                    self.settings.motion_gain = 1.0
                    self.settings.iteration_gain = 1.0
                    self.settings.trap_mix_gain = 1.0
                    self.settings.glow_gain = 1.0
                    self.settings.palette_id = 0
                    self.settings.hue_offset = 0.0
                    self.settings.palette_saturation = 0.9
                    self.settings.fractal_type = 0
                    self.settings.scale = 2.4
                    self.settings.iterations_base = 150.0
                    self.settings.bailout_radius = 8.0
                    self.settings.morph_gain = 1.0
                    self.settings.ship_gain = 1.0
                    self.settings.trap_radius_scale = 1.0
                    self.settings.bend_gain = 1.0
                    self.settings.view_angle_deg = 0.0
                    self.settings.view_center_x = 0.5
                    self.settings.view_center_y = 0.5
                    for k in [
                        "master",
                        "exposure",
                        "contrast",
                        "palette_id",
                        "hue_offset",
                        "palette_saturation",
                        "fractal_type",
                        "motion_gain",
                        "iteration_gain",
                        "trap_mix_gain",
                        "glow_gain",
                        "scale",
                        "iterations_base",
                        "bailout_radius",
                        "morph_gain",
                        "ship_gain",
                        "trap_radius_scale",
                        "bend_gain",
                        "view_angle_deg",
                        "view_center_x",
                        "view_center_y",
                    ]:
                        try:
                            self._vars[k + "_scale"].set(float(getattr(self.settings, k)))
                        except Exception:
                            pass

                def do_save():
                    try:
                        self.save_callback(self.settings)
                    except Exception:
                        pass

                tk.Button(btn_frame, text="Reset", command=do_reset).pack(side=tk.LEFT, padx=6)
                tk.Button(btn_frame, text="Save", command=do_save).pack(side=tk.LEFT, padx=6)

                self._root.mainloop()
            finally:
                self._running = False
                self._root = None

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _on_close(self):
        try:
            if self._root is not None:
                self._root.destroy()
        except Exception:
            pass
        self._running = False

    def close(self) -> None:
        try:
            if self._root is not None:
                self._root.after(0, self._root.destroy)
        except Exception:
            pass


