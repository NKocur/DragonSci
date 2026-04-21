"""
Interactive benchmark for dragonsci.

- Embeds a Scatter3D in a Tkinter window
- Sliders to control: N points, point size, colormap, distribution shape
- New random cloud every second (configurable)
- Live stats panel: FPS, frame-time, upload time, point count
"""

from __future__ import annotations

import time
import tkinter as tk
from collections import deque

import numpy as np

from dragonsci import Scatter3D, link_cameras, unlink_cameras

# ── Config ────────────────────────────────────────────────────────────────────

COLORMAPS = ["viridis", "plasma", "inferno", "magma", "coolwarm", "hot", "gray",
             "turbo", "cividis", "blues", "greens", "reds"]
SHAPES    = ["uniform", "gaussian", "torus", "helix", "sphere"]
FPS_WINDOW = 60  # frames to average for FPS display


# ── Point cloud generators ────────────────────────────────────────────────────

def make_cloud(shape: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions (N,3), scalars (N,)) float32."""
    if shape == "gaussian":
        pts = rng.standard_normal((n, 3)).astype(np.float32)
        scalars = np.linalg.norm(pts, axis=1).astype(np.float32)

    elif shape == "torus":
        theta = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        phi   = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        r     = (2.0 + np.cos(phi)).astype(np.float32)
        pts   = np.stack([r * np.cos(theta), r * np.sin(theta), np.sin(phi)], axis=1)
        scalars = r

    elif shape == "helix":
        t      = np.linspace(0, 8 * np.pi, n, dtype=np.float32)
        noise  = rng.standard_normal((n, 3)).astype(np.float32) * 0.1
        pts    = np.stack([np.cos(t), np.sin(t), t / (8 * np.pi)], axis=1) + noise
        scalars = t / t.max()

    elif shape == "sphere":
        theta  = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        phi    = np.arccos(rng.uniform(-1, 1, n)).astype(np.float32)
        r      = rng.uniform(0.8, 1.0, n).astype(np.float32)
        pts    = np.stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ], axis=1)
        scalars = pts[:, 2]  # color by Z

    else:  # uniform
        pts     = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
        scalars = pts[:, 0]

    return pts, scalars.astype(np.float32)


# ── Main app ──────────────────────────────────────────────────────────────────

class BenchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("dragonsci — benchmark")
        self.geometry("1280x800")
        self.configure(bg="#1e1e1e")

        self._rng = np.random.default_rng(0)

        # Timing state — draw and upload tracked separately
        self._draw_times: deque[float] = deque(maxlen=FPS_WINDOW)   # render() only
        self._upload_ms: float = 0.0
        self._gen_ms: float = 0.0
        self._total_frames: int = 0

        # New-cloud timer
        self._last_cloud_ts: float = time.perf_counter()

        self._build_ui()

        # Trigger first cloud after layout is ready
        self.after(200, self._refresh_cloud)

    # ── UI builder ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Left: scatter widget ──────────────────────────────────────────────
        # vsync=False so render() returns without blocking on the display.
        # The timed block measures CPU submission time (queue.submit + present),
        # not GPU execution time — GPU work runs asynchronously after submit.
        self._scatter = Scatter3D(self, fps=60, vsync=False, bg="#0d0d0d")
        self._scatter.pack(side="left", fill="both", expand=True)

        # Intercept the render tick to record frame times
        self._scatter._render_tick = self._patched_render_tick

        # ── Right: scrollable control panel ──────────────────────────────────
        sidebar = tk.Frame(self, bg="#1e1e1e", width=296)
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)

        canvas = tk.Canvas(sidebar, bg="#1e1e1e", highlightthickness=0, width=280)
        scrollbar = tk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        panel = tk.Frame(canvas, bg="#1e1e1e")
        panel_window = canvas.create_window((0, 0), window=panel, anchor="nw")

        def _on_panel_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        panel.bind("<Configure>", _on_panel_configure)

        def _on_canvas_configure(event):
            canvas.itemconfig(panel_window, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _bind_scroll(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind_scroll(event):
            canvas.unbind_all("<MouseWheel>")
        canvas.bind("<Enter>", _bind_scroll)
        canvas.bind("<Leave>", _unbind_scroll)
        panel.bind("<Enter>", _bind_scroll)
        panel.bind("<Leave>", _unbind_scroll)

        def section(label):
            tk.Label(panel, text=label, bg="#1e1e1e", fg="#888",
                     font=("Consolas", 9)).pack(anchor="w", pady=(12, 2))

        def labeled_slider(parent, text, var, from_, to, resolution=1, fmt=None):
            row = tk.Frame(parent, bg="#1e1e1e")
            row.pack(fill="x", pady=1)
            tk.Label(row, text=text, bg="#1e1e1e", fg="#ccc",
                     font=("Consolas", 10), width=14, anchor="w").pack(side="left")
            val_lbl = tk.Label(row, bg="#1e1e1e", fg="#4fc",
                               font=("Consolas", 10), width=8, anchor="e")
            val_lbl.pack(side="right")
            def update_label(*_):
                v = var.get()
                val_lbl.config(text=(fmt(v) if fmt else str(v)))
            var.trace_add("write", update_label)
            tk.Scale(parent, variable=var, from_=from_, to=to,
                     resolution=resolution, orient="horizontal",
                     bg="#1e1e1e", fg="#ccc", troughcolor="#333",
                     highlightthickness=0, bd=0,
                     command=lambda *_: self._on_slider_change()
                     ).pack(fill="x")
            update_label()

        # ── Point count ───────────────────────────────────────────────────────
        section("POINT CLOUD")
        self._n_var = tk.IntVar(value=100_000)
        labeled_slider(panel, "N points", self._n_var,
                       from_=1_000, to=500_000, resolution=1_000,
                       fmt=lambda v: f"{v:,}")

        self._size_var = tk.DoubleVar(value=3.0)
        labeled_slider(panel, "Point size", self._size_var,
                       from_=1.0, to=12.0, resolution=0.5,
                       fmt=lambda v: f"{v:.1f}px")

        self._zscale_var = tk.DoubleVar(value=1.0)
        labeled_slider(panel, "Z scale", self._zscale_var,
                       from_=0.0, to=1.0, resolution=0.01,
                       fmt=lambda v: f"{v:.2f}")

        # ── Colormap ──────────────────────────────────────────────────────────
        section("COLORMAP")
        self._cmap_var = tk.StringVar(value="plasma")
        for name in COLORMAPS:
            rb = tk.Radiobutton(panel, text=name, variable=self._cmap_var, value=name,
                                bg="#1e1e1e", fg="#ccc", selectcolor="#333",
                                activebackground="#1e1e1e", activeforeground="#4fc",
                                font=("Consolas", 10),
                                command=self._on_slider_change)
            rb.pack(anchor="w")

        # ── Shape ─────────────────────────────────────────────────────────────
        section("DISTRIBUTION")
        self._shape_var = tk.StringVar(value="torus")
        for s in SHAPES:
            rb = tk.Radiobutton(panel, text=s, variable=self._shape_var, value=s,
                                bg="#1e1e1e", fg="#ccc", selectcolor="#333",
                                activebackground="#1e1e1e", activeforeground="#4fc",
                                font=("Consolas", 10),
                                command=self._refresh_cloud)
            rb.pack(anchor="w")

        # ── Scalar pipeline ───────────────────────────────────────────────────
        section("SCALAR PIPELINE")
        self._scalar_bar_var = tk.BooleanVar(value=False)
        self._log_scale_var  = tk.BooleanVar(value=False)
        self._clim_var       = tk.BooleanVar(value=False)
        self._clim_lo_var    = tk.DoubleVar(value=-2.0)
        self._clim_hi_var    = tk.DoubleVar(value=2.0)

        tk.Checkbutton(panel, text="Scalar bar", variable=self._scalar_bar_var,
                       bg="#1e1e1e", fg="#ccc", selectcolor="#333",
                       activebackground="#1e1e1e", activeforeground="#4fc",
                       font=("Consolas", 10),
                       command=self._on_slider_change).pack(anchor="w")
        tk.Checkbutton(panel, text="Log scale", variable=self._log_scale_var,
                       bg="#1e1e1e", fg="#ccc", selectcolor="#333",
                       activebackground="#1e1e1e", activeforeground="#4fc",
                       font=("Consolas", 10),
                       command=self._on_slider_change).pack(anchor="w")
        tk.Checkbutton(panel, text="Fix clim", variable=self._clim_var,
                       bg="#1e1e1e", fg="#ccc", selectcolor="#333",
                       activebackground="#1e1e1e", activeforeground="#4fc",
                       font=("Consolas", 10),
                       command=self._on_slider_change).pack(anchor="w")
        labeled_slider(panel, "clim lo", self._clim_lo_var,
                       from_=-5.0, to=5.0, resolution=0.1,
                       fmt=lambda v: f"{v:.1f}")
        labeled_slider(panel, "clim hi", self._clim_hi_var,
                       from_=-5.0, to=5.0, resolution=0.1,
                       fmt=lambda v: f"{v:.1f}")

        # ── Axis ticks ────────────────────────────────────────────────────────
        section("AXIS TICKS  (0 = auto)")
        self._ticks_x_var = tk.IntVar(value=0)
        self._ticks_y_var = tk.IntVar(value=0)
        self._ticks_z_var = tk.IntVar(value=0)
        tick_fmt = lambda v: "auto" if v == 0 else str(v)
        labeled_slider(panel, "X ticks", self._ticks_x_var,
                       from_=0, to=10, resolution=1, fmt=tick_fmt)
        labeled_slider(panel, "Y ticks", self._ticks_y_var,
                       from_=0, to=10, resolution=1, fmt=tick_fmt)
        labeled_slider(panel, "Z ticks", self._ticks_z_var,
                       from_=0, to=10, resolution=1, fmt=tick_fmt)

        # ── Auto-refresh interval ─────────────────────────────────────────────
        section("AUTO REFRESH")
        self._interval_var = tk.DoubleVar(value=1.0)
        labeled_slider(panel, "Interval (s)", self._interval_var,
                       from_=0.25, to=10.0, resolution=0.25,
                       fmt=lambda v: f"{v:.2f}s")

        tk.Button(panel, text="Refresh now", bg="#333", fg="#4fc",
                  activebackground="#444", font=("Consolas", 10),
                  relief="flat", bd=0, pady=4,
                  command=self._refresh_cloud).pack(fill="x", pady=(8, 0))

        tk.Button(panel, text="Reset camera", bg="#333", fg="#fc8",
                  activebackground="#444", font=("Consolas", 10),
                  relief="flat", bd=0, pady=4,
                  command=lambda: self._scatter.reset_camera()).pack(fill="x", pady=(4, 0))

        def _save_screenshot():
            import tkinter.filedialog as fd
            path = fd.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
                title="Save screenshot",
            )
            if path:
                try:
                    self._scatter.save_png(path)
                except Exception as exc:
                    import tkinter.messagebox as mb
                    mb.showerror("Screenshot failed", str(exc))

        # ── Animation export ──────────────────────────────────────────────────
        section("ANIMATION")
        self._anim_info_var = tk.StringVar(value="—")
        tk.Label(panel, textvariable=self._anim_info_var,
                 bg="#1e1e1e", fg="#fc8", font=("Consolas", 9),
                 justify="left", anchor="w").pack(fill="x")

        self._anim_frames_var = tk.IntVar(value=60)
        self._anim_fps_var = tk.IntVar(value=20)

        tk.Label(panel, text="Frames", bg="#1e1e1e", fg="#888",
                 font=("Consolas", 9)).pack(anchor="w", pady=(4, 0))
        tk.Scale(panel, from_=10, to=180, resolution=10, orient="horizontal",
                 variable=self._anim_frames_var, bg="#1e1e1e", fg="#ccc",
                 troughcolor="#333", highlightthickness=0,
                 font=("Consolas", 9)).pack(fill="x")
        tk.Label(panel, text="FPS", bg="#1e1e1e", fg="#888",
                 font=("Consolas", 9)).pack(anchor="w")
        tk.Scale(panel, from_=5, to=30, resolution=5, orient="horizontal",
                 variable=self._anim_fps_var, bg="#1e1e1e", fg="#ccc",
                 troughcolor="#333", highlightthickness=0,
                 font=("Consolas", 9)).pack(fill="x")

        def _save_orbit_gif():
            import tkinter.filedialog as fd
            path = fd.asksaveasfilename(
                title="Save orbit GIF",
                defaultextension=".gif",
                filetypes=[("GIF", "*.gif"), ("All files", "*.*")],
            )
            if not path:
                return
            n = self._anim_frames_var.get()
            fps = self._anim_fps_var.get()
            self._anim_info_var.set("Recording…")
            panel.update()

            def _progress(i, total):
                self._anim_info_var.set(f"Frame {i + 1}/{total}")
                panel.update()

            try:
                self._scatter.orbit_gif(path, n_frames=n, fps=fps,
                                         on_progress=_progress)
                self._anim_info_var.set(f"Saved {n} frames → {path.split('/')[-1]}")
            except Exception as exc:
                self._anim_info_var.set(f"Error: {exc}")

        tk.Button(panel, text="Save orbit GIF…", bg="#333", fg="#4fc",
                  activebackground="#444", font=("Consolas", 10),
                  relief="flat", bd=0, pady=3,
                  command=_save_orbit_gif).pack(fill="x", pady=(4, 0))

        # ── Export ────────────────────────────────────────────────────────────
        section("EXPORT")
        tk.Button(panel, text="Save screenshot…", bg="#333", fg="#8cf",
                  activebackground="#444", font=("Consolas", 10),
                  relief="flat", bd=0, pady=4,
                  command=_save_screenshot).pack(fill="x", pady=(4, 0))

        # ── Multi-actor ───────────────────────────────────────────────────────
        section("MULTI-ACTOR")
        self._actor_handles: list[int] = []
        self._actor_count_var = tk.StringVar(value="Actors: 0")
        tk.Label(panel, textvariable=self._actor_count_var,
                 bg="#1e1e1e", fg="#4fc", font=("Consolas", 10)).pack(anchor="w")

        def _add_actor():
            n      = self._n_var.get()
            cmap   = self._cmap_var.get()
            size   = self._size_var.get()
            shape  = self._shape_var.get()
            offset = (len(self._actor_handles) + 1) * 3.0
            pts, scalars = make_cloud(shape, n // 4, self._rng)
            pts = pts + np.array([offset, 0.0, 0.0], dtype=np.float32)
            h = self._scatter.add_points(pts, scalars=scalars, colormap=cmap, point_size=size)
            self._actor_handles.append(h)
            self._actor_count_var.set(f"Actors: {len(self._actor_handles)}")

        def _clear_actors():
            self._actor_handles.clear()
            self._scatter.clear()
            self._actor_count_var.set("Actors: 0")

        def _hide_last():
            if self._actor_handles:
                h = self._actor_handles[-1]
                self._scatter.set_actor_visibility(h, False)

        def _show_all():
            for h in self._actor_handles:
                self._scatter.set_actor_visibility(h, True)

        def _remove_last():
            if self._actor_handles:
                h = self._actor_handles.pop()
                self._scatter.remove_actor(h)
                self._actor_count_var.set(f"Actors: {len(self._actor_handles)}")

        for lbl, cmd in [
            ("Add actor",   _add_actor),
            ("Hide last",   _hide_last),
            ("Show all",    _show_all),
            ("Remove last", _remove_last),
            ("Clear all",   _clear_actors),
        ]:
            tk.Button(panel, text=lbl, bg="#333", fg="#4fc",
                      activebackground="#444", font=("Consolas", 10),
                      relief="flat", bd=0, pady=3,
                      command=cmd).pack(fill="x", pady=(2, 0))

        # ── Camera presets ────────────────────────────────────────────────────
        section("CAMERA PRESETS")
        for label, cmd in [
            ("XY  (top)",      lambda: self._scatter.view_xy()),
            ("XZ  (front)",    lambda: self._scatter.view_xz()),
            ("YZ  (side)",     lambda: self._scatter.view_yz()),
            ("Isometric",      lambda: self._scatter.view_isometric()),
        ]:
            tk.Button(panel, text=label, bg="#333", fg="#4fc",
                      activebackground="#444", font=("Consolas", 10),
                      relief="flat", bd=0, pady=3,
                      command=cmd).pack(fill="x", pady=(2, 0))

        self._parallel_var = tk.BooleanVar(value=False)
        def _toggle_parallel():
            self._scatter.parallel_projection = self._parallel_var.get()
        tk.Checkbutton(panel, text="Parallel projection", variable=self._parallel_var,
                       bg="#1e1e1e", fg="#ccc", selectcolor="#333",
                       activebackground="#1e1e1e", activeforeground="#4fc",
                       font=("Consolas", 10),
                       command=_toggle_parallel).pack(anchor="w", pady=(6, 0))

        tk.Button(panel, text="Fit to data", bg="#333", fg="#4fc",
                  activebackground="#444", font=("Consolas", 10),
                  relief="flat", bd=0, pady=3,
                  command=lambda: self._scatter.fit()).pack(fill="x", pady=(4, 0))

        # ── Rendering modes ───────────────────────────────────────────────────
        section("RENDERING")
        self._style_var = tk.StringVar(value="circle")

        def _set_style(*_):
            self._scatter.point_style = self._style_var.get()

        for val, lbl in [("circle", "Circle (soft)"), ("square", "Square"), ("gaussian", "Gaussian")]:
            tk.Radiobutton(panel, text=lbl, variable=self._style_var, value=val,
                           bg="#1e1e1e", fg="#ccc", selectcolor="#333",
                           activebackground="#1e1e1e", activeforeground="#4fc",
                           font=("Consolas", 10),
                           command=_set_style).pack(anchor="w")

        self._opacity_var = tk.DoubleVar(value=1.0)

        def _opacity_changed(*_):
            self._refresh_cloud()

        tk.Label(panel, text="Opacity", bg="#1e1e1e", fg="#888",
                 font=("Consolas", 9)).pack(anchor="w", pady=(6, 0))
        tk.Scale(panel, from_=0.05, to=1.0, resolution=0.05, orient="horizontal",
                 variable=self._opacity_var, bg="#1e1e1e", fg="#ccc",
                 troughcolor="#333", highlightthickness=0, font=("Consolas", 9),
                 command=_opacity_changed).pack(fill="x")

        self._lod_var = tk.BooleanVar(value=True)

        def _toggle_lod():
            self._scatter._lod_enabled = self._lod_var.get()

        tk.Checkbutton(panel, text="Interaction LOD", variable=self._lod_var,
                       bg="#1e1e1e", fg="#ccc", selectcolor="#333",
                       activebackground="#1e1e1e", activeforeground="#4fc",
                       font=("Consolas", 10),
                       command=_toggle_lod).pack(anchor="w", pady=(4, 0))

        # ── Picking ───────────────────────────────────────────────────────────
        section("PICKING")
        self._pick_info_var = tk.StringVar(value="—")
        tk.Label(panel, textvariable=self._pick_info_var,
                 bg="#1e1e1e", fg="#fc8", font=("Consolas", 9),
                 justify="left", anchor="w", wraplength=256).pack(fill="x")

        self._pick_mode_var = tk.StringVar(value="none")

        def _set_pick_mode(*_):
            mode = self._pick_mode_var.get()
            if mode == "point":
                self._scatter.enable_point_picking()
            elif mode == "rect":
                self._scatter.enable_rectangle_picking()
            elif mode == "both":
                self._scatter.enable_point_picking()
                self._scatter.enable_rectangle_picking()
            else:
                self._scatter.disable_picking()

        for val, lbl in [("none", "Off"), ("point", "Point pick"),
                         ("rect", "Rect select"), ("both", "Both")]:
            tk.Radiobutton(panel, text=lbl, variable=self._pick_mode_var, value=val,
                           bg="#1e1e1e", fg="#ccc", selectcolor="#333",
                           activebackground="#1e1e1e", activeforeground="#4fc",
                           font=("Consolas", 10),
                           command=_set_pick_mode).pack(anchor="w")

        def _on_pick(event):
            w = event.widget
            if w.picked_point is not None:
                p = w.picked_point
                self._pick_info_var.set(
                    f"Actor {w.picked_actor}  idx {w.picked_index}\n"
                    f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                )

        def _on_select(event):
            sel = event.widget.selected or []
            self._pick_info_var.set(f"{len(sel)} points selected")

        self._scatter.bind("<<PointPicked>>", _on_pick, add="+")
        self._scatter.bind("<<SelectionChanged>>", _on_select, add="+")

        # ── Overlays ──────────────────────────────────────────────────────────
        section("OVERLAYS")
        self._overlay_handles: list[int] = []

        def _add_box_overlay():
            """Add a bounding box around the current data."""
            h = self._scatter.add_box((-2, -2, -2, 2, 2, 2), color=(1.0, 1.0, 0.0))
            if h >= 0:
                self._overlay_handles.append(h)
                self._overlay_info_var.set(f"{len(self._overlay_handles)} overlay(s)")

        def _add_axes_lines():
            """Add XYZ axis lines through the origin."""
            segs = np.array([
                [-3, 0, 0, 3, 0, 0],
                [0, -3, 0, 0, 3, 0],
                [0, 0, -3, 0, 0, 3],
            ], dtype=np.float32)
            colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.5, 1.0)]
            for i, color in enumerate(colors):
                h = self._scatter.add_lines(segs[i:i+1], color=color)
                if h >= 0:
                    self._overlay_handles.append(h)
            self._overlay_info_var.set(f"{len(self._overlay_handles)} overlay(s)")

        def _clear_overlays():
            self._scatter.clear_overlays()
            self._overlay_handles.clear()
            self._overlay_info_var.set("0 overlay(s)")

        def _toggle_axes_widget():
            self._axes_visible = not getattr(self, "_axes_visible", False)
            self._scatter.show_orientation_axes(self._axes_visible)
            _axes_btn.config(text="Hide axes widget" if self._axes_visible else "Show axes widget")

        self._overlay_info_var = tk.StringVar(value="0 overlay(s)")
        tk.Label(panel, textvariable=self._overlay_info_var,
                 bg="#1e1e1e", fg="#fc8", font=("Consolas", 9),
                 justify="left", anchor="w").pack(fill="x")

        tk.Button(panel, text="Add bounding box", bg="#333", fg="#4fc",
                  activebackground="#444", font=("Consolas", 10),
                  relief="flat", bd=0, pady=3,
                  command=_add_box_overlay).pack(fill="x", pady=(2, 0))
        tk.Button(panel, text="Add axis lines", bg="#333", fg="#4fc",
                  activebackground="#444", font=("Consolas", 10),
                  relief="flat", bd=0, pady=3,
                  command=_add_axes_lines).pack(fill="x", pady=(2, 0))
        tk.Button(panel, text="Clear overlays", bg="#333", fg="#c88",
                  activebackground="#444", font=("Consolas", 10),
                  relief="flat", bd=0, pady=3,
                  command=_clear_overlays).pack(fill="x", pady=(2, 0))
        _axes_btn = tk.Button(panel, text="Show axes widget", bg="#333", fg="#4fc",
                              activebackground="#444", font=("Consolas", 10),
                              relief="flat", bd=0, pady=3,
                              command=_toggle_axes_widget)
        _axes_btn.pack(fill="x", pady=(2, 0))

        # ── Linked views ──────────────────────────────────────────────────────
        section("LINKED VIEWS")
        self._linked_windows: list = []  # (Toplevel, Scatter3D) pairs
        self._linked_info_var = tk.StringVar(value="0 linked window(s)")
        tk.Label(panel, textvariable=self._linked_info_var,
                 bg="#1e1e1e", fg="#fc8", font=("Consolas", 9),
                 justify="left", anchor="w").pack(fill="x")

        def _open_linked_window():
            top = tk.Toplevel(self)
            top.title(f"Linked view {len(self._linked_windows) + 1}")
            top.geometry("600x500")
            sw = Scatter3D(top, width=600, height=500)
            sw.pack(fill="both", expand=True)
            # Copy current cloud into new widget and link cameras
            n = self._n_var.get()
            shape = self._shape_var.get()
            pts, scalars = make_cloud(shape, n, np.random.default_rng())
            sw.set_points(pts, scalars=scalars,
                          colormap=self._cmap_var.get(),
                          point_size=self._size_var.get())
            link_cameras(self._scatter, sw)
            self._linked_windows.append((top, sw))
            self._linked_info_var.set(f"{len(self._linked_windows)} linked window(s)")

            def _on_close():
                unlink_cameras(self._scatter, sw)
                self._linked_windows[:] = [(t, w) for t, w in self._linked_windows if t is not top]
                self._linked_info_var.set(f"{len(self._linked_windows)} linked window(s)")
                top.destroy()

            top.protocol("WM_DELETE_WINDOW", _on_close)

        def _close_all_linked():
            for top, sw in list(self._linked_windows):
                unlink_cameras(self._scatter, sw)
                top.destroy()
            self._linked_windows.clear()
            self._linked_info_var.set("0 linked window(s)")

        tk.Button(panel, text="Open linked window", bg="#333", fg="#4fc",
                  activebackground="#444", font=("Consolas", 10),
                  relief="flat", bd=0, pady=3,
                  command=_open_linked_window).pack(fill="x", pady=(2, 0))
        tk.Button(panel, text="Close all linked", bg="#333", fg="#c88",
                  activebackground="#444", font=("Consolas", 10),
                  relief="flat", bd=0, pady=3,
                  command=_close_all_linked).pack(fill="x", pady=(2, 0))

        # ── Appearance ────────────────────────────────────────────────────────
        section("APPEARANCE")

        self._grid_var = tk.BooleanVar(value=True)
        def _toggle_grid():
            self._scatter.show_grid(self._grid_var.get())
        tk.Checkbutton(panel, text="Show grid", variable=self._grid_var,
                       command=_toggle_grid,
                       bg="#1e1e1e", fg="#ccc", selectcolor="#2a2a3a",
                       activebackground="#1e1e1e", font=("Consolas", 10),
                       anchor="w").pack(fill="x")

        # Background colour presets
        _BG_PRESETS = [
            ("Dark (default)", (0.05, 0.05, 0.07)),
            ("Black",          (0.0,  0.0,  0.0 )),
            ("White",          (1.0,  1.0,  1.0 )),
            ("Navy",           (0.05, 0.05, 0.20)),
        ]
        bg_row = tk.Frame(panel, bg="#1e1e1e")
        bg_row.pack(fill="x", pady=(2, 0))
        tk.Label(bg_row, text="Background:", bg="#1e1e1e", fg="#aaa",
                 font=("Consolas", 9)).pack(side="left")
        self._bg_preset_var = tk.StringVar(value="Dark (default)")
        def _on_bg_preset(*_):
            label = self._bg_preset_var.get()
            for name, color in _BG_PRESETS:
                if name == label:
                    self._scatter.set_background(color)
                    break
        bg_menu = tk.OptionMenu(bg_row, self._bg_preset_var,
                                *[n for n, _ in _BG_PRESETS],
                                command=_on_bg_preset)
        bg_menu.config(bg="#333", fg="#ccc", activebackground="#444",
                       font=("Consolas", 9), bd=0, highlightthickness=0)
        bg_menu.pack(side="left", fill="x", expand=True)

        # Axis title entry fields
        ax_row = tk.Frame(panel, bg="#1e1e1e")
        ax_row.pack(fill="x", pady=(4, 0))
        tk.Label(ax_row, text="Axis labels:", bg="#1e1e1e", fg="#aaa",
                 font=("Consolas", 9)).pack(side="left")
        self._ax_x_var = tk.StringVar(value="X")
        self._ax_y_var = tk.StringVar(value="Y")
        self._ax_z_var = tk.StringVar(value="Z")
        def _apply_axis_labels(*_):
            self._scatter.set_axes(
                self._ax_x_var.get(),
                self._ax_y_var.get(),
                self._ax_z_var.get(),
            )
        for var, label in [(self._ax_x_var, "X"), (self._ax_y_var, "Y"), (self._ax_z_var, "Z")]:
            entry = tk.Entry(ax_row, textvariable=var, width=5,
                             bg="#2a2a2a", fg="#ccc", insertbackground="#ccc",
                             font=("Consolas", 9), bd=1, relief="solid")
            entry.pack(side="left", padx=(2, 0))
            entry.bind("<Return>", _apply_axis_labels)
            entry.bind("<FocusOut>", _apply_axis_labels)

        # ── Stats ─────────────────────────────────────────────────────────────
        section("STATISTICS")
        self._stats_text = tk.StringVar(value="—")
        stats_lbl = tk.Label(panel, textvariable=self._stats_text,
                             bg="#1e1e1e", fg="#4fc", font=("Consolas", 10),
                             justify="left", anchor="w")
        stats_lbl.pack(fill="x", pady=(2, 0))

        # ── Controls hint ─────────────────────────────────────────────────────
        section("MOUSE")
        hint = "L-drag       : orbit\nShift+L-drag : pan\nM-drag       : pan\nScroll       : zoom\nDbl-click    : reset"
        tk.Label(panel, text=hint, bg="#1e1e1e", fg="#666",
                 font=("Consolas", 9), justify="left").pack(anchor="w")

    # ── Render-loop patch ─────────────────────────────────────────────────────

    def _patched_render_tick(self):
        """Replaces Scatter3D._render_tick; separates submit time from upload."""
        r = self._scatter._renderer

        # Auto-refresh runs before submit so upload cost stays out of submit timer
        interval = self._interval_var.get()
        if time.perf_counter() - self._last_cloud_ts >= interval:
            self._refresh_cloud()

        self._scatter._after_id = None  # cleared so _mark_dirty can re-arm
        if r is not None and self._scatter._dirty:
            t0 = time.perf_counter()
            try:
                r.render()
                self._scatter._dirty = False  # only clear on success; error keeps dirty
                self._draw_times.append(time.perf_counter() - t0)
                self._total_frames += 1
            except Exception:
                pass

        self._update_stats()
        # Re-arm unconditionally in benchmark mode (auto-refresh needs continuous loop)
        self._scatter._schedule_render()

    # ── Cloud upload ──────────────────────────────────────────────────────────

    def _refresh_cloud(self, *_):
        n      = self._n_var.get()
        cmap   = self._cmap_var.get()
        size   = self._size_var.get()
        shape  = self._shape_var.get()
        zscale = self._zscale_var.get()

        t0 = time.perf_counter()
        pts, scalars = make_cloud(shape, n, self._rng)
        gen_ms = (time.perf_counter() - t0) * 1000

        if zscale != 1.0:
            pts = pts.copy()
            pts[:, 2] *= zscale

        log  = self._log_scale_var.get()
        clim = None
        if self._clim_var.get():
            lo, hi = self._clim_lo_var.get(), self._clim_hi_var.get()
            if lo < hi:
                clim = (lo, hi)

        t1 = time.perf_counter()
        self._scatter.set_points(pts, scalars=scalars, colormap=cmap, point_size=size,
                                  clim=clim, log_scale=log,
                                  opacity=self._opacity_var.get())
        self._upload_ms = (time.perf_counter() - t1) * 1000
        self._gen_ms    = gen_ms

        # Update scalar bar to match current settings
        if self._scalar_bar_var.get():
            vmin, vmax = (clim if clim else (float(scalars.min()), float(scalars.max())))
            self._scatter.scalar_bar(True, vmin=vmin, vmax=vmax,
                                      log_scale=log, colormap=cmap, title=cmap)
        else:
            self._scatter.scalar_bar(False)

        self._last_cloud_ts = time.perf_counter()

    def _apply_ticks(self):
        to_override = lambda v: None if v == 0 else v
        self._scatter.set_ticks(
            x=to_override(self._ticks_x_var.get()),
            y=to_override(self._ticks_y_var.get()),
            z=to_override(self._ticks_z_var.get()),
        )

    def _on_slider_change(self, *_):
        self._apply_ticks()
        self._refresh_cloud()

    # ── Stats display ─────────────────────────────────────────────────────────

    def _update_stats(self):
        if not self._draw_times:
            return
        # submit_times = CPU submission only (queue.submit + present, no vsync, no upload)
        avg_sub = sum(self._draw_times) / len(self._draw_times) * 1000
        min_sub = min(self._draw_times) * 1000
        max_sub = max(self._draw_times) * 1000
        # Max theoretical FPS if submission were the only bottleneck
        fps_cap = 1000.0 / avg_sub if avg_sub > 0 else 0

        n = self._n_var.get()
        pts_per_sec = n / (avg_sub / 1000) if avg_sub > 0 else 0

        text = (
            f"Submit avg {avg_sub:6.2f} ms\n"
            f"Submit min {min_sub:6.2f} ms\n"
            f"Submit max {max_sub:6.2f} ms\n"
            f"FPS cap    {fps_cap:6.1f}\n"
            f"─────────────────\n"
            f"N points   {n:>9,}\n"
            f"Upload     {self._upload_ms:6.2f} ms\n"
            f"Generate   {self._gen_ms:6.2f} ms\n"
            f"─────────────────\n"
            f"Pts/frame  {n/1e3:6.1f}k\n"
            f"Pts/sec    {pts_per_sec/1e6:6.2f}M\n"
            f"Total frms {self._total_frames:>9,}"
        )
        self._stats_text.set(text)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = BenchApp()
    app.mainloop()
