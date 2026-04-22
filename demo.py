"""
DragonSci feature demo.

One section per feature.  Left sidebar navigates; right panel is a single
shared Scatter3D widget that is reconfigured when you switch sections.
Planned features show a clearly labelled stub.

Usage
-----
    python demo.py
"""
from __future__ import annotations

import time
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np

from dragonsci import Scatter3D, Scatter2D, Line2D, Figure, link_cameras, unlink_cameras

# ── Theme ──────────────────────────────────────────────────────────────────────

BG    = "#1a1a2e"
PANEL = "#16213e"
FG    = "#e0e0e0"
DIM   = "#777777"
ACT   = "#4fc3f7"
WARN  = "#ffcc80"
RED   = "#ef9a9a"
PLAN  = "#444444"
FM    = ("Consolas", 10)
FH    = ("Consolas", 11, "bold")

CMAPS = ["viridis", "plasma", "inferno", "magma", "coolwarm",
         "hot", "gray", "turbo", "cividis", "blues", "greens", "reds"]

RNG = np.random.default_rng(42)


# ── Synthetic data generators ──────────────────────────────────────────────────

def _torus(n: int) -> tuple[np.ndarray, np.ndarray]:
    theta = RNG.uniform(0, 2 * np.pi, n).astype(np.float32)
    phi   = RNG.uniform(0, 2 * np.pi, n).astype(np.float32)
    r = (2.0 + np.cos(phi)).astype(np.float32)
    pts = np.stack([r * np.cos(theta), r * np.sin(theta), np.sin(phi)], axis=1)
    return pts, r


def _sphere(n: int) -> tuple[np.ndarray, np.ndarray]:
    theta = RNG.uniform(0, 2 * np.pi, n).astype(np.float32)
    phi   = np.arccos(RNG.uniform(-1, 1, n)).astype(np.float32)
    pts = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], axis=1)
    return pts, pts[:, 2].copy()


def _helix(n: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 8 * np.pi, n, dtype=np.float32)
    noise = RNG.standard_normal((n, 3)).astype(np.float32) * 0.05
    pts = np.stack([np.cos(t), np.sin(t), t / (8 * np.pi)], axis=1) + noise
    return pts, (t / t.max()).astype(np.float32)


def _gaussian(n: int) -> tuple[np.ndarray, np.ndarray]:
    pts = RNG.standard_normal((n, 3)).astype(np.float32)
    return pts, np.linalg.norm(pts, axis=1).astype(np.float32)


SHAPES = {"torus": _torus, "sphere": _sphere, "helix": _helix, "gaussian": _gaussian}


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _head(p, text: str) -> None:
    tk.Label(p, text=text, bg=PANEL, fg=DIM, font=("Consolas", 9),
             anchor="w").pack(fill="x", pady=(10, 2))


def _label(p, text="", textvariable=None, fg=FG) -> tk.Label:
    lbl = tk.Label(p, text=text, textvariable=textvariable,
                   bg=PANEL, fg=fg, font=FM, anchor="w",
                   justify="left", wraplength=230)
    lbl.pack(fill="x")
    return lbl


def _btn(p, text: str, cmd, fg=ACT) -> tk.Button:
    b = tk.Button(p, text=text, command=cmd, bg="#222", fg=fg,
                  activebackground="#333", font=FM, relief="flat", bd=0, pady=3)
    b.pack(fill="x", pady=1)
    return b


def _check(p, text: str, var: tk.BooleanVar, cmd) -> None:
    tk.Checkbutton(p, text=text, variable=var, command=cmd,
                   bg=PANEL, fg=FG, selectcolor="#333",
                   activebackground=PANEL, activeforeground=ACT,
                   font=FM, anchor="w").pack(fill="x")


def _radio_group(p, var, choices: list[tuple[str, object]], cmd) -> None:
    for lbl, val in choices:
        tk.Radiobutton(p, text=lbl, variable=var, value=val,
                       bg=PANEL, fg=FG, selectcolor="#333",
                       activebackground=PANEL, font=FM, anchor="w",
                       command=cmd).pack(fill="x")


# ── Main application ───────────────────────────────────────────────────────────

class DemoApp(tk.Tk):
    """Feature demo.  Left = section nav + controls.  Right = Scatter3D."""

    # (display name, builder method, is_planned)
    SECTIONS: list[tuple[str, str, bool]] = [
        ("Scatter Basics",    "_sec_basics",      False),
        ("DataFrame API",     "_sec_dataframe",   False),
        ("Multi-Actor",       "_sec_multiactor",  False),
        ("Picking",           "_sec_picking",     False),
        ("Camera",            "_sec_camera",      False),
        ("Overlays",          "_sec_overlays",    False),
        ("Export",            "_sec_export",      False),
        ("Hover Tooltips",    "_sec_hover",       False),
        ("Categ. Color",      "_sec_catcolor",    False),
        ("Size by Column",    "_sec_sizebycolumn", False),
        ("2D Mode",           "_sec_2dmode",      False),
        ("Marginal Hists",    "_sec_marginals",   False),
        ("Point Labels",      "_sec_labels",      False),
        ("Figure Subplots",   "_sec_figure",      False),
        ("Statistical Overlays", "_sec_stat_overlays", False),
        ("── planned ──",    "",                 True),   # separator
        ("Lasso Selection",   "_sec_lasso",       False),
        ("Streaming",         "_sec_streaming",   False),
        ("Line2D",            "_sec_line2d",      False),
    ]

    def __init__(self):
        super().__init__()
        self.title("DragonSci — Feature Demo")
        self.geometry("1320x840")
        self.configure(bg=BG)

        self._nav_btns: dict[str, tk.Button] = {}
        self._active: str = ""
        self._actor_handles: list[int] = []
        self._linked_wins: list[tuple] = []
        self._scatter_parent: "tk.Frame | None" = None
        self._figure: "Figure | None" = None

        self._build_layout()
        self.after(200, lambda: self._switch("_sec_basics"))

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        # ── Left sidebar ──────────────────────────────────────────────────────
        sidebar = tk.Frame(self, bg=BG, width=260)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Title + nav buttons
        nav = tk.Frame(sidebar, bg=BG)
        nav.pack(fill="x", padx=6, pady=(8, 4))
        tk.Label(nav, text="DRAGONSCI  DEMO", bg=BG, fg=ACT, font=FH,
                 anchor="w").pack(fill="x", pady=(0, 6))

        for name, method, is_planned in self.SECTIONS:
            if not method:                       # separator row
                tk.Label(nav, text=name, bg=BG, fg=PLAN,
                         font=("Consolas", 8), anchor="w").pack(fill="x", pady=(6, 0))
                continue
            fg = PLAN if is_planned else FG
            b = tk.Button(nav, text=name,
                          command=lambda m=method: self._switch(m),
                          bg=BG, fg=fg, activebackground="#222",
                          font=FM, relief="flat", bd=0, pady=2, anchor="w")
            b.pack(fill="x")
            self._nav_btns[method] = b

        tk.Frame(sidebar, bg=DIM, height=1).pack(fill="x", padx=6, pady=6)

        # Scrollable controls area
        ctrl_outer = tk.Frame(sidebar, bg=PANEL)
        ctrl_outer.pack(fill="both", expand=True, padx=4, pady=4)

        self._ctrl_canvas = tk.Canvas(ctrl_outer, bg=PANEL, highlightthickness=0)
        sb = tk.Scrollbar(ctrl_outer, orient="vertical",
                          command=self._ctrl_canvas.yview)
        self._ctrl_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._ctrl_canvas.pack(side="left", fill="both", expand=True)

        self._ctrl_frame = tk.Frame(self._ctrl_canvas, bg=PANEL)
        self._ctrl_win = self._ctrl_canvas.create_window(
            (0, 0), window=self._ctrl_frame, anchor="nw")

        self._ctrl_frame.bind("<Configure>",
            lambda e: self._ctrl_canvas.configure(
                scrollregion=self._ctrl_canvas.bbox("all")))
        self._ctrl_canvas.bind("<Configure>",
            lambda e: self._ctrl_canvas.itemconfig(self._ctrl_win, width=e.width))

        # ── Right: scatter + status ───────────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side="right", fill="both", expand=True)

        self._scatter_parent = right
        self.scatter = Scatter3D(right, fps=60, bg="#0d0d0d")
        self.scatter.pack(fill="both", expand=True)

        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(right, textvariable=self._status_var,
                 bg="#111", fg=DIM, font=("Consolas", 9),
                 anchor="w").pack(fill="x", padx=4, pady=2)

    # ── Section switching ──────────────────────────────────────────────────────

    def _swap_scatter(self, cls) -> None:
        """Replace the right-panel scatter widget with an instance of cls."""
        if type(self.scatter) is cls:
            return
        self.scatter.destroy()
        self.scatter = cls(self._scatter_parent, fps=60, bg="#0d0d0d")
        self.scatter.pack(fill="both", expand=True)

    def _scatter_class_for_section(self, method: str):
        """Return the widget class the target section expects."""
        if method == "_sec_line2d":
            return Line2D
        if method in ("_sec_2dmode", "_sec_marginals"):
            return Scatter2D
        return Scatter3D

    def _switch(self, method: str) -> None:
        # Tear down Figure if leaving that section.
        if self._figure is not None and method != "_sec_figure":
            self._figure.destroy()
            self._figure = None
            self.scatter.pack(fill="both", expand=True)

        # Ensure the shared widget type matches the section we're entering.
        target_cls = self._scatter_class_for_section(method)
        if type(self.scatter) is not target_cls:
            self._swap_scatter(target_cls)

        # Reset scatter state that may have been changed by previous section
        self.scatter.disable_picking()
        self.scatter.bind("<<PointPicked>>",      lambda e: None)
        self.scatter.bind("<<SelectionChanged>>", lambda e: None)
        self.scatter.clear_labels()

        # Clear controls
        for w in self._ctrl_frame.winfo_children():
            w.destroy()

        # Update nav highlight
        if self._active and self._active in self._nav_btns:
            self._nav_btns[self._active].configure(bg=BG)
        self._active = method
        if method in self._nav_btns:
            self._nav_btns[method].configure(bg="#222")

        builder = getattr(self, method, None)
        if builder:
            builder(self._ctrl_frame)

        self._ctrl_canvas.yview_moveto(0)

    def _status(self, msg: str) -> None:
        self._status_var.set(msg)

    # ── Section: Scatter Basics ────────────────────────────────────────────────

    def _sec_basics(self, p: tk.Frame) -> None:
        shape_var = tk.StringVar(value="torus")
        n_var     = tk.IntVar(value=250_000)
        cmap_var  = tk.StringVar(value="plasma")
        size_var  = tk.DoubleVar(value=3.0)

        def _load(*_):
            pts, sc = SHAPES[shape_var.get()](n_var.get())
            self.scatter.set_points(pts, scalars=sc,
                                    colormap=cmap_var.get(),
                                    point_size=size_var.get())
            self._status(f"{shape_var.get()}  {n_var.get():,} pts  cmap={cmap_var.get()}")

        _head(p, "SHAPE")
        _radio_group(p, shape_var, [(s, s) for s in SHAPES], _load)

        _head(p, "POINT COUNT")
        _radio_group(p, n_var,
                     [(f"{n:,}", n) for n in [50_000, 250_000, 500_000, 1_000_000]],
                     _load)

        _head(p, "COLORMAP")
        _radio_group(p, cmap_var, [(c, c) for c in CMAPS], _load)

        _head(p, "POINT STYLE")
        style_var = tk.StringVar(value="circle")
        _radio_group(p, style_var,
                     [("Circle (soft)", "circle"), ("Square", "square"),
                      ("Gaussian", "gaussian")],
                     lambda: setattr(self.scatter, "point_style", style_var.get()))

        _head(p, "POINT SIZE")
        tk.Scale(p, variable=size_var, from_=1.0, to=12.0, resolution=0.5,
                 orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                 highlightthickness=0, command=_load).pack(fill="x")

        _load()

    # ── Section: DataFrame API ─────────────────────────────────────────────────

    def _sec_dataframe(self, p: tk.Frame) -> None:
        try:
            import pandas as pd
        except ImportError:
            _label(p, "pandas is not installed.\npip install pandas", fg=WARN)
            return

        # Build a synthetic DataFrame with clear column roles
        n = 8_000
        pts, _ = _torus(n)
        df = pd.DataFrame({
            "lon":        pts[:, 0].astype(float),
            "lat":        pts[:, 1].astype(float),
            "elev":       pts[:, 2].astype(float),
            "temp":       (pts[:, 2] * 10 + 20
                           + RNG.standard_normal(n) * 2).astype(float),
            "station_id": [f"WX{i:04d}" for i in range(n)],
        })

        col_names = list(df.columns)
        x_var   = tk.StringVar(value="lon")
        y_var   = tk.StringVar(value="lat")
        z_var   = tk.StringVar(value="elev")
        col_var = tk.StringVar(value="temp")
        info_var = tk.StringVar(value="—")

        def _load(*_):
            z = z_var.get() or None
            c = col_var.get() or None
            try:
                self.scatter.set_points(df,
                                        x=x_var.get(), y=y_var.get(), z=z,
                                        color=c, hover=["station_id", "temp"],
                                        colormap="plasma")
                info_var.set(f"{len(df):,} rows loaded\n"
                             f"x={x_var.get()}  y={y_var.get()}"
                             f"  z={z or '(zeros)'}\ncolor={c or '(none)'}")
                self._status(f"DataFrame {len(df):,} rows  color={c}")
            except NotImplementedError as e:
                info_var.set(f"Not implemented:\n{e}")
                self._status(str(e))
            except Exception as e:
                info_var.set(f"Error: {e}")
                self._status(f"Error: {e}")

        _head(p, "COLUMN MAPPING")
        for label, var in [("x=", x_var), ("y=", y_var),
                           ("z=", z_var), ("color=", col_var)]:
            row = tk.Frame(p, bg=PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, bg=PANEL, fg=FG,
                     font=FM, width=7, anchor="w").pack(side="left")
            om = tk.OptionMenu(row, var, *col_names, "")
            om.configure(bg="#222", fg=FG, activebackground="#333",
                         font=FM, bd=0, highlightthickness=0)
            om.pack(side="left", fill="x", expand=True)

        _btn(p, "Load DataFrame", _load)

        _head(p, "DATASET")
        _label(p, text="Columns: " + ", ".join(col_names) +
               f"\n{len(df)} rows — torus geometry", fg=DIM)

        _head(p, "NOTES")
        _label(p, text="Numeric color= → scalar colormap path.\n"
                        "String / low-cardinality int color= → categorical\n"
                        "  palette with automatic legend.", fg=DIM)

        _head(p, "STATUS")
        _label(p, textvariable=info_var, fg=ACT)

        _load()

    # ── Section: Multi-Actor ───────────────────────────────────────────────────

    def _sec_multiactor(self, p: tk.Frame) -> None:
        self._actor_handles.clear()
        self.scatter.clear()

        count_var  = tk.StringVar(value="0 actors")
        OFFSETS    = [(0,0,0), (5,0,0), (-5,0,0), (0,5,0), (0,-5,0)]
        ACTOR_MAPS = ["viridis", "plasma", "inferno", "coolwarm", "turbo"]

        def _refresh_count():
            count_var.set(f"{len(self._actor_handles)} actor(s)")

        def _add():
            i   = len(self._actor_handles)
            pts, sc = _torus(25_000)
            pts += np.array(OFFSETS[i % len(OFFSETS)], dtype=np.float32)
            h = self.scatter.add_points(pts, scalars=sc,
                                        colormap=ACTOR_MAPS[i % len(ACTOR_MAPS)],
                                        point_size=3.0)
            self._actor_handles.append(h)
            _refresh_count()
            self._status(f"Added actor handle={h}")

        def _hide_last():
            if self._actor_handles:
                self.scatter.set_actor_visibility(self._actor_handles[-1], False)

        def _show_all():
            for h in self._actor_handles:
                self.scatter.set_actor_visibility(h, True)

        def _remove_last():
            if self._actor_handles:
                h = self._actor_handles.pop()
                self.scatter.remove_actor(h)
                _refresh_count()
                self._status(f"Removed actor handle={h}")

        def _clear_all():
            self._actor_handles.clear()
            self.scatter.clear()
            _refresh_count()
            self._status("Cleared all actors")

        _head(p, "ACTORS")
        _label(p, textvariable=count_var, fg=ACT)
        for lbl, cmd in [("Add actor (+offset)", _add),
                         ("Hide last",           _hide_last),
                         ("Show all",            _show_all),
                         ("Remove last",         _remove_last),
                         ("Clear all",           _clear_all)]:
            _btn(p, lbl, cmd)

        _add()   # start with one actor

    # ── Section: Picking ──────────────────────────────────────────────────────

    def _sec_picking(self, p: tk.Frame) -> None:
        pts, sc = _torus(100_000)
        self.scatter.set_points(pts, scalars=sc, colormap="plasma", point_size=3.0)

        pick_var = tk.StringVar(value="—")
        sel_var  = tk.StringVar(value="—")

        def _on_pick(event):
            w = event.widget
            if w.picked_point is not None:
                px, py, pz = w.picked_point
                pick_var.set(f"actor={w.picked_actor}  idx={w.picked_index}\n"
                              f"({px:.3f}, {py:.3f}, {pz:.3f})")

        def _on_select(event):
            sel = event.widget.selected or []
            sel_var.set(f"{len(sel)} point(s) selected")

        self.scatter.bind("<<PointPicked>>",      _on_pick)
        self.scatter.bind("<<SelectionChanged>>", _on_select)

        _head(p, "PICK MODE")
        mode_var = tk.StringVar(value="none")

        def _set_mode():
            m = mode_var.get()
            if   m == "point": self.scatter.enable_point_picking()
            elif m == "rect":  self.scatter.enable_rectangle_picking()
            elif m == "both":
                self.scatter.enable_point_picking()
                self.scatter.enable_rectangle_picking()
            else:
                self.scatter.disable_picking()

        _radio_group(p, mode_var,
                     [("Off",                   "none"),
                      ("Point pick (click)",    "point"),
                      ("Rect select (Shift+drag)", "rect"),
                      ("Both",                  "both")],
                     _set_mode)

        _head(p, "LAST PICKED POINT")
        _label(p, textvariable=pick_var, fg=ACT)

        _head(p, "SELECTION")
        _label(p, textvariable=sel_var, fg=WARN)

        _head(p, "NOTES")
        _label(p, text="Lasso selection available in the\n"
                        "\"Lasso Selection\" section.\n"
                        "Hover tooltips are planned.", fg=DIM)

    # ── Section: Camera ───────────────────────────────────────────────────────

    def _sec_camera(self, p: tk.Frame) -> None:
        pts, sc = _sphere(100_000)
        self.scatter.set_points(pts, scalars=sc, colormap="viridis", point_size=3.0)

        cam_info = tk.StringVar(value="—")
        saved: dict = {}

        _head(p, "PRESETS")
        for lbl, cmd in [("Reset",       self.scatter.reset_camera),
                         ("XY  (top)",   self.scatter.view_xy),
                         ("XZ  (front)", self.scatter.view_xz),
                         ("YZ  (side)",  self.scatter.view_yz),
                         ("Isometric",   self.scatter.view_isometric)]:
            _btn(p, lbl, cmd)

        _head(p, "FLATTEN VIEW")
        _label(p, "Snap to plane + orthographic:", fg=DIM)
        for plane_pair in [("XY (+Z)", "xy", "XY (−Z)", "xy-"),
                           ("XZ (+Y)", "xz", "XZ (−Y)", "xz-"),
                           ("YZ (+X)", "yz", "YZ (−X)", "yz-")]:
            row = tk.Frame(p, bg=PANEL)
            row.pack(fill="x")
            for i in range(0, 4, 2):
                lbl, pl = plane_pair[i], plane_pair[i + 1]
                tk.Button(
                    row, text=lbl, bg=PANEL, fg=FG, font=FM,
                    relief="flat", activebackground=ACT, cursor="hand2",
                    command=lambda pl=pl: self.scatter.flatten_view(pl),
                ).pack(side="left", expand=True, fill="x", padx=(0, 2), pady=1)

        _head(p, "PROJECTION")
        pp_var = tk.BooleanVar(value=False)
        _check(p, "Parallel projection",
               pp_var, lambda: setattr(self.scatter, "parallel_projection", pp_var.get()))

        _head(p, "SAVE / RESTORE STATE")
        def _save():
            saved.clear()
            saved.update(self.scatter.get_camera())
            cam_info.set("State saved.")
        def _restore():
            if saved:
                self.scatter.set_camera(saved)
                cam_info.set("State restored.")
            else:
                cam_info.set("Nothing saved yet.")
        _btn(p, "Save state",    _save)
        _btn(p, "Restore state", _restore)
        _label(p, textvariable=cam_info, fg=DIM)

        _head(p, "LINKED CAMERAS")
        linked_info = tk.StringVar(value="0 linked")

        def _open_linked():
            top = tk.Toplevel(self)
            top.title(f"Linked view {len(self._linked_wins) + 1}")
            top.geometry("600x500")
            sw = Scatter3D(top)
            sw.pack(fill="both", expand=True)
            sw.set_points(pts, scalars=sc, colormap="plasma", point_size=3.0)
            link_cameras(self.scatter, sw)
            self._linked_wins.append((top, sw))
            linked_info.set(f"{len(self._linked_wins)} linked")
            def _on_close():
                unlink_cameras(self.scatter, sw)
                self._linked_wins[:] = [
                    (t, w) for t, w in self._linked_wins if t is not top]
                linked_info.set(f"{len(self._linked_wins)} linked")
                top.destroy()
            top.protocol("WM_DELETE_WINDOW", _on_close)

        def _close_all():
            for top, sw in list(self._linked_wins):
                unlink_cameras(self.scatter, sw)
                top.destroy()
            self._linked_wins.clear()
            linked_info.set("0 linked")

        _btn(p, "Open linked window", _open_linked)
        _btn(p, "Close all linked",   _close_all, fg=RED)
        _label(p, textvariable=linked_info, fg=DIM)

    # ── Section: Overlays ─────────────────────────────────────────────────────

    def _sec_overlays(self, p: tk.Frame) -> None:
        pts, sc = _torus(100_000)
        self.scatter.set_points(pts, scalars=sc, colormap="plasma", point_size=3.0)

        ov_handles: list[int] = []
        ov_info = tk.StringVar(value="0 overlay(s)")

        def _upd():
            ov_info.set(f"{len(ov_handles)} overlay(s)")

        def _add_box():
            h = self.scatter.add_box((-3, -3, -2, 3, 3, 2), color=(1.0, 1.0, 0.0))
            if h >= 0:
                ov_handles.append(h)
                _upd()

        def _add_axes_lines():
            segs   = np.array([[-4,0,0, 4,0,0], [0,-4,0, 0,4,0], [0,0,-2, 0,0,2]],
                               dtype=np.float32)
            colors = [(1.0, 0.3, 0.3), (0.3, 1.0, 0.3), (0.3, 0.5, 1.0)]
            for i, color in enumerate(colors):
                h = self.scatter.add_lines(segs[i:i+1], color=color)
                if h >= 0:
                    ov_handles.append(h)
            _upd()

        def _clear_ov():
            self.scatter.clear_overlays()
            ov_handles.clear()
            _upd()

        _head(p, "LINE OVERLAYS")
        _label(p, textvariable=ov_info, fg=DIM)
        _btn(p, "Add bounding box",  _add_box)
        _btn(p, "Add axis lines",    _add_axes_lines)
        _btn(p, "Clear overlays",    _clear_ov)

        _head(p, "GRID")
        grid_var = tk.BooleanVar(value=True)
        _check(p, "Show grid",
               grid_var, lambda: self.scatter.show_grid(grid_var.get()))

        major_var = tk.BooleanVar(value=False)
        minor_var = tk.BooleanVar(value=False)
        def _update_planes(*_):
            self.scatter.show_grid_planes(major_var.get(), minor_var.get())
        _check(p, "Major grid planes", major_var, _update_planes)
        _check(p, "Minor grid planes", minor_var, _update_planes)

        _head(p, "TICK COUNT")
        _label(p, "Max ticks per axis (0 = auto):", fg=DIM)
        tick_vars = {ax: tk.IntVar(value=0) for ax in ("X", "Y", "Z")}

        def _apply_ticks(*_):
            def _val(ax):
                v = tick_vars[ax].get()
                return v if v > 0 else None
            self.scatter.set_ticks(x=_val("X"), y=_val("Y"), z=_val("Z"))

        for ax in ("X", "Y", "Z"):
            row = tk.Frame(p, bg=PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{ax}:", bg=PANEL, fg=FG, font=FM,
                     width=3, anchor="w").pack(side="left")
            tk.Scale(row, variable=tick_vars[ax], from_=0, to=20, resolution=1,
                     orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                     highlightthickness=0, command=_apply_ticks,
                     ).pack(side="left", fill="x", expand=True)

        axes_var = tk.BooleanVar(value=False)
        _check(p, "Orientation axes widget",
               axes_var, lambda: self.scatter.show_orientation_axes(axes_var.get()))

        _head(p, "SCALAR BAR")
        sb_var = tk.BooleanVar(value=False)
        def _toggle_sb():
            self.scatter.scalar_bar(sb_var.get(),
                                    vmin=float(sc.min()), vmax=float(sc.max()),
                                    colormap="plasma", title="radius")
        _check(p, "Show scalar bar", sb_var, _toggle_sb)

        _head(p, "BACKGROUND")
        for lbl, color in [("Dark (default)", (0.05, 0.05, 0.07)),
                            ("Black",          (0.0,  0.0,  0.0 )),
                            ("White",          (1.0,  1.0,  1.0 )),
                            ("Navy",           (0.05, 0.05, 0.20))]:
            _btn(p, lbl, lambda c=color: self.scatter.set_background(c), fg=FG)

    # ── Section: Export ───────────────────────────────────────────────────────

    def _sec_export(self, p: tk.Frame) -> None:
        pts, sc = _torus(100_000)
        self.scatter.set_points(pts, scalars=sc, colormap="plasma", point_size=3.0)

        exp_info  = tk.StringVar(value="—")
        n_frames  = tk.IntVar(value=60)
        gif_fps   = tk.IntVar(value=20)

        def _save_png():
            path = fd.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
                title="Save screenshot")
            if path:
                self.scatter.save_png(path)
                exp_info.set(f"Saved: {path.split('/')[-1]}")
                self._status(f"PNG saved: {path}")

        def _orbit_gif():
            path = fd.asksaveasfilename(
                defaultextension=".gif",
                filetypes=[("GIF animation", "*.gif"), ("All files", "*.*")],
                title="Save orbit GIF")
            if not path:
                return
            def _prog(i, total):
                exp_info.set(f"Frame {i + 1} / {total}")
                p.update()
            self.scatter.orbit_gif(path, n_frames=n_frames.get(),
                                   fps=gif_fps.get(), on_progress=_prog)
            exp_info.set(f"Saved {n_frames.get()} frames")
            self._status(f"Orbit GIF saved: {path}")

        _head(p, "SCREENSHOT")
        _btn(p, "Save PNG…", _save_png)

        _head(p, "ORBIT GIF")
        for lbl, var, lo, hi in [("Frames", n_frames, 10, 120),
                                  ("FPS",    gif_fps,   5,  30)]:
            row = tk.Frame(p, bg=PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=lbl, bg=PANEL, fg=FG,
                     font=FM, width=8, anchor="w").pack(side="left")
            tk.Scale(row, variable=var, from_=lo, to=hi, orient="horizontal",
                     bg=PANEL, fg=FG, troughcolor="#333", highlightthickness=0,
                     font=("Consolas", 9)).pack(side="left", fill="x", expand=True)
        _btn(p, "Save orbit GIF…", _orbit_gif)

        _head(p, "NOTES")
        _label(p, text="High-res PNG export planned.\n"
                        "SVG export planned (2D scenes only).", fg=DIM)

        _head(p, "STATUS")
        _label(p, textvariable=exp_info, fg=ACT)

    # ── Planned stubs ──────────────────────────────────────────────────────────

    def _planned_stub(self, p: tk.Frame, title: str, plan: str, notes: str) -> None:
        tk.Label(p, text=title, bg=PANEL, fg=PLAN, font=FH,
                 anchor="w").pack(fill="x", pady=(8, 4))
        tk.Label(p, text="NOT YET IMPLEMENTED", bg=PANEL, fg="#c62828",
                 font=FM, anchor="w").pack(fill="x")
        tk.Label(p, text=f"Plan: {plan}", bg=PANEL, fg=PLAN,
                 font=("Consolas", 9), anchor="w",
                 wraplength=230).pack(fill="x", pady=(2, 0))
        tk.Label(p, text=notes, bg=PANEL, fg=PLAN,
                 font=("Consolas", 9), anchor="w", justify="left",
                 wraplength=230).pack(fill="x", pady=(6, 0))

    # ── Section: Hover Tooltips ────────────────────────────────────────────────

    def _sec_hover(self, p: tk.Frame) -> None:
        try:
            import pandas as pd
        except ImportError:
            pd = None

        # Two datasets: raw numpy (coord-only tooltip) and DataFrame (enriched)
        n = 6_000
        pts_raw, sc_raw = _sphere(n)

        info_var   = tk.StringVar(value="—")
        mode_var   = tk.StringVar(value="dataframe")
        enabled_var = tk.BooleanVar(value=True)

        def _load(*_):
            mode = mode_var.get()
            if mode == "numpy":
                self.scatter.set_points(pts_raw, scalars=sc_raw,
                                        colormap="viridis", point_size=4.0)
                info_var.set("Raw numpy — tooltip shows\nx / y / z world coords.")
                self._status("Hover demo: raw numpy")
            elif mode == "dataframe" and pd is not None:
                pts_df, _ = _torus(n)
                df = pd.DataFrame({
                    "lon":  pts_df[:, 0].astype(float),
                    "lat":  pts_df[:, 1].astype(float),
                    "elev": pts_df[:, 2].astype(float),
                    "temp": (pts_df[:, 2] * 10 + 20
                             + RNG.standard_normal(n) * 2).astype(float),
                    "id":   [f"PT{i:04d}" for i in range(n)],
                })
                self.scatter.set_points(df, x="lon", y="lat", z="elev",
                                        color="elev", hover=["id", "temp"],
                                        colormap="plasma", point_size=4.0)
                info_var.set("DataFrame — tooltip shows\nlon / lat / elev\n+ id and temp columns.")
                self._status("Hover demo: DataFrame")
            else:
                self.scatter.set_points(pts_raw, scalars=sc_raw,
                                        colormap="viridis", point_size=4.0)
                info_var.set("pandas not installed — using raw numpy.")

        def _toggle_tooltip():
            self.scatter.hover_tooltip = enabled_var.get()
            self._status(f"hover_tooltip = {enabled_var.get()}")

        _head(p, "DATA SOURCE")
        for label, val in [("Raw numpy", "numpy"), ("DataFrame (pandas)", "dataframe")]:
            tk.Radiobutton(
                p, text=label, variable=mode_var, value=val,
                bg=PANEL, fg=FG, selectcolor=PANEL,
                activebackground=PANEL, activeforeground=ACT,
                font=FM, anchor="w", command=_load,
            ).pack(fill="x", pady=1)

        _head(p, "CONTROLS")
        tk.Checkbutton(
            p, text="hover_tooltip enabled", variable=enabled_var,
            bg=PANEL, fg=FG, selectcolor=PANEL,
            activebackground=PANEL, activeforeground=ACT,
            font=FM, anchor="w", command=_toggle_tooltip,
        ).pack(fill="x", pady=1)

        _head(p, "HOW IT WORKS")
        _label(p, text="Move the mouse over any point\n"
                        "and a tooltip appears near the cursor.\n\n"
                        "Tooltip contents:\n"
                        "• raw numpy → x / y / z world coords\n"
                        "• DataFrame → column names + hover cols\n\n"
                        "Debounce: 30 ms  ·  no Rust changes.", fg=DIM)

        _head(p, "STATUS")
        _label(p, textvariable=info_var, fg=ACT)

        _load()

    # ── Section: Categorical Color ─────────────────────────────────────────────

    def _sec_catcolor(self, p: tk.Frame) -> None:
        try:
            import pandas as pd
        except ImportError:
            _label(p, "pandas is not installed.\npip install pandas", fg=WARN)
            return

        n = 6_000
        pts, _ = _torus(n)

        # Three demo datasets showing different categorical scenarios
        species_labels = (["setosa"] * (n // 3)
                          + ["versicolor"] * (n // 3)
                          + ["virginica"] * (n - 2 * (n // 3)))
        cluster_ids = np.tile(np.arange(5), n // 5 + 1)[:n]
        many_labels = [f"group-{i % 14}" for i in range(n)]

        datasets = {
            "strings  (3 species)":     ("species",  species_labels),
            "integers (5 clusters)":    ("cluster",  cluster_ids.tolist()),
            "14 groups (palette cycle)":("group",    many_labels),
        }

        mode_var = tk.StringVar(value=list(datasets)[0])
        legend_var = tk.BooleanVar(value=True)
        pos_var = tk.StringVar(value="top-right")
        info_var = tk.StringVar(value="—")

        def _load(*_):
            name = mode_var.get()
            col, values = datasets[name]
            df = pd.DataFrame({
                "x": pts[:, 0].astype(float),
                "y": pts[:, 1].astype(float),
                "z": pts[:, 2].astype(float),
                col: values,
            })
            self.scatter.set_points(df, x="x", y="y", z="z",
                                    color=col, point_size=3.0)
            unique = len(set(values))
            info_var.set(f"{col!r}  —  {unique} categories\n{n:,} points")
            self._status(f"Categorical color: {col!r} ({unique} categories)")

        def _toggle_legend():
            self.scatter.show_legend(legend_var.get())

        def _set_position(*_):
            self.scatter.legend_position = pos_var.get()

        _head(p, "DATASET")
        for label in datasets:
            tk.Radiobutton(
                p, text=label, variable=mode_var, value=label,
                bg=PANEL, fg=FG, selectcolor=PANEL,
                activebackground=PANEL, activeforeground=ACT,
                font=FM, anchor="w", command=_load,
            ).pack(fill="x", pady=1)

        _head(p, "LEGEND")
        tk.Checkbutton(
            p, text="show legend", variable=legend_var,
            bg=PANEL, fg=FG, selectcolor=PANEL,
            activebackground=PANEL, activeforeground=ACT,
            font=FM, anchor="w", command=_toggle_legend,
        ).pack(fill="x", pady=1)

        row = tk.Frame(p, bg=PANEL)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="position", bg=PANEL, fg=FG,
                 font=FM, width=9, anchor="w").pack(side="left")
        om = tk.OptionMenu(row, pos_var,
                           "top-right", "top-left",
                           "bottom-right", "bottom-left",
                           command=_set_position)
        om.configure(bg="#222", fg=FG, activebackground="#333",
                     font=FM, bd=0, highlightthickness=0)
        om.pack(side="left", fill="x", expand=True)

        _head(p, "STATUS")
        _label(p, textvariable=info_var, fg=ACT)

        _load()

    # ── Section: Size by Column ───────────────────────────────────────────────

    def _sec_sizebycolumn(self, p: tk.Frame) -> None:
        try:
            import pandas as pd
        except ImportError:
            _label(p, "pandas is not installed.\npip install pandas", fg=WARN)
            return

        n = 4_000
        pts, _ = _torus(n)

        # Three scenarios: numeric column, NaN values, and raw point_sizes array
        magnitudes = np.linalg.norm(pts, axis=1).astype(float)
        magnitudes_nan = magnitudes.copy()
        magnitudes_nan[::8] = float("nan")

        datasets = {
            "magnitude (no NaN)": magnitudes,
            "magnitude (with NaN)": magnitudes_nan,
        }

        mode_var     = tk.StringVar(value=list(datasets)[0])
        range_lo_var = tk.DoubleVar(value=2.0)
        range_hi_var = tk.DoubleVar(value=16.0)
        info_var     = tk.StringVar(value="—")

        def _load(*_):
            name   = mode_var.get()
            lo, hi = range_lo_var.get(), range_hi_var.get()
            values = datasets[name]
            df = pd.DataFrame({
                "x":  pts[:, 0].astype(float),
                "y":  pts[:, 1].astype(float),
                "z":  pts[:, 2].astype(float),
                "mag": values,
            })
            self.scatter.set_points(df, x="x", y="y", z="z",
                                    size="mag", size_range=(lo, hi),
                                    colormap="plasma")
            n_nan = int(np.sum(~np.isfinite(values)))
            info_var.set(f"{n:,} pts  range=[{lo:.0f}, {hi:.0f}]px\n"
                         f"NaN values: {n_nan} → fallback size")
            self._status(f"Size by column: {name!r}  range=[{lo:.0f},{hi:.0f}]")

        def _load_raw(*_):
            """Raw point_sizes= array — bypass the DataFrame path."""
            rng2 = np.random.default_rng(55)
            sizes = rng2.uniform(range_lo_var.get(), range_hi_var.get(), n).astype(np.float32)
            self.scatter.set_points(pts, point_sizes=sizes, colormap="plasma")
            info_var.set(f"{n:,} pts  raw point_sizes array\n"
                         f"range=[{sizes.min():.1f}, {sizes.max():.1f}]px")
            self._status("Size by column: raw point_sizes array")

        _head(p, "SIZE COLUMN")
        for label in datasets:
            tk.Radiobutton(
                p, text=label, variable=mode_var, value=label,
                bg=PANEL, fg=FG, selectcolor=PANEL,
                activebackground=PANEL, activeforeground=ACT,
                font=FM, anchor="w", command=_load,
            ).pack(fill="x", pady=1)
        _btn(p, "Raw point_sizes= array", _load_raw)

        _head(p, "SIZE RANGE (px)")
        for lbl, var in [("min px", range_lo_var), ("max px", range_hi_var)]:
            row = tk.Frame(p, bg=PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=lbl, bg=PANEL, fg=FG,
                     font=FM, width=8, anchor="w").pack(side="left")
            tk.Scale(row, variable=var, from_=1.0, to=30.0, resolution=1.0,
                     orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                     highlightthickness=0, command=_load).pack(side="left", fill="x", expand=True)

        _head(p, "NOTES")
        _label(p, text="size= column values are mapped\n"
                        "linearly to [min_px, max_px].\n"
                        "NaN → fallback (midpoint) size.\n\n"
                        "point_sizes= accepts a raw float32\n"
                        "numpy array for direct control.", fg=DIM)

        _head(p, "STATUS")
        _label(p, textvariable=info_var, fg=ACT)

        _load()

    # ── Section: 2D Mode ──────────────────────────────────────────────────────

    def _sec_2dmode(self, p: tk.Frame) -> None:
        self._swap_scatter(Scatter2D)
        scatter = self.scatter   # local alias — always a Scatter2D here

        try:
            import pandas as _pd
            _has_pd = True
        except ImportError:
            _pd = None
            _has_pd = False

        n = 80_000
        info_var = tk.StringVar(value="—")

        def _load_numpy(*_):
            rng = np.random.default_rng(7)
            # 2D Gaussian clusters — purely X/Y data
            centers = [(-2, -2), (2, -2), (0, 2), (-2, 2), (2, 2)]
            chunks = []
            per = n // len(centers)
            for cx, cy in centers:
                c = rng.standard_normal((per, 2)).astype(np.float32) * 0.6
                c[:, 0] += cx
                c[:, 1] += cy
                chunks.append(c)
            xy = np.concatenate(chunks)
            pts = np.zeros((len(xy), 3), dtype=np.float32)
            pts[:, :2] = xy
            scalars = np.hypot(pts[:, 0], pts[:, 1]).astype(np.float32)
            scatter.set_points(pts, scalars=scalars, colormap="plasma", point_size=2.0)
            info_var.set(f"numpy  {len(pts):,} pts\n5 Gaussian clusters")
            self._status(f"2D demo: numpy {len(pts):,} pts")

        def _load_df(*_):
            if not _has_pd:
                info_var.set("pandas not installed")
                return
            rng = np.random.default_rng(8)
            # UMAP-style blobs — natural 2D scatter use case
            labels = [f"cluster_{i}" for i in range(6)]
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
            rows = []
            per = n // len(labels)
            for lbl, ang in zip(labels, angles):
                cx, cy = 3 * np.cos(ang), 3 * np.sin(ang)
                pts_l = rng.standard_normal((per, 2)) * 0.5
                pts_l[:, 0] += cx
                pts_l[:, 1] += cy
                for x, y in pts_l:
                    rows.append({"x": float(x), "y": float(y), "cluster": lbl})
            df = _pd.DataFrame(rows)
            scatter.set_points(df, x="x", y="y", color="cluster", point_size=2.0)
            info_var.set(f"DataFrame  {len(df):,} pts\n6 clusters  categorical color")
            self._status(f"2D demo: DataFrame {len(df):,} pts")

        def _reset(*_):
            scatter.reset_camera()
            self._status("Camera refit to data bounds")

        _head(p, "DATA SOURCE")
        _btn(p, "Load numpy (sphere)",   _load_numpy)
        _btn(p, "Load DataFrame (torus)", _load_df)

        _head(p, "CAMERA")
        _btn(p, "Reset / refit",  _reset)
        _label(p, text="Left-drag  →  pan\n"
                        "Scroll      →  zoom\n"
                        "Dbl-click →  reset", fg=DIM)

        _head(p, "GRID")
        grid_var_2d = tk.BooleanVar(value=True)
        _check(p, "Show grid",
               grid_var_2d, lambda: scatter.show_grid(grid_var_2d.get()))

        major_var_2d = tk.BooleanVar(value=False)
        minor_var_2d = tk.BooleanVar(value=False)
        def _update_planes_2d(*_):
            scatter.show_grid_planes(major_var_2d.get(), minor_var_2d.get())
        _check(p, "Major grid planes", major_var_2d, _update_planes_2d)
        _check(p, "Minor grid planes", minor_var_2d, _update_planes_2d)

        _head(p, "NOTES")
        _label(p, text="Scatter2D is a thin subclass of\n"
                        "Scatter3D. Same renderer, same\n"
                        "DataFrame / hover / picking API.\n\n"
                        "Camera at +Z → XY plane is the\n"
                        "screen (X right, Y up).\n"
                        "Parallel projection locked on.\n"
                        "Z is always zeroed.", fg=DIM)

        _head(p, "STATUS")
        _label(p, textvariable=info_var, fg=ACT)

        _load_numpy()

    # ── Section: Point Labels ─────────────────────────────────────────────────

    def _sec_labels(self, p: tk.Frame) -> None:
        # Five Gaussian clusters; label each centroid
        n_clusters = 5
        n_per      = 10_000
        rng        = np.random.default_rng(7)
        CLUSTER_COLORS = [
            (1.0, 0.4, 0.4), (0.4, 1.0, 0.4), (0.3, 0.6, 1.0),
            (1.0, 0.9, 0.2), (1.0, 0.5, 1.0),
        ]
        angles   = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
        centers  = [(3 * np.cos(a), 3 * np.sin(a), 0.0) for a in angles]
        names    = [f"Cluster {i}" for i in range(n_clusters)]

        chunks = []
        for cx, cy, cz in centers:
            c = rng.standard_normal((n_per, 3)).astype(np.float32) * 0.5
            c[:, 0] += cx; c[:, 1] += cy; c[:, 2] += cz
            chunks.append(c)
        all_pts = np.concatenate(chunks)
        scalars = np.repeat(np.arange(n_clusters, dtype=np.float32), n_per)

        label_handles: list[int] = []
        info_var  = tk.StringVar(value=f"{len(all_pts):,} pts  ·  {n_clusters} labels")
        vis_var   = tk.BooleanVar(value=True)

        def _load():
            self.scatter.clear_labels()
            label_handles.clear()
            self.scatter.set_points(all_pts, scalars=scalars,
                                    colormap="plasma", point_size=2.0)
            for i, (cx, cy, cz) in enumerate(centers):
                h = self.scatter.add_label(
                    (cx, cy, cz + 0.6), names[i],
                    color=CLUSTER_COLORS[i], size=13.0, anchor="bottom",
                )
                label_handles.append(h)
            info_var.set(f"{len(all_pts):,} pts  ·  {n_clusters} labels")
            self._status("Labels demo loaded")

        def _rename():
            for i, h in enumerate(label_handles):
                self.scatter.update_label(h, text=f"Grp {i}  (n={n_per:,})")
            info_var.set("Labels renamed")

        def _move_up():
            for i, h in enumerate(label_handles):
                cx, cy, cz = centers[i]
                self.scatter.update_label(h, (cx, cy, cz + 1.5))
            info_var.set("Labels moved up")

        def _resize():
            for h in label_handles:
                self.scatter.update_label(h, size=20.0)
            info_var.set("Labels resized → 20 pt")

        def _toggle_vis():
            v = vis_var.get()
            for h in label_handles:
                self.scatter.set_label_visibility(h, v)
            info_var.set(f"Labels {'shown' if v else 'hidden'}")

        def _remove_one():
            if label_handles:
                h = label_handles.pop()
                self.scatter.remove_label(h)
                info_var.set(f"{len(label_handles)} label(s) remaining")

        def _clear():
            self.scatter.clear_labels()
            label_handles.clear()
            info_var.set("Labels cleared")

        def _anchor_demo():
            self.scatter.clear_labels()
            label_handles.clear()
            pos = (0.0, 0.0, 0.0)
            for i, (anchor, dy) in enumerate(
                    [("center", 0), ("left", 0), ("right", 0),
                     ("top", 0), ("bottom", 0)]):
                offset = (i - 2) * 2.5
                h = self.scatter.add_label(
                    (offset, 0.0, 0.0), anchor,
                    color=(0.9, 0.9, 0.9), size=13.0, anchor=anchor,
                )
                label_handles.append(h)
            info_var.set("Anchor demo — 5 alignment modes")

        _head(p, "SCENE")
        _btn(p, "Load (cluster centroids)", _load)
        _btn(p, "Anchor alignment demo",    _anchor_demo)

        _head(p, "MUTATIONS")
        _btn(p, "Rename labels",    _rename)
        _btn(p, "Move labels up",   _move_up)
        _btn(p, "Resize → 20 pt",   _resize)

        _head(p, "VISIBILITY")
        _check(p, "Show labels", vis_var, _toggle_vis)

        _head(p, "REMOVAL")
        _btn(p, "Remove last label", _remove_one, fg=WARN)
        _btn(p, "Clear all labels",  _clear,      fg=RED)

        _head(p, "STATUS")
        _label(p, textvariable=info_var, fg=ACT)

        _head(p, "NOTES")
        _label(p, text="anchor=  center | left | right | top | bottom\n"
                        "color=   (r,g,b) or \"#RRGGBB\"\n"
                        "size=    font size in points\n\n"
                        "Labels project from 3D world space\n"
                        "every frame — no screen-space drift.", fg=DIM)

        _load()

    # ── Section: Figure Subplots ──────────────────────────────────────────────

    def _sec_figure(self, p: tk.Frame) -> None:
        # Destroy any previous Figure from a re-entry before creating a new one.
        if self._figure is not None:
            self._figure.destroy()
            self._figure = None
        # Hide the shared scatter widget; the Figure occupies the right panel.
        self.scatter.pack_forget()
        fig = Figure(self._scatter_parent, rows=2, cols=2,
                     width=1040, height=780, padding=4)
        fig.pack(fill="both", expand=True)
        self._figure = fig

        rng = np.random.default_rng(42)
        n = 20_000
        pts_torus, sc_torus = _torus(n)
        pts_sphere, sc_sphere = _sphere(n)
        pts_helix, sc_helix = _helix(n)
        pts_gauss, sc_gauss = _gaussian(n)

        info_var  = tk.StringVar(value="—")
        link_var  = tk.BooleanVar(value=False)

        DATASETS = [
            (pts_torus,  sc_torus,  "plasma",   "Torus"),
            (pts_sphere, sc_sphere, "viridis",  "Sphere"),
            (pts_helix,  sc_helix,  "inferno",  "Helix"),
            (pts_gauss,  sc_gauss,  "coolwarm", "Gaussian"),
        ]
        cells = [(0, 0), (0, 1), (1, 0), (1, 1)]

        def _load(*_):
            for (r, c), (pts, sc, cmap, name) in zip(cells, DATASETS):
                fig[r, c].set_points(pts, scalars=sc, colormap=cmap,
                                     point_size=2.0)
                fig[r, c].add_label(
                    (float(pts[:, 0].mean()), float(pts[:, 1].mean()),
                     float(pts[:, 2].max()) + 0.3),
                    name, color=(1.0, 0.9, 0.7), size=12.0, anchor="bottom",
                )
            info_var.set(f"4 subplots  ·  {n:,} pts each")
            self._status("Figure demo loaded")

        def _toggle_link():
            if link_var.get():
                fig.link_cameras((0, 0), (0, 1), (1, 0), (1, 1))
                info_var.set("Cameras linked — orbit any subplot")
                self._status("Cameras linked")
            else:
                unlink_cameras(*fig.axes)
                info_var.set("Cameras unlinked")
                self._status("Cameras unlinked")

        def _link_top_row():
            fig.link_cameras((0, 0), (0, 1))
            info_var.set("Top row cameras linked")
            self._status("Top row cameras linked")

        def _scalar_bar_row0():
            fig.scalar_bar(row=0, colormap="plasma", vmin=0.0, vmax=1.0,
                           title="scalar")
            info_var.set("Scalar bar on row 0 (rightmost cell)")

        def _scalar_bar_all():
            fig.scalar_bar(colormap="viridis", vmin=0.0, vmax=1.0)
            info_var.set("Scalar bar on all rows (rightmost cell each)")

        def _reset_cameras():
            for w in fig.axes:
                w.reset_camera()
            info_var.set("All cameras reset")

        _head(p, "SCENE")
        _btn(p, "Load all subplots", _load)

        _head(p, "CAMERAS")
        _check(p, "Link all cameras", link_var, _toggle_link)
        _btn(p, "Link top row only", _link_top_row)
        _btn(p, "Reset all cameras", _reset_cameras)

        _head(p, "SCALAR BAR  (v1 workaround)")
        _btn(p, "scalar_bar(row=0)",  _scalar_bar_row0)
        _btn(p, "scalar_bar(all)",    _scalar_bar_all)

        _head(p, "NOTES")
        _label(p, text="Figure(rows=2, cols=2, …)\n"
                        "fig[row, col]  →  Scatter3D\n"
                        "fig.axes       →  flat list\n"
                        "fig.link_cameras(*cells)\n\n"
                        "scalar_bar(row=r) shows the bar\n"
                        "in the rightmost cell per row\n"
                        "(v1 workaround — not a shared bar).", fg=DIM)

        _head(p, "STATUS")
        _label(p, textvariable=info_var, fg=ACT)

        _load()

    def _sec_stat_overlays(self, p: tk.Frame) -> None:
        try:
            from scipy.spatial import ConvexHull as _CH  # noqa: F401
            _has_scipy = True
        except ImportError:
            _has_scipy = False

        if not _has_scipy:
            _head(p, "SCIPY NOT INSTALLED")
            _label(p, text="Install with:  pip install dragonsci[stats]", fg=WARN)
            return

        rng = np.random.default_rng(7)
        n_per_cluster = 300
        n_clusters = 4
        centers = np.array([
            [0, 0, 0], [4, 0, 0], [2, 4, 0], [2, 2, 4],
        ], dtype=np.float32)
        pts_list, lbl_list = [], []
        for i, c in enumerate(centers):
            cluster = rng.standard_normal((n_per_cluster, 3)).astype(np.float32) * 0.8 + c
            pts_list.append(cluster)
            lbl_list.extend([i] * n_per_cluster)
        all_pts = np.vstack(pts_list)
        labels  = np.array(lbl_list)

        mesh_handles: "list[int]" = []
        overlay_var   = tk.StringVar(value="none")
        opacity_var   = tk.DoubleVar(value=0.25)
        nstd_var      = tk.DoubleVar(value=2.0)
        wireframe_var = tk.BooleanVar(value=False)

        def _rebuild(*_):
            self.scatter.clear_meshes()
            mesh_handles.clear()
            mode    = overlay_var.get()
            opacity = opacity_var.get()
            n_std   = nstd_var.get()
            wf      = wireframe_var.get()
            if mode == "hulls":
                hs = self.scatter.add_cluster_hulls(
                    all_pts, labels, opacity=opacity)
                if wf:
                    for h in hs:
                        self.scatter.update_convex_hull(h, wireframe=True)
                mesh_handles.extend(hs)
            elif mode == "ellipsoids":
                hs = self.scatter.add_cluster_ellipsoids(
                    all_pts, labels, opacity=opacity, n_std=n_std)
                if wf:
                    for h in hs:
                        self.scatter.update_ellipsoid(h, wireframe=True)
                mesh_handles.extend(hs)
            elif mode == "both":
                hs1 = self.scatter.add_cluster_hulls(
                    all_pts, labels, opacity=max(0.1, opacity * 0.6))
                hs2 = self.scatter.add_cluster_ellipsoids(
                    all_pts, labels, opacity=opacity, n_std=n_std)
                if wf:
                    for h in hs1 + hs2:
                        self.scatter.update_convex_hull(h, wireframe=True) \
                            if h in hs1 else \
                            self.scatter.update_ellipsoid(h, wireframe=True)
                mesh_handles.extend(hs1 + hs2)

        def _load():
            self.scatter.set_points(all_pts,
                                    scalars=labels.astype(np.float32),
                                    colormap="tab10", point_size=3.0)
            _rebuild()

        _load()

        _head(p, "OVERLAY TYPE")
        _radio_group(p, overlay_var,
                     [("None",               "none"),
                      ("Convex Hulls",       "hulls"),
                      ("Ellipsoids",         "ellipsoids"),
                      ("Hulls + Ellipsoids", "both")],
                     lambda: _rebuild())

        _head(p, "WIREFRAME")
        tk.Checkbutton(p, text="Draw as wireframe", variable=wireframe_var,
                       bg=PANEL, fg=FG, selectcolor=PANEL, activebackground=PANEL,
                       activeforeground=ACT, font=FM,
                       command=_rebuild).pack(anchor="w", padx=6, pady=2)

        _head(p, "OPACITY")
        tk.Scale(p, variable=opacity_var, from_=0.05, to=1.0, resolution=0.05,
                 orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                 highlightthickness=0, command=lambda _: _rebuild()
                 ).pack(fill="x", padx=6, pady=2)

        _head(p, "ELLIPSOID N_STD")
        tk.Scale(p, variable=nstd_var, from_=0.5, to=4.0, resolution=0.25,
                 orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                 highlightthickness=0, command=lambda _: _rebuild()
                 ).pack(fill="x", padx=6, pady=2)

        _head(p, "CONTROLS")
        _btn(p, "Reload Points", _load, fg=ACT)
        _btn(p, "Clear Overlays",
             lambda: (self.scatter.clear_meshes(), mesh_handles.clear()),
             fg=WARN)

    def _sec_lasso(self, p: tk.Frame) -> None:
        try:
            import pandas as _pd
            _has_pd = True
        except ImportError:
            _pd = None
            _has_pd = False

        rng = np.random.default_rng(99)
        n = 60_000
        info_var = tk.StringVar(value="—")
        sel_var  = tk.StringVar(value="Draw a selection")
        mode_var = tk.StringVar(value="none")

        # Load numpy data by default; optionally load DataFrame for index demo.
        def _load_numpy(*_):
            pts, sc = _torus(n)
            self.scatter.set_points(pts, scalars=sc, colormap="plasma", point_size=2.0)
            info_var.set(f"numpy  {n:,} pts  (torus)")
            sel_var.set("Draw a selection")
            self._status(f"Lasso demo: {n:,} pts loaded")

        def _load_df(*_):
            if not _has_pd:
                info_var.set("pandas not installed")
                return
            labels = [f"grp_{i}" for i in range(5)]
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
            rows = []
            per = n // len(labels)
            for lbl, ang in zip(labels, angles):
                cx, cy = 3 * np.cos(ang), 3 * np.sin(ang)
                noise = rng.standard_normal((per, 3)).astype(np.float32) * 0.5
                noise[:, 0] += cx
                noise[:, 1] += cy
                for x, y, z in noise:
                    rows.append({"x": float(x), "y": float(y), "z": float(z), "grp": lbl})
            df = _pd.DataFrame(rows)
            self.scatter.set_points(df, x="x", y="y", z="z", color="grp", point_size=2.0)
            info_var.set(f"DataFrame  {len(df):,} pts  5 groups")
            sel_var.set("Draw a selection")
            self._status(f"Lasso demo: DataFrame {len(df):,} pts loaded")

        def _on_select(event):
            w = event.widget
            idx = w.selected_indices or []
            vals = w.selected_index_values
            if not idx:
                sel_var.set("(empty)")
                return
            preview = ", ".join(str(i) for i in idx[:5])
            if len(idx) > 5:
                preview += f" … (+{len(idx) - 5} more)"
            line2 = ""
            if vals is not None:
                vpreview = ", ".join(str(v) for v in vals[:3])
                if len(vals) > 3:
                    vpreview += " …"
                line2 = f"\nindex_values: {vpreview}"
            sel_var.set(f"{len(idx)} point(s) selected\niloc: {preview}{line2}")
            self._status(f"Selection: {len(idx)} points")

        self.scatter.bind("<<SelectionChanged>>", _on_select)

        def _set_mode():
            m = mode_var.get()
            if m == "lasso":
                self.scatter.enable_lasso_picking()
            elif m == "rect":
                self.scatter.enable_rectangle_picking()
            elif m == "both":
                self.scatter.enable_lasso_picking()
                self.scatter.enable_rectangle_picking()
            else:
                self.scatter.disable_picking()
                sel_var.set("Draw a selection")

        _head(p, "DATA")
        _btn(p, "Load numpy (torus)",      _load_numpy)
        _btn(p, "Load DataFrame (blobs)",  _load_df)

        _head(p, "STATUS")
        _label(p, textvariable=info_var, fg=ACT)

        _head(p, "PICK MODE")
        _radio_group(p, mode_var,
                     [("Off",                       "none"),
                      ("Lasso  (Ctrl+drag)",         "lasso"),
                      ("Rect   (Shift+drag)",        "rect"),
                      ("Both",                       "both")],
                     _set_mode)

        _head(p, "SELECTION")
        _label(p, textvariable=sel_var, fg=WARN)

        _head(p, "NOTES")
        _label(p, text="selected_indices  → iloc positions\n"
                        "selected_index_values → pandas labels\n"
                        "  (None for numpy arrays)\n\n"
                        "Both lasso and rect fire\n"
                        "<<SelectionChanged>>.", fg=DIM)

        _load_numpy()
        mode_var.set("lasso")
        _set_mode()

    def _sec_streaming(self, p: tk.Frame) -> None:
        stream_handle: "list[int | None]" = [None]
        running:       "list[bool]"        = [False]
        after_id:      "list[str | None]"  = [None]
        total_pts:     "list[int]"         = [0]

        n_var    = tk.IntVar(value=500)
        rate_var = tk.IntVar(value=33)    # ms between ticks ≈ 30 fps
        max_var  = tk.IntVar(value=50_000)
        mode_var = tk.StringVar(value="ring")
        shape_var = tk.StringVar(value="helix")
        cmap_var  = tk.StringVar(value="plasma")
        count_var = tk.StringVar(value="0 pts streamed")

        def _gen_pts(n: int):
            t_start = total_pts[0] / 2_000.0
            t_end   = (total_pts[0] + n) / 2_000.0
            t = np.linspace(t_start, t_end, n, dtype=np.float32)
            shape = shape_var.get()
            if shape == "helix":
                pts = np.stack([np.cos(t * 8), np.sin(t * 8), t * 0.5], axis=1)
                scl = (t * 0.5).astype(np.float32)
            elif shape == "wave":
                decay = np.exp(-t * 0.3).astype(np.float32)
                pts = np.stack([
                    t - t_start,
                    np.sin(t * 12) * decay,
                    np.cos(t * 7)  * decay * 0.5,
                ], axis=1)
                scl = decay
            else:  # "random"
                pts = RNG.standard_normal((n, 3)).astype(np.float32) * 1.5
                scl = np.linalg.norm(pts, axis=1).astype(np.float32)
            return pts, scl

        def _make_stream():
            if stream_handle[0] is not None:
                self.scatter.remove_actor(stream_handle[0])
            stream_handle[0] = self.scatter.add_stream(
                max_points=max_var.get(),
                mode=mode_var.get(),
            )
            total_pts[0] = 0
            count_var.set("0 pts streamed")

        def _tick():
            if not running[0] or stream_handle[0] is None:
                after_id[0] = None
                return
            pts, scl = _gen_pts(n_var.get())
            self.scatter.stream(stream_handle[0], pts, scalars=scl,
                                colormap=cmap_var.get(), point_size=2.0)
            total_pts[0] += n_var.get()
            count_var.set(f"{total_pts[0]:,} pts streamed")
            after_id[0] = self.scatter.after(rate_var.get(), _tick)

        def _start():
            if stream_handle[0] is None:
                _make_stream()
            running[0] = True
            if after_id[0] is None:
                _tick()
            self._status("Streaming…")

        def _pause():
            running[0] = False
            self._status("Paused.")

        def _reset():
            was_running = running[0]
            running[0] = False
            if after_id[0] is not None:
                try:
                    self.scatter.after_cancel(after_id[0])
                except Exception:
                    pass
                after_id[0] = None
            _make_stream()
            if was_running:
                _start()

        def _cleanup(*_):
            running[0] = False
            if after_id[0] is not None:
                try:
                    self.scatter.after_cancel(after_id[0])
                except Exception:
                    pass
                after_id[0] = None

        p.bind("<Destroy>", lambda e: _cleanup() if e.widget is p else None)

        _head(p, "BUFFER")
        _label(p, "max_points (capacity):", fg=DIM)
        tk.Scale(p, variable=max_var, from_=5_000, to=500_000, resolution=5_000,
                 orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                 highlightthickness=0, command=lambda _: _reset()).pack(fill="x")

        _head(p, "RATE")
        for lbl, var, lo, hi, res in [
            ("pts / tick", n_var,    100, 5_000, 100),
            ("ms / tick",  rate_var,  16,   500,   1),
        ]:
            row = tk.Frame(p, bg=PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=lbl, bg=PANEL, fg=FG, font=FM,
                     width=11, anchor="w").pack(side="left")
            tk.Scale(row, variable=var, from_=lo, to=hi, resolution=res,
                     orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                     highlightthickness=0).pack(side="left", fill="x", expand=True)

        _head(p, "MODE")
        _radio_group(p, mode_var,
                     [("Ring  — overwrite oldest", "ring"),
                      ("Append — stop when full",  "append")],
                     lambda: _reset())

        _head(p, "DATA SHAPE")
        _radio_group(p, shape_var,
                     [("Helix",  "helix"),
                      ("Wave",   "wave"),
                      ("Random", "random")],
                     lambda: _reset())

        _head(p, "COLORMAP")
        _radio_group(p, cmap_var,
                     [(c, c) for c in ["plasma", "viridis", "turbo", "coolwarm"]],
                     lambda: _reset())

        _head(p, "CONTROLS")
        _btn(p, "Start / Resume",  _start, fg=ACT)
        _btn(p, "Pause",           _pause, fg=WARN)
        _btn(p, "Reset / Rebuild", _reset, fg=RED)

        _head(p, "STATUS")
        _label(p, textvariable=count_var, fg=ACT)

        # Create stream actor and start immediately
        _make_stream()
        _start()


    # ── Section: Marginal Histograms ─────────────────────────────────────────

    def _sec_marginals(self, p: tk.Frame) -> None:
        self._swap_scatter(Scatter2D)
        scatter = self.scatter

        rng = np.random.default_rng(99)
        info_var = tk.StringVar(value="—")

        # Controls
        bins_var    = tk.IntVar(value=50)
        alpha_var   = tk.DoubleVar(value=0.7)
        size_var    = tk.IntVar(value=80)
        orient_var  = tk.StringVar(value="both")
        color_var   = tk.StringVar(value="#4c8eff")
        visible_var = tk.BooleanVar(value=True)

        def _apply_marginals(*_):
            if visible_var.get():
                scatter.show_marginals(
                    True,
                    bins=bins_var.get(),
                    color=color_var.get(),
                    alpha=alpha_var.get(),
                    size=size_var.get(),
                    orientation=orient_var.get(),
                )
            else:
                scatter.show_marginals(False)

        def _load_clusters(*_):
            n = 60_000
            centers = [(-2, -2), (2, -2), (0, 2.5), (-2.5, 1.5), (2.5, 1.5)]
            chunks = []
            for cx, cy in centers:
                c = rng.standard_normal((n // len(centers), 2)).astype(np.float32) * 0.5
                c[:, 0] += cx
                c[:, 1] += cy
                chunks.append(c)
            xy = np.concatenate(chunks)
            pts = np.zeros((len(xy), 3), dtype=np.float32)
            pts[:, :2] = xy
            scalars = np.hypot(pts[:, 0], pts[:, 1]).astype(np.float32)
            scatter.set_points(pts, scalars=scalars, colormap="plasma", point_size=2.0)
            info_var.set(f"{len(pts):,} pts  5 clusters")
            _apply_marginals()

        def _load_bivariate(*_):
            n = 40_000
            pts = np.zeros((n, 3), dtype=np.float32)
            pts[:, 0] = rng.standard_normal(n).astype(np.float32)
            pts[:, 1] = (pts[:, 0] * 0.7 + rng.standard_normal(n) * 0.5).astype(np.float32)
            scatter.set_points(pts, point_size=1.5)
            info_var.set(f"{n:,} pts  bivariate normal")
            _apply_marginals()

        def _toggle_marginals(*_):
            _apply_marginals()

        _head(p, "DATA")
        _btn(p, "Load clusters",    _load_clusters)
        _btn(p, "Load bivariate",   _load_bivariate)

        _head(p, "MARGINALS")
        _check(p, "Show marginals", visible_var, _toggle_marginals)

        _head(p, "ORIENTATION")
        _radio_group(p, orient_var,
                     [("Both",    "both"),
                      ("X only",  "x"),
                      ("Y only",  "y")],
                     _apply_marginals)

        _head(p, "BINS")
        tk.Scale(p, variable=bins_var, from_=5, to=200, resolution=5,
                 orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                 highlightthickness=0, command=_apply_marginals).pack(fill="x")

        _head(p, "OPACITY")
        tk.Scale(p, variable=alpha_var, from_=0.1, to=1.0, resolution=0.05,
                 orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                 highlightthickness=0, command=_apply_marginals).pack(fill="x")

        _head(p, "SIZE (px)")
        tk.Scale(p, variable=size_var, from_=30, to=200, resolution=5,
                 orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                 highlightthickness=0, command=_apply_marginals).pack(fill="x")

        _head(p, "COLOR")
        _radio_group(p, color_var,
                     [("Blue",   "#4c8eff"),
                      ("Teal",   "#26c6da"),
                      ("Orange", "#ffa726"),
                      ("Pink",   "#ec407a")],
                     _apply_marginals)

        _head(p, "STATUS")
        _label(p, textvariable=info_var, fg=ACT)

        _load_clusters()

    def _sec_line2d(self, p: tk.Frame) -> None:
        self._swap_scatter(Line2D)
        w = self.scatter          # type: Line2D

        # ── State ────────────────────────────────────────────────────────────
        running:   "list[bool]"       = [False]
        after_id:  "list[str | None]" = [None]
        total_pts: "list[int]"        = [0]

        color_map = {
            "Blue":   (0.3,  0.7,  1.0),
            "Orange": (1.0,  0.55, 0.1),
            "Green":  (0.2,  0.9,  0.4),
            "Pink":   (0.95, 0.4,  0.75),
        }

        n_var     = tk.IntVar(value=50)
        rate_var  = tk.IntVar(value=30)
        max_var   = tk.IntVar(value=3_000)
        shape_var = tk.StringVar(value="sine")
        line_a_color_var = tk.StringVar(value="Blue")
        line_b_color_var = tk.StringVar(value="Orange")
        line_c_color_var = tk.StringVar(value="Green")
        stream_color_var = tk.StringVar(value="Blue")
        line_a_width_var = tk.DoubleVar(value=2.5)
        line_b_width_var = tk.DoubleVar(value=2.5)
        line_c_width_var = tk.DoubleVar(value=2.5)
        stream_width_var = tk.DoubleVar(value=2.5)
        y_tick_var = tk.DoubleVar(value=1.0)
        current_plot: "list[str]" = ["static"]
        count_var = tk.StringVar(value="—")

        # ── Helpers ───────────────────────────────────────────────────────────
        def _color_dropdown(parent: tk.Frame, label: str, var: tk.StringVar, cmd) -> None:
            row = tk.Frame(parent, bg=PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, bg=PANEL, fg=FG, font=FM,
                     anchor="w").pack(side="left")
            menu = tk.OptionMenu(row, var, *color_map.keys(), command=lambda *_: cmd())
            menu.configure(bg="#222", fg=FG, activebackground="#333",
                           activeforeground=ACT, relief="flat", bd=0,
                           font=FM, highlightthickness=0)
            menu["menu"].configure(bg="#222", fg=FG, activebackground="#333",
                                   activeforeground=ACT, font=FM)
            menu.pack(side="right", fill="x", expand=True)

        def _reset_widget():
            """Clear all line geometry and reset axis state."""
            _stop_stream()
            # Remove the active time-series stream if there is one.
            if ts_handle[0] is not None:
                try:
                    w.remove_line_stream(ts_handle[0])
                except Exception:
                    pass
                ts_handle[0] = None
            w.clear_overlays()
            # Remove primary line from renderer if present.
            if w._primary_handle is not None:
                if w._renderer is not None:
                    w._renderer.chart2d_remove_line(w._primary_handle)
                w._primary_handle = None
            w._pending_primary = None
            # Remove named lines from renderer.
            if w._renderer is not None:
                for h in list(w._named_lines):
                    w._renderer.chart2d_remove_line(h)
            w._named_lines.clear()
            w._pending_named_lines.clear()
            w._nhandle_map.clear()
            w._xlim = None
            w._ylim = None
            w._x_limits_frozen = False
            w._y_limits_frozen = False
            w._limits_frozen = False
            w._chart2d_sent = False
            w.set_y_tick_interval(y_tick_var.get())

        # ── Static examples ───────────────────────────────────────────────────
        def _load_static(*_):
            _reset_widget()
            current_plot[0] = "static"
            x = np.linspace(0, 4 * np.pi, 1_000, dtype=np.float32)
            # set_line triggers _auto_fit on first call → derives xlim/ylim from data
            w.set_line(
                x,
                np.sin(x),
                color=color_map[line_a_color_var.get()],
                line_width=line_a_width_var.get(),
            )
            w.add_line(
                x,
                np.cos(x),
                color=color_map[line_b_color_var.get()],
                line_width=line_b_width_var.get(),
            )
            w.add_line(
                x,
                np.sin(2 * x) * 0.5,
                color=color_map[line_c_color_var.get()],
                line_width=line_c_width_var.get(),
            )
            count_var.set("3 static lines")
            self._status("Line2D — static plot loaded")

        def _load_lissajous(*_):
            _reset_widget()
            current_plot[0] = "lissajous"
            t = np.linspace(0, 2 * np.pi, 2_000, dtype=np.float32)
            colors = [
                color_map[line_a_color_var.get()],
                color_map[line_b_color_var.get()],
                color_map[line_c_color_var.get()],
            ]
            widths = [
                line_a_width_var.get(),
                line_b_width_var.get(),
                line_c_width_var.get(),
            ]
            for i, (a, b, delta) in enumerate([(3, 2, np.pi / 4),
                                               (5, 4, np.pi / 2),
                                               (7, 6, np.pi / 3)]):
                w.add_line(np.sin(a * t + delta), np.sin(b * t),
                           color=colors[i],
                           line_width=widths[i])
            count_var.set("3 Lissajous curves")
            self._status("Line2D — Lissajous curves")

        # ── Streaming ─────────────────────────────────────────────────────────
        def _apply_static_appearance() -> None:
            if running[0]:
                return
            if current_plot[0] == "static":
                _load_static()
            elif current_plot[0] == "lissajous":
                _load_lissajous()

        # Time-series style: x = elapsed wall time (seconds), sliding window.
        # Data tick: emits sample batches at the requested callback cadence.
        # Axis tick: slides xlim at ~60 fps using the same wall-clock origin.
        ts_handle:         "list[int | None]" = [None]
        ts_start_wall:     "list[float]"      = [0.0]
        ts_last_batch_wall:"list[float]"      = [0.0]
        noise_y_last:      "list[float]"      = [0.0]
        anim_id:           "list[str | None]" = [None]
        max_scale:         list               = [None]  # Scale widget ref for disable during stream
        WINDOW_SEC = 10.0

        def _gen_samples(n: int):
            """Return (x_times, y_values) for the elapsed wall-time since the last batch."""
            now_wall = time.monotonic()
            t0_wall = ts_last_batch_wall[0]
            t1_wall = max(now_wall, t0_wall + 1e-6)
            dt_sample = (t1_wall - t0_wall) / max(n, 1)
            t0 = t0_wall - ts_start_wall[0]
            times = (t0 + (np.arange(n, dtype=np.float64) + 1.0) * dt_sample).astype(np.float32)
            ts_last_batch_wall[0] = t1_wall
            shape = shape_var.get()
            if shape == "sine":
                y = np.sin(times * 4).astype(np.float32)
            elif shape == "sawtooth":
                y = ((times % 1.0) * 2 - 1).astype(np.float32)
            else:  # noise / random walk
                steps = RNG.standard_normal(n).astype(np.float32) * 0.08
                y = np.cumsum(steps, dtype=np.float32) + noise_y_last[0]
                noise_y_last[0] = float(y[-1])
            return times, y

        def _make_stream():
            if ts_handle[0] is not None:
                w.remove_line_stream(ts_handle[0])
                ts_handle[0] = None
            current_plot[0] = "stream"
            seed_now = time.monotonic()
            seed_dt = rate_var.get() / 1000.0
            ts_start_wall[0] = seed_now - seed_dt
            ts_last_batch_wall[0] = ts_start_wall[0]
            noise_y_last[0] = 0.0
            clr = color_map[stream_color_var.get()]
            ts_handle[0] = w.add_line_stream(
                max_points=max_var.get(),
                mode="ring",
                color=clr,
                line_width=stream_width_var.get(),
            )
            w.set_y_tick_interval(y_tick_var.get())
            w.set_xlim(0.0, WINDOW_SEC)
            count_var.set("0 pts")

        def _apply_stream_appearance() -> None:
            if ts_handle[0] is None:
                return
            st = w._line_streams.get(ts_handle[0])
            if st is None:
                return
            st["color"] = color_map[stream_color_var.get()]
            st["line_width"] = float(stream_width_var.get())
            if st["render_handle"] is None or w._renderer is None:
                return
            xs, ys = w._stream_ordered(st)
            if xs is None:
                return
            w._renderer.chart2d_update_line(
                st["render_handle"], xs, ys, st["color"], st["line_width"])
            w._mark_dirty()
            self._status("Line2D stream appearance updated")

        def _axis_animate():
            """Runs at ~60 fps to slide xlim smoothly, independent of data rate."""
            if not running[0]:
                anim_id[0] = None
                return
            t_now = time.monotonic() - ts_start_wall[0]
            w.set_xlim(max(0.0, t_now - WINDOW_SEC), max(WINDOW_SEC, t_now))
            anim_id[0] = w.after(16, _axis_animate)

        def _tick():
            if not running[0] or ts_handle[0] is None:
                after_id[0] = None
                return
            n  = n_var.get()
            ms = rate_var.get()
            xs, ys = _gen_samples(n)
            w.stream_line(ts_handle[0], xs, ys)
            total_pts[0] += n
            count_var.set(f"{total_pts[0]:,} pts streamed")
            after_id[0] = w.after(ms, _tick)

        def _set_stream_controls(streaming: bool) -> None:
            if max_scale[0] is not None:
                max_scale[0].configure(state="disabled" if streaming else "normal")

        def _start_stream():
            if running[0]:
                return
            _reset_widget()
            _make_stream()
            running[0] = True
            _tick()
            _axis_animate()
            _set_stream_controls(True)
            self._status("Line2D stream running")

        def _stop_stream():
            running[0] = False
            for id_cell in (after_id, anim_id):
                if id_cell[0] is not None:
                    try:
                        w.after_cancel(id_cell[0])
                    except Exception:
                        pass
                    id_cell[0] = None
            _set_stream_controls(False)

        def _restart_stream():
            _stop_stream()
            _make_stream()
            running[0] = True
            _tick()
            _axis_animate()
            _set_stream_controls(True)

        def _clear_stream():
            if ts_handle[0] is not None:
                w.clear_line_stream(ts_handle[0])
                seed_now = time.monotonic()
                seed_dt = rate_var.get() / 1000.0
                ts_start_wall[0] = seed_now - seed_dt
                ts_last_batch_wall[0] = ts_start_wall[0]
                noise_y_last[0] = 0.0
                total_pts[0] = 0
                w.set_xlim(0.0, WINDOW_SEC)
                count_var.set("cleared")

        # Stop timers both when the widget itself is replaced and when the
        # section controls are torn down during navigation.
        w.bind("<Destroy>", lambda _: _stop_stream(), add="+")
        p.bind("<Destroy>", lambda e: _stop_stream() if e.widget is p else None, add="+")

        # ── Controls ──────────────────────────────────────────────────────────
        _head(p, "STATIC PLOTS")
        _btn(p, "sin / cos / sin(2x)", _load_static)
        _btn(p, "Lissajous curves",    _load_lissajous)

        _head(p, "STATIC LINE COLORS")
        _color_dropdown(p, "Line A", line_a_color_var, _apply_static_appearance)
        _color_dropdown(p, "Line B", line_b_color_var, _apply_static_appearance)
        _color_dropdown(p, "Line C", line_c_color_var, _apply_static_appearance)

        _head(p, "STATIC LINE WIDTHS")
        for label, var in [
            ("Line A", line_a_width_var),
            ("Line B", line_b_width_var),
            ("Line C", line_c_width_var),
        ]:
            row = tk.Frame(p, bg=PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, bg=PANEL, fg=FG, font=FM,
                     anchor="w").pack(fill="x")
            tk.Scale(
                row, variable=var, from_=1.0, to=10.0, resolution=0.5,
                orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                highlightthickness=0,
                command=lambda _v, cmd=_apply_static_appearance: cmd(),
            ).pack(fill="x")

        _head(p, "STREAM CONTROLS")
        _btn(p, "Start stream",  _start_stream)
        _btn(p, "Stop",          _stop_stream)
        _btn(p, "Restart",       _restart_stream)
        _btn(p, "Clear buffer",  _clear_stream)

        _head(p, "SHAPE")
        _radio_group(p, shape_var,
                     [("Sine",     "sine"),
                      ("Sawtooth", "sawtooth"),
                      ("Noise",    "noise")],
                     lambda *_: None)

        _head(p, "COLOR")
        _radio_group(p, stream_color_var,
                     [("Blue",   "Blue"),
                      ("Orange", "Orange"),
                      ("Green",  "Green"),
                      ("Pink",   "Pink")],
                     _apply_stream_appearance)

        _head(p, "STREAM LINE WIDTH")
        tk.Scale(p, variable=stream_width_var, from_=1.0, to=10.0, resolution=0.5,
                 orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                 highlightthickness=0,
                 command=lambda _v: _apply_stream_appearance()).pack(fill="x")

        _head(p, "SAMPLES / UPDATE")
        tk.Scale(p, variable=n_var, from_=1, to=500, resolution=1,
                 orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                 highlightthickness=0).pack(fill="x")

        _head(p, "UPDATE EVERY (ms)")
        tk.Scale(p, variable=rate_var, from_=16, to=200, resolution=4,
                 orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                 highlightthickness=0).pack(fill="x")

        _head(p, "MAX POINTS (restart to apply)")
        _s = tk.Scale(p, variable=max_var, from_=100, to=20_000, resolution=100,
                      orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                      highlightthickness=0)
        _s.pack(fill="x")
        max_scale[0] = _s

        _head(p, "Y GRID INTERVAL")
        tk.Scale(p, variable=y_tick_var, from_=0.1, to=5.0, resolution=0.1,
                 orient="horizontal", bg=PANEL, fg=FG, troughcolor="#333",
                 highlightthickness=0,
                 command=lambda v: w.set_y_tick_interval(float(v))).pack(fill="x")

        _head(p, "STATUS")
        _label(p, textvariable=count_var, fg=ACT)

        _load_static()


if __name__ == "__main__":
    app = DemoApp()
    app.mainloop()
