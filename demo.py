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

import tkinter as tk
import tkinter.filedialog as fd
import numpy as np

from dragonsci import Scatter3D, Scatter2D, link_cameras, unlink_cameras

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
        ("── planned ──",    "",                 True),   # separator
        ("Lasso Selection",   "_sec_lasso",       False),
        ("Streaming",         "_sec_streaming",   False),
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

    def _switch(self, method: str) -> None:
        # If leaving 2D mode, restore the Scatter3D widget.
        if isinstance(self.scatter, Scatter2D) and method != "_sec_2dmode":
            self._swap_scatter(Scatter3D)

        # Reset scatter state that may have been changed by previous section
        self.scatter.disable_picking()
        self.scatter.bind("<<PointPicked>>",      lambda e: None)
        self.scatter.bind("<<SelectionChanged>>", lambda e: None)

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


if __name__ == "__main__":
    app = DemoApp()
    app.mainloop()
