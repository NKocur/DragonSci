"""Jupyter / JupyterLab widget for dragonsci.

Install the notebook extra to use this module::

    pip install dragonsci[notebook]

Usage::

    from dragonsci import scatter3d   # auto-detects Jupyter kernel
    import numpy as np

    w = scatter3d(width=800, height=600)
    w.set_points(np.random.randn(50_000, 3).astype("f4"), colormap="viridis")
    w  # display in cell

    # Or construct directly:
    from dragonsci import JupyterScatter3D
    w = JupyterScatter3D(width=800, height=600)

Event surface differences vs Scatter3D
---------------------------------------
- ``<<PointPicked>>`` Tkinter virtual events do not exist.  Use the
  ``on_pick=`` callback parameter in ``enable_point_picking()`` instead.
- ``<<SelectionChanged>>`` / selection is not supported in v1.
- Camera linking (``link_cameras``) is not supported in v1.
- DataFrames are not yet supported — pass numpy (N, 3) float32 arrays.
"""
from __future__ import annotations

import io
from typing import Optional, TYPE_CHECKING

import numpy as np

from ._dragonsci import ScatterRenderer

if TYPE_CHECKING:
    pass

try:
    from jupyter_rfb import RemoteFrameBuffer
    _RFB_AVAILABLE = True
except ImportError:
    _RFB_AVAILABLE = False
    # Provide a stub so the class definition doesn't fail at import time.
    class RemoteFrameBuffer:  # type: ignore[no-redef]
        pass

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _require_notebook_deps() -> None:
    if not _RFB_AVAILABLE:
        raise ImportError(
            "jupyter_rfb is required for JupyterScatter3D.\n"
            "Install it with:  pip install dragonsci[notebook]"
        )
    if not _PIL_AVAILABLE:
        raise ImportError(
            "Pillow is required for JPEG encoding in JupyterScatter3D.\n"
            "Install it with:  pip install dragonsci[notebook]"
        )


class JupyterScatter3D(RemoteFrameBuffer):
    """GPU-accelerated 3-D scatter plot widget for JupyterLab.

    Parameters
    ----------
    width, height : int
        Initial canvas size in pixels.
    jpeg_quality : int
        JPEG encoding quality 1–95.  Higher = better fidelity, larger payload.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        jpeg_quality: int = 85,
    ) -> None:
        _require_notebook_deps()
        super().__init__()
        self._renderer: ScatterRenderer = ScatterRenderer.create_offscreen(width, height)
        self._width = width
        self._height = height
        self._jpeg_quality = jpeg_quality

        # Mouse state for drag translation
        self._drag_btn: Optional[int] = None
        self._drag_x: float = 0.0
        self._drag_y: float = 0.0
        self._drag_start_x: float = 0.0
        self._drag_start_y: float = 0.0

        # Picking callback (no Tk virtual events; selection not supported in v1)
        self._on_pick_cb = None

    # ── jupyter_rfb interface ─────────────────────────────────────────────────

    def get_frame_data(self) -> bytes:
        """Render one frame and return JPEG bytes for the browser."""
        rgba = bytes(self._renderer.render_offscreen())
        img = _PILImage.frombytes("RGBA", (self._width, self._height), rgba).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self._jpeg_quality)
        return buf.getvalue()

    def handle_event(self, event: dict, send_callback=None) -> None:
        """Translate jupyter_rfb pointer/wheel/resize events to camera commands."""
        etype = event.get("event_type", "")

        if etype == "resize":
            w = max(int(event.get("width", self._width)), 1)
            h = max(int(event.get("height", self._height)), 1)
            self._width = w
            self._height = h
            self._renderer.resize(w, h)
            self.request_draw()
            return

        if etype == "pointer_down":
            self._drag_btn = event.get("button", 1)
            x = float(event.get("x", 0))
            y = float(event.get("y", 0))
            self._drag_x = x
            self._drag_y = y
            self._drag_start_x = x
            self._drag_start_y = y
            return

        if etype == "pointer_up":
            if self._on_pick_cb is not None and self._drag_btn is not None:
                ddx = self._drag_x - self._drag_start_x
                ddy = self._drag_y - self._drag_start_y
                if ddx * ddx + ddy * ddy < 25.0:  # <5 px radius → treat as click
                    result = self._renderer.pick_point(self._drag_start_x, self._drag_start_y)
                    if result is not None:
                        self._on_pick_cb(result)
            self._drag_btn = None
            return

        if etype == "pointer_move":
            if self._drag_btn is None:
                return
            x = float(event.get("x", self._drag_x))
            y = float(event.get("y", self._drag_y))
            dx = x - self._drag_x
            dy = y - self._drag_y
            self._drag_x = x
            self._drag_y = y
            # Left-drag → orbit (button 1); middle/right-drag → pan (button 2)
            modifiers = event.get("modifiers", [])
            shift = "Shift" in modifiers
            effective = 2 if (self._drag_btn == 1 and shift) else self._drag_btn
            self._renderer.mouse_drag(dx, dy, effective)
            self.request_draw()
            return

        if etype == "wheel":
            dy = float(event.get("dy", 0))
            # jupyter_rfb sends pixel delta; normalise to ~scroll-click units
            self._renderer.scroll(-dy / 100.0)
            self.request_draw()
            return

        if etype == "double_click":
            self._renderer.reset_camera()
            self.request_draw()
            return

    # ── Scene API ─────────────────────────────────────────────────────────────

    def set_points(
        self,
        positions: np.ndarray,
        *,
        colors: Optional[np.ndarray] = None,
        scalars: Optional[np.ndarray] = None,
        colormap: str = "viridis",
        point_size: float = 4.0,
        point_sizes: Optional[np.ndarray] = None,
        clim: "tuple[float, float] | None" = None,
        nan_color: "tuple[float, float, float] | None" = None,
        log_scale: bool = False,
        opacity: float = 1.0,
    ) -> None:
        """Replace the scene with a single point cloud."""
        pos, clr, scl, sizes = _prepare_numpy_inputs(positions, colors, scalars, point_sizes)
        clim_l = list(clim) if clim is not None else None
        nan_l = list(nan_color) if nan_color is not None else None
        self._renderer.set_points(
            pos, clr, scl, colormap, float(point_size),
            sizes, clim_l, nan_l, bool(log_scale), float(opacity),
        )
        self.request_draw()

    def add_points(
        self,
        positions: np.ndarray,
        *,
        colors: Optional[np.ndarray] = None,
        scalars: Optional[np.ndarray] = None,
        colormap: str = "viridis",
        point_size: float = 4.0,
        point_sizes: Optional[np.ndarray] = None,
        clim: "tuple[float, float] | None" = None,
        nan_color: "tuple[float, float, float] | None" = None,
        log_scale: bool = False,
        opacity: float = 1.0,
    ) -> int:
        """Add a point cloud actor on top of the scene. Returns a handle."""
        pos, clr, scl, sizes = _prepare_numpy_inputs(positions, colors, scalars, point_sizes)
        clim_l = list(clim) if clim is not None else None
        nan_l = list(nan_color) if nan_color is not None else None
        handle = int(self._renderer.add_points(
            pos, clr, scl, colormap, float(point_size),
            sizes, clim_l, nan_l, bool(log_scale), float(opacity),
        ))
        self.request_draw()
        return handle

    def update_actor(
        self,
        handle: int,
        positions: np.ndarray,
        *,
        colors: Optional[np.ndarray] = None,
        scalars: Optional[np.ndarray] = None,
        colormap: str = "viridis",
        point_size: float = 4.0,
        point_sizes: Optional[np.ndarray] = None,
        clim: "tuple[float, float] | None" = None,
        nan_color: "tuple[float, float, float] | None" = None,
        log_scale: bool = False,
        opacity: float = 1.0,
    ) -> None:
        """Replace data in an existing actor in-place."""
        pos, clr, scl, sizes = _prepare_numpy_inputs(positions, colors, scalars, point_sizes)
        self._renderer.update_actor(
            handle, pos, clr, scl, colormap, float(point_size),
            sizes, list(clim) if clim else None,
            list(nan_color) if nan_color else None,
            bool(log_scale), float(opacity),
        )
        self.request_draw()

    def remove_actor(self, handle: int) -> None:
        self._renderer.remove_actor(handle)
        self.request_draw()

    def set_actor_visibility(self, handle: int, visible: bool) -> None:
        self._renderer.set_actor_visibility(handle, visible)
        self.request_draw()

    def clear(self) -> None:
        """Remove all actors, overlays, and user labels."""
        self._renderer.clear_actors()
        self._renderer.clear_user_labels()
        self.request_draw()

    # ── Camera ────────────────────────────────────────────────────────────────

    def reset_camera(self) -> None:
        self._renderer.reset_camera()
        self.request_draw()

    def fit(self, bounds=None) -> None:
        self._renderer.fit(bounds)
        self.request_draw()

    def view_xy(self) -> None:
        self._renderer.view_xy()
        self.request_draw()

    def view_xz(self) -> None:
        self._renderer.view_xz()
        self.request_draw()

    def view_yz(self) -> None:
        self._renderer.view_yz()
        self.request_draw()

    def view_isometric(self) -> None:
        self._renderer.view_isometric()
        self.request_draw()

    def get_camera(self) -> dict:
        return self._renderer.get_camera()

    def set_camera(self, state: dict) -> None:
        self._renderer.set_camera(state)
        self.request_draw()

    @property
    def parallel_projection(self) -> bool:
        return self._renderer.get_parallel_projection()

    @parallel_projection.setter
    def parallel_projection(self, value: bool) -> None:
        self._renderer.set_parallel_projection(value)
        self.request_draw()

    # ── Appearance ────────────────────────────────────────────────────────────

    def set_background(self, color) -> None:
        if isinstance(color, str):
            color = color.lstrip("#")
            r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
            color = (r / 255.0, g / 255.0, b / 255.0)
        self._renderer.set_background_color(*color)
        self.request_draw()

    def show_grid(self, visible: bool = True) -> None:
        self._renderer.show_grid(visible)
        self.request_draw()

    def set_axes(self, x: str = "X", y: str = "Y", z: str = "Z") -> None:
        self._renderer.set_axis_labels(x, y, z)
        self.request_draw()

    def scalar_bar(
        self,
        visible: bool = True,
        *,
        vmin: float = 0.0,
        vmax: float = 1.0,
        log_scale: bool = False,
        colormap: str = "viridis",
        title: str = "",
    ) -> None:
        self._renderer.show_scalar_bar(visible, vmin, vmax, log_scale, colormap, title)
        self.request_draw()

    # ── Export ────────────────────────────────────────────────────────────────

    def screenshot(self) -> np.ndarray:
        """Return current frame as (H, W, 4) uint8 RGBA array."""
        rgba = bytes(self._renderer.render_offscreen())
        return np.frombuffer(rgba, dtype=np.uint8).reshape(self._height, self._width, 4)

    def save_png(self, path: str) -> None:
        img = _PILImage.frombytes(
            "RGBA", (self._width, self._height),
            bytes(self._renderer.render_offscreen()),
        )
        img.save(path)

    # ── Picking callbacks (no Tk virtual events) ──────────────────────────────

    def enable_point_picking(self, on_pick=None) -> None:
        self._on_pick_cb = on_pick
        self._renderer.set_pick_storage(True)

    def disable_picking(self) -> None:
        self._on_pick_cb = None

    # ── User-defined world-space text labels ──────────────────────────────────

    def add_label(
        self,
        position: "tuple[float, float, float] | np.ndarray",
        text: str,
        *,
        color: "tuple[float, float, float] | str" = (1.0, 1.0, 1.0),
        size: float = 14.0,
        anchor: str = "center",
    ) -> int:
        """Pin a text label at a 3-D world-space position. Returns a handle."""
        from .widget import _parse_label_position, _parse_label_color, _LABEL_ANCHOR_MAP
        pos3 = _parse_label_position(position)
        rgba = _parse_label_color(color)
        anch = _LABEL_ANCHOR_MAP.get(anchor.lower(), 0)
        handle = int(self._renderer.add_user_label(
            pos3[0], pos3[1], pos3[2], text, rgba, float(size), anch,
        ))
        self.request_draw()
        return handle

    def update_label(
        self,
        handle: int,
        position: "tuple[float, float, float] | np.ndarray | None" = None,
        text: "str | None" = None,
        *,
        color: "tuple[float, float, float] | str | None" = None,
        size: "float | None" = None,
        anchor: "str | None" = None,
    ) -> None:
        from .widget import _parse_label_position, _parse_label_color, _LABEL_ANCHOR_MAP
        pos3 = _parse_label_position(position) if position is not None else None
        rgba = _parse_label_color(color) if color is not None else None
        anch = _LABEL_ANCHOR_MAP.get(anchor.lower(), 0) if anchor is not None else None
        self._renderer.update_user_label(handle, pos=pos3, text=text,
                                         color=rgba, size=size, anchor=anch)
        self.request_draw()

    def remove_label(self, handle: int) -> None:
        self._renderer.remove_user_label(handle)
        self.request_draw()

    def set_label_visibility(self, handle: int, visible: bool) -> None:
        self._renderer.set_user_label_visible(handle, visible)
        self.request_draw()

    def clear_labels(self) -> None:
        self._renderer.clear_user_labels()
        self.request_draw()

    @staticmethod
    def colormap_names() -> list:
        return ScatterRenderer.colormap_names()


class JupyterScatter2D(JupyterScatter3D):
    """2-D scatter plot widget for JupyterLab (XY plane, parallel projection).

    Accepts both (N, 2) and (N, 3) position arrays.  When (N, 2) is given the
    Z column is set to 0.  Nonzero Z values in (N, 3) arrays are also zeroed
    out to keep points on the XY plane.

    Camera methods that would break the 2-D view (``view_isometric``,
    ``view_xy``, ``view_yz``, ``set_camera``, ``reset_camera``) are overridden
    to re-lock the front view after execution.  ``parallel_projection`` is
    always True and cannot be changed.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._relock_2d()

    def _relock_2d(self) -> None:
        self._renderer.view_xz()
        self._renderer.set_parallel_projection(True)

    @staticmethod
    def _to_3d(positions: np.ndarray) -> np.ndarray:
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        if pos.ndim == 2 and pos.shape[1] == 2:
            pos = np.column_stack([pos, np.zeros(len(pos), dtype=np.float32)])
        elif pos.ndim == 2 and pos.shape[1] == 3:
            pos = pos.copy()
            pos[:, 2] = 0.0
        return pos

    # ── Data overrides (flatten Z, re-lock camera for set_points) ─────────────

    def set_points(self, positions: np.ndarray, **kwargs) -> None:
        super().set_points(self._to_3d(positions), **kwargs)
        self._relock_2d()

    def add_points(self, positions: np.ndarray, **kwargs) -> int:
        return super().add_points(self._to_3d(positions), **kwargs)

    def update_actor(self, handle: int, positions: np.ndarray, **kwargs) -> None:
        super().update_actor(handle, self._to_3d(positions), **kwargs)

    # ── Camera overrides — always restore 2-D front view ──────────────────────

    def reset_camera(self) -> None:
        super().reset_camera()
        self._relock_2d()
        self.request_draw()

    def set_camera(self, state: dict) -> None:
        super().set_camera(state)
        self._relock_2d()
        self.request_draw()

    def view_isometric(self) -> None:
        self._relock_2d()
        self.request_draw()

    def view_xy(self) -> None:
        self._relock_2d()
        self.request_draw()

    def view_yz(self) -> None:
        self._relock_2d()
        self.request_draw()

    def view_xz(self) -> None:
        self._relock_2d()
        self.request_draw()

    @property
    def parallel_projection(self) -> bool:
        return True

    @parallel_projection.setter
    def parallel_projection(self, value: bool) -> None:
        pass  # always parallel in 2D mode


# ── Module-level helpers ──────────────────────────────────────────────────────

def _prepare_numpy_inputs(
    positions: np.ndarray,
    colors: Optional[np.ndarray],
    scalars: Optional[np.ndarray],
    point_sizes: Optional[np.ndarray],
):
    pos = np.ascontiguousarray(positions, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be shape (N, 3), got {pos.shape}")
    n = pos.shape[0]

    clr = None
    if colors is not None:
        clr = np.ascontiguousarray(colors, dtype=np.float32)
        if clr.ndim != 2 or clr.shape[0] != n or clr.shape[1] != 3:
            raise ValueError(f"colors must be shape (N, 3), got {clr.shape}")

    scl = None
    if scalars is not None:
        scl = np.ascontiguousarray(scalars, dtype=np.float32)
        if scl.ndim != 1 or scl.shape[0] != n:
            raise ValueError(f"scalars length {scl.shape[0]} must match N={n}")

    sizes = None
    if point_sizes is not None:
        sizes = np.ascontiguousarray(point_sizes, dtype=np.float32)
        if sizes.ndim != 1 or sizes.shape[0] != n:
            raise ValueError(f"point_sizes length {sizes.shape[0]} must match N={n}")

    return pos, clr, scl, sizes
