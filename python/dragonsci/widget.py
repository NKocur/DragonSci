"""Tkinter widget that wraps the Rust/wgpu ScatterRenderer."""

from __future__ import annotations

import ctypes
import math as _math
import sys
import tkinter as tk
from typing import Optional

import numpy as np

from ._dragonsci import ScatterRenderer


def _series_to_numpy(values) -> np.ndarray:
    """Best-effort conversion for pandas / polars series-like objects."""
    to_numpy = getattr(values, "to_numpy", None)
    if callable(to_numpy):
        try:
            arr = to_numpy(copy=False)
        except TypeError:
            arr = to_numpy()
    else:
        arr = np.asarray(values)
    return np.asarray(arr)


def _extract_frame_column(data, name: str) -> np.ndarray:
    try:
        values = data[name]
    except Exception as exc:
        raise ValueError(f"Unknown column {name!r}") from exc
    return _series_to_numpy(values)


def _extract_numeric_frame_column(data, name: str) -> np.ndarray:
    values = _extract_frame_column(data, name)
    try:
        return np.ascontiguousarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {name!r} must be numeric") from exc


def _is_supported_dataframe(data) -> bool:
    if isinstance(data, np.ndarray):
        return False

    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None and isinstance(data, pd.DataFrame):
        return True

    try:
        import polars as pl
    except ImportError:
        pl = None
    if pl is not None and isinstance(data, pl.DataFrame):
        return True

    return False


_CATEGORY_MISSING = object()
_CATEGORICAL_THRESHOLD = 20
_CATEGORICAL_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.122, 0.467, 0.706),
    (1.000, 0.498, 0.055),
    (0.173, 0.627, 0.173),
    (0.839, 0.153, 0.157),
    (0.580, 0.404, 0.741),
    (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761),
    (0.498, 0.498, 0.498),
    (0.737, 0.741, 0.133),
    (0.090, 0.745, 0.812),
)
_FLATTEN_PLANES: dict[str, tuple[float, float]] = {
    # Maps plane name → (yaw_rad, pitch_rad) for the camera position formula
    # pos = target + (cos(p)*sin(y), sin(p), cos(p)*cos(y)) * dist
    "xy":  (0.0,                         0.0),                      # from +Z: X right, Y up
    "xy-": (_math.pi,                    0.0),                      # from -Z
    "xz":  (0.0,                         _math.pi / 2 - 0.001),    # from +Y: X right, Z "up"
    "xz-": (0.0,                        -(_math.pi / 2 - 0.001)),  # from -Y
    "yz":  (_math.pi / 2,                0.0),                      # from +X: Y up, Z right
    "yz-": (-_math.pi / 2,               0.0),                      # from -X
}

_LEGEND_POSITION_IDX: dict[str, int] = {
    "top-right": 0,
    "top-left": 1,
    "bottom-right": 2,
    "bottom-left": 3,
}


def _is_missing_category_value(value) -> bool:
    if value is None:
        return True
    try:
        if bool(np.isnan(value)):
            return True
    except (TypeError, ValueError):
        pass
    try:
        if bool(np.isnat(value)):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _normalize_category_value(value) -> tuple[object, str]:
    if isinstance(value, np.generic):
        value = value.item()
    if _is_missing_category_value(value):
        return _CATEGORY_MISSING, "Missing"
    try:
        hash(value)
        key = value
    except TypeError:
        key = repr(value)
    return key, str(value)


def _categorical_palette_color(index: int) -> tuple[float, float, float]:
    base = _CATEGORICAL_PALETTE[index % len(_CATEGORICAL_PALETTE)]
    cycle = index // len(_CATEGORICAL_PALETTE)
    if cycle == 0:
        return base
    if cycle % 2 == 1:
        return tuple(min(1.0, c + (1.0 - c) * 0.22) for c in base)
    return tuple(max(0.0, c * 0.78) for c in base)


def _clamp_sizes(arr: "np.ndarray") -> "np.ndarray":
    """Replace non-finite size values with 0 and clamp negatives to 0.

    Prevents negative or infinite pixel sizes from reaching the shader.
    """
    return np.where(np.isfinite(arr), np.maximum(arr, 0.0), 0.0).astype(np.float32)


def _try_encode_categorical(
    values: np.ndarray,
) -> "tuple[np.ndarray, list[tuple[str, tuple[float, float, float]]]] | None":
    """Single-pass categorical detection + encoding.

    Returns ``(rgb_array, legend_items)`` when ``values`` looks categorical,
    ``None`` otherwise.  Stops early when the cardinality exceeds
    ``_CATEGORICAL_THRESHOLD`` (so non-categorical integer columns pay at most
    that many iterations instead of a full double-scan).
    """
    arr = np.asarray(values).reshape(-1)
    dtype = arr.dtype

    # Non-numeric dtypes are always categorical — skip the threshold check.
    always_categorical = np.issubdtype(dtype, np.bool_) or dtype.kind in ("U", "S", "O")
    if not always_categorical and not np.issubdtype(dtype, np.integer):
        return None

    colors = np.empty((arr.shape[0], 3), dtype=np.float32)
    key_to_index: dict[object, int] = {}
    legend_items: list[tuple[str, tuple[float, float, float]]] = []

    for i, raw in enumerate(arr):
        key, label = _normalize_category_value(raw)
        idx = key_to_index.get(key)
        if idx is None:
            if not always_categorical and len(legend_items) >= _CATEGORICAL_THRESHOLD:
                return None   # too many distinct values — not categorical
            idx = len(legend_items)
            key_to_index[key] = idx
            legend_items.append((label, _categorical_palette_color(idx)))
        colors[i] = legend_items[idx][1]

    return colors, legend_items


def _rgb01_to_hex(color: tuple[float, float, float]) -> str:
    r = max(0, min(255, round(color[0] * 255)))
    g = max(0, min(255, round(color[1] * 255)))
    b = max(0, min(255, round(color[2] * 255)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _copy_legend_items(
    items: "list[tuple[str, tuple[float, float, float]]] | None",
) -> "list[tuple[str, tuple[float, float, float]]] | None":
    if items is None:
        return None
    return [(str(label), tuple(color)) for label, color in items]


def _coerce_dataframe(data, x, y, z=None, color=None, size=None, hover=None) -> dict:
    if not _is_supported_dataframe(data):
        raise TypeError("Expected a pandas or polars DataFrame")
    if x is None or y is None:
        raise ValueError("x and y are required when positions is a DataFrame")

    x_values = _extract_numeric_frame_column(data, x)
    y_values = _extract_numeric_frame_column(data, y)
    if x_values.shape[0] != y_values.shape[0]:
        raise ValueError("x and y columns must have the same length")

    if z is None:
        z_values = np.zeros(x_values.shape[0], dtype=np.float32)
    else:
        z_values = _extract_numeric_frame_column(data, z)

    if z_values.shape[0] != x_values.shape[0]:
        raise ValueError("z column length must match x and y")

    positions = np.column_stack((x_values, y_values, z_values)).astype(np.float32, copy=False)
    color_values = _extract_frame_column(data, color) if color is not None else None
    size_values = _extract_numeric_frame_column(data, size) if size is not None else None

    hover_cols: list[str]
    if hover is None:
        hover_cols = []
    elif isinstance(hover, str):
        hover_cols = [hover]
    else:
        hover_cols = list(hover)

    hover_data = {name: _extract_frame_column(data, name) for name in hover_cols}
    row_positions = np.arange(len(data), dtype=np.int64)

    row_labels = None
    if hasattr(data, "index"):
        try:
            row_labels = np.asarray(data.index.to_numpy(copy=False))
        except TypeError:
            row_labels = np.asarray(data.index.to_numpy())

    columns = {"x": str(x), "y": str(y), "z": str(z) if z is not None else "z"}
    if color is not None:
        columns["color"] = str(color)
    if size is not None:
        columns["size"] = str(size)
    if hover_cols:
        columns["hover"] = list(hover_cols)

    return {
        "positions": positions,
        "color_values": color_values,
        "size_values": size_values,
        "hover_data": hover_data,
        "row_positions": row_positions,
        "row_labels": row_labels,
        "columns": columns,
        "legend_items": None,
        "legend_title": None,
    }


def _gif_lzw_compress(indices: "list[int]", min_code_size: int = 8) -> bytes:
    """GIF LZW compression.  Returns bytes in GIF sub-block format."""
    clear = 1 << min_code_size
    eoi = clear + 1

    table: "dict[tuple, int]" = {(i,): i for i in range(clear)}
    next_code = eoi + 1
    code_size = min_code_size + 1

    pairs: "list[tuple[int, int]]" = [(clear, code_size)]

    if indices:
        prefix: "tuple[int, ...]" = (indices[0],)
        for sym in indices[1:]:
            ext = prefix + (sym,)
            if ext in table:
                prefix = ext
            else:
                pairs.append((table[prefix], code_size))
                if next_code < 4096:
                    table[ext] = next_code
                    next_code += 1
                    if next_code > (1 << code_size) and code_size < 12:
                        code_size += 1
                else:
                    pairs.append((clear, code_size))
                    table = {(i,): i for i in range(clear)}
                    next_code = eoi + 1
                    code_size = min_code_size + 1
                prefix = (sym,)
        pairs.append((table[prefix], code_size))

    pairs.append((eoi, code_size))

    # Pack codes LSB-first into bytes
    raw = bytearray()
    acc = acc_bits = 0
    for code, nbits in pairs:
        acc |= code << acc_bits
        acc_bits += nbits
        while acc_bits >= 8:
            raw.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        raw.append(acc & 0xFF)

    # Wrap in GIF sub-blocks (max 255 bytes each)
    result = bytearray([min_code_size])
    i = 0
    while i < len(raw):
        block = raw[i : i + 255]
        result.append(len(block))
        result.extend(block)
        i += 255
    result.append(0)
    return bytes(result)


def _write_gif_stdlib(
    path: str,
    frames: "list[np.ndarray]",
    fps: int,
    loop: int,
) -> None:
    """Write an animated GIF using only stdlib + numpy (3-3-2 colour quantisation)."""
    import struct

    if not frames:
        return
    h, w = frames[0].shape[:2]
    delay = max(2, round(100 / fps))  # centiseconds

    # Build 3-3-2 palette: index = (r3 << 5) | (g3 << 2) | b2
    pal = bytearray(256 * 3)
    for idx in range(256):
        r3 = (idx >> 5) & 7
        g3 = (idx >> 2) & 7
        b2 = idx & 3
        pal[idx * 3]     = (r3 * 255) // 7 if r3 else 0
        pal[idx * 3 + 1] = (g3 * 255) // 7 if g3 else 0
        pal[idx * 3 + 2] = (b2 * 255) // 3 if b2 else 0

    buf = bytearray()
    buf += b"GIF89a"
    buf += struct.pack("<HH", w, h)
    buf += bytes([0xF7, 0, 0])  # global CT=1, 256 colours, bg=0, aspect=0
    buf += pal

    # Netscape loop extension
    buf += b"\x21\xFF\x0BNETSCAPE2.0\x03\x01"
    buf += struct.pack("<HB", loop & 0xFFFF, 0)

    for rgba in frames:
        r = rgba[:, :, 0].astype(np.uint16)
        g = rgba[:, :, 1].astype(np.uint16)
        b = rgba[:, :, 2].astype(np.uint16)
        qi = (((r >> 5) << 5) | ((g >> 5) << 2) | (b >> 6)).astype(np.uint8)
        indices = qi.flatten().tolist()

        # Graphic Control Extension
        buf += b"\x21\xF9\x04\x00"
        buf += struct.pack("<HB", delay, 0)

        # Image Descriptor
        buf += b"\x2C"
        buf += struct.pack("<HHHHB", 0, 0, w, h, 0)

        buf += _gif_lzw_compress(indices, 8)

    buf += b"\x3B"
    with open(path, "wb") as f:
        f.write(buf)


def _write_png(path: str, rgba: "np.ndarray") -> None:
    """Minimal PNG writer using only stdlib — no Pillow required."""
    import struct, zlib
    h, w = rgba.shape[:2]

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    # IHDR: width, height, bit_depth=8, color_type=6 (RGBA), compression=0, filter=0, interlace=0
    ihdr = chunk(b"IHDR", struct.pack(">II", w, h) + bytes([8, 6, 0, 0, 0]))
    # IDAT: filter byte 0 (None) prepended to every scanline
    raw_rows = b"".join(b"\x00" + bytes(row.tobytes()) for row in rgba)
    idat = chunk(b"IDAT", zlib.compress(raw_rows, 6))
    iend = chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend)


def _get_display_id() -> int:
    if sys.platform != "linux":
        return 0
    try:
        xlib = ctypes.cdll.LoadLibrary("libX11.so.6")
        xlib.XOpenDisplay.restype = ctypes.c_void_p
        ptr = xlib.XOpenDisplay(None)
        return int(ptr) if ptr else 0
    except OSError:
        return 0


def _platform_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


_DISPLAY_ID: int = _get_display_id()
_HOVER_DEBOUNCE_MS: int = 30


class Scatter3D(tk.Frame):
    """A Tkinter widget that renders a 3-D scatter plot using wgpu (Rust).

    Usage
    -----
    ::

        import tkinter as tk
        import numpy as np
        from dragonsci import Scatter3D

        root = tk.Tk()
        w = Scatter3D(root, width=800, height=600)
        w.pack(fill="both", expand=True)

        pts = np.random.rand(250_000, 3).astype(np.float32)
        w.set_points(pts)

        root.mainloop()
    """

    def __init__(
        self,
        master: tk.Misc,
        width: int = 800,
        height: int = 600,
        fps: int = 60,
        vsync: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(master, width=width, height=height, **kwargs)

        if fps < 1:
            raise ValueError(f"fps must be >= 1, got {fps}")
        self._fps = fps
        self._vsync = vsync
        self._renderer: Optional[ScatterRenderer] = None

        # Pending data: set before the renderer is ready
        self._pending: Optional[dict] = None
        self._pending_point_meta: Optional[dict] = None
        self._pending_actors: list = []   # queued add_points() calls: (kwargs, vhandle)
        self._pending_actor_meta: dict[int, dict] = {}
        self._pending_actor_visibility: "dict[int, bool]" = {}  # vhandle → visibility (pre-map)
        self._pending_streams: list = []  # queued add_stream() calls: (kwargs, vhandle)
        self._stream_handles: "set[int]" = set()  # real handles of stream actors
        self._next_phandle: int = 0
        self._phandle_map: "dict[int, int]" = {}   # virtual → real actor handle
        self._pending_ticks: Optional[tuple] = None
        # Python-side shadow so parallel_projection is readable before renderer init
        self._parallel_projection: bool = False

        # Dirty-frame model: only call render() when something changed
        self._dirty: bool = False

        # Drag state
        self._drag_btn: Optional[int] = None
        self._drag_x: int = 0
        self._drag_y: int = 0

        # Picking state
        self._pick_mode: str = "none"   # "none" | "point" | "rect" | "both"
        self._press_x: int = 0          # ButtonPress coords, for click vs drag
        self._press_y: int = 0
        self._sel_x0: int = 0           # rectangle start (screen coords)
        self._sel_y0: int = 0
        self._rect_active: bool = False  # Shift+drag rectangle in progress
        self._pick_threshold: int = 5   # px — less than this = click, not drag

        # Public result attributes — read after virtual events
        self.picked_point: "list[float] | None" = None
        self.picked_index: "int | None" = None
        self.picked_actor: "int | None" = None
        self.selected: "list[dict] | None" = None       # raw [{"actor","index"},...]
        self.selected_indices: "list[int] | None" = None        # plotted row positions (iloc)
        self.selected_index_values: "list | None" = None        # pandas index labels when available

        # Lasso state (lasso_pts are stored in Rust; only the active flag lives here)
        self._lasso_enabled: bool = False
        self._lasso_active: bool = False

        # Linked-camera state
        self._camera_links: "set[Scatter3D]" = set()
        self._propagating: bool = False  # re-entrancy guard

        # Animation recording state
        self._gif_frames: "list | None" = None
        self._gif_path: "str | None" = None
        self._gif_fps: int = 20
        self._gif_loop: int = 0
        self._gif_tmp_dir: "str | None" = None  # temp dir for on-disk frame cache

        # Hover tooltip state
        self._hover_tooltip: bool = True
        self._tooltip_win: "tk.Toplevel | None" = None
        self._hover_after_id: "str | None" = None
        self._hover_last_x: int = 0
        self._hover_last_y: int = 0
        self._legend_visible: bool = True
        self._legend_position: str = "top-right"
        self._pending_legend: "tuple | None" = None
        self._major_grid_planes: bool = False
        self._minor_grid_planes: bool = False

        # Rendering modes
        self._point_style: str = "circle"
        self._lod_enabled: bool = True
        self._lod_threshold: int = 200_000  # activate LOD above this many total points
        self._lod_factor: int = 8           # draw 1-in-8 points during interaction
        self._total_n: int = 0              # running total of uploaded points
        self._actor_n: "dict[int, int]" = {}  # real handle → point count for LOD accounting

        # Visual appearance
        self._grid_visible: bool = True
        self._bg_color: tuple = (0.05, 0.05, 0.07)
        self._axis_labels: tuple = ("X", "Y", "Z")
        self._axis_visible: tuple = (True, True, True)

        # Metadata for DataFrame-backed plots
        self._scene_hover: dict[str, np.ndarray] = {}
        self._scene_columns: dict[str, object] = {}
        self._scene_row_positions: "np.ndarray | None" = None
        self._scene_row_labels: "np.ndarray | None" = None
        self._scene_actor_handle: "int | None" = None
        self._scene_legend: "list[tuple[str, tuple[float, float, float]]] | None" = None
        self._scene_legend_title: "str | None" = None
        self._actor_hover: dict[int, dict[str, np.ndarray]] = {}
        self._actor_columns: dict[int, dict[str, object]] = {}
        self._actor_row_positions: dict[int, np.ndarray] = {}
        self._actor_row_labels: dict[int, np.ndarray] = {}
        self._actor_legend: dict[int, list[tuple[str, tuple[float, float, float]]]] = {}
        self._actor_legend_title: dict[int, str] = {}
        self._legend_order: "list[tuple[str, int | None]]" = []

        # Pre-map overlay queue
        self._orientation_axes_visible: bool = False
        self._pending_scalar_bar: "dict | None" = None
        self._pending_overlays: "list[tuple]" = []  # (method, segments, color, vhandle)
        self._pending_overlay_visibility: "dict[int, bool]" = {}  # vhandle → visibility (pre-map)
        self._next_vhandle: int = 0
        self._vhandle_map: "dict[int, int]" = {}   # virtual → real handle

        self._after_id: Optional[str] = None
        self._resize_after_id: Optional[str] = None

        self.pack_propagate(False)
        self.grid_propagate(False)

        self.bind("<Map>", self._on_map, add="+")
        self.bind("<Configure>", self._on_configure, add="+")

        self.bind("<ButtonPress-1>", lambda e: self._drag_start(e, 1))
        self.bind("<ButtonPress-2>", lambda e: self._drag_start(e, 2))
        self.bind("<B1-Motion>", lambda e: self._drag_move(e, 1))
        self.bind("<B2-Motion>", lambda e: self._drag_move(e, 2))
        self.bind("<ButtonRelease-1>", self._drag_end)
        self.bind("<ButtonRelease-2>", self._drag_end)
        self.bind("<MouseWheel>", self._on_scroll)
        self.bind("<Button-4>", self._on_scroll_up_x11)
        self.bind("<Button-5>", self._on_scroll_down_x11)
        self.bind("<Double-Button-1>", lambda _e: self.reset_camera())
        self.bind("<Motion>", self._on_hover_motion, add="+")
        self.bind("<Leave>", self._on_hover_leave, add="+")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _on_map(self, _event: tk.Event) -> None:
        if self._renderer is not None:
            return
        self.update_idletasks()
        self._init_renderer()

    def _init_renderer(self) -> None:
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        try:
            self._renderer = ScatterRenderer(
                self.winfo_id(),
                _DISPLAY_ID,
                w,
                h,
                _platform_name(),
                self._vsync,
            )
        except Exception as exc:
            import warnings
            warnings.warn(f"dragonsci: renderer init failed: {exc}", stacklevel=2)
            return

        # Apply any pre-renderer state that was set before the window was mapped.
        if self._parallel_projection:
            self._renderer.set_parallel_projection(True)
        if self._point_style != "circle":
            self._renderer.set_point_style(self._STYLE_MAP[self._point_style])
        if not self._grid_visible:
            self._renderer.show_grid(False)
        if self._major_grid_planes or self._minor_grid_planes:
            self._renderer.show_grid_planes(self._major_grid_planes, self._minor_grid_planes)
        if self._bg_color != (0.05, 0.05, 0.07):
            self._renderer.set_background_color(*self._bg_color)
        if self._axis_labels != ("X", "Y", "Z"):
            self._renderer.set_axis_labels(*self._axis_labels)
        if self._axis_visible != (True, True, True):
            self._renderer.set_axis_visible(*self._axis_visible)
        if self._orientation_axes_visible:
            self._renderer.show_orientation_axes(True)
        if self._pending_scalar_bar is not None:
            sb = self._pending_scalar_bar
            self._pending_scalar_bar = None
            self._renderer.show_scalar_bar(
                sb["visible"], sb["vmin"], sb["vmax"],
                sb["log_scale"], sb["colormap"], sb["title"],
            )
        if self._pending_legend is not None:
            visible, title, items_list, position_idx = self._pending_legend
            self._pending_legend = None
            self._renderer.set_legend(visible, title, items_list, position_idx)
        for _method, segments, color, vhandle in self._pending_overlays:
            real = int(self._renderer.add_lines(segments, color))
            self._vhandle_map[vhandle] = real
        self._pending_overlays.clear()
        for vhandle, visible in self._pending_overlay_visibility.items():
            real = self._vhandle_map.get(vhandle, vhandle)
            self._renderer.set_overlay_visibility(real, visible)
        self._pending_overlay_visibility.clear()
        if self._pending_ticks is not None:
            x, y, z = self._pending_ticks
            self._pending_ticks = None
            self._renderer.set_ticks(x=x, y=y, z=z)
        did_something = False
        if self._pending is not None:
            pending = self._pending
            self._pending = None
            pending_meta = self._pending_point_meta
            self._pending_point_meta = None
            self.set_points(**pending)   # calls _mark_dirty internally; sets _scene_actor_handle
            if pending_meta is not None:
                # Preserve the handle assigned above so _translate_hits can map
                # actor IDs to DataFrame row positions and index labels.
                self._set_scene_metadata(pending_meta, actor_handle=self._scene_actor_handle)
            did_something = True
        if self._pending_actors:
            self._clear_scene_metadata()
            for kwargs, vhandle in self._pending_actors:
                real = int(self._renderer.add_points(**kwargs))
                if real != 0xFFFFFFFF:   # u32::MAX means empty dataset
                    self._phandle_map[vhandle] = real
                    self._store_actor_metadata(real, self._pending_actor_meta.get(vhandle))
                    n = kwargs["positions"].shape[0]
                    self._actor_n[real] = n
                    self._total_n += n
            self._pending_actors.clear()
            self._pending_actor_meta.clear()
            for vhandle, visible in self._pending_actor_visibility.items():
                real = self._phandle_map.get(vhandle)
                if real is not None:
                    self._renderer.set_actor_visibility(real, visible)
            self._pending_actor_visibility.clear()
            self._mark_dirty()
            did_something = True
        if self._pending_streams:
            for kwargs, vhandle in self._pending_streams:
                real = int(self._renderer.create_stream(**kwargs))
                self._phandle_map[vhandle] = real
                self._stream_handles.add(real)
            self._pending_streams.clear()
            self._mark_dirty()
            did_something = True
        if not did_something:
            self._schedule_render()

    def _on_configure(self, event: tk.Event) -> None:
        if self._renderer is None:
            return
        # Debounce: resize is expensive — only execute 50 ms after the last event
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(
            50, lambda: self._do_resize(event.width, event.height)
        )

    def _do_resize(self, w: int, h: int) -> None:
        self._resize_after_id = None
        if self._renderer is not None:
            self._renderer.resize(max(w, 1), max(h, 1))
            self._refresh_legend()
            self._mark_dirty()

    def destroy(self) -> None:
        for id_ in (self._after_id, self._resize_after_id, self._hover_after_id):
            if id_ is not None:
                self.after_cancel(id_)
        if self._tooltip_win is not None:
            try:
                self._tooltip_win.destroy()
            except Exception:
                pass
            self._tooltip_win = None
        self._clear_all_point_metadata()
        self._renderer = None
        super().destroy()

    # ── Render loop ───────────────────────────────────────────────────────────

    def _schedule_render(self) -> None:
        interval = max(1, 1000 // self._fps)
        self._after_id = self.after(interval, self._render_tick)

    def _mark_dirty(self) -> None:
        """Mark a redraw needed; restarts the timer if it was stopped."""
        self._dirty = True
        if self._after_id is None and self._renderer is not None:
            self._schedule_render()

    def _render_tick(self) -> None:
        self._after_id = None  # cleared first so _mark_dirty can re-arm
        if self._renderer is not None and self._dirty:
            try:
                self._renderer.render()
                self._dirty = False  # only clear on success; error keeps dirty for retry
            except Exception:
                pass
        # Re-arm only while there is work to do; stops firing when idle
        if self._dirty and self._after_id is None:
            self._schedule_render()

    # ── Data API ──────────────────────────────────────────────────────────────

    def _remember_legend_source(self, source: "tuple[str, int | None]") -> None:
        self._legend_order = [entry for entry in self._legend_order if entry != source]
        self._legend_order.append(source)

    def _forget_legend_source(self, source: "tuple[str, int | None]") -> None:
        self._legend_order = [entry for entry in self._legend_order if entry != source]

    def _current_legend_payload(
        self,
    ) -> "tuple[str | None, list[tuple[str, tuple[float, float, float]]]] | None":
        for kind, handle in reversed(self._legend_order):
            if kind == "scene":
                if self._scene_legend:
                    return self._scene_legend_title, self._scene_legend
                continue
            if handle is None:
                continue
            items = self._actor_legend.get(handle)
            if items:
                return self._actor_legend_title.get(handle), items
        return None

    def _hide_legend(self) -> None:
        if self._renderer is not None:
            self._renderer.set_legend(False, "", [], 0)
            self._mark_dirty()
        else:
            self._pending_legend = (False, "", [], 0)

    def _refresh_legend(self) -> None:
        if not self._legend_visible:
            self._hide_legend()
            return

        payload = self._current_legend_payload()
        if payload is None:
            self._hide_legend()
            return

        title, items = payload
        position_idx = _LEGEND_POSITION_IDX.get(self._legend_position, 0)
        items_list = [(label, list(color)) for label, color in items]

        if self._renderer is not None:
            self._renderer.set_legend(True, title or "", items_list, position_idx)
            self._mark_dirty()
        else:
            self._pending_legend = (True, title or "", items_list, position_idx)

    def _clear_scene_metadata(self, refresh: bool = True) -> None:
        self._scene_hover.clear()
        self._scene_columns.clear()
        self._scene_row_positions = None
        self._scene_row_labels = None
        self._scene_actor_handle = None
        self._scene_legend = None
        self._scene_legend_title = None
        self._forget_legend_source(("scene", None))
        if refresh:
            self._refresh_legend()

    def _set_scene_metadata(self, meta: Optional[dict], actor_handle: "int | None" = None) -> None:
        self._clear_scene_metadata(refresh=False)
        self._scene_actor_handle = actor_handle   # preserve even when meta is None
        if meta is None:
            self._refresh_legend()
            return
        self._scene_hover = dict(meta["hover_data"])
        self._scene_columns = dict(meta["columns"])
        self._scene_row_positions = meta["row_positions"]
        self._scene_row_labels = meta["row_labels"]
        self._scene_legend = _copy_legend_items(meta.get("legend_items"))
        legend_title = meta.get("legend_title")
        self._scene_legend_title = str(legend_title) if legend_title is not None else None
        if self._scene_legend:
            self._remember_legend_source(("scene", None))
        self._refresh_legend()

    def _drop_actor_metadata(self, handle: int, refresh: bool = True) -> None:
        self._actor_hover.pop(handle, None)
        self._actor_columns.pop(handle, None)
        self._actor_row_positions.pop(handle, None)
        self._actor_row_labels.pop(handle, None)
        self._actor_legend.pop(handle, None)
        self._actor_legend_title.pop(handle, None)
        self._forget_legend_source(("actor", handle))
        if refresh:
            self._refresh_legend()

    def _store_actor_metadata(self, handle: int, meta: Optional[dict]) -> None:
        self._drop_actor_metadata(handle, refresh=False)
        if meta is None:
            self._refresh_legend()
            return
        self._actor_hover[handle] = dict(meta["hover_data"])
        self._actor_columns[handle] = dict(meta["columns"])
        self._actor_row_positions[handle] = meta["row_positions"]
        if meta["row_labels"] is not None:
            self._actor_row_labels[handle] = meta["row_labels"]
        legend_items = _copy_legend_items(meta.get("legend_items"))
        if legend_items:
            self._actor_legend[handle] = legend_items
            legend_title = meta.get("legend_title")
            if legend_title is not None:
                self._actor_legend_title[handle] = str(legend_title)
            self._remember_legend_source(("actor", handle))
        self._refresh_legend()

    def _clear_all_point_metadata(self) -> None:
        self._clear_scene_metadata(refresh=False)
        self._actor_hover.clear()
        self._actor_columns.clear()
        self._actor_row_positions.clear()
        self._actor_row_labels.clear()
        self._actor_legend.clear()
        self._actor_legend_title.clear()
        self._legend_order.clear()
        self._pending_point_meta = None
        self._pending_actor_meta.clear()
        self._refresh_legend()

    @staticmethod
    def _coerce_color_array(colors: Optional["np.ndarray"], n: int) -> "np.ndarray | None":
        if colors is None:
            return None
        clr = np.ascontiguousarray(colors, dtype=np.float32)
        if clr.ndim != 2 or clr.shape[0] != n or clr.shape[1] != 3:
            raise ValueError(f"colors must be shape (N, 3), got {clr.shape}")
        return clr

    @staticmethod
    def _coerce_scalar_array(scalars: Optional["np.ndarray"], n: int) -> "np.ndarray | None":
        if scalars is None:
            return None
        scl = np.ascontiguousarray(scalars, dtype=np.float32)
        if scl.ndim != 1 or scl.shape[0] != n:
            raise ValueError(f"scalars length {scl.shape[0]} must match N={n}")
        return scl

    @staticmethod
    def _normalize_sizes(
        values: "np.ndarray",
        min_px: float = 2.0,
        max_px: float = 20.0,
        fallback: float = 4.0,
    ) -> "np.ndarray":
        """Map *values* linearly into [min_px, max_px].  NaN → *fallback*.

        Both *min_px* and *max_px* must be non-negative; negative values would
        produce invisible or undefined rendering.
        """
        if min_px < 0.0 or max_px < 0.0:
            raise ValueError(
                f"size_range must be non-negative, got ({min_px}, {max_px})"
            )
        arr = np.asarray(values, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return np.full(arr.shape, fallback, dtype=np.float32)
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmax == vmin:
            mid = (min_px + max_px) * 0.5
            return np.where(np.isfinite(arr), mid, fallback).astype(np.float32)
        t = (arr - vmin) / (vmax - vmin)
        return np.where(
            np.isfinite(arr),
            min_px + t * (max_px - min_px),
            fallback,
        ).astype(np.float32)

    def _prepare_point_inputs(
        self,
        positions,
        *,
        x=None,
        y=None,
        z=None,
        colors: Optional["np.ndarray"] = None,
        scalars: Optional["np.ndarray"] = None,
        color=None,
        size=None,
        size_range: "tuple[float, float]" = (2.0, 20.0),
        point_sizes: Optional["np.ndarray"] = None,
        hover=None,
    ) -> "tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, dict | None]":
        is_dataframe = _is_supported_dataframe(positions)

        if is_dataframe:
            if color is not None and (colors is not None or scalars is not None):
                raise ValueError("color= is mutually exclusive with colors= and scalars=")

            meta = _coerce_dataframe(positions, x, y, z=z, color=color, size=size, hover=hover)
            pos = meta["positions"]
            n = pos.shape[0]
            clr = self._coerce_color_array(colors, n)
            scl = self._coerce_scalar_array(scalars, n)

            if color is not None:
                color_values = meta["color_values"]
                if color_values is None:
                    raise ValueError("color column was not extracted")
                cat = _try_encode_categorical(color_values)
                if cat is not None:
                    encoded_colors, legend_items = cat
                    clr = self._coerce_color_array(encoded_colors, n)
                    scl = None
                    meta["legend_items"] = legend_items
                    meta["legend_title"] = meta["columns"].get("color")
                else:
                    scl = self._coerce_scalar_array(color_values, n)

            # Resolve per-point sizes: size= column OR point_sizes= array, not both.
            sizes: "np.ndarray | None" = None
            raw_sizes = meta.get("size_values")
            if raw_sizes is not None and point_sizes is not None:
                raise ValueError(
                    "size= and point_sizes= are mutually exclusive; "
                    "use size= for DataFrame column normalization "
                    "or point_sizes= for a direct pre-computed array"
                )
            if raw_sizes is not None:
                min_px, max_px = size_range
                sizes = self._normalize_sizes(raw_sizes, min_px=min_px, max_px=max_px)
            elif point_sizes is not None:
                szs = np.ascontiguousarray(point_sizes, dtype=np.float32)
                if szs.ndim != 1 or szs.shape[0] != n:
                    raise ValueError(f"point_sizes length {szs.shape[0]} must match N={n}")
                sizes = _clamp_sizes(szs)

            return pos, clr, scl, sizes, meta

        if any(value is not None for value in (x, y, z, color, size, hover)):
            raise ValueError(
                "x, y, z, color, size, and hover are only supported when positions is a DataFrame"
            )
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must be shape (N, 3), got {pos.shape}")
        n = pos.shape[0]
        clr = self._coerce_color_array(colors, n)
        scl = self._coerce_scalar_array(scalars, n)
        # Accept a pre-computed per-point sizes array for the raw-numpy path.
        szs: "np.ndarray | None" = None
        if point_sizes is not None:
            szs = np.ascontiguousarray(point_sizes, dtype=np.float32)
            if szs.ndim != 1 or szs.shape[0] != n:
                raise ValueError(f"point_sizes length {szs.shape[0]} must match N={n}")
            szs = _clamp_sizes(szs)
        return pos, clr, scl, szs, None

    def set_points(
        self,
        positions,
        *,
        x=None,
        y=None,
        z=None,
        colors: Optional["np.ndarray"] = None,
        scalars: Optional["np.ndarray"] = None,
        color=None,
        size=None,
        size_range: "tuple[float, float]" = (2.0, 20.0),
        point_sizes: Optional["np.ndarray"] = None,
        hover=None,
        colormap: str = "viridis",
        point_size: float = 4.0,
        clim: "tuple[float, float] | None" = None,
        nan_color: "tuple[float, float, float] | None" = None,
        log_scale: bool = False,
        opacity: float = 1.0,
    ) -> None:
        """Replace all point actors with a single new point cloud.

        Line overlays added via :meth:`add_lines` or :meth:`add_box` are
        **not** affected. Call :meth:`clear` first if you want to reset the
        entire scene including overlays.

        Parameters
        ----------
        positions : (N, 3) float32 array or DataFrame
            XYZ coordinates as a numpy array, or a pandas / polars DataFrame.
        x, y, z : str, optional
            Column names used when *positions* is a DataFrame. ``x`` and ``y``
            are required in that mode. ``z`` defaults to 0 when omitted.
        colors : (N, 3) float32 array, optional
            Per-point RGB in [0, 1]. Highest priority; ignores *scalars* and
            *colormap* when provided.
        scalars : (N,) float32 array, optional
            Per-point scalar values mapped through *colormap*. Used when
            *colors* is not provided.
        color : str, optional
            DataFrame column used for color. Continuous numeric columns map
            through *colormap*; categorical columns are encoded to per-point
            RGB colors with an automatic legend.
        size : str, optional
            DataFrame column whose values control per-point diameter in pixels.
            Values are normalized to *size_range*.
        size_range : (min_px, max_px), optional
            Pixel range for the *size* column normalization. Defaults to
            ``(2.0, 20.0)``.
        point_sizes : (N,) float32 array, optional
            Per-point sizes in pixels.  Mutually exclusive with *size*: raises
            ``ValueError`` if both are provided for DataFrame input.  Non-finite
            and negative values are clamped to 0.
        colormap : str
            Colormap applied to *scalars*, or to the Z coordinate when neither
            *colors* nor *scalars* are given. Defaults to ``"viridis"``.
        point_size : float
            Uniform point diameter in pixels (used when neither *size* column
            nor *point_sizes* array is provided). Defaults to ``4.0``.
        clim : (vmin, vmax), optional
            Fix the colormap range. Values outside the range are clamped.
        nan_color : (r, g, b), optional
            RGB color in [0, 1] for NaN / non-finite scalars.
        log_scale : bool
            Apply logarithmic normalization before colormap sampling.
        """
        pos, clr, scl, sizes, meta = self._prepare_point_inputs(
            positions,
            x=x,
            y=y,
            z=z,
            colors=colors,
            scalars=scalars,
            color=color,
            size=size,
            size_range=size_range,
            point_sizes=point_sizes,
            hover=hover,
        )
        n = pos.shape[0]

        clim_arr = list(clim) if clim is not None else None
        nan_arr  = list(nan_color) if nan_color is not None else None

        if self._renderer is None:
            self._pending = dict(
                positions=pos, colors=clr, scalars=scl, point_sizes=sizes,
                colormap=colormap, point_size=point_size,
                clim=clim, nan_color=nan_color, log_scale=log_scale,
                opacity=opacity,
            )
            self._pending_point_meta = meta
            return

        self._clear_all_point_metadata()
        self._actor_n.clear()
        _handle = self._renderer.set_points(pos, clr, scl, colormap, float(point_size),
                                             sizes, clim_arr, nan_arr,
                                             bool(log_scale), float(opacity))
        self._set_scene_metadata(meta, actor_handle=_handle)
        self._actor_n[int(_handle)] = n
        self._total_n = n
        self._mark_dirty()

    def add_points(
        self,
        positions,
        *,
        x=None,
        y=None,
        z=None,
        colors: Optional["np.ndarray"] = None,
        scalars: Optional["np.ndarray"] = None,
        color=None,
        size=None,
        size_range: "tuple[float, float]" = (2.0, 20.0),
        point_sizes: Optional["np.ndarray"] = None,
        hover=None,
        colormap: str = "viridis",
        point_size: float = 4.0,
        clim: "tuple[float, float] | None" = None,
        nan_color: "tuple[float, float, float] | None" = None,
        log_scale: bool = False,
        opacity: float = 1.0,
    ) -> int:
        """Add a new point cloud actor on top of the existing scene.

        Returns an integer handle. Pass it to ``update_actor()``,
        ``remove_actor()``, or ``set_actor_visibility()`` to manipulate the
        actor later. A virtual handle (non-negative int) is returned even before
        the widget is mapped; it is resolved to a real renderer handle when the
        widget is first mapped.
        """
        pos, clr, scl, sizes, meta = self._prepare_point_inputs(
            positions,
            x=x,
            y=y,
            z=z,
            colors=colors,
            scalars=scalars,
            color=color,
            size=size,
            size_range=size_range,
            point_sizes=point_sizes,
            hover=hover,
        )
        n = pos.shape[0]

        kwargs = dict(positions=pos, colors=clr, scalars=scl, point_sizes=sizes,
                      colormap=colormap, point_size=point_size,
                      clim=clim, nan_color=nan_color, log_scale=log_scale,
                      opacity=float(opacity))

        if self._renderer is None:
            vhandle = self._next_phandle
            self._next_phandle += 1
            self._pending_actors.append((kwargs, vhandle))
            if meta is not None:
                self._pending_actor_meta[vhandle] = meta
            return vhandle

        self._clear_scene_metadata()
        handle = int(self._renderer.add_points(**kwargs))
        if handle != 0xFFFFFFFF:
            self._store_actor_metadata(handle, meta)
            self._actor_n[handle] = n
            self._total_n += n
        self._mark_dirty()
        return handle if handle != 0xFFFFFFFF else -1

    def _resolve_actor_handle(self, handle: int) -> int:
        """Translate a virtual (pre-map) actor handle to the real renderer handle."""
        return self._phandle_map.get(handle, handle)

    def update_actor(
        self,
        handle: int,
        positions: "np.ndarray",
        *,
        colors: Optional["np.ndarray"] = None,
        scalars: Optional["np.ndarray"] = None,
        point_sizes: Optional["np.ndarray"] = None,
        colormap: str = "viridis",
        point_size: float = 4.0,
        clim: "tuple[float, float] | None" = None,
        nan_color: "tuple[float, float, float] | None" = None,
        log_scale: bool = False,
        opacity: float = 1.0,
    ) -> None:
        """Replace the data of an existing actor in place."""
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must be shape (N, 3), got {pos.shape}")
        n = pos.shape[0]
        clr = np.ascontiguousarray(colors, dtype=np.float32) if colors is not None else None
        scl = np.ascontiguousarray(scalars, dtype=np.float32) if scalars is not None else None
        szs: "np.ndarray | None" = None
        if point_sizes is not None:
            szs = np.ascontiguousarray(point_sizes, dtype=np.float32)
            if szs.ndim != 1 or szs.shape[0] != n:
                raise ValueError(f"point_sizes length {szs.shape[0]} must match N={n}")
            szs = _clamp_sizes(szs)
        if self._renderer is None:
            for i, (kwargs, vhandle) in enumerate(self._pending_actors):
                if vhandle == handle:
                    self._pending_actors[i] = (dict(
                        positions=pos, colors=clr, scalars=scl, point_sizes=szs,
                        colormap=colormap, point_size=float(point_size),
                        clim=clim, nan_color=nan_color, log_scale=bool(log_scale),
                        opacity=float(opacity),
                    ), vhandle)
                    self._pending_actor_meta.pop(handle, None)
                    return
            return
        real = self._resolve_actor_handle(handle)
        self._drop_actor_metadata(real)
        self._renderer.update_actor(real, pos,
                                     colors=clr, scalars=scl, point_sizes=szs,
                                     colormap=colormap, point_size=point_size,
                                     clim=clim, nan_color=nan_color, log_scale=log_scale,
                                     opacity=float(opacity))
        old_n = self._actor_n.get(real, 0)
        self._actor_n[real] = n
        self._total_n = max(0, self._total_n - old_n + n)
        self._mark_dirty()

    # ── Streaming API ─────────────────────────────────────────────────────────

    def add_stream(
        self,
        positions=None,
        *,
        max_points: int,
        mode: str = "ring",
        colormap: str = "viridis",
        point_size: float = 4.0,
        colors: Optional["np.ndarray"] = None,
        scalars: Optional["np.ndarray"] = None,
        clim: "tuple[float, float] | None" = None,
        nan_color: "tuple[float, float, float] | None" = None,
        log_scale: bool = False,
        opacity: float = 1.0,
    ) -> int:
        """Pre-allocate a streaming point cloud actor.

        The GPU buffer is sized once for `max_points` and never reallocated.

        Parameters
        ----------
        positions:
            Optional (N, 3) float32 array of initial seed points.
        max_points:
            Maximum number of points the buffer can hold.
        mode:
            ``"ring"`` (overwrite oldest when full) or ``"append"`` (stop when full).

        Returns
        -------
        int
            Handle for use with :meth:`stream` and :meth:`clear_stream`.
        """
        mode_int = 0 if mode == "append" else 1

        pos: "np.ndarray | None" = None
        if positions is not None:
            pos = np.ascontiguousarray(positions, dtype=np.float32)
            if pos.ndim == 2 and pos.shape[1] == 2:
                pos = np.column_stack([pos, np.zeros(pos.shape[0], dtype=np.float32)])
            if pos.ndim != 2 or pos.shape[1] != 3:
                raise ValueError(f"positions must be shape (N,3) or (N,2), got {pos.shape}")

        clr = np.ascontiguousarray(colors, dtype=np.float32) if colors is not None else None
        scl = np.ascontiguousarray(scalars, dtype=np.float32) if scalars is not None else None

        kwargs = dict(
            max_points=max_points, mode=mode_int,
            positions=pos, colors=clr, scalars=scl,
            colormap=colormap, point_size=float(point_size),
            clim=clim, nan_color=nan_color,
            log_scale=log_scale, opacity=float(opacity),
        )

        if self._renderer is None:
            vhandle = self._next_phandle
            self._next_phandle += 1
            self._pending_streams.append((kwargs, vhandle))
            return vhandle

        handle = int(self._renderer.create_stream(**kwargs))
        self._stream_handles.add(handle)
        self._mark_dirty()
        return handle

    def stream(
        self,
        handle: int,
        positions,
        *,
        colormap: str = "viridis",
        point_size: float = 4.0,
        colors: Optional["np.ndarray"] = None,
        scalars: Optional["np.ndarray"] = None,
        clim: "tuple[float, float] | None" = None,
        nan_color: "tuple[float, float, float] | None" = None,
        log_scale: bool = False,
        opacity: float = 1.0,
    ) -> None:
        """Append new points to a stream actor.

        For ``mode="append"``: stops accepting points once the buffer is full.
        For ``mode="ring"``:   overwrites the oldest points when the buffer is full.

        Parameters
        ----------
        handle:
            Handle returned by :meth:`add_stream`.
        positions:
            (N, 3) float32 array of new points to append.
        """
        if self._renderer is None:
            return
        real = self._resolve_actor_handle(handle)
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        if pos.ndim == 2 and pos.shape[1] == 2:
            pos = np.column_stack([pos, np.zeros(pos.shape[0], dtype=np.float32)])
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must be shape (N,3) or (N,2), got {pos.shape}")
        clr = np.ascontiguousarray(colors, dtype=np.float32) if colors is not None else None
        scl = np.ascontiguousarray(scalars, dtype=np.float32) if scalars is not None else None
        self._renderer.stream_points(
            real, pos,
            colors=clr, scalars=scl,
            colormap=colormap, point_size=float(point_size),
            clim=clim, nan_color=nan_color, log_scale=log_scale,
            opacity=float(opacity),
        )
        self._mark_dirty()

    def clear_stream(self, handle: int) -> None:
        """Reset a stream actor to empty; the pre-allocated GPU capacity is kept."""
        if self._renderer is None:
            return
        real = self._resolve_actor_handle(handle)
        self._renderer.clear_stream(real)
        self._mark_dirty()

    def remove_actor(self, handle: int) -> None:
        """Remove a point cloud actor by handle."""
        if self._renderer is None:
            self._pending_actors = [(kw, vh) for kw, vh in self._pending_actors if vh != handle]
            self._pending_actor_meta.pop(handle, None)
            self._pending_actor_visibility.pop(handle, None)
            return
        real = self._resolve_actor_handle(handle)
        self._renderer.remove_actor(real)
        self._drop_actor_metadata(real)
        self._stream_handles.discard(real)
        self._total_n = max(0, self._total_n - self._actor_n.pop(real, 0))
        self._mark_dirty()

    def set_actor_visibility(self, handle: int, visible: bool) -> None:
        """Show or hide an actor without removing it."""
        if self._renderer is None:
            self._pending_actor_visibility[handle] = visible
            return
        self._renderer.set_actor_visibility(self._resolve_actor_handle(handle), visible)
        self._mark_dirty()

    def clear(self) -> None:
        """Remove all point actors, all line overlays, and clear the scene."""
        self._pending = None
        self._pending_actors.clear()
        self._pending_streams.clear()
        self._stream_handles.clear()
        self._phandle_map.clear()
        self._next_phandle = 0
        self._total_n = 0
        self._actor_n.clear()
        self._clear_all_point_metadata()
        self._pending_overlays.clear()
        self._pending_actor_visibility.clear()
        self._pending_overlay_visibility.clear()
        self._vhandle_map.clear()
        self._next_vhandle = 0
        if self._renderer is not None:
            self._renderer.clear_actors()
            self._renderer.clear_overlays()
            self._mark_dirty()

    # ── Line / overlay actors ─────────────────────────────────────────────────

    def add_lines(
        self,
        segments: "np.ndarray",
        color: "tuple[float, float, float]" = (1.0, 1.0, 1.0),
    ) -> int:
        """Add a set of line segments as an overlay actor in world space.

        Parameters
        ----------
        segments : (N, 6) float32 array
            Each row is ``[x0, y0, z0, x1, y1, z1]`` defining one segment.
        color : (r, g, b)
            RGB colour in ``[0, 1]``.

        Returns
        -------
        int
            A handle to pass to ``update_lines``, ``remove_overlay``, etc.
            Returns a virtual handle (non-negative int) even before the renderer
            is initialized; the handle is resolved to a real renderer handle when
            the widget is first mapped.
        """
        seg = np.ascontiguousarray(segments, dtype=np.float32)
        if seg.ndim != 2 or seg.shape[1] != 6:
            raise ValueError(f"segments must be shape (N, 6), got {seg.shape}")
        if self._renderer is None:
            vhandle = self._next_vhandle
            self._next_vhandle += 1
            self._pending_overlays.append(("lines", seg, color, vhandle))
            return vhandle
        handle = int(self._renderer.add_lines(seg, color))
        self._mark_dirty()
        return handle

    def add_box(
        self,
        bounds: "tuple[float, float, float, float, float, float]",
        color: "tuple[float, float, float]" = (1.0, 1.0, 0.0),
    ) -> int:
        """Add a wireframe bounding box.

        Parameters
        ----------
        bounds : (xmin, ymin, zmin, xmax, ymax, zmax)
        color : (r, g, b) in ``[0, 1]``.

        Returns
        -------
        int
            A handle usable with ``remove_overlay``, ``set_overlay_visibility``,
            etc. Returns a virtual handle before the renderer is initialized;
            see :meth:`add_lines`.
        """
        x0, y0, z0, x1, y1, z1 = bounds
        edges = np.array([
            [x0, y0, z0, x1, y0, z0], [x1, y0, z0, x1, y1, z0],
            [x1, y1, z0, x0, y1, z0], [x0, y1, z0, x0, y0, z0],
            [x0, y0, z1, x1, y0, z1], [x1, y0, z1, x1, y1, z1],
            [x1, y1, z1, x0, y1, z1], [x0, y1, z1, x0, y0, z1],
            [x0, y0, z0, x0, y0, z1], [x1, y0, z0, x1, y0, z1],
            [x1, y1, z0, x1, y1, z1], [x0, y1, z0, x0, y1, z1],
        ], dtype=np.float32)
        return self.add_lines(edges, color)

    def _resolve_handle(self, handle: int) -> int:
        """Translate a virtual (pre-map) handle to the real renderer handle."""
        return self._vhandle_map.get(handle, handle)

    def update_lines(
        self,
        handle: int,
        segments: "np.ndarray",
        color: "tuple[float, float, float]" = (1.0, 1.0, 1.0),
    ) -> None:
        """Replace the geometry of an existing line overlay actor."""
        seg = np.ascontiguousarray(segments, dtype=np.float32)
        if seg.ndim != 2 or seg.shape[1] != 6:
            raise ValueError(f"segments must be shape (N, 6), got {seg.shape}")
        if self._renderer is None:
            for i, (method, _seg, _color, vhandle) in enumerate(self._pending_overlays):
                if vhandle == handle:
                    self._pending_overlays[i] = (method, seg, color, vhandle)
                    return
            return
        self._renderer.update_lines(self._resolve_handle(handle), seg, color)
        self._mark_dirty()

    def remove_overlay(self, handle: int) -> None:
        """Remove a line overlay actor by handle."""
        if self._renderer is None:
            self._pending_overlays = [(m, s, c, vh) for m, s, c, vh in self._pending_overlays if vh != handle]
            self._pending_overlay_visibility.pop(handle, None)
            return
        self._renderer.remove_overlay(self._resolve_handle(handle))
        self._mark_dirty()

    def set_overlay_visibility(self, handle: int, visible: bool) -> None:
        """Show or hide a line overlay actor."""
        if self._renderer is None:
            self._pending_overlay_visibility[handle] = visible
            return
        self._renderer.set_overlay_visibility(self._resolve_handle(handle), visible)
        self._mark_dirty()

    def clear_overlays(self) -> None:
        """Remove all line overlay actors."""
        self._pending_overlays.clear()
        self._pending_overlay_visibility.clear()
        self._vhandle_map.clear()
        self._next_vhandle = 0
        if self._renderer is not None:
            self._renderer.clear_overlays()
            self._mark_dirty()

    def show_orientation_axes(self, visible: bool = True) -> None:
        """Show or hide the orientation axes widget in the bottom-left corner."""
        self._orientation_axes_visible = visible
        if self._renderer is not None:
            self._renderer.show_orientation_axes(visible)
            self._mark_dirty()

    # ── Export ────────────────────────────────────────────────────────────────

    def screenshot(self) -> "np.ndarray | None":
        """Capture the current scene as an RGBA uint8 NumPy array of shape (H, W, 4).

        Returns ``None`` when the renderer has not been initialized yet (i.e.
        the widget has never been mapped on screen).
        """
        if self._renderer is None:
            return None
        w, h, raw = self._renderer.screenshot()
        # raw is already a 1-D uint8 numpy array (zero-copy from Rust); reshape as view.
        return np.asarray(raw).reshape(h, w, 4)

    def save_png(self, path: str) -> None:
        """Save the current scene to a PNG file.

        Uses *Pillow* when available; falls back to a pure-stdlib PNG writer
        otherwise, so no optional dependency is required.

        Parameters
        ----------
        path : str
            Destination file path (should end in ``.png``).
        """
        img = self.screenshot()
        if img is None:
            raise RuntimeError("Widget has not been mapped — call after the window is shown.")
        try:
            from PIL import Image as _PILImage
            _PILImage.fromarray(img, mode="RGBA").save(path)
        except ImportError:
            _write_png(path, img)

    # ── Animation export ──────────────────────────────────────────────────────

    def open_gif(self, path: str, fps: int = 20, loop: int = 0) -> None:
        """Begin recording frames for an animated GIF.

        Call :meth:`write_frame` for each frame you want, then
        :meth:`close_gif` to write the file.

        Parameters
        ----------
        path : str
            Output ``.gif`` path.
        fps : int
            Target playback speed in frames per second.
        loop : int
            Number of times the GIF loops; ``0`` = infinite.
        """
        if fps < 1:
            raise ValueError(f"fps must be >= 1, got {fps}")
        self._gif_frames = []
        self._gif_path = path
        self._gif_fps = fps
        self._gif_loop = loop
        self._gif_tmp_dir = None

    def write_frame(self) -> None:
        """Capture the current scene and append it to the active GIF recording.

        Raises ``RuntimeError`` if called before :meth:`open_gif`.
        """
        if self._gif_frames is None:
            raise RuntimeError("Call open_gif() before write_frame().")
        img = self.screenshot()
        if img is None:
            return
        # Stream each frame to a temp file on disk so the capture loop holds
        # only one frame in RAM at a time.  Fall back to in-memory list when
        # PIL is absent (temp files need PIL to write single-frame GIFs).
        try:
            from PIL import Image as _PILImage
            import tempfile as _tempfile, os as _os
            if self._gif_tmp_dir is None:
                self._gif_tmp_dir = _tempfile.mkdtemp(prefix="dragonsci_gif_")
            idx = len(self._gif_frames)
            frame_path = _os.path.join(
                self._gif_tmp_dir, f"frame_{idx:06d}.gif"
            )
            _PILImage.fromarray(img, "RGBA").convert(
                "P", palette=_PILImage.Palette.ADAPTIVE
            ).save(frame_path, format="GIF")
            self._gif_frames.append(frame_path)
        except ImportError:
            self._gif_frames.append(img)

    def close_gif(self) -> None:
        """Finalise and write the GIF file started by :meth:`open_gif`.

        Uses Pillow when available for better colour quality; falls back to a
        pure-stdlib encoder with 3-3-2 quantisation.  Safe to call multiple
        times (second call is a no-op).
        """
        if self._gif_frames is None:
            return
        frames, path, fps, loop, tmp_dir = (
            self._gif_frames, self._gif_path, self._gif_fps, self._gif_loop,
            self._gif_tmp_dir,
        )
        self._gif_frames = None
        self._gif_path = None
        self._gif_tmp_dir = None
        if not frames or path is None:
            return
        try:
            from PIL import Image as _Im
            import numpy as _np, os as _os
            delay_ms = max(20, round(1000 / fps))
            if frames and isinstance(frames[0], str):
                # Frames are temp-file paths written during capture; open lazily
                # so PIL loads each frame on demand rather than all at once.
                imgs = [_Im.open(p) for p in frames]
            elif frames and isinstance(frames[0], _np.ndarray):
                # PIL was absent at capture time; convert now.
                imgs = [_Im.fromarray(f, "RGBA").convert("P", palette=_Im.Palette.ADAPTIVE)
                        for f in frames]
            else:
                imgs = frames  # already PIL P-mode (legacy in-memory path)
            imgs[0].save(
                path,
                save_all=True,
                append_images=imgs[1:],
                duration=delay_ms,
                loop=loop,
                optimize=False,
            )
            for img in imgs:
                img.close()
        except ImportError:
            _write_gif_stdlib(path, frames, fps, loop)
        finally:
            # Clean up temp files regardless of success or error.
            if tmp_dir is not None:
                import os as _os
                for p in frames:
                    if isinstance(p, str):
                        try:
                            _os.unlink(p)
                        except OSError:
                            pass
                try:
                    _os.rmdir(tmp_dir)
                except OSError:
                    pass

    def orbit_gif(
        self,
        path: str,
        n_frames: int = 60,
        fps: int = 20,
        loop: int = 0,
        elevation: "float | None" = None,
        on_progress: "callable | None" = None,
    ) -> None:
        """Orbit the camera 360° and save as an animated GIF.

        Parameters
        ----------
        path : str
            Output ``.gif`` path.
        n_frames : int
            Number of frames in the animation (default 60 = 3 s at 20 fps).
        fps : int
            Playback speed.
        loop : int
            ``0`` = infinite loop.
        elevation : float or None
            Camera pitch in radians.  ``None`` keeps the current pitch.
        on_progress : callable or None
            Called as ``on_progress(frame_index, n_frames)`` after each frame.
            Useful for updating a progress label.
        """
        if self._renderer is None:
            raise RuntimeError("Widget must be mapped before recording.")
        import math
        saved = self._renderer.get_camera()
        pitch = elevation if elevation is not None else saved.get("pitch", 0.3)
        yaw0 = float(saved.get("yaw", 0.0))

        self.open_gif(path, fps=fps, loop=loop)
        try:
            for i in range(n_frames):
                state = dict(saved)
                state["yaw"] = yaw0 + 2.0 * math.pi * i / n_frames
                state["pitch"] = pitch
                self._renderer.set_camera(state)
                self.write_frame()
                if on_progress is not None:
                    on_progress(i, n_frames)
        finally:
            self.close_gif()
            self._renderer.set_camera(saved)
            self._mark_dirty()

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
        """Show or hide the scalar bar overlay.

        Parameters
        ----------
        visible : bool
            Show the scalar bar when True, hide it when False.
        vmin, vmax : float
            The data range the colormap spans.
        log_scale : bool
            Mirror the log_scale used in set_points so tick labels are correct.
        colormap : str
            Colormap name (should match the one used in set_points).
        title : str
            Optional label drawn above the bar.
        """
        if self._renderer is not None:
            self._renderer.show_scalar_bar(visible, vmin, vmax, log_scale, colormap, title)
            self._mark_dirty()
        else:
            self._pending_scalar_bar = {
                "visible": visible, "vmin": vmin, "vmax": vmax,
                "log_scale": log_scale, "colormap": colormap, "title": title,
            }

    # ── Linked-camera support ─────────────────────────────────────────────────

    def _propagate_camera(self) -> None:
        """Push our current camera state to all linked widgets."""
        if self._propagating or self._renderer is None or not self._camera_links:
            return
        state = self._renderer.get_camera()
        dead: list = []
        for other in self._camera_links:
            try:
                other._receive_camera(state)
            except Exception:
                dead.append(other)
        for d in dead:
            self._camera_links.discard(d)

    def _receive_camera(self, state: dict) -> None:
        """Apply a camera state coming from a linked widget (no further propagation)."""
        if self._renderer is None:
            return
        self._propagating = True
        try:
            self._renderer.set_camera(state)
            self._mark_dirty()
        finally:
            self._propagating = False

    def reset_camera(self) -> None:
        """Reset the camera to the fitted view for the current dataset."""
        if self._renderer is not None:
            self._renderer.reset_camera()
            self._mark_dirty()
            self._propagate_camera()

    # ── Camera presets ────────────────────────────────────────────────────────

    def view_xy(self) -> None:
        """Bird's-eye view: camera at +Y looking down at the XZ plane."""
        if self._renderer is not None:
            self._renderer.view_xy()
            self._mark_dirty()
            self._propagate_camera()

    def view_xz(self) -> None:
        """Front view: camera at +Z looking at the XY plane (X right, Y up).

        This is also the default view used by :class:`Scatter2D`.
        """
        if self._renderer is not None:
            self._renderer.view_xz()
            self._mark_dirty()
            self._propagate_camera()

    def view_yz(self) -> None:
        """Look along -X onto the YZ plane (side view)."""
        if self._renderer is not None:
            self._renderer.view_yz()
            self._mark_dirty()
            self._propagate_camera()

    def view_isometric(self) -> None:
        """45°/45° isometric view."""
        if self._renderer is not None:
            self._renderer.view_isometric()
            self._mark_dirty()
            self._propagate_camera()

    def flatten_view(self, plane: str = "xy") -> None:
        """Snap to an axis-aligned view of *plane* and enable parallel
        (orthographic) projection for a true flat look.

        Parameters
        ----------
        plane:
            Which plane to look straight at:

            - ``"xy"``  / ``"xy-"`` — from +Z / −Z (X right, Y up)
            - ``"xz"``  / ``"xz-"`` — from +Y / −Y (X right, Z "up")
            - ``"yz"``  / ``"yz-"`` — from +X / −X (Y up, Z right)
        """
        if plane not in _FLATTEN_PLANES:
            raise ValueError(
                f"plane must be one of: {', '.join(sorted(_FLATTEN_PLANES))}"
            )
        yaw, pitch = _FLATTEN_PLANES[plane]
        if self._renderer is None:
            return
        state = self._renderer.get_camera()
        state["yaw"]     = yaw
        state["pitch"]   = pitch
        state["parallel"] = True
        self._renderer.set_camera(state)
        self._parallel_projection = True
        self._mark_dirty()
        self._propagate_camera()

    @property
    def parallel_projection(self) -> bool:
        """True when orthographic projection is active."""
        return self._parallel_projection

    @parallel_projection.setter
    def parallel_projection(self, on: bool) -> None:
        self._parallel_projection = bool(on)
        if self._renderer is not None:
            self._renderer.set_parallel_projection(self._parallel_projection)
            self._mark_dirty()
            self._propagate_camera()

    def fit(self, bounds: "tuple[float,...] | None" = None) -> None:
        """Fit camera to *bounds* ``(xmin,ymin,zmin,xmax,ymax,zmax)`` or to the current dataset."""
        if self._renderer is not None:
            self._renderer.fit(list(bounds) if bounds is not None else None)
            self._mark_dirty()
            self._propagate_camera()

    def get_camera(self) -> dict:
        """Return the current camera state as a dict (serialisable, passable to set_camera)."""
        if self._renderer is not None:
            return self._renderer.get_camera()
        return {}

    def set_camera(self, state: dict) -> None:
        """Restore a camera state dict previously returned by get_camera()."""
        if self._renderer is not None:
            self._renderer.set_camera(state)
            self._mark_dirty()
            self._propagate_camera()

    # ── Rendering modes ───────────────────────────────────────────────────────

    _STYLE_MAP = {"circle": 0, "square": 1, "gaussian": 2}

    @property
    def point_style(self) -> str:
        """Point rendering style: ``"circle"`` (default), ``"square"``, or ``"gaussian"``."""
        return self._point_style

    @point_style.setter
    def point_style(self, style: str) -> None:
        code = self._STYLE_MAP.get(style)
        if code is None:
            raise ValueError(f"point_style must be 'circle', 'square', or 'gaussian', got {style!r}")
        self._point_style = style
        if self._renderer is not None:
            self._renderer.set_point_style(code)
            self._mark_dirty()

    def set_ticks(
        self,
        x: int | None = None,
        y: int | None = None,
        z: int | None = None,
    ) -> None:
        """Set max tick count per axis. Pass None to restore auto-scaling."""
        if self._renderer is not None:
            self._renderer.set_ticks(x=x, y=y, z=z)
            self._mark_dirty()
        else:
            self._pending_ticks = (x, y, z)

    # ── Visual appearance ─────────────────────────────────────────────────────

    def show_grid(self, visible: bool = True) -> None:
        """Show or hide the grid lines and tick labels."""
        self._grid_visible = visible
        if self._renderer is not None:
            self._renderer.show_grid(visible)
            self._mark_dirty()

    def show_grid_planes(self, major: bool = True, minor: bool = False) -> None:
        """Show or hide grid lines on the axis planes.

        Parameters
        ----------
        major:
            Draw lines at each tick position on the three near faces of the
            bounding box.  Default ``True``.
        minor:
            Additionally draw lines that subdivide each major interval into 5.
            Default ``False``.
        """
        self._major_grid_planes = bool(major)
        self._minor_grid_planes = bool(minor)
        if self._renderer is not None:
            self._renderer.show_grid_planes(self._major_grid_planes, self._minor_grid_planes)
            self._mark_dirty()

    def show_legend(self, visible: bool = True) -> None:
        """Show or hide the categorical legend overlay."""
        self._legend_visible = bool(visible)
        self._refresh_legend()

    @property
    def legend_position(self) -> str:
        """Legend corner placement: top-right, top-left, bottom-right, or bottom-left."""
        return self._legend_position

    @legend_position.setter
    def legend_position(self, position: str) -> None:
        if position not in _LEGEND_POSITION_IDX:
            raise ValueError(
                "legend_position must be one of: "
                + ", ".join(sorted(_LEGEND_POSITION_IDX))
            )
        self._legend_position = position
        self._refresh_legend()

    def set_background(self, color) -> None:
        """Set the background colour.

        Parameters
        ----------
        color : tuple or str
            Either an ``(r, g, b)`` tuple with float values in ``[0, 1]`` or a
            hex string such as ``"#0d0d12"``.
        """
        if isinstance(color, str):
            h = color.lstrip("#")
            if len(h) == 6:
                r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            else:
                raise ValueError(f"set_background: expected '#RRGGBB' hex string, got {color!r}")
        else:
            r, g, b = float(color[0]), float(color[1]), float(color[2])
        self._bg_color = (r, g, b)
        if self._renderer is not None:
            self._renderer.set_background_color(r, g, b)
            self._mark_dirty()

    def set_axes(
        self,
        x: str = "X",
        y: str = "Y",
        z: str = "Z",
    ) -> None:
        """Set the axis title labels displayed at the grid extents.

        Parameters
        ----------
        x, y, z : str
            Label text for each axis. Pass an empty string ``""`` to hide a
            title without affecting the other two.
        """
        self._axis_labels = (x, y, z)
        if self._renderer is not None:
            self._renderer.set_axis_labels(x, y, z)
            self._mark_dirty()

    def set_axis_visibility(
        self,
        x: bool = True,
        y: bool = True,
        z: bool = True,
    ) -> None:
        """Show or hide individual axes.

        Hiding an axis removes its tick marks, tick labels, axis title, and
        the four bounding-box edges that run parallel to it.

        Parameters
        ----------
        x, y, z : bool
            ``True`` to show the axis, ``False`` to hide it.
        """
        self._axis_visible = (bool(x), bool(y), bool(z))
        if self._renderer is not None:
            self._renderer.set_axis_visible(*self._axis_visible)
            self._mark_dirty()

    # ── Picking helpers ───────────────────────────────────────────────────────

    def _translate_hits(self, hits: list) -> "tuple[list[int], list | None]":
        """Translate raw (actor, index) hits to (row_positions, index_labels).

        ``row_positions`` are plotted row positions suitable for ``df.iloc[...]``.
        For numpy-only actors the raw per-actor buffer index is returned.
        ``index_labels`` is populated when at least one actor was DataFrame-backed
        and had a non-trivial pandas index; otherwise ``None``.
        """
        indices: list = []
        labels: list = []
        any_labels = False
        for h in hits:
            actor = h["actor"]
            idx = int(h["index"])
            # Actor-level metadata (add_points path).
            row_pos = self._actor_row_positions.get(actor)
            if row_pos is not None and idx < len(row_pos):
                indices.append(int(row_pos[idx]))
                lbl_arr = self._actor_row_labels.get(actor)
                if lbl_arr is not None and idx < len(lbl_arr):
                    labels.append(lbl_arr[idx])
                    any_labels = True
                else:
                    labels.append(None)
            elif actor == self._scene_actor_handle and self._scene_row_positions is not None \
                    and idx < len(self._scene_row_positions):
                # Scene-level metadata (set_points path).
                indices.append(int(self._scene_row_positions[idx]))
                if self._scene_row_labels is not None and idx < len(self._scene_row_labels):
                    labels.append(self._scene_row_labels[idx])
                    any_labels = True
                else:
                    labels.append(None)
            else:
                indices.append(idx)
                labels.append(None)
        return indices, (labels if any_labels else None)

    def _fire_selection(self, hits: list) -> None:
        """Store translated selection results and fire ``<<SelectionChanged>>``."""
        self.selected = hits
        self.selected_indices, self.selected_index_values = self._translate_hits(hits)
        self.event_generate("<<SelectionChanged>>")

    # ── Picking API ───────────────────────────────────────────────────────────

    def enable_point_picking(self, on_pick=None) -> None:
        """Activate point picking.

        A left-click with no drag finds the nearest visible point and fires
        ``<<PointPicked>>``. Read ``widget.picked_point``,
        ``widget.picked_index``, and ``widget.picked_actor`` in the handler.

        Parameters
        ----------
        on_pick : callable, optional
            Convenience callback bound to ``<<PointPicked>>``. Receives the Tk
            event object; read widget attributes for pick results.
        """
        self._pick_mode = "both" if self._pick_mode == "rect" else "point"
        if on_pick is not None:
            self.bind("<<PointPicked>>", on_pick, add="+")

    def enable_rectangle_picking(self, on_select=None) -> None:
        """Activate rectangle selection via Shift+left-drag.

        On release, fires ``<<SelectionChanged>>``. Read
        ``widget.selected`` (list of ``{"actor": int, "index": int}`` dicts)
        in the handler.

        Parameters
        ----------
        on_select : callable, optional
            Convenience callback bound to ``<<SelectionChanged>>``.
        """
        self._pick_mode = "both" if self._pick_mode == "point" else "rect"
        if on_select is not None:
            self.bind("<<SelectionChanged>>", on_select, add="+")

    def enable_lasso_picking(self, on_select=None) -> None:
        """Activate freehand lasso selection via Ctrl+left-drag.

        On release, fires ``<<SelectionChanged>>``. Read
        ``widget.selected_indices`` (plotted row positions, compatible with
        ``df.iloc[selected_indices]``) and ``widget.selected`` (raw hits) in
        the handler.

        Parameters
        ----------
        on_select : callable, optional
            Convenience callback bound to ``<<SelectionChanged>>``.
        """
        self._lasso_enabled = True
        if on_select is not None:
            self.bind("<<SelectionChanged>>", on_select, add="+")

    def disable_picking(self) -> None:
        """Return to orbit-only mode (no picking)."""
        self._pick_mode = "none"
        self._lasso_enabled = False
        self._lasso_active = False
        if self._renderer is not None:
            self._renderer.clear_selection_rect()
            self._renderer.lasso_cancel()
            self._mark_dirty()

    @staticmethod
    def colormap_names() -> list[str]:
        return ScatterRenderer.colormap_names()

    # ── Hover tooltip ─────────────────────────────────────────────────────────

    @property
    def hover_tooltip(self) -> bool:
        """Show a tooltip on hover when True (default)."""
        return self._hover_tooltip

    @hover_tooltip.setter
    def hover_tooltip(self, enabled: bool) -> None:
        self._hover_tooltip = bool(enabled)
        if not self._hover_tooltip:
            if self._hover_after_id is not None:
                self.after_cancel(self._hover_after_id)
                self._hover_after_id = None
            self._hide_tooltip()

    def _on_hover_motion(self, event: tk.Event) -> None:
        if not self._hover_tooltip or self._drag_btn is not None:
            return
        self._hover_last_x = event.x
        self._hover_last_y = event.y
        if self._hover_after_id is not None:
            self.after_cancel(self._hover_after_id)
        self._hover_after_id = self.after(_HOVER_DEBOUNCE_MS, self._do_hover_pick)

    def _on_hover_leave(self, _event: tk.Event) -> None:
        if self._hover_after_id is not None:
            self.after_cancel(self._hover_after_id)
            self._hover_after_id = None
        self._hide_tooltip()

    def _do_hover_pick(self) -> None:
        self._hover_after_id = None
        if self._renderer is None or not self._hover_tooltip:
            return
        result = self._renderer.pick_point(
            float(self._hover_last_x), float(self._hover_last_y)
        )
        if result is None:
            self._hide_tooltip()
            return
        text = self._build_tooltip_text(result)
        screen_x = self.winfo_rootx() + self._hover_last_x + 16
        screen_y = self.winfo_rooty() + self._hover_last_y - 8
        self._show_tooltip(screen_x, screen_y, text)

    def _build_tooltip_text(self, result: dict) -> str:
        actor = result["actor"]
        index = result["index"]
        point = result["point"]   # [wx, wy, wz] world coords

        # Actor-specific hover data takes priority (add_points path).
        # Fall back to scene-level data (set_points path).
        hover_data = self._actor_hover.get(actor) or self._scene_hover
        columns = self._actor_columns.get(actor) or self._scene_columns

        x_label = columns.get("x", "x") if columns else "x"
        y_label = columns.get("y", "y") if columns else "y"
        z_label = columns.get("z", "z") if columns else "z"

        lines = [
            f"{x_label}: {point[0]:.4g}",
            f"{y_label}: {point[1]:.4g}",
            f"{z_label}: {point[2]:.4g}",
        ]

        hover_col_names = columns.get("hover", []) if columns else []
        for name in hover_col_names:
            arr = hover_data.get(name) if hover_data else None
            if arr is not None and index < len(arr):
                val = arr[index]
                if isinstance(val, (float, np.floating)):
                    lines.append(f"{name}: {val:.4g}")
                else:
                    lines.append(f"{name}: {val}")

        return "\n".join(lines)

    def _show_tooltip(self, screen_x: int, screen_y: int, text: str) -> None:
        font = ("Consolas", 9) if sys.platform == "win32" else ("Monospace", 9)
        if self._tooltip_win is None:
            win = tk.Toplevel(self)
            win.wm_overrideredirect(True)
            win.wm_attributes("-topmost", True)
            tk.Label(
                win, text=text, justify="left",
                background="#ffffcc", foreground="#111111",
                relief="solid", borderwidth=1, font=font,
                padx=5, pady=3,
            ).pack()
            win.wm_withdraw()
            self._tooltip_win = win
        else:
            self._tooltip_win.winfo_children()[0].configure(text=text)

        win = self._tooltip_win
        win.update_idletasks()
        tip_w = win.winfo_reqwidth()
        tip_h = win.winfo_reqheight()

        # Keep tooltip inside widget bounds
        wx = self.winfo_rootx()
        wy = self.winfo_rooty()
        ww = self.winfo_width()
        wh = self.winfo_height()

        x = screen_x
        y = screen_y - tip_h
        if x + tip_w > wx + ww:
            x = screen_x - tip_w - 16
        x = max(wx, x)
        if y < wy:
            y = screen_y + 16
        if y + tip_h > wy + wh:
            y = screen_y - tip_h - 8
        y = max(wy, y)

        win.wm_geometry(f"+{x}+{y}")
        win.wm_deiconify()

    def _hide_tooltip(self) -> None:
        if self._tooltip_win is not None:
            self._tooltip_win.wm_withdraw()

    # ── Mouse handling ────────────────────────────────────────────────────────

    def _engage_lod(self) -> None:
        if self._lod_enabled and self._total_n > self._lod_threshold and self._renderer is not None:
            self._renderer.set_lod_factor(self._lod_factor)

    def _disengage_lod(self) -> None:
        if self._renderer is not None:
            self._renderer.set_lod_factor(1)
            self._mark_dirty()

    def _drag_start(self, event: tk.Event, button: int) -> None:
        self._drag_btn = button
        self._drag_x = event.x
        self._drag_y = event.y
        self._press_x = event.x
        self._press_y = event.y
        self.focus_set()

        shift = bool(event.state & 0x0001)
        ctrl  = bool(event.state & 0x0004)

        # Ctrl+left → lasso (highest priority)
        if button == 1 and ctrl and self._lasso_enabled:
            self._lasso_active = True
            if self._renderer is not None:
                self._renderer.lasso_begin(float(event.x), float(event.y))
        # Shift+left → rectangle selection
        elif button == 1 and shift and self._pick_mode in ("rect", "both"):
            self._sel_x0 = event.x
            self._sel_y0 = event.y
            self._rect_active = True
        else:
            self._engage_lod()

    def _try_lasso_move(self, event: tk.Event) -> bool:
        """Extend the in-progress lasso path.  Returns True if lasso consumed the event."""
        if not self._lasso_active:
            return False
        # One float-pair per event; Rust manages the point list and updates the
        # overlay buffer incrementally (O(1) GPU writes per call).
        self._renderer.lasso_extend(float(event.x), float(event.y))
        self._mark_dirty()
        return True

    def _try_lasso_end(self, event: tk.Event) -> bool:
        """Finish the lasso, pick inside it, fire <<SelectionChanged>>.
        Returns True if lasso consumed the event."""
        if not self._lasso_active:
            return False
        self._lasso_active = False
        # lasso_end() picks the accumulated polygon, clears the overlay, and returns hits.
        hits = self._renderer.lasso_end()
        self._fire_selection(hits)
        self._mark_dirty()
        self._disengage_lod()
        self._drag_btn = None
        return True

    def _drag_move(self, event: tk.Event, button: int) -> None:
        if self._renderer is None or self._drag_btn != button:
            return

        if self._try_lasso_move(event):
            return

        # Rectangle selection: Shift+left-drag
        if button == 1 and self._rect_active:
            self._renderer.set_selection_rect(
                float(self._sel_x0), float(self._sel_y0),
                float(event.x), float(event.y),
            )
            self._mark_dirty()
            return

        dx = event.x - self._drag_x
        dy = event.y - self._drag_y
        self._drag_x = event.x
        self._drag_y = event.y
        # Shift+left-drag → pan (button 2); plain left-drag → orbit (button 1)
        shift = bool(event.state & 0x0001)
        effective = 2 if (button == 1 and shift) else button
        self._renderer.mouse_drag(float(dx), float(dy), effective)
        self._mark_dirty()
        self._propagate_camera()

    def _drag_end(self, event: tk.Event) -> None:
        if self._renderer is not None and self._try_lasso_end(event):
            return

        if self._rect_active:
            self._rect_active = False
            if self._renderer is not None:
                self._renderer.clear_selection_rect()
                x0, y0 = float(self._sel_x0), float(self._sel_y0)
                x1, y1 = float(event.x), float(event.y)
                if abs(x1 - x0) > 2 and abs(y1 - y0) > 2:
                    hits = self._renderer.pick_rectangle(x0, y0, x1, y1)
                    self._fire_selection(hits)
                self._mark_dirty()
            self._disengage_lod()
            self._drag_btn = None
            return

        # Point pick: left-click with minimal drag
        if (self._drag_btn == 1
                and self._pick_mode in ("point", "both")
                and self._renderer is not None):
            dx = abs(event.x - self._press_x)
            dy = abs(event.y - self._press_y)
            if dx <= self._pick_threshold and dy <= self._pick_threshold:
                result = self._renderer.pick_point(float(event.x), float(event.y))
                if result is not None:
                    self.picked_actor = result["actor"]
                    self.picked_index = result["index"]
                    self.picked_point = result["point"]
                else:
                    self.picked_actor = None
                    self.picked_index = None
                    self.picked_point = None
                self.event_generate("<<PointPicked>>")

        self._disengage_lod()
        self._drag_btn = None

    def _on_scroll(self, event: tk.Event) -> None:
        if self._renderer is None:
            return
        self._renderer.scroll(event.delta / 120.0)
        self._mark_dirty()
        self._propagate_camera()

    def _on_scroll_up_x11(self, _event: tk.Event) -> None:
        if self._renderer is not None:
            self._renderer.scroll(1.0)
            self._mark_dirty()
            self._propagate_camera()

    def _on_scroll_down_x11(self, _event: tk.Event) -> None:
        if self._renderer is not None:
            self._renderer.scroll(-1.0)
            self._mark_dirty()
            self._propagate_camera()


# ── 2D scatter widget ─────────────────────────────────────────────────────────

class Scatter2D(Scatter3D):
    """A Tkinter widget that renders a 2-D scatter plot using wgpu (Rust).

    Identical to :class:`Scatter3D` except:

    - the view is locked to top-down orthographic (parallel projection,
      looking along +Z onto the XY plane)
    - left-drag pans instead of orbiting
    - Z coordinates are always set to zero, regardless of input

    Usage
    -----
    ::

        import tkinter as tk
        import numpy as np
        from dragonsci import Scatter2D

        root = tk.Tk()
        w = Scatter2D(root, width=800, height=600)
        w.pack(fill="both", expand=True)

        pts = np.random.rand(100_000, 3).astype(np.float32)
        w.set_points(pts)   # z column is ignored and zeroed

        root.mainloop()

    Parameters
    ----------
    z : ignored
        Any ``z=`` column passed to :meth:`set_points` or
        :meth:`add_points` is silently replaced with zeros. Pass ``x``
        and ``y`` only for clarity.
    """

    def __init__(
        self,
        master: tk.Misc,
        width: int = 800,
        height: int = 600,
        fps: int = 60,
        vsync: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(master, width=width, height=height,
                         fps=fps, vsync=vsync, **kwargs)
        # Shadow flags: parent's _init_renderer reads these before the
        # renderer exists and applies them automatically.
        self._parallel_projection = True
        self._axis_labels = ("X", "Y", "")
        self._axis_visible = (True, True, False)  # hide Z axis entirely

    # ── Lifecycle override ────────────────────────────────────────────────

    def _init_renderer(self) -> None:
        super()._init_renderer()
        if self._renderer is None:
            return
        # Snap to front view (camera at +Z, looking at the XY plane) after
        # all pending data has been replayed so the auto-fit is correct first.
        self._renderer.view_xz()
        self._mark_dirty()

    # ── Data overrides ────────────────────────────────────────────────────

    def update_actor(
        self,
        handle: int,
        positions: "np.ndarray",
        **kwargs,
    ) -> None:
        """Flatten z to 0 then delegate to the parent update path."""
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        if pos.ndim == 2 and pos.shape[1] >= 3:
            pos = pos.copy()
            pos[:, 2] = 0.0
        super().update_actor(handle, pos, **kwargs)

    def _prepare_point_inputs(
        self,
        positions,
        *,
        x=None,
        y=None,
        z=None,
        colors: Optional["np.ndarray"] = None,
        scalars: Optional["np.ndarray"] = None,
        color=None,
        size=None,
        size_range: "tuple[float, float]" = (2.0, 20.0),
        point_sizes: Optional["np.ndarray"] = None,
        hover=None,
    ):
        """Always flatten z to 0 before passing to the renderer."""
        is_df = _is_supported_dataframe(positions)
        pos, clr, scl, sizes, meta = super()._prepare_point_inputs(
            positions,
            x=x, y=y,
            z=None if is_df else z,   # ignore z column for DataFrames
            colors=colors, scalars=scalars,
            color=color, size=size, size_range=size_range,
            point_sizes=point_sizes, hover=hover,
        )
        if not is_df:
            pos = pos.copy()
            pos[:, 2] = 0.0
        return pos, clr, scl, sizes, meta

    def set_points(self, positions, **kwargs) -> None:
        """Load a point cloud and snap the camera to the 2D front view."""
        super().set_points(positions, **kwargs)
        if self._renderer is not None:
            self._renderer.view_xz()
            self._mark_dirty()

    # ── Camera overrides ──────────────────────────────────────────────────

    def reset_camera(self) -> None:
        """Fit the camera to the data and restore the 2D front view."""
        super().reset_camera()
        if self._renderer is not None:
            self._renderer.view_xz()
            self._mark_dirty()
            self._propagate_camera()

    def view_xy(self) -> None:
        """No-op in 2D mode — camera is locked to the XY plane (front view)."""

    def view_xz(self) -> None:
        """No-op in 2D mode — camera is locked to the XY plane (front view)."""

    def view_yz(self) -> None:
        """No-op in 2D mode — camera is locked to the XY plane (front view)."""

    def view_isometric(self) -> None:
        """No-op in 2D mode — camera is locked to the XY plane (front view)."""

    def set_camera(self, state: dict) -> None:
        """Apply a camera state then re-lock to top-down orthographic.

        Overrides the parent fully so that only one propagation fires —
        after the relock — ensuring linked peers always receive the
        corrected 2D state, never the intermediate perspective state.
        """
        if self._renderer is None:
            return
        self._renderer.set_camera(state)
        self._renderer.view_xz()
        self._renderer.set_parallel_projection(True)
        self._mark_dirty()
        self._propagate_camera()

    def _receive_camera(self, state: dict) -> None:
        """Apply a linked camera state then re-lock to front orthographic.

        Keeps ``_propagating = True`` for the entire relock so that the
        orientation and projection corrections never propagate back to peers.
        """
        if self._renderer is None:
            return
        self._propagating = True
        try:
            self._renderer.set_camera(state)
            self._renderer.view_xz()
            self._renderer.set_parallel_projection(True)
            self._mark_dirty()
        finally:
            self._propagating = False

    @property
    def parallel_projection(self) -> bool:
        """Always ``True`` — 2D mode is always orthographic."""
        return True

    @parallel_projection.setter
    def parallel_projection(self, on: bool) -> None:
        pass  # no-op: cannot disable orthographic in 2D mode

    # ── Mouse override ────────────────────────────────────────────────────

    def _drag_move(self, event: tk.Event, button: int) -> None:
        """Override: left-drag always pans; orbiting is disabled."""
        if self._renderer is None or self._drag_btn != button:
            return

        if self._try_lasso_move(event):
            return

        # Shift+left rectangle selection still works normally.
        if button == 1 and self._rect_active:
            self._renderer.set_selection_rect(
                float(self._sel_x0), float(self._sel_y0),
                float(event.x), float(event.y),
            )
            self._mark_dirty()
            return

        dx = event.x - self._drag_x
        dy = event.y - self._drag_y
        self._drag_x = event.x
        self._drag_y = event.y
        # Always pan (2) regardless of button or Shift state.
        self._renderer.mouse_drag(float(dx), float(dy), 2)
        self._mark_dirty()
        self._propagate_camera()


# ── Linked-camera module API ──────────────────────────────────────────────────

def link_cameras(*widgets: Scatter3D) -> None:
    """Synchronise the camera across two or more ``Scatter3D`` instances.

    After linking, orbiting, panning, zooming, or applying any camera preset on
    any widget immediately mirrors the view on all others.  Widgets can be in
    different Tk windows.

    Call ``unlink_cameras(*widgets)`` to break the link.

    Parameters
    ----------
    *widgets
        Two or more ``Scatter3D`` instances to link together.
    """
    if len(widgets) < 2:
        return
    for i, w in enumerate(widgets):
        for j, other in enumerate(widgets):
            if i != j:
                w._camera_links.add(other)


def unlink_cameras(*widgets: Scatter3D) -> None:
    """Remove camera synchronisation between the given widgets.

    Removes all cross-links *among* the supplied widgets.  Links to widgets
    not listed are left intact.

    Parameters
    ----------
    *widgets
        The widgets to unlink from each other.
    """
    widget_set = set(widgets)
    for w in widgets:
        w._camera_links -= widget_set
