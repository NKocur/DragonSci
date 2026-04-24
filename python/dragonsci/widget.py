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
_LEGEND_POSITION_IDX: dict[str, int] = {
    "top-right":    0,
    "top-left":     1,
    "bottom-right": 2,
    "bottom-left":  3,
}

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


def _xy_to_segments(x: "np.ndarray", y: "np.ndarray") -> "np.ndarray":
    """Convert parallel (N,) x and y arrays into an (N-1, 6) segment array (z=0).

    Each output row is ``[x0, y0, 0, x1, y1, 0]`` — the format expected by
    :meth:`Scatter3D.add_lines`.  Returns a single degenerate zero-segment when
    ``len(x) < 2`` so the GPU buffer is never empty.
    """
    n = len(x)
    if n < 2:
        return np.zeros((1, 6), dtype=np.float32)
    segs = np.zeros((n - 1, 6), dtype=np.float32)
    segs[:, 0] = x[:-1]
    segs[:, 1] = y[:-1]
    segs[:, 3] = x[1:]
    segs[:, 4] = y[1:]
    return segs


def _nice_bounds_1d(lo: float, hi: float) -> "tuple[float, float]":
    """Compute 'nice' rounded axis bounds — mirrors the Rust grid::nice_bounds logic.

    Targets ~5 ticks; step rounds to 1/2/5×10^n.  When the range is
    degenerate (< 1e-10) a ±0.5 fallback is returned.
    """
    import math
    rng = abs(hi - lo)
    if rng < 1e-10:
        return lo - 0.5, hi + 0.5
    rough = rng / 5.0
    mag = 10.0 ** math.floor(math.log10(rough))
    norm = rough / mag
    if norm <= 1.0:
        step = 1.0
    elif norm <= 2.0:
        step = 2.0
    elif norm <= 5.0:
        step = 5.0
    else:
        step = 10.0
    step *= mag
    return math.floor(lo / step) * step, math.ceil(hi / step) * step


def _normalize_line2d_width(line_width: float) -> float:
    try:
        value = float(line_width)
    except Exception as exc:
        raise ValueError("line width must be a number") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("line width must be a positive finite number")
    return value


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
    """Vectorized categorical detection + encoding.

    Returns ``(rgb_array, legend_items)`` when ``values`` looks categorical,
    ``None`` otherwise.  Uses numpy factorization so the Python loop runs
    over unique values only (bounded by ``_CATEGORICAL_THRESHOLD``), not
    over every row.
    """
    arr = np.asarray(values).reshape(-1)
    dtype = arr.dtype

    # Non-numeric dtypes are always categorical — skip the threshold check.
    always_categorical = np.issubdtype(dtype, np.bool_) or dtype.kind in ("U", "S", "O")
    if not always_categorical and not np.issubdtype(dtype, np.integer):
        return None

    # np.unique gives sorted unique values + inverse indices for free.
    # For very large integer arrays that are unlikely to be categorical, do a
    # cheap cardinality probe on a prefix before paying the full sort cost.
    if not always_categorical and arr.shape[0] > 1000:
        probe = np.unique(arr[:1000])
        if len(probe) > _CATEGORICAL_THRESHOLD:
            return None

    unique_vals, inverse = np.unique(arr, return_inverse=True)
    if not always_categorical and len(unique_vals) > _CATEGORICAL_THRESHOLD:
        return None

    # Build legend items for unique values only (max _CATEGORICAL_THRESHOLD iterations).
    legend_items: list[tuple[str, tuple[float, float, float]]] = []
    for raw in unique_vals:
        _, label = _normalize_category_value(raw)
        legend_items.append((label, _categorical_palette_color(len(legend_items))))

    # Vectorized color assignment via numpy advanced indexing.
    palette = np.array([c for _, c in legend_items], dtype=np.float32)
    colors = palette[inverse]

    return colors, legend_items


def _factorize_labels(
    lbl_arr: "np.ndarray",
) -> "list[object]":
    """Return unique label values in the same order as ``_try_encode_categorical``.

    Mirrors ``np.unique`` (sorted) so overlay palette slots match point colors.
    Falls back to first-seen order only when labels are not sortable (e.g. mixed
    int/str), which is the same situation where ``np.unique`` would also fail.
    """
    try:
        unique_vals = np.unique(lbl_arr).tolist()
    except TypeError:
        seen: "dict[object, None]" = {}
        for v in lbl_arr.tolist():
            seen[v] = None
        unique_vals = list(seen)
    return unique_vals


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

_LABEL_ANCHOR_MAP: "dict[str, int]" = {
    "center": 0, "left": 1, "right": 2, "top": 3, "bottom": 4,
}


def _parse_label_position(position) -> "tuple[float, float, float]":
    import numpy as _np
    if isinstance(position, _np.ndarray):
        position = position.flatten()
    x, y, z = float(position[0]), float(position[1]), float(position[2])
    return (x, y, z)


def _parse_label_color(color) -> "list[float]":
    if isinstance(color, str):
        color = color.lstrip("#")
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        return [r / 255.0, g / 255.0, b / 255.0, 1.0]
    if len(color) == 3:
        return [float(color[0]), float(color[1]), float(color[2]), 1.0]
    return [float(color[0]), float(color[1]), float(color[2]), float(color[3])]


def _compute_convex_hull(
    points: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Return (vertices float32 Nx3, triangle_indices uint32 Mx3) for the convex hull of *points*."""
    try:
        from scipy.spatial import ConvexHull
    except ImportError as exc:
        raise ImportError(
            "scipy is required for convex hull overlays. "
            "Install it with: pip install dragonsci[stats]"
        ) from exc
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise ValueError(
            f"convex hull requires at least 4 points with shape (N, 3); "
            f"got shape {pts.shape}"
        )
    try:
        hull = ConvexHull(pts)
    except Exception as exc:
        raise ValueError(f"convex hull computation failed: {exc}") from exc
    verts = pts[hull.vertices].astype(np.float32)
    # hull.simplices index into hull.vertices, not pts directly — remap
    old_to_new = {old: new for new, old in enumerate(hull.vertices)}
    idxs = np.array(
        [[old_to_new[i] for i in tri] for tri in hull.simplices], dtype=np.uint32
    )
    return verts, idxs


def _compute_ellipsoid(
    center: "np.ndarray",
    covariance: "np.ndarray",
    n_std: float = 2.0,
    u_res: int = 20,
    v_res: int = 10,
) -> "tuple[np.ndarray, np.ndarray]":
    """Return (vertices float32 Nx3, triangle_indices uint32 Mx3) for an ellipsoid.

    The ellipsoid is derived by eigendecomposing *covariance* and scaling a UV
    sphere by ``n_std * sqrt(eigenvalues)`` along the principal axes.
    """
    vals, vecs = np.linalg.eigh(np.asarray(covariance, dtype=np.float64))
    vals = np.maximum(vals, 0.0)  # numerical safety
    radii = n_std * np.sqrt(vals)

    # UV sphere in [-pi,pi] x [-pi/2,pi/2]
    u = np.linspace(0, 2 * np.pi, u_res + 1)
    v = np.linspace(-np.pi / 2, np.pi / 2, v_res + 1)
    uu, vv = np.meshgrid(u, v)
    sx = np.cos(vv) * np.cos(uu)
    sy = np.cos(vv) * np.sin(uu)
    sz = np.sin(vv)
    sphere = np.stack([sx.ravel(), sy.ravel(), sz.ravel()], axis=1)  # Nx3

    # Transform: center + vecs @ diag(radii) @ sphere.T → Nx3
    pts = (vecs * radii) @ sphere.T  # 3×N
    pts = pts.T + np.asarray(center, dtype=np.float64)
    verts = pts.astype(np.float32)

    # Build triangle indices for the UV grid
    nv1 = v_res + 1
    nu1 = u_res + 1
    tris = []
    for row in range(v_res):
        for col in range(u_res):
            a = row * nu1 + col
            b = a + 1
            c = a + nu1
            d = c + 1
            tris.append([a, c, b])
            tris.append([b, c, d])
    idxs = np.array(tris, dtype=np.uint32)
    return verts, idxs


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

        # User label pre-map state (virtual handle system mirrors actor handles)
        self._next_lhandle: int = 0
        self._lhandle_map: "dict[int, int]" = {}   # virtual → real (u64) Rust handle
        self._label_handles: "set[int]" = set()    # live virtual handles
        self._pending_labels: "list[tuple[int, str, dict]]" = []

        # Mesh overlay pre-map state (convex hulls, ellipsoids)
        self._next_mhandle: int = 0
        self._mhandle_map: "dict[int, int]" = {}   # virtual → real (u64) Rust handle
        self._mesh_handles: "set[int]" = set()
        self._pending_meshes: "list[tuple[int, str, dict]]" = []
        self._mesh_meta: "dict[int, dict]" = {}    # handle → {"color", "wireframe", ...}
        # Python-side shadow so parallel_projection is readable before renderer init
        self._parallel_projection: bool = False

        # Dirty-frame model: only call render() when something changed
        self._dirty: bool = False
        self._render_fail_count: int = 0  # consecutive render() failures

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

        # Sub-classes can point this at a child widget to redirect the renderer
        # HWND, winfo_width/height queries, and configure-driven resizes to a
        # different surface than the outer frame.  Defaults to self.
        self._render_target_widget: tk.Misc = self

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
        self._render_target_widget.update_idletasks()
        self._init_renderer()

    def _init_renderer(self) -> None:
        tgt = self._render_target_widget
        w = max(tgt.winfo_width(), 1)
        h = max(tgt.winfo_height(), 1)
        try:
            self._renderer = ScatterRenderer(
                tgt.winfo_id(),
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
        if self._pending_labels:
            for vhandle, action, payload in self._pending_labels:
                if action == "add":
                    real = int(self._renderer.add_user_label(**payload))
                    self._lhandle_map[vhandle] = real
                    self._label_handles.add(vhandle)
                elif action in ("update", "remove", "visibility", "clear"):
                    real = self._lhandle_map.get(vhandle)
                    if action == "update" and real is not None:
                        self._renderer.update_user_label(real, **payload)
                    elif action == "remove" and real is not None:
                        self._renderer.remove_user_label(real)
                        self._label_handles.discard(vhandle)
                    elif action == "visibility" and real is not None:
                        self._renderer.set_user_label_visible(real, payload["visible"])
                    elif action == "clear":
                        self._renderer.clear_user_labels()
                        self._lhandle_map.clear()
                        self._label_handles.clear()
            self._pending_labels.clear()
            self._mark_dirty()
            did_something = True
        if self._pending_meshes:
            for vhandle, action, payload in self._pending_meshes:
                if action == "add":
                    real = int(self._renderer.add_mesh(**payload))
                    self._mhandle_map[vhandle] = real
                    self._mesh_handles.add(vhandle)
                elif action == "update":
                    real = self._mhandle_map.get(vhandle)
                    if real is not None:
                        self._renderer.update_mesh(real, **payload)
                elif action == "style":
                    real = self._mhandle_map.get(vhandle)
                    if real is not None:
                        self._renderer.update_mesh_style(
                            real, payload["color"], payload["wireframe"])
                elif action == "remove":
                    real = self._mhandle_map.pop(vhandle, None)
                    if real is not None:
                        self._renderer.remove_mesh(real)
                        self._mesh_handles.discard(vhandle)
                        self._mesh_meta.pop(vhandle, None)
                elif action == "visibility":
                    real = self._mhandle_map.get(vhandle)
                    if real is not None:
                        self._renderer.set_mesh_visibility(real, payload["visible"])
                elif action == "clear":
                    self._renderer.clear_meshes()
                    self._mhandle_map.clear()
                    self._mesh_handles.clear()
                    self._mesh_meta.clear()
            self._pending_meshes.clear()
            self._mark_dirty()
            did_something = True
        if not did_something:
            self._schedule_render()

    def _on_configure(self, event: tk.Event) -> None:
        if self._renderer is None:
            return
        # Debounce: resize is expensive — only execute 50 ms after the last event.
        # Always query the render-target widget directly so sub-class overrides
        # (e.g. Scatter2D with a _render_frame sub-frame) get the correct size.
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        tgt = self._render_target_widget
        self._resize_after_id = self.after(
            50, lambda: self._do_resize(
                max(tgt.winfo_width(), 1),
                max(tgt.winfo_height(), 1),
            )
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
        try:
            self.close_gif()  # flush and clean up any in-progress recording
        except Exception:
            pass
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

    _RENDER_FAIL_LIMIT = 5  # consecutive failures before giving up

    def _render_tick(self) -> None:
        self._after_id = None  # cleared first so _mark_dirty can re-arm
        if self._renderer is not None and self._dirty:
            try:
                self._renderer.render()
                self._dirty = False
                self._render_fail_count = 0
            except Exception as exc:
                self._render_fail_count += 1
                if self._render_fail_count >= self._RENDER_FAIL_LIMIT:
                    import warnings
                    warnings.warn(
                        f"DragonSci: render() failed {self._render_fail_count} times in a row "
                        f"({type(exc).__name__}: {exc}); stopping render loop.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._dirty = False  # stop retrying
                    self._render_fail_count = 0
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
        self._pending_labels.clear()
        self._label_handles.clear()
        self._lhandle_map.clear()
        self._pending_meshes.clear()
        self._mesh_handles.clear()
        self._mhandle_map.clear()
        self._mesh_meta.clear()
        if self._renderer is not None:
            self._renderer.clear_actors()
            self._renderer.clear_overlays()
            self._renderer.clear_user_labels()
            self._renderer.clear_meshes()
            self._mark_dirty()

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
        """Pin a text label at a 3-D world-space position.

        Returns a handle that can be passed to :meth:`update_label`,
        :meth:`remove_label`, and :meth:`set_label_visibility`.
        """
        pos3 = _parse_label_position(position)
        rgba = _parse_label_color(color)
        anch = _LABEL_ANCHOR_MAP.get(anchor.lower(), 0)
        payload = dict(x=pos3[0], y=pos3[1], z=pos3[2], text=text,
                       color=rgba, size=float(size), anchor=anch)
        if self._renderer is None:
            vhandle = self._next_lhandle
            self._next_lhandle += 1
            self._pending_labels.append((vhandle, "add", payload))
            self._label_handles.add(vhandle)
            return vhandle
        real = int(self._renderer.add_user_label(**payload))
        self._label_handles.add(real)
        self._mark_dirty()
        return real

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
        """Update one or more fields of an existing label in-place."""
        pos3 = _parse_label_position(position) if position is not None else None
        rgba = _parse_label_color(color) if color is not None else None
        anch = _LABEL_ANCHOR_MAP.get(anchor.lower(), 0) if anchor is not None else None
        payload = dict(pos=pos3, text=text, color=rgba, size=size, anchor=anch)
        if self._renderer is None:
            self._pending_labels.append((handle, "update", payload))
            return
        real = self._lhandle_map.get(handle, handle)
        self._renderer.update_user_label(real, **payload)
        self._mark_dirty()

    def remove_label(self, handle: int) -> None:
        """Remove a label by handle."""
        if self._renderer is None:
            self._pending_labels.append((handle, "remove", {}))
            self._label_handles.discard(handle)
            return
        real = self._lhandle_map.get(handle, handle)
        self._renderer.remove_user_label(real)
        self._label_handles.discard(handle)
        self._mark_dirty()

    def set_label_visibility(self, handle: int, visible: bool) -> None:
        """Show or hide a label without removing it."""
        if self._renderer is None:
            self._pending_labels.append((handle, "visibility", {"visible": visible}))
            return
        real = self._lhandle_map.get(handle, handle)
        self._renderer.set_user_label_visible(real, visible)
        self._mark_dirty()

    def clear_labels(self) -> None:
        """Remove all user-defined labels."""
        self._pending_labels.clear()
        self._label_handles.clear()
        if self._renderer is not None:
            self._renderer.clear_user_labels()
            self._mark_dirty()
        else:
            # Pre-map: single sentinel so _init_renderer replays the clear.
            self._pending_labels.append((-1, "clear", {}))

    # ── Mesh overlays (convex hulls, ellipsoids) ──────────────────────────────

    def add_convex_hull(
        self,
        points: "np.ndarray",
        *,
        color: "tuple[float, float, float] | str" = (1.0, 1.0, 0.0),
        opacity: float = 0.3,
        wireframe: bool = False,
    ) -> int:
        """Add a convex hull overlay around *points*.

        Requires ``scipy``:  ``pip install dragonsci[stats]``.

        Parameters
        ----------
        points : (N, 3) float32 array  — N ≥ 4.
        color  : RGB tuple or ``"#RRGGBB"`` hex string.
        opacity: alpha in [0, 1].
        wireframe : draw edges only instead of filled faces.

        Returns
        -------
        int — handle for update / remove.
        """
        verts, idxs = _compute_convex_hull(points)
        rgba = list(_parse_label_color(color))[:3] + [float(opacity)]
        return self._add_mesh_actor(verts, idxs, rgba, wireframe,
                                    meta={"color": rgba, "wireframe": wireframe})

    def update_convex_hull(
        self,
        handle: int,
        points: "np.ndarray | None" = None,
        *,
        color: "tuple[float, float, float] | str | None" = None,
        opacity: "float | None" = None,
        wireframe: "bool | None" = None,
    ) -> None:
        """Update an existing convex hull's geometry and/or style.

        All parameters are optional.  Omitting *points* performs a style-only
        update without recomputing the geometry.
        """
        meta = self._mesh_meta.get(handle, {})
        cur_color = meta.get("color", [1.0, 1.0, 0.0, 0.3])
        if color is not None:
            rgb = list(_parse_label_color(color))[:3]
            alpha = float(opacity) if opacity is not None else cur_color[3]
            rgba = rgb + [alpha]
        elif opacity is not None:
            rgba = cur_color[:3] + [float(opacity)]
        else:
            rgba = list(cur_color)
        wf = wireframe if wireframe is not None else meta.get("wireframe", False)

        if points is not None:
            verts, idxs = _compute_convex_hull(points)
            self._update_mesh_actor(handle, verts, idxs, rgba, wf)
        else:
            # Style-only: reuse existing geometry, just push new color/wireframe.
            self._update_mesh_actor(handle, None, None, rgba, wf)
        if handle in self._mesh_meta:
            self._mesh_meta[handle].update({"color": rgba, "wireframe": wf})

    def add_ellipsoid(
        self,
        center: "np.ndarray | tuple",
        covariance: "np.ndarray",
        *,
        color: "tuple[float, float, float] | str" = (1.0, 0.2, 0.2),
        opacity: float = 0.3,
        n_std: float = 2.0,
        wireframe: bool = False,
    ) -> int:
        """Add an ellipsoid overlay defined by *center* and *covariance*.

        Parameters
        ----------
        center     : (3,) world-space centroid.
        covariance : (3, 3) covariance matrix; axes from eigendecomposition.
        n_std      : number of standard deviations (default 2 ≈ 95 % of a Gaussian).
        color, opacity, wireframe : same as :meth:`add_convex_hull`.
        """
        verts, idxs = _compute_ellipsoid(center, covariance, n_std)
        rgba = list(_parse_label_color(color))[:3] + [float(opacity)]
        return self._add_mesh_actor(verts, idxs, rgba, wireframe,
                                    meta={"color": rgba, "wireframe": wireframe,
                                          "n_std": float(n_std),
                                          "_center": np.asarray(center, dtype=np.float64).copy(),
                                          "_covariance": np.asarray(covariance, dtype=np.float64).copy()})

    def update_ellipsoid(
        self,
        handle: int,
        center: "np.ndarray | tuple | None" = None,
        covariance: "np.ndarray | None" = None,
        *,
        color: "tuple[float, float, float] | str | None" = None,
        opacity: "float | None" = None,
        n_std: "float | None" = None,
        wireframe: "bool | None" = None,
    ) -> None:
        """Update an existing ellipsoid's geometry and/or style.

        All parameters are optional.  Omitting *center* and *covariance*
        performs a style-only update without recomputing the geometry.
        If either *center* or *covariance* is supplied, both must be supplied.
        """
        if (center is None) != (covariance is None):
            raise ValueError("supply both center and covariance, or neither")
        meta = self._mesh_meta.get(handle, {})
        cur_color = meta.get("color", [1.0, 0.2, 0.2, 0.3])
        if color is not None:
            rgb = list(_parse_label_color(color))[:3]
            alpha = float(opacity) if opacity is not None else cur_color[3]
            rgba = rgb + [alpha]
        elif opacity is not None:
            rgba = cur_color[:3] + [float(opacity)]
        else:
            rgba = list(cur_color)
        wf = wireframe if wireframe is not None else meta.get("wireframe", False)
        std = float(n_std) if n_std is not None else meta.get("n_std", 2.0)

        if center is not None:
            # Full geometry update with new center + covariance.
            verts, idxs = _compute_ellipsoid(center, covariance, std)
            self._update_mesh_actor(handle, verts, idxs, rgba, wf)
            if handle in self._mesh_meta:
                self._mesh_meta[handle]["_center"]     = np.asarray(center,     dtype=np.float64).copy()
                self._mesh_meta[handle]["_covariance"] = np.asarray(covariance, dtype=np.float64).copy()
        elif n_std is not None:
            # n_std changed — recompute geometry using cached center + covariance.
            c_cached   = meta.get("_center")
            cov_cached = meta.get("_covariance")
            if c_cached is not None and cov_cached is not None:
                verts, idxs = _compute_ellipsoid(c_cached, cov_cached, std)
                self._update_mesh_actor(handle, verts, idxs, rgba, wf)
            else:
                self._update_mesh_actor(handle, None, None, rgba, wf)
        else:
            # Style-only (color / opacity / wireframe).
            self._update_mesh_actor(handle, None, None, rgba, wf)
        if handle in self._mesh_meta:
            self._mesh_meta[handle].update({"color": rgba, "wireframe": wf, "n_std": std})

    def set_mesh_visibility(self, handle: int, visible: bool) -> None:
        """Show or hide a convex hull or ellipsoid by its handle."""
        if self._renderer is not None:
            real = self._mhandle_map.get(handle)
            if real is not None:
                self._renderer.set_mesh_visibility(real, visible)
                self._mark_dirty()
        else:
            self._pending_meshes.append((handle, "visibility", {"visible": visible}))

    def remove_mesh(self, handle: int) -> None:
        """Remove any mesh overlay (hull or ellipsoid) by its handle."""
        if self._renderer is not None:
            real = self._mhandle_map.pop(handle, None)
            if real is not None:
                self._renderer.remove_mesh(real)
                self._mesh_handles.discard(handle)
                self._mesh_meta.pop(handle, None)
                self._mark_dirty()
        else:
            self._pending_meshes.append((handle, "remove", {}))
            self._mesh_handles.discard(handle)
            self._mesh_meta.pop(handle, None)

    # Convenience aliases kept for backward-compat with plan API
    remove_convex_hull = remove_mesh
    remove_ellipsoid   = remove_mesh

    def clear_meshes(self) -> None:
        """Remove all mesh overlays (hulls and ellipsoids)."""
        self._pending_meshes.clear()
        self._mesh_handles.clear()
        self._mhandle_map.clear()
        self._mesh_meta.clear()
        if self._renderer is not None:
            self._renderer.clear_meshes()
            self._mark_dirty()
        else:
            self._pending_meshes.append((-1, "clear", {}))

    def add_cluster_hulls(
        self,
        positions: "np.ndarray",
        labels: "list | np.ndarray",
        *,
        colormap: str = "tab10",
        opacity: float = 0.25,
    ) -> "list[int]":
        """Add one convex hull per unique label value.

        Parameters
        ----------
        positions : (N, 3) float32 array.
        labels    : length-N sequence of category values (ints, strings, …).
        colormap  : ignored in v1; uses the built-in categorical palette.
        opacity   : alpha for all hulls.

        Returns
        -------
        list[int] — one handle per unique label (skips groups with < 4 points).
        """
        pts = np.asarray(positions, dtype=np.float32)
        lbl_arr = np.asarray(labels)
        unique = _factorize_labels(lbl_arr)
        handles = []
        for i, lbl in enumerate(unique):
            mask = lbl_arr == lbl
            grp = pts[mask]
            if len(grp) < 4:
                continue
            color = _CATEGORICAL_PALETTE[i % len(_CATEGORICAL_PALETTE)]
            handles.append(self.add_convex_hull(grp, color=color, opacity=opacity))
        return handles

    def add_cluster_ellipsoids(
        self,
        positions: "np.ndarray",
        labels: "list | np.ndarray",
        *,
        colormap: str = "tab10",
        opacity: float = 0.25,
        n_std: float = 2.0,
    ) -> "list[int]":
        """Add one ellipsoid per unique label value.

        Parameters
        ----------
        positions, labels, colormap, opacity : same as :meth:`add_cluster_hulls`.
        n_std : number of standard deviations for the ellipsoid axes.

        Returns
        -------
        list[int] — one handle per unique label (skips groups with < 4 points).
        """
        pts = np.asarray(positions, dtype=np.float64)
        lbl_arr = np.asarray(labels)
        unique = _factorize_labels(lbl_arr)
        handles = []
        for i, lbl in enumerate(unique):
            mask = lbl_arr == lbl
            grp = pts[mask]
            if len(grp) < 4:
                continue
            center = grp.mean(axis=0)
            cov = np.cov(grp.T)
            if cov.ndim == 0:
                cov = np.diag([float(cov)] * 3)
            color = _CATEGORICAL_PALETTE[i % len(_CATEGORICAL_PALETTE)]
            handles.append(self.add_ellipsoid(center, cov, color=color,
                                              opacity=opacity, n_std=n_std))
        return handles

    # ── Internal mesh helpers ─────────────────────────────────────────────────

    def _add_mesh_actor(self, verts, idxs, rgba, wireframe, *, meta) -> int:
        vhandle = self._next_mhandle
        self._next_mhandle += 1
        self._mesh_handles.add(vhandle)
        meta["_verts"] = np.ascontiguousarray(verts, dtype=np.float32)
        meta["_idxs"]  = np.ascontiguousarray(idxs,  dtype=np.uint32)
        self._mesh_meta[vhandle] = meta
        payload = {
            "vertices": np.ascontiguousarray(verts, dtype=np.float32),
            "indices":  np.ascontiguousarray(idxs,  dtype=np.uint32),
            "color":    list(rgba),
            "wireframe": wireframe,
        }
        if self._renderer is not None:
            real = int(self._renderer.add_mesh(**payload))
            self._mhandle_map[vhandle] = real
            self._mark_dirty()
        else:
            self._pending_meshes.append((vhandle, "add", payload))
        return vhandle

    def _update_mesh_actor(self, handle, verts, idxs, rgba, wireframe) -> None:
        style_only = verts is None or idxs is None
        if style_only:
            # Style-only (color / opacity / wireframe toggle): Rust uses its stored
            # positions/triangle-indices to re-bake the buffers without Python
            # retransferring geometry arrays.
            if self._renderer is not None:
                real = self._mhandle_map.get(handle)
                if real is not None:
                    self._renderer.update_mesh_style(real, list(rgba), bool(wireframe))
                    self._mark_dirty()
            else:
                self._pending_meshes.append(
                    (handle, "style", {"color": list(rgba), "wireframe": bool(wireframe)}))
        else:
            # Full geometry update: persist geometry in meta for Rust-side storage.
            if handle in self._mesh_meta:
                self._mesh_meta[handle]["_verts"] = np.ascontiguousarray(verts, dtype=np.float32)
                self._mesh_meta[handle]["_idxs"]  = np.ascontiguousarray(idxs,  dtype=np.uint32)
            payload = {
                "vertices": np.ascontiguousarray(verts, dtype=np.float32),
                "indices":  np.ascontiguousarray(idxs,  dtype=np.uint32),
                "color":    list(rgba),
                "wireframe": wireframe,
            }
            if self._renderer is not None:
                real = self._mhandle_map.get(handle)
                if real is not None:
                    self._renderer.update_mesh(real, **payload)
                    self._mark_dirty()
            else:
                self._pending_meshes.append((handle, "update", payload))

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

    def _translate_hits(self, actor_ids, point_indices) -> "tuple[list[int], list | None]":
        """Translate raw (actor_ids, point_indices) hit arrays to (row_positions, index_labels).

        ``row_positions`` are plotted row positions suitable for ``df.iloc[...]``.
        For numpy-only actors the raw per-actor buffer index is returned.
        ``index_labels`` is populated when at least one actor was DataFrame-backed
        and had a non-trivial pandas index; otherwise ``None``.
        """
        indices: list = []
        labels: list = []
        any_labels = False
        for actor, idx in zip(actor_ids, point_indices):
            idx = int(idx)
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

    def _fire_selection(self, actor_ids, point_indices) -> None:
        """Store translated selection results and fire ``<<SelectionChanged>>``."""
        # Reconstruct self.selected as list-of-dicts for backward compatibility.
        self.selected = [{"actor": int(a), "index": int(i)}
                         for a, i in zip(actor_ids, point_indices)]
        self.selected_indices, self.selected_index_values = self._translate_hits(actor_ids, point_indices)
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
        # lasso_end() returns (actor_ids, point_indices) arrays.
        hits = self._renderer.lasso_end()
        self._fire_selection(*hits)
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
                    self._fire_selection(*hits)
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

        # ── Marginal histogram state ──────────────────────────────────────
        self._marginal_coords: "dict[int, tuple[np.ndarray, np.ndarray]]" = {}
        self._hidden_actors: "set[int]" = set()
        self._marginals_visible: bool = False
        self._marginals_bins: "int | str" = 50
        self._marginals_color: str = "#4c8eff"
        self._marginals_alpha: float = 0.7
        self._marginals_size: int = 80
        self._marginals_orientation: str = "both"
        self._x_hist_canvas: "tk.Canvas | None" = None
        self._y_hist_canvas: "tk.Canvas | None" = None
        self._last_bounds_hash: "tuple | None" = None
        self._marginal_stream_cap: int = 50_000
        self._last_prep_pos: "np.ndarray | None" = None  # set by _prepare_point_inputs

        # ── Sub-frame layout ──────────────────────────────────────────────
        # Renderer draws into _render_frame (row=1, col=0).  Marginal canvases
        # occupy row=0 (X hist) and col=1 (Y hist) when visible, shrinking the
        # scatter area rather than overlaying it.
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)   # Y hist column — sized by minsize
        self.rowconfigure(0, weight=0)       # X hist row  — sized by minsize
        self.rowconfigure(1, weight=1)       # scatter row

        self._render_frame = tk.Frame(self, bg="black")
        self._render_frame.grid(row=1, column=0, sticky="nsew")

        # Redirect renderer to the inner frame.
        self._render_target_widget = self._render_frame

        # Bind render-frame Configure so the renderer resizes when the panel
        # area changes (e.g. marginals shown/hidden or outer widget resized).
        self._render_frame.bind("<Configure>", self._on_configure, add="+")

        # Rebind input events from outer frame to the inner render frame so
        # event.x/y coords are in renderer-surface space (origin = top-left of
        # _render_frame, matching the wgpu surface origin).
        _INPUT_EVENTS = (
            "<ButtonPress-1>", "<ButtonPress-2>",
            "<B1-Motion>", "<B2-Motion>",
            "<ButtonRelease-1>", "<ButtonRelease-2>",
            "<MouseWheel>", "<Button-4>", "<Button-5>",
            "<Double-Button-1>", "<Motion>", "<Leave>",
        )
        for ev in _INPUT_EVENTS:
            self.unbind(ev)
        self._render_frame.bind("<ButtonPress-1>",   lambda e: self._drag_start(e, 1))
        self._render_frame.bind("<ButtonPress-2>",   lambda e: self._drag_start(e, 2))
        self._render_frame.bind("<B1-Motion>",       lambda e: self._drag_move(e, 1))
        self._render_frame.bind("<B2-Motion>",       lambda e: self._drag_move(e, 2))
        self._render_frame.bind("<ButtonRelease-1>", self._drag_end)
        self._render_frame.bind("<ButtonRelease-2>", self._drag_end)
        self._render_frame.bind("<MouseWheel>",      self._on_scroll)
        self._render_frame.bind("<Button-4>",        self._on_scroll_up_x11)
        self._render_frame.bind("<Button-5>",        self._on_scroll_down_x11)
        self._render_frame.bind("<Double-Button-1>", lambda _e: self.reset_camera())
        self._render_frame.bind("<Motion>",          self._on_hover_motion, add="+")
        self._render_frame.bind("<Leave>",           self._on_hover_leave, add="+")

    # ── Lifecycle override ────────────────────────────────────────────────

    def _init_renderer(self) -> None:
        # Save pre-map add_points state BEFORE parent replays it directly via
        # self._renderer.add_points() (bypassing our add_points hook).
        saved_actors = list(self._pending_actors)
        saved_vis    = dict(self._pending_actor_visibility)

        super()._init_renderer()
        if self._renderer is None:
            return

        # Snap to front view after data replay so auto-fit is 2D-correct.
        # Data replay may call fit_camera() which resets parallel→False, so
        # restore orthographic projection here unconditionally.
        self._renderer.set_parallel_projection(True)
        self._renderer.view_xz()
        self._mark_dirty()

        # Sync _marginal_coords for actors that were replayed by the parent.
        # _phandle_map is now fully populated: vhandle → real handle.
        for kwargs, vhandle in saved_actors:
            real = self._phandle_map.get(vhandle)
            if real is None:
                continue
            pos = kwargs.get("positions")
            if (pos is not None and hasattr(pos, "ndim")
                    and pos.ndim == 2 and pos.shape[1] >= 2):
                self._marginal_coords[real] = (pos[:, 0].copy(), pos[:, 1].copy())

        # Mirror the visibility replay into _hidden_actors.
        for vhandle, visible in saved_vis.items():
            real = self._phandle_map.get(vhandle, vhandle)
            if not visible:
                self._hidden_actors.add(real)

    # ── Data overrides ────────────────────────────────────────────────────

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
        # Save the normalised (x,y,z) array so set_points/add_points can extract
        # the correct x/y columns for marginals without re-touching the raw
        # positions argument (which may be a DataFrame with non-numeric columns).
        self._last_prep_pos = pos
        return pos, clr, scl, sizes, meta

    # ── Camera overrides ──────────────────────────────────────────────────

    def reset_camera(self) -> None:
        """Fit the camera to the data and restore the 2D front view."""
        super().reset_camera()
        if self._renderer is not None:
            self._renderer.set_parallel_projection(True)  # reset_camera/fit_camera resets parallel→False
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

    # ── Marginal histogram API ────────────────────────────────────────────

    def show_marginals(
        self,
        visible: bool = True,
        *,
        bins: "int | str" = 50,
        color: str = "#4c8eff",
        alpha: float = 0.7,
        size: int = 80,
        orientation: str = "both",
    ) -> None:
        """Show or hide marginal histograms above (X) and to the right (Y) of the scatter.

        Parameters
        ----------
        visible : bool
            Whether to display the marginals.
        bins : int or "auto"
            Number of histogram bins.
        color : str
            Bar fill color as a hex string.
        alpha : float
            Bar opacity in [0, 1].
        size : int
            Height of the X histogram / width of the Y histogram in pixels.
        orientation : {"both", "x", "y"}
            Which marginals to show.
        """
        self._marginals_bins = bins
        self._marginals_color = color
        self._marginals_alpha = float(alpha)
        self._marginals_size = int(size)
        self._marginals_orientation = orientation
        self._marginals_visible = bool(visible)

        if not visible:
            self._destroy_marginal_canvases()
            return

        self._create_marginal_canvases()
        # Defer drawing until after Tk processes the grid placement so the
        # canvases have their final sizes.  Camera-hash polling in _render_tick
        # will keep histograms in sync with subsequent pan/zoom/resize.
        self.after_idle(self.update_marginals)

    def update_marginals(self) -> None:
        """Recompute histograms from current point data and redraw the canvases."""
        if not self._marginals_visible:
            return
        self._draw_x_hist()
        self._draw_y_hist()

    def _create_marginal_canvases(self) -> None:
        """Create/destroy histogram canvases and apply grid layout."""
        s = self._marginals_size
        if self._marginals_orientation in ("x", "both"):
            if self._x_hist_canvas is None:
                # height= pins the canvas's requested height so the grid row
                # is exactly s pixels tall; width=1 lets sticky="nsew" expand
                # it horizontally without requesting extra space.
                self._x_hist_canvas = tk.Canvas(
                    self, bg="#1a1a2e", highlightthickness=0, height=s, width=1
                )
            else:
                self._x_hist_canvas.configure(height=s)
        else:
            if self._x_hist_canvas is not None:
                self._x_hist_canvas.grid_remove()
                self._x_hist_canvas.destroy()
                self._x_hist_canvas = None

        if self._marginals_orientation in ("y", "both"):
            if self._y_hist_canvas is None:
                # width= pins the canvas's requested width so the grid column
                # is exactly s pixels wide; height=1 lets sticky="nsew" expand.
                self._y_hist_canvas = tk.Canvas(
                    self, bg="#1a1a2e", highlightthickness=0, width=s, height=1
                )
            else:
                self._y_hist_canvas.configure(width=s)
        else:
            if self._y_hist_canvas is not None:
                self._y_hist_canvas.grid_remove()
                self._y_hist_canvas.destroy()
                self._y_hist_canvas = None

        self._place_marginal_canvases()

    def _place_marginal_canvases(self) -> None:
        """Apply grid geometry to the histogram canvases."""
        s = self._marginals_size
        if self._x_hist_canvas is not None:
            # Sync canvas height in case size= changed since creation.
            self._x_hist_canvas.configure(height=s)
            self._x_hist_canvas.grid(row=0, column=0, sticky="nsew")
            self.rowconfigure(0, weight=0)
        else:
            self.rowconfigure(0, weight=0, minsize=0)

        if self._y_hist_canvas is not None:
            self._y_hist_canvas.configure(width=s)
            self._y_hist_canvas.grid(row=0, column=1, rowspan=2, sticky="nsew")
            self.columnconfigure(1, weight=0)
        else:
            self.columnconfigure(1, weight=0, minsize=0)

    def _destroy_marginal_canvases(self) -> None:
        for cv in (self._x_hist_canvas, self._y_hist_canvas):
            if cv is not None:
                cv.grid_remove()
                cv.destroy()
        self._x_hist_canvas = None
        self._y_hist_canvas = None
        self.rowconfigure(0, weight=0, minsize=0)
        self.columnconfigure(1, weight=0, minsize=0)

    def _aggregate_marginal_coords(self) -> "tuple[np.ndarray, np.ndarray]":
        xs, ys = [], []
        for h, (x, y) in self._marginal_coords.items():
            if h not in self._hidden_actors:
                xs.append(x)
                ys.append(y)
        return (
            np.concatenate(xs) if xs else np.empty(0, dtype=np.float32),
            np.concatenate(ys) if ys else np.empty(0, dtype=np.float32),
        )

    @staticmethod
    def _blend_color(hex_color: str, alpha: float, bg: str = "#1a1a2e") -> str:
        """Alpha-blend hex_color over bg and return a hex color string."""
        def _parse(h: str) -> "tuple[int, int, int]":
            h = h.lstrip("#")
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

        fr, fg, fb = _parse(hex_color)
        br, bg_, bb = _parse(bg)
        a = max(0.0, min(1.0, alpha))
        r = int(fr * a + br * (1 - a))
        g = int(fg * a + bg_ * (1 - a))
        b = int(fb * a + bb * (1 - a))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _draw_x_hist(self) -> None:
        cv = self._x_hist_canvas
        if cv is None or self._renderer is None:
            return
        cw = cv.winfo_width()
        ch = cv.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        bounds = self._renderer.get_view_bounds_2d()
        xmin, xmax = bounds[0], bounds[2]
        if xmax <= xmin:
            return

        xs, _ = self._aggregate_marginal_coords()
        cv.delete("all")
        if xs.size == 0:
            return

        bins = self._marginals_bins
        counts, edges = np.histogram(xs, bins=bins, range=(xmin, xmax))
        max_count = counts.max()
        if max_count == 0:
            return

        fill = self._blend_color(self._marginals_color, self._marginals_alpha)
        x_range = xmax - xmin
        for i, count in enumerate(counts):
            x0_w = edges[i]
            x1_w = edges[i + 1]
            px0 = int((x0_w - xmin) / x_range * cw)
            px1 = int((x1_w - xmin) / x_range * cw)
            bar_h = int(count / max_count * (ch - 4))
            if bar_h < 1:
                continue
            cv.create_rectangle(px0, ch - bar_h, px1, ch, fill=fill, outline="")

    def _draw_y_hist(self) -> None:
        cv = self._y_hist_canvas
        if cv is None or self._renderer is None:
            return
        cw = cv.winfo_width()
        ch = cv.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        bounds = self._renderer.get_view_bounds_2d()
        ymin, ymax = bounds[1], bounds[3]
        if ymax <= ymin:
            return

        _, ys = self._aggregate_marginal_coords()
        cv.delete("all")
        if ys.size == 0:
            return

        bins = self._marginals_bins
        counts, edges = np.histogram(ys, bins=bins, range=(ymin, ymax))
        max_count = counts.max()
        if max_count == 0:
            return

        fill = self._blend_color(self._marginals_color, self._marginals_alpha)
        y_range = ymax - ymin
        for i, count in enumerate(counts):
            y0_w = edges[i]
            y1_w = edges[i + 1]
            # In canvas coords: y=0 is top; ymax world = canvas top
            py0 = int((ymax - y1_w) / y_range * ch)
            py1 = int((ymax - y0_w) / y_range * ch)
            bar_w = int(count / max_count * (cw - 4))
            if bar_w < 1:
                continue
            cv.create_rectangle(0, py0, bar_w, py1, fill=fill, outline="")

    def _check_camera_changed_for_marginals(self) -> None:
        if not self._marginals_visible or self._renderer is None:
            return
        # Hash the actual world-space view bounds so that both pan/zoom AND
        # aspect-ratio changes (from widget resize) trigger a redraw.
        bounds = tuple(self._renderer.get_view_bounds_2d())
        if bounds != self._last_bounds_hash:
            self._last_bounds_hash = bounds
            self._update_marginals_async()

    def _update_marginals_async(self) -> None:
        self.after_idle(self.update_marginals)

    # ── Lifecycle overrides ───────────────────────────────────────────────

    def destroy(self) -> None:
        self._destroy_marginal_canvases()
        super().destroy()

    def _render_tick(self) -> None:
        was_dirty = self._dirty
        super()._render_tick()
        # After a successful render (_dirty flipped to False), check whether the
        # camera moved so marginal histograms can be updated.
        if was_dirty and not self._dirty:
            self._check_camera_changed_for_marginals()

    # ── Data lifecycle hooks ──────────────────────────────────────────────

    def set_points(self, positions, **kwargs) -> None:
        """Load a point cloud, snap to front view, and update marginal coords."""
        self._marginal_coords.clear()
        self._hidden_actors.clear()
        self._last_prep_pos = None
        super().set_points(positions, **kwargs)
        if self._renderer is not None:
            self._renderer.set_parallel_projection(True)  # fit_camera() in set_points resets parallel→False
            self._renderer.view_xz()
            self._mark_dirty()
            h = self._scene_actor_handle
            pos = self._last_prep_pos  # normalised by _prepare_point_inputs
            if h is not None and pos is not None and pos.ndim == 2 and pos.shape[1] >= 2:
                self._marginal_coords[h] = (pos[:, 0].copy(), pos[:, 1].copy())
            if self._marginals_visible:
                self._update_marginals_async()

    def add_points(self, positions, **kwargs) -> int:
        """Add a point cloud actor and register its coords for marginals."""
        self._last_prep_pos = None
        handle = super().add_points(positions, **kwargs)
        pos = self._last_prep_pos  # normalised by _prepare_point_inputs
        if handle >= 0 and pos is not None and pos.ndim == 2 and pos.shape[1] >= 2:
            self._marginal_coords[handle] = (pos[:, 0].copy(), pos[:, 1].copy())
        if handle >= 0 and self._marginals_visible:
            self._update_marginals_async()
        return handle

    def update_actor(self, handle: int, positions: "np.ndarray", **kwargs) -> None:
        """Flatten z to 0, update actor data, and refresh marginal coords."""
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        if pos.ndim == 2 and pos.shape[1] >= 3:
            pos = pos.copy()
            pos[:, 2] = 0.0
        super().update_actor(handle, pos, **kwargs)
        real = self._resolve_actor_handle(handle)
        if pos.ndim == 2 and pos.shape[1] >= 2:
            self._marginal_coords[real] = (pos[:, 0].copy(), pos[:, 1].copy())
        if self._marginals_visible:
            self._update_marginals_async()

    def remove_actor(self, handle: int) -> None:
        """Remove an actor and drop its marginal coords."""
        real = self._resolve_actor_handle(handle)
        super().remove_actor(handle)
        self._marginal_coords.pop(real, None)
        self._marginal_coords.pop(handle, None)
        self._hidden_actors.discard(real)
        self._hidden_actors.discard(handle)
        if self._marginals_visible:
            self._update_marginals_async()

    def set_actor_visibility(self, handle: int, visible: bool) -> None:
        """Show/hide an actor and update marginal exclusion set."""
        super().set_actor_visibility(handle, visible)
        real = self._resolve_actor_handle(handle)
        if visible:
            self._hidden_actors.discard(real)
        else:
            self._hidden_actors.add(real)
        if self._marginals_visible:
            self._update_marginals_async()

    def stream(self, handle: int, positions, **kwargs) -> None:
        """Stream new points into an actor and update marginal coords (capped)."""
        super().stream(handle, positions, **kwargs)
        real = self._resolve_actor_handle(handle)
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        if pos.ndim == 2 and pos.shape[1] >= 2:
            new_x = pos[:, 0].copy()
            new_y = pos[:, 1].copy()
            if real in self._marginal_coords:
                existing_x, existing_y = self._marginal_coords[real]
                combined_x = np.concatenate([existing_x, new_x])
                combined_y = np.concatenate([existing_y, new_y])
                if len(combined_x) > self._marginal_stream_cap:
                    combined_x = combined_x[-self._marginal_stream_cap:]
                    combined_y = combined_y[-self._marginal_stream_cap:]
                self._marginal_coords[real] = (combined_x, combined_y)
            else:
                self._marginal_coords[real] = (new_x, new_y)
        if self._marginals_visible:
            self._update_marginals_async()

    def clear(self) -> None:
        """Clear the scene and drop all marginal coords."""
        self._marginal_coords.clear()
        self._hidden_actors.clear()
        super().clear()
        if self._marginals_visible:
            self._update_marginals_async()

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


# ── Line2D ───────────────────────────────────────────────────────────────────

class Line2D(Scatter3D):
    """A Tkinter widget that renders 2-D line charts using wgpu (Rust).

    Uses a dedicated screen-space chart rendering path instead of the 3D
    camera — axes are anchored to the window, data lines are clipped to the
    plot rect, and ticks/labels are positioned in pixel space.  No camera
    pan/orbit/zoom; the chart frame is stable during streaming.

    Usage — static plot
    --------------------
    ::

        import tkinter as tk
        import numpy as np
        from dragonsci import Line2D

        root = tk.Tk()
        w = Line2D(root, width=800, height=600)
        w.pack(fill="both", expand=True)

        x = np.linspace(0, 4 * np.pi, 1_000)
        w.set_line(x, np.sin(x))

        root.mainloop()

    Usage — streaming / live plot
    ------------------------------
    ::

        w = Line2D(root, width=800, height=600)
        w.pack(fill="both", expand=True)
        w.set_xlim(0, 100)
        w.set_ylim(-1, 1)

        handle = w.add_line_stream(max_points=5_000, mode="ring")

        def tick():
            w.stream_line(handle, new_x, new_y)
            w.after(33, tick)

        w.after(33, tick)
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
        super().__init__(master, width=width, height=height,
                         fps=fps, vsync=vsync, **kwargs)

        # Axis limits — drive the chart2d transform.
        self._xlim: "tuple[float, float] | None" = None
        self._ylim: "tuple[float, float] | None" = None

        # Axis / chart label text
        self._xlabel: str = "X"
        self._ylabel: str = "Y"
        self._title:  str = ""
        self._y_tick_interval: "float | None" = None
        self._x_tick_interval: "float | None" = None

        # Axis freeze flags: x is often explicit for time-series windows while y stays auto.
        self._x_limits_frozen: bool = False
        self._y_limits_frozen: bool = False
        self._limits_frozen: bool = False

        # Running data extent — updated by _refit_all_static/_refit_from_all_sources; used by home/autoscale.
        self._data_xmin: "float | None" = None
        self._data_xmax: "float | None" = None
        self._data_ymin: "float | None" = None
        self._data_ymax: "float | None" = None

        # Stored geometry for the primary (set_line) line; kept so that
        # _current_data_bounds() can scan it after update_line / clear.
        self._primary_x: "np.ndarray | None" = None
        self._primary_y: "np.ndarray | None" = None

        # Plot rect (viewport fractions). Left/right/top/bottom from the edges.
        # top < bottom because both are measured from the top of the window.
        self._pad_left:   float = 0.13
        self._pad_right:  float = 0.97
        self._pad_top:    float = 0.04
        self._pad_bottom: float = 0.88

        # Primary (set_line) handle — replaced on each set_line call.
        self._primary_handle: "int | None" = None
        self._primary_color: tuple = (0.3, 0.7, 1.0)
        self._primary_width: float = 2.0
        self._primary_label: "str | None" = None
        # Pending primary line (set before renderer existed); 5-tuple (x,y,color,lw,label)
        self._pending_primary: "tuple | None" = None

        # Named extra lines: handle -> {"color": tuple, "line_width": float,
        #                               "label": str|None, "visible": bool}
        self._named_lines: "dict[int, dict]" = {}
        # Pre-renderer queue: vhandle -> (x, y, color, line_width, label, visible)
        self._pending_named_lines: "dict[int, tuple]" = {}
        self._next_nhandle: int = 0
        self._nhandle_map: "dict[int, int]" = {}

        # Legend state
        self._legend_visible:  bool = False
        self._legend_position: str  = "top-right"

        # Streaming line state: stream-handle (int) → state dict
        self._line_streams: "dict[int, dict]" = {}
        self._next_stream: int = 0

        # True once set_chart2d has been sent to the renderer at least once.
        # Used to gate the fast-path chart2d_update_xlim optimization.
        self._chart2d_sent: bool = False

        # Reference overlays: overlay-handle -> {"kind": str, "args": tuple}
        # kind in {"hspan", "vspan", "hline", "vline"}
        # Stored for replay after renderer is created and for clear_chart_overlays.
        self._overlay_meta: "dict[int, dict]" = {}
        # Maps overlay handle back to the Rust id (for remove_overlay).
        self._overlay_handle_map: "dict[int, int]" = {}
        # Next virtual overlay handle.
        self._next_overlay_handle: int = 0

        # Axis formatting / scale (persisted so they survive renderer re-init).
        self._x_tick_format: str = "default"
        self._y_tick_format: str = "default"
        self._x_log_scale: bool = False
        self._y_log_scale: bool = False

        # Cursor and box-zoom state
        self._cursor_enabled: bool = False
        self._box_zoom_enabled: bool = False
        self._box_zoom_active: bool = False   # True while streaming is frozen by box zoom
        self._bz_dragging: bool = False
        self._bz_px0: int = 0
        self._bz_py0: int = 0

        # ── Sub-frame layout ──────────────────────────────────────────────
        # Row 0: toolbar strip (fixed height)
        # Row 1: render surface (expands)
        # Row 2: status readout strip (fixed height)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)   # toolbar
        self.rowconfigure(1, weight=1)   # render surface
        self.rowconfigure(2, weight=0)   # status readout

        self._toolbar_frame = tk.Frame(self, bg="#111118", height=28)
        self._toolbar_frame.grid(row=0, column=0, sticky="ew")
        self._toolbar_frame.pack_propagate(False)

        self._render_frame = tk.Frame(self, bg="black")
        self._render_frame.grid(row=1, column=0, sticky="nsew")

        self._status_frame = tk.Frame(self, bg="#0d0d1a", height=18)
        self._status_frame.grid(row=2, column=0, sticky="ew")
        self._status_frame.pack_propagate(False)
        self._status_label = tk.Label(
            self._status_frame, text="",
            bg="#0d0d1a", fg="#666688", font=("Consolas", 8),
            anchor="w", padx=8,
        )
        self._status_label.pack(fill="x")

        # Redirect renderer surface to the inner render frame.
        self._render_target_widget = self._render_frame

        # Resize events on the render frame trigger the debounced resize.
        self._render_frame.bind("<Configure>", self._on_configure, add="+")

        # Re-bind all input events from the outer frame to the render frame so
        # event.x/y are in render-surface space (origin = top-left of _render_frame).
        _INPUT_EVENTS = (
            "<ButtonPress-1>", "<ButtonPress-2>",
            "<B1-Motion>", "<B2-Motion>",
            "<ButtonRelease-1>", "<ButtonRelease-2>",
            "<MouseWheel>", "<Button-4>", "<Button-5>",
            "<Double-Button-1>", "<Motion>", "<Leave>",
        )
        for ev in _INPUT_EVENTS:
            self.unbind(ev)
        self._render_frame.bind("<ButtonPress-1>",   lambda e: self._drag_start(e, 1))
        self._render_frame.bind("<ButtonPress-2>",   lambda e: self._drag_start(e, 2))
        self._render_frame.bind("<B1-Motion>",       lambda e: self._drag_move(e, 1))
        self._render_frame.bind("<B2-Motion>",       lambda e: self._drag_move(e, 2))
        self._render_frame.bind("<ButtonRelease-1>", self._drag_end)
        self._render_frame.bind("<ButtonRelease-2>", self._drag_end)
        self._render_frame.bind("<MouseWheel>",      self._on_scroll)
        self._render_frame.bind("<Button-4>",        self._on_scroll_up_x11)
        self._render_frame.bind("<Button-5>",        self._on_scroll_down_x11)
        self._render_frame.bind("<Motion>",          self._on_hover_motion, add="+")
        self._render_frame.bind("<Leave>",           self._on_hover_leave, add="+")

        self._build_toolbar()

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        """Populate the compact chart toolbar strip."""
        _BTN_BG  = "#1e1e2e"
        _BTN_FG  = "#c0c0d0"
        _BTN_ABG = "#2a2a3e"
        _BTN_AFG = "#7ecfff"
        _FONT    = ("Consolas", 9)

        def _tbtn(text: str, cmd, tooltip: str = "") -> tk.Button:
            b = tk.Button(
                self._toolbar_frame,
                text=text,
                command=cmd,
                bg=_BTN_BG, fg=_BTN_FG,
                activebackground=_BTN_ABG, activeforeground=_BTN_AFG,
                relief="flat", bd=0,
                font=_FONT,
                padx=8, pady=3,
                cursor="hand2",
            )
            b.pack(side="left", padx=1, pady=2)
            if tooltip:
                self._toolbar_tooltip(b, tooltip)
            return b

        _tbtn("⌂ Home",         self.home,           "Reset to full data extent")
        _tbtn("↕ Autoscale Y",  self.autoscale_y,    "Fit Y axis to data")
        _tbtn("⤢ Autoscale",    self.autoscale_both, "Fit both axes to data")

        # Separator
        tk.Frame(self._toolbar_frame, bg="#333344", width=1).pack(
            side="left", fill="y", padx=4, pady=4)

        _tbtn("▶ Resume Live",  self.resume_live,    "Resume live axis scrolling after box zoom")

        # Separator
        tk.Frame(self._toolbar_frame, bg="#333344", width=1).pack(
            side="left", fill="y", padx=4, pady=4)

        _tbtn("💾 Save PNG",    self._toolbar_save,  "Save chart to PNG")

    def _toolbar_tooltip(self, widget: tk.Widget, text: str) -> None:
        """Attach a simple hover tooltip to a toolbar button."""
        tip: "list[tk.Toplevel | None]" = [None]

        def _show(event: tk.Event) -> None:
            if tip[0] is not None:
                return
            x = widget.winfo_rootx() + widget.winfo_width() // 2
            y = widget.winfo_rooty() + widget.winfo_height() + 4
            tw = tk.Toplevel(widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            tk.Label(tw, text=text, bg="#2a2a3e", fg="#c0c0d0",
                     relief="flat", font=("Consolas", 8), padx=4, pady=2).pack()
            tip[0] = tw

        def _hide(_event: tk.Event) -> None:
            if tip[0] is not None:
                tip[0].destroy()
                tip[0] = None

        widget.bind("<Enter>", _show, add="+")
        widget.bind("<Leave>", _hide, add="+")

    def _toolbar_save(self) -> None:
        """Prompt for a path and save the chart as PNG."""
        import tkinter.filedialog as _fd
        path = _fd.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            title="Save chart as PNG",
        )
        if path:
            self.save_png(path)

    # ── Toolbar actions ───────────────────────────────────────────────────────

    def _current_data_bounds(self) -> "tuple[float,float,float,float] | None":
        """Return (xmin, xmax, ymin, ymax) across all currently-buffered data.

        Scans stored geometry in ``_named_lines`` and the primary line directly,
        so bounds are always accurate even after ``update_line`` shrinks data or
        ``remove_line`` drops a series.  Stream buffers are always scanned live.
        """
        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        # Primary (set_line) geometry.
        if self._primary_x is not None and len(self._primary_x) > 0:
            xmins.append(float(self._primary_x.min()))
            xmaxs.append(float(self._primary_x.max()))
            ymins.append(float(self._primary_y.min()))
            ymaxs.append(float(self._primary_y.max()))
        # Named static lines — scan stored geometry directly (not monotonic accumulator).
        for meta in self._named_lines.values():
            x, y = meta.get("x"), meta.get("y")
            if x is not None and len(x) > 0:
                xmins.append(float(x.min())); xmaxs.append(float(x.max()))
                ymins.append(float(y.min())); ymaxs.append(float(y.max()))
        # Pre-renderer pending named lines.
        for (x, y, *_rest) in self._pending_named_lines.values():
            if len(x) > 0:
                xmins.append(float(x.min())); xmaxs.append(float(x.max()))
                ymins.append(float(y.min())); ymaxs.append(float(y.max()))
        # Live stream buffers — always re-scan so scrolled-off data is excluded.
        for st in self._line_streams.values():
            cnt = st["count"]
            if cnt < 1:
                continue
            xs, ys = self._stream_ordered(st)
            if xs is None:
                xs, ys = st["buf_x"][:cnt], st["buf_y"][:cnt]
            xmins.append(float(xs.min())); xmaxs.append(float(xs.max()))
            ymins.append(float(ys.min())); ymaxs.append(float(ys.max()))
        if not xmins:
            return None
        return min(xmins), max(xmaxs), min(ymins), max(ymaxs)

    def _refit_all_static(self) -> None:
        """Recompute data bounds from all static line geometry and apply to unfrozen axes.

        Unlike the old monotonic accumulator, this
        scans the current stored geometry — including pre-renderer pending lines —
        so adding, removing, or shrinking a line immediately corrects both the
        accumulator and the visible range.
        """
        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        if self._primary_x is not None and len(self._primary_x) > 0:
            xmins.append(float(self._primary_x.min()))
            xmaxs.append(float(self._primary_x.max()))
            ymins.append(float(self._primary_y.min()))
            ymaxs.append(float(self._primary_y.max()))
        for meta in self._named_lines.values():
            x, y = meta.get("x"), meta.get("y")
            if x is not None and len(x) > 0:
                xmins.append(float(x.min())); xmaxs.append(float(x.max()))
                ymins.append(float(y.min())); ymaxs.append(float(y.max()))
        # Also scan lines that are queued but not yet sent to the renderer.
        for (x, y, *_rest) in self._pending_named_lines.values():
            if len(x) > 0:
                xmins.append(float(x.min())); xmaxs.append(float(x.max()))
                ymins.append(float(y.min())); ymaxs.append(float(y.max()))
        if xmins:
            self._data_xmin = min(xmins); self._data_xmax = max(xmaxs)
            self._data_ymin = min(ymins); self._data_ymax = max(ymaxs)
            if not self._x_limits_frozen:
                self._apply_xlim(*_nice_bounds_1d(self._data_xmin, self._data_xmax), freeze=False)
            if not self._y_limits_frozen:
                self._apply_ylim(*_nice_bounds_1d(self._data_ymin, self._data_ymax), freeze=False)
        else:
            # All static data removed; reset accumulator but leave limits alone.
            self._data_xmin = self._data_xmax = None
            self._data_ymin = self._data_ymax = None

    def _refit_from_all_sources(self) -> None:
        """Refit unfrozen axes from *all* data (static + streams).

        Used after a stream is cleared or removed so that stale limits are
        corrected.  When no data remains at all, ``_xlim`` and ``_ylim`` are
        reset to ``None`` so the chart shows a default empty state.
        """
        bounds = self._current_data_bounds()
        if bounds is None:
            self._data_xmin = self._data_xmax = None
            self._data_ymin = self._data_ymax = None
            # Only reset unfrozen axes; frozen axes keep the user-specified range.
            if not self._x_limits_frozen:
                self._xlim = None
            if not self._y_limits_frozen:
                self._ylim = None
            # Always push so the renderer gets an updated empty state regardless
            # of which combination of axes is frozen.
            self._push_chart2d()
        else:
            xmin, xmax, ymin, ymax = bounds
            self._data_xmin = xmin; self._data_xmax = xmax
            self._data_ymin = ymin; self._data_ymax = ymax
            if not self._x_limits_frozen:
                self._apply_xlim(*_nice_bounds_1d(xmin, xmax), freeze=False)
            if not self._y_limits_frozen:
                self._apply_ylim(*_nice_bounds_1d(ymin, ymax), freeze=False)

    def home(self) -> None:
        """Reset both axes to the full recorded data extent."""
        self._box_zoom_active = False  # release any streaming freeze
        bounds = self._current_data_bounds()
        if bounds is None:
            return
        xmin, xmax, ymin, ymax = bounds
        self._x_limits_frozen = False
        self._y_limits_frozen = False
        self._sync_limit_freeze()
        self._apply_xlim(*_nice_bounds_1d(xmin, xmax), freeze=False)
        self._apply_ylim(*_nice_bounds_1d(ymin, ymax), freeze=False)

    def autoscale_y(self) -> None:
        """Unfreeze Y and fit it to currently-buffered data."""
        bounds = self._current_data_bounds()
        if bounds is None:
            return
        _, _, ymin, ymax = bounds
        self._y_limits_frozen = False
        self._sync_limit_freeze()
        self._apply_ylim(*_nice_bounds_1d(ymin, ymax), freeze=False)

    def autoscale_both(self) -> None:
        """Alias for :meth:`home` — unfreeze both axes and fit to data."""
        self.home()

    # ── Cursor and box-zoom API ───────────────────────────────────────────────

    def enable_cursor(self, enabled: bool, *, snap: bool = False) -> None:
        """Show a crosshair cursor that tracks the mouse over the plot area.

        Parameters
        ----------
        enabled : bool
        snap : bool
            Must be ``False`` in v1.  Passing ``True`` raises
            ``NotImplementedError`` until nearest-sample snapping is implemented.
        """
        if snap:
            raise NotImplementedError(
                "snap=True is not yet supported; use snap=False")
        self._cursor_enabled = bool(enabled)
        if not enabled and self._renderer is not None:
            self._renderer.chart2d_set_cursor(0.0, 0.0, False)
            self._status_label.configure(text="")
            self._mark_dirty()

    def enable_box_zoom(self, enabled: bool) -> None:
        """Enable or disable left-drag box zoom on the plot area.

        When box zoom is active on a streaming chart the live axis animation is
        suspended until the user calls :meth:`resume_live`, :meth:`home`, or
        :meth:`autoscale_both`.
        """
        self._box_zoom_enabled = bool(enabled)
        if not enabled:
            self._box_zoom_active = False
            self._bz_dragging = False

    def resume_live(self) -> None:
        """Release the box-zoom streaming freeze so axis animation resumes."""
        self._box_zoom_active = False

    # ── Mouse overrides ───────────────────────────────────────────────────────

    def _drag_start(self, event: tk.Event, button: int) -> None:
        if button == 1 and self._box_zoom_enabled:
            self._bz_dragging = True
            self._bz_px0 = event.x
            self._bz_py0 = event.y
            self._drag_btn = button
            return
        super()._drag_start(event, button)

    def _drag_end(self, event: tk.Event) -> None:
        if self._bz_dragging:
            self._bz_dragging = False
            self._drag_btn = None
            if self._renderer is not None:
                self._renderer.clear_selection_rect()
                self._mark_dirty()
            self._apply_box_zoom(self._bz_px0, self._bz_py0, event.x, event.y)
            return
        super()._drag_end(event)

    def _on_hover_motion(self, event: tk.Event) -> None:
        if self._cursor_enabled:
            self._update_cursor(event.x, event.y)

    def _on_hover_leave(self, event: tk.Event) -> None:
        if self._cursor_enabled and self._renderer is not None:
            self._renderer.chart2d_set_cursor(0.0, 0.0, False)
            self._mark_dirty()
        self._status_label.configure(text="")

    def _update_cursor(self, mx: int, my: int) -> None:
        """Convert pixel position to data coords and update the renderer cursor."""
        tgt = self._render_target_widget
        w = tgt.winfo_width()
        h = tgt.winfo_height()
        if w < 1 or h < 1 or self._xlim is None or self._ylim is None:
            return
        pl = self._pad_left   * w
        pr = self._pad_right  * w
        pt = self._pad_top    * h
        pb = self._pad_bottom * h
        if not (pl <= mx <= pr and pt <= my <= pb):
            if self._renderer is not None:
                self._renderer.chart2d_set_cursor(0.0, 0.0, False)
                self._mark_dirty()
            self._status_label.configure(text="")
            return
        x0, x1 = self._xlim
        y0, y1 = self._ylim
        x_data = x0 + (mx - pl) / (pr - pl) * (x1 - x0)
        y_data = y1 - (my - pt) / (pb - pt) * (y1 - y0)
        if self._renderer is not None:
            self._renderer.chart2d_set_cursor(float(x_data), float(y_data), True)
            self._mark_dirty()
        self._status_label.configure(text=f"x = {x_data:.4g}   y = {y_data:.4g}")

    def _apply_box_zoom(self, px0: int, py0: int, px1: int, py1: int) -> None:
        """Convert a pixel drag rect to data-space limits and apply them."""
        tgt = self._render_target_widget
        w = tgt.winfo_width()
        h = tgt.winfo_height()
        if w < 1 or h < 1 or self._xlim is None or self._ylim is None:
            return
        pl = self._pad_left   * w
        pr = self._pad_right  * w
        pt = self._pad_top    * h
        pb = self._pad_bottom * h
        pw = pr - pl; ph = pb - pt
        if pw < 1 or ph < 1:
            return
        x0, x1 = self._xlim
        y0, y1 = self._ylim
        # Map pixel corners to data space, clamped to plot rect.
        xa = x0 + (max(pl, min(pr, float(px0))) - pl) / pw * (x1 - x0)
        xb = x0 + (max(pl, min(pr, float(px1))) - pl) / pw * (x1 - x0)
        ya = y1 - (max(pt, min(pb, float(py0))) - pt) / ph * (y1 - y0)
        yb = y1 - (max(pt, min(pb, float(py1))) - pt) / ph * (y1 - y0)
        new_x0, new_x1 = min(xa, xb), max(xa, xb)
        new_y0, new_y1 = min(ya, yb), max(ya, yb)
        if abs(new_x1 - new_x0) < 1e-10 or abs(new_y1 - new_y0) < 1e-10:
            return  # Degenerate drag — ignore
        self._apply_xlim(new_x0, new_x1, freeze=True)
        self._apply_ylim(new_y0, new_y1, freeze=True)
        self._box_zoom_active = True  # Freeze streaming axis animation

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all lines, overlays, and streams and reset chart bounds."""
        # Python-side state
        self._pending_primary = None
        self._primary_handle = None
        self._primary_label = None
        self._primary_x = self._primary_y = None
        self._pending_named_lines.clear()
        self._named_lines.clear()
        self._nhandle_map.clear()
        self._next_nhandle = 0
        self._line_streams.clear()
        self._next_stream = 0
        self._overlay_meta.clear()
        self._overlay_handle_map.clear()
        self._next_overlay_handle = 0
        self._data_xmin = self._data_xmax = None
        self._data_ymin = self._data_ymax = None
        self._xlim = None
        self._ylim = None
        self._x_limits_frozen = False
        self._y_limits_frozen = False
        self._chart2d_sent = False
        # Renderer-side state
        if self._renderer is not None:
            self._renderer.chart2d_clear_lines()
            self._renderer.chart2d_clear_overlays()
            self._push_chart2d()
            self._push_legend()
            self._mark_dirty()

    def _init_renderer(self) -> None:
        super()._init_renderer()
        if self._renderer is None:
            return
        self._push_chart2d()
        # Replay primary line set before renderer existed.
        if self._pending_primary is not None:
            x, y, color, line_width, label = self._pending_primary
            self._pending_primary = None
            self._primary_label = label
            self._primary_handle = self._renderer.chart2d_add_line(
                x, y, color, line_width
            )
            self._mark_dirty()
        # Replay any streams that have data buffered already.
        for vhandle, (x, y, color, line_width, label, visible) in list(self._pending_named_lines.items()):
            real = self._renderer.chart2d_add_line(x, y, color, line_width)
            self._nhandle_map[vhandle] = real
            self._named_lines[real] = {
                "color": color, "line_width": line_width,
                "label": label, "visible": visible,
                "x": x.copy(), "y": y.copy(),
            }
            if not visible:
                self._renderer.chart2d_set_line_visible(real, False)
            self._mark_dirty()
        self._pending_named_lines.clear()
        for st in self._line_streams.values():
            cnt = st["count"]
            if cnt >= 2 and st["render_handle"] is None:
                xs, ys = self._stream_ordered(st)
                if xs is not None:
                    st["render_handle"] = self._renderer.chart2d_add_line(
                        xs, ys, st["color"], st["line_width"]
                    )
                    self._mark_dirty()
        # Replay any overlays that were added before the renderer existed.
        for vhandle, meta in list(self._overlay_meta.items()):
            kind = meta["kind"]
            args = meta["args"]
            if kind == "hspan":
                rust_id = self._renderer.chart2d_add_hspan(*args)
            elif kind == "vspan":
                rust_id = self._renderer.chart2d_add_vspan(*args)
            elif kind == "hline":
                rust_id = self._renderer.chart2d_add_hline(*args)
            elif kind == "vline":
                rust_id = self._renderer.chart2d_add_vline(*args)
            else:
                continue
            self._overlay_handle_map[vhandle] = rust_id
            self._mark_dirty()
        # Restore axis formatting and log scale settings.
        if self._x_tick_format != "default":
            self._renderer.chart2d_set_tick_format("x", self._x_tick_format)
            self._mark_dirty()
        if self._y_tick_format != "default":
            self._renderer.chart2d_set_tick_format("y", self._y_tick_format)
            self._mark_dirty()
        if self._x_log_scale:
            self._renderer.chart2d_set_log_scale("x", True)
            self._mark_dirty()
        if self._y_log_scale:
            self._renderer.chart2d_set_log_scale("y", True)
            self._mark_dirty()
        # Replay legend (show_legend() before renderer init only stored the flag).
        if self._legend_visible:
            self._push_legend()

    def _do_resize(self, w: int, h: int) -> None:
        super()._do_resize(w, h)
        # Force a full set_chart2d rebuild after resize (axis geometry is in px).
        self._chart2d_sent = False
        self._push_chart2d()
        self._push_legend()  # legend positions are pixel-space; recompute on resize

    # ── Axis setup ────────────────────────────────────────────────────────────

    def set_xlim(self, xmin: float, xmax: float) -> None:
        """Set the visible X range."""
        self._apply_xlim(float(xmin), float(xmax), freeze=True)

    def set_ylim(self, ymin: float, ymax: float) -> None:
        """Set the visible Y range."""
        self._apply_ylim(float(ymin), float(ymax), freeze=True)

    def set_xlabel(self, label: str) -> None:
        """Set the X axis title."""
        self._xlabel = label
        self._push_chart2d()

    def set_ylabel(self, label: str) -> None:
        """Set the Y axis title."""
        self._ylabel = label
        self._push_chart2d()

    def set_y_tick_interval(self, step: "float | None") -> None:
        """Set a fixed Y-axis grid/tick interval, or ``None`` to auto-pick it."""
        if step is None:
            self._y_tick_interval = None
        else:
            step = float(step)
            if not np.isfinite(step) or step <= 0.0:
                raise ValueError("y tick interval must be a positive finite number")
            self._y_tick_interval = step
        self._push_chart2d()

    def set_x_tick_interval(self, step: "float | None") -> None:
        """Set a fixed X-axis grid/tick interval, or ``None`` to auto-pick it."""
        if step is None:
            self._x_tick_interval = None
        else:
            step = float(step)
            if not np.isfinite(step) or step <= 0.0:
                raise ValueError("x tick interval must be a positive finite number")
            self._x_tick_interval = step
        self._push_chart2d()

    def set_title(self, title: str) -> None:
        """Set the chart title displayed above the plot area."""
        self._title = str(title)
        # Reserve extra top padding so the title clears the frame.
        self._pad_top = 0.08 if title else 0.04
        self._chart2d_sent = False  # resize of top margin requires full rebuild
        self._push_chart2d()

    # ── Legend API ────────────────────────────────────────────────────────────

    def show_legend(self, visible: bool = True) -> None:
        """Show or hide the chart legend.  Entries come from ``label=`` on lines."""
        self._legend_visible = bool(visible)
        self._push_legend()

    @property
    def legend_position(self) -> str:
        """Legend anchor: ``"top-right"``, ``"top-left"``, ``"bottom-right"``, or
        ``"bottom-left"``."""
        return self._legend_position

    @legend_position.setter
    def legend_position(self, value: str) -> None:
        valid = {"top-right", "top-left", "bottom-right", "bottom-left"}
        if value not in valid:
            raise ValueError(
                f"legend_position must be one of {sorted(valid)!r}, got {value!r}")
        self._legend_position = value
        if self._legend_visible:
            self._push_legend()

    def set_line_visibility(self, handle: int, visible: bool) -> None:
        """Show or hide a named line (from :meth:`add_line`) without removing it."""
        visible = bool(visible)
        real = self._resolve_line_handle(handle)
        if real in self._named_lines:
            self._named_lines[real]["visible"] = visible
        elif handle in self._pending_named_lines:
            t = self._pending_named_lines[handle]
            self._pending_named_lines[handle] = (t[0], t[1], t[2], t[3], t[4], visible)
        if self._renderer is not None:
            self._renderer.chart2d_set_line_visible(real, visible)
            self._mark_dirty()

    def reset_camera(self) -> None:
        """Re-send chart2d state (no camera to reset in chart mode)."""
        self._push_chart2d()

    # ── Static / multi-line API ───────────────────────────────────────────────

    def set_line(
        self,
        x,
        y,
        *,
        color: "tuple[float, float, float]" = (0.3, 0.7, 1.0),
        line_width: float = 2.0,
        label: "str | None" = None,
    ) -> None:
        """Replace the primary polyline.

        Parameters
        ----------
        x, y : array-like, shape (N,)
            Coordinate arrays of equal length.
        color : (r, g, b) in [0, 1]
        line_width : float, optional
            Line width in screen pixels.
        label : str, optional
            Series label shown in the legend when :meth:`show_legend` is True.
        """
        x = np.asarray(x, dtype=np.float32).ravel()
        y = np.asarray(y, dtype=np.float32).ravel()
        line_width = _normalize_line2d_width(line_width)
        if len(x) != len(y):
            raise ValueError(
                f"x and y must have equal length, got {len(x)} vs {len(y)}")
        self._primary_x = x.copy()
        self._primary_y = y.copy()
        self._refit_all_static()  # union of all current static lines, not just this one
        label = str(label) if label is not None else None
        self._primary_label = label
        if self._renderer is None:
            self._pending_primary = (x.copy(), y.copy(), color, line_width, label)
            self._primary_color = color
            self._primary_width = line_width
            return
        if self._primary_handle is None:
            self._primary_handle = self._renderer.chart2d_add_line(
                x, y, color, line_width
            )
        else:
            self._renderer.chart2d_update_line(
                self._primary_handle, x, y, color, line_width
            )
        self._primary_color = color
        self._primary_width = line_width
        if self._legend_visible:
            self._push_legend()
        self._mark_dirty()

    def add_line(
        self,
        x,
        y,
        *,
        color: "tuple[float, float, float]" = (0.9, 0.5, 0.2),
        line_width: float = 2.0,
        label: "str | None" = None,
    ) -> int:
        """Add a new polyline; returns a handle.

        Parameters
        ----------
        x, y : array-like, shape (N,)
        color : (r, g, b) in [0, 1]
        line_width : float, optional
            Line width in screen pixels.
        label : str, optional
            Series label shown in the legend when :meth:`show_legend` is True.

        Returns
        -------
        int
            Handle for :meth:`update_line` and :meth:`remove_line`.
        """
        x = np.asarray(x, dtype=np.float32).ravel()
        y = np.asarray(y, dtype=np.float32).ravel()
        line_width = _normalize_line2d_width(line_width)
        if len(x) != len(y):
            raise ValueError(
                f"x and y must have equal length, got {len(x)} vs {len(y)}")
        label = str(label) if label is not None else None
        if self._renderer is None:
            vhandle = self._next_nhandle
            self._next_nhandle += 1
            self._pending_named_lines[vhandle] = (
                x.copy(), y.copy(), color, line_width, label, True
            )
            # Refit from all static geometry (including this new pending line).
            self._refit_all_static()
            return vhandle
        handle = self._renderer.chart2d_add_line(x, y, color, line_width)
        self._named_lines[handle] = {
            "color": color, "line_width": line_width,
            "label": label, "visible": True,
            "x": x.copy(), "y": y.copy(),
        }
        # Refit from all static geometry now that the new line is stored.
        self._refit_all_static()
        if self._legend_visible and label is not None:
            self._push_legend()
        self._mark_dirty()
        return handle

    def _resolve_line_handle(self, handle: int) -> int:
        return self._nhandle_map.get(handle, handle)

    def update_line(
        self,
        handle: int,
        x,
        y,
        *,
        color: "tuple[float, float, float] | None" = None,
        line_width: "float | None" = None,
        label: "str | None" = None,
    ) -> None:
        """Replace the geometry of an existing line.

        Parameters
        ----------
        handle : int
            Handle returned by :meth:`add_line`.
        x, y : array-like, shape (N,)
        color : (r, g, b), optional — keeps existing colour when omitted.
        line_width : float, optional — keeps existing width when omitted.
        label : str, optional — updates the legend label when provided.
        """
        x = np.asarray(x, dtype=np.float32).ravel()
        y = np.asarray(y, dtype=np.float32).ravel()
        if len(x) != len(y):
            raise ValueError(
                f"x and y must have equal length, got {len(x)} vs {len(y)}")
        if self._renderer is None and handle in self._pending_named_lines:
            _old_x, _old_y, old_color, old_width, old_label, old_vis = \
                self._pending_named_lines[handle]
            c = color if color is not None else old_color
            lw = (
                _normalize_line2d_width(line_width)
                if line_width is not None
                else old_width
            )
            new_label = str(label) if label is not None else old_label
            self._pending_named_lines[handle] = (
                x.copy(), y.copy(), c, lw, new_label, old_vis
            )
            self._refit_all_static()
            return

        real = self._resolve_line_handle(handle)
        meta = self._named_lines.get(real, {})
        c = color if color is not None else meta.get("color", (0.3, 0.7, 1.0))
        lw = (
            _normalize_line2d_width(line_width)
            if line_width is not None
            else meta.get("line_width", 2.0)
        )
        if self._renderer is not None:
            self._renderer.chart2d_update_line(real, x, y, c, lw)
        # Upsert into _named_lines so geometry is always available for bounds recomputation.
        if real not in self._named_lines:
            self._named_lines[real] = {"color": c, "line_width": lw,
                                       "label": None, "visible": True,
                                       "x": x.copy(), "y": y.copy()}
        else:
            self._named_lines[real]["color"] = c
            self._named_lines[real]["line_width"] = lw
            self._named_lines[real]["x"] = x.copy()
            self._named_lines[real]["y"] = y.copy()
        if label is not None:
            new_label = str(label)
            self._named_lines[real]["label"] = new_label
            if self._legend_visible:
                self._push_legend()
        # Recompute bounds from scratch so shrinking data doesn't leave stale extents.
        self._refit_all_static()
        self._mark_dirty()

    def remove_line(self, handle: int) -> None:
        """Remove a line by handle."""
        if self._renderer is None:
            self._pending_named_lines.pop(handle, None)
            self._refit_all_static()
            return
        real = self._resolve_line_handle(handle)
        if self._renderer is not None:
            self._renderer.chart2d_remove_line(real)
        self._named_lines.pop(real, None)
        self._refit_all_static()
        if self._legend_visible:
            self._push_legend()
        self._mark_dirty()

    # ── Reference overlays ────────────────────────────────────────────────────

    def axhspan(
        self,
        ymin: float,
        ymax: float,
        *,
        color: "tuple[float, float, float, float]" = (0.4, 0.6, 1.0, 0.20),
    ) -> int:
        """Add a filled horizontal band from *ymin* to *ymax* (data coordinates).

        The band spans the full x extent of the chart and is clipped by the plot
        scissor rect.  Returns an overlay handle for :meth:`remove_overlay`.

        Parameters
        ----------
        ymin, ymax : float
            Data-space y bounds of the band.
        color : (r, g, b, a) in [0, 1]
            Fill colour including alpha (e.g. 0.2 for 20 % opacity).
        """
        r, g, b, a = float(color[0]), float(color[1]), float(color[2]), float(color[3])
        vhandle = self._next_overlay_handle
        self._next_overlay_handle += 1
        args = (float(ymin), float(ymax), (r, g, b, a))
        self._overlay_meta[vhandle] = {"kind": "hspan", "args": args}
        if self._renderer is not None:
            rust_id = self._renderer.chart2d_add_hspan(*args)
            self._overlay_handle_map[vhandle] = rust_id
            self._mark_dirty()
        return vhandle

    def axvspan(
        self,
        xmin: float,
        xmax: float,
        *,
        color: "tuple[float, float, float, float]" = (0.4, 0.6, 1.0, 0.20),
    ) -> int:
        """Add a filled vertical band from *xmin* to *xmax* (data coordinates).

        The band spans the full y extent of the chart.
        Returns an overlay handle for :meth:`remove_overlay`.
        """
        r, g, b, a = float(color[0]), float(color[1]), float(color[2]), float(color[3])
        vhandle = self._next_overlay_handle
        self._next_overlay_handle += 1
        args = (float(xmin), float(xmax), (r, g, b, a))
        self._overlay_meta[vhandle] = {"kind": "vspan", "args": args}
        if self._renderer is not None:
            rust_id = self._renderer.chart2d_add_vspan(*args)
            self._overlay_handle_map[vhandle] = rust_id
            self._mark_dirty()
        return vhandle

    def axhline(
        self,
        y: float,
        *,
        color: "tuple[float, float, float]" = (0.8, 0.8, 0.3),
        line_width: float = 1.5,
    ) -> int:
        """Add an infinite horizontal reference line at data y = *y*.

        Returns an overlay handle for :meth:`remove_overlay`.
        """
        color = (float(color[0]), float(color[1]), float(color[2]))
        vhandle = self._next_overlay_handle
        self._next_overlay_handle += 1
        args = (float(y), color, float(line_width))
        self._overlay_meta[vhandle] = {"kind": "hline", "args": args}
        if self._renderer is not None:
            rust_id = self._renderer.chart2d_add_hline(*args)
            self._overlay_handle_map[vhandle] = rust_id
            self._mark_dirty()
        return vhandle

    def axvline(
        self,
        x: float,
        *,
        color: "tuple[float, float, float]" = (0.8, 0.8, 0.3),
        line_width: float = 1.5,
    ) -> int:
        """Add an infinite vertical reference line at data x = *x*.

        Returns an overlay handle for :meth:`remove_overlay`.
        """
        color = (float(color[0]), float(color[1]), float(color[2]))
        vhandle = self._next_overlay_handle
        self._next_overlay_handle += 1
        args = (float(x), color, float(line_width))
        self._overlay_meta[vhandle] = {"kind": "vline", "args": args}
        if self._renderer is not None:
            rust_id = self._renderer.chart2d_add_vline(*args)
            self._overlay_handle_map[vhandle] = rust_id
            self._mark_dirty()
        return vhandle

    def remove_overlay(self, handle: int) -> None:
        """Remove a reference overlay (span or line) by the handle from :meth:`axhspan` etc."""
        self._overlay_meta.pop(handle, None)
        rust_id = self._overlay_handle_map.pop(handle, None)
        if rust_id is not None and self._renderer is not None:
            self._renderer.chart2d_remove_overlay(rust_id)
            self._mark_dirty()

    def clear_chart_overlays(self) -> None:
        """Remove all reference spans and lines."""
        self._overlay_meta.clear()
        self._overlay_handle_map.clear()
        if self._renderer is not None:
            self._renderer.chart2d_clear_overlays()
            self._mark_dirty()

    # ── Axis formatting / scale ───────────────────────────────────────────────

    def set_x_tick_formatter(self, fmt: str) -> None:
        """Set the tick-label format for the X axis.

        Parameters
        ----------
        fmt : ``"default"`` | ``"sci"`` | ``"int"`` | ``"time"``
            * ``"default"`` — smart decimal / scientific notation
            * ``"sci"``     — always scientific (``1.23e4``)
            * ``"int"``     — integer (no decimal point)
            * ``"time"``    — seconds → ``MM:SS`` or ``H:MM:SS``
        """
        self._x_tick_format = fmt
        if self._renderer is not None:
            self._renderer.chart2d_set_tick_format("x", fmt)
            self._mark_dirty()

    def set_y_tick_formatter(self, fmt: str) -> None:
        """Set the tick-label format for the Y axis.

        Parameters
        ----------
        fmt : ``"default"`` | ``"sci"`` | ``"int"`` | ``"time"``
        """
        self._y_tick_format = fmt
        if self._renderer is not None:
            self._renderer.chart2d_set_tick_format("y", fmt)
            self._mark_dirty()

    def set_xscale(self, scale: str) -> None:
        """Set the X-axis scale.

        Parameters
        ----------
        scale : ``"linear"`` | ``"log"``
        """
        enabled = scale == "log"
        self._x_log_scale = enabled
        if self._renderer is not None:
            self._renderer.chart2d_set_log_scale("x", enabled)
            self._mark_dirty()

    def set_yscale(self, scale: str) -> None:
        """Set the Y-axis scale.

        Parameters
        ----------
        scale : ``"linear"`` | ``"log"``
        """
        enabled = scale == "log"
        self._y_log_scale = enabled
        if self._renderer is not None:
            self._renderer.chart2d_set_log_scale("y", enabled)
            self._mark_dirty()

    # ── Streaming API ─────────────────────────────────────────────────────────

    def add_line_stream(
        self,
        *,
        max_points: int,
        mode: str = "ring",
        color: "tuple[float, float, float]" = (0.3, 0.7, 1.0),
        line_width: float = 2.0,
        label: "str | None" = None,
    ) -> int:
        """Pre-allocate a streaming polyline buffer.

        Parameters
        ----------
        max_points : int
            Maximum number of *(x, y)* points the buffer can hold.
        mode : ``"ring"`` | ``"append"``
        color : (r, g, b) in [0, 1]
        line_width : float, optional
            Line width in screen pixels.
        label : str, optional
            Series label shown in the legend when :meth:`show_legend` is True.

        Returns
        -------
        int
            Handle for :meth:`stream_line`, :meth:`clear_line_stream`, and
            :meth:`remove_line_stream`.
        """
        if max_points < 2:
            raise ValueError("max_points must be >= 2")
        line_width = _normalize_line2d_width(line_width)
        sid = self._next_stream
        self._next_stream += 1
        self._line_streams[sid] = {
            "buf_x":        np.zeros(max_points, dtype=np.float32),
            "buf_y":        np.zeros(max_points, dtype=np.float32),
            # Pre-allocated reorder buffer — avoids heap allocation every frame
            # for wrapped ring buffers (avoids np.concatenate in _stream_ordered).
            "out_x":        np.empty(max_points, dtype=np.float32),
            "out_y":        np.empty(max_points, dtype=np.float32),
            "head":         0,
            "count":        0,
            "max_pts":      max_points,
            "mode":         mode,
            "color":        color,
            "line_width":   line_width,
            "label":        str(label) if label is not None else None,
            "render_handle": None,  # chart2d line handle, created on first upload
        }
        if self._legend_visible and label is not None:
            self._push_legend()
        return sid

    def stream_line(self, handle: int, x, y) -> None:
        """Append new *(x, y)* samples to a streaming line.

        Parameters
        ----------
        handle : int
            Handle returned by :meth:`add_line_stream`.
        x, y : scalar or array-like
        """
        st = self._line_streams.get(handle)
        if st is None:
            raise KeyError(f"No line stream with handle {handle!r}")

        x = np.asarray(x, dtype=np.float32).ravel()
        y = np.asarray(y, dtype=np.float32).ravel()
        if len(x) != len(y):
            raise ValueError(
                f"x and y must have equal length, got {len(x)} vs {len(y)}")
        n = len(x)
        if n == 0:
            return

        buf_x, buf_y, max_pts = st["buf_x"], st["buf_y"], st["max_pts"]
        if st["mode"] == "append":
            remaining = max_pts - st["count"]
            if remaining <= 0:
                return
            n = min(n, remaining)
            s = st["count"]
            buf_x[s:s + n] = x[:n]; buf_y[s:s + n] = y[:n]
            st["count"] += n
        else:  # ring
            head  = st["head"]
            space = max_pts - head
            if n <= space:
                buf_x[head:head + n] = x; buf_y[head:head + n] = y
                st["head"] = (head + n) % max_pts
            else:
                buf_x[head:] = x[:space]; buf_y[head:] = y[:space]
                overflow = min(n - space, max_pts)
                buf_x[:overflow] = x[space:space + overflow]
                buf_y[:overflow] = y[space:space + overflow]
                st["head"] = overflow
            st["count"] = min(st["count"] + n, max_pts)

        cnt = st["count"]
        if cnt < 2:
            return

        if not self._box_zoom_active:
            self._refit_from_all_sources()

        if self._renderer is None:
            return

        xs, ys = self._stream_ordered(st)
        if xs is None:
            return

        if st["render_handle"] is None:
            st["render_handle"] = self._renderer.chart2d_add_line(
                xs, ys, st["color"], st["line_width"]
            )
        else:
            self._renderer.chart2d_update_line(
                st["render_handle"], xs, ys, st["color"], st["line_width"]
            )
        self._mark_dirty()

    def _stream_ordered(self, st: dict):
        """Return (x_ordered, y_ordered) arrays oldest-to-newest, or (None, None).

        When the ring buffer has not yet wrapped, returns zero-copy slices of the
        underlying buffers.  When it has wrapped, writes into the pre-allocated
        ``out_x``/``out_y`` buffers (avoids per-frame heap allocation).
        """
        cnt = st["count"]
        if cnt < 2:
            return None, None
        buf_x, buf_y, max_pts = st["buf_x"], st["buf_y"], st["max_pts"]
        if cnt < max_pts:
            return buf_x[:cnt], buf_y[:cnt]  # zero-copy slice
        head = st["head"]
        out_x, out_y = st["out_x"], st["out_y"]
        tail = max_pts - head
        out_x[:tail] = buf_x[head:]
        out_x[tail:] = buf_x[:head]
        out_y[:tail] = buf_y[head:]
        out_y[tail:] = buf_y[:head]
        return out_x, out_y

    def clear_line_stream(self, handle: int) -> None:
        """Reset a streaming line to empty; pre-allocated buffer is kept."""
        st = self._line_streams.get(handle)
        if st is None:
            raise KeyError(f"No line stream with handle {handle!r}")
        st["count"] = 0; st["head"] = 0
        if st["render_handle"] is not None and self._renderer is not None:
            empty = np.array([], dtype=np.float32)
            self._renderer.chart2d_update_line(
                st["render_handle"], empty, empty, st["color"], st["line_width"]
            )
        self._refit_from_all_sources()
        self._mark_dirty()

    def remove_line_stream(self, handle: int) -> None:
        """Remove a streaming line actor and release its buffer."""
        st = self._line_streams.pop(handle, None)
        if st is None:
            raise KeyError(f"No line stream with handle {handle!r}")
        if st["render_handle"] is not None and self._renderer is not None:
            self._renderer.chart2d_remove_line(st["render_handle"])
        self._refit_from_all_sources()
        if self._legend_visible:
            self._push_legend()
        self._mark_dirty()

    # ── Mouse / camera overrides ──────────────────────────────────────────────

    def _drag_move(self, event: tk.Event, button: int) -> None:
        """No camera pan in chart mode — box zoom and selection drags pass through."""
        if self._renderer is None or self._drag_btn != button:
            return
        # Box zoom: show the drag rect as a selection rect overlay.
        if button == 1 and self._bz_dragging:
            self._renderer.set_selection_rect(
                float(self._bz_px0), float(self._bz_py0),
                float(event.x), float(event.y))
            self._mark_dirty()
            return
        if self._try_lasso_move(event):
            return
        if button == 1 and self._rect_active:
            self._renderer.set_selection_rect(
                float(self._sel_x0), float(self._sel_y0),
                float(event.x), float(event.y))
            self._mark_dirty()
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_scroll(self, event: tk.Event) -> None:
        """Ignore scroll; chart mode has fixed axis limits."""

    def _on_scroll_up_x11(self, _event: tk.Event) -> None:
        """Ignore scroll."""

    def _on_scroll_down_x11(self, _event: tk.Event) -> None:
        """Ignore scroll."""

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _push_chart2d(self) -> None:
        """Send current chart2d state to the Rust renderer."""
        if self._renderer is None:
            return
        # When no data has arrived yet (or was just cleared), use a neutral
        # (0, 1) range so the renderer shows a clean empty grid rather than
        # retaining the previous chart's axes/ticks.
        xlim = self._xlim if self._xlim is not None else (0.0, 1.0)
        ylim = self._ylim if self._ylim is not None else (0.0, 1.0)
        self._renderer.set_chart2d(
            self._pad_left, self._pad_right,
            self._pad_top,  self._pad_bottom,
            xlim[0], xlim[1],
            ylim[0], ylim[1],
            self._xlabel,   self._ylabel,
            self._y_tick_interval,
            self._title,
            self._x_tick_interval,
        )
        self._chart2d_sent = True
        self._mark_dirty()

    def _push_legend(self) -> None:
        """Recompute and send legend entries to the Rust renderer."""
        if self._renderer is None:
            return
        if not self._legend_visible:
            self._renderer.chart2d_set_legend([], self._legend_position, False)
            self._mark_dirty()
            return

        entries: "list[tuple[str, tuple[float, float, float]]]" = []

        # Primary line (set_line)
        if self._primary_label is not None and self._primary_handle is not None:
            entries.append((self._primary_label, self._primary_color))

        # Named extra lines (add_line)
        for meta in self._named_lines.values():
            lbl = meta.get("label")
            if lbl is not None:
                entries.append((lbl, meta["color"]))

        # Streaming lines (add_line_stream)
        for st in self._line_streams.values():
            lbl = st.get("label")
            if lbl is not None:
                entries.append((lbl, st["color"]))

        self._renderer.chart2d_set_legend(entries, self._legend_position, True)
        self._mark_dirty()

    def _sync_limit_freeze(self) -> None:
        self._limits_frozen = self._x_limits_frozen and self._y_limits_frozen

    def _apply_xlim(self, xmin: float, xmax: float, *, freeze: bool) -> None:
        self._xlim = (float(xmin), float(xmax))
        self._x_limits_frozen = freeze
        self._sync_limit_freeze()
        # Fast path: skip full set_chart2d (glyph reshape) when only xlim slides.
        if self._chart2d_sent and self._renderer is not None:
            self._renderer.chart2d_update_xlim(float(xmin), float(xmax))
            self._mark_dirty()
            return
        self._push_chart2d()

    def _apply_ylim(self, ymin: float, ymax: float, *, freeze: bool) -> None:
        self._ylim = (float(ymin), float(ymax))
        self._y_limits_frozen = freeze
        self._sync_limit_freeze()
        # Fast path for auto-y updates: keep the existing Y interval fixed.
        if self._chart2d_sent and self._renderer is not None and not freeze:
            self._renderer.chart2d_update_ylim(float(ymin), float(ymax))
            self._mark_dirty()
            return
        self._push_chart2d()


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
