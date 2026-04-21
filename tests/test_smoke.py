"""Smoke tests for dragonsci.

The entire module is skipped gracefully when the Rust extension has not been
built yet. To build it:

    maturin develop --release

Renderer tests additionally require a real display and skip automatically in
headless environments.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module if the extension is absent — gives a clear skip
# message rather than a confusing ImportError mid-collection.
pytest.importorskip(
    "dragonsci._dragonsci",
    reason="Rust extension not built — run: maturin develop --release",
)
import dragonsci
from dragonsci import Scatter3D, Scatter2D, link_cameras, unlink_cameras


def _scene_metadata(widget):
    if widget._renderer is None and widget._pending_point_meta is not None:
        return widget._pending_point_meta
    return {
        "columns": widget._scene_columns,
        "hover_data": widget._scene_hover,
        "row_positions": widget._scene_row_positions,
        "row_labels": widget._scene_row_labels,
        "legend_items": widget._scene_legend,
        "legend_title": widget._scene_legend_title,
    }


def _actor_metadata(widget, handle):
    if widget._renderer is None:
        return widget._pending_actor_meta.get(handle)
    return {
        "columns": widget._actor_columns.get(handle, {}),
        "hover_data": widget._actor_hover.get(handle, {}),
        "row_positions": widget._actor_row_positions.get(handle),
        "row_labels": widget._actor_row_labels.get(handle),
        "legend_items": widget._actor_legend.get(handle),
        "legend_title": widget._actor_legend_title.get(handle),
    }


# ── Headless-safe tests ───────────────────────────────────────────────────────

def test_import():
    assert hasattr(dragonsci, "Scatter3D")


def test_version():
    assert dragonsci.__version__ != "unknown"


def test_colormap_names():
    names = Scatter3D.colormap_names()
    assert isinstance(names, list)
    assert "viridis" in names
    assert "plasma" in names
    assert "turbo" in names


# ── Display-dependent fixture ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def root():
    """A real Tk root; skips the module if no display is available."""
    tk = pytest.importorskip("tkinter")
    try:
        r = tk.Tk()
        r.withdraw()
        yield r
        r.destroy()
    except tk.TclError as exc:
        pytest.skip(f"No display available: {exc}")


@pytest.fixture()
def widget(root):
    w = Scatter3D(root, width=200, height=200)
    w.pack()
    root.update_idletasks()
    yield w
    w.destroy()


# ── Widget smoke tests ────────────────────────────────────────────────────────

def test_widget_creates(widget):
    assert isinstance(widget, Scatter3D)


def test_set_points_basic(widget):
    pts = np.random.default_rng(0).standard_normal((1_000, 3)).astype(np.float32)
    widget.set_points(pts)


def test_set_points_empty(widget):
    widget.set_points(np.zeros((0, 3), dtype=np.float32))


def test_set_points_scalars(widget):
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((5_000, 3)).astype(np.float32)
    scalars = rng.random(5_000).astype(np.float32)
    widget.set_points(pts, scalars=scalars, colormap="plasma")


def test_set_points_colors(widget):
    rng = np.random.default_rng(2)
    pts = rng.standard_normal((5_000, 3)).astype(np.float32)
    colors = rng.random((5_000, 3)).astype(np.float32)
    widget.set_points(pts, colors=colors)


def test_set_points_pandas(widget):
    pd = pytest.importorskip("pandas")
    n = 1_000
    df = pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, n),
        "y": np.linspace(1.0, -1.0, n),
        "z": np.linspace(0.0, 2.0, n),
        "temp": np.linspace(10.0, 20.0, n),
        "label": [f"pt-{i}" for i in range(n)],
    })
    widget.set_points(df, x="x", y="y", z="z", color="temp", hover=["label"])
    meta = _scene_metadata(widget)
    assert meta["columns"]["x"] == "x"
    assert meta["columns"]["y"] == "y"
    assert meta["columns"]["z"] == "z"
    assert meta["columns"]["color"] == "temp"
    assert "label" in meta["hover_data"]
    assert meta["row_positions"] is not None
    assert meta["row_positions"].shape == (n,)
    assert meta["legend_items"] is None
    if widget._renderer is None:
        assert widget._pending is not None
        assert widget._pending["colors"] is None
        assert widget._pending["scalars"] is not None


def test_set_points_pandas_without_z(widget):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, 100),
        "y": np.linspace(1.0, -1.0, 100),
    })
    widget.set_points(df, x="x", y="y")
    meta = _scene_metadata(widget)
    assert meta["row_positions"] is not None
    assert meta["row_positions"].shape == (100,)


def test_set_points_pandas_categorical_color(widget):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, 100),
        "y": np.linspace(1.0, -1.0, 100),
        "species": ["a", "b"] * 50,
    })
    widget.set_points(df, x="x", y="y", color="species")
    meta = _scene_metadata(widget)
    assert meta["legend_title"] == "species"
    assert meta["legend_items"] is not None
    assert [label for label, _color in meta["legend_items"]] == ["a", "b"]
    if widget._renderer is None:
        assert widget._pending is not None
        assert widget._pending["colors"] is not None
        assert widget._pending["colors"].shape == (100, 3)
        assert widget._pending["scalars"] is None


def test_set_points_pandas_low_cardinality_integer_color_is_categorical(widget):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, 120),
        "y": np.linspace(1.0, -1.0, 120),
        "cluster": np.tile(np.array([0, 1, 2]), 40),
    })
    widget.set_points(df, x="x", y="y", color="cluster")
    meta = _scene_metadata(widget)
    assert meta["legend_items"] is not None
    assert [label for label, _color in meta["legend_items"]] == ["0", "1", "2"]


def test_show_legend_toggle(widget):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, 60),
        "y": np.linspace(1.0, -1.0, 60),
        "species": ["a", "b", "c"] * 20,
    })
    widget.set_points(df, x="x", y="y", color="species")
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no legend frame)")
    assert widget._legend_frame is not None
    assert widget._legend_frame.winfo_manager() == "place"
    widget.show_legend(False)
    assert widget._legend_visible is False
    assert widget._legend_frame.winfo_manager() == ""
    widget.show_legend(True)
    assert widget._legend_visible is True
    assert widget._legend_frame is not None
    assert widget._legend_frame.winfo_manager() == "place"
    widget.legend_position = "bottom-left"
    assert widget.legend_position == "bottom-left"


def test_set_points_polars(widget):
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({
        "x": np.linspace(-1.0, 1.0, 100),
        "y": np.linspace(1.0, -1.0, 100),
        "z": np.linspace(0.0, 2.0, 100),
    })
    widget.set_points(df, x="x", y="y", z="z")


def test_set_points_before_map():
    """set_points() before map must queue, not crash."""
    import tkinter as tk
    try:
        r = tk.Tk()
        r.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"No display: {exc}")
    w = Scatter3D(r, width=200, height=200)
    pts = np.random.default_rng(3).standard_normal((1_000, 3)).astype(np.float32)
    w.set_points(pts)
    r.destroy()


def test_set_points_pandas_before_map():
    """DataFrame input before map must queue the coerced data and metadata."""
    import tkinter as tk
    pd = pytest.importorskip("pandas")
    try:
        r = tk.Tk()
        r.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"No display: {exc}")
    w = Scatter3D(r, width=200, height=200)
    df = pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, 50),
        "y": np.linspace(1.0, -1.0, 50),
        "z": np.linspace(0.0, 2.0, 50),
    })
    w.set_points(df, x="x", y="y", z="z")
    assert w._pending is not None
    assert w._pending_point_meta is not None
    r.destroy()


def test_wrong_positions_shape(widget):
    with pytest.raises(Exception):
        widget.set_points(np.zeros((100, 2), dtype=np.float32))


def test_scalar_length_mismatch(widget):
    with pytest.raises(Exception):
        widget.set_points(np.zeros((100, 3), dtype=np.float32),
                          scalars=np.zeros(50, dtype=np.float32))


def test_reset_camera(widget):
    pts = np.random.default_rng(4).standard_normal((1_000, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.reset_camera()
    # Subsequent set_points() must NOT re-fit the camera (camera_fitted stays True).
    widget.set_points(pts)


def test_colormap_without_scalars(widget):
    """colormap= must be respected even when no scalars are provided (Z-default path)."""
    pts = np.random.default_rng(8).standard_normal((1_000, 3)).astype(np.float32)
    # Should not raise and should use the requested colormap, not always viridis.
    widget.set_points(pts, colormap="plasma")
    widget.set_points(pts, colormap="hot")


def test_all_colormaps(widget):
    pts = np.random.default_rng(5).standard_normal((1_000, 3)).astype(np.float32)
    scalars = pts[:, 2].copy()
    for name in Scatter3D.colormap_names():
        widget.set_points(pts, scalars=scalars, colormap=name)


def test_set_ticks_before_points(widget):
    """set_ticks() before any data must not crash."""
    widget.set_ticks(x=3, y=3, z=3)


def test_set_ticks_after_points(widget):
    """set_ticks() on a loaded dataset updates immediately (no upload needed)."""
    pts = np.random.default_rng(6).standard_normal((1_000, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.set_ticks(x=5, y=5, z=2)
    widget.set_ticks(z=None)   # restore one axis to auto


def test_set_ticks_reset_to_auto(widget):
    """Passing all-None restores full auto-scaling without error."""
    pts = np.random.default_rng(7).standard_normal((1_000, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.set_ticks(x=4)
    widget.set_ticks()   # all back to auto


def test_clim(widget):
    """clim= overrides the auto data range."""
    pts = np.random.default_rng(13).standard_normal((1_000, 3)).astype(np.float32)
    scalars = pts[:, 2].copy()
    widget.set_points(pts, scalars=scalars, colormap="plasma", clim=(-1.0, 1.0))


def test_log_scale(widget):
    """log_scale= must not crash (positive scalars)."""
    rng = np.random.default_rng(14)
    pts = rng.standard_normal((1_000, 3)).astype(np.float32)
    scalars = (rng.random(1_000) + 0.1).astype(np.float32)
    widget.set_points(pts, scalars=scalars, colormap="viridis", log_scale=True)


def test_nan_color(widget):
    """nan_color= is accepted; NaN scalars must not crash."""
    pts = np.random.default_rng(15).standard_normal((500, 3)).astype(np.float32)
    scalars = pts[:, 2].copy()
    scalars[::10] = float("nan")
    widget.set_points(pts, scalars=scalars, nan_color=(1.0, 0.0, 0.0))


def test_scalar_bar(widget):
    """scalar_bar() must not crash and must update visible state."""
    pts = np.random.default_rng(16).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts, scalars=pts[:, 2].copy(), colormap="viridis", clim=(-2.0, 2.0))
    widget.scalar_bar(True, vmin=-2.0, vmax=2.0, colormap="viridis", title="Z")
    if widget._renderer is not None:
        assert widget._renderer.inner.scalar_bar_visible if hasattr(widget._renderer, "inner") else True
    widget.scalar_bar(False)


def test_scalar_bar_before_renderer(root):
    """scalar_bar() before renderer init must queue state, not drop it."""
    w = Scatter3D(root, width=100, height=100)
    w.scalar_bar(True, vmin=-1.0, vmax=1.0, colormap="plasma", title="T")
    assert w._pending_scalar_bar is not None
    assert w._pending_scalar_bar["visible"] is True
    assert w._pending_scalar_bar["colormap"] == "plasma"
    assert w._pending_scalar_bar["title"] == "T"
    w.destroy()


def test_set_ticks_before_map():
    """set_ticks() before map must queue, not crash."""
    import tkinter as tk
    try:
        r = tk.Tk()
        r.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"No display: {exc}")
    w = Scatter3D(r, width=200, height=200)
    w.set_ticks(x=3, y=3, z=3)
    r.destroy()


# ── Multi-actor tests ─────────────────────────────────────────────────────────

def test_add_points_before_map_returns_virtual_handle(root):
    """add_points() before map must return a non-negative virtual handle and queue the call."""
    w = Scatter3D(root, width=200, height=200)
    pts = np.random.default_rng(90).standard_normal((200, 3)).astype(np.float32)
    h = w.add_points(pts)
    assert isinstance(h, int) and h >= 0
    assert len(w._pending_actors) == 1
    assert w._pending_actors[0][1] == h   # vhandle stored as second element
    w.destroy()


def test_add_points_pandas_before_map_tracks_metadata(root):
    pd = pytest.importorskip("pandas")
    w = Scatter3D(root, width=200, height=200)
    df = pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, 20),
        "y": np.linspace(1.0, -1.0, 20),
        "z": np.linspace(0.0, 2.0, 20),
        "label": [f"pt-{i}" for i in range(20)],
    })
    h = w.add_points(df, x="x", y="y", z="z", hover=["label"])
    meta = _actor_metadata(w, h)
    assert meta is not None
    assert meta["columns"]["x"] == "x"
    assert "label" in meta["hover_data"]
    w.destroy()


def test_add_points_pandas_tracks_metadata(widget):
    pd = pytest.importorskip("pandas")
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    df = pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, 25),
        "y": np.linspace(1.0, -1.0, 25),
        "z": np.linspace(0.0, 2.0, 25),
        "label": [f"pt-{i}" for i in range(25)],
    })
    h = widget.add_points(df, x="x", y="y", z="z", hover=["label"])
    meta = _actor_metadata(widget, h)
    assert meta is not None
    assert meta["columns"]["x"] == "x"
    assert "label" in meta["hover_data"]
    assert meta["row_positions"] is not None


def test_add_points_pandas_categorical_before_map_tracks_legend(root):
    pd = pytest.importorskip("pandas")
    w = Scatter3D(root, width=200, height=200)
    df = pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, 24),
        "y": np.linspace(1.0, -1.0, 24),
        "species": ["a", "b", "c"] * 8,
    })
    h = w.add_points(df, x="x", y="y", color="species")
    meta = _actor_metadata(w, h)
    assert meta is not None
    assert meta["legend_title"] == "species"
    assert meta["legend_items"] is not None
    assert [label for label, _color in meta["legend_items"]] == ["a", "b", "c"]
    queued = next(kwargs for kwargs, vhandle in w._pending_actors if vhandle == h)
    assert queued["colors"] is not None
    assert queued["colors"].shape == (24, 3)
    assert queued["scalars"] is None
    w.destroy()


def test_add_points_pandas_categorical_tracks_legend(widget):
    pd = pytest.importorskip("pandas")
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    df = pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, 18),
        "y": np.linspace(1.0, -1.0, 18),
        "species": ["a", "b"] * 9,
    })
    h = widget.add_points(df, x="x", y="y", color="species")
    meta = _actor_metadata(widget, h)
    assert meta is not None
    assert meta["legend_title"] == "species"
    assert meta["legend_items"] is not None


def test_remove_actor_hides_categorical_legend(widget):
    pd = pytest.importorskip("pandas")
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window â€” no Map event)")
    df = pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, 18),
        "y": np.linspace(1.0, -1.0, 18),
        "species": ["a", "b"] * 9,
    })
    h = widget.add_points(df, x="x", y="y", color="species")
    assert widget._legend_frame is not None
    assert widget._legend_frame.winfo_manager() == "place"
    assert h in widget._actor_legend
    widget.remove_actor(h)
    assert h not in widget._actor_legend
    assert widget._legend_frame is not None
    assert widget._legend_frame.winfo_manager() == ""


def test_add_points_multiple_before_map_distinct_handles(root):
    """Multiple pre-map add_points() calls must each get distinct virtual handles."""
    w = Scatter3D(root, width=200, height=200)
    pts = np.random.default_rng(91).standard_normal((100, 3)).astype(np.float32)
    h1 = w.add_points(pts)
    h2 = w.add_points(pts)
    assert h1 != h2
    assert len(w._pending_actors) == 2
    w.destroy()


def test_clear_resets_pending_actor_queue(root):
    """clear() before map must also empty the pending actor queue."""
    w = Scatter3D(root, width=200, height=200)
    pts = np.random.default_rng(92).standard_normal((100, 3)).astype(np.float32)
    w.add_points(pts)
    assert len(w._pending_actors) == 1
    w.clear()
    assert len(w._pending_actors) == 0
    assert len(w._phandle_map) == 0
    w.destroy()


def test_add_points_returns_handle(widget):
    """add_points() must return a non-negative integer handle."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(20).standard_normal((500, 3)).astype(np.float32)
    h = widget.add_points(pts)
    assert isinstance(h, int)
    assert h >= 0


def test_add_multiple_actors(widget):
    """Multiple actors must coexist without crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    rng = np.random.default_rng(21)
    h1 = widget.add_points(rng.standard_normal((500, 3)).astype(np.float32))
    h2 = widget.add_points(rng.standard_normal((500, 3)).astype(np.float32))
    assert h1 != h2


def test_update_actor(widget):
    """update_actor() must replace data without crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    rng = np.random.default_rng(22)
    h = widget.add_points(rng.standard_normal((500, 3)).astype(np.float32))
    widget.update_actor(h, rng.standard_normal((300, 3)).astype(np.float32))


def test_remove_actor(widget):
    """remove_actor() must not crash and must remove the actor."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(23).standard_normal((500, 3)).astype(np.float32)
    h = widget.add_points(pts)
    widget.remove_actor(h)


def test_actor_visibility(widget):
    """set_actor_visibility() must not crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(24).standard_normal((500, 3)).astype(np.float32)
    h = widget.add_points(pts)
    widget.set_actor_visibility(h, False)
    widget.set_actor_visibility(h, True)


def test_clear_actors(widget):
    """clear() must remove all actors without crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    rng = np.random.default_rng(25)
    widget.add_points(rng.standard_normal((200, 3)).astype(np.float32))
    widget.add_points(rng.standard_normal((200, 3)).astype(np.float32))
    widget.clear()


def test_add_points_before_map():
    """add_points() before map must return a virtual handle, not -1."""
    import tkinter as tk
    try:
        r = tk.Tk()
        r.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"No display: {exc}")
    w = Scatter3D(r, width=200, height=200)
    pts = np.random.default_rng(26).standard_normal((500, 3)).astype(np.float32)
    h = w.add_points(pts)
    assert isinstance(h, int) and h >= 0   # virtual handle, not -1
    r.destroy()


def test_set_points_after_add_points(widget):
    """set_points() after add_points() must replace, not accumulate."""
    rng = np.random.default_rng(27)
    widget.add_points(rng.standard_normal((200, 3)).astype(np.float32))
    widget.add_points(rng.standard_normal((200, 3)).astype(np.float32))
    # set_points should wipe the multi-actor state
    widget.set_points(rng.standard_normal((300, 3)).astype(np.float32))


# ── Export tests ─────────────────────────────────────────────────────────────

def test_screenshot_none_before_map():
    """screenshot() must return None when the widget has not been mapped."""
    import tkinter as tk
    try:
        r = tk.Tk()
        r.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"No display: {exc}")
    w = Scatter3D(r, width=200, height=200)
    assert w.screenshot() is None
    r.destroy()


def test_screenshot_returns_array(widget):
    """screenshot() must return a (H, W, 4) uint8 RGBA array and not be all-zero."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(40).standard_normal((200, 3)).astype(np.float32)
    widget.set_points(pts)
    img = widget.screenshot()
    assert img is not None
    assert img.ndim == 3 and img.shape[2] == 4
    assert img.dtype == np.uint8
    assert img.shape[:2] == (200, 200)  # matches widget dimensions
    # Image should not be entirely black — at least some pixels are non-zero
    assert img[..., :3].max() > 0


def test_save_png(widget, tmp_path):
    """save_png() must write a readable PNG file."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(41).standard_normal((200, 3)).astype(np.float32)
    widget.set_points(pts)
    out = tmp_path / "shot.png"
    widget.save_png(str(out))
    assert out.exists() and out.stat().st_size > 0
    # Verify it's a valid PNG (magic bytes)
    with open(out, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


# ── Picking / selection tests ─────────────────────────────────────────────────

def test_pick_point_returns_dict_or_none(widget):
    """pick_point() must not crash and return a dict or None."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(30).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts)
    result = widget._renderer.pick_point(100.0, 100.0)
    assert result is None or isinstance(result, dict)
    if result is not None:
        assert {"actor", "index", "point"} <= result.keys()


def test_pick_point_global_nearest_contract(widget):
    """pick_point must return the globally nearest point, even when the cursor
    is far from all points (exercises the fallback past the ±32 px local band).

    Regression test for the ScreenPickCache fast-path / fallback split:
    when ``best_dist_sq > R²`` after the local search the fallback must fire,
    not just when ``best.is_none()``.
    """
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    # Single point at origin.  After camera auto-fit it projects near screen
    # centre, so a cursor at (0, 0) is hundreds of pixels away — well outside
    # the ±32 px local band.  The fallback global scan must still find it.
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    widget.set_points(pts)
    result = widget._renderer.pick_point(0.0, 0.0)
    assert result is not None, (
        "pick_point must return the single visible point regardless of cursor distance"
    )
    assert result["index"] == 0, (
        f"pick_point must return index 0 (the only point); got {result}"
    )


def test_pick_rectangle_returns_list(widget):
    """pick_rectangle() must return a list of dicts."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(31).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts)
    hits = widget._renderer.pick_rectangle(0.0, 0.0, 200.0, 200.0)
    assert isinstance(hits, list)
    for h in hits:
        assert {"actor", "index"} <= h.keys()


def test_enable_point_picking(widget):
    """enable_point_picking() must not crash."""
    widget.enable_point_picking()
    assert widget._pick_mode in ("point", "both")


def test_enable_rectangle_picking(widget):
    """enable_rectangle_picking() must not crash."""
    widget.enable_rectangle_picking()
    assert widget._pick_mode in ("rect", "both")


def test_disable_picking(widget):
    """disable_picking() must restore mode to 'none'."""
    widget.enable_point_picking()
    widget.disable_picking()
    assert widget._pick_mode == "none"


def test_pick_empty_scene(widget):
    """Picking on an empty scene must return None / empty list."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    widget.set_points(np.zeros((0, 3), dtype=np.float32))
    assert widget._renderer.pick_point(100.0, 100.0) is None
    assert widget._renderer.pick_rectangle(0.0, 0.0, 200.0, 200.0) == []


def test_enable_lasso_picking(widget):
    """enable_lasso_picking() must set _lasso_enabled and not crash."""
    widget.enable_lasso_picking()
    assert widget._lasso_enabled is True


def test_disable_picking_clears_lasso(widget):
    """disable_picking() must clear lasso state."""
    widget.enable_lasso_picking()
    widget.disable_picking()
    assert widget._pick_mode == "none"
    assert widget._lasso_enabled is False
    assert widget._lasso_active is False


def test_pick_polygon_returns_list(widget):
    """pick_polygon() must return a list of dicts with actor/index keys."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(40).standard_normal((200, 3)).astype(np.float32)
    widget.set_points(pts)
    poly = [[0.0, 0.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0], [0.0, 0.0]]
    hits = widget._renderer.pick_polygon(poly)
    assert isinstance(hits, list)
    for h in hits:
        assert {"actor", "index"} <= h.keys()


def test_selected_indices_populated_after_lasso(widget):
    """_fire_selection must populate selected_indices from raw pick hits."""
    widget.set_points(np.random.default_rng(41).standard_normal((100, 3)).astype(np.float32))
    # When renderer is absent _scene_actor_handle is None; use raw-index fallback path.
    fake_hits = [{"actor": 0, "index": 0}, {"actor": 0, "index": 5}]
    widget._fire_selection([h["actor"] for h in fake_hits], [h["index"] for h in fake_hits])
    assert widget.selected_indices == [0, 5]
    assert widget.selected_index_values is None   # numpy array — no pandas labels


def test_selected_index_values_with_dataframe(root):
    """_translate_hits must return pandas index labels for DataFrame-backed scenes."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    w = Scatter3D(root, width=200, height=200)
    w.pack()
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {"x": rng.standard_normal(10).astype(np.float32),
         "y": rng.standard_normal(10).astype(np.float32),
         "z": rng.standard_normal(10).astype(np.float32)},
        index=[f"row_{i}" for i in range(10)],
    )
    w.set_points(df, x="x", y="y", z="z")
    # Even without a renderer the scene metadata is populated via _pending.
    # We can test _translate_hits directly: scene_actor_handle is None when
    # renderer is absent, so we go through the raw-index fallback.
    # Load with a mock renderer to exercise the full path.
    from unittest.mock import MagicMock
    mock_r = MagicMock()
    mock_r.set_points.return_value = 7   # fake handle
    w._renderer = mock_r
    w._init_renderer = lambda: None       # prevent re-init
    # Re-run set_points so _scene_actor_handle is set via the mock.
    w.set_points(df, x="x", y="y", z="z")
    assert w._scene_actor_handle == 7
    fake_hits = [{"actor": 7, "index": 2}, {"actor": 7, "index": 9}]
    indices, labels = w._translate_hits([h["actor"] for h in fake_hits], [h["index"] for h in fake_hits])
    assert indices == [2, 9]             # iloc positions (identity mapping)
    assert labels is not None
    assert list(labels) == ["row_2", "row_9"]


def test_deferred_set_points_preserves_scene_handle(root):
    """_on_map replay must preserve _scene_actor_handle through the metadata overwrite.

    Regression test for the bug where set_points(df) before the widget is mapped
    would lose _scene_actor_handle (and therefore pandas label translation) because
    _on_map called _set_scene_metadata(pending_meta) without passing actor_handle,
    resetting _scene_actor_handle back to None.
    """
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    from unittest.mock import MagicMock

    w = Scatter3D(root, width=200, height=200)
    w.pack()
    rng = np.random.default_rng(43)
    df = pd.DataFrame(
        {"x": rng.standard_normal(8).astype(np.float32),
         "y": rng.standard_normal(8).astype(np.float32),
         "z": rng.standard_normal(8).astype(np.float32)},
        index=[f"item_{i}" for i in range(8)],
    )

    # Call set_points while renderer is absent — goes into pending storage.
    assert w._renderer is None
    w.set_points(df, x="x", y="y", z="z")
    assert w._pending is not None, "expected deferred storage"
    assert w._pending_point_meta is not None

    # Inject a mock renderer (simulates what _init_renderer produces).
    mock_r = MagicMock()
    mock_r.set_points.return_value = 99   # fake actor handle
    w._renderer = mock_r
    w._init_renderer = lambda: None

    # Replay the deferred pending — mirrors exactly what _on_map does.
    pending = w._pending
    w._pending = None
    pending_meta = w._pending_point_meta
    w._pending_point_meta = None
    w.set_points(**pending)
    if pending_meta is not None:
        w._set_scene_metadata(pending_meta, actor_handle=w._scene_actor_handle)

    # The handle must survive the second _set_scene_metadata call.
    assert w._scene_actor_handle == 99, (
        "_scene_actor_handle was reset by the metadata overwrite"
    )
    fake_hits = [{"actor": 99, "index": 3}, {"actor": 99, "index": 7}]
    indices, labels = w._translate_hits([h["actor"] for h in fake_hits], [h["index"] for h in fake_hits])
    assert indices == [3, 7]
    assert labels is not None, "pandas labels lost after deferred replay"
    assert list(labels) == ["item_3", "item_7"]


# ── Overlay / line actor tests ───────────────────────────────────────────────

def test_add_lines_returns_handle(widget):
    """add_lines() must return a non-negative integer handle."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.array([[0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 1, 0]], dtype=np.float32)
    h = widget.add_lines(segs, color=(1.0, 0.0, 0.0))
    assert isinstance(h, int) and h >= 0


def test_add_multiple_overlays(widget):
    """Multiple line overlay actors must coexist."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.zeros((3, 6), dtype=np.float32)
    h1 = widget.add_lines(segs)
    h2 = widget.add_lines(segs, color=(0.0, 1.0, 0.0))
    assert h1 != h2


def test_update_lines(widget):
    """update_lines() must not crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.zeros((2, 6), dtype=np.float32)
    h = widget.add_lines(segs)
    new_segs = np.ones((4, 6), dtype=np.float32)
    widget.update_lines(h, new_segs, color=(0.5, 0.5, 0.5))


def test_overlay_visibility(widget):
    """set_overlay_visibility() must not crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.zeros((2, 6), dtype=np.float32)
    h = widget.add_lines(segs)
    widget.set_overlay_visibility(h, False)
    widget.set_overlay_visibility(h, True)


def test_remove_overlay(widget):
    """remove_overlay() must not crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.zeros((2, 6), dtype=np.float32)
    h = widget.add_lines(segs)
    widget.remove_overlay(h)


def test_clear_overlays(widget):
    """clear_overlays() must remove all line actors without crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.zeros((2, 6), dtype=np.float32)
    widget.add_lines(segs)
    widget.add_lines(segs)
    widget.clear_overlays()


def test_add_box(widget):
    """add_box() must return a valid handle."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    h = widget.add_box((-1, -1, -1, 1, 1, 1))
    assert isinstance(h, int) and h >= 0


def test_orientation_axes(widget):
    """show_orientation_axes() must update state on toggle."""
    widget.show_orientation_axes(True)
    assert widget._orientation_axes_visible is True
    widget.show_orientation_axes(False)
    assert widget._orientation_axes_visible is False


def test_orientation_axes_before_map(root):
    """show_orientation_axes() before map must persist to renderer on init."""
    w = Scatter3D(root, width=100, height=100)
    w.show_orientation_axes(True)
    assert w._orientation_axes_visible is True
    w.destroy()


def test_add_lines_before_map(root):
    """add_lines() before map must return a valid virtual handle and queue the overlay."""
    w = Scatter3D(root, width=200, height=200)
    segs = np.zeros((2, 6), dtype=np.float32)
    h = w.add_lines(segs)
    assert isinstance(h, int) and h >= 0
    assert len(w._pending_overlays) == 1
    assert w._pending_overlays[0][3] == h   # vhandle stored in queue
    w.destroy()


def test_add_box_before_map(root):
    """add_box() before map must queue via add_lines path."""
    w = Scatter3D(root, width=200, height=200)
    h = w.add_box((-1, -1, -1, 1, 1, 1))
    assert isinstance(h, int) and h >= 0
    assert len(w._pending_overlays) == 1
    w.destroy()


def test_multiple_overlays_before_map(root):
    """Multiple pre-map overlays must each get distinct virtual handles."""
    w = Scatter3D(root, width=200, height=200)
    segs = np.zeros((2, 6), dtype=np.float32)
    h1 = w.add_lines(segs)
    h2 = w.add_lines(segs, color=(0.0, 1.0, 0.0))
    assert h1 != h2
    assert len(w._pending_overlays) == 2
    w.destroy()


def test_clear_overlays_before_map(root):
    """clear_overlays() before map must empty the queue."""
    w = Scatter3D(root, width=200, height=200)
    segs = np.zeros((2, 6), dtype=np.float32)
    w.add_lines(segs)
    w.add_lines(segs)
    w.clear_overlays()
    assert len(w._pending_overlays) == 0
    w.destroy()


def test_clear_removes_overlays(widget):
    """clear() must remove line overlays, not just point actors."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    pts = np.random.default_rng(80).standard_normal((300, 3)).astype(np.float32)
    widget.set_points(pts)
    segs = np.array([[10.0, 20.0, 30.0, 11.0, 21.0, 31.0]], dtype=np.float32)
    widget.add_lines(segs)
    assert widget._renderer.actor_union_bounds() is not None
    widget.clear()
    # Both point actors and line overlays must be gone
    assert widget._renderer.actor_union_bounds() is None


def test_clear_resets_pending_overlay_queue(root):
    """clear() before map must also empty the pending overlay queue."""
    w = Scatter3D(root, width=200, height=200)
    segs = np.zeros((2, 6), dtype=np.float32)
    w.add_lines(segs)
    assert len(w._pending_overlays) == 1
    w.clear()
    assert len(w._pending_overlays) == 0
    w.destroy()


def test_wrong_segments_shape(widget):
    """add_lines() with wrong shape must raise."""
    with pytest.raises(Exception):
        widget.add_lines(np.zeros((3, 3), dtype=np.float32))


# ── Animation export tests ───────────────────────────────────────────────────

def test_write_frame_without_open(widget):
    """write_frame() before open_gif() must raise RuntimeError."""
    with pytest.raises(RuntimeError):
        widget.write_frame()


def test_close_gif_without_open(widget):
    """close_gif() without open_gif() must be a silent no-op."""
    widget.close_gif()   # must not raise


def test_open_write_close_no_renderer(root):
    """open/write/close on an unmapped widget must not crash."""
    w = Scatter3D(root, width=100, height=100)
    w.open_gif("/dev/null", fps=10)
    w.write_frame()  # screenshot returns None — frame silently skipped
    w.close_gif()
    w.destroy()


def test_orbit_gif_before_renderer(root):
    """orbit_gif() before map must raise RuntimeError (renderer absent)."""
    w = Scatter3D(root, width=100, height=100)
    with pytest.raises(RuntimeError):
        w.orbit_gif("irrelevant.gif")
    w.destroy()


def test_orbit_gif_produces_file(widget, tmp_path):
    """orbit_gif() must create a non-empty file."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    pts = np.random.default_rng(70).standard_normal((200, 3)).astype(np.float32)
    widget.set_points(pts)
    out = tmp_path / "orbit.gif"
    widget.orbit_gif(str(out), n_frames=4, fps=10)
    assert out.exists() and out.stat().st_size > 0
    # Verify GIF magic bytes
    with open(out, "rb") as f:
        assert f.read(6) == b"GIF89a"


def test_manual_gif_write(widget, tmp_path):
    """Manual open/write/close must produce a valid GIF."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    pts = np.random.default_rng(71).standard_normal((200, 3)).astype(np.float32)
    widget.set_points(pts)
    out = tmp_path / "manual.gif"
    widget.open_gif(str(out), fps=10, loop=0)
    for _ in range(3):
        widget.write_frame()
    widget.close_gif()
    assert out.exists() and out.stat().st_size > 0
    with open(out, "rb") as f:
        assert f.read(6) == b"GIF89a"


# ── Rendering mode tests ─────────────────────────────────────────────────────

def test_point_style_property(widget):
    """point_style round-trips for all valid values."""
    for style in ("circle", "square", "gaussian"):
        widget.point_style = style
        assert widget.point_style == style
    widget.point_style = "circle"   # restore default


def test_point_style_invalid(widget):
    """Unknown style must raise ValueError."""
    with pytest.raises(ValueError):
        widget.point_style = "blob"


def test_opacity(widget):
    """opacity= must not crash for values in [0, 1]."""
    pts = np.random.default_rng(60).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts, opacity=0.5)
    widget.set_points(pts, opacity=1.0)
    widget.set_points(pts, opacity=0.0)


def test_point_style_before_renderer(root):
    """Setting point_style before renderer is initialized must not crash."""
    w = Scatter3D(root, width=100, height=100)
    w.point_style = "gaussian"
    assert w.point_style == "gaussian"
    w.destroy()


# ── Linked-camera tests ───────────────────────────────────────────────────────

def test_link_cameras_populates_links(root):
    """link_cameras() must add cross-references to _camera_links on each widget."""
    w1 = Scatter3D(root, width=100, height=100)
    w2 = Scatter3D(root, width=100, height=100)
    assert w2 not in w1._camera_links
    link_cameras(w1, w2)
    assert w2 in w1._camera_links
    assert w1 in w2._camera_links
    w1.destroy()
    w2.destroy()


def test_link_cameras_three_way(root):
    """Three-way link must be fully connected."""
    w1 = Scatter3D(root, width=100, height=100)
    w2 = Scatter3D(root, width=100, height=100)
    w3 = Scatter3D(root, width=100, height=100)
    link_cameras(w1, w2, w3)
    assert w2 in w1._camera_links and w3 in w1._camera_links
    assert w1 in w2._camera_links and w3 in w2._camera_links
    assert w1 in w3._camera_links and w2 in w3._camera_links
    w1.destroy(); w2.destroy(); w3.destroy()


def test_unlink_cameras(root):
    """unlink_cameras() must remove cross-references."""
    w1 = Scatter3D(root, width=100, height=100)
    w2 = Scatter3D(root, width=100, height=100)
    link_cameras(w1, w2)
    unlink_cameras(w1, w2)
    assert w2 not in w1._camera_links
    assert w1 not in w2._camera_links
    w1.destroy()
    w2.destroy()


def test_link_cameras_no_crash_before_renderer(root):
    """link_cameras() and camera mutations must not crash when renderer is absent."""
    w1 = Scatter3D(root, width=100, height=100)
    w2 = Scatter3D(root, width=100, height=100)
    link_cameras(w1, w2)
    # Calling camera methods before renderer is initialized must be silent.
    w1.view_xy()
    w1.reset_camera()
    w1.destroy()
    w2.destroy()


def test_linked_camera_propagates(root):
    """Camera state must propagate from w1 to w2 and from w2 to w1 after link."""
    w1 = Scatter3D(root, width=200, height=200)
    w2 = Scatter3D(root, width=200, height=200)
    w1.pack()
    w2.pack()
    root.update_idletasks()
    if w1._renderer is None or w2._renderer is None:
        w1.destroy(); w2.destroy()
        pytest.skip("renderer not initialized")
    pts = np.random.default_rng(50).standard_normal((500, 3)).astype(np.float32)
    w1.set_points(pts)
    w2.set_points(pts)
    link_cameras(w1, w2)
    # w1 → w2 propagation
    w1.view_xy()
    s1, s2 = w1.get_camera(), w2.get_camera()
    assert abs(s1["pitch"] - s2["pitch"]) < 1e-4
    assert abs(s1["yaw"] - s2["yaw"]) < 1e-4
    # w2 → w1 propagation (reverse direction)
    w2.view_yz()
    s1, s2 = w1.get_camera(), w2.get_camera()
    assert abs(s1["pitch"] - s2["pitch"]) < 1e-4
    assert abs(s1["yaw"] - s2["yaw"]) < 1e-4
    w1.destroy()
    w2.destroy()


def test_overlay_bounds_included_in_union(widget):
    """add_lines() bounds must be included in actor_union_bounds()."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.array([[100.0, 200.0, 300.0, 101.0, 201.0, 301.0]], dtype=np.float32)
    widget.add_lines(segs, color=(1.0, 1.0, 0.0))
    bounds = widget._renderer.actor_union_bounds()
    assert bounds is not None
    bmin, bmax = bounds
    assert bmax[0] >= 101.0
    assert bmax[1] >= 201.0
    assert bmax[2] >= 301.0


def test_overlay_only_scene_gets_camera_fit(widget):
    """An overlay added to an empty scene must trigger camera fitting."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    widget.clear()
    segs = np.array([[100.0, 200.0, 300.0, 101.0, 201.0, 301.0]], dtype=np.float32)
    widget.add_lines(segs)
    cam_after = widget.get_camera()
    target = cam_after["target"]
    assert max(abs(t) for t in target) > 50.0


def test_hidden_overlay_excluded_from_bounds(widget):
    """Hidden overlays must not contribute to actor_union_bounds."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    widget.clear()
    segs = np.array([[100.0, 200.0, 300.0, 101.0, 201.0, 301.0]], dtype=np.float32)
    h = widget.add_lines(segs)
    assert widget._renderer.actor_union_bounds() is not None
    widget.set_overlay_visibility(h, False)
    # After hiding the only overlay, bounds should be empty
    assert widget._renderer.actor_union_bounds() is None


def test_hidden_actor_excluded_from_bounds(widget):
    """Hidden point actors must not contribute to actor_union_bounds."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    pts = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
    h = widget.add_points(pts)
    assert widget._renderer.actor_union_bounds() is not None
    widget.set_actor_visibility(h, False)
    assert widget._renderer.actor_union_bounds() is None
    # Restoring visibility brings bounds back
    widget.set_actor_visibility(h, True)
    assert widget._renderer.actor_union_bounds() is not None


def test_clear_last_overlay_resets_camera_fit(widget):
    """Clearing the last overlay must reset camera_fitted so the next add refits."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    widget.clear()
    segs1 = np.array([[100.0, 200.0, 300.0, 101.0, 201.0, 301.0]], dtype=np.float32)
    widget.add_lines(segs1)
    cam1 = widget.get_camera()

    # Clear all overlays — camera_fitted should reset
    widget.clear_overlays()
    assert widget._renderer.camera_fitted is False

    # Add a new overlay at a completely different location
    segs2 = np.array([[-500.0, -500.0, -500.0, -499.0, -499.0, -499.0]], dtype=np.float32)
    widget.add_lines(segs2)
    cam2 = widget.get_camera()
    # Camera target should now be near segs2, not segs1
    t1 = cam1["target"]
    t2 = cam2["target"]
    dist = sum((a - b) ** 2 for a, b in zip(t1, t2)) ** 0.5
    assert dist > 100.0, "Camera target should have moved after refit"


def test_remove_last_overlay_resets_camera_fit(widget):
    """Removing the last overlay must reset camera_fitted."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    widget.clear()
    segs = np.array([[50.0, 60.0, 70.0, 51.0, 61.0, 71.0]], dtype=np.float32)
    h = widget.add_lines(segs)
    widget.remove_overlay(h)
    assert widget._renderer.camera_fitted is False


# ── Camera preset tests ───────────────────────────────────────────────────────

def test_camera_presets(widget):
    """All view presets must run without error."""
    pts = np.random.default_rng(9).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.view_xy()
    widget.view_xz()
    widget.view_yz()
    widget.view_isometric()
    widget.reset_camera()


def test_parallel_projection(widget):
    """Toggling parallel projection must not crash and must round-trip."""
    pts = np.random.default_rng(10).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts)
    assert widget.parallel_projection is False
    widget.parallel_projection = True
    assert widget.parallel_projection is True
    widget.parallel_projection = False
    assert widget.parallel_projection is False


def test_fit_to_bounds(widget):
    """fit() with explicit bounds must not crash."""
    pts = np.random.default_rng(11).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.fit((-2, -2, -2, 2, 2, 2))
    widget.fit()   # re-fit to data


def test_get_set_camera(widget):
    """get_camera / set_camera must round-trip without error."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(12).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts)
    state = widget.get_camera()
    assert {"target", "distance", "yaw", "pitch", "parallel"} <= state.keys()
    widget.view_xy()
    widget.set_camera(state)   # restore original state


# ── set_axes / show_grid / set_background tests ───────────────────────────────

def test_show_grid_toggle(widget):
    """show_grid() must not crash when toggled."""
    pts = np.random.default_rng(70).standard_normal((300, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.show_grid(False)
    assert widget._grid_visible is False
    widget.show_grid(True)
    assert widget._grid_visible is True


def test_show_grid_before_renderer(root):
    """show_grid() before renderer is initialized must persist to renderer on map."""
    w = Scatter3D(root, width=100, height=100)
    w.show_grid(False)
    assert w._grid_visible is False
    w.destroy()


def test_set_background_tuple(widget):
    """set_background() with an (r, g, b) tuple must not crash."""
    pts = np.random.default_rng(71).standard_normal((300, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.set_background((0.1, 0.1, 0.2))
    assert widget._bg_color == (0.1, 0.1, 0.2)


def test_set_background_hex(widget):
    """set_background() with a hex string must parse and apply correctly."""
    pts = np.random.default_rng(72).standard_normal((300, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.set_background("#0d0d12")
    r, g, b = widget._bg_color
    assert abs(r - 0x0d / 255) < 1e-4
    assert abs(g - 0x0d / 255) < 1e-4
    assert abs(b - 0x12 / 255) < 1e-4


def test_set_background_invalid(widget):
    """set_background() with a bad string must raise ValueError."""
    with pytest.raises(ValueError):
        widget.set_background("not-a-color")


def test_set_background_before_renderer(root):
    """set_background() before renderer init must persist state."""
    w = Scatter3D(root, width=100, height=100)
    w.set_background((0.5, 0.5, 0.5))
    assert w._bg_color == (0.5, 0.5, 0.5)
    w.destroy()


def test_set_axes(widget):
    """set_axes() must accept label strings and persist state."""
    pts = np.random.default_rng(73).standard_normal((300, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.set_axes("Time", "Amplitude", "Phase")
    assert widget._axis_labels == ("Time", "Amplitude", "Phase")


def test_set_axes_empty_labels(widget):
    """set_axes() with empty strings must not crash."""
    pts = np.random.default_rng(74).standard_normal((300, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.set_axes("", "", "")
    assert widget._axis_labels == ("", "", "")


def test_set_axes_before_renderer(root):
    """set_axes() before renderer init must persist state."""
    w = Scatter3D(root, width=100, height=100)
    w.set_axes("A", "B", "C")
    assert w._axis_labels == ("A", "B", "C")
    w.destroy()


# ── Scatter2D smoke tests ─────────────────────────────────────────────────────

@pytest.fixture()
def widget2d(root):
    w = Scatter2D(root, width=200, height=200)
    w.pack()
    root.update_idletasks()
    yield w
    w.destroy()


def test_scatter2d_creates(widget2d):
    """Scatter2D must instantiate without crash."""
    assert isinstance(widget2d, Scatter2D)


def test_scatter2d_parallel_projection_always_true(widget2d):
    """parallel_projection must always be True and the setter must be a no-op."""
    assert widget2d.parallel_projection is True
    widget2d.parallel_projection = False
    assert widget2d.parallel_projection is True


def test_scatter2d_set_points_numpy_zeroes_z(widget2d):
    """set_points with a raw (N,3) array must zero the z column."""
    rng = np.random.default_rng(200)
    pts = rng.standard_normal((500, 3)).astype(np.float32)
    widget2d.set_points(pts)
    # _pending["positions"] holds the buffered array when renderer is not yet up
    pending = widget2d._pending
    if pending is not None:
        assert np.all(pending["positions"][:, 2] == 0.0)


def test_scatter2d_set_points_pandas_no_z(widget2d):
    """set_points with a DataFrame x/y must not crash and must not need z."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(201)
    df = pd.DataFrame({
        "x": rng.standard_normal(200).astype(np.float32),
        "y": rng.standard_normal(200).astype(np.float32),
    })
    widget2d.set_points(df, x="x", y="y")


def test_scatter2d_set_points_pandas_z_ignored(widget2d):
    """set_points with a DataFrame z= column must zero z in the pending buffer."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(202)
    df = pd.DataFrame({
        "x": rng.standard_normal(200).astype(np.float32),
        "y": rng.standard_normal(200).astype(np.float32),
        "z": rng.standard_normal(200).astype(np.float32),
    })
    widget2d.set_points(df, x="x", y="y", z="z")
    pending = widget2d._pending
    if pending is not None:
        assert np.all(pending["positions"][:, 2] == 0.0)


def test_scatter2d_update_actor_zeroes_z(root):
    """Scatter2D.update_actor must zero z before handing off to Scatter3D."""
    import unittest.mock as mock
    w = Scatter2D(root, width=200, height=200)
    rng = np.random.default_rng(203)
    pts = rng.standard_normal((300, 3)).astype(np.float32)
    h = w.add_points(pts)

    new_pts = rng.standard_normal((200, 3)).astype(np.float32)
    new_pts[:, 2] = 99.0  # deliberately non-zero

    captured = {}
    _orig = Scatter3D.update_actor

    def _spy(self_inner, handle, positions, **kwargs):
        captured["positions"] = positions.copy()
        _orig(self_inner, handle, positions, **kwargs)

    with mock.patch.object(Scatter3D, "update_actor", _spy):
        w.update_actor(h, new_pts)

    assert "positions" in captured, "Scatter3D.update_actor was never called"
    assert np.all(captured["positions"][:, 2] == 0.0), "z was not zeroed before delegation"
    w.destroy()


def test_scatter2d_all_view_presets_are_noops(root):
    """All four view-preset methods must be no-ops in Scatter2D.

    Uses a mock renderer so the test is not gated on a live display.
    Each method must not call *any* camera-changing method on the renderer.
    """
    import unittest.mock as mock
    w = Scatter2D(root, width=200, height=200)
    mock_r = mock.MagicMock()
    w._renderer = mock_r

    camera_methods = ("view_xy", "view_xz", "view_yz", "view_isometric",
                      "set_view_direction")

    for preset in ("view_xy", "view_xz", "view_yz", "view_isometric"):
        mock_r.reset_mock()
        getattr(w, preset)()
        for cam_method in camera_methods:
            assert not getattr(mock_r, cam_method).called, (
                f"Scatter2D.{preset}() called renderer.{cam_method}() — "
                f"must be a complete no-op"
            )

    w.destroy()


def test_scatter2d_set_camera_snaps_back(root):
    """set_camera() must re-lock to the front view and re-enable parallel projection."""
    import unittest.mock as mock
    w = Scatter2D(root, width=200, height=200)
    mock_r = mock.MagicMock()
    w._renderer = mock_r

    w.set_camera({"target": [0, 0, 0], "distance": 5.0,
                  "yaw": 0.9, "pitch": 0.9, "parallel": False})

    mock_r.set_camera.assert_called_once()
    mock_r.view_xz.assert_called_once()
    mock_r.set_parallel_projection.assert_called_once_with(True)

    w.destroy()


def test_scatter2d_axis_labels(widget2d):
    """Scatter2D must default to ('X', 'Y', '') axis labels."""
    assert widget2d._axis_labels == ("X", "Y", "")


# ── Size-by-column tests ──────────────────────────────────────────────────────

def test_set_points_point_sizes_array(widget):
    """point_sizes= numpy array produces per-point sizes without crash."""
    rng = np.random.default_rng(301)
    pts = rng.standard_normal((500, 3)).astype(np.float32)
    sizes = rng.uniform(2.0, 12.0, 500).astype(np.float32)
    widget.set_points(pts, point_sizes=sizes)


def test_set_points_point_sizes_length_mismatch(widget):
    """point_sizes= with wrong length must raise."""
    with pytest.raises(Exception):
        widget.set_points(
            np.zeros((100, 3), dtype=np.float32),
            point_sizes=np.ones(50, dtype=np.float32),
        )


def test_set_points_pandas_size_column(widget):
    """size= column in DataFrame maps to per-point sizes."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(302)
    n = 500
    df = pd.DataFrame({
        "x": rng.standard_normal(n).astype(float),
        "y": rng.standard_normal(n).astype(float),
        "z": rng.standard_normal(n).astype(float),
        "magnitude": np.linspace(1.0, 10.0, n),
    })
    widget.set_points(df, x="x", y="y", z="z", size="magnitude")
    if widget._renderer is None:
        assert widget._pending is not None
        assert widget._pending["point_sizes"] is not None
        assert widget._pending["point_sizes"].shape == (n,)


def test_set_points_pandas_size_with_nan(widget):
    """NaN values in the size column fall back to the fallback scalar."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(303)
    n = 100
    magnitudes = np.linspace(1.0, 10.0, n)
    magnitudes[::5] = float("nan")
    df = pd.DataFrame({
        "x": rng.standard_normal(n).astype(float),
        "y": rng.standard_normal(n).astype(float),
        "magnitude": magnitudes,
    })
    widget.set_points(df, x="x", y="y", size="magnitude")


def test_normalize_sizes_basic():
    """_normalize_sizes must map values linearly into [min_px, max_px]."""
    values = [0.0, 5.0, 10.0]
    result = Scatter3D._normalize_sizes(values, min_px=2.0, max_px=12.0)
    assert result.dtype == np.float32
    assert abs(float(result[0]) - 2.0) < 1e-4
    assert abs(float(result[1]) - 7.0) < 1e-4
    assert abs(float(result[2]) - 12.0) < 1e-4


def test_normalize_sizes_all_equal():
    """_normalize_sizes with constant input must return the midpoint."""
    result = Scatter3D._normalize_sizes([5.0, 5.0, 5.0], min_px=2.0, max_px=12.0)
    assert abs(float(result[0]) - 7.0) < 1e-4


def test_normalize_sizes_nan_fallback():
    """NaN values must map to the fallback scalar."""
    values = [1.0, float("nan"), 10.0]
    result = Scatter3D._normalize_sizes(values, min_px=2.0, max_px=12.0, fallback=4.0)
    assert np.isfinite(result[1])
    assert abs(float(result[1]) - 4.0) < 1e-4


def test_normalize_sizes_negative_range_raises():
    """Negative min_px or max_px must raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        Scatter3D._normalize_sizes([1.0, 2.0], min_px=-1.0, max_px=10.0)
    with pytest.raises(ValueError, match="non-negative"):
        Scatter3D._normalize_sizes([1.0, 2.0], min_px=1.0, max_px=-5.0)


def test_set_points_point_sizes_clamps_nonfinite(widget):
    """Non-finite values in point_sizes= must be clamped to 0, not crash."""
    rng = np.random.default_rng(304)
    pts = rng.standard_normal((10, 3)).astype(np.float32)
    sizes = np.array([4.0, float("nan"), float("inf"), float("-inf"),
                      -3.0, 2.0, 5.0, 1.0, 3.0, 6.0], dtype=np.float32)
    widget.set_points(pts, point_sizes=sizes)


def test_set_points_pandas_point_sizes_used_without_size_col(widget):
    """point_sizes= array must work for DataFrame input when size= is absent."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(305)
    n = 50
    df = pd.DataFrame({
        "x": rng.standard_normal(n).astype(float),
        "y": rng.standard_normal(n).astype(float),
    })
    sizes = np.linspace(2.0, 10.0, n).astype(np.float32)
    widget.set_points(df, x="x", y="y", point_sizes=sizes)
    if widget._renderer is None:
        assert widget._pending is not None
        assert widget._pending["point_sizes"] is not None


def test_set_points_pandas_size_and_point_sizes_raises(widget):
    """Providing both size= and point_sizes= for a DataFrame must raise ValueError."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(306)
    n = 30
    df = pd.DataFrame({
        "x": rng.standard_normal(n).astype(float),
        "y": rng.standard_normal(n).astype(float),
        "mag": np.linspace(1.0, 5.0, n),
    })
    with pytest.raises(ValueError, match="mutually exclusive"):
        widget.set_points(df, x="x", y="y", size="mag",
                          point_sizes=np.ones(n, dtype=np.float32))


def test_update_actor_point_sizes(widget):
    """update_actor(point_sizes=...) must forward sizes without crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    rng = np.random.default_rng(307)
    h = widget.add_points(rng.standard_normal((200, 3)).astype(np.float32))
    new_pts = rng.standard_normal((150, 3)).astype(np.float32)
    sizes = np.linspace(2.0, 8.0, 150).astype(np.float32)
    widget.update_actor(h, new_pts, point_sizes=sizes)


def test_update_actor_point_sizes_length_mismatch(widget):
    """update_actor with wrong-length point_sizes must raise ValueError in Python."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    h = widget.add_points(np.zeros((50, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="point_sizes length"):
        widget.update_actor(h, np.zeros((50, 3), dtype=np.float32),
                            point_sizes=np.ones(30, dtype=np.float32))
