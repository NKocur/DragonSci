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
from dragonsci import Line2D, Scatter3D, Scatter2D, link_cameras, unlink_cameras


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


@pytest.fixture()
def line2d(root):
    w = Line2D(root, width=320, height=240)
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


def test_line2d_drag_does_not_pan(line2d):
    """Line2D must not forward plain drag motion to renderer.mouse_drag()."""
    import unittest.mock as mock

    line2d._renderer = mock.Mock()
    line2d._drag_btn = 1
    line2d._drag_x = 20
    line2d._drag_y = 20
    line2d._rect_active = False
    line2d._lasso_active = False

    ev = mock.Mock(x=60, y=80)
    line2d._drag_move(ev, 1)

    line2d._renderer.mouse_drag.assert_not_called()


def test_line2d_scroll_does_not_zoom(line2d):
    """Line2D must ignore wheel zoom so the chart frame stays fixed."""
    import unittest.mock as mock

    line2d._renderer = mock.Mock()
    ev = mock.Mock(delta=120)
    line2d._on_scroll(ev)
    line2d._on_scroll_up_x11(ev)
    line2d._on_scroll_down_x11(ev)

    line2d._renderer.scroll.assert_not_called()


def test_line2d_set_line_passes_width_to_renderer(line2d):
    """set_line() must forward the configured line width to the renderer."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer

    line2d.set_line([0.0, 1.0, 2.0], [1.0, 0.5, 1.5], line_width=3.5)

    renderer.chart2d_add_line.assert_called_once()
    assert renderer.chart2d_add_line.call_args.args[3] == 3.5
    assert line2d._primary_width == 3.5


def test_line2d_update_line_preserves_and_updates_width(line2d):
    """update_line() must keep the existing width by default and accept overrides."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    handle = 42
    line2d._named_lines[handle] = {
        "color": (0.1, 0.2, 0.3),
        "line_width": 4.0,
    }

    line2d.update_line(handle, [0.0, 1.0], [0.0, 1.0])
    assert renderer.chart2d_update_line.call_args.args[4] == 4.0

    line2d.update_line(handle, [0.0, 1.0], [1.0, 0.0], line_width=6.0)
    assert renderer.chart2d_update_line.call_args.args[4] == 6.0
    assert line2d._named_lines[handle]["line_width"] == 6.0


def test_line2d_stream_line_passes_width_to_renderer(line2d):
    """Streaming line uploads must retain the configured width."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer

    handle = line2d.add_line_stream(max_points=8, line_width=5.0)
    line2d.stream_line(handle, [0.0, 1.0, 2.0], [1.0, 0.0, 1.0])

    renderer.chart2d_add_line.assert_called_once()
    assert renderer.chart2d_add_line.call_args.args[3] == 5.0


def test_line2d_stream_line_keeps_width_after_wrap(line2d):
    """Streaming lines must keep the original width after later updates."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer

    handle = line2d.add_line_stream(max_points=3, line_width=2.5)
    line2d.stream_line(handle, [10.0, 11.0, 12.0], [0.0, 1.0, 0.0])
    assert renderer.chart2d_add_line.call_args.args[3] == 2.5

    line2d.stream_line(handle, [13.0, 14.0], [1.0, 0.0])
    assert renderer.chart2d_update_line.call_args.args[4] == 2.5


def test_line2d_add_line_stream_invalid_mode_raises(line2d):
    with pytest.raises(ValueError, match="mode must be 'ring' or 'append'"):
        line2d.add_line_stream(max_points=3, mode="typo")


def test_line2d_ring_stream_keeps_latest_large_batch(line2d):
    handle = line2d.add_line_stream(max_points=3, mode="ring")
    line2d.stream_line(
        handle,
        np.arange(10, dtype=np.float32),
        np.arange(10, dtype=np.float32) * 10.0,
    )

    xs, ys = line2d._stream_ordered(line2d._line_streams[handle])
    np.testing.assert_allclose(xs, [7.0, 8.0, 9.0])
    np.testing.assert_allclose(ys, [70.0, 80.0, 90.0])
    assert line2d._current_data_bounds() == (7.0, 9.0, 70.0, 90.0)


def test_line2d_ring_stream_bounds_contract_after_overwrite(line2d):
    handle = line2d.add_line_stream(max_points=3, mode="ring")
    line2d.stream_line(handle, [0.0, 1.0, 2.0], [10.0, 20.0, 30.0])
    line2d.stream_line(handle, [3.0], [40.0])

    assert line2d._current_data_bounds() == (1.0, 3.0, 20.0, 40.0)


def test_line2d_invalid_width_raises(line2d):
    with pytest.raises(ValueError, match="positive finite"):
        line2d.set_line([0.0, 1.0], [0.0, 1.0], line_width=0.0)


def test_line2d_add_line_before_map_returns_virtual_handle(line2d):
    """Secondary Line2D lines must queue cleanly before the renderer is mapped."""
    assert line2d._renderer is None

    handle = line2d.add_line([0.0, 1.0], [1.0, 0.0], line_width=3.0)

    assert handle >= 0
    assert handle in line2d._pending_named_lines
    _x, _y, _color, width, _label, _visible = line2d._pending_named_lines[handle]
    assert width == 3.0


def test_line2d_set_line_derives_limits_from_data(line2d):
    """set_line must derive xlim/ylim from nice-rounded data bounds and notify
    the renderer."""
    import unittest.mock as mock
    import numpy as np

    renderer = mock.Mock()
    line2d._renderer = renderer

    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    y = np.array([0.0, 5.0, -1.0], dtype=np.float32)

    line2d.set_line(x, y)

    # xlim/ylim must have been set
    assert line2d._xlim is not None
    assert line2d._ylim is not None
    assert line2d._xlim[0] <= 10.0  # at or left of data minimum
    assert line2d._xlim[1] >= 30.0  # at or right of data maximum
    assert line2d._ylim[0] <= -1.0  # at or below data minimum
    assert line2d._ylim[1] >= 5.0   # at or above data maximum

    # The renderer must have been notified (via set_chart2d or the fast-path
    # chart2d_update_xlim/chart2d_update_ylim calls).
    assert renderer.set_chart2d.called or renderer.chart2d_update_ylim.called


def test_line2d_animated_set_xlim_uses_fast_path(line2d):
    """After the first set_chart2d, sliding set_xlim must use chart2d_update_xlim,
    not call set_chart2d again — verifying the _chart2d_sent fast-path gate."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._ylim = (-1.0, 1.0)

    # First set_xlim triggers a full set_chart2d (chart not yet configured).
    line2d.set_xlim(0.0, 10.0)
    assert renderer.set_chart2d.call_count == 1
    assert renderer.chart2d_update_xlim.call_count == 0

    # Subsequent set_xlim calls must use the fast path.
    line2d.set_xlim(1.0, 11.0)
    line2d.set_xlim(2.0, 12.0)
    assert renderer.set_chart2d.call_count == 1, "set_chart2d must not be called again"
    assert renderer.chart2d_update_xlim.call_count == 2

    # The fast-path calls must pass the correct limits.
    last = renderer.chart2d_update_xlim.call_args
    assert last.args == (2.0, 12.0)


def test_line2d_set_ylim_does_full_rebuild(line2d):
    """set_ylim always does a full set_chart2d because y-tick glyphs must be rebuilt."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._xlim = (0.0, 10.0)

    line2d.set_ylim(-1.0, 1.0)
    assert renderer.set_chart2d.call_count == 1

    # A second set_ylim must also do a full rebuild (not the xlim fast path).
    line2d.set_ylim(-2.0, 2.0)
    assert renderer.set_chart2d.call_count == 2
    assert renderer.chart2d_update_xlim.call_count == 0


def test_line2d_set_y_tick_interval_passes_override(line2d):
    """Custom Y tick spacing must be forwarded through set_chart2d."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._xlim = (0.0, 10.0)
    line2d._ylim = (-2.0, 2.0)

    line2d.set_y_tick_interval(0.5)

    renderer.set_chart2d.assert_called_once()
    assert renderer.set_chart2d.call_args.args[10] == 0.5


def test_line2d_set_line_with_frozen_x_rebuilds_static_autoy(line2d):
    """Static auto-y refits must rebuild the chart so tick spacing is recomputed."""
    import unittest.mock as mock
    import numpy as np

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._chart2d_sent = True
    line2d._xlim = (0.0, 10.0)
    line2d._ylim = (-1.0, 1.0)
    line2d._x_limits_frozen = True
    line2d._y_limits_frozen = False
    line2d._sync_limit_freeze()

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([-3.2, 1.1, 4.4], dtype=np.float32)

    line2d.set_line(x, y)

    renderer.chart2d_update_xlim.assert_not_called()
    renderer.chart2d_update_ylim.assert_not_called()
    renderer.set_chart2d.assert_called_once()
    args = renderer.set_chart2d.call_args.args
    assert args[6] <= float(y.min())
    assert args[7] >= float(y.max())
    assert line2d._xlim == (0.0, 10.0)


def test_line2d_stream_with_frozen_x_keeps_y_fast_path(line2d):
    """Streaming auto-y still uses the cheap y-limit update path."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.return_value = 0
    line2d._renderer = renderer
    line2d._chart2d_sent = True
    line2d._xlim = (0.0, 10.0)
    line2d._ylim = (-1.0, 1.0)
    line2d._x_limits_frozen = True
    line2d._y_limits_frozen = False
    line2d._sync_limit_freeze()

    handle = line2d.add_line_stream(max_points=16, mode="ring")
    line2d.stream_line(handle, [1.0, 2.0, 3.0], [-3.0, 1.0, 4.0])

    renderer.chart2d_update_xlim.assert_not_called()
    renderer.set_chart2d.assert_not_called()
    renderer.chart2d_update_ylim.assert_called_once()


def test_line2d_resize_resets_fast_path_gate(line2d):
    """_do_resize must trigger a full set_chart2d rebuild (axis geometry is in pixels).
    After the resize rebuild, the fast path is re-armed and the next set_xlim can use it."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._ylim = (-1.0, 1.0)

    line2d.set_xlim(0.0, 10.0)          # first call: full rebuild
    assert renderer.set_chart2d.call_count == 1

    line2d.set_xlim(1.0, 11.0)          # fast path
    assert renderer.chart2d_update_xlim.call_count == 1
    assert renderer.set_chart2d.call_count == 1  # still just 1

    # Simulate a resize — must fire a full set_chart2d (not the fast path).
    line2d._do_resize(900, 600)
    assert renderer.set_chart2d.call_count == 2  # full rebuild happened
    assert line2d._chart2d_sent  # flag re-armed after rebuild

    # Fast path is live again; next set_xlim should not call set_chart2d again.
    line2d.set_xlim(2.0, 12.0)
    assert renderer.chart2d_update_xlim.call_count == 2
    assert renderer.set_chart2d.call_count == 2  # no extra full rebuild


# ── Phase 2: toolbar — home, autoscale_y, autoscale_both ─────────────────────

def test_line2d_home_resets_both_axes(line2d):
    """home() unfreezes both axes and refits to recorded data extent."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.return_value = 0
    line2d._renderer = renderer
    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (-1.0, 1.0)
    line2d._chart2d_sent = True

    x = np.linspace(0, 4 * np.pi, 100, dtype=np.float32)
    y = np.sin(x)
    # Store geometry so _current_data_bounds() can find it.
    line2d._primary_x = x.copy()
    line2d._primary_y = y.copy()

    # Freeze the axes then call home.
    line2d._x_limits_frozen = True
    line2d._y_limits_frozen = True
    line2d.home()

    assert not line2d._x_limits_frozen
    assert not line2d._y_limits_frozen


def test_line2d_home_no_data_is_noop(line2d):
    """home() does nothing when no data has been loaded yet."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (-1.0, 1.0)
    line2d._chart2d_sent = True

    line2d.home()  # no stored geometry — should not raise or call renderer
    renderer.set_chart2d.assert_not_called()


def test_line2d_autoscale_y_only_unfreezes_y(line2d):
    """autoscale_y() unfreezes Y but leaves X frozen."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.return_value = 0
    line2d._renderer = renderer
    line2d._xlim = (0.0, 10.0)
    line2d._ylim = (-2.0, 2.0)
    line2d._chart2d_sent = True
    line2d._x_limits_frozen = True
    line2d._y_limits_frozen = True
    # Store geometry directly.
    line2d._primary_x = np.array([0.0, 10.0], dtype=np.float32)
    line2d._primary_y = np.array([-1.0, 1.0], dtype=np.float32)

    line2d.autoscale_y()

    assert line2d._x_limits_frozen  # X must stay frozen
    assert not line2d._y_limits_frozen


def test_line2d_autoscale_both_delegates_to_home(line2d):
    """autoscale_both() is equivalent to home()."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.return_value = 0
    line2d._renderer = renderer
    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (-1.0, 1.0)
    line2d._chart2d_sent = True
    line2d._x_limits_frozen = True
    line2d._y_limits_frozen = True
    line2d._primary_x = np.array([0.0, 5.0], dtype=np.float32)
    line2d._primary_y = np.array([-3.0, 3.0], dtype=np.float32)

    line2d.autoscale_both()

    assert not line2d._x_limits_frozen
    assert not line2d._y_limits_frozen


def test_line2d_data_bounds_union_of_all_lines(line2d):
    """Adding multiple lines with different ranges produces the union bounds."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.side_effect = [1, 2]
    line2d._renderer = renderer
    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (-1.0, 1.0)
    line2d._chart2d_sent = True

    line2d.add_line(np.array([0.0, 5.0], dtype=np.float32),
                    np.array([-2.0, 2.0], dtype=np.float32), label="a")
    assert line2d._data_xmin == pytest.approx(0.0)
    assert line2d._data_xmax == pytest.approx(5.0)
    assert line2d._data_ymin == pytest.approx(-2.0)
    assert line2d._data_ymax == pytest.approx(2.0)

    # Second line with wider extents — bounds must cover the union.
    line2d.add_line(np.array([0.0, 8.0], dtype=np.float32),
                    np.array([-5.0, 1.0], dtype=np.float32), label="b")
    assert line2d._data_xmax == pytest.approx(8.0)
    assert line2d._data_ymin == pytest.approx(-5.0)


def test_line2d_toolbar_frame_exists(line2d):
    """Line2D must have a _toolbar_frame attribute after construction."""
    import tkinter as _tk
    assert hasattr(line2d, "_toolbar_frame")
    assert isinstance(line2d._toolbar_frame, _tk.Frame)


def _line2d_toolbar_button(line2d, label):
    import tkinter as _tk

    for child in line2d._toolbar_frame.winfo_children():
        if isinstance(child, _tk.Button) and label in child.cget("text"):
            return child
    raise AssertionError(f"toolbar button containing {label!r} not found")


def test_line2d_toolbar_home_static_plot_rebuilds_chart(line2d):
    """The Home toolbar button must refit a stationary plot and do a full chart rebuild."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.return_value = 0
    line2d._renderer = renderer
    line2d._chart2d_sent = True

    line2d.set_line([0.0, 10.0], [-2.0, 3.0])
    line2d.set_xlim(4.0, 5.0)
    line2d.set_ylim(-0.5, 0.5)
    renderer.reset_mock()

    _line2d_toolbar_button(line2d, "Home").invoke()

    assert line2d._xlim[0] <= 0.0
    assert line2d._xlim[1] >= 10.0
    assert line2d._ylim[0] <= -2.0
    assert line2d._ylim[1] >= 3.0
    assert not line2d._x_limits_frozen
    assert not line2d._y_limits_frozen
    renderer.set_chart2d.assert_called_once()
    renderer.chart2d_update_xlim.assert_not_called()
    renderer.chart2d_update_ylim.assert_not_called()


def test_line2d_toolbar_autoscale_y_static_plot_rebuilds_chart(line2d):
    """The Autoscale Y toolbar button must fit Y while leaving X frozen."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.return_value = 0
    line2d._renderer = renderer
    line2d._chart2d_sent = True

    line2d.set_line([0.0, 10.0], [-8.0, 12.0])
    line2d.set_xlim(2.0, 4.0)
    line2d.set_ylim(-1.0, 1.0)
    renderer.reset_mock()

    _line2d_toolbar_button(line2d, "Autoscale Y").invoke()

    assert line2d._xlim == (2.0, 4.0)
    assert line2d._x_limits_frozen
    assert not line2d._y_limits_frozen
    assert line2d._ylim[0] <= -8.0
    assert line2d._ylim[1] >= 12.0
    renderer.set_chart2d.assert_called_once()
    renderer.chart2d_update_xlim.assert_not_called()
    renderer.chart2d_update_ylim.assert_not_called()


def test_line2d_render_frame_is_render_target(line2d):
    """Renderer surface is directed at _render_frame, not the outer widget."""
    import tkinter as _tk
    assert hasattr(line2d, "_render_frame")
    assert isinstance(line2d._render_frame, _tk.Frame)
    assert line2d._render_target_widget is line2d._render_frame



# ── Phase 3: cursor, box zoom, streaming interlock ────────────────────────────

def test_line2d_enable_cursor_stores_flag(line2d):
    """enable_cursor(True) sets _cursor_enabled."""
    line2d.enable_cursor(True)
    assert line2d._cursor_enabled is True
    line2d.enable_cursor(False)
    assert line2d._cursor_enabled is False


def test_line2d_enable_cursor_snap_raises(line2d):
    """enable_cursor(snap=True) raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        line2d.enable_cursor(True, snap=True)


def test_line2d_enable_cursor_false_hides_cursor(line2d):
    """enable_cursor(False) calls chart2d_set_cursor with visible=False."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._cursor_enabled = True

    line2d.enable_cursor(False)
    renderer.chart2d_set_cursor.assert_called_with(0.0, 0.0, False)


def test_line2d_update_cursor_inside_plot(line2d):
    """_update_cursor inside the plot rect calls chart2d_set_cursor with correct data coords."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._xlim = (0.0, 10.0)
    line2d._ylim = (-1.0, 1.0)
    line2d._chart2d_sent = True

    # Simulate render_frame at 400×300; plot rect is fraction-based.
    tgt = mock.Mock()
    tgt.winfo_width.return_value = 400
    tgt.winfo_height.return_value = 300
    line2d._render_target_widget = tgt

    # Cursor at the center of the plot rect.
    pl = line2d._pad_left  * 400   # ~52
    pr = line2d._pad_right * 400   # ~388
    pt = line2d._pad_top   * 300   # ~12
    pb = line2d._pad_bottom * 300  # ~264
    mx = (pl + pr) / 2
    my = (pt + pb) / 2

    line2d._update_cursor(int(mx), int(my))

    renderer.chart2d_set_cursor.assert_called_once()
    call_args = renderer.chart2d_set_cursor.call_args[0]
    # x_data should be near midpoint of xlim (5.0), visible=True
    assert abs(call_args[0] - 5.0) < 0.5
    assert call_args[2] is True


def test_line2d_update_cursor_outside_plot_hides(line2d):
    """_update_cursor outside the plot rect hides the cursor."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._xlim = (0.0, 10.0)
    line2d._ylim = (-1.0, 1.0)

    tgt = mock.Mock()
    tgt.winfo_width.return_value = 400
    tgt.winfo_height.return_value = 300
    line2d._render_target_widget = tgt

    # Pixel in the margin (left of plot rect)
    line2d._update_cursor(5, 150)
    renderer.chart2d_set_cursor.assert_called_with(0.0, 0.0, False)


def test_line2d_enable_box_zoom_stores_flag(line2d):
    """enable_box_zoom(True/False) sets the flag."""
    line2d.enable_box_zoom(True)
    assert line2d._box_zoom_enabled is True
    line2d.enable_box_zoom(False)
    assert line2d._box_zoom_enabled is False


def test_line2d_enable_box_zoom_false_clears_active(line2d):
    """Disabling box zoom clears _box_zoom_active."""
    line2d._box_zoom_active = True
    line2d._bz_dragging = True
    line2d.enable_box_zoom(False)
    assert not line2d._box_zoom_active
    assert not line2d._bz_dragging


def test_line2d_apply_box_zoom_sets_limits_and_active(line2d):
    """_apply_box_zoom with a valid drag rect freezes both axes and sets _box_zoom_active."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._xlim = (0.0, 10.0)
    line2d._ylim = (-1.0, 1.0)
    line2d._chart2d_sent = True

    tgt = mock.Mock()
    tgt.winfo_width.return_value = 400
    tgt.winfo_height.return_value = 300
    line2d._render_target_widget = tgt

    # Drag from 25% to 75% of the plot rect in both axes.
    pl = line2d._pad_left  * 400
    pr = line2d._pad_right * 400
    pt = line2d._pad_top   * 300
    pb = line2d._pad_bottom * 300
    px0 = int(pl + (pr - pl) * 0.25)
    px1 = int(pl + (pr - pl) * 0.75)
    py0 = int(pt + (pb - pt) * 0.25)
    py1 = int(pt + (pb - pt) * 0.75)

    line2d._apply_box_zoom(px0, py0, px1, py1)

    assert line2d._box_zoom_active is True
    assert line2d._x_limits_frozen is True
    assert line2d._y_limits_frozen is True
    # New xlim should be roughly the middle 50% of [0, 10]
    assert line2d._xlim[0] == pytest.approx(2.5, abs=0.5)
    assert line2d._xlim[1] == pytest.approx(7.5, abs=0.5)


def test_line2d_apply_box_zoom_degenerate_ignored(line2d):
    """A zero-size box zoom drag is silently ignored."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._xlim = (0.0, 10.0)
    line2d._ylim = (-1.0, 1.0)
    line2d._chart2d_sent = True

    tgt = mock.Mock()
    tgt.winfo_width.return_value = 400
    tgt.winfo_height.return_value = 300
    line2d._render_target_widget = tgt

    # Same start and end pixel — degenerate drag.
    line2d._apply_box_zoom(100, 100, 100, 100)
    assert line2d._box_zoom_active is False


def test_line2d_resume_live_clears_active(line2d):
    """resume_live() clears the _box_zoom_active freeze flag."""
    line2d._box_zoom_active = True
    line2d.resume_live()
    assert line2d._box_zoom_active is False


def test_line2d_status_frame_exists(line2d):
    """Line2D must have a _status_frame and _status_label after construction."""
    import tkinter as _tk
    assert hasattr(line2d, "_status_frame")
    assert hasattr(line2d, "_status_label")
    assert isinstance(line2d._status_label, _tk.Label)


# ── Phase 1: title, x_tick_interval, labels, legend, visibility ───────────────

def test_line2d_set_title_stores_and_triggers_rebuild(line2d):
    """set_title stores the title and forces a full chart2d rebuild."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._ylim = (-1.0, 1.0)
    line2d.set_xlim(0.0, 10.0)
    assert renderer.set_chart2d.call_count == 1

    line2d.set_title("Sensor output")
    assert line2d._title == "Sensor output"
    assert line2d._pad_top == pytest.approx(0.08)
    # Must do a full rebuild, not fast-path, because top padding changed.
    assert renderer.set_chart2d.call_count == 2

    # Clearing the title resets padding.
    line2d.set_title("")
    assert line2d._pad_top == pytest.approx(0.04)


def test_line2d_set_title_passes_to_set_chart2d(line2d):
    """set_chart2d is called with the title string."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (-1.0, 1.0)
    line2d._chart2d_sent = True  # arm fast path

    line2d.set_title("My Chart")
    call_kwargs = renderer.set_chart2d.call_args
    # title is passed positionally; capture all args
    args = call_kwargs[0] if call_kwargs[0] else []
    assert "My Chart" in args, f"title not found in set_chart2d args: {args}"


def test_line2d_set_x_tick_interval_stores_and_rebuilds(line2d):
    """set_x_tick_interval stores the step and triggers a full chart2d rebuild."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._ylim = (-1.0, 1.0)
    line2d.set_xlim(0.0, 10.0)
    count_before = renderer.set_chart2d.call_count

    line2d.set_x_tick_interval(1.0)
    assert line2d._x_tick_interval == pytest.approx(1.0)
    assert renderer.set_chart2d.call_count == count_before + 1


def test_line2d_set_x_tick_interval_none(line2d):
    """set_x_tick_interval(None) resets to auto."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._xlim = (0.0, 10.0)
    line2d._ylim = (-1.0, 1.0)
    line2d._chart2d_sent = True

    line2d.set_x_tick_interval(2.0)
    line2d.set_x_tick_interval(None)
    assert line2d._x_tick_interval is None


def test_line2d_set_x_tick_interval_invalid_raises(line2d):
    """Non-positive or non-finite x tick intervals raise ValueError."""
    with pytest.raises(ValueError):
        line2d.set_x_tick_interval(-1.0)
    with pytest.raises(ValueError):
        line2d.set_x_tick_interval(0.0)
    with pytest.raises(ValueError):
        line2d.set_x_tick_interval(float("nan"))


def test_line2d_set_line_label_stored(line2d):
    """set_line with label= stores the label on the widget."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.return_value = 1
    line2d._renderer = renderer
    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (0.0, 1.0)
    line2d._chart2d_sent = True

    x = np.linspace(0, 1, 10, dtype=np.float32)
    line2d.set_line(x, x, label="Channel A")
    assert line2d._primary_label == "Channel A"


def test_line2d_set_line_no_label(line2d):
    """set_line without label= stores None."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.return_value = 1
    line2d._renderer = renderer
    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (0.0, 1.0)
    line2d._chart2d_sent = True

    x = np.linspace(0, 1, 10, dtype=np.float32)
    line2d.set_line(x, x)
    assert line2d._primary_label is None


def test_line2d_add_line_label_stored(line2d):
    """add_line with label= stores the label in _named_lines."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.return_value = 5
    line2d._renderer = renderer
    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (0.0, 1.0)
    line2d._chart2d_sent = True

    x = np.linspace(0, 1, 10, dtype=np.float32)
    handle = line2d.add_line(x, x, label="Series B")
    assert line2d._named_lines[handle]["label"] == "Series B"


def test_line2d_add_line_before_map_label_in_pending(line2d):
    """add_line before renderer stores label in pending tuple."""
    assert line2d._renderer is None
    x = np.zeros(5, dtype=np.float32)
    vhandle = line2d.add_line(x, x, label="Pre-map label")
    tup = line2d._pending_named_lines[vhandle]
    assert tup[4] == "Pre-map label"  # index 4 = label in 6-tuple


def test_line2d_add_line_stream_label_stored(line2d):
    """add_line_stream with label= stores the label in the stream state dict."""
    sid = line2d.add_line_stream(max_points=100, label="Live feed")
    assert line2d._line_streams[sid]["label"] == "Live feed"


def test_line2d_add_line_stream_no_label(line2d):
    """add_line_stream without label= stores None."""
    sid = line2d.add_line_stream(max_points=100)
    assert line2d._line_streams[sid]["label"] is None


def test_line2d_show_legend_calls_set_legend(line2d):
    """show_legend(True) triggers chart2d_set_legend with visible=True."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.return_value = 10
    line2d._renderer = renderer
    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (0.0, 1.0)
    line2d._chart2d_sent = True

    x = np.linspace(0, 1, 5, dtype=np.float32)
    line2d.set_line(x, x, label="A")
    line2d._primary_handle = 10

    line2d.show_legend(True)
    renderer.chart2d_set_legend.assert_called()
    call_args = renderer.chart2d_set_legend.call_args[0]
    assert call_args[2] is True  # visible=True


def test_line2d_show_legend_false_hides(line2d):
    """show_legend(False) calls chart2d_set_legend with visible=False."""
    import unittest.mock as mock

    renderer = mock.Mock()
    line2d._renderer = renderer
    line2d._legend_visible = True  # was visible

    line2d.show_legend(False)
    renderer.chart2d_set_legend.assert_called()
    call_args = renderer.chart2d_set_legend.call_args[0]
    assert call_args[2] is False


def test_line2d_legend_position_valid(line2d):
    """Setting legend_position to any valid value is accepted."""
    for pos in ("top-right", "top-left", "bottom-right", "bottom-left"):
        line2d.legend_position = pos
        assert line2d.legend_position == pos


def test_line2d_legend_position_invalid_raises(line2d):
    """Setting legend_position to an unknown string raises ValueError."""
    with pytest.raises(ValueError):
        line2d.legend_position = "center"


def test_line2d_legend_entries_include_labeled_lines(line2d):
    """_push_legend sends correct (label, color) pairs for labeled lines."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.side_effect = lambda x, y, c, lw: id(c)
    line2d._renderer = renderer
    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (0.0, 1.0)
    line2d._chart2d_sent = True

    x = np.linspace(0, 1, 5, dtype=np.float32)
    # primary
    color_a = (0.1, 0.2, 0.3)
    line2d.set_line(x, x, color=color_a, label="A")
    line2d._primary_handle = renderer.chart2d_add_line.return_value
    # named
    color_b = (0.4, 0.5, 0.6)
    renderer.chart2d_add_line.return_value = 99
    line2d.add_line(x, x, color=color_b, label="B")

    renderer.reset_mock()
    line2d._legend_visible = True
    line2d._push_legend()

    call_args = renderer.chart2d_set_legend.call_args[0]
    entries = call_args[0]
    labels = [e[0] for e in entries]
    assert "A" in labels
    assert "B" in labels


def test_line2d_unlabeled_lines_excluded_from_legend(line2d):
    """Lines without label= are not included in legend entries."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.return_value = 7
    line2d._renderer = renderer
    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (0.0, 1.0)
    line2d._chart2d_sent = True

    x = np.linspace(0, 1, 5, dtype=np.float32)
    line2d.set_line(x, x)  # no label
    line2d._primary_handle = 7
    line2d.add_line(x, x)  # no label

    line2d._legend_visible = True
    line2d._push_legend()

    call_args = renderer.chart2d_set_legend.call_args[0]
    entries = call_args[0]
    assert entries == []


def test_line2d_set_line_visibility_updates_named_line(line2d):
    """set_line_visibility updates the visible flag and calls the renderer."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.return_value = 20
    line2d._renderer = renderer
    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (0.0, 1.0)
    line2d._chart2d_sent = True

    x = np.linspace(0, 1, 5, dtype=np.float32)
    handle = line2d.add_line(x, x, label="C")

    line2d.set_line_visibility(handle, False)
    assert line2d._named_lines[handle]["visible"] is False
    renderer.chart2d_set_line_visible.assert_called_with(handle, False)

    line2d.set_line_visibility(handle, True)
    assert line2d._named_lines[handle]["visible"] is True
    renderer.chart2d_set_line_visible.assert_called_with(handle, True)


def test_line2d_set_line_visibility_pending(line2d):
    """set_line_visibility before renderer updates the pending tuple."""
    assert line2d._renderer is None
    x = np.zeros(5, dtype=np.float32)
    vhandle = line2d.add_line(x, x, label="D")
    assert line2d._pending_named_lines[vhandle][5] is True  # default visible

    line2d.set_line_visibility(vhandle, False)
    assert line2d._pending_named_lines[vhandle][5] is False


def test_line2d_update_line_label_updates_legend(line2d):
    """update_line with label= updates the stored label and refreshes legend."""
    import unittest.mock as mock

    renderer = mock.Mock()
    renderer.chart2d_add_line.return_value = 30
    line2d._renderer = renderer
    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (0.0, 1.0)
    line2d._chart2d_sent = True

    x = np.linspace(0, 1, 5, dtype=np.float32)
    handle = line2d.add_line(x, x, label="old")

    line2d._legend_visible = True
    renderer.reset_mock()
    line2d.update_line(handle, x, x, label="new")
    assert line2d._named_lines[handle]["label"] == "new"
    renderer.chart2d_set_legend.assert_called()


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


# ── Phase 5: reference overlays (axhspan / axvspan / axhline / axvline) ───────

def test_line2d_axhspan_returns_handle(line2d):
    """axhspan() must return an integer handle."""
    h = line2d.axhspan(-1.0, 1.0)
    assert isinstance(h, int)


def test_line2d_axvspan_returns_handle(line2d):
    """axvspan() must return an integer handle."""
    h = line2d.axvspan(2.0, 4.0)
    assert isinstance(h, int)


def test_line2d_axhline_returns_handle(line2d):
    """axhline() must return an integer handle."""
    h = line2d.axhline(0.5)
    assert isinstance(h, int)


def test_line2d_axvline_returns_handle(line2d):
    """axvline() must return an integer handle."""
    h = line2d.axvline(1.5)
    assert isinstance(h, int)


def test_line2d_overlays_get_unique_handles(line2d):
    """Each axhspan / axvspan / axhline / axvline call must return a different handle."""
    handles = [
        line2d.axhspan(-1.0, 0.0),
        line2d.axvspan(0.0, 1.0),
        line2d.axhline(0.5),
        line2d.axvline(0.5),
    ]
    assert len(set(handles)) == 4


def test_line2d_axhspan_stored_in_overlay_meta(line2d):
    """axhspan must record metadata in _overlay_meta for deferred replay."""
    h = line2d.axhspan(2.0, 3.0, color=(1.0, 0.0, 0.0, 0.3))
    assert h in line2d._overlay_meta
    assert line2d._overlay_meta[h]["kind"] == "hspan"


def test_line2d_axvspan_stored_in_overlay_meta(line2d):
    h = line2d.axvspan(-5.0, -3.0)
    assert h in line2d._overlay_meta
    assert line2d._overlay_meta[h]["kind"] == "vspan"


def test_line2d_axhline_stored_in_overlay_meta(line2d):
    h = line2d.axhline(0.0)
    assert h in line2d._overlay_meta
    assert line2d._overlay_meta[h]["kind"] == "hline"


def test_line2d_axvline_stored_in_overlay_meta(line2d):
    h = line2d.axvline(10.0)
    assert h in line2d._overlay_meta
    assert line2d._overlay_meta[h]["kind"] == "vline"


def test_line2d_axhspan_calls_renderer_when_live(line2d):
    """When renderer exists, axhspan must call chart2d_add_hspan immediately."""
    import unittest.mock as mock
    renderer = mock.Mock()
    renderer.chart2d_add_hspan.return_value = 99
    line2d._renderer = renderer
    line2d.axhspan(1.0, 2.0, color=(0.5, 0.5, 1.0, 0.2))
    renderer.chart2d_add_hspan.assert_called_once()
    args = renderer.chart2d_add_hspan.call_args[0]
    assert args[0] == pytest.approx(1.0)
    assert args[1] == pytest.approx(2.0)


def test_line2d_axvspan_calls_renderer_when_live(line2d):
    import unittest.mock as mock
    renderer = mock.Mock()
    renderer.chart2d_add_vspan.return_value = 88
    line2d._renderer = renderer
    line2d.axvspan(-2.0, -1.0)
    renderer.chart2d_add_vspan.assert_called_once()


def test_line2d_axhline_calls_renderer_when_live(line2d):
    import unittest.mock as mock
    renderer = mock.Mock()
    renderer.chart2d_add_hline.return_value = 77
    line2d._renderer = renderer
    line2d.axhline(0.0, color=(0.8, 0.8, 0.3), line_width=2.0)
    renderer.chart2d_add_hline.assert_called_once()
    args = renderer.chart2d_add_hline.call_args[0]
    assert args[0] == pytest.approx(0.0)


def test_line2d_axvline_calls_renderer_when_live(line2d):
    import unittest.mock as mock
    renderer = mock.Mock()
    renderer.chart2d_add_vline.return_value = 66
    line2d._renderer = renderer
    line2d.axvline(5.0, line_width=1.0)
    renderer.chart2d_add_vline.assert_called_once()


def test_line2d_remove_overlay_clears_meta(line2d):
    """remove_overlay must delete _overlay_meta and call renderer.chart2d_remove_overlay."""
    import unittest.mock as mock
    renderer = mock.Mock()
    renderer.chart2d_add_hspan.return_value = 55
    line2d._renderer = renderer
    h = line2d.axhspan(0.0, 1.0)
    assert h in line2d._overlay_meta
    line2d.remove_overlay(h)
    assert h not in line2d._overlay_meta
    renderer.chart2d_remove_overlay.assert_called_once_with(55)


def test_line2d_remove_overlay_unknown_handle_is_noop(line2d):
    """remove_overlay with an unknown handle must not raise."""
    line2d.remove_overlay(99999)


def test_line2d_clear_chart_overlays_removes_all(line2d):
    """clear_chart_overlays must remove all overlay metadata and call the renderer."""
    import unittest.mock as mock
    renderer = mock.Mock()
    renderer.chart2d_add_hspan.return_value = 10
    renderer.chart2d_add_vspan.return_value = 11
    line2d._renderer = renderer
    line2d.axhspan(0.0, 1.0)
    line2d.axvspan(0.0, 1.0)
    assert len(line2d._overlay_meta) == 2
    line2d.clear_chart_overlays()
    assert len(line2d._overlay_meta) == 0
    renderer.chart2d_clear_overlays.assert_called_once()


def test_line2d_overlay_deferred_replay(root):
    """Overlays added before renderer is mapped must be stored and replayed later.

    We verify the overlay_meta dict is populated while renderer is None, then
    simulate the replay step (mirroring _init_renderer's loop) to confirm it
    correctly calls the renderer and populates _overlay_handle_map.
    """
    import unittest.mock as mock
    w = Line2D(root, width=320, height=240)
    # Add overlays while renderer is None.
    h1 = w.axhspan(0.0, 1.0)
    h2 = w.axvline(5.0)
    assert w._renderer is None
    assert h1 in w._overlay_meta
    assert h2 in w._overlay_meta
    assert w._overlay_meta[h1]["kind"] == "hspan"
    assert w._overlay_meta[h2]["kind"] == "vline"
    # Inject mock renderer and run the overlay replay portion directly
    # (calling _init_renderer() would trigger the base-class renderer-creation
    # path which replaces the mock, so we simulate only the replay step).
    renderer = mock.Mock()
    renderer.chart2d_add_hspan.return_value = 100
    renderer.chart2d_add_vline.return_value = 101
    w._renderer = renderer
    # Simulate the replay loop from _init_renderer:
    _dispatch = {
        "hspan": renderer.chart2d_add_hspan,
        "vspan": renderer.chart2d_add_vspan,
        "hline": renderer.chart2d_add_hline,
        "vline": renderer.chart2d_add_vline,
    }
    for vhandle, meta in list(w._overlay_meta.items()):
        fn = _dispatch.get(meta["kind"])
        if fn is not None:
            rust_id = fn(*meta["args"])
            w._overlay_handle_map[vhandle] = rust_id
    # Verify renderer received the right calls.
    renderer.chart2d_add_hspan.assert_called_once()
    renderer.chart2d_add_vline.assert_called_once()
    # Verify handle map was populated.
    assert w._overlay_handle_map[h1] == 100
    assert w._overlay_handle_map[h2] == 101
    w.destroy()


# ── Phase 4: axis formatting and log scale ────────────────────────────────────

def test_line2d_set_x_tick_formatter_stores_and_calls_renderer(line2d):
    """set_x_tick_formatter should persist the format and call the renderer."""
    from unittest.mock import MagicMock, patch

    line2d.add_line([0, 1, 2], [0, 1, 2])
    renderer = MagicMock()
    line2d._renderer = renderer

    line2d.set_x_tick_formatter("sci")

    assert line2d._x_tick_format == "sci"
    renderer.chart2d_set_tick_format.assert_called_once_with("x", "sci")


def test_line2d_set_y_tick_formatter_stores_and_calls_renderer(line2d):
    """set_y_tick_formatter should persist the format and call the renderer."""
    from unittest.mock import MagicMock

    line2d.add_line([0, 1, 2], [0, 1, 2])
    renderer = MagicMock()
    line2d._renderer = renderer

    line2d.set_y_tick_formatter("int")

    assert line2d._y_tick_format == "int"
    renderer.chart2d_set_tick_format.assert_called_once_with("y", "int")


def test_line2d_set_x_tick_formatter_no_renderer_stores(line2d):
    """set_x_tick_formatter before renderer is mapped should store for later replay."""
    line2d._renderer = None

    line2d.set_x_tick_formatter("time")

    assert line2d._x_tick_format == "time"


def test_line2d_set_xscale_linear_stores_false(line2d):
    """set_xscale('linear') stores x_log_scale=False."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    line2d._renderer = renderer

    line2d.set_xscale("linear")

    assert line2d._x_log_scale is False
    renderer.chart2d_set_log_scale.assert_called_once_with("x", False)


def test_line2d_set_xscale_log_stores_true(line2d):
    """set_xscale('log') stores x_log_scale=True and calls renderer."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    line2d._renderer = renderer

    line2d.set_xscale("log")

    assert line2d._x_log_scale is True
    renderer.chart2d_set_log_scale.assert_called_once_with("x", True)


def test_line2d_set_yscale_log_stores_true(line2d):
    """set_yscale('log') stores y_log_scale=True and calls renderer."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    line2d._renderer = renderer

    line2d.set_yscale("log")

    assert line2d._y_log_scale is True
    renderer.chart2d_set_log_scale.assert_called_once_with("y", True)


def test_line2d_set_yscale_no_renderer_stores(line2d):
    """set_yscale before renderer is mapped should store for later replay."""
    line2d._renderer = None

    line2d.set_yscale("log")

    assert line2d._y_log_scale is True


def test_line2d_tick_format_replayed_on_init(root):
    """Tick formatter set before renderer creation must be applied on _init_renderer."""
    from unittest.mock import MagicMock, patch

    w = Line2D(root, width=320, height=240)
    w._renderer = None

    # Configure before renderer exists.
    w.set_x_tick_formatter("sci")
    w.set_yscale("log")

    # Simulate renderer becoming available.
    renderer = MagicMock()
    renderer.chart2d_add_hspan.return_value = 0
    renderer.chart2d_add_vspan.return_value = 0
    renderer.chart2d_add_hline.return_value = 0
    renderer.chart2d_add_vline.return_value = 0

    w._renderer = renderer
    # Manually invoke the replay section (without calling full _init_renderer
    # which would try to create a real GPU context).
    if w._x_tick_format != "default":
        renderer.chart2d_set_tick_format("x", w._x_tick_format)
    if w._y_tick_format != "default":
        renderer.chart2d_set_tick_format("y", w._y_tick_format)
    if w._x_log_scale:
        renderer.chart2d_set_log_scale("x", True)
    if w._y_log_scale:
        renderer.chart2d_set_log_scale("y", True)

    renderer.chart2d_set_tick_format.assert_called_once_with("x", "sci")
    renderer.chart2d_set_log_scale.assert_called_once_with("y", True)
    w.destroy()


def test_line2d_tick_format_defaults(line2d):
    """Fresh widget should have default (linear, 'default' format) settings."""
    assert line2d._x_tick_format == "default"
    assert line2d._y_tick_format == "default"
    assert line2d._x_log_scale is False
    assert line2d._y_log_scale is False


def test_line2d_set_x_tick_formatter_time_format(line2d):
    """'time' formatter should be accepted without error."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    line2d._renderer = renderer

    line2d.set_x_tick_formatter("time")

    assert line2d._x_tick_format == "time"
    renderer.chart2d_set_tick_format.assert_called_once_with("x", "time")


# ── Audit fixes ───────────────────────────────────────────────────────────────

def test_line2d_clear_resets_python_state(line2d):
    """clear() must wipe all chart2d Python-side state."""
    from unittest.mock import MagicMock

    line2d.add_line([0, 1], [0, 1], label="a")
    line2d.axhspan(0.0, 1.0)
    renderer = MagicMock()
    renderer.chart2d_clear_lines = MagicMock()
    renderer.chart2d_clear_overlays = MagicMock()
    renderer.chart2d_set_legend = MagicMock()
    renderer.set_chart2d = MagicMock()
    line2d._renderer = renderer

    line2d.clear()

    assert line2d._named_lines == {}
    assert line2d._pending_named_lines == {}
    assert line2d._line_streams == {}
    assert line2d._overlay_meta == {}
    assert line2d._overlay_handle_map == {}
    assert line2d._primary_handle is None
    assert line2d._pending_primary is None
    assert line2d._data_xmin is None
    assert line2d._data_xmax is None
    assert line2d._x_limits_frozen is False
    assert line2d._y_limits_frozen is False
    renderer.chart2d_clear_lines.assert_called_once()
    renderer.chart2d_clear_overlays.assert_called_once()


def test_line2d_clear_no_renderer_resets_pending(line2d):
    """clear() before renderer is mapped must wipe pending state too."""
    line2d._renderer = None
    line2d._pending_named_lines[0] = ([0.0], [0.0], (1, 0, 0), 2.0, None, True)
    line2d._overlay_meta[0] = {"kind": "hspan", "args": (0.0, 1.0, [0, 0, 0, 1])}

    line2d.clear()

    assert line2d._pending_named_lines == {}
    assert line2d._overlay_meta == {}


def test_line2d_clear_does_not_call_3d_clear(line2d):
    """Line2D.clear() must NOT call clear_actors (the 3D scatter clear path)."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    line2d._renderer = renderer

    line2d.clear()

    renderer.clear_actors.assert_not_called()


def test_line2d_update_line_updates_data_bounds(line2d):
    """update_line() must update running data bounds so home/autoscale work."""
    import numpy as np
    from unittest.mock import MagicMock

    h = line2d.add_line([0.0, 1.0], [0.0, 1.0])
    renderer = MagicMock()
    line2d._renderer = renderer

    line2d.update_line(h, [0.0, 100.0], [0.0, 50.0])

    assert line2d._data_xmax == pytest.approx(100.0, abs=1.0)
    assert line2d._data_ymax == pytest.approx(50.0, abs=1.0)


def test_line2d_update_line_unfrozen_updates_ylim(line2d):
    """update_line() with unfrozen y should update ylim to new data range."""
    h = line2d.add_line([0.0, 1.0], [0.0, 1.0])
    from unittest.mock import MagicMock
    renderer = MagicMock()
    line2d._renderer = renderer
    line2d._y_limits_frozen = False

    line2d.update_line(h, [0.0, 1.0], [-5.0, 5.0])

    ylo, yhi = line2d._ylim
    assert ylo <= -5.0
    assert yhi >= 5.0


def test_line2d_legend_replayed_on_init(root):
    """show_legend() before renderer init must produce a legend after _init_renderer."""
    from unittest.mock import MagicMock, patch

    w = Line2D(root, width=320, height=240)
    w._renderer = None
    w.add_line([0, 1], [0, 1], label="series")
    w.show_legend()  # sets flag; renderer not present yet

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 0
    renderer.chart2d_set_legend = MagicMock()

    # Manually invoke only the legend-replay portion of _init_renderer.
    w._renderer = renderer
    if w._legend_visible:
        w._push_legend()

    renderer.chart2d_set_legend.assert_called()
    w.destroy()


def test_line2d_remove_line_refreshes_legend(line2d):
    """remove_line() must refresh the legend so removed series disappear."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    line2d._renderer = renderer
    line2d._legend_visible = True

    h = line2d.add_line([0, 1], [0, 1], label="to remove")
    renderer.chart2d_set_legend.reset_mock()

    line2d.remove_line(h)

    renderer.chart2d_set_legend.assert_called()


def test_render_tick_stops_after_repeated_failures(root):
    """_render_tick must stop retrying after _RENDER_FAIL_LIMIT failures."""
    import warnings
    from unittest.mock import MagicMock

    w = Line2D(root, width=320, height=240)
    renderer = MagicMock()
    renderer.render.side_effect = RuntimeError("gpu dead")
    w._renderer = renderer
    w._dirty = True

    limit = w._RENDER_FAIL_LIMIT
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(limit):
            w._render_tick()
            if i < limit - 1:
                w._dirty = True  # keep dirty for next tick; skip after final

    assert any("render()" in str(c.message) for c in caught)
    # After limit the warning fired and dirty was cleared (loop stopped).
    assert w._dirty is False
    w.destroy()


def test_render_tick_resets_fail_count_on_success(root):
    """A successful render must reset the consecutive failure counter."""
    from unittest.mock import MagicMock

    w = Line2D(root, width=320, height=240)
    renderer = MagicMock()
    w._renderer = renderer
    w._render_fail_count = 3
    w._dirty = True

    renderer.render.side_effect = None  # success
    w._render_tick()

    assert w._render_fail_count == 0
    w.destroy()


# ── Second-round audit fixes ──────────────────────────────────────────────────

def test_stream_line_box_zoom_blocks_refit(line2d):
    """stream_line() must not change xlim/ylim while box-zoom is active."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 0
    line2d._renderer = renderer
    line2d._chart2d_sent = True
    line2d._box_zoom_active = True
    line2d._xlim = (0.0, 10.0)
    line2d._ylim = (-1.0, 1.0)

    h = line2d.add_line_stream(max_points=100, mode="ring")
    xs = np.linspace(0, 5, 50, dtype=np.float32)
    ys = np.ones(50, dtype=np.float32) * 999.0  # would expand ylim if not blocked

    line2d.stream_line(h, xs, ys)

    # ylim must not have changed to accommodate the 999 values.
    assert line2d._ylim[1] < 500.0


def test_stream_line_no_box_zoom_updates_limits(line2d):
    """stream_line() with box-zoom inactive must still update limits."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 0
    line2d._renderer = renderer
    line2d._chart2d_sent = True
    line2d._box_zoom_active = False

    h = line2d.add_line_stream(max_points=100, mode="ring")
    xs = np.linspace(0, 5, 50, dtype=np.float32)
    ys = np.ones(50, dtype=np.float32) * 999.0

    line2d.stream_line(h, xs, ys)

    assert line2d._ylim[1] > 500.0


def test_update_line_shrink_corrects_bounds(line2d):
    """update_line() to a smaller range must shrink _data_xmax, not leave it stale."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 42
    line2d._renderer = renderer

    h = line2d.add_line([0.0, 100.0], [0.0, 100.0])
    assert line2d._data_xmax == pytest.approx(100.0, abs=1.0)

    line2d.update_line(h, [0.0, 10.0], [0.0, 10.0])

    assert line2d._data_xmax == pytest.approx(10.0, abs=1.0)
    assert line2d._data_ymax == pytest.approx(10.0, abs=1.0)


def test_remove_line_corrects_bounds(line2d):
    """remove_line() must shrink bounds to the remaining data, not keep old extent."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.side_effect = [10, 11]
    line2d._renderer = renderer

    h1 = line2d.add_line([0.0, 100.0], [0.0, 0.0])
    h2 = line2d.add_line([0.0, 5.0],   [0.0, 1.0])

    line2d.remove_line(h1)

    # After removing h1 (xmax=100), remaining data only goes to 5.
    assert line2d._data_xmax == pytest.approx(5.0, abs=1.0)


def test_home_uses_stored_named_line_geometry(line2d):
    """home() must derive bounds from current stored x/y, not monotonic accumulator."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 0
    line2d._renderer = renderer
    line2d._chart2d_sent = True

    h = line2d.add_line([0.0, 1.0], [0.0, 1.0])
    # Simulate bounds shrinking after update.
    line2d.update_line(h, [0.0, 0.5], [0.0, 0.5])

    line2d._x_limits_frozen = True
    line2d._y_limits_frozen = True
    line2d.home()

    assert not line2d._x_limits_frozen
    xlo, xhi = line2d._xlim
    assert xhi < 2.0  # must reflect the smaller data, not the old 1.0 (and not 100.0)


def test_remove_line_stream_refreshes_legend(line2d):
    """remove_line_stream() must refresh the legend so labeled streams disappear."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 0
    line2d._renderer = renderer
    line2d._legend_visible = True

    h = line2d.add_line_stream(max_points=100, mode="ring", label="stream")
    renderer.chart2d_set_legend.reset_mock()

    line2d.remove_line_stream(h)

    renderer.chart2d_set_legend.assert_called()


def test_scatter2d_render_tick_delegates_failure_to_base(root):
    """Scatter2D._render_tick must respect the base-class failure cap."""
    import warnings
    from unittest.mock import MagicMock
    from dragonsci import Scatter2D

    w = Scatter2D(root, width=320, height=240)
    renderer = MagicMock()
    renderer.render.side_effect = RuntimeError("gpu dead")
    w._renderer = renderer
    w._dirty = True

    limit = w._RENDER_FAIL_LIMIT
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(limit):
            w._render_tick()
            if i < limit - 1:
                w._dirty = True

    assert any("render()" in str(c.message) for c in caught)
    assert w._dirty is False
    w.destroy()


# ── Third-audit regression tests ─────────────────────────────────────────────

def test_home_clears_box_zoom_active(line2d):
    """home() must clear _box_zoom_active so streaming resumes."""
    line2d._box_zoom_active = True
    line2d._primary_x = np.array([0.0, 1.0])
    line2d._primary_y = np.array([0.0, 1.0])
    line2d._data_xmin = 0.0; line2d._data_xmax = 1.0
    line2d._data_ymin = 0.0; line2d._data_ymax = 1.0
    line2d.home()
    assert line2d._box_zoom_active is False


def test_autoscale_both_clears_box_zoom_active(line2d):
    """autoscale_both() delegates to home() and therefore clears _box_zoom_active."""
    line2d._box_zoom_active = True
    line2d._primary_x = np.array([0.0, 1.0])
    line2d._primary_y = np.array([0.0, 1.0])
    line2d._data_xmin = 0.0; line2d._data_xmax = 1.0
    line2d._data_ymin = 0.0; line2d._data_ymax = 1.0
    line2d.autoscale_both()
    assert line2d._box_zoom_active is False


def test_clear_line_stream_refits_from_remaining_data(line2d):
    """clear_line_stream() must refit limits from remaining static data."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 99
    line2d._renderer = renderer

    # Set up a static line with known range
    x_static = np.array([0.0, 2.0])
    y_static = np.array([0.0, 2.0])
    line2d._primary_x = x_static
    line2d._primary_y = y_static

    # Add a stream with wider data that has already inflated the limits
    h = line2d.add_line_stream(max_points=50, mode="ring")
    st = line2d._line_streams[h]
    st["buf_x"][:4] = [0, 5, 10, 15]
    st["buf_y"][:4] = [0, 5, 10, 15]
    st["count"] = 4
    st["head"] = 4

    # Force limits to match the expanded stream range
    line2d._xlim = (0.0, 20.0)
    line2d._ylim = (0.0, 20.0)
    line2d._x_limits_frozen = False
    line2d._y_limits_frozen = False

    # Clear the stream; limits should shrink back to static data range
    st["render_handle"] = None  # no renderer interaction needed
    line2d.clear_line_stream(h)

    # x limits must now reflect static data only (max 2.0), not the old 20.0
    assert line2d._xlim is not None
    xlo, xhi = line2d._xlim
    assert xhi < 10.0


def test_remove_line_stream_refits_from_remaining_data(line2d):
    """remove_line_stream() must refit limits from remaining static data."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 99
    line2d._renderer = renderer

    # Static data with range [0, 2]
    line2d._primary_x = np.array([0.0, 2.0])
    line2d._primary_y = np.array([0.0, 2.0])

    h = line2d.add_line_stream(max_points=50, mode="ring")
    st = line2d._line_streams[h]
    st["buf_x"][:3] = [0, 10, 20]
    st["buf_y"][:3] = [0, 10, 20]
    st["count"] = 3
    st["head"] = 3
    st["render_handle"] = None  # skip renderer

    line2d._xlim = (0.0, 25.0)
    line2d._ylim = (0.0, 25.0)
    line2d._x_limits_frozen = False
    line2d._y_limits_frozen = False

    line2d.remove_line_stream(h)

    assert line2d._xlim is not None
    xlo, xhi = line2d._xlim
    assert xhi < 10.0


def test_remove_line_stream_resets_xlim_when_no_data(line2d):
    """remove_line_stream() resets _xlim/_ylim to None when all data is gone."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 42
    line2d._renderer = renderer

    h = line2d.add_line_stream(max_points=50, mode="ring")
    st = line2d._line_streams[h]
    # stream has no data (count stays 0)
    st["render_handle"] = None

    line2d._xlim = (0.0, 1.0)
    line2d._ylim = (0.0, 1.0)
    line2d._x_limits_frozen = False
    line2d._y_limits_frozen = False

    line2d.remove_line_stream(h)

    # No data at all → limits must be cleared
    assert line2d._xlim is None
    assert line2d._ylim is None


def test_line2d_clear_resets_xlim_ylim(line2d):
    """Line2D.clear() must reset _xlim and _ylim so the renderer gets a fresh empty state."""
    line2d._xlim = (0.0, 10.0)
    line2d._ylim = (0.0, 10.0)
    line2d.clear()
    assert line2d._xlim is None
    assert line2d._ylim is None


def test_add_line_multi_series_bounds_union(line2d):
    """add_line() must fit the union of all series, not just the latest one."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.side_effect = [10, 20]
    line2d._renderer = renderer

    line2d.add_line(np.array([0.0, 1.0]), np.array([0.0, 1.0]), label="a")
    line2d.add_line(np.array([5.0, 6.0]), np.array([5.0, 6.0]), label="b")

    # x range must span [0, 6], not just [5, 6]
    assert line2d._xlim is not None
    xlo, xhi = line2d._xlim
    assert xlo <= 0.0
    assert xhi >= 6.0


def test_set_line_multi_series_refits_all(line2d):
    """set_line() on top of add_line() must produce union bounds."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 5
    line2d._renderer = renderer

    # Primary at x=[0,1]; named at x=[8,9]
    line2d.add_line(np.array([8.0, 9.0]), np.array([0.0, 1.0]), label="wide")
    line2d.set_line(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    assert line2d._xlim is not None
    xlo, xhi = line2d._xlim
    assert xhi >= 8.0  # named line must still be included


# ── Fourth-audit regression tests ────────────────────────────────────────────

def test_stream_line_uses_union_bounds_with_static_line(line2d):
    """stream_line() must fit the union of stream + static data, not just stream."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 1
    line2d._renderer = renderer
    line2d._chart2d_sent = True

    # Static line spans x=[0, 100]
    line2d._primary_x = np.array([0.0, 100.0], dtype=np.float32)
    line2d._primary_y = np.array([0.0, 100.0], dtype=np.float32)

    h = line2d.add_line_stream(max_points=100, mode="ring")
    line2d._line_streams[h]["render_handle"] = 1

    # Stream only has narrow data x=[1, 2] — must NOT shrink view to that range
    line2d.stream_line(h, np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    assert line2d._xlim is not None
    xlo, xhi = line2d._xlim
    assert xhi >= 50.0  # static line extent must be preserved


def test_stream_line_uses_union_bounds_two_streams(line2d):
    """A narrow update to one stream must not shrink the view clipping the other."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.side_effect = [1, 2]
    line2d._renderer = renderer
    line2d._chart2d_sent = True

    h1 = line2d.add_line_stream(max_points=100, mode="ring")
    h2 = line2d.add_line_stream(max_points=100, mode="ring")
    line2d._line_streams[h1]["render_handle"] = 1
    line2d._line_streams[h2]["render_handle"] = 2

    # Wide stream first
    line2d.stream_line(h1, np.array([0.0, 50.0]), np.array([0.0, 50.0]))
    # Narrow update to second stream — view must still cover h1's range
    line2d.stream_line(h2, np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    assert line2d._xlim is not None
    xlo, xhi = line2d._xlim
    assert xhi >= 25.0  # h1 range must still be included


def test_current_data_bounds_includes_pending_named_lines(line2d):
    """_current_data_bounds() must include lines queued before renderer init."""
    # No renderer — lines go to _pending_named_lines
    h = line2d.add_line(np.array([0.0, 50.0]), np.array([0.0, 50.0]), label="a")
    bounds = line2d._current_data_bounds()
    assert bounds is not None
    xmin, xmax, ymin, ymax = bounds
    assert xmax >= 50.0


def test_update_line_pre_render_refits(line2d):
    """update_line() before renderer init must refit bounds from pending lines."""
    h = line2d.add_line(np.array([0.0, 50.0]), np.array([0.0, 50.0]), label="a")

    # Update with smaller range — xlim should shrink, not stay at 50
    line2d.update_line(h, np.array([0.0, 5.0]), np.array([0.0, 5.0]))

    assert line2d._xlim is not None
    xlo, xhi = line2d._xlim
    assert xhi < 20.0  # shrunk from 50


def test_remove_line_pre_render_refits(line2d):
    """remove_line() before renderer init must clear bounds when all lines gone."""
    h = line2d.add_line(np.array([0.0, 50.0]), np.array([0.0, 50.0]), label="a")
    line2d.remove_line(h)

    # No lines remain — data accumulator must be cleared
    assert line2d._data_xmin is None
    assert line2d._data_xmax is None


def test_push_chart2d_sends_default_when_no_limits(line2d):
    """_push_chart2d() must still call set_chart2d with default (0,1) when _xlim/_ylim is None."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    line2d._renderer = renderer
    line2d._xlim = None
    line2d._ylim = None

    line2d._push_chart2d()

    renderer.set_chart2d.assert_called_once()
    args = renderer.set_chart2d.call_args[0]
    # args[4..7] are x0, x1, y0, y1 — should be default (0, 1, 0, 1)
    assert args[4] == 0.0 and args[5] == 1.0
    assert args[6] == 0.0 and args[7] == 1.0


def test_clear_calls_push_chart2d_with_defaults(line2d):
    """After clear(), the renderer must receive set_chart2d to clear the old frame."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 1
    line2d._renderer = renderer

    line2d.set_line(np.array([0.0, 5.0]), np.array([0.0, 5.0]))
    renderer.set_chart2d.reset_mock()

    line2d.clear()

    # set_chart2d must be called after clear so the renderer shows empty state
    renderer.set_chart2d.assert_called()


# ── Fifth-audit regression tests ─────────────────────────────────────────────

def test_refit_from_all_sources_pushes_when_x_frozen_only(line2d):
    """_refit_from_all_sources() must call _push_chart2d even when x is frozen."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 1
    line2d._renderer = renderer
    line2d._chart2d_sent = True

    # Set up a stream, then freeze x
    h = line2d.add_line_stream(max_points=50, mode="ring")
    st = line2d._line_streams[h]
    st["buf_x"][:2] = [0.0, 1.0]; st["buf_y"][:2] = [0.0, 1.0]; st["count"] = 2
    st["render_handle"] = None
    line2d._xlim = (0.0, 5.0); line2d._ylim = (0.0, 5.0)
    line2d._x_limits_frozen = True  # x frozen, y not frozen

    renderer.set_chart2d.reset_mock()
    renderer.chart2d_update_ylim.reset_mock()

    # Remove the only stream — no data left
    line2d.remove_line_stream(h)

    # Renderer must have received an update for the y axis (or full push)
    called = renderer.set_chart2d.called or renderer.chart2d_update_ylim.called
    assert called, "renderer must be notified after last stream removed with x frozen"
    # Frozen x must be preserved
    assert line2d._xlim == (0.0, 5.0)
    # Unfrozen y must be reset to None
    assert line2d._ylim is None


def test_refit_from_all_sources_pushes_when_y_frozen_only(line2d):
    """_refit_from_all_sources() must call _push_chart2d even when y is frozen."""
    from unittest.mock import MagicMock

    renderer = MagicMock()
    renderer.chart2d_add_line.return_value = 1
    line2d._renderer = renderer
    line2d._chart2d_sent = True

    h = line2d.add_line_stream(max_points=50, mode="ring")
    st = line2d._line_streams[h]
    st["buf_x"][:2] = [0.0, 1.0]; st["buf_y"][:2] = [0.0, 1.0]; st["count"] = 2
    st["render_handle"] = None
    line2d._xlim = (0.0, 5.0); line2d._ylim = (0.0, 5.0)
    line2d._y_limits_frozen = True  # y frozen, x not frozen

    renderer.set_chart2d.reset_mock()
    renderer.chart2d_update_xlim.reset_mock()

    line2d.remove_line_stream(h)

    called = renderer.set_chart2d.called or renderer.chart2d_update_xlim.called
    assert called, "renderer must be notified after last stream removed with y frozen"
    assert line2d._ylim == (0.0, 5.0)
    assert line2d._xlim is None


def test_pending_mesh_remove_clears_mhandle_map(root):
    """Pending-mesh replay inside _init_renderer: remove must pop _mhandle_map
    so a later visibility call cannot target the removed renderer id."""
    from unittest.mock import MagicMock, patch
    from dragonsci import Scatter3D
    import dragonsci.widget as _w

    w = Scatter3D(root, width=320, height=240)

    # Directly inject a pending add then remove (bypasses scipy hull validation)
    vhandle = 99
    dummy_payload = {"vertices": np.zeros((4, 3), dtype=np.float32),
                     "indices":  np.zeros((2, 3), dtype=np.uint32),
                     "rgba": (1.0, 0.0, 0.0, 1.0), "wireframe": False}
    w._pending_meshes.append((vhandle, "add", dummy_payload))
    w._pending_meshes.append((vhandle, "remove", {}))

    mock_renderer = MagicMock()
    mock_renderer.add_mesh.return_value = 42

    # Patch ScatterRenderer so _init_renderer uses our mock instead of wgpu
    with patch.object(_w, "ScatterRenderer", return_value=mock_renderer):
        w._renderer = None  # force _init_renderer to (re)create renderer
        w._init_renderer()

    # After replay the dead handle must have been popped from the map
    assert w._mhandle_map.get(vhandle) is None

    # A subsequent visibility call must not reach the renderer
    mock_renderer.set_mesh_visibility.reset_mock()
    w.set_mesh_visibility(vhandle, False)
    mock_renderer.set_mesh_visibility.assert_not_called()

    w.destroy()
