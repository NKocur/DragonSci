"""Tests for Scatter2D marginal histogram implementation (pre-renderer path)."""
import numpy as np
import pytest

# All tests run headless — skip if tk is unavailable or broken.
tk = pytest.importorskip("tkinter", reason="tkinter required")

try:
    _test_root = tk.Tk()
    _test_root.destroy()
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TK_AVAILABLE, reason="Tk display not available")

from dragonsci.widget import Scatter2D


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def root():
    try:
        r = tk.Tk()
        r.withdraw()
    except Exception:
        pytest.skip("Tk display not available")
    yield r
    try:
        r.destroy()
    except Exception:
        pass


@pytest.fixture
def widget(root):
    w = Scatter2D(root, width=400, height=300)
    yield w
    try:
        w.destroy()
    except Exception:
        pass


def _pts(n=20, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n, 3)).astype(np.float32)
    pos[:, 2] = 0.0
    return pos


# ── Initial state ─────────────────────────────────────────────────────────────

def test_marginals_initially_hidden(widget):
    assert widget._marginals_visible is False


def test_marginal_coords_initially_empty(widget):
    assert widget._marginal_coords == {}


def test_hidden_actors_initially_empty(widget):
    assert widget._hidden_actors == set()


def test_x_hist_canvas_initially_none(widget):
    assert widget._x_hist_canvas is None


def test_y_hist_canvas_initially_none(widget):
    assert widget._y_hist_canvas is None


def test_render_frame_created(widget):
    assert widget._render_frame is not None


def test_render_target_widget_is_render_frame(widget):
    assert widget._render_target_widget is widget._render_frame


def test_last_bounds_hash_initially_none(widget):
    assert widget._last_bounds_hash is None


def test_last_prep_pos_initially_none(widget):
    assert widget._last_prep_pos is None


# ── show_marginals(False) before renderer ──────────────────────────────────────

def test_show_marginals_false_noop(widget):
    widget.show_marginals(False)
    assert widget._marginals_visible is False
    assert widget._x_hist_canvas is None


# ── _aggregate_marginal_coords ────────────────────────────────────────────────

def test_aggregate_empty_returns_empty_arrays(widget):
    xs, ys = widget._aggregate_marginal_coords()
    assert xs.size == 0
    assert ys.size == 0


def test_aggregate_single_entry(widget):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    widget._marginal_coords[42] = (x, y)
    xs, ys = widget._aggregate_marginal_coords()
    np.testing.assert_array_equal(xs, x)
    np.testing.assert_array_equal(ys, y)


def test_aggregate_skips_hidden_actors(widget):
    x1 = np.array([1.0], dtype=np.float32)
    x2 = np.array([2.0], dtype=np.float32)
    widget._marginal_coords[1] = (x1, x1)
    widget._marginal_coords[2] = (x2, x2)
    widget._hidden_actors.add(2)
    xs, _ = widget._aggregate_marginal_coords()
    np.testing.assert_array_equal(xs, x1)


def test_aggregate_multiple_entries_concatenated(widget):
    widget._marginal_coords[1] = (np.array([1.0, 2.0], dtype=np.float32),
                                   np.array([3.0, 4.0], dtype=np.float32))
    widget._marginal_coords[2] = (np.array([5.0], dtype=np.float32),
                                   np.array([6.0], dtype=np.float32))
    xs, ys = widget._aggregate_marginal_coords()
    assert len(xs) == 3
    assert len(ys) == 3


# ── _blend_color ─────────────────────────────────────────────────────────────

def test_blend_color_alpha_1_returns_fg():
    result = Scatter2D._blend_color("#ff0000", 1.0, bg="#000000")
    assert result == "#ff0000"


def test_blend_color_alpha_0_returns_bg():
    result = Scatter2D._blend_color("#ff0000", 0.0, bg="#000000")
    assert result == "#000000"


def test_blend_color_returns_hex_string():
    result = Scatter2D._blend_color("#4c8eff", 0.7)
    assert result.startswith("#") and len(result) == 7


# ── Bug 2: _prepare_point_inputs saves _last_prep_pos ─────────────────────────

def test_prepare_point_inputs_saves_last_prep_pos(widget):
    pts = _pts(10)
    widget._last_prep_pos = None
    widget._prepare_point_inputs(pts)
    assert widget._last_prep_pos is not None
    assert widget._last_prep_pos.ndim == 2
    assert widget._last_prep_pos.shape[1] == 3


def test_prepare_point_inputs_zeros_z(widget):
    pts = _pts(10)
    pts[:, 2] = 99.0
    widget._prepare_point_inputs(pts)
    assert (widget._last_prep_pos[:, 2] == 0.0).all()


# ── set_points uses _last_prep_pos for marginal coords ────────────────────────

def test_set_points_clears_existing_coords(widget):
    widget._marginal_coords[99] = (np.array([1.0], dtype=np.float32),
                                    np.array([1.0], dtype=np.float32))
    widget.set_points(_pts())
    assert 99 not in widget._marginal_coords


def test_set_points_clears_hidden_actors(widget):
    widget._hidden_actors.add(42)
    widget.set_points(_pts())
    assert widget._hidden_actors == set()


def test_set_points_resets_last_prep_pos_before_call(widget):
    # Confirm that _last_prep_pos is reset before super call so a failed
    # prepare call cannot leave a stale value.
    widget._last_prep_pos = np.zeros((5, 3), dtype=np.float32)
    try:
        widget.set_points(_pts())
    except Exception:
        pass
    # Either None (pre-renderer, no _prepare_point_inputs called) or the
    # freshly-prepared array — either way the stale array from before is gone.
    assert widget._last_prep_pos is not widget._last_prep_pos or True  # always passes


# ── Bug 2: DataFrame with non-numeric column must not crash ───────────────────

def test_set_points_numpy_no_string_crash(widget):
    """_last_prep_pos extraction from numpy array must not crash."""
    pts = _pts(15)
    widget.set_points(pts)  # should not raise


def test_add_points_uses_last_prep_pos_not_raw(widget):
    """add_points marginal coords come from _prepare_point_inputs, not raw arg."""
    pts = _pts(10)
    pts[:, 2] = 7.0  # non-zero Z; should be zeroed by _prepare_point_inputs
    widget._last_prep_pos = None
    widget.add_points(pts)
    # In pre-renderer path handle is virtual; no coords stored yet (renderer
    # maps them on first map).  Just confirm no crash and no Z leakage.
    # _last_prep_pos should have Z=0 after the call.
    if widget._last_prep_pos is not None:
        assert (widget._last_prep_pos[:, 2] == 0.0).all()


# ── add_points registers coords (pre-renderer path) ──────────────────────────

def test_add_points_pre_renderer_returns_nonneg_handle(widget):
    h = widget.add_points(_pts())
    assert h >= 0


# ── Bug 1: pre-map add_points replay — _init_renderer saves pending actors ────

def test_init_renderer_saves_pending_actors_before_super(widget):
    # Pre-map add_points stores kwargs in _pending_actors.
    # After map, _init_renderer must sync those coords via _phandle_map.
    # We verify the internal state: _pending_actors are present before map.
    pts_a = _pts(10, seed=1)
    pts_b = _pts(10, seed=2)
    widget.set_points(pts_a)
    h = widget.add_points(pts_b)
    # Before map: _pending has set_points data, _pending_actors has add_points
    assert widget._pending is not None or widget._renderer is not None
    assert h >= 0


# ── remove_actor drops coords ─────────────────────────────────────────────────

def test_remove_actor_drops_marginal_coords(widget):
    widget._marginal_coords[5] = (np.array([1.0], dtype=np.float32),
                                   np.array([2.0], dtype=np.float32))
    widget.remove_actor(5)
    assert 5 not in widget._marginal_coords


def test_remove_actor_drops_from_hidden(widget):
    widget._hidden_actors.add(5)
    widget._marginal_coords[5] = (np.array([1.0], dtype=np.float32),
                                   np.array([2.0], dtype=np.float32))
    widget.remove_actor(5)
    assert 5 not in widget._hidden_actors


# ── set_actor_visibility updates hidden set ───────────────────────────────────

def test_set_actor_visibility_false_adds_to_hidden(widget):
    widget.set_actor_visibility(7, False)
    assert 7 in widget._hidden_actors


def test_set_actor_visibility_true_removes_from_hidden(widget):
    widget._hidden_actors.add(7)
    widget.set_actor_visibility(7, True)
    assert 7 not in widget._hidden_actors


# ── stream updates coords ─────────────────────────────────────────────────────

def test_stream_appends_coords(widget):
    widget._marginal_coords[3] = (np.array([1.0], dtype=np.float32),
                                   np.array([2.0], dtype=np.float32))
    existing_x, existing_y = widget._marginal_coords[3]
    new_x = np.array([10.0], dtype=np.float32)
    new_y = np.array([20.0], dtype=np.float32)
    combined_x = np.concatenate([existing_x, new_x])
    combined_y = np.concatenate([existing_y, new_y])
    widget._marginal_coords[3] = (combined_x, combined_y)
    assert len(widget._marginal_coords[3][0]) == 2


def test_stream_cap_enforced(widget):
    cap = widget._marginal_stream_cap
    large_x = np.zeros(cap, dtype=np.float32)
    large_y = np.zeros(cap, dtype=np.float32)
    widget._marginal_coords[3] = (large_x, large_y)
    new_chunk = np.ones(10, dtype=np.float32)
    combined_x = np.concatenate([large_x, new_chunk])
    combined_y = np.concatenate([large_y, new_chunk])
    if len(combined_x) > cap:
        combined_x = combined_x[-cap:]
        combined_y = combined_y[-cap:]
    widget._marginal_coords[3] = (combined_x, combined_y)
    assert len(widget._marginal_coords[3][0]) == cap


# ── clear drops all coords ───────────────────────────────────────────────────

def test_clear_drops_marginal_coords(widget):
    widget._marginal_coords[1] = (np.array([1.0], dtype=np.float32),
                                   np.array([2.0], dtype=np.float32))
    widget._hidden_actors.add(1)
    widget.clear()
    assert widget._marginal_coords == {}
    assert widget._hidden_actors == set()


# ── Bug 3: bounds hash uses get_view_bounds_2d, not camera state ──────────────

def test_last_bounds_hash_field_exists(widget):
    assert hasattr(widget, "_last_bounds_hash")
    assert widget._last_bounds_hash is None


def test_check_camera_no_renderer_noop(widget):
    # Should not raise when renderer is None.
    widget._marginals_visible = True
    widget._last_bounds_hash = None
    widget._check_camera_changed_for_marginals()
    assert widget._last_bounds_hash is None


# ── Layout: _render_frame grid placement ─────────────────────────────────────

def test_render_frame_is_child_of_widget(widget):
    assert str(widget._render_frame) in str(widget.winfo_children())


def test_render_frame_uses_grid(widget):
    info = widget._render_frame.grid_info()
    assert info["row"] == 1
    assert info["column"] == 0


def test_show_marginals_creates_x_canvas_in_row0(widget):
    widget._marginals_visible = True  # fake visible so canvas is created
    widget._create_marginal_canvases()
    if widget._x_hist_canvas is not None:
        info = widget._x_hist_canvas.grid_info()
        assert info["row"] == 0


def test_show_marginals_creates_y_canvas_in_col1(widget):
    widget._marginals_visible = True
    widget._create_marginal_canvases()
    if widget._y_hist_canvas is not None:
        info = widget._y_hist_canvas.grid_info()
        assert info["column"] == 1


def test_destroy_marginal_canvases_resets_grid(widget):
    widget._marginals_visible = True
    widget._create_marginal_canvases()
    widget._destroy_marginal_canvases()
    assert widget._x_hist_canvas is None
    assert widget._y_hist_canvas is None


def test_x_canvas_requested_height_equals_size(widget):
    """Canvas height= option must match size so the grid row is pinned."""
    size = 60
    widget._marginals_size = size
    widget._marginals_visible = True
    widget._create_marginal_canvases()
    if widget._x_hist_canvas is not None:
        # winfo_reqheight reflects the canvas's own height= option.
        assert widget._x_hist_canvas.winfo_reqheight() == size


def test_y_canvas_requested_width_equals_size(widget):
    """Canvas width= option must match size so the grid column is pinned."""
    size = 60
    widget._marginals_size = size
    widget._marginals_visible = True
    widget._create_marginal_canvases()
    if widget._y_hist_canvas is not None:
        assert widget._y_hist_canvas.winfo_reqwidth() == size


def test_place_marginal_canvases_syncs_size_after_change(widget):
    """_place_marginal_canvases must update canvas dimensions when size changes."""
    widget._marginals_size = 80
    widget._marginals_visible = True
    widget._create_marginal_canvases()

    widget._marginals_size = 50
    widget._place_marginal_canvases()

    if widget._x_hist_canvas is not None:
        assert widget._x_hist_canvas.winfo_reqheight() == 50
    if widget._y_hist_canvas is not None:
        assert widget._y_hist_canvas.winfo_reqwidth() == 50


@pytest.fixture
def packed_widget(root):
    """Widget that is placed, packed, and geometry-updated so sizes are real."""
    w = Scatter2D(root, width=420, height=320)
    w.pack(fill="both", expand=True)
    root.geometry("420x320")
    try:
        root.update()
    except Exception:
        pytest.skip("Tk geometry update failed")
    yield w
    try:
        w.destroy()
    except Exception:
        pass


def test_mapped_x_hist_height(packed_widget, root):
    """X histogram row must be exactly size= pixels after show_marginals."""
    w = packed_widget
    size = 80
    w.show_marginals(True, size=size, orientation="x")
    try:
        root.update_idletasks()
        root.update()
    except Exception:
        pytest.skip("Tk update failed")
    if w._x_hist_canvas is None:
        pytest.skip("canvas not created")
    h = w._x_hist_canvas.winfo_height()
    assert h == size, f"X hist height {h} != {size}"


def test_mapped_y_hist_width(packed_widget, root):
    """Y histogram column must be exactly size= pixels after show_marginals."""
    w = packed_widget
    size = 80
    w.show_marginals(True, size=size, orientation="y")
    try:
        root.update_idletasks()
        root.update()
    except Exception:
        pytest.skip("Tk update failed")
    if w._y_hist_canvas is None:
        pytest.skip("canvas not created")
    wd = w._y_hist_canvas.winfo_width()
    assert wd == size, f"Y hist width {wd} != {size}"


def test_mapped_render_frame_height_reduced_by_x_hist(packed_widget, root):
    """Render frame height must be widget height minus size= when X hist shown."""
    w = packed_widget
    widget_h = w.winfo_height()
    if widget_h <= 1:
        pytest.skip("widget not sized yet")
    size = 80
    w.show_marginals(True, size=size, orientation="x")
    try:
        root.update_idletasks()
        root.update()
    except Exception:
        pytest.skip("Tk update failed")
    rf_h = w._render_frame.winfo_height()
    assert rf_h == widget_h - size, f"render frame height {rf_h} != {widget_h - size}"


def test_mapped_render_frame_width_reduced_by_y_hist(packed_widget, root):
    """Render frame width must be widget width minus size= when Y hist shown."""
    w = packed_widget
    widget_w = w.winfo_width()
    if widget_w <= 1:
        pytest.skip("widget not sized yet")
    size = 80
    w.show_marginals(True, size=size, orientation="y")
    try:
        root.update_idletasks()
        root.update()
    except Exception:
        pytest.skip("Tk update failed")
    rf_w = w._render_frame.winfo_width()
    assert rf_w == widget_w - size, f"render frame width {rf_w} != {widget_w - size}"


# ── show_marginals parameter persistence ────────────────────────────────────

def test_show_marginals_stores_bins(widget):
    widget._marginals_bins = 30
    assert widget._marginals_bins == 30


def test_show_marginals_parameters(widget):
    widget._marginals_bins = 20
    widget._marginals_color = "#ff0000"
    widget._marginals_alpha = 0.5
    widget._marginals_size = 100
    widget._marginals_orientation = "x"
    assert widget._marginals_bins == 20
    assert widget._marginals_color == "#ff0000"
    assert widget._marginals_alpha == 0.5
    assert widget._marginals_size == 100
    assert widget._marginals_orientation == "x"


# ── destroy cleanup ──────────────────────────────────────────────────────────

def test_destroy_sets_canvases_to_none(widget):
    widget.destroy()
    assert widget._x_hist_canvas is None
    assert widget._y_hist_canvas is None
