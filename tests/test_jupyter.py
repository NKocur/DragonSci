"""Headless smoke tests for JupyterScatter3D / JupyterScatter2D."""
import numpy as np
import pytest

dragonsci = pytest.importorskip("dragonsci._dragonsci")
jupyter_rfb = pytest.importorskip("jupyter_rfb")
PIL = pytest.importorskip("PIL")

from dragonsci.jupyter_widget import JupyterScatter3D, JupyterScatter2D


@pytest.fixture
def widget():
    w = JupyterScatter3D(width=128, height=128)
    yield w


def _random_pts(n=500):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 3)).astype(np.float32)


def test_create_offscreen():
    w = JupyterScatter3D(width=64, height=64)
    assert w._width == 64 and w._height == 64


def test_get_frame_data_returns_jpeg(widget):
    pts = _random_pts()
    widget.set_points(pts, colormap="viridis")
    data = widget.get_frame_data()
    assert isinstance(data, bytes)
    assert data[:2] == b"\xff\xd8", "expected JPEG magic bytes"


def test_set_and_add_points(widget):
    pts = _random_pts()
    widget.set_points(pts)
    h = widget.add_points(_random_pts(200))
    assert isinstance(h, int)
    widget.remove_actor(h)


def test_update_actor(widget):
    pts = _random_pts()
    h = widget.add_points(pts)
    widget.update_actor(h, _random_pts(300))


def test_clear(widget):
    widget.set_points(_random_pts())
    widget.clear()
    frame = widget.get_frame_data()
    assert len(frame) > 0


def test_camera_round_trip(widget):
    widget.set_points(_random_pts())
    cam = widget.get_camera()
    widget.set_camera(cam)


def test_resize_event(widget):
    widget.handle_event({"event_type": "resize", "width": 256, "height": 200})
    assert widget._width == 256 and widget._height == 200


def test_pointer_drag_orbit(widget):
    widget.set_points(_random_pts())
    widget.handle_event({"event_type": "pointer_down", "button": 1, "x": 64, "y": 64})
    widget.handle_event({"event_type": "pointer_move", "x": 80, "y": 70})
    widget.handle_event({"event_type": "pointer_up"})


def test_wheel_zoom(widget):
    widget.set_points(_random_pts())
    widget.handle_event({"event_type": "wheel", "dy": 120})


def test_double_click_reset(widget):
    widget.set_points(_random_pts())
    widget.handle_event({"event_type": "double_click"})


def test_screenshot_shape(widget):
    widget.set_points(_random_pts())
    arr = widget.screenshot()
    assert arr.shape == (128, 128, 4)
    assert arr.dtype == np.uint8


def test_parallel_projection(widget):
    widget.parallel_projection = True
    assert widget.parallel_projection is True
    widget.parallel_projection = False


def test_set_background_str(widget):
    widget.set_background("#1a2b3c")


def test_set_background_tuple(widget):
    widget.set_background((0.1, 0.2, 0.3))


def test_show_grid(widget):
    widget.show_grid(True)
    widget.show_grid(False)


def test_set_axes(widget):
    widget.set_axes("PC1", "PC2", "PC3")


def test_scalar_bar(widget):
    widget.scalar_bar(True, vmin=0.0, vmax=1.0, colormap="plasma", title="val")
    widget.scalar_bar(False)


def test_colormap_names():
    names = JupyterScatter3D.colormap_names()
    assert isinstance(names, list)
    assert "viridis" in names


def test_2d_widget_3d_input():
    w = JupyterScatter2D(width=128, height=128)
    assert w.parallel_projection is True
    pts = _random_pts()
    w.set_points(pts)
    data = w.get_frame_data()
    assert data[:2] == b"\xff\xd8"


def test_2d_widget_2d_input():
    w = JupyterScatter2D(width=128, height=128)
    rng = np.random.default_rng(1)
    pts2d = rng.standard_normal((300, 2)).astype(np.float32)
    w.set_points(pts2d)  # must not raise
    data = w.get_frame_data()
    assert data[:2] == b"\xff\xd8"


def test_2d_widget_add_points_2d():
    w = JupyterScatter2D(width=128, height=128)
    rng = np.random.default_rng(2)
    pts2d = rng.standard_normal((200, 2)).astype(np.float32)
    h = w.add_points(pts2d)
    assert isinstance(h, int)
    w.remove_actor(h)


def test_2d_widget_update_actor_2d():
    w = JupyterScatter2D(width=128, height=128)
    rng = np.random.default_rng(3)
    h = w.add_points(rng.standard_normal((100, 2)).astype(np.float32))
    w.update_actor(h, rng.standard_normal((150, 2)).astype(np.float32))


def test_picking_enable_disable():
    w = JupyterScatter3D(width=128, height=128)
    results = []
    w.enable_point_picking(on_pick=results.append)
    assert w._on_pick_cb is not None
    w.disable_picking()
    assert w._on_pick_cb is None


def test_picking_click_fires_callback():
    w = JupyterScatter3D(width=128, height=128)
    rng = np.random.default_rng(0)
    w.set_points(rng.standard_normal((1000, 3)).astype(np.float32))
    results = []
    w.enable_point_picking(on_pick=results.append)
    # Simulate a click: pointer_down then pointer_up at same position
    w.handle_event({"event_type": "pointer_down", "button": 1, "x": 64.0, "y": 64.0})
    w.handle_event({"event_type": "pointer_up", "x": 64.0, "y": 64.0})
    # Callback may or may not fire (depends on whether a point is under cursor),
    # but it must not raise.  If a hit occurred, the result dict has expected keys.
    for r in results:
        assert "actor" in r and "index" in r and "point" in r


def test_picking_drag_does_not_fire_callback():
    w = JupyterScatter3D(width=128, height=128)
    w.set_points(np.zeros((100, 3), dtype=np.float32))
    fired = []
    w.enable_point_picking(on_pick=fired.append)
    # Simulate a drag (>5px movement) — must NOT trigger pick
    w.handle_event({"event_type": "pointer_down", "button": 1, "x": 0.0, "y": 0.0})
    w.handle_event({"event_type": "pointer_move", "x": 30.0, "y": 30.0})
    w.handle_event({"event_type": "pointer_up", "x": 30.0, "y": 30.0})
    assert len(fired) == 0


# ── Regression tests for the three findings fixed in this pass ────────────────

def test_2d_parallel_projection_locked_after_set_points():
    """set_points must not allow the camera to lose parallel projection."""
    w = JupyterScatter2D(width=128, height=128)
    rng = np.random.default_rng(0)
    w.set_points(rng.standard_normal((500, 2)).astype(np.float32))
    assert w.get_camera()["parallel"] is True


def test_2d_view_isometric_is_noop():
    """view_isometric must re-lock to the 2D front view, not go perspective."""
    w = JupyterScatter2D(width=128, height=128)
    w.set_points(_random_pts())
    cam_before = w.get_camera()
    w.view_isometric()
    cam_after = w.get_camera()
    assert cam_after["parallel"] is True
    # pitch and yaw should match the locked 2D front view, not an isometric angle
    assert abs(cam_after["pitch"] - cam_before["pitch"]) < 1e-3
    assert abs(cam_after["yaw"] - cam_before["yaw"]) < 1e-3


def test_2d_set_camera_relocks():
    """set_camera with a perspective dict must still restore parallel projection."""
    w = JupyterScatter2D(width=128, height=128)
    w.set_points(_random_pts())
    perspective_cam = w.get_camera()
    perspective_cam["parallel"] = False
    w.set_camera(perspective_cam)
    assert w.get_camera()["parallel"] is True


def test_picking_works_when_enabled_after_set_points():
    """Picking must work when enable_point_picking() is called after data is loaded."""
    w = JupyterScatter3D(width=128, height=128)
    # All points at the centre so a click at (64, 64) should always hit
    pts = np.zeros((200, 3), dtype=np.float32)
    w.set_points(pts)
    w.fit()  # ensure points fill the view
    fired = []
    w.enable_point_picking(on_pick=fired.append)
    w.handle_event({"event_type": "pointer_down", "button": 1, "x": 64.0, "y": 64.0})
    w.handle_event({"event_type": "pointer_up", "x": 64.0, "y": 64.0})
    assert len(fired) == 1, "pick_point() returned None — CPU positions not stored"
    assert "index" in fired[0]
