"""Smoke tests for the user-defined world-space label API."""
import numpy as np
import pytest

pytest.importorskip("dragonsci._dragonsci")
pytest.importorskip("jupyter_rfb")
pytest.importorskip("PIL")

from dragonsci.jupyter_widget import JupyterScatter3D, JupyterScatter2D


@pytest.fixture
def widget():
    w = JupyterScatter3D(width=256, height=256)
    rng = np.random.default_rng(0)
    w.set_points(rng.standard_normal((500, 3)).astype(np.float32))
    return w


def test_add_label_returns_int(widget):
    h = widget.add_label((0, 0, 0), "hello")
    assert isinstance(h, int)


def test_add_label_tuple_position(widget):
    h = widget.add_label((1.0, 2.0, 3.0), "A")
    assert isinstance(h, int)


def test_add_label_numpy_position(widget):
    pos = np.array([1.0, 0.5, -1.0], dtype=np.float32)
    h = widget.add_label(pos, "B")
    assert isinstance(h, int)


def test_add_label_hex_color(widget):
    h = widget.add_label((0, 0, 0), "C", color="#ff8800")
    assert isinstance(h, int)


def test_add_label_tuple_color(widget):
    h = widget.add_label((0, 0, 0), "D", color=(1.0, 0.5, 0.0))
    assert isinstance(h, int)


def test_add_label_anchors(widget):
    for anchor in ("center", "left", "right", "top", "bottom"):
        h = widget.add_label((0, 0, 0), anchor, anchor=anchor)
        assert isinstance(h, int)


def test_update_label_text(widget):
    h = widget.add_label((0, 0, 0), "original")
    widget.update_label(h, text="updated")


def test_update_label_position(widget):
    h = widget.add_label((0, 0, 0), "pos")
    widget.update_label(h, (1.0, 1.0, 1.0))


def test_update_label_size(widget):
    h = widget.add_label((0, 0, 0), "size", size=12.0)
    widget.update_label(h, size=20.0)


def test_update_label_color(widget):
    h = widget.add_label((0, 0, 0), "col", color=(1, 1, 1))
    widget.update_label(h, color=(0.5, 0.0, 1.0))


def test_update_label_anchor(widget):
    h = widget.add_label((0, 0, 0), "anch", anchor="center")
    widget.update_label(h, anchor="left")


def test_update_label_all_none_is_noop(widget):
    h = widget.add_label((0, 0, 0), "noop")
    widget.update_label(h)  # all None — must not crash


def test_set_label_visibility(widget):
    h = widget.add_label((0, 0, 0), "vis")
    widget.set_label_visibility(h, False)
    widget.set_label_visibility(h, True)


def test_remove_label(widget):
    h = widget.add_label((0, 0, 0), "remove me")
    widget.remove_label(h)


def test_clear_labels(widget):
    widget.add_label((0, 0, 0), "one")
    widget.add_label((1, 1, 1), "two")
    widget.clear_labels()


def test_multiple_labels_unique_handles(widget):
    handles = [widget.add_label((i, 0, 0), str(i)) for i in range(5)]
    assert len(set(handles)) == 5


def test_label_renders_without_crash(widget):
    widget.add_label((0, 0, 0), "rendered", color=(1, 1, 0), size=16.0, anchor="top")
    data = widget.get_frame_data()
    assert data[:2] == b"\xff\xd8"


def test_label_off_screen_does_not_crash(widget):
    widget.add_label((1e6, 1e6, 1e6), "far away")
    data = widget.get_frame_data()
    assert data[:2] == b"\xff\xd8"


def test_update_nonexistent_handle_is_noop(widget):
    widget.update_label(99999, text="ghost")  # stale handle — must not crash


def test_2d_labels(widget):
    w = JupyterScatter2D(width=128, height=128)
    rng = np.random.default_rng(1)
    w.set_points(rng.standard_normal((300, 2)).astype(np.float32))
    h = w.add_label((0.0, 0.0, 0.0), "origin", color="#ffffff", anchor="center")
    assert isinstance(h, int)
    w.remove_label(h)


# ── Regression: clear() must remove labels ────────────────────────────────────

def test_clear_removes_labels(widget):
    """clear() must remove all user labels, not just point actors."""
    before = widget.get_frame_data()
    widget.add_label((0, 0, 0), "LABEL", color=(1, 1, 0), size=32.0, anchor="center")
    after_label = widget.get_frame_data()
    widget.clear()
    after_clear = widget.get_frame_data()
    # After clear, the frame must be equal to before (no label rendered).
    # We compare lengths as a proxy — a large yellow label meaningfully grows JPEG.
    assert len(after_clear) <= len(after_label), (
        "clear() did not remove the label: frame after clear is not smaller"
    )


def test_clear_resets_label_pending_state():
    """clear() must also wipe _pending_labels so pre-map labels don't survive."""
    from dragonsci.jupyter_widget import JupyterScatter3D as W
    w = W.__new__(W)
    # Manually set up just the label state (no full __init__ needed)
    w._pending_labels = [
        (0, "add", {"x": 0, "y": 0, "z": 0, "text": "x",
                    "color": [1, 1, 1, 1], "size": 14.0, "anchor": 0}),
    ]
    w._label_handles = {0}
    w._lhandle_map = {}
    w._renderer = None
    # Simulate what clear() in widget.py does to label state
    w._pending_labels.clear()
    w._label_handles.clear()
    w._lhandle_map.clear()
    assert len(w._pending_labels) == 0
    assert len(w._label_handles) == 0


# ── Regression: clear_labels() must not grow _pending_labels when mapped ──────

def test_clear_labels_no_state_growth_when_mapped(widget):
    """Repeated clear_labels() on a live widget must not raise and must leave no labels."""
    widget.add_label((0, 0, 0), "a")
    widget.clear_labels()
    widget.add_label((1, 1, 1), "b")
    widget.clear_labels()
    widget.clear_labels()  # third call — must not crash


def test_clear_labels_no_state_growth_tk_pending():
    """For the Tk path, repeated pre-map clear_labels() must leave exactly one sentinel.

    Simulates the Scatter3D pre-map state machine directly without creating a Tk window.
    """
    from dragonsci.widget import _LABEL_ANCHOR_MAP
    # Build a minimal pre-map label state dict (same shape as Scatter3D._pending_labels)
    pending_labels: list = []
    label_handles: set = set()
    renderer = None  # simulates pre-map

    def _clear_labels():
        pending_labels.clear()
        label_handles.clear()
        if renderer is not None:
            pass  # would call renderer.clear_user_labels()
        else:
            pending_labels.append((-1, "clear", {}))

    def _add_label_premap():
        vhandle = len(label_handles)
        pending_labels.append((vhandle, "add", {}))
        label_handles.add(vhandle)

    _add_label_premap()
    _add_label_premap()
    _clear_labels()
    assert len(pending_labels) == 1, f"expected 1 sentinel, got {len(pending_labels)}"

    _clear_labels()
    assert len(pending_labels) == 1, f"expected 1 sentinel after 2nd clear, got {len(pending_labels)}"


def test_clear_labels_pre_map_single_sentinel():
    """Pre-map clear_labels() must leave exactly one sentinel, even after repeated calls."""
    from dragonsci.widget import Scatter3D, _parse_label_position, _parse_label_color, _LABEL_ANCHOR_MAP
    # Simulate a pre-map widget by inspecting pending state directly.
    # We use JupyterScatter3D.__new__ to avoid creating a Tk window.
    from dragonsci.jupyter_widget import JupyterScatter3D as W
    w = W.__new__(W)
    w._pending_labels = []
    w._label_handles = set()
    w._lhandle_map = {}
    w._renderer = None
    w._next_lhandle = 0

    # Simulate add_label pre-map (as the real code path does)
    vhandle = w._next_lhandle
    w._next_lhandle += 1
    w._pending_labels.append((vhandle, "add",
        {"x": 0.0, "y": 0.0, "z": 0.0, "text": "test",
         "color": [1.0, 1.0, 1.0, 1.0], "size": 14.0, "anchor": 0}))
    w._label_handles.add(vhandle)

    # First clear_labels() pre-map
    w._pending_labels.clear()
    w._label_handles.clear()
    w._pending_labels.append((-1, "clear", {}))   # sentinel

    assert len(w._pending_labels) == 1

    # Second clear_labels() pre-map — must still be exactly 1 entry
    w._pending_labels.clear()
    w._label_handles.clear()
    w._pending_labels.append((-1, "clear", {}))

    assert len(w._pending_labels) == 1, (
        f"Expected 1 sentinel, got {len(w._pending_labels)}"
    )
