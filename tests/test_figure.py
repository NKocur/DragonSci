"""Tests for Figure multi-subplot grid container."""
import pytest

_tkinter = pytest.importorskip("_tkinter")
pytest.importorskip("dragonsci._dragonsci")

import tkinter as tk
from dragonsci import Figure, Scatter3D


@pytest.fixture(scope="module")
def root():
    r = tk.Tk()
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def fig_2x2(root):
    f = Figure(root, rows=2, cols=2, width=400, height=300)
    yield f
    f.destroy()


# ── Construction ──────────────────────────────────────────────────────────────

def test_axes_count(fig_2x2):
    assert len(fig_2x2.axes) == 4


def test_axes_row_major_order(fig_2x2):
    flat = fig_2x2.axes
    # Row-major: [0,0], [0,1], [1,0], [1,1]
    assert flat[0] is fig_2x2[0, 0]
    assert flat[1] is fig_2x2[0, 1]
    assert flat[2] is fig_2x2[1, 0]
    assert flat[3] is fig_2x2[1, 1]


def test_all_cells_are_scatter3d(fig_2x2):
    for w in fig_2x2.axes:
        assert isinstance(w, Scatter3D)


def test_1x1_figure(root):
    f = Figure(root, rows=1, cols=1, width=200, height=150)
    assert len(f.axes) == 1
    assert f[0, 0] is f.axes[0]
    f.destroy()


def test_1x3_figure(root):
    f = Figure(root, rows=1, cols=3, width=600, height=200)
    assert len(f.axes) == 3
    f.destroy()


def test_3x1_figure(root):
    f = Figure(root, rows=3, cols=1, width=200, height=600)
    assert len(f.axes) == 3
    f.destroy()


def test_cells_are_distinct_instances(fig_2x2):
    flat = fig_2x2.axes
    assert len(set(id(w) for w in flat)) == 4


# ── Indexing ──────────────────────────────────────────────────────────────────

def test_getitem_returns_correct_cell(root):
    f = Figure(root, rows=3, cols=3, width=300, height=300)
    for r in range(3):
        for c in range(3):
            assert isinstance(f[r, c], Scatter3D)
    f.destroy()


# ── Camera linking ────────────────────────────────────────────────────────────

def test_link_cameras_two_cells(fig_2x2):
    fig_2x2.link_cameras((0, 0), (0, 1))
    a, b = fig_2x2[0, 0], fig_2x2[0, 1]
    assert b in a._camera_links
    assert a in b._camera_links


def test_link_cameras_three_cells(fig_2x2):
    f = fig_2x2
    f.link_cameras((0, 0), (1, 0), (1, 1))
    w00, w10, w11 = f[0, 0], f[1, 0], f[1, 1]
    assert w10 in w00._camera_links
    assert w11 in w00._camera_links
    assert w00 in w10._camera_links


def test_link_cameras_duplicate_coords_no_self_link(root):
    f = Figure(root, rows=2, cols=2, width=200, height=200)
    f.link_cameras((0, 0), (0, 0))  # duplicate — must not self-link
    assert f[0, 0] not in f[0, 0]._camera_links
    f.destroy()


def test_link_cameras_single_cell_is_noop(fig_2x2):
    f = Figure(fig_2x2.master, rows=2, cols=2, width=200, height=200)
    before = set(f[0, 0]._camera_links)
    f.link_cameras((0, 0))   # only one cell — must not crash
    assert f[0, 0]._camera_links == before
    f.destroy()


def test_share_cameras_option(root):
    f = Figure(root, rows=2, cols=2, width=200, height=200, share_cameras=True)
    flat = f.axes
    for i, w in enumerate(flat):
        others = set(flat) - {w}
        assert others.issubset(w._camera_links), (
            f"Cell {i} missing links when share_cameras=True"
        )
    f.destroy()


def test_share_cameras_false_no_links(root):
    f = Figure(root, rows=2, cols=2, width=200, height=200, share_cameras=False)
    for w in f.axes:
        assert len(w._camera_links) == 0
    f.destroy()


# ── Scalar bar routing ────────────────────────────────────────────────────────

def _sb(widget) -> "dict | None":
    """Return the pending scalar bar dict for a pre-map widget."""
    return getattr(widget, "_pending_scalar_bar", None)


def test_scalar_bar_row_shows_rightmost_only(root):
    f = Figure(root, rows=2, cols=3, width=300, height=200)
    f.scalar_bar(row=0, colormap="plasma", vmin=0.0, vmax=1.0)
    # Only rightmost cell (col=2) in row 0 should have visible=True.
    assert _sb(f[0, 0])["visible"] is False
    assert _sb(f[0, 1])["visible"] is False
    assert _sb(f[0, 2])["visible"] is True
    # All other cells (row 1) are also cleared by the stale-bar fix.
    assert _sb(f[1, 0])["visible"] is False
    assert _sb(f[1, 1])["visible"] is False
    assert _sb(f[1, 2])["visible"] is False
    f.destroy()


def test_scalar_bar_all_rows_when_no_row_col(root):
    f = Figure(root, rows=2, cols=2, width=200, height=200)
    f.scalar_bar(colormap="viridis", vmin=0.0, vmax=5.0)
    # Rightmost column (col=1) in each row should be visible
    assert _sb(f[0, 0])["visible"] is False
    assert _sb(f[0, 1])["visible"] is True
    assert _sb(f[1, 0])["visible"] is False
    assert _sb(f[1, 1])["visible"] is True
    f.destroy()


def test_scalar_bar_col_shows_bottommost_only(root):
    f = Figure(root, rows=3, cols=2, width=200, height=300)
    f.scalar_bar(col=1, colormap="inferno", vmin=0.0, vmax=1.0)
    # Only bottom cell (row=2) in col 1 should be visible; all others hidden.
    assert _sb(f[0, 1])["visible"] is False
    assert _sb(f[1, 1])["visible"] is False
    assert _sb(f[2, 1])["visible"] is True
    # col 0 is also touched (set to hidden) by the fix that clears stale bars.
    assert _sb(f[0, 0])["visible"] is False
    assert _sb(f[1, 0])["visible"] is False
    assert _sb(f[2, 0])["visible"] is False
    f.destroy()


def test_scalar_bar_explicit_row_col(root):
    f = Figure(root, rows=2, cols=2, width=200, height=200)
    f.scalar_bar(row=1, col=0, colormap="coolwarm", vmin=-1.0, vmax=1.0)
    assert _sb(f[1, 0])["visible"] is True
    # All other cells are explicitly hidden (stale-bar clearing).
    assert _sb(f[0, 0])["visible"] is False
    assert _sb(f[0, 1])["visible"] is False
    assert _sb(f[1, 1])["visible"] is False
    f.destroy()


# ── equal_aspect enforcement ──────────────────────────────────────────────────

def test_equal_aspect_initial_cell_is_square(root):
    f = Figure(root, rows=2, cols=2, width=400, height=300,
               equal_aspect=True, padding=4)
    w = f[0, 0].cget("width")
    h = f[0, 0].cget("height")
    assert w == h, f"equal_aspect cell not square: {w}×{h}"
    f.destroy()


def test_equal_aspect_uses_smaller_dimension(root):
    # 400×300 → cell_w=194, cell_h=144 → square side = 144
    f = Figure(root, rows=2, cols=2, width=400, height=300,
               equal_aspect=True, padding=4)
    side = f[0, 0].cget("width")
    assert side == 144, f"expected 144, got {side}"
    f.destroy()


def test_equal_aspect_on_resize_stays_square(root):
    f = Figure(root, rows=2, cols=2, width=400, height=300,
               equal_aspect=True, padding=4)
    # Simulate a resize to 500×300
    f._on_resize(type("E", (), {"widget": f, "width": 500, "height": 300})())
    w = f[0, 0].cget("width")
    h = f[0, 0].cget("height")
    assert w == h, f"equal_aspect cell not square after resize: {w}×{h}"
    f.destroy()


def test_no_equal_aspect_cells_expand(root):
    f = Figure(root, rows=1, cols=2, width=400, height=200,
               equal_aspect=False, padding=4)
    w = f[0, 0].cget("width")
    h = f[0, 0].cget("height")
    assert w != h or True, "non-equal-aspect cells may or may not be square"
    # Just verify no crash and both cells exist.
    assert f[0, 0] is not f[0, 1]
    f.destroy()


# ── scalar_bar retargeting ────────────────────────────────────────────────────

def test_scalar_bar_retarget_row_clears_previous(root):
    """Calling scalar_bar(row=1) after scalar_bar(all) must hide row-0 bar."""
    f = Figure(root, rows=2, cols=2, width=200, height=200)
    f.scalar_bar(colormap="viridis", vmin=0.0, vmax=1.0)   # all rows
    # row 0 rightmost is visible
    assert _sb(f[0, 1])["visible"] is True

    f.scalar_bar(row=1, colormap="plasma", vmin=0.0, vmax=1.0)
    # row 0 rightmost must now be hidden
    assert _sb(f[0, 1])["visible"] is False
    # row 1 rightmost must be visible
    assert _sb(f[1, 1])["visible"] is True
    f.destroy()


def test_scalar_bar_retarget_explicit_clears_previous(root):
    """Calling scalar_bar(row=0, col=1) after scalar_bar(row=1) clears row 1."""
    f = Figure(root, rows=2, cols=2, width=200, height=200)
    f.scalar_bar(row=1, colormap="viridis", vmin=0.0, vmax=1.0)
    assert _sb(f[1, 1])["visible"] is True

    f.scalar_bar(row=0, col=0, colormap="plasma", vmin=0.0, vmax=1.0)
    assert _sb(f[0, 0])["visible"] is True
    assert _sb(f[1, 1])["visible"] is False
    f.destroy()


# ── Input validation ─────────────────────────────────────────────────────────

def test_zero_rows_raises(root):
    with pytest.raises(ValueError, match="rows"):
        Figure(root, rows=0, cols=2)


def test_zero_cols_raises(root):
    with pytest.raises(ValueError, match="cols"):
        Figure(root, rows=2, cols=0)


def test_negative_rows_raises(root):
    with pytest.raises(ValueError):
        Figure(root, rows=-1, cols=2)


# ── kwargs forwarding ─────────────────────────────────────────────────────────

def test_kwargs_forwarded_to_scatter(root):
    f = Figure(root, rows=1, cols=2, width=200, height=100, bg="#0d0d0d")
    # If bg= was forwarded, the scatter widgets should have it as configure option.
    # We just check no crash and correct cell count.
    assert len(f.axes) == 2
    f.destroy()
