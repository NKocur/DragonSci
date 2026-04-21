"""Tests for statistical overlay helpers and Scatter3D mesh API (pre-renderer path)."""
import numpy as np
import pytest

scipy = pytest.importorskip("scipy", reason="scipy required for these tests")

from dragonsci.widget import _compute_convex_hull, _compute_ellipsoid


# ── _compute_convex_hull ──────────────────────────────────────────────────────

def _cube_points():
    return np.array([
        [0,0,0],[1,0,0],[0,1,0],[1,1,0],
        [0,0,1],[1,0,1],[0,1,1],[1,1,1],
    ], dtype=np.float32)


def test_hull_returns_float32_verts():
    verts, _ = _compute_convex_hull(_cube_points())
    assert verts.dtype == np.float32


def test_hull_returns_uint32_indices():
    _, idxs = _compute_convex_hull(_cube_points())
    assert idxs.dtype == np.uint32


def test_hull_indices_in_range():
    verts, idxs = _compute_convex_hull(_cube_points())
    assert idxs.max() < len(verts)


def test_hull_triangles_shape():
    _, idxs = _compute_convex_hull(_cube_points())
    assert idxs.ndim == 2 and idxs.shape[1] == 3


def test_hull_fewer_than_4_points_raises():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32)
    with pytest.raises(Exception):
        _compute_convex_hull(pts)


def test_hull_verts_subset_of_input():
    pts = _cube_points()
    verts, _ = _compute_convex_hull(pts)
    # Every hull vertex must equal some original point.
    for v in verts:
        assert any(np.allclose(v, p) for p in pts), f"vertex {v} not in input"


def test_hull_accepts_float64_input():
    pts = _cube_points().astype(np.float64)
    verts, idxs = _compute_convex_hull(pts)
    assert verts.dtype == np.float32


def test_hull_large_random():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((500, 3)).astype(np.float32)
    verts, idxs = _compute_convex_hull(pts)
    assert len(verts) > 0
    assert len(idxs) > 0


# ── _compute_ellipsoid ────────────────────────────────────────────────────────

def _identity_cov():
    return np.eye(3, dtype=np.float64)


def _identity_center():
    return np.array([0.0, 0.0, 0.0])


def test_ellipsoid_returns_float32_verts():
    verts, _ = _compute_ellipsoid(_identity_center(), _identity_cov())
    assert verts.dtype == np.float32


def test_ellipsoid_returns_uint32_indices():
    _, idxs = _compute_ellipsoid(_identity_center(), _identity_cov())
    assert idxs.dtype == np.uint32


def test_ellipsoid_triangles_shape():
    _, idxs = _compute_ellipsoid(_identity_center(), _identity_cov())
    assert idxs.ndim == 2 and idxs.shape[1] == 3


def test_ellipsoid_indices_in_range():
    verts, idxs = _compute_ellipsoid(_identity_center(), _identity_cov())
    assert idxs.max() < len(verts)


def test_ellipsoid_center_offset():
    center = np.array([10.0, 20.0, 30.0])
    verts, _ = _compute_ellipsoid(center, _identity_cov(), n_std=1.0)
    centroid = verts.mean(axis=0)
    assert np.allclose(centroid, center, atol=0.1), f"centroid {centroid} != center {center}"


def test_ellipsoid_n_std_scales_radius():
    center = _identity_center()
    cov = _identity_cov()
    v1, _ = _compute_ellipsoid(center, cov, n_std=1.0)
    v2, _ = _compute_ellipsoid(center, cov, n_std=2.0)
    # Max distance from origin should double with n_std
    r1 = np.linalg.norm(v1, axis=1).max()
    r2 = np.linalg.norm(v2, axis=1).max()
    assert abs(r2 / r1 - 2.0) < 0.05, f"expected radius ratio ~2, got {r2/r1}"


def test_ellipsoid_singular_cov_no_crash():
    # Degenerate: all points in a plane — one eigenvalue is 0.
    cov = np.array([[1,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    verts, idxs = _compute_ellipsoid(_identity_center(), cov)
    assert len(verts) > 0


def test_ellipsoid_resolution_param():
    v_lo, i_lo = _compute_ellipsoid(_identity_center(), _identity_cov(), u_res=8, v_res=4)
    v_hi, i_hi = _compute_ellipsoid(_identity_center(), _identity_cov(), u_res=32, v_res=16)
    assert len(v_hi) > len(v_lo)
    assert len(i_hi) > len(i_lo)


# ── Scatter3D pre-renderer mesh API ──────────────────────────────────────────

_tkinter = pytest.importorskip("_tkinter")
pytest.importorskip("dragonsci._dragonsci")

import tkinter as tk
from dragonsci import Scatter3D


@pytest.fixture(scope="module")
def root():
    r = tk.Tk()
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def w(root):
    widget = Scatter3D(root, width=200, height=150)
    yield widget
    widget.destroy()


def test_add_convex_hull_returns_int(w):
    pts = _cube_points()
    h = w.add_convex_hull(pts)
    assert isinstance(h, int)


def test_add_convex_hull_increments_handle(w):
    pts = _cube_points()
    h1 = w.add_convex_hull(pts)
    h2 = w.add_convex_hull(pts)
    assert h2 > h1


def test_add_ellipsoid_returns_int(w):
    h = w.add_ellipsoid(_identity_center(), _identity_cov())
    assert isinstance(h, int)


def test_remove_mesh_removes_handle(w):
    pts = _cube_points()
    h = w.add_convex_hull(pts)
    assert h in w._mesh_handles
    w.remove_mesh(h)
    assert h not in w._mesh_handles


def test_clear_meshes_empties_all(w):
    pts = _cube_points()
    w.add_convex_hull(pts)
    w.add_ellipsoid(_identity_center(), _identity_cov())
    w.clear_meshes()
    assert len(w._mesh_handles) == 0
    assert len(w._mesh_meta) == 0


def test_meta_stored_for_hull(w):
    pts = _cube_points()
    h = w.add_convex_hull(pts, color=(0.5, 0.5, 0.0), opacity=0.4)
    meta = w._mesh_meta[h]
    assert abs(meta["color"][3] - 0.4) < 1e-6


def test_meta_stored_for_ellipsoid(w):
    h = w.add_ellipsoid(_identity_center(), _identity_cov(), n_std=3.0)
    assert w._mesh_meta[h]["n_std"] == 3.0


def test_pending_meshes_queued_pre_renderer(w):
    pts = _cube_points()
    before = len(w._pending_meshes)
    w.add_convex_hull(pts)
    assert len(w._pending_meshes) > before


def test_add_cluster_hulls_returns_handles(w):
    pts = np.vstack([
        np.random.default_rng(0).standard_normal((20, 3)).astype(np.float32),
        np.random.default_rng(1).standard_normal((20, 3)).astype(np.float32) + 5,
    ])
    labels = [0]*20 + [1]*20
    handles = w.add_cluster_hulls(pts, labels)
    assert len(handles) == 2
    assert all(isinstance(h, int) for h in handles)


def test_add_cluster_hulls_skips_small_groups(w):
    pts = np.vstack([
        np.random.default_rng(2).standard_normal((20, 3)).astype(np.float32),
        np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32),  # only 3 — skipped
    ])
    labels = [0]*20 + [1]*3
    handles = w.add_cluster_hulls(pts, labels)
    assert len(handles) == 1


def test_add_cluster_ellipsoids_returns_handles(w):
    pts = np.vstack([
        np.random.default_rng(3).standard_normal((20, 3)).astype(np.float32),
        np.random.default_rng(4).standard_normal((20, 3)).astype(np.float32) + 5,
    ])
    labels = [0]*20 + [1]*20
    handles = w.add_cluster_ellipsoids(pts, labels)
    assert len(handles) == 2


# ── Color-parity contract ─────────────────────────────────────────────────────

def _point_palette_index_for_label(label_value, all_labels):
    """Return the palette slot that _try_encode_categorical assigns to label_value.

    Mirrors the np.unique path: sorted unique values, slot = position in that list.
    """
    unique = np.unique(np.asarray(all_labels)).tolist()
    return unique.index(label_value)


def test_cluster_hulls_color_matches_point_encoder_sorted_labels(w):
    """Overlay color for label 0 must match palette[0], label 1 → palette[1]."""
    from dragonsci.widget import _CATEGORICAL_PALETTE
    pts = np.vstack([
        np.random.default_rng(10).standard_normal((20, 3)).astype(np.float32),
        np.random.default_rng(11).standard_normal((20, 3)).astype(np.float32) + 5,
    ])
    labels = [1]*20 + [0]*20   # intentionally unsorted to catch regression
    handles = w.add_cluster_hulls(pts, labels)
    assert len(handles) == 2

    all_lbl = np.asarray(labels)
    for h, lbl in zip(handles, np.unique(all_lbl).tolist()):
        expected_slot = _point_palette_index_for_label(lbl, labels)
        expected_color = list(_CATEGORICAL_PALETTE[expected_slot % len(_CATEGORICAL_PALETTE)])
        actual_rgb = w._mesh_meta[h]["color"][:3]
        assert actual_rgb == pytest.approx(expected_color, abs=1e-5), (
            f"label {lbl}: overlay color {actual_rgb} != point color {expected_color}"
        )


def test_cluster_ellipsoids_color_matches_point_encoder(w):
    """Same parity check for ellipsoids with labels [1,1,...,0,0,...]."""
    from dragonsci.widget import _CATEGORICAL_PALETTE
    pts = np.vstack([
        np.random.default_rng(12).standard_normal((20, 3)).astype(np.float32),
        np.random.default_rng(13).standard_normal((20, 3)).astype(np.float32) + 5,
    ])
    labels = [1]*20 + [0]*20
    handles = w.add_cluster_ellipsoids(pts, labels)
    assert len(handles) == 2

    for h, lbl in zip(handles, np.unique(np.asarray(labels)).tolist()):
        expected_slot = _point_palette_index_for_label(lbl, labels)
        expected_color = list(_CATEGORICAL_PALETTE[expected_slot % len(_CATEGORICAL_PALETTE)])
        actual_rgb = w._mesh_meta[h]["color"][:3]
        assert actual_rgb == pytest.approx(expected_color, abs=1e-5), (
            f"label {lbl}: overlay color {actual_rgb} != point color {expected_color}"
        )


def test_cluster_hulls_mixed_type_labels(w):
    """Mixed int/str labels must not raise TypeError."""
    pts = np.vstack([
        np.random.default_rng(5).standard_normal((20, 3)).astype(np.float32),
        np.random.default_rng(6).standard_normal((20, 3)).astype(np.float32) + 5,
    ])
    labels = [0]*20 + ["a"]*20
    handles = w.add_cluster_hulls(pts, labels)
    assert len(handles) == 2


def test_cluster_ellipsoids_mixed_type_labels(w):
    """Mixed int/str labels must not raise TypeError."""
    pts = np.vstack([
        np.random.default_rng(7).standard_normal((20, 3)).astype(np.float32),
        np.random.default_rng(8).standard_normal((20, 3)).astype(np.float32) + 5,
    ])
    labels = [0]*20 + ["b"]*20
    handles = w.add_cluster_ellipsoids(pts, labels)
    assert len(handles) == 2


def test_color_rgba_length_is_4(w):
    pts = _cube_points()
    h = w.add_convex_hull(pts, color=(1.0, 0.0, 0.0), opacity=0.5)
    assert len(w._mesh_meta[h]["color"]) == 4


def test_hex_color_accepted(w):
    pts = _cube_points()
    h = w.add_convex_hull(pts, color="#ff0000", opacity=0.5)
    meta = w._mesh_meta[h]
    assert abs(meta["color"][0] - 1.0) < 1e-3  # red channel ≈ 1.0


# ── Style-only update (High regression) ──────────────────────────────────────

def test_update_hull_style_only_no_TypeError(w):
    """update_convex_hull(handle, color=..., opacity=...) must not raise."""
    pts = _cube_points()
    h = w.add_convex_hull(pts, color=(1.0, 1.0, 0.0), opacity=0.3)
    w.update_convex_hull(h, color=(0.0, 1.0, 0.0), opacity=0.6)
    assert abs(w._mesh_meta[h]["color"][3] - 0.6) < 1e-6


def test_update_hull_color_only(w):
    pts = _cube_points()
    h = w.add_convex_hull(pts, color=(1.0, 0.0, 0.0), opacity=0.4)
    w.update_convex_hull(h, color=(0.0, 0.0, 1.0))
    assert abs(w._mesh_meta[h]["color"][2] - 1.0) < 1e-3  # blue
    assert abs(w._mesh_meta[h]["color"][3] - 0.4) < 1e-6  # opacity preserved


def test_update_hull_opacity_only(w):
    pts = _cube_points()
    h = w.add_convex_hull(pts, color=(1.0, 0.0, 0.0), opacity=0.4)
    w.update_convex_hull(h, opacity=0.9)
    assert abs(w._mesh_meta[h]["color"][3] - 0.9) < 1e-6


def test_update_hull_wireframe_only(w):
    pts = _cube_points()
    h = w.add_convex_hull(pts, wireframe=False)
    w.update_convex_hull(h, wireframe=True)
    assert w._mesh_meta[h]["wireframe"] is True


def test_update_ellipsoid_style_only_no_TypeError(w):
    """update_ellipsoid(handle, opacity=...) must not raise."""
    h = w.add_ellipsoid(_identity_center(), _identity_cov(), opacity=0.3)
    w.update_ellipsoid(h, opacity=0.7)
    assert abs(w._mesh_meta[h]["color"][3] - 0.7) < 1e-6


def test_update_ellipsoid_color_only(w):
    h = w.add_ellipsoid(_identity_center(), _identity_cov(), color=(1.0, 0.0, 0.0))
    w.update_ellipsoid(h, color=(0.0, 1.0, 0.0))
    assert abs(w._mesh_meta[h]["color"][1] - 1.0) < 1e-3  # green


def test_update_ellipsoid_nstd_only_resizes_geometry(w):
    """n_std change must recompute vertices, not just update metadata."""
    h = w.add_ellipsoid(_identity_center(), _identity_cov(), n_std=1.0)
    r_before = np.linalg.norm(w._mesh_meta[h]["_verts"], axis=1).max()
    w.update_ellipsoid(h, n_std=3.0)
    r_after = np.linalg.norm(w._mesh_meta[h]["_verts"], axis=1).max()
    assert w._mesh_meta[h]["n_std"] == 3.0
    assert r_after / r_before > 2.5, f"expected ~3x radius, got ratio {r_after/r_before:.2f}"


def test_update_ellipsoid_partial_geometry_raises(w):
    """Supplying center but not covariance (or vice versa) must raise ValueError."""
    h = w.add_ellipsoid(_identity_center(), _identity_cov())
    with pytest.raises(ValueError, match="both"):
        w.update_ellipsoid(h, center=_identity_center())


def test_update_hull_with_points_updates_geometry(w):
    pts1 = _cube_points()
    h = w.add_convex_hull(pts1)
    verts_before = w._mesh_meta[h]["_verts"].copy()
    pts2 = _cube_points() * 2  # scaled — different geometry
    w.update_convex_hull(h, pts2)
    verts_after = w._mesh_meta[h]["_verts"]
    assert not np.allclose(verts_before, verts_after)


# ── Input validation (Medium) ─────────────────────────────────────────────────

def test_hull_too_few_points_raises_value_error():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="at least 4"):
        _compute_convex_hull(pts)


def test_hull_wrong_shape_raises_value_error():
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    with pytest.raises(ValueError):
        _compute_convex_hull(pts)


def test_hull_degenerate_coplanar_raises_value_error():
    # All points in the same plane — QhullError should be caught and re-raised
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], dtype=np.float32)
    with pytest.raises(ValueError):
        _compute_convex_hull(pts)
