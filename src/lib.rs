mod camera;
mod colormap;
mod grid;
mod renderer;

use std::num::NonZeroIsize;

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use glam::Vec3;
use renderer::{PointInstance, Renderer, StreamMode};

// ── Platform handle construction ─────────────────────────────────────────────

#[allow(unused_variables)] // display_id used only on Linux
fn make_handles(
    window_id: isize,
    display_id: isize,
    platform: &str,
) -> Result<(RawWindowHandle, RawDisplayHandle), String> {
    match platform {
        "windows" => {
            #[cfg(target_os = "windows")]
            {
                use raw_window_handle::{Win32WindowHandle, WindowsDisplayHandle};
                use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
                let hwnd = NonZeroIsize::new(window_id).ok_or("HWND is zero")?;
                let hinstance = unsafe {
                    NonZeroIsize::new(GetModuleHandleW(std::ptr::null()) as isize)
                };
                let mut handle = Win32WindowHandle::new(hwnd);
                handle.hinstance = hinstance;
                Ok((
                    RawWindowHandle::Win32(handle),
                    RawDisplayHandle::Windows(WindowsDisplayHandle::new()),
                ))
            }
            #[cfg(not(target_os = "windows"))]
            Err("Windows handles only available on Windows".into())
        }
        "linux" => {
            #[cfg(target_os = "linux")]
            {
                use raw_window_handle::{XlibDisplayHandle, XlibWindowHandle};
                use std::ptr::NonNull;
                let win = XlibWindowHandle::new(window_id as u64);
                let dpy = XlibDisplayHandle::new(
                    NonNull::new(display_id as *mut std::ffi::c_void),
                    0,
                );
                Ok((RawWindowHandle::Xlib(win), RawDisplayHandle::Xlib(dpy)))
            }
            #[cfg(not(target_os = "linux"))]
            Err("Xlib handles only available on Linux".into())
        }
        "darwin" => {
            #[cfg(target_os = "macos")]
            {
                use raw_window_handle::{AppKitDisplayHandle, AppKitWindowHandle};
                use std::ptr::NonNull;
                let ns_view =
                    NonNull::new(window_id as *mut std::ffi::c_void).ok_or("NSView is null")?;
                Ok((
                    RawWindowHandle::AppKit(AppKitWindowHandle::new(ns_view)),
                    RawDisplayHandle::AppKit(AppKitDisplayHandle::new()),
                ))
            }
            #[cfg(not(target_os = "macos"))]
            Err("AppKit handles only available on macOS".into())
        }
        other => Err(format!("Unknown platform: {other}")),
    }
}

// ── Instance building ─────────────────────────────────────────────────────────

/// Build a `Vec<PointInstance>` from numpy arrays.
///
/// Returns `None` when `positions` is empty (caller should clear GPU state).
/// Returns `Some((instances, positions_cpu, n, bmin, bmax))` otherwise.
fn build_instances(
    positions: PyReadonlyArray2<f32>,
    colors: Option<PyReadonlyArray2<f32>>,
    scalars: Option<PyReadonlyArray1<f32>>,
    sizes: Option<PyReadonlyArray1<f32>>,
    cmap: &str,
    size: f32,
    clim: Option<[f32; 2]>,
    nan_color: Option<[f32; 3]>,
    log_scale: bool,
    opacity: f32,
) -> PyResult<Option<(Vec<PointInstance>, Vec<[f32; 3]>, usize, glam::Vec3, glam::Vec3)>> {
    let pos_array = positions.as_array();
    let n = pos_array.nrows();
    if pos_array.ncols() != 3 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("positions must be shape (N, 3)"));
    }
    if n == 0 { return Ok(None); }

    // Build the CPU positions vec first — single allocation.
    // For contiguous C-order input we copy the slice; for non-contiguous we
    // materialise it row-by-row.  Either way there is exactly one allocation and
    // no second copy at the end of the function.
    let positions_cpu: Vec<[f32; 3]> = if let Ok(s) = positions.as_slice() {
        bytemuck::cast_slice(s).to_vec()
    } else {
        (0..n).map(|i| [pos_array[[i,0]], pos_array[[i,1]], pos_array[[i,2]]]).collect()
    };
    let flat_pos: &[[f32; 3]] = &positions_cpu;

    if let Some(ref rgb) = colors {
        let ca = rgb.as_array();
        if ca.nrows() != n || ca.ncols() != 3 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("colors must be shape (N, 3)"));
        }
    }
    if let Some(ref scl) = scalars {
        if scl.as_array().len() != n {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("scalars length must match N"));
        }
    }

    // Resolve per-point sizes once.  NaN entries fall back to the uniform scalar.
    let sizes_vec: Option<Vec<f32>> = if let Some(ref sa) = sizes {
        let sv = sa.as_array();
        if sv.len() != n {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "point_sizes length must match N"
            ));
        }
        Some(match sa.as_slice() {
            Ok(s) => s.iter().map(|&v| if v.is_nan() { size } else { v }).collect(),
            Err(_) => sv.iter().map(|&v| if v.is_nan() { size } else { v }).collect(),
        })
    } else {
        None
    };
    let pt_size = |i: usize| -> f32 { sizes_vec.as_ref().map_or(size, |v| v[i]) };

    enum ColorData<'a> {
        Rgb { slice: Option<&'a [[f32; 3]]>, owned: Option<Vec<[f32; 3]>> },
        Scalar { slice: Option<&'a [f32]>, owned: Option<Vec<f32>>, vmin: f32, range: f32, cpts: &'static [[f32; 3]] },
        ZDefault { cpts: &'static [[f32; 3]] },
    }

    let color_data: ColorData<'_> = if let Some(ref rgb) = colors {
        let ca = rgb.as_array();
        match rgb.as_slice() {
            Ok(s) => ColorData::Rgb { slice: Some(bytemuck::cast_slice(s)), owned: None },
            Err(_) => ColorData::Rgb { slice: None, owned: Some((0..n).map(|i| [ca[[i,0]], ca[[i,1]], ca[[i,2]]]).collect()) },
        }
    } else if let Some(ref scl) = scalars {
        let sv = scl.as_array();
        let (sl, ov) = match scl.as_slice() {
            Ok(s) => (Some(s), None),
            Err(_) => (None, Some(sv.iter().copied().collect::<Vec<f32>>())),
        };
        let (vmin, vmax) = if let Some([a, b]) = clim {
            (a, b)
        } else if let Some(s) = sl {
            s.iter().copied().filter(|v| v.is_finite()).fold((f32::INFINITY, f32::NEG_INFINITY), |(mn,mx), v| (mn.min(v), mx.max(v)))
        } else {
            ov.as_deref().unwrap().iter().copied().filter(|v| v.is_finite()).fold((f32::INFINITY, f32::NEG_INFINITY), |(mn,mx), v| (mn.min(v), mx.max(v)))
        };
        let range = if (vmax - vmin).abs() < 1e-10 { 1.0 } else { vmax - vmin };
        ColorData::Scalar { slice: sl, owned: ov, vmin, range, cpts: colormap::resolve(cmap) }
    } else {
        ColorData::ZDefault { cpts: colormap::resolve(cmap) }
    };

    let nan_col = nan_color.unwrap_or([0.4, 0.4, 0.4]);
    let normalize = |v: f32, vmin: f32, range: f32| -> Option<f32> {
        if !v.is_finite() { return None; }
        let t = if log_scale {
            let lmin = vmin.max(1e-30).ln();
            let lmax = (vmin + range).max(1e-30).ln();
            let lv   = v.max(1e-30).ln();
            if (lmax - lmin).abs() < 1e-10 { 0.5 } else { (lv - lmin) / (lmax - lmin) }
        } else { (v - vmin) / range };
        Some(t.clamp(0.0, 1.0))
    };

    let mut bmin = glam::Vec3::splat(f32::INFINITY);
    let mut bmax = glam::Vec3::splat(f32::NEG_INFINITY);

    let instances: Vec<PointInstance> = match &color_data {
        ColorData::ZDefault { cpts } => {
            for p in flat_pos.iter() {
                let pos = glam::Vec3::from(*p);
                bmin = bmin.min(pos);
                bmax = bmax.max(pos);
            }
            let (vmin, range) = if let Some([a, b]) = clim {
                (a, if (b - a).abs() < 1e-10 { 1.0 } else { b - a })
            } else {
                let dz = bmax.z - bmin.z;
                (bmin.z, if dz.abs() < 1e-10 { 1.0 } else { dz })
            };
            flat_pos.iter().enumerate().map(|(i, p)| {
                let color = normalize(p[2], vmin, range).map(|t| colormap::sample(cpts, t)).unwrap_or(nan_col);
                PointInstance { position: *p, size: pt_size(i), color, alpha: opacity }
            }).collect()
        }
        _ => {
            flat_pos.iter().enumerate().map(|(i, p)| {
                let pos = glam::Vec3::from(*p);
                bmin = bmin.min(pos);
                bmax = bmax.max(pos);
                let color = match &color_data {
                    ColorData::Rgb { slice: Some(s), .. } => s[i],
                    ColorData::Rgb { owned: Some(v), .. } => v[i],
                    ColorData::Rgb { .. } => unreachable!(),
                    ColorData::Scalar { slice: Some(s), vmin, range, cpts, .. } =>
                        normalize(s[i], *vmin, *range).map(|t| colormap::sample(cpts, t)).unwrap_or(nan_col),
                    ColorData::Scalar { owned: Some(v), vmin, range, cpts, .. } =>
                        normalize(v[i], *vmin, *range).map(|t| colormap::sample(cpts, t)).unwrap_or(nan_col),
                    ColorData::Scalar { .. } | ColorData::ZDefault { .. } => unreachable!(),
                };
                PointInstance { position: *p, size: pt_size(i), color, alpha: opacity }
            }).collect()
        }
    };

    Ok(Some((instances, positions_cpu, n, bmin, bmax)))
}

// ── Line vertex building ──────────────────────────────────────────────────────

/// Convert a flat `(N, 2)` or `(N, 3)` numpy array of endpoint pairs into a
/// `Vec<LineVertex>` suitable for `add_line_actor`.
///
/// `segments` must have shape `(N, 2, 3)` — N line segments, each defined by
/// two XYZ endpoints.  Returns an error for wrong shapes.
fn build_line_vertices(
    segments: PyReadonlyArray2<f32>,
    color: [f32; 3],
) -> PyResult<Vec<grid::LineVertex>> {
    let arr = segments.as_array();
    let rows = arr.nrows();
    if arr.ncols() != 6 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "segments must be shape (N, 6) — each row is [x0,y0,z0, x1,y1,z1]",
        ));
    }
    let mut verts = Vec::with_capacity(rows * 2);
    for i in 0..rows {
        verts.push(grid::LineVertex { position: [arr[[i,0]], arr[[i,1]], arr[[i,2]]], color });
        verts.push(grid::LineVertex { position: [arr[[i,3]], arr[[i,4]], arr[[i,5]]], color });
    }
    Ok(verts)
}

// ── Python-facing class ───────────────────────────────────────────────────────

#[pyclass(name = "ScatterRenderer")]
struct PyScatterRenderer {
    inner: Renderer,
    camera_fitted: bool,
    // Scalar bar state — kept here so resize can rebuild it.
    scalar_bar_visible: bool,
    scalar_bar_vmin: f32,
    scalar_bar_vmax: f32,
    scalar_bar_log: bool,
    scalar_bar_cmap: String,
    scalar_bar_title: String,
}

impl PyScatterRenderer {
    /// Recompute grid and optionally fit camera from the current union bounds.
    /// `fit_camera` should be `true` only on first load (when camera_fitted is false).
    fn refresh_scene_bounds(&mut self, fit_if_unfitted: bool) {
        if let Some((union_min, union_max)) = self.inner.actor_union_bounds() {
            let (nice_min, nice_max) = grid::nice_bounds(union_min, union_max);
            self.inner.set_grid(union_min, union_max, nice_min, nice_max);
            if fit_if_unfitted && !self.camera_fitted {
                let center = (nice_min + nice_max) * 0.5;
                let radius = (nice_max - nice_min).length() * 0.5;
                self.inner.fit_camera(center, radius);
                self.camera_fitted = true;
            }
        } else {
            // Scene is now empty — clear grid and reset fit state so the next
            // add_points / add_lines triggers a fresh automatic camera fit.
            self.inner.clear_grid();
            self.camera_fitted = false;
        }
    }
}

#[pymethods]
impl PyScatterRenderer {
    #[new]
    #[pyo3(signature = (window_id, display_id, width, height, platform, vsync=true))]
    fn new(
        window_id: isize,
        display_id: isize,
        width: u32,
        height: u32,
        platform: &str,
        vsync: bool,
    ) -> PyResult<Self> {
        let (raw_window, raw_display) =
            make_handles(window_id, display_id, platform).map_err(PyRuntimeError::new_err)?;

        let present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };

        let inner = Renderer::new(raw_window, raw_display, width, height, present_mode)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            inner,
            camera_fitted: false,
            scalar_bar_visible: false,
            scalar_bar_vmin: 0.0,
            scalar_bar_vmax: 1.0,
            scalar_bar_log: false,
            scalar_bar_cmap: "viridis".to_string(),
            scalar_bar_title: String::new(),
        })
    }

    /// Create a headless offscreen renderer for use in Jupyter / server-side rendering.
    /// No OS window handle is required.
    #[staticmethod]
    #[pyo3(signature = (width, height))]
    fn create_offscreen(width: u32, height: u32) -> PyResult<Self> {
        let inner = Renderer::new_offscreen(width, height)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner,
            camera_fitted: false,
            scalar_bar_visible: false,
            scalar_bar_vmin: 0.0,
            scalar_bar_vmax: 1.0,
            scalar_bar_log: false,
            scalar_bar_cmap: "viridis".to_string(),
            scalar_bar_title: String::new(),
        })
    }

    /// Render one frame and return raw RGBA bytes (width × height × 4).
    /// Only valid for renderers created with `create_offscreen()`.
    fn render_offscreen(&mut self, _py: Python<'_>) -> PyResult<Vec<u8>> {
        self.inner
            .render_offscreen()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Upload point cloud data, replacing any previous scene content.
    ///
    /// Priority: explicit ``colors`` > ``scalars`` mapped through ``colormap`` > Z-position mapped through ``colormap``.
    #[pyo3(signature = (positions, colors=None, scalars=None, colormap=None, point_size=None, point_sizes=None, clim=None, nan_color=None, log_scale=false, opacity=1.0))]
    fn set_points(
        &mut self,
        _py: Python<'_>,
        positions: PyReadonlyArray2<f32>,
        colors: Option<PyReadonlyArray2<f32>>,
        scalars: Option<PyReadonlyArray1<f32>>,
        colormap: Option<&str>,
        point_size: Option<f32>,
        point_sizes: Option<PyReadonlyArray1<f32>>,
        clim: Option<[f32; 2]>,
        nan_color: Option<[f32; 3]>,
        log_scale: bool,
        opacity: f32,
    ) -> PyResult<Option<u32>> {
        let cmap = colormap.unwrap_or("viridis");
        let size = point_size.unwrap_or(4.0);

        let result = build_instances(positions, colors, scalars, point_sizes, cmap, size, clim, nan_color, log_scale, opacity)?;
        let (instances, pos_cpu, n, bmin, bmax) = match result {
            None => {
                self.inner.clear_actors();
                self.inner.clear_grid();
                self.camera_fitted = false;
                return Ok(None);
            }
            Some(v) => v,
        };

        let (min, max) = grid::nice_bounds(bmin, bmax);
        let handle = self.inner.set_points(&instances, pos_cpu, n as u32, bmin, bmax);
        self.inner.set_grid(bmin, bmax, min, max);

        if !self.camera_fitted {
            let center = (min + max) * 0.5;
            let radius = (max - min).length() * 0.5;
            self.inner.fit_camera(center, radius);
            self.camera_fitted = true;
        }

        Ok(handle)
    }

    /// Add a new point cloud actor on top of the existing scene.
    ///
    /// Returns an integer handle that can be passed to ``update_actor``,
    /// ``remove_actor``, and ``set_actor_visibility``.
    #[pyo3(signature = (positions, colors=None, scalars=None, colormap=None, point_size=None, point_sizes=None, clim=None, nan_color=None, log_scale=false, opacity=1.0))]
    fn add_points(
        &mut self,
        _py: Python<'_>,
        positions: PyReadonlyArray2<f32>,
        colors: Option<PyReadonlyArray2<f32>>,
        scalars: Option<PyReadonlyArray1<f32>>,
        colormap: Option<&str>,
        point_size: Option<f32>,
        point_sizes: Option<PyReadonlyArray1<f32>>,
        clim: Option<[f32; 2]>,
        nan_color: Option<[f32; 3]>,
        log_scale: bool,
        opacity: f32,
    ) -> PyResult<u32> {
        let cmap = colormap.unwrap_or("viridis");
        let size = point_size.unwrap_or(4.0);

        let result = build_instances(positions, colors, scalars, point_sizes, cmap, size, clim, nan_color, log_scale, opacity)?;
        let (instances, pos_cpu, n, bmin, bmax) = match result {
            None => return Ok(u32::MAX),
            Some(v) => v,
        };

        let handle = self.inner.add_actor(&instances, pos_cpu, n as u32, bmin, bmax);
        self.refresh_scene_bounds(true);
        Ok(handle)
    }

    /// Replace the data in an existing actor in place.
    #[pyo3(signature = (handle, positions, colors=None, scalars=None, colormap=None, point_size=None, point_sizes=None, clim=None, nan_color=None, log_scale=false, opacity=1.0))]
    fn update_actor(
        &mut self,
        _py: Python<'_>,
        handle: u32,
        positions: PyReadonlyArray2<f32>,
        colors: Option<PyReadonlyArray2<f32>>,
        scalars: Option<PyReadonlyArray1<f32>>,
        colormap: Option<&str>,
        point_size: Option<f32>,
        point_sizes: Option<PyReadonlyArray1<f32>>,
        clim: Option<[f32; 2]>,
        nan_color: Option<[f32; 3]>,
        log_scale: bool,
        opacity: f32,
    ) -> PyResult<()> {
        let cmap = colormap.unwrap_or("viridis");
        let size = point_size.unwrap_or(4.0);

        let result = build_instances(positions, colors, scalars, point_sizes, cmap, size, clim, nan_color, log_scale, opacity)?;
        match result {
            None => { self.inner.remove_actor(handle); }
            Some((instances, pos_cpu, n, bmin, bmax)) => {
                self.inner.update_actor_data(handle, &instances, pos_cpu, n as u32, bmin, bmax);
            }
        }
        self.refresh_scene_bounds(false);
        Ok(())
    }

    /// Remove a point cloud actor by handle.
    fn remove_actor(&mut self, handle: u32) {
        self.inner.remove_actor(handle);
        self.refresh_scene_bounds(false);
    }

    /// Show or hide an actor without removing it.
    fn set_actor_visibility(&mut self, handle: u32, visible: bool) {
        self.inner.set_actor_visibility(handle, visible);
        self.refresh_scene_bounds(false);
    }

    /// Remove all actors and clear the scene.
    fn clear_actors(&mut self) {
        self.inner.clear_actors();
        self.inner.clear_grid();
        self.camera_fitted = false;
    }

    // ── Streaming ────────────────────────────────────────────────────────────

    /// Pre-allocate a fixed-capacity stream actor and return its handle.
    ///
    /// `mode` is `0` for append (stop when full) or `1` for ring (overwrite oldest).
    /// `positions` and coloring arrays are optional initial seed data.
    #[pyo3(signature = (max_points, mode, positions=None, colors=None, scalars=None,
                        colormap=None, point_size=None, point_sizes=None,
                        clim=None, nan_color=None, log_scale=false, opacity=1.0))]
    fn create_stream(
        &mut self,
        _py: Python<'_>,
        max_points: u32,
        mode: u8,
        positions: Option<PyReadonlyArray2<f32>>,
        colors: Option<PyReadonlyArray2<f32>>,
        scalars: Option<PyReadonlyArray1<f32>>,
        colormap: Option<&str>,
        point_size: Option<f32>,
        point_sizes: Option<PyReadonlyArray1<f32>>,
        clim: Option<[f32; 2]>,
        nan_color: Option<[f32; 3]>,
        log_scale: bool,
        opacity: f32,
    ) -> PyResult<u32> {
        let stream_mode = if mode == 0 { StreamMode::Append } else { StreamMode::Ring };
        let cmap = colormap.unwrap_or("viridis");
        let size = point_size.unwrap_or(4.0);

        let (instances, pos_cpu, count, bmin, bmax) = if let Some(pos) = positions {
            match build_instances(pos, colors, scalars, point_sizes, cmap, size, clim, nan_color, log_scale, opacity)? {
                None => (vec![], vec![], 0u32, Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                Some((inst, cpu, n, mn, mx)) => (inst, cpu, n as u32, mn, mx),
            }
        } else {
            (vec![], vec![], 0u32, Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY))
        };

        let handle = self.inner.create_stream(max_points, stream_mode, &instances, pos_cpu, count, bmin, bmax);
        self.refresh_scene_bounds(true);
        Ok(handle)
    }

    /// Append new points to a stream actor.
    ///
    /// Returns `false` when `handle` does not refer to a stream actor.
    #[pyo3(signature = (handle, positions, colors=None, scalars=None, colormap=None,
                        point_size=None, point_sizes=None,
                        clim=None, nan_color=None, log_scale=false, opacity=1.0))]
    fn stream_points(
        &mut self,
        _py: Python<'_>,
        handle: u32,
        positions: PyReadonlyArray2<f32>,
        colors: Option<PyReadonlyArray2<f32>>,
        scalars: Option<PyReadonlyArray1<f32>>,
        colormap: Option<&str>,
        point_size: Option<f32>,
        point_sizes: Option<PyReadonlyArray1<f32>>,
        clim: Option<[f32; 2]>,
        nan_color: Option<[f32; 3]>,
        log_scale: bool,
        opacity: f32,
    ) -> PyResult<bool> {
        let cmap = colormap.unwrap_or("viridis");
        let size = point_size.unwrap_or(4.0);

        let (instances, pos_cpu, bmin, bmax) =
            match build_instances(positions, colors, scalars, point_sizes, cmap, size, clim, nan_color, log_scale, opacity)? {
                None => return Ok(true),
                Some((inst, cpu, _, mn, mx)) => (inst, cpu, mn, mx),
            };

        let (ok, bounds_grew) = self.inner.append_to_stream(handle, &instances, &pos_cpu, bmin, bmax);
        if ok && bounds_grew { self.refresh_scene_bounds(false); }
        Ok(ok)
    }

    /// Reset a stream actor to empty; keeps its pre-allocated GPU capacity.
    ///
    /// Returns `false` when `handle` does not refer to a stream actor.
    fn clear_stream(&mut self, handle: u32) -> bool {
        let ok = self.inner.clear_stream(handle);
        if ok { self.refresh_scene_bounds(false); }
        ok
    }

    /// Enable or disable CPU-side position storage used for picking and hover.
    ///
    /// When disabled, pick/hover operations silently find nothing, saving
    /// one full-sized `Vec<[f32;3]>` per actor.  Call before adding points.
    fn set_pick_storage(&mut self, enabled: bool) {
        self.inner.set_pick_storage(enabled);
    }

    fn render(&mut self) -> PyResult<()> {
        self.inner.render().map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.inner.resize(width, height);
        self.rebuild_scalar_bar();
    }

    fn rebuild_scalar_bar(&mut self) {
        if !self.scalar_bar_visible { return; }
        let cpts = colormap::resolve(&self.scalar_bar_cmap);
        self.inner.set_scalar_bar(
            true,
            self.scalar_bar_vmin,
            self.scalar_bar_vmax,
            self.scalar_bar_log,
            cpts,
            &self.scalar_bar_title.clone(),
        );
    }

    fn mouse_drag(&mut self, dx: f32, dy: f32, button: u8) {
        self.inner.mouse_drag(dx, dy, button);
    }

    fn scroll(&mut self, delta: f32) {
        self.inner.scroll(delta);
    }

    fn reset_camera(&mut self) {
        self.inner.reset_camera();
        // Keep camera_fitted = true: the camera is already in the fitted state
        // after reset, so the next set_points() should not re-fit over the user's
        // explicit reset request.
    }

    // ── Camera presets ────────────────────────────────────────────────────────

    /// Bird's-eye view: camera at +Y looking down at the XZ plane.
    fn view_xy(&mut self) {
        self.inner.set_view_direction(0.0, std::f32::consts::FRAC_PI_2 - 0.001);
    }

    /// Front view: camera at +Z looking at the XY plane (X right, Y up).
    /// This is the correct view for 2D scatter data stored in the XY plane (z=0).
    fn view_xz(&mut self) {
        self.inner.set_view_direction(0.0, 0.0);
    }

    /// Side view: camera at -X looking at the YZ plane.
    fn view_yz(&mut self) {
        self.inner.set_view_direction(-std::f32::consts::FRAC_PI_2, 0.0);
    }

    /// Isometric-style view: equal footing on all three axes.
    fn view_isometric(&mut self) {
        self.inner.set_view_direction(
            std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_4,
        );
    }

    /// Toggle between perspective (False) and parallel/orthographic (True) projection.
    fn set_parallel_projection(&mut self, on: bool) {
        self.inner.set_parallel_projection(on);
    }

    fn get_parallel_projection(&self) -> bool {
        self.inner.camera.parallel
    }

    /// Whether the camera has been fitted to data at least once (read-only diagnostic).
    #[getter]
    fn camera_fitted(&self) -> bool {
        self.camera_fitted
    }

    /// Fit the camera to explicit world-space bounds ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
    /// When called with no argument, re-fits to the last uploaded dataset.
    #[pyo3(signature = (bounds=None))]
    fn fit(&mut self, bounds: Option<[f32; 6]>) {
        if let Some(b) = bounds {
            self.inner.fit_to_bounds(b);
        } else {
            self.inner.reset_camera();
        }
    }

    /// Return the current camera state as a dict.
    fn get_camera(&self) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        use pyo3::types::PyDict;
        Python::with_gil(|py| {
            let s = self.inner.get_camera_state();
            let d = PyDict::new_bound(py);
            d.set_item("target", s.target.to_vec())?;
            d.set_item("distance", s.distance)?;
            d.set_item("yaw", s.yaw)?;
            d.set_item("pitch", s.pitch)?;
            d.set_item("parallel", s.parallel)?;
            Ok(d.into())
        })
    }

    /// Return [xmin, ymin, xmax, ymax] world-space view bounds for the 2D parallel view.
    fn get_view_bounds_2d(&self) -> [f32; 4] {
        let cam = &self.inner.camera;
        let half_h = cam.distance * (cam.fov_y * 0.5).tan();
        let half_w = half_h * cam.aspect;
        [
            cam.target.x - half_w,
            cam.target.y - half_h,
            cam.target.x + half_w,
            cam.target.y + half_h,
        ]
    }

    /// Set independent half-extents for orthographic projection.
    /// After this call the viewport always shows exactly
    /// [target.x - half_w .. target.x + half_w] × [target.y - half_h .. target.y + half_h].
    /// Parallel projection is enabled automatically.
    fn set_parallel_scale(&mut self, half_w: f32, half_h: f32) {
        self.inner.set_parallel_scale(half_w, half_h);
    }

    // ── Chart2D dedicated rendering path ──────────────────────────────────────

    /// Enable 2D chart mode with the given frame.
    ///
    /// ``plot_left/right/top/bottom`` are viewport fractions in [0, 1].
    /// ``top`` and ``bottom`` are measured from the *top* of the window
    /// (so ``top < bottom``).
    #[pyo3(signature = (plot_left, plot_right, plot_top, plot_bottom,
                        x0, x1, y0, y1, x_label="", y_label="", y_tick_step=None))]
    #[allow(clippy::too_many_arguments)]
    fn set_chart2d(
        &mut self,
        plot_left: f32, plot_right: f32, plot_top: f32, plot_bottom: f32,
        x0: f32, x1: f32, y0: f32, y1: f32,
        x_label: &str, y_label: &str,
        y_tick_step: Option<f32>,
    ) {
        self.inner.set_chart2d(
            plot_left, plot_right, plot_top, plot_bottom,
            x0, x1, y0, y1,
            x_label.to_string(), y_label.to_string(),
            y_tick_step,
        );
    }

    /// Add a polyline in data (x, y) space; returns a handle.
    /// The chart2d affine transform converts to NDC at render time.
    #[pyo3(signature = (x, y, color, line_width=2.0))]
    fn chart2d_add_line(
        &mut self,
        _py: Python<'_>,
        x: PyReadonlyArray1<f32>,
        y: PyReadonlyArray1<f32>,
        color: (f32, f32, f32),
        line_width: f32,
    ) -> PyResult<u32> {
        let xs = x.as_slice()?;
        let ys = y.as_slice()?;
        Ok(self.inner.chart2d_add_line(
            xs,
            ys,
            [color.0, color.1, color.2],
            line_width,
        ))
    }

    /// Replace a chart2d polyline's geometry.
    #[pyo3(signature = (handle, x, y, color, line_width=2.0))]
    fn chart2d_update_line(
        &mut self,
        _py: Python<'_>,
        handle: u32,
        x: PyReadonlyArray1<f32>,
        y: PyReadonlyArray1<f32>,
        color: (f32, f32, f32),
        line_width: f32,
    ) -> bool {
        let xs = x.as_slice().unwrap_or(&[]);
        let ys = y.as_slice().unwrap_or(&[]);
        self.inner.chart2d_update_line(
            handle,
            xs,
            ys,
            [color.0, color.1, color.2],
            line_width,
        )
    }

    /// Remove a chart2d polyline by handle.
    fn chart2d_remove_line(&mut self, handle: u32) -> bool {
        self.inner.chart2d_remove_line(handle)
    }

    /// Remove all chart2d polylines.
    fn chart2d_clear_lines(&mut self) {
        self.inner.chart2d_clear_lines();
    }

    /// Fast path: update only the x data-range; skips y-tick / title glyph rebuild.
    fn chart2d_update_xlim(&mut self, x0: f32, x1: f32) {
        self.inner.chart2d_update_xlim(x0, x1);
    }

    /// Fast path: update only the y data-range; keeps the Y tick interval fixed.
    fn chart2d_update_ylim(&mut self, y0: f32, y1: f32) {
        self.inner.chart2d_update_ylim(y0, y1);
    }

    /// Restore a camera state previously returned by ``get_camera()``.
    fn set_camera(&mut self, state: &pyo3::Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
        use pyo3::types::PyAnyMethods;
        use crate::camera::CameraState;
        let target_list: Vec<f32> = state.get_item("target")?.ok_or_else(||
            PyRuntimeError::new_err("missing key: target"))?.extract()?;
        if target_list.len() != 3 {
            return Err(PyRuntimeError::new_err("target must have 3 elements"));
        }
        let cs = CameraState {
            target:   [target_list[0], target_list[1], target_list[2]],
            distance: state.get_item("distance")?.ok_or_else(|| PyRuntimeError::new_err("missing key: distance"))?.extract()?,
            yaw:      state.get_item("yaw")?.ok_or_else(|| PyRuntimeError::new_err("missing key: yaw"))?.extract()?,
            pitch:    state.get_item("pitch")?.ok_or_else(|| PyRuntimeError::new_err("missing key: pitch"))?.extract()?,
            parallel: state.get_item("parallel")?.ok_or_else(|| PyRuntimeError::new_err("missing key: parallel"))?.extract()?,
        };
        self.inner.set_camera_state(cs);
        Ok(())
    }

    // ── Scalar bar ────────────────────────────────────────────────────────────

    /// Show or update the scalar bar overlay.
    #[pyo3(signature = (visible=true, vmin=0.0, vmax=1.0, log_scale=false, colormap="viridis", title=""))]
    fn show_scalar_bar(
        &mut self,
        visible: bool,
        vmin: f32,
        vmax: f32,
        log_scale: bool,
        colormap: &str,
        title: &str,
    ) {
        self.scalar_bar_visible = visible;
        self.scalar_bar_vmin = vmin;
        self.scalar_bar_vmax = vmax;
        self.scalar_bar_log = log_scale;
        self.scalar_bar_cmap = colormap.to_string();
        self.scalar_bar_title = title.to_string();
        let cpts = crate::colormap::resolve(colormap);
        self.inner.set_scalar_bar(visible, vmin, vmax, log_scale, cpts, title);
    }

    // ── Legend ────────────────────────────────────────────────────────────────

    /// Show or update the categorical legend overlay.
    /// `items` is a list of (label, [r, g, b]) pairs.
    /// `position`: 0=top-right, 1=top-left, 2=bottom-right, 3=bottom-left.
    #[pyo3(signature = (visible=true, title="", items=vec![], position=0))]
    fn set_legend(
        &mut self,
        visible: bool,
        title: &str,
        items: Vec<(String, Vec<f32>)>,
        position: u8,
    ) {
        let item_refs: Vec<(&str, [f32; 3])> = items
            .iter()
            .map(|(label, color)| {
                let c = [
                    color.first().copied().unwrap_or(0.0),
                    color.get(1).copied().unwrap_or(0.0),
                    color.get(2).copied().unwrap_or(0.0),
                ];
                (label.as_str(), c)
            })
            .collect();
        self.inner.set_legend(visible, title, &item_refs, position);
    }

    // ── Export ───────────────────────────────────────────────────────────────

    /// Render the current scene and return ``(width, height, rgba_array)``.
    ///
    /// ``rgba_array`` is a 1-D ``numpy.ndarray[uint8]`` of length ``width * height * 4``.
    /// Reshape with ``.reshape(height, width, 4)`` to get the full image.
    /// Ownership is transferred to Python — no copy required.
    fn screenshot<'py>(&mut self, py: Python<'py>) -> PyResult<PyObject> {
        let (w, h, pixels) = self.inner.screenshot()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        // Zero-copy: move Vec<u8> ownership directly into a numpy array.
        let arr = pixels.into_pyarray_bound(py);
        Ok((w, h, arr).into_py(py))
    }

    // ── Picking ───────────────────────────────────────────────────────────────

    /// Return the nearest point to a screen coordinate.
    ///
    /// Returns ``{"actor": int, "index": int, "point": [x, y, z]}`` or ``None``.
    fn pick_point(&mut self, screen_x: f32, screen_y: f32) -> Option<pyo3::Py<pyo3::types::PyDict>> {
        self.inner.pick_point(screen_x, screen_y).map(|(actor_id, idx, pos)| {
            Python::with_gil(|py| {
                let d = pyo3::types::PyDict::new_bound(py);
                d.set_item("actor", actor_id).ok();
                d.set_item("index", idx).ok();
                d.set_item("point", pos.to_vec()).ok();
                d.into()
            })
        })
    }

    /// Return all points inside a screen-space rectangle.
    ///
    /// Returns ``(actor_ids, point_indices)`` — two parallel ``list[int]`` arrays.
    fn pick_rectangle(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) -> (Vec<u32>, Vec<u32>) {
        let hits = self.inner.pick_rectangle(x0, y0, x1, y1);
        let actors: Vec<u32> = hits.iter().map(|(a, _)| *a).collect();
        let indices: Vec<u32> = hits.iter().map(|(_, i)| *i).collect();
        (actors, indices)
    }

    /// Return all points inside a screen-space polygon.
    ///
    /// `verts` is a list of ``[x, y]`` screen-coordinate pairs (pixels).
    /// Returns ``(actor_ids, point_indices)`` — two parallel ``list[int]`` arrays.
    fn pick_polygon(&mut self, verts: Vec<[f32; 2]>) -> (Vec<u32>, Vec<u32>) {
        let hits = self.inner.pick_polygon(&verts);
        let actors: Vec<u32> = hits.iter().map(|(a, _)| *a).collect();
        let indices: Vec<u32> = hits.iter().map(|(_, i)| *i).collect();
        (actors, indices)
    }

    /// Show an in-progress selection rectangle (screen coords, pixels).
    fn set_selection_rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        self.inner.set_selection_rect(x0, y0, x1, y1);
    }

    /// Hide the selection rectangle overlay.
    fn clear_selection_rect(&mut self) {
        self.inner.clear_selection_rect();
    }

    /// Update the in-progress lasso polyline (screen coords, pixels).
    /// `verts` is a list of ``[x, y]`` pairs recorded during the drag.
    fn set_lasso_path(&mut self, verts: Vec<[f32; 2]>) {
        self.inner.set_lasso_path(&verts);
    }

    /// Clear the lasso overlay.
    fn clear_lasso_path(&mut self) {
        self.inner.clear_lasso_path();
    }

    // ── Incremental lasso API ─────────────────────────────────────────────────

    /// Start a new freehand lasso at screen point ``(sx, sy)`` (pixels).
    /// Clears any active lasso path.
    fn lasso_begin(&mut self, sx: f32, sy: f32) {
        self.inner.lasso_begin(sx, sy);
    }

    /// Extend the active lasso by one screen-space point.
    /// Updates the overlay in O(1) GPU writes per call.
    fn lasso_extend(&mut self, sx: f32, sy: f32) {
        self.inner.lasso_extend(sx, sy);
    }

    /// Finish the lasso: run polygon picking against the recorded path,
    /// clear the overlay, and return ``(actor_ids, point_indices)`` arrays.
    fn lasso_end(&mut self) -> (Vec<u32>, Vec<u32>) {
        let hits = self.inner.lasso_end();
        let actors: Vec<u32> = hits.iter().map(|(a, _)| *a).collect();
        let indices: Vec<u32> = hits.iter().map(|(_, i)| *i).collect();
        (actors, indices)
    }

    /// Cancel the active lasso without picking.
    fn lasso_cancel(&mut self) {
        self.inner.lasso_cancel();
    }

    // ── Line / overlay actors ─────────────────────────────────────────────────

    /// Add a set of line segments as a new overlay actor.
    ///
    /// `segments` must be a ``(N, 6)`` float32 array where each row is
    /// ``[x0, y0, z0, x1, y1, z1]``.  `color` is an RGB tuple ``(r, g, b)``
    /// with values in ``[0, 1]``.
    ///
    /// Returns a non-negative integer handle.
    #[pyo3(signature = (segments, color=(1.0, 1.0, 1.0)))]
    fn add_lines(
        &mut self,
        _py: Python<'_>,
        segments: PyReadonlyArray2<f32>,
        color: (f32, f32, f32),
    ) -> PyResult<u32> {
        let verts = build_line_vertices(segments, [color.0, color.1, color.2])?;
        let handle = self.inner.add_line_actor(&verts);
        self.refresh_scene_bounds(true);
        Ok(handle)
    }

    /// Replace the geometry of an existing line overlay actor.
    #[pyo3(signature = (handle, segments, color=(1.0, 1.0, 1.0)))]
    fn update_lines(
        &mut self,
        _py: Python<'_>,
        handle: u32,
        segments: PyReadonlyArray2<f32>,
        color: (f32, f32, f32),
    ) -> PyResult<()> {
        let verts = build_line_vertices(segments, [color.0, color.1, color.2])?;
        self.inner.update_line_actor_data(handle, &verts);
        self.refresh_scene_bounds(false);
        Ok(())
    }

    /// Remove a line overlay actor by handle.
    fn remove_overlay(&mut self, handle: u32) {
        self.inner.remove_line_actor(handle);
        self.refresh_scene_bounds(false);
    }

    /// Show or hide a line overlay actor.
    fn set_overlay_visibility(&mut self, handle: u32, visible: bool) {
        self.inner.set_line_actor_visibility(handle, visible);
        self.refresh_scene_bounds(false);
    }

    /// Remove all line overlay actors.
    fn clear_overlays(&mut self) {
        self.inner.clear_line_actors();
        self.refresh_scene_bounds(false);
    }

    /// Return the union bounds of all actors (points + overlays) as ``((xmin,ymin,zmin),(xmax,ymax,zmax))``.
    /// Returns ``None`` when the scene is empty.
    fn actor_union_bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        self.inner.actor_union_bounds()
            .map(|(bmin, bmax)| (bmin.to_array(), bmax.to_array()))
    }

    /// Show or hide the orientation axes widget in the bottom-left corner.
    fn show_orientation_axes(&mut self, visible: bool) {
        self.inner.set_orientation_axes_visible(visible);
    }

    // ── Rendering modes ───────────────────────────────────────────────────────

    /// Set the point rendering style.
    ///
    /// ``style`` must be one of:
    /// - ``0`` — circle (soft anti-aliased disc, default)
    /// - ``1`` — square (full quad, no clipping)
    /// - ``2`` — gaussian (smooth exponential falloff, no hard edge)
    fn set_point_style(&mut self, style: u32) {
        self.inner.set_point_style(style);
    }

    /// Set the LOD factor for interaction.  When ``factor > 1`` each actor
    /// draws only ``count // factor`` instances, trading density for speed.
    /// Reset to ``1`` to restore full quality.
    fn set_lod_factor(&mut self, factor: u32) {
        self.inner.set_lod_factor(factor);
    }

    /// Override the maximum number of ticks shown on each axis.
    /// Pass ``None`` (Python ``None``) for any axis to restore auto-scaling.
    #[pyo3(signature = (x=None, y=None, z=None))]
    fn set_ticks(
        &mut self,
        x: Option<usize>,
        y: Option<usize>,
        z: Option<usize>,
    ) {
        self.inner.set_tick_override(x, y, z);
    }

    // ── Visual appearance ─────────────────────────────────────────────────────

    /// Show or hide the grid lines and tick labels.
    fn show_grid(&mut self, visible: bool) {
        self.inner.set_grid_visible(visible);
    }

    /// Show or hide major/minor grid lines on the axis planes.
    /// `major` draws lines at each tick position; `minor` subdivides each
    /// major interval into 5 for denser reference lines.
    #[pyo3(signature = (major=true, minor=false))]
    fn show_grid_planes(&mut self, major: bool, minor: bool) {
        self.inner.set_grid_planes(major, minor);
    }

    /// Set the background clear colour (linear RGB, 0.0–1.0 each channel).
    fn set_background_color(&mut self, r: f64, g: f64, b: f64) {
        self.inner.set_background_color(r, g, b);
    }

    /// Set the axis title labels shown at the grid extents.
    /// Pass empty strings to hide individual titles.
    #[pyo3(signature = (x="X", y="Y", z="Z"))]
    fn set_axis_labels(&mut self, x: &str, y: &str, z: &str) {
        self.inner.set_axis_labels(x.to_string(), y.to_string(), z.to_string());
    }

    /// Show or hide individual axes (ticks, labels, and bounding-box edges).
    /// Hiding an axis removes its tick marks and the four bounding-box edges
    /// that run parallel to it, without affecting the other axes.
    #[pyo3(signature = (x=true, y=true, z=true))]
    fn set_axis_visible(&mut self, x: bool, y: bool, z: bool) {
        self.inner.set_axis_visible(x, y, z);
    }

    // ── User label API ─────────────────────────────────────────────────────────

    /// Add a world-space text label.  Returns an opaque handle (u64).
    /// `anchor`: 0=Center, 1=Left, 2=Right, 3=Top, 4=Bottom
    #[pyo3(signature = (x, y, z, text, color, size, anchor=0))]
    fn add_user_label(
        &mut self,
        x: f32, y: f32, z: f32,
        text: &str,
        color: [f32; 4],
        size: f32,
        anchor: u8,
    ) -> u64 {
        self.inner.add_user_label(x, y, z, text, color, size, anchor)
    }

    /// Update fields of an existing label.  Pass ``None`` to keep the current value.
    #[pyo3(signature = (id, pos=None, text=None, color=None, size=None, anchor=None))]
    fn update_user_label(
        &mut self,
        id: u64,
        pos: Option<[f32; 3]>,
        text: Option<&str>,
        color: Option<[f32; 4]>,
        size: Option<f32>,
        anchor: Option<u8>,
    ) {
        self.inner.update_user_label(id, pos, text, color, size, anchor);
    }

    fn remove_user_label(&mut self, id: u64) {
        self.inner.remove_user_label(id);
    }

    fn set_user_label_visible(&mut self, id: u64, visible: bool) {
        self.inner.set_user_label_visible(id, visible);
    }

    fn clear_user_labels(&mut self) {
        self.inner.clear_user_labels();
    }

    // ── Mesh overlay API ───────────────────────────────────────────────────────

    fn add_mesh(
        &mut self,
        vertices: PyReadonlyArray2<f32>,
        indices: PyReadonlyArray2<u32>,
        color: [f32; 4],
        wireframe: bool,
    ) -> PyResult<u64> {
        let v = vertices.as_array();
        let nv = v.nrows();
        let verts: Vec<[f32; 3]> = (0..nv).map(|i| [v[[i,0]], v[[i,1]], v[[i,2]]]).collect();
        let idx = indices.as_array();
        let nf = idx.nrows();
        let idxs: Vec<[u32; 3]> = (0..nf).map(|i| [idx[[i,0]], idx[[i,1]], idx[[i,2]]]).collect();
        let handle = self.inner.add_mesh_actor(&verts, &idxs, color, wireframe);
        self.refresh_scene_bounds(true);
        Ok(handle)
    }

    fn update_mesh(
        &mut self,
        handle: u64,
        vertices: PyReadonlyArray2<f32>,
        indices: PyReadonlyArray2<u32>,
        color: [f32; 4],
        wireframe: bool,
    ) {
        let v = vertices.as_array();
        let nv = v.nrows();
        let verts: Vec<[f32; 3]> = (0..nv).map(|i| [v[[i,0]], v[[i,1]], v[[i,2]]]).collect();
        let idx = indices.as_array();
        let nf = idx.nrows();
        let idxs: Vec<[u32; 3]> = (0..nf).map(|i| [idx[[i,0]], idx[[i,1]], idx[[i,2]]]).collect();
        self.inner.update_mesh_actor(handle, &verts, &idxs, color, wireframe);
        self.refresh_scene_bounds(false);
    }

    fn remove_mesh(&mut self, handle: u64) {
        self.inner.remove_mesh_actor(handle);
        self.refresh_scene_bounds(false);
    }

    fn set_mesh_visibility(&mut self, handle: u64, visible: bool) {
        self.inner.set_mesh_actor_visibility(handle, visible);
    }

    fn clear_meshes(&mut self) {
        self.inner.clear_mesh_actors();
        self.refresh_scene_bounds(false);
    }

    #[staticmethod]
    fn colormap_names() -> Vec<&'static str> {
        colormap::COLORMAP_NAMES.to_vec()
    }
}

// ── Module ────────────────────────────────────────────────────────────────────

#[pymodule]
fn _dragonsci(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScatterRenderer>()?;
    Ok(())
}
