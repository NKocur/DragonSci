use glam::Vec3;

/// Expand raw data bounds to the nearest "nice" round numbers so the grid
/// stays visually stable between frames that share a similar data range.
/// Targets ~5 ticks per axis; step rounds to 1 / 2 / 5 × 10^n.
pub fn nice_bounds(min: Vec3, max: Vec3) -> (Vec3, Vec3) {
    let nice_axis = |lo: f32, hi: f32| -> (f32, f32) {
        let range = (hi - lo).abs();
        if range < 1e-10 {
            return (lo - 0.5, hi + 0.5);
        }
        let rough_step = range / 5.0;
        let mag = 10_f32.powf(rough_step.log10().floor());
        let norm = rough_step / mag;
        let nice_step = if norm <= 1.0 { 1.0 }
            else if norm <= 2.0 { 2.0 }
            else if norm <= 5.0 { 5.0 }
            else { 10.0 } * mag;
        let nice_min = (lo / nice_step).floor() * nice_step;
        let nice_max = (hi / nice_step).ceil()  * nice_step;
        (nice_min, nice_max)
    };
    let (x0, x1) = nice_axis(min.x, max.x);
    let (y0, y1) = nice_axis(min.y, max.y);
    let (z0, z1) = nice_axis(min.z, max.z);
    (Vec3::new(x0, y0, z0), Vec3::new(x1, y1, z1))
}

/// Generate tick positions at multiples of the nice step for [lo, hi].
/// Targets ~5 ticks and caps at MAX_TICKS; steps up to the next nice
/// increment if the initial step produces too many.
/// Returns an empty vec for degenerate ranges.
/// `max_ticks` is the caller-supplied cap; never exceeds this count.
fn axis_ticks(lo: f32, hi: f32, max_ticks: usize) -> Vec<f32> {
    let range = hi - lo;
    if range < 1e-10 || max_ticks == 0 {
        return vec![];
    }

    let initial_step = {
        let rough = range / max_ticks as f32;
        let mag = 10_f32.powf(rough.log10().floor());
        let norm = rough / mag;
        (if norm <= 1.0 { 1.0 } else if norm <= 2.0 { 2.0 } else if norm <= 5.0 { 5.0 } else { 10.0 }) * mag
    };

    let mut step = initial_step;
    loop {
        let first = (lo / step).ceil() * step;
        let mut ticks = Vec::with_capacity(max_ticks + 1);
        let mut t = first;
        while t <= hi + step * 1e-4 {
            ticks.push(t);
            t += step;
        }
        if ticks.len() <= max_ticks {
            return ticks;
        }
        // Too many — move to next nice step up (1→2→5→10 pattern).
        let mag = 10_f32.powf(step.log10().floor());
        let norm = (step / mag).round() as i32;
        step = match norm { 1 => 2.0, 2 => 5.0, _ => 10.0 } * mag;
    }
}

/// Generate minor tick positions between major ticks.
/// Subdivides each major interval into `subdivisions` equal parts and returns
/// only the interior positions (major tick positions are excluded).
fn minor_ticks(lo: f32, hi: f32, major_step: f32, subdivisions: u32) -> Vec<f32> {
    if major_step <= 0.0 || subdivisions <= 1 {
        return vec![];
    }
    let minor_step = major_step / subdivisions as f32;
    let first = (lo / minor_step).ceil() * minor_step;
    let mut result = Vec::new();
    let mut t = first;
    while t <= hi + minor_step * 1e-4 {
        // Exclude positions that coincide with a major tick.
        let dist = ((t / major_step).round() * major_step - t).abs();
        if dist > major_step * 1e-4 {
            result.push(t);
        }
        t += minor_step;
    }
    result
}

/// A line segment vertex: position + RGB color.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

pub struct LabelAnchor {
    pub world_pos: Vec3,
    /// The point on the grid edge the label refers to (before the world-space offset).
    /// Used at render time to compute the screen-space push direction and enforce a
    /// minimum pixel gap, so labels stay readable in any projection mode.
    pub tick_pos: Vec3,
    pub text: String,
    /// When true, rendered with the larger axis-title font (14 px) instead of the
    /// tick-value font (11 px).
    pub is_axis_title: bool,
}

pub struct GridGeometry {
    pub vertices: Vec<LineVertex>,
    pub labels: Vec<LabelAnchor>,
}

/// Builds the bounding-box wireframe plus axis tick marks and their text anchors.
///
/// `data_min`/`data_max` are the raw point-cloud bounds (used to detect flat
/// axes and suppress their ticks). `nice_min`/`nice_max` are the rounded bounds
/// used for the box extent and tick value positions.
/// `tick_override` lets callers pin the max tick count per axis [x, y, z].
/// `None` means auto (proportional to axis length).
/// `axis_visible` hides an axis entirely (its edges and tick marks are omitted).
///
/// `camera_eye` is the current camera position in world space.  It controls
/// which face of the bounding box each axis draws its ticks on, so that labels
/// always project *outside* the box silhouette regardless of viewing angle.
///
/// `axis_texts` are the axis title strings (e.g. `["X", "Y", "Z"]`).  An empty
/// string suppresses the title for that axis.
pub fn build_grid(
    data_min: Vec3,
    data_max: Vec3,
    nice_min: Vec3,
    nice_max: Vec3,
    tick_override: [Option<usize>; 3],
    axis_visible: [bool; 3],
    camera_eye: Vec3,
    axis_texts: &[String; 3],
    show_major_planes: bool,
    show_minor_planes: bool,
) -> GridGeometry {
    let mut verts: Vec<LineVertex> = Vec::new();
    let mut labels: Vec<LabelAnchor> = Vec::new();

    let extent = nice_max - nice_min;
    let box_color = [0.45_f32, 0.45, 0.45];
    let x_col = [0.90_f32, 0.30, 0.30];
    let y_col = [0.30_f32, 0.90, 0.30];
    let z_col = [0.30_f32, 0.50, 0.90];

    // Center is needed both for face selection and for the back-corner filter.
    let center = (nice_min + nice_max) * 0.5;

    // ── Bounding box (9 edges in 3D, 4 edges in 2D) ───────────────────────────
    // Corners are indexed by (x_bit | y_bit<<1 | z_bit<<2), bit=0 → min, bit=1 → max.
    let c = [
        Vec3::new(nice_min.x, nice_min.y, nice_min.z), // 0: xmin ymin zmin
        Vec3::new(nice_max.x, nice_min.y, nice_min.z), // 1: xmax ymin zmin
        Vec3::new(nice_min.x, nice_max.y, nice_min.z), // 2: xmin ymax zmin
        Vec3::new(nice_max.x, nice_max.y, nice_min.z), // 3: xmax ymax zmin
        Vec3::new(nice_min.x, nice_min.y, nice_max.z), // 4: xmin ymin zmax
        Vec3::new(nice_max.x, nice_min.y, nice_max.z), // 5: xmax ymin zmax
        Vec3::new(nice_min.x, nice_max.y, nice_max.z), // 6: xmin ymax zmax
        Vec3::new(nice_max.x, nice_max.y, nice_max.z), // 7: xmax ymax zmax
    ];
    let edges: [(usize, usize); 12] = [
        (0, 1), (2, 3), (4, 5), (6, 7), // X-parallel
        (0, 2), (1, 3), (4, 6), (5, 7), // Y-parallel
        (0, 4), (1, 5), (2, 6), (3, 7), // Z-parallel
    ];
    // In 3D the back corner is the vertex diagonally opposite to where the three
    // near faces meet.  Its index bit is 1 for each axis where the camera is on
    // the positive side (because the near face is the min face there, making the
    // far face the max face).  The 3 edges that touch this corner don't lie on
    // any active plane, so they are omitted — leaving the 9 edges that form the
    // three visible planes (open-corner look instead of full wireframe).
    let far_idx: usize =
          (camera_eye.x >= center.x) as usize
        | ((camera_eye.y >= center.y) as usize) << 1
        | ((camera_eye.z >= center.z) as usize) << 2;

    for (i, (a, b)) in edges.iter().enumerate() {
        if !axis_visible[2] {
            // 2D mode: emit only the 4 edges of the z=zmin face (indices 0,1,4,5).
            // These are the X and Y edges at z=nice_min.z, which is also where
            // the tick marks are anchored. Skipping the duplicate z=zmax face
            // (indices 2,3,6,7) and all Z-parallel edges (8-11) removes the
            // second box that appears when looking straight at the XY plane.
            match i { 0 | 1 | 4 | 5 => {} _ => continue, }
        } else if *a == far_idx || *b == far_idx {
            // 3D mode: skip the 3 edges that touch the back corner.
            continue;
        }
        verts.push(LineVertex { position: c[*a].to_array(), color: box_color });
        verts.push(LineVertex { position: c[*b].to_array(), color: box_color });
    }

    // ── Detect flat axes ──────────────────────────────────────────────────────
    let data_range = data_max - data_min;
    let diagonal = data_range.length().max(1e-10);
    let flat_x = data_range.x.abs() / diagonal < 0.01;
    let flat_y = data_range.y.abs() / diagonal < 0.01;
    let flat_z = data_range.z.abs() / diagonal < 0.01;

    let tick_len = extent.length() * 0.025;
    let label_offset = tick_len * 2.0;
    let pad = extent.length() * 0.12;

    // Scale max ticks per axis by its fraction of the longest axis.
    let max_ne = extent.x.max(extent.y).max(extent.z).max(1e-10);
    let ticks_for = |e: f32| -> usize {
        let r = e / max_ne;
        if r < 0.15 { 2 } else if r < 0.40 { 3 } else { 5 }
    };
    let x_ticks = tick_override[0].unwrap_or_else(|| ticks_for(extent.x));
    let y_ticks = tick_override[1].unwrap_or_else(|| ticks_for(extent.y));
    let z_ticks = tick_override[2].unwrap_or_else(|| ticks_for(extent.z));

    // ── Dynamic face selection ────────────────────────────────────────────────
    // For each axis we pick the bounding-box edge (face) on which tick marks and
    // labels are anchored.  Rule: choose the face on the *near* side of the camera
    // (minimum world-space distance), so labels project outside the box silhouette
    // in screen space regardless of viewing angle.
    //
    // Concretely: if the camera is above center (eye.y ≥ center.y) we place X-axis
    // ticks on the floor (y = nice_min) and push them further downward (−Y sign),
    // so they appear below the box in screen space.  Mirroring happens in every
    // other axis/face pair by the same logic.
    //
    // Each axis uses ONE perpendicular direction for the offset (±Y for X-axis ticks;
    // ±X for Y- and Z-axis ticks) to keep labels tidy and non-overlapping.
    // The Z-axis always uses the *opposite* X-face from the Y-axis so they can
    // never share a corner and collide.
    // (center was computed earlier for the back-corner edge filter)

    // X-axis ticks (along X): anchor on the near Y-face, push further in ±Y.
    let (x_y_edge, x_y_sign): (f32, f32) = if camera_eye.y >= center.y {
        (nice_min.y, -1.0)   // camera above → anchor at floor (y=min), push −Y
    } else {
        (nice_max.y,  1.0)   // camera below → anchor at ceiling (y=max), push +Y
    };
    // Also snap to the near Z-face so X ticks sit on the visible box edge.
    let x_z_edge: f32 = if camera_eye.z >= center.z { nice_min.z } else { nice_max.z };

    // Y-axis ticks (along Y): anchor on the near X-face, push further in ±X.
    let (y_x_edge, y_x_sign): (f32, f32) = if camera_eye.x >= center.x {
        (nice_min.x, -1.0)   // camera at +X → anchor at left wall (x=min), push −X
    } else {
        (nice_max.x,  1.0)   // camera at −X → anchor at right wall (x=max), push +X
    };
    let y_z_edge: f32 = if camera_eye.z >= center.z { nice_min.z } else { nice_max.z };

    // Z-axis ticks (along Z): anchor on the near X-face, but *opposite* from Y-axis.
    // Swapping faces guarantees Y and Z labels land on different X-faces and
    // can never pile up at the same corner.
    let (z_x_edge, z_x_sign): (f32, f32) = if camera_eye.x >= center.x {
        (nice_max.x,  1.0)   // camera at +X → anchor at right wall (x=max), push +X
    } else {
        (nice_min.x, -1.0)   // camera at −X → anchor at left wall (x=min), push −X
    };
    // Z ticks share the same Y-face choice as X-axis for consistent appearance.
    let z_y_edge = x_y_edge;

    // ── Depth-axis detection ──────────────────────────────────────────────────
    // When the camera is near-axis-aligned (e.g. after flatten_view), the depth
    // axis's tick labels all project to the same screen point — suppress them.
    // For the remaining axes whose normal push direction is along the depth axis,
    // fall back to a perpendicular direction that stays visible on screen.
    let cam_dir = (center - camera_eye).normalize_or_zero();
    let depth_x = cam_dir.x.abs() > 0.97;
    let depth_y = cam_dir.y.abs() > 0.97;
    let depth_z = cam_dir.z.abs() > 0.97;
    // Sign for Z-direction fallback push (used when X or Y is the depth axis).
    let z_out: f32 = if camera_eye.z >= center.z { -1.0 } else { 1.0 };

    // ── Pre-compute tick value vectors (reused for grid planes and tick marks) ─
    let x_show = axis_visible[0] && !depth_x && (!flat_x || tick_override[0].is_some());
    let y_show = axis_visible[1] && !depth_y && (!flat_y || tick_override[1].is_some());
    let z_show = axis_visible[2] && !depth_z && (!flat_z || tick_override[2].is_some());

    let x_vals = if x_show { axis_ticks(nice_min.x, nice_max.x, x_ticks) } else { vec![] };
    let y_vals = if y_show { axis_ticks(nice_min.y, nice_max.y, y_ticks) } else { vec![] };
    let z_vals = if z_show { axis_ticks(nice_min.z, nice_max.z, z_ticks) } else { vec![] };

    // ── Grid planes ───────────────────────────────────────────────────────────
    // Major lines align with the tick positions; minor lines subdivide each
    // major interval into 5.  Lines are drawn on the same near face as the
    // tick marks so they appear as a background grid for the data.
    if show_major_planes || show_minor_planes {
        let major_col = [0.20_f32, 0.20, 0.25];
        let minor_col = [0.13_f32, 0.13, 0.17];

        let x_step = if x_vals.len() >= 2 { x_vals[1] - x_vals[0] } else { 0.0 };
        let y_step = if y_vals.len() >= 2 { y_vals[1] - y_vals[0] } else { 0.0 };
        let z_step = if z_vals.len() >= 2 { z_vals[1] - z_vals[0] } else { 0.0 };

        let x_minor = if show_minor_planes { minor_ticks(nice_min.x, nice_max.x, x_step, 5) } else { vec![] };
        let y_minor = if show_minor_planes { minor_ticks(nice_min.y, nice_max.y, y_step, 5) } else { vec![] };
        let z_minor = if show_minor_planes { minor_ticks(nice_min.z, nice_max.z, z_step, 5) } else { vec![] };

        // Helper: push a line segment.
        let mut seg = |a: Vec3, b: Vec3, col: [f32; 3]| {
            verts.push(LineVertex { position: a.to_array(), color: col });
            verts.push(LineVertex { position: b.to_array(), color: col });
        };

        if !axis_visible[2] {
            // ── 2D grid (XY plane at z = nice_min.z) ─────────────────────────
            // Vertical lines at each X tick, horizontal lines at each Y tick.
            let z = nice_min.z;
            if show_major_planes {
                for &x in &x_vals {
                    seg(Vec3::new(x, nice_min.y, z), Vec3::new(x, nice_max.y, z), major_col);
                }
                for &y in &y_vals {
                    seg(Vec3::new(nice_min.x, y, z), Vec3::new(nice_max.x, y, z), major_col);
                }
            }
            if show_minor_planes {
                for &x in &x_minor {
                    seg(Vec3::new(x, nice_min.y, z), Vec3::new(x, nice_max.y, z), minor_col);
                }
                for &y in &y_minor {
                    seg(Vec3::new(nice_min.x, y, z), Vec3::new(nice_max.x, y, z), minor_col);
                }
            }
        } else {
            // ── 3D floor plane (y = x_y_edge): X and Z grid lines ────────────
            if x_show || z_show {
                let y = x_y_edge;
                if show_major_planes {
                    for &x in &x_vals {
                        seg(Vec3::new(x, y, nice_min.z), Vec3::new(x, y, nice_max.z), major_col);
                    }
                    for &z in &z_vals {
                        seg(Vec3::new(nice_min.x, y, z), Vec3::new(nice_max.x, y, z), major_col);
                    }
                }
                if show_minor_planes {
                    for &x in &x_minor {
                        seg(Vec3::new(x, y, nice_min.z), Vec3::new(x, y, nice_max.z), minor_col);
                    }
                    for &z in &z_minor {
                        seg(Vec3::new(nice_min.x, y, z), Vec3::new(nice_max.x, y, z), minor_col);
                    }
                }
            }
        }

        // Side wall (x = y_x_edge): Y lines at each Z tick, Z lines at each Y tick.
        // Only meaningful in 3D mode.
        if axis_visible[2] && (y_show || z_show) {
            let x = y_x_edge;
            if show_major_planes {
                for &y in &y_vals {
                    seg(Vec3::new(x, y, nice_min.z), Vec3::new(x, y, nice_max.z), major_col);
                }
                for &z in &z_vals {
                    seg(Vec3::new(x, nice_min.y, z), Vec3::new(x, nice_max.y, z), major_col);
                }
            }
            if show_minor_planes {
                for &y in &y_minor {
                    seg(Vec3::new(x, y, nice_min.z), Vec3::new(x, y, nice_max.z), minor_col);
                }
                for &z in &z_minor {
                    seg(Vec3::new(x, nice_min.y, z), Vec3::new(x, nice_max.y, z), minor_col);
                }
            }
        }

        // Back wall (z = x_z_edge): X lines at each Y tick, Y lines at each X tick.
        // Only meaningful in 3D mode.
        if axis_visible[2] && (x_show || y_show) {
            let z = x_z_edge;
            if show_major_planes {
                for &x in &x_vals {
                    seg(Vec3::new(x, nice_min.y, z), Vec3::new(x, nice_max.y, z), major_col);
                }
                for &y in &y_vals {
                    seg(Vec3::new(nice_min.x, y, z), Vec3::new(nice_max.x, y, z), major_col);
                }
            }
            if show_minor_planes {
                for &x in &x_minor {
                    seg(Vec3::new(x, nice_min.y, z), Vec3::new(x, nice_max.y, z), minor_col);
                }
                for &y in &y_minor {
                    seg(Vec3::new(nice_min.x, y, z), Vec3::new(nice_max.x, y, z), minor_col);
                }
            }
        }
    }

    // ── X ticks ───────────────────────────────────────────────────────────────
    if x_show {
        // When depth_y (camera along Y, viewing XZ plane) the normal ±Y push
        // goes into the depth — switch to ±Z so labels remain visible.
        let (x_tick_off, x_label_off, x_pad_off) = if !depth_y {
            (Vec3::new(0.0, x_y_sign * tick_len,     0.0),
             Vec3::new(0.0, x_y_sign * label_offset, 0.0),
             Vec3::new(0.0, x_y_sign * pad,           0.0))
        } else {
            (Vec3::new(0.0, 0.0, z_out * tick_len),
             Vec3::new(0.0, 0.0, z_out * label_offset),
             Vec3::new(0.0, 0.0, z_out * pad))
        };
        for &val in &x_vals {
            let v   = Vec3::new(val, x_y_edge, x_z_edge);
            let end = v + x_tick_off;
            verts.push(LineVertex { position: v.to_array(),   color: x_col });
            verts.push(LineVertex { position: end.to_array(), color: x_col });
            labels.push(LabelAnchor {
                world_pos:    end + x_label_off,
                tick_pos:     end,
                text:         format_tick(val),
                is_axis_title: false,
            });
        }
        if !axis_texts[0].is_empty() {
            let mid = Vec3::new(center.x, x_y_edge, x_z_edge);
            labels.push(LabelAnchor {
                world_pos:    mid + x_pad_off,
                tick_pos:     mid,
                text:         axis_texts[0].clone(),
                is_axis_title: true,
            });
        }
    }

    // ── Y ticks ───────────────────────────────────────────────────────────────
    if y_show {
        // When depth_x (camera along X, viewing YZ plane) the normal ±X push
        // goes into the depth — switch to ±Z so labels remain visible.
        let (y_tick_off, y_label_off, y_pad_off) = if !depth_x {
            (Vec3::new(y_x_sign * tick_len,     0.0, 0.0),
             Vec3::new(y_x_sign * label_offset, 0.0, 0.0),
             Vec3::new(y_x_sign * pad,          0.0, 0.0))
        } else {
            (Vec3::new(0.0, 0.0, z_out * tick_len),
             Vec3::new(0.0, 0.0, z_out * label_offset),
             Vec3::new(0.0, 0.0, z_out * pad))
        };
        for &val in &y_vals {
            let v   = Vec3::new(y_x_edge, val, y_z_edge);
            let end = v + y_tick_off;
            verts.push(LineVertex { position: v.to_array(),   color: y_col });
            verts.push(LineVertex { position: end.to_array(), color: y_col });
            labels.push(LabelAnchor {
                world_pos:    end + y_label_off,
                tick_pos:     end,
                text:         format_tick(val),
                is_axis_title: false,
            });
        }
        if !axis_texts[1].is_empty() {
            let mid = Vec3::new(y_x_edge, center.y, y_z_edge);
            labels.push(LabelAnchor {
                world_pos:    mid + y_pad_off,
                tick_pos:     mid,
                text:         axis_texts[1].clone(),
                is_axis_title: true,
            });
        }
    }

    // ── Z ticks ───────────────────────────────────────────────────────────────
    if z_show {
        // When depth_x (camera along X, viewing YZ plane) the normal ±X push
        // goes into the depth — switch to ±Y so labels remain visible.
        let (z_tick_off, z_label_off, z_pad_off) = if !depth_x {
            (Vec3::new(z_x_sign * tick_len,     0.0, 0.0),
             Vec3::new(z_x_sign * label_offset, 0.0, 0.0),
             Vec3::new(z_x_sign * pad,          0.0, 0.0))
        } else {
            (Vec3::new(0.0, x_y_sign * tick_len,     0.0),
             Vec3::new(0.0, x_y_sign * label_offset, 0.0),
             Vec3::new(0.0, x_y_sign * pad,          0.0))
        };
        for &val in &z_vals {
            let v   = Vec3::new(z_x_edge, z_y_edge, val);
            let end = v + z_tick_off;
            verts.push(LineVertex { position: v.to_array(),   color: z_col });
            verts.push(LineVertex { position: end.to_array(), color: z_col });
            labels.push(LabelAnchor {
                world_pos:    end + z_label_off,
                tick_pos:     end,
                text:         format_tick(val),
                is_axis_title: false,
            });
        }
        if !axis_texts[2].is_empty() {
            let mid = Vec3::new(z_x_edge, z_y_edge, center.z);
            labels.push(LabelAnchor {
                world_pos:    mid + z_pad_off,
                tick_pos:     mid,
                text:         axis_texts[2].clone(),
                is_axis_title: true,
            });
        }
    }

    GridGeometry { vertices: verts, labels }
}

pub fn format_tick_pub(v: f32) -> String { format_tick(v) }

fn format_tick(v: f32) -> String {
    if v.abs() >= 1000.0 || (v.abs() < 0.01 && v != 0.0) {
        format!("{:.2e}", v)
    } else {
        format!("{:.3}", v)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

/// Compute a 6-bit sentinel that identifies which face of the bounding box the
/// camera is on along each perpendicular axis, plus whether each axis is a
/// depth axis (camera nearly aligned with it).  The grid geometry must be
/// rebuilt whenever this value changes.
///
/// Bit layout:
///   bit 0 — camera.y < center.y       (below  → X ticks flip to ceiling)
///   bit 1 — camera.z < center.z       (behind → X/Y ticks flip to back face)
///   bit 2 — camera.x < center.x       (left   → Y/Z ticks flip X-face)
///   bit 3 — depth axis is X  (|cam_dir.x| > 0.97 → X ticks/grid suppressed)
///   bit 4 — depth axis is Y  (|cam_dir.y| > 0.97 → Y ticks/grid suppressed)
///   bit 5 — depth axis is Z  (|cam_dir.z| > 0.97 → Z ticks/grid suppressed)
pub fn face_bits(camera_eye: Vec3, center: Vec3) -> u8 {
    let side =
          ((camera_eye.y < center.y) as u8)
        | (((camera_eye.z < center.z) as u8) << 1)
        | (((camera_eye.x < center.x) as u8) << 2);
    let cam_dir = (center - camera_eye).normalize_or_zero();
    let depth =
          ((cam_dir.x.abs() > 0.97) as u8) << 3
        | ((cam_dir.y.abs() > 0.97) as u8) << 4
        | ((cam_dir.z.abs() > 0.97) as u8) << 5;
    side | depth
}
