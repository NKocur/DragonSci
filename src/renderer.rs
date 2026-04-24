use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use wgpu::util::DeviceExt;

use crate::camera::{Camera, CameraState};
use crate::grid::{axis_ticks_log, build_grid, face_bits, format_tick_with_fmt, tick_step, LineVertex};

// ── GPU data structures ───────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    screen_size: [f32; 2],
    /// 0 = circle (soft), 1 = square, 2 = gaussian
    style: u32,
    _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PointInstance {
    pub position: [f32; 3],
    pub size: f32,
    pub color: [f32; 3],
    pub alpha: f32,
}

// ── Growable GPU buffer ───────────────────────────────────────────────────────

struct GrowableBuffer {
    buf: Option<wgpu::Buffer>,
    capacity: u64,
    usage: wgpu::BufferUsages,
}

impl GrowableBuffer {
    fn new(usage: wgpu::BufferUsages) -> Self {
        Self { buf: None, capacity: 0, usage }
    }

    /// Upload `data` bytes. Reallocates (with 1.5x headroom) only when capacity is exceeded.
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) {
        let needed = data.len() as u64;
        if needed == 0 {
            return;
        }
        if needed > self.capacity {
            let new_cap = needed + needed / 2;
            self.buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: new_cap,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.capacity = new_cap;
        }
        queue.write_buffer(self.buf.as_ref().unwrap(), 0, data);
    }

    fn slice(&self) -> Option<wgpu::BufferSlice<'_>> {
        self.buf.as_ref().map(|b| b.slice(..))
    }

    /// Grow the buffer to hold at least `needed` bytes, potentially reallocating.
    /// Returns `true` when a new GPU buffer was created (all previous data lost).
    fn ensure_capacity(&mut self, device: &wgpu::Device, needed: u64) -> bool {
        if needed <= self.capacity { return false; }
        let new_cap = needed + needed / 2;
        self.buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: new_cap,
            usage: self.usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.capacity = new_cap;
        true
    }

    /// Write `data` at `byte_offset` into the already-allocated buffer.
    fn write_at(&self, queue: &wgpu::Queue, byte_offset: u64, data: &[u8]) {
        if let Some(buf) = &self.buf {
            queue.write_buffer(buf, byte_offset, data);
        }
    }
}

// ── Line overlay actor ────────────────────────────────────────────────────────

struct LineActor {
    id: u32,
    buf: GrowableBuffer,
    vertex_count: u32,
    visible: bool,
    data_min: Vec3,
    data_max: Vec3,
}

struct Chart2DLineActor {
    id: u32,
    buf: GrowableBuffer,
    vertex_count: u32,
    visible: bool,
    x: Vec<f32>,
    y: Vec<f32>,
    color: [f32; 3],
    line_width: f32,
    /// Marker style: 0 = none, 1 = square.
    marker_style: u8,
    /// Marker size in screen pixels.
    marker_size_px: f32,
    /// Draw one marker every N data points (1 = every point).
    marker_every: u32,
    /// Marker geometry in NDC space (rendered with overlay/identity bind group).
    marker_buf: GrowableBuffer,
    marker_vertex_count: u32,
}

/// A filled reference band in data space (axhspan / axvspan).
/// Use `f32::NEG_INFINITY`/`f32::INFINITY` for the "full extent" axis (x for hspan, y for vspan).
struct Chart2DSpanActor {
    id: u32,
    x0: f32, x1: f32,
    y0: f32, y1: f32,
    color: [f32; 4],  // RGBA — alpha is honoured by the alpha-blending pipeline
    buf: GrowableBuffer,
    vertex_count: u32,
}

/// An infinite reference line in data space (axhline / axvline).
/// The line geometry is regenerated from the current axis limits on each rebuild.
struct Chart2DRefLine {
    id: u32,
    value: f32,        // y coordinate for hline, x coordinate for vline
    is_vertical: bool, // false = horizontal (axhline), true = vertical (axvline)
    color: [f32; 3],
    line_width: f32,
    buf: GrowableBuffer,
    vertex_count: u32,
}

// ── Mesh overlay actor (convex hulls, ellipsoids) ────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

/// Vertex format for GPU-side chart2d thick lines (48 bytes).
/// Stores chart-space coordinates (log10 already applied for log scales) so that
/// axis-limit changes only require a uniform buffer write — no vertex rebuilds.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ThickLineVert {
    pos_prev:   [f32; 2],
    pos_curr:   [f32; 2],
    pos_next:   [f32; 2],
    side:       f32,       // +1.0 = left edge, -1.0 = right edge
    line_width: f32,       // pixels
    color:      [f32; 4],
}

struct MeshActor {
    id: u64,
    vbuf: GrowableBuffer,
    ibuf: GrowableBuffer,
    index_count: u32,
    visible: bool,
    wireframe: bool,
    color: [f32; 4],
    data_min: Vec3,
    data_max: Vec3,
    /// Original vertex positions (data-space) — retained so style-only updates
    /// can re-bake a new color without Python retransferring geometry arrays.
    positions: Vec<[f32; 3]>,
    /// Original triangle indices — retained so wireframe/solid toggling can
    /// rebuild the index buffer without Python retransferring index arrays.
    triangle_indices: Vec<[u32; 3]>,
}

// ── Screen-space pick cache (projected positions + 16 px spatial grid) ────────

const GRID_CELL_PX: f32 = 16.0;

/// Per-actor cache of screen-projected positions plus a coarse spatial grid for
/// sub-linear pick queries.  Rebuilt lazily when VP matrix or pixel dimensions change.
struct ScreenPickCache {
    vp: glam::Mat4,
    w: f32,
    h: f32,
    cols: u32,
    rows: u32,
    /// Per-point screen position in pixels (`None` = clipped / behind camera).
    screen_xy: Vec<Option<[f32; 2]>>,
    /// Prefix-sum: `cell_start[i]` is the first index into `sorted_pts` for grid cell `i`.
    cell_start: Vec<u32>,
    /// Point indices sorted by their row-major grid cell id.
    sorted_pts: Vec<u32>,
}

impl ScreenPickCache {
    fn build(positions: &[[f32; 3]], vp: glam::Mat4, w: f32, h: f32) -> Self {
        let cols = ((w / GRID_CELL_PX).ceil() as u32).max(1);
        let rows = ((h / GRID_CELL_PX).ceil() as u32).max(1);
        let n_cells = (cols * rows) as usize;

        let screen_xy: Vec<Option<[f32; 2]>> = positions.iter().map(|&p| {
            let clip = vp * Vec3::from(p).extend(1.0);
            if clip.w <= 0.0 { return None; }
            let ndc = clip.truncate() / clip.w;
            if ndc.x.abs() > 1.05 || ndc.y.abs() > 1.05 { return None; }
            Some([(ndc.x + 1.0) * 0.5 * w, (1.0 - ndc.y) * 0.5 * h])
        }).collect();

        // First pass: count points per cell.
        let mut cell_count = vec![0u32; n_cells];
        for xy in &screen_xy {
            if let Some([sx, sy]) = xy {
                let cx = (*sx / GRID_CELL_PX) as u32;
                let cy = (*sy / GRID_CELL_PX) as u32;
                if cx < cols && cy < rows {
                    cell_count[(cy * cols + cx) as usize] += 1;
                }
            }
        }

        // Build prefix-sum table.
        let mut cell_start = vec![0u32; n_cells + 1];
        for i in 0..n_cells {
            cell_start[i + 1] = cell_start[i] + cell_count[i];
        }

        // Second pass: fill sorted_pts using per-cell write cursors.
        let total = *cell_start.last().unwrap() as usize;
        let mut sorted_pts = vec![0u32; total];
        let mut cursors = cell_start[..n_cells].to_vec();
        for (pt_idx, xy) in screen_xy.iter().enumerate() {
            if let Some([sx, sy]) = xy {
                let cx = (*sx / GRID_CELL_PX) as u32;
                let cy = (*sy / GRID_CELL_PX) as u32;
                if cx < cols && cy < rows {
                    let cell = (cy * cols + cx) as usize;
                    sorted_pts[cursors[cell] as usize] = pt_idx as u32;
                    cursors[cell] += 1;
                }
            }
        }

        Self { vp, w, h, cols, rows, screen_xy, cell_start, sorted_pts }
    }

    /// Invoke `f(point_index, [sx, sy])` for every point whose grid cell overlaps
    /// `[sx0, sy0]–[sx1, sy1]`.  Points in border cells that fall outside the
    /// rectangle are also visited; the caller does the exact test.
    fn for_each_in_rect<F: FnMut(u32, [f32; 2])>(
        &self, sx0: f32, sy0: f32, sx1: f32, sy1: f32, mut f: F,
    ) {
        let cx0 = ((sx0 / GRID_CELL_PX).floor() as i32).max(0) as u32;
        let cy0 = ((sy0 / GRID_CELL_PX).floor() as i32).max(0) as u32;
        let cx1 = ((sx1 / GRID_CELL_PX).ceil() as i32)
            .min(self.cols as i32 - 1).max(0) as u32;
        let cy1 = ((sy1 / GRID_CELL_PX).ceil() as i32)
            .min(self.rows as i32 - 1).max(0) as u32;
        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                let cell = (cy * self.cols + cx) as usize;
                let start = self.cell_start[cell] as usize;
                let end   = self.cell_start[cell + 1] as usize;
                for &pt in &self.sorted_pts[start..end] {
                    if let Some(xy) = self.screen_xy[pt as usize] {
                        f(pt, xy);
                    }
                }
            }
        }
    }

    /// Like `for_each_in_rect` but skips cells that are entirely within the
    /// inner rectangle `[ix0, iy0]–[ix1, iy1]` (already searched).
    /// Used by the expanding-ring fallback in `pick_point`.
    fn for_each_in_ring<F: FnMut(u32, [f32; 2])>(
        &self,
        ox0: f32, oy0: f32, ox1: f32, oy1: f32,  // outer box
        ix0: f32, iy0: f32, ix1: f32, iy1: f32,  // inner box (skip)
        mut f: F,
    ) {
        let cx0 = ((ox0 / GRID_CELL_PX).floor() as i32).max(0) as u32;
        let cy0 = ((oy0 / GRID_CELL_PX).floor() as i32).max(0) as u32;
        let cx1 = ((ox1 / GRID_CELL_PX).ceil() as i32)
            .min(self.cols as i32 - 1).max(0) as u32;
        let cy1 = ((oy1 / GRID_CELL_PX).ceil() as i32)
            .min(self.rows as i32 - 1).max(0) as u32;
        // Inner cell range (inclusive) — cells fully inside inner box.
        let icx0 = ((ix0 / GRID_CELL_PX).ceil() as i32).max(0) as u32;
        let icy0 = ((iy0 / GRID_CELL_PX).ceil() as i32).max(0) as u32;
        let icx1 = ((ix1 / GRID_CELL_PX).floor() as i32)
            .min(self.cols as i32 - 1).max(0) as u32;
        let icy1 = ((iy1 / GRID_CELL_PX).floor() as i32)
            .min(self.rows as i32 - 1).max(0) as u32;
        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                // Skip cells entirely within the previously searched inner box.
                if cx >= icx0 && cx <= icx1 && cy >= icy0 && cy <= icy1 {
                    continue;
                }
                let cell = (cy * self.cols + cx) as usize;
                let start = self.cell_start[cell] as usize;
                let end   = self.cell_start[cell + 1] as usize;
                for &pt in &self.sorted_pts[start..end] {
                    if let Some(xy) = self.screen_xy[pt as usize] {
                        f(pt, xy);
                    }
                }
            }
        }
    }
}

// ── Streaming buffer metadata ─────────────────────────────────────────────────

/// Whether a stream actor stops when its buffer is full (Append) or overwrites
/// the oldest points in a circular fashion (Ring).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// Accept points until `capacity` is reached; further writes are silently ignored.
    Append,
    /// Overwrite the oldest points when the buffer is full (circular buffer).
    Ring,
}

struct StreamInfo {
    capacity: u32,
    /// Index of the next slot to write (modulo capacity).
    write_head: u32,
    mode: StreamMode,
}

// ── Actor (a single uploadable point cloud) ───────────────────────────────────

struct Actor {
    id: u32,
    buf: GrowableBuffer,
    positions: Vec<[f32; 3]>,   // CPU copy for picking / pick-cache rebuild
    count: u32,
    visible: bool,
    data_min: Vec3,
    data_max: Vec3,
    /// Lazily-built screen-projection cache with spatial grid for sub-linear picks.
    /// Keyed on `(vp, w, h)`; invalidated on camera change or resize.
    pick_cache: Option<ScreenPickCache>,
    /// Present only for stream actors; absent for regular actors.
    stream: Option<StreamInfo>,
}

impl Actor {
    /// Rebuild `pick_cache` when VP matrix or pixel dimensions have changed.
    /// No-op when positions are not stored (pick storage disabled).
    fn ensure_pick_cache(&mut self, vp: glam::Mat4, w: f32, h: f32) {
        if self.positions.is_empty() {
            self.pick_cache = None;
            return;
        }
        if let Some(ref c) = self.pick_cache {
            if c.vp == vp && c.w == w && c.h == h { return; }
        }
        self.pick_cache = Some(ScreenPickCache::build(&self.positions, vp, w, h));
    }
}

// ── Screenshot resource cache ─────────────────────────────────────────────────

struct ScreenshotCache {
    w: u32,
    h: u32,
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,
    readback: wgpu::Buffer,
    padded_row: u32,
}

// ── Render surface (windowed vs. offscreen) ───────────────────────────────────

enum RenderSurface {
    Windowed {
        surface: wgpu::Surface<'static>,
        surface_config: wgpu::SurfaceConfiguration,
    },
    /// No OS window — renders to an off-screen texture and returns raw bytes.
    Offscreen,
}

// ── Cached label (pre-shaped, world position only) ────────────────────────────

struct CachedLabel {
    glyph_buf: Buffer,
    world_pos: Vec3,
    tick_pos: Vec3,
    is_axis_title: bool,
}

/// Screen-space label for the scalar bar (position in pixels).
struct ScalarBarLabel {
    glyph_buf: Buffer,
    px: f32,
    py: f32,
}

// ── User label (world-space, per-label color/size/anchor) ─────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LabelAnchor {
    Center,
    Left,
    Right,
    Top,
    Bottom,
}

impl LabelAnchor {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Left,
            2 => Self::Right,
            3 => Self::Top,
            4 => Self::Bottom,
            _ => Self::Center,
        }
    }
}

struct UserLabel {
    id: u64,
    text: String,
    glyph_buf: Buffer,
    world_pos: Vec3,
    color: [f32; 4],
    size: f32,
    anchor: LabelAnchor,
    visible: bool,
}

fn build_label_buffer(font_system: &mut FontSystem, text: &str, size: f32) -> Buffer {
    let line_h = size * 1.4;
    let mut buf = Buffer::new(font_system, Metrics::new(size, line_h));
    buf.set_size(font_system, Some(512.0), Some(line_h * 2.0));
    buf.set_text(font_system, text, Attrs::new().family(Family::SansSerif), Shaping::Basic);
    buf.shape_until_scroll(font_system, false);
    buf
}

// ── Chart2D dedicated rendering path ─────────────────────────────────────────

/// Which role a chart2d text label plays (determines pixel-position formula).
#[derive(Clone, Copy, PartialEq)]
enum Chart2DLabelKind { XTick, YTick, XTitle, YTitle, ChartTitle }

/// Pre-shaped legend text label (pixel-space; position known at build time).
struct Chart2DLegendLabel {
    glyph_buf: Buffer,
    px: f32,
    py: f32,
}

/// A pre-shaped text label for the chart2d axis system.
/// Pixel positions are computed each frame from the stored chart state + window size,
/// so the labels stay correct across window resizes without re-shaping.
struct Chart2DLabel {
    /// NDC x (for XTick/XTitle) or NDC y (for YTick/YTitle) of the label's anchor.
    ndc_val: f32,
    glyph_buf: Buffer,
    kind: Chart2DLabelKind,
}

/// Describes the 2D chart frame: data domain, viewport fractions, axis titles.
struct Chart2DState {
    plot_left:   f32,   // viewport fraction [0,1], left edge of the plot area
    plot_right:  f32,   // viewport fraction [0,1], right edge
    plot_top:    f32,   // viewport fraction [0,1], top edge (smaller = closer to top of window)
    plot_bottom: f32,   // viewport fraction [0,1], bottom edge
    x0: f32, x1: f32,  // xlim (data domain)
    y0: f32, y1: f32,  // ylim (data domain)
    x_label: String,
    y_label: String,
    /// Optional chart-level title displayed above the plot area.
    title: String,
    /// Cached tick values for x axis — used to skip glyph reshaping when unchanged.
    x_tick_cache: Vec<f32>,
    /// Stable major-step interval for X ticks/grid during animated x-range updates.
    x_tick_step: f32,
    /// Stable major-step interval for Y ticks/grid during autoscaled y-range updates.
    y_tick_step: f32,
    /// Tick label format for X axis: "default" | "sci" | "time" | "int"
    x_tick_format: String,
    /// Tick label format for Y axis.
    y_tick_format: String,
    /// When true, X axis uses a log₁₀ scale.
    x_log_scale: bool,
    /// When true, Y axis uses a log₁₀ scale.
    y_log_scale: bool,
}

impl Chart2DState {
    /// Map a data-space x value to chart (transform-input) space.
    /// When log scale is on this is log₁₀(v); otherwise v itself.
    #[inline] fn chart_x(&self, v: f32) -> f32 {
        if self.x_log_scale { v.max(1e-30).log10() } else { v }
    }
    #[inline] fn chart_y(&self, v: f32) -> f32 {
        if self.y_log_scale { v.max(1e-30).log10() } else { v }
    }
    /// Effective x0/x1 in chart space (log-transformed when log scale is on).
    #[inline] fn effective_x0(&self) -> f32 { self.chart_x(self.x0) }
    #[inline] fn effective_x1(&self) -> f32 { self.chart_x(self.x1) }
    #[inline] fn effective_y0(&self) -> f32 { self.chart_y(self.y0) }
    #[inline] fn effective_y1(&self) -> f32 { self.chart_y(self.y1) }

    /// NDC x = scale_x * chart_x(data_x) + offset_x
    fn scale_x(&self) -> f32 {
        2.0 * (self.plot_right - self.plot_left) / (self.effective_x1() - self.effective_x0()).abs().max(1e-10)
    }
    fn offset_x(&self) -> f32 {
        (2.0 * self.plot_left - 1.0) - self.effective_x0() * self.scale_x()
    }
    /// NDC y = scale_y * chart_y(data_y) + offset_y
    fn scale_y(&self) -> f32 {
        2.0 * (self.plot_bottom - self.plot_top) / (self.effective_y1() - self.effective_y0()).abs().max(1e-10)
    }
    fn offset_y(&self) -> f32 {
        (1.0 - 2.0 * self.plot_bottom) - self.effective_y0() * self.scale_y()
    }
    /// Pack the 2D affine transform as a full Uniforms struct so the same line
    /// pipeline and bind-group layout can be reused without a new pipeline.
    /// The view_proj matrix encodes: ndc = M * [data_x, data_y, 0, 1]^T
    fn as_uniforms(&self, screen_w: f32, screen_h: f32) -> Uniforms {
        let sx = self.scale_x();
        let sy = self.scale_y();
        let ox = self.offset_x();
        let oy = self.offset_y();
        // Column-major 4×4: col[i][j] = row j of column i
        Uniforms {
            view_proj: [
                [sx,  0.0, 0.0, 0.0],  // col 0
                [0.0, sy,  0.0, 0.0],  // col 1
                [0.0, 0.0, 1.0, 0.0],  // col 2
                [ox,  oy,  0.0, 1.0],  // col 3 (translation)
            ],
            screen_size: [screen_w, screen_h],
            style: 0,
            _pad: 0.0,
        }
    }
}

// ── Renderer ─────────────────────────────────────────────────────────────────

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_surface: RenderSurface,

    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    point_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,

    actors: Vec<Actor>,
    next_actor_id: u32,
    /// ID of the actor created by the most recent `set_points` call.
    /// Kept alive across calls so its GPU buffer can be reused.
    scene_actor_id: Option<u32>,

    line_buf: GrowableBuffer,
    line_count: u32,

    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    pub camera: Camera,
    fit_center: Vec3,
    fit_radius: f32,

    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    viewport: Viewport,
    cached_labels: Vec<CachedLabel>,
    atlas_trim_counter: u32,
    last_grid_min: Option<Vec3>,
    last_grid_max: Option<Vec3>,
    last_data_min: Option<Vec3>,
    last_data_max: Option<Vec3>,
    tick_override: [Option<usize>; 3],
    axis_visible: [bool; 3],

    // Scalar bar overlay (screen-space, drawn with identity view_proj)
    scalar_bar_buf: GrowableBuffer,
    scalar_bar_line_count: u32,
    overlay_bind_group: wgpu::BindGroup,  // identity-matrix uniform
    scalar_bar_labels: Vec<ScalarBarLabel>,
    scalar_bar_visible: bool,

    // Legend overlay (screen-space, same pipeline as scalar bar)
    legend_buf: GrowableBuffer,
    legend_line_count: u32,
    legend_labels: Vec<ScalarBarLabel>,
    legend_visible: bool,
    // Stored parameters so the legend can be rebuilt when scalar bar visibility changes.
    legend_title_stored: String,
    legend_items_stored: Vec<(String, [f32; 3])>,
    legend_position_stored: u8,

    // Selection rectangle overlay (screen-space, same pipeline as scalar bar)
    sel_rect_buf: GrowableBuffer,
    sel_rect_visible: bool,

    // Lasso overlay (screen-space polyline, same pipeline)
    lasso_buf: GrowableBuffer,
    lasso_vert_count: u32,
    lasso_visible: bool,
    /// Accumulated screen-space points for the active lasso gesture (screen pixels).
    lasso_pts: Vec<[f32; 2]>,

    // Screenshot resource cache (reused across calls when dimensions match)
    screenshot_cache: Option<ScreenshotCache>,

    // User-defined line overlay actors (depth-tested, world space)
    line_actors: Vec<LineActor>,
    next_line_actor_id: u32,

    // Orientation axes (computed per-frame from camera rotation, drawn as overlay)
    axes_buf: GrowableBuffer,
    axes_visible: bool,

    line_pipeline_nodepth: wgpu::RenderPipeline,

    surface_format: wgpu::TextureFormat,
    width: u32,
    height: u32,

    /// Active point style: 0 = circle, 1 = square, 2 = gaussian
    point_style: u32,
    /// LOD divisor: draw only first `count / lod_factor` instances (1 = full quality)
    lod_factor: u32,

    // ── Visual appearance ─────────────────────────────────────────────────────
    grid_visible: bool,
    major_grid_planes: bool,
    minor_grid_planes: bool,
    bg_color: [f64; 4],
    axis_label_texts: [String; 3],
    /// 3-bit sentinel tracking which bounding-box face the camera is on per axis.
    /// 0xFF on init so the first render always builds the geometry with the real eye.
    grid_face_bits: u8,

    /// When false, actor position vecs are kept empty to save RAM.
    /// Pick/hover operations silently skip actors without stored positions.
    pub store_pick_data: bool,

    // ── User-defined world-space text labels ──────────────────────────────────
    user_labels: Vec<UserLabel>,
    next_user_label_id: u64,

    // ── Mesh overlay actors (convex hulls, ellipsoids) ────────────────────────
    mesh_actors: Vec<MeshActor>,
    next_mesh_actor_id: u64,
    mesh_pipeline_opaque: wgpu::RenderPipeline,
    mesh_pipeline_transparent: wgpu::RenderPipeline,
    mesh_pipeline_wireframe: wgpu::RenderPipeline,
    /// Pipeline for chart2d spans, ref lines, markers and cursors (MeshVertex / chart-space).
    chart2d_line_pipeline: wgpu::RenderPipeline,
    /// Pipeline for chart2d data lines (ThickLineVert / GPU-side miter join).
    chart2d_thick_pipeline: wgpu::RenderPipeline,

    // ── Chart2D dedicated rendering path ─────────────────────────────────────
    /// When Some, the renderer is in 2D chart mode: world-space grid/scatter is
    /// replaced by NDC-space axis geometry + chart-transform data lines.
    chart2d_state: Option<Chart2DState>,
    /// Uniform buffer holding the chart2d 4×4 affine transform (same Uniforms
    /// layout as the main buffer — reuses the existing pipeline/bind-group layout).
    chart2d_uniform_buf: wgpu::Buffer,
    /// Bind group pointing at chart2d_uniform_buf (same layout as uniform_bind_group).
    chart2d_bind_group: wgpu::BindGroup,
    /// NDC-space chart chrome: frame, X/Y grid lines, and X/Y tick marks.
    chart2d_axis_buf: GrowableBuffer,
    chart2d_axis_count: u32,
    /// Data-space chart lines rendered as thick triangle meshes.
    chart2d_line_actors: Vec<Chart2DLineActor>,
    next_chart2d_line_id: u32,
    /// Pre-shaped text labels for the chart2d axis; positions computed each frame.
    chart2d_text_labels: Vec<Chart2DLabel>,
    /// Chart2D legend swatches (colored rectangles in NDC/screen space).
    chart2d_legend_buf: GrowableBuffer,
    chart2d_legend_swatch_count: u32,
    /// Chart2D legend text labels (pixel-space positions fixed at build time).
    chart2d_legend_text_labels: Vec<Chart2DLegendLabel>,
    chart2d_legend_visible: bool,
    /// Chart2D cursor crosshair — two thick line segments (H + V) in data space,
    /// rendered inside the plot-rect scissor.
    chart2d_cursor_buf: GrowableBuffer,
    chart2d_cursor_vertex_count: u32,
    chart2d_cursor_visible: bool,
    /// Reference bands (axhspan / axvspan).
    chart2d_span_actors: Vec<Chart2DSpanActor>,
    /// Reference lines (axhline / axvline).
    chart2d_ref_lines: Vec<Chart2DRefLine>,
    /// Monotone ID counter shared by spans and ref lines.
    next_chart2d_overlay_id: u32,
}

impl Renderer {
    pub fn new(
        raw_window_handle: raw_window_handle::RawWindowHandle,
        raw_display_handle: raw_window_handle::RawDisplayHandle,
        width: u32,
        height: u32,
        present_mode: wgpu::PresentMode,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface: wgpu::Surface<'static> = unsafe {
            let s = instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle,
                raw_window_handle,
            })?;
            std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(s)
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or("No suitable GPU adapter found")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("dragonsci"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let (depth_texture, depth_view) = make_depth_texture(&device, width.max(1), height.max(1));

        let dummy_uniforms = Uniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            screen_size: [width as f32, height as f32],
            style: 0,
            _pad: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&dummy_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform_bg"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Overlay uniform: identity view_proj so vertices are in NDC space directly.
        let overlay_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("overlay_uniforms"),
            contents: bytemuck::bytes_of(&dummy_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let overlay_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay_bg"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: overlay_uniform_buf.as_entire_binding(),
            }],
        });

        // Chart2D uniform: same layout, but COPY_DST so set_chart2d() can update it.
        let chart2d_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chart2d_uniforms"),
            contents: bytemuck::bytes_of(&dummy_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chart2d_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chart2d_bg"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: chart2d_uniform_buf.as_entire_binding(),
            }],
        });

        let point_pipeline = build_point_pipeline(
            &device,
            &uniform_layout,
            surface_format,
            include_str!("shaders/points.wgsl"),
        );
        let line_pipeline = build_line_pipeline(
            &device, &uniform_layout, surface_format,
            include_str!("shaders/lines.wgsl"), true,
        );
        let line_pipeline_nodepth = build_line_pipeline(
            &device, &uniform_layout, surface_format,
            include_str!("shaders/lines.wgsl"), false,
        );

        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let glyph_cache = Cache::new(&device);
        let viewport = Viewport::new(&device, &glyph_cache);
        let mut text_atlas = TextAtlas::new(&device, &queue, &glyph_cache, surface_format);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        );

        let camera = Camera::fit(Vec3::ZERO, 1.0, width as f32 / height.max(1) as f32);

        let mesh_wgsl = include_str!("shaders/mesh.wgsl");
        let mesh_pipeline_opaque      = build_mesh_pipeline(&device, &uniform_layout, surface_format, mesh_wgsl, false);
        let mesh_pipeline_transparent = build_mesh_pipeline(&device, &uniform_layout, surface_format, mesh_wgsl, true);
        let mesh_pipeline_wireframe   = build_wireframe_pipeline(&device, &uniform_layout, surface_format, mesh_wgsl);
        let chart2d_line_pipeline = build_chart2d_line_pipeline(
            &device, &uniform_layout, surface_format, mesh_wgsl,
        );
        let chart2d_thick_wgsl = include_str!("shaders/chart2d_lines.wgsl");
        let chart2d_thick_pipeline = build_chart2d_thick_pipeline(
            &device, &uniform_layout, surface_format, chart2d_thick_wgsl,
        );

        Ok(Self {
            device,
            queue,
            render_surface: RenderSurface::Windowed { surface, surface_config },
            depth_texture,
            depth_view,
            point_pipeline,
            line_pipeline,
            actors: Vec::new(),
            next_actor_id: 0,
            scene_actor_id: None,
            line_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            line_count: 0,
            uniform_buffer,
            uniform_bind_group,
            camera,
            fit_center: Vec3::ZERO,
            fit_radius: 1.0,
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            viewport,
            cached_labels: Vec::new(),
            atlas_trim_counter: 0,
            last_grid_min: None,
            last_grid_max: None,
            last_data_min: None,
            last_data_max: None,
            tick_override: [None; 3],
            axis_visible: [true; 3],
            scalar_bar_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            scalar_bar_line_count: 0,
            overlay_bind_group,
            scalar_bar_labels: Vec::new(),
            scalar_bar_visible: false,
            legend_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            legend_line_count: 0,
            legend_labels: Vec::new(),
            legend_visible: false,
            legend_title_stored: String::new(),
            legend_items_stored: Vec::new(),
            legend_position_stored: 0,
            sel_rect_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            sel_rect_visible: false,
            lasso_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            lasso_vert_count: 0,
            lasso_visible: false,
            lasso_pts: Vec::new(),
            screenshot_cache: None,
            line_actors: Vec::new(),
            next_line_actor_id: 0,
            axes_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            axes_visible: false,
            line_pipeline_nodepth,
            surface_format,
            width,
            height,
            point_style: 0,
            lod_factor: 1,
            grid_visible: true,
            major_grid_planes: false,
            minor_grid_planes: false,
            bg_color: [0.05, 0.05, 0.07, 1.0],
            axis_label_texts: ["X".to_string(), "Y".to_string(), "Z".to_string()],
            grid_face_bits: 0xFF,
            store_pick_data: true,
            user_labels: Vec::new(),
            next_user_label_id: 0,
            mesh_actors: Vec::new(),
            next_mesh_actor_id: 0,
            mesh_pipeline_opaque,
            mesh_pipeline_transparent,
            mesh_pipeline_wireframe,
            chart2d_line_pipeline,
            chart2d_thick_pipeline,
            chart2d_state: None,
            chart2d_uniform_buf,
            chart2d_bind_group,
            chart2d_axis_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            chart2d_axis_count: 0,
            chart2d_line_actors: Vec::new(),
            next_chart2d_line_id: 0,
            chart2d_text_labels: Vec::new(),
            chart2d_legend_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            chart2d_legend_swatch_count: 0,
            chart2d_legend_text_labels: Vec::new(),
            chart2d_legend_visible: false,
            chart2d_cursor_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            chart2d_cursor_vertex_count: 0,
            chart2d_cursor_visible: false,
            chart2d_span_actors: Vec::new(),
            chart2d_ref_lines: Vec::new(),
            next_chart2d_overlay_id: 0,
        })
    }

    /// Create a headless renderer that renders to an off-screen texture.
    /// No OS window is required; suitable for Jupyter / server-side rendering.
    pub fn new_offscreen(
        width: u32,
        height: u32,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // No surface — adapter is selected on power preference alone.
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or("No suitable GPU adapter found for offscreen rendering")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("dragonsci_offscreen"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))?;

        // Fixed known-good format for offscreen render targets.
        let surface_format = wgpu::TextureFormat::Rgba8UnormSrgb;

        let (depth_texture, depth_view) = make_depth_texture(&device, width.max(1), height.max(1));

        let dummy_uniforms = Uniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            screen_size: [width as f32, height as f32],
            style: 0,
            _pad: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&dummy_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform_bg"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let overlay_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("overlay_uniforms"),
            contents: bytemuck::bytes_of(&dummy_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let overlay_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay_bg"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: overlay_uniform_buf.as_entire_binding(),
            }],
        });

        let chart2d_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chart2d_uniforms"),
            contents: bytemuck::bytes_of(&dummy_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chart2d_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chart2d_bg"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: chart2d_uniform_buf.as_entire_binding(),
            }],
        });

        let point_pipeline = build_point_pipeline(
            &device, &uniform_layout, surface_format,
            include_str!("shaders/points.wgsl"),
        );
        let line_pipeline = build_line_pipeline(
            &device, &uniform_layout, surface_format,
            include_str!("shaders/lines.wgsl"), true,
        );
        let line_pipeline_nodepth = build_line_pipeline(
            &device, &uniform_layout, surface_format,
            include_str!("shaders/lines.wgsl"), false,
        );

        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let glyph_cache = Cache::new(&device);
        let viewport = Viewport::new(&device, &glyph_cache);
        let mut text_atlas = TextAtlas::new(&device, &queue, &glyph_cache, surface_format);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        );

        let camera = Camera::fit(Vec3::ZERO, 1.0, width as f32 / height.max(1) as f32);

        let mesh_wgsl = include_str!("shaders/mesh.wgsl");
        let mesh_pipeline_opaque      = build_mesh_pipeline(&device, &uniform_layout, surface_format, mesh_wgsl, false);
        let mesh_pipeline_transparent = build_mesh_pipeline(&device, &uniform_layout, surface_format, mesh_wgsl, true);
        let mesh_pipeline_wireframe   = build_wireframe_pipeline(&device, &uniform_layout, surface_format, mesh_wgsl);
        let chart2d_line_pipeline = build_chart2d_line_pipeline(
            &device, &uniform_layout, surface_format, mesh_wgsl,
        );
        let chart2d_thick_wgsl = include_str!("shaders/chart2d_lines.wgsl");
        let chart2d_thick_pipeline = build_chart2d_thick_pipeline(
            &device, &uniform_layout, surface_format, chart2d_thick_wgsl,
        );

        Ok(Self {
            device,
            queue,
            render_surface: RenderSurface::Offscreen,
            depth_texture,
            depth_view,
            point_pipeline,
            line_pipeline,
            actors: Vec::new(),
            next_actor_id: 0,
            scene_actor_id: None,
            line_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            line_count: 0,
            uniform_buffer,
            uniform_bind_group,
            camera,
            fit_center: Vec3::ZERO,
            fit_radius: 1.0,
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            viewport,
            cached_labels: Vec::new(),
            atlas_trim_counter: 0,
            last_grid_min: None,
            last_grid_max: None,
            last_data_min: None,
            last_data_max: None,
            tick_override: [None; 3],
            axis_visible: [true; 3],
            scalar_bar_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            scalar_bar_line_count: 0,
            overlay_bind_group,
            scalar_bar_labels: Vec::new(),
            scalar_bar_visible: false,
            legend_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            legend_line_count: 0,
            legend_labels: Vec::new(),
            legend_visible: false,
            legend_title_stored: String::new(),
            legend_items_stored: Vec::new(),
            legend_position_stored: 0,
            sel_rect_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            sel_rect_visible: false,
            lasso_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            lasso_vert_count: 0,
            lasso_visible: false,
            lasso_pts: Vec::new(),
            screenshot_cache: None,
            line_actors: Vec::new(),
            next_line_actor_id: 0,
            axes_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            axes_visible: false,
            line_pipeline_nodepth,
            surface_format,
            width,
            height,
            point_style: 0,
            lod_factor: 1,
            grid_visible: true,
            major_grid_planes: false,
            minor_grid_planes: false,
            bg_color: [0.05, 0.05, 0.07, 1.0],
            axis_label_texts: ["X".to_string(), "Y".to_string(), "Z".to_string()],
            grid_face_bits: 0xFF,
            store_pick_data: true, // always store CPU positions in offscreen mode
            user_labels: Vec::new(),
            next_user_label_id: 0,
            mesh_actors: Vec::new(),
            next_mesh_actor_id: 0,
            mesh_pipeline_opaque,
            mesh_pipeline_transparent,
            mesh_pipeline_wireframe,
            chart2d_line_pipeline,
            chart2d_thick_pipeline,
            chart2d_state: None,
            chart2d_uniform_buf,
            chart2d_bind_group,
            chart2d_axis_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            chart2d_axis_count: 0,
            chart2d_line_actors: Vec::new(),
            next_chart2d_line_id: 0,
            chart2d_text_labels: Vec::new(),
            chart2d_legend_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            chart2d_legend_swatch_count: 0,
            chart2d_legend_text_labels: Vec::new(),
            chart2d_legend_visible: false,
            chart2d_cursor_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            chart2d_cursor_vertex_count: 0,
            chart2d_cursor_visible: false,
            chart2d_span_actors: Vec::new(),
            chart2d_ref_lines: Vec::new(),
            next_chart2d_overlay_id: 0,
        })
    }

    pub fn set_pick_storage(&mut self, enabled: bool) {
        self.store_pick_data = enabled;
        if !enabled {
            for actor in &mut self.actors {
                actor.positions.clear();
                actor.positions.shrink_to_fit();
                actor.pick_cache = None;
            }
        }
    }

    // ── Scalar bar ────────────────────────────────────────────────────────────

    /// Build or update the scalar bar overlay.
    /// `cpts` is the resolved colormap table; `vmin`/`vmax` are the display limits;
    /// `log_scale` mirrors the normalization used for the points.
    pub fn set_scalar_bar(
        &mut self,
        visible: bool,
        vmin: f32,
        vmax: f32,
        log_scale: bool,
        cpts: &[[f32; 3]],
        title: &str,
    ) {
        self.scalar_bar_visible = visible;
        if !visible {
            self.scalar_bar_line_count = 0;
            self.scalar_bar_labels.clear();
            return;
        }

        // Scalar bar geometry: a vertical gradient strip in NDC space.
        // The bar occupies the top-right corner; exact pixel sizes are computed
        // from the current viewport dimensions.
        let (w, h) = (self.width as f32, self.height as f32);
        // Bar dimensions & position in pixels
        let bar_w = 16.0_f32;
        let bar_h = (h * 0.45).min(220.0).max(60.0);
        let margin_r = 52.0_f32;  // from right edge
        let margin_t = 32.0_f32;  // from top edge
        let bar_x1 = w - margin_r - bar_w;  // left edge in pixels
        let bar_x2 = w - margin_r;           // right edge
        let bar_y1 = margin_t;               // top edge
        let bar_y2 = margin_t + bar_h;       // bottom edge

        // Convert pixel coords to NDC [-1, 1]
        let to_ndc = |px: f32, py: f32| -> [f32; 3] {
            [(px / w) * 2.0 - 1.0, 1.0 - (py / h) * 2.0, 0.0]
        };

        // Gradient: N horizontal line pairs, each colored by the colormap.
        const GRAD_STEPS: usize = 64;
        let mut verts: Vec<LineVertex> = Vec::with_capacity(GRAD_STEPS * 2);
        for i in 0..GRAD_STEPS {
            let t_top = i as f32 / GRAD_STEPS as f32;
            let t_bot = (i + 1) as f32 / GRAD_STEPS as f32;
            let t_mid = (t_top + t_bot) * 0.5;
            // t=0 → vmax (top), t=1 → vmin (bottom)
            let color = crate::colormap::sample(cpts, 1.0 - t_mid);
            let y_top = bar_y1 + t_top * bar_h;
            let y_bot = bar_y1 + t_bot * bar_h;
            // Draw top and bottom edges of this band (both same color → solid band)
            verts.push(LineVertex { position: to_ndc(bar_x1, y_top), color });
            verts.push(LineVertex { position: to_ndc(bar_x2, y_top), color });
            verts.push(LineVertex { position: to_ndc(bar_x1, y_bot), color });
            verts.push(LineVertex { position: to_ndc(bar_x2, y_bot), color });
        }
        // Thin white border around the bar
        let border = [0.7_f32, 0.7, 0.7];
        let corners = [
            to_ndc(bar_x1, bar_y1), to_ndc(bar_x2, bar_y1),
            to_ndc(bar_x2, bar_y1), to_ndc(bar_x2, bar_y2),
            to_ndc(bar_x2, bar_y2), to_ndc(bar_x1, bar_y2),
            to_ndc(bar_x1, bar_y2), to_ndc(bar_x1, bar_y1),
        ];
        for i in (0..corners.len()).step_by(2) {
            verts.push(LineVertex { position: corners[i],   color: border });
            verts.push(LineVertex { position: corners[i+1], color: border });
        }

        self.scalar_bar_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
        self.scalar_bar_line_count = verts.len() as u32;

        // Labels: title + tick values
        self.scalar_bar_labels.clear();
        let label_x = bar_x2 + 4.0;  // just right of bar

        let mut add_label = |text: String, px: f32, py: f32| {
            let mut buf = Buffer::new(&mut self.font_system, Metrics::new(11.0, 14.0));
            buf.set_size(&mut self.font_system, Some(80.0), Some(20.0));
            buf.set_text(
                &mut self.font_system,
                &text,
                Attrs::new().family(Family::SansSerif),
                Shaping::Basic,
            );
            buf.shape_until_scroll(&mut self.font_system, false);
            self.scalar_bar_labels.push(ScalarBarLabel { glyph_buf: buf, px, py });
        };

        // Tick labels: vmax at top, vmin at bottom, one or two intermediate
        let tick_count = 5_usize;
        for i in 0..=tick_count {
            let t = i as f32 / tick_count as f32;  // 0 = top (vmax), 1 = bottom (vmin)
            let val = if log_scale {
                let lmin = vmin.max(1e-10).ln();
                let lmax = vmax.max(1e-10).ln();
                (lmin + t * (lmax - lmin)).exp()
            } else {
                vmax + t * (vmin - vmax)
            };
            let py = bar_y1 + t * bar_h - 5.0;
            add_label(crate::grid::format_tick_pub(val), label_x, py);
        }
        if !title.is_empty() {
            // Title above the bar
            add_label(title.to_string(), bar_x1, bar_y1 - 16.0);
        }

        // If a top-right legend is active, rebuild it so it repositions below
        // (or back to the top) now that scalar bar visibility has changed.
        self.rebuild_legend_overlay();
    }

    // ── Legend overlay ────────────────────────────────────────────────────────

    /// Build or update the categorical legend overlay.
    /// `items` is a slice of (label, rgb_color) pairs.
    /// `position`: 0=top-right, 1=top-left, 2=bottom-right, 3=bottom-left.
    /// When position is top-right and the scalar bar is visible, the legend is
    /// automatically placed below the scalar bar to avoid overlap.
    pub fn set_legend(
        &mut self,
        visible: bool,
        title: &str,
        items: &[(&str, [f32; 3])],
        position: u8,
    ) {
        self.legend_visible = visible;
        self.legend_title_stored = title.to_string();
        self.legend_items_stored = items.iter().map(|&(l, c)| (l.to_string(), c)).collect();
        self.legend_position_stored = position;
        self.rebuild_legend_overlay();
    }

    /// Re-layout the legend using stored parameters.  Called by `set_legend()`
    /// and by `set_scalar_bar()` so that toggling the scalar bar automatically
    /// shifts the legend out of the way without requiring a round-trip to Python.
    fn rebuild_legend_overlay(&mut self) {
        if !self.legend_visible || self.legend_items_stored.is_empty() {
            self.legend_line_count = 0;
            self.legend_labels.clear();
            return;
        }

        // Clone stored state to avoid borrow conflicts with font_system.
        let title   = self.legend_title_stored.clone();
        let items   = self.legend_items_stored.clone();
        let position = self.legend_position_stored;

        let (w, h) = (self.width as f32, self.height as f32);

        const MARGIN: f32 = 10.0;
        const PAD_H: f32 = 8.0;
        const PAD_V: f32 = 7.0;
        const SWATCH: f32 = 11.0;
        const SWATCH_GAP: f32 = 5.0;
        const ROW_H: f32 = 15.0;
        const ROW_GAP: f32 = 3.0;
        const TITLE_H: f32 = 15.0;
        const TITLE_GAP: f32 = 5.0;
        const TEXT_W: f32 = 130.0;
        const BOX_W: f32 = PAD_H + SWATCH + SWATCH_GAP + TEXT_W + PAD_H;

        let n = items.len() as f32;
        let has_title = !title.is_empty();
        let box_h = PAD_V
            + if has_title { TITLE_H + TITLE_GAP } else { 0.0 }
            + n * ROW_H
            + (n - 1.0).max(0.0) * ROW_GAP
            + PAD_V;

        let (box_x1, box_y1) = match position {
            1 => (MARGIN, MARGIN),
            2 => (w - MARGIN - BOX_W, h - MARGIN - box_h),
            3 => (MARGIN, h - MARGIN - box_h),
            _ => {
                // top-right: stack below the scalar bar when it is visible so
                // the two overlays never overlap without any user intervention.
                let y_start = if self.scalar_bar_visible {
                    let bar_h = (h * 0.45).min(220.0_f32).max(60.0_f32);
                    32.0_f32 + bar_h + 10.0  // 10 px gap below the bar
                } else {
                    MARGIN
                };
                (w - MARGIN - BOX_W, y_start)
            }
        };
        let box_x2 = box_x1 + BOX_W;
        let box_y2 = box_y1 + box_h;

        let to_ndc = |px: f32, py: f32| -> [f32; 3] {
            [(px / w) * 2.0 - 1.0, 1.0 - (py / h) * 2.0, 0.0]
        };

        let mut verts: Vec<LineVertex> = Vec::new();

        // Background fill
        let bg = [0.10_f32, 0.10, 0.13];
        for y in (box_y1 as i32)..=(box_y2 as i32) {
            verts.push(LineVertex { position: to_ndc(box_x1, y as f32), color: bg });
            verts.push(LineVertex { position: to_ndc(box_x2, y as f32), color: bg });
        }

        // Swatch fills
        let swatch_x1 = box_x1 + PAD_H;
        let swatch_x2 = swatch_x1 + SWATCH;
        let mut cursor_y = box_y1 + PAD_V + if has_title { TITLE_H + TITLE_GAP } else { 0.0 };
        for &(_, color) in &items {
            let sy1 = cursor_y + (ROW_H - SWATCH) * 0.5;
            let sy2 = sy1 + SWATCH;
            for y in (sy1 as i32)..=(sy2 as i32) {
                verts.push(LineVertex { position: to_ndc(swatch_x1, y as f32), color });
                verts.push(LineVertex { position: to_ndc(swatch_x2, y as f32), color });
            }
            cursor_y += ROW_H + ROW_GAP;
        }

        // Border
        let border = [0.45_f32, 0.45, 0.52];
        let corners = [
            to_ndc(box_x1, box_y1), to_ndc(box_x2, box_y1),
            to_ndc(box_x2, box_y1), to_ndc(box_x2, box_y2),
            to_ndc(box_x2, box_y2), to_ndc(box_x1, box_y2),
            to_ndc(box_x1, box_y2), to_ndc(box_x1, box_y1),
        ];
        for i in (0..corners.len()).step_by(2) {
            verts.push(LineVertex { position: corners[i],   color: border });
            verts.push(LineVertex { position: corners[i+1], color: border });
        }

        self.legend_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
        self.legend_line_count = verts.len() as u32;

        // Text labels
        self.legend_labels.clear();
        let text_x = swatch_x2 + SWATCH_GAP;
        cursor_y = box_y1 + PAD_V;

        if has_title {
            let mut buf = Buffer::new(&mut self.font_system, Metrics::new(12.0, 15.0));
            buf.set_size(&mut self.font_system, Some(BOX_W - PAD_H * 2.0), Some(TITLE_H));
            buf.set_text(&mut self.font_system, &title,
                Attrs::new().family(Family::SansSerif), Shaping::Basic);
            buf.shape_until_scroll(&mut self.font_system, false);
            self.legend_labels.push(ScalarBarLabel { glyph_buf: buf, px: box_x1 + PAD_H, py: cursor_y });
            cursor_y += TITLE_H + TITLE_GAP;
        }
        for (label, _) in &items {
            let mut buf = Buffer::new(&mut self.font_system, Metrics::new(11.0, 14.0));
            buf.set_size(&mut self.font_system, Some(TEXT_W), Some(ROW_H));
            buf.set_text(&mut self.font_system, label,
                Attrs::new().family(Family::SansSerif), Shaping::Basic);
            buf.shape_until_scroll(&mut self.font_system, false);
            let ty = cursor_y + (ROW_H - 11.0) * 0.5;
            self.legend_labels.push(ScalarBarLabel { glyph_buf: buf, px: text_x, py: ty });
            cursor_y += ROW_H + ROW_GAP;
        }
    }

    // ── Data upload / actor management ───────────────────────────────────────

    /// Replace the entire scene with a single point cloud.
    ///
    /// Reuses the existing scene actor's GPU buffer when one is present so that
    /// repeated full-scene refreshes avoid GPU reallocation when the point count
    /// stays similar or shrinks (GrowableBuffer only reallocates on growth).
    /// Returns the actor ID, or `None` when `count == 0`.
    pub fn set_points(&mut self, instances: &[PointInstance], positions: Vec<[f32; 3]>, count: u32, data_min: Vec3, data_max: Vec3) -> Option<u32> {
        // Drop add_points actors but keep the scene slot so its GPU buffer survives.
        match self.scene_actor_id {
            Some(sid) => self.actors.retain(|a| a.id == sid),
            None => self.actors.clear(),
        }

        if count == 0 {
            self.actors.clear();
            self.scene_actor_id = None;
            return None;
        }

        // Reuse the existing scene actor's GPU buffer.
        if let Some(sid) = self.scene_actor_id {
            if let Some(actor) = self.actors.iter_mut().find(|a| a.id == sid) {
                actor.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(instances));
                actor.positions = if self.store_pick_data { positions } else { vec![] };
                actor.count = count;
                actor.data_min = data_min;
                actor.data_max = data_max;
                actor.visible = true;
                actor.pick_cache = None;
                return Some(sid);
            }
        }

        // First call — create the persistent scene slot.
        let id = self._add_actor_buf(instances, positions, count, data_min, data_max);
        self.scene_actor_id = Some(id);
        Some(id)
    }

    fn _add_actor_buf(&mut self, instances: &[PointInstance], mut positions: Vec<[f32; 3]>, count: u32, data_min: Vec3, data_max: Vec3) -> u32 {
        if !self.store_pick_data {
            positions.clear();
            positions.shrink_to_fit();
        }
        let id = self.next_actor_id;
        self.next_actor_id += 1;
        let mut actor = Actor {
            id,
            buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            positions,
            count,
            visible: true,
            data_min,
            data_max,
            pick_cache: None,
            stream: None,
        };
        actor.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(instances));
        self.actors.push(actor);
        id
    }

    /// Add a new point cloud actor and return its ID.
    pub fn add_actor(&mut self, instances: &[PointInstance], positions: Vec<[f32; 3]>, count: u32, data_min: Vec3, data_max: Vec3) -> u32 {
        self._add_actor_buf(instances, positions, count, data_min, data_max)
    }

    /// Replace data for an existing actor in-place. Returns false if not found.
    pub fn update_actor_data(&mut self, id: u32, instances: &[PointInstance], positions: Vec<[f32; 3]>, count: u32, data_min: Vec3, data_max: Vec3) -> bool {
        let store = self.store_pick_data;
        if let Some(a) = self.actors.iter_mut().find(|a| a.id == id) {
            a.count = count;
            a.data_min = data_min;
            a.data_max = data_max;
            a.positions = if store { positions } else { vec![] };
            a.pick_cache = None;  // positions changed — invalidate pick cache
            if count > 0 {
                a.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(instances));
            }
            true
        } else {
            false
        }
    }

    // ── Streaming actor management ───────────────────────────────────────────

    /// Pre-allocate a fixed-capacity stream actor.
    ///
    /// The GPU buffer is sized to hold exactly `max_points` instances and is
    /// never reallocated.  `instances` / `positions` are optional initial data
    /// (at most `max_points` entries are consumed).
    pub fn create_stream(
        &mut self,
        max_points: u32,
        mode: StreamMode,
        instances: &[PointInstance],
        positions: Vec<[f32; 3]>,
        count: u32,
        data_min: Vec3,
        data_max: Vec3,
    ) -> u32 {
        let id = self.next_actor_id;
        self.next_actor_id += 1;
        let inst_size = std::mem::size_of::<PointInstance>() as u64;
        let cap_bytes = max_points as u64 * inst_size;

        let mut buf = GrowableBuffer::new(wgpu::BufferUsages::VERTEX);
        buf.ensure_capacity(&self.device, cap_bytes);

        let fill = count.min(max_points);
        if fill > 0 {
            buf.write_at(&self.queue, 0, bytemuck::cast_slice(&instances[..fill as usize]));
        }

        let stream_positions = if self.store_pick_data {
            positions[..fill as usize].to_vec()
        } else {
            vec![]
        };
        self.actors.push(Actor {
            id,
            buf,
            positions: stream_positions,
            count: fill,
            visible: true,
            data_min,
            data_max,
            pick_cache: None,
            stream: Some(StreamInfo {
                capacity: max_points,
                write_head: fill % max_points.max(1),
                mode,
            }),
        });
        id
    }

    /// Append `instances` to the stream actor identified by `id`.
    ///
    /// - **Append mode**: ignores points that would overflow the fixed capacity.
    /// - **Ring mode**: overwrites the oldest points using two contiguous GPU
    ///   writes (head → capacity, 0 → head) to avoid per-point overhead.
    ///
    /// The running bounds are expanded (never contracted) to include the new data.
    /// Returns `(false, _)` when the ID is not a stream actor; `(true, bounds_grew)` otherwise.
    pub fn append_to_stream(
        &mut self,
        id: u32,
        instances: &[PointInstance],
        new_positions: &[[f32; 3]],
        new_data_min: Vec3,
        new_data_max: Vec3,
    ) -> (bool, bool) {
        let actor = match self.actors.iter_mut().find(|a| a.id == id) {
            Some(a) => a,
            None => return (false, false),
        };
        let si = match actor.stream.as_mut() {
            Some(si) => si,
            None => return (false, false),
        };

        let n = instances.len();
        if n == 0 { return (true, false); }
        let inst_size = std::mem::size_of::<PointInstance>();
        let cap = si.capacity as usize;

        match si.mode {
            StreamMode::Append => {
                let space = cap.saturating_sub(actor.count as usize);
                let to_write = n.min(space);
                if to_write == 0 { return (true, false); }
                let offset = actor.count as u64 * inst_size as u64;
                actor.buf.write_at(&self.queue, offset,
                    bytemuck::cast_slice(&instances[..to_write]));
                if !actor.positions.is_empty() {
                    actor.positions.extend_from_slice(&new_positions[..to_write]);
                }
                actor.count += to_write as u32;
            }
            StreamMode::Ring => {
                let to_write = n.min(cap);
                let head = si.write_head as usize;

                // Two-segment write to handle the wrap-around boundary.
                let seg1_end = (head + to_write).min(cap);
                let seg1_len = seg1_end - head;
                let seg2_len = to_write - seg1_len;

                actor.buf.write_at(&self.queue, (head * inst_size) as u64,
                    bytemuck::cast_slice(&instances[..seg1_len]));
                if seg2_len > 0 {
                    actor.buf.write_at(&self.queue, 0,
                        bytemuck::cast_slice(&instances[seg1_len..to_write]));
                }

                let new_count = (actor.count as usize + to_write).min(cap);
                // Mirror the ring write in the CPU positions vec (for picking).
                if !actor.positions.is_empty() {
                    if actor.positions.len() < new_count {
                        actor.positions.resize(new_count, [0.0; 3]);
                    }
                    for i in 0..seg1_len {
                        actor.positions[head + i] = new_positions[i];
                    }
                    for i in 0..seg2_len {
                        actor.positions[i] = new_positions[seg1_len + i];
                    }
                }

                si.write_head = ((head + to_write) % cap) as u32;
                actor.count = new_count as u32;
            }
        }

        // Expand bounds (we never contract — acceptable for streaming visualisation).
        let prev_min = actor.data_min;
        let prev_max = actor.data_max;
        actor.data_min = actor.data_min.min(new_data_min);
        actor.data_max = actor.data_max.max(new_data_max);
        let bounds_grew = actor.data_min != prev_min || actor.data_max != prev_max;
        actor.pick_cache = None;
        (true, bounds_grew)
    }

    /// Reset a stream actor to empty; preserves the pre-allocated GPU capacity.
    /// Returns `false` when the ID is not a stream actor.
    pub fn clear_stream(&mut self, id: u32) -> bool {
        let actor = match self.actors.iter_mut().find(|a| a.id == id) {
            Some(a) => a,
            None => return false,
        };
        if actor.stream.is_none() { return false; }
        actor.count = 0;
        actor.positions.clear();
        actor.pick_cache = None;
        if let Some(ref mut si) = actor.stream {
            si.write_head = 0;
        }
        // Reset bounds to "empty" so they don't affect actor_union_bounds.
        actor.data_min = Vec3::splat(f32::INFINITY);
        actor.data_max = Vec3::splat(f32::NEG_INFINITY);
        true
    }

    /// Remove an actor by ID. Returns false if not found.
    pub fn remove_actor(&mut self, id: u32) -> bool {
        if let Some(pos) = self.actors.iter().position(|a| a.id == id) {
            self.actors.remove(pos);
            if self.scene_actor_id == Some(id) {
                self.scene_actor_id = None;
            }
            true
        } else {
            false
        }
    }

    /// Show or hide an actor. Returns false if not found.
    pub fn set_actor_visibility(&mut self, id: u32, visible: bool) -> bool {
        if let Some(a) = self.actors.iter_mut().find(|a| a.id == id) {
            a.visible = visible;
            true
        } else {
            false
        }
    }

    /// Remove all actors.
    pub fn clear_actors(&mut self) {
        self.actors.clear();
        self.scene_actor_id = None;
    }

    /// Union of visible point actor and line overlay bounds. None when the scene is empty.
    pub fn actor_union_bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut bmin = Vec3::splat(f32::INFINITY);
        let mut bmax = Vec3::splat(f32::NEG_INFINITY);
        let mut any = false;
        for a in &self.actors {
            if !a.visible || a.count == 0 { continue; }
            bmin = bmin.min(a.data_min);
            bmax = bmax.max(a.data_max);
            any = true;
        }
        for la in &self.line_actors {
            if !la.visible || la.vertex_count == 0 { continue; }
            bmin = bmin.min(la.data_min);
            bmax = bmax.max(la.data_max);
            any = true;
        }
        for ma in &self.mesh_actors {
            if !ma.visible || ma.index_count == 0 { continue; }
            bmin = bmin.min(ma.data_min);
            bmax = bmax.max(ma.data_max);
            any = true;
        }
        if any { Some((bmin, bmax)) } else { None }
    }

    // ── Picking ───────────────────────────────────────────────────────────────

    /// Return `(actor_id, point_index, world_pos)` for the point closest to
    /// the given screen position. `None` when the scene is empty.
    ///
    /// Uses a per-actor screen-projection cache to avoid recomputing the
    /// view-projection transform on every call while the camera is stationary.
    pub fn pick_point(&mut self, screen_x: f32, screen_y: f32) -> Option<(u32, u32, [f32; 3])> {
        let vp = self.camera.view_proj();
        let (w, h) = (self.width as f32, self.height as f32);
        let mut best_dist_sq = f32::MAX;
        let mut best: Option<(u32, u32, [f32; 3])> = None;

        // Fast path: search only cells within a ±2-cell radius (32 px).
        // Any point whose screen projection is within ~48 px of the cursor is
        // guaranteed to be in the searched cells (cell boundary slack included).
        const R: f32 = GRID_CELL_PX * 2.0;
        for actor in &mut self.actors {
            if !actor.visible { continue; }
            actor.ensure_pick_cache(vp, w, h);
            let (cache, positions, actor_id) = match actor.pick_cache.as_ref() {
                Some(c) => (c, &actor.positions, actor.id),
                None => continue,
            };
            cache.for_each_in_rect(
                screen_x - R, screen_y - R, screen_x + R, screen_y + R,
                |i, [sx, sy]| {
                    let d_sq = (sx - screen_x).powi(2) + (sy - screen_y).powi(2);
                    if d_sq < best_dist_sq {
                        best_dist_sq = d_sq;
                        best = Some((actor_id, i, positions[i as usize]));
                    }
                },
            );
        }

        // Fallback: expand the search ring outward one cell at a time.
        // After visiting all cells in the outer box of radius `search_r`, any
        // unvisited point lies in a cell whose near edge is at least `search_r`
        // screen-pixels away — so if best_dist_sq ≤ search_r², we cannot improve
        // further and stop early.  This avoids the O(N) full-scan in most scenes.
        if best_dist_sq > R * R {
            let mut inner_r = R;
            let max_r = w.hypot(h); // screen diagonal — no point is farther
            while inner_r < max_r {
                let outer_r = inner_r + GRID_CELL_PX;
                for actor in &mut self.actors {
                    if !actor.visible { continue; }
                    // Cache is already built from the fast-path loop above.
                    let (cache, positions, actor_id) = match actor.pick_cache.as_ref() {
                        Some(c) => (c, &actor.positions, actor.id),
                        None => continue,
                    };
                    cache.for_each_in_ring(
                        screen_x - outer_r, screen_y - outer_r,
                        screen_x + outer_r, screen_y + outer_r,
                        screen_x - inner_r, screen_y - inner_r,
                        screen_x + inner_r, screen_y + inner_r,
                        |i, [sx, sy]| {
                            let d_sq = (sx - screen_x).powi(2) + (sy - screen_y).powi(2);
                            if d_sq < best_dist_sq {
                                best_dist_sq = d_sq;
                                best = Some((actor_id, i, positions[i as usize]));
                            }
                        },
                    );
                }
                // Every unvisited point is in a cell starting at or beyond inner_r.
                if best_dist_sq <= inner_r * inner_r {
                    break;
                }
                inner_r = outer_r;
            }
        }
        best
    }

    /// Return `(actor_id, point_index)` for all visible points whose screen
    /// projection falls inside the given screen-space rectangle.
    pub fn pick_rectangle(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<(u32, u32)> {
        let vp = self.camera.view_proj();
        let (w, h) = (self.width as f32, self.height as f32);
        let sx_min = x0.min(x1);
        let sx_max = x0.max(x1);
        let sy_min = y0.min(y1);
        let sy_max = y0.max(y1);

        let mut result = Vec::new();
        for actor in &mut self.actors {
            if !actor.visible { continue; }
            actor.ensure_pick_cache(vp, w, h);
            let (cache, actor_id) = match actor.pick_cache.as_ref() {
                Some(c) => (c, actor.id),
                None => continue,
            };
            cache.for_each_in_rect(sx_min, sy_min, sx_max, sy_max, |i, [sx, sy]| {
                if sx >= sx_min && sx <= sx_max && sy >= sy_min && sy <= sy_max {
                    result.push((actor_id, i));
                }
            });
        }
        result
    }

    // ── Selection rectangle overlay ───────────────────────────────────────────

    /// Draw an in-progress selection rectangle (screen coords, pixels).
    pub fn set_selection_rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        let (w, h) = (self.width as f32, self.height as f32);
        let to_ndc = |px: f32, py: f32| -> [f32; 3] {
            [(px / w) * 2.0 - 1.0, 1.0 - (py / h) * 2.0, 0.0]
        };
        let tl = to_ndc(x0, y0);
        let tr = to_ndc(x1, y0);
        let br = to_ndc(x1, y1);
        let bl = to_ndc(x0, y1);
        let col = [0.4_f32, 0.8, 1.0];
        let verts: [LineVertex; 8] = [
            LineVertex { position: tl, color: col },
            LineVertex { position: tr, color: col },
            LineVertex { position: tr, color: col },
            LineVertex { position: br, color: col },
            LineVertex { position: br, color: col },
            LineVertex { position: bl, color: col },
            LineVertex { position: bl, color: col },
            LineVertex { position: tl, color: col },
        ];
        self.sel_rect_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
        self.sel_rect_visible = true;
    }

    pub fn clear_selection_rect(&mut self) {
        self.sel_rect_visible = false;
    }

    // ── Lasso overlay + picking ───────────────────────────────────────────────

    /// Update the in-progress lasso polyline (screen coords, pixels).
    /// Draws an open polyline; the caller appends the first point again to close it.
    pub fn set_lasso_path(&mut self, screen_verts: &[[f32; 2]]) {
        let n = screen_verts.len();
        if n < 2 {
            self.lasso_visible = false;
            return;
        }
        let (w, h) = (self.width as f32, self.height as f32);
        let to_ndc = |px: f32, py: f32| -> [f32; 3] {
            [(px / w) * 2.0 - 1.0, 1.0 - (py / h) * 2.0, 0.0]
        };
        let col = [0.4_f32, 0.8, 1.0];
        let mut verts = Vec::with_capacity((n - 1) * 2);
        for i in 0..(n - 1) {
            let [ax, ay] = screen_verts[i];
            let [bx, by] = screen_verts[i + 1];
            verts.push(LineVertex { position: to_ndc(ax, ay), color: col });
            verts.push(LineVertex { position: to_ndc(bx, by), color: col });
        }
        self.lasso_vert_count = verts.len() as u32;
        self.lasso_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
        self.lasso_visible = true;
    }

    pub fn clear_lasso_path(&mut self) {
        self.lasso_visible = false;
    }

    // ── Incremental lasso API ─────────────────────────────────────────────────

    /// Start a new freehand lasso gesture at screen point `(sx, sy)` (pixels).
    /// Clears any previous lasso path.
    pub fn lasso_begin(&mut self, sx: f32, sy: f32) {
        self.lasso_pts.clear();
        self.lasso_pts.push([sx, sy]);
        self.lasso_visible = false;
        self.lasso_vert_count = 0;
    }

    /// Extend the active lasso by one screen-space point, updating the overlay
    /// incrementally (O(1) GPU writes, amortised O(1) per call).
    pub fn lasso_extend(&mut self, sx: f32, sy: f32) {
        const SV: u64 = std::mem::size_of::<LineVertex>() as u64;
        let col = [0.4_f32, 0.8, 1.0];
        let n = self.lasso_pts.len();
        if n == 0 { return; }
        self.lasso_pts.push([sx, sy]);
        let new_n = n + 1;   // point count after push

        let (w, h) = (self.width as f32, self.height as f32);
        let to_v = |px: f32, py: f32| LineVertex {
            position: [(px / w) * 2.0 - 1.0, 1.0 - (py / h) * 2.0, 0.0],
            color: col,
        };

        // Segment count: 1 open seg if new_n==2; new_n segs (n-1 open + 1 close) if new_n>=3.
        let seg_count = if new_n >= 3 { new_n as u64 } else { 1u64 };
        let needed = seg_count * 2 * SV;

        let reallocated = self.lasso_buf.ensure_capacity(&self.device, needed);

        if reallocated {
            // Full rebuild after reallocation (rare — only when gesture exceeds previous capacity).
            let pts = &self.lasso_pts;
            let m = pts.len();
            let mut verts: Vec<LineVertex> = Vec::with_capacity(seg_count as usize * 2);
            for i in 0..(m - 1) {
                let [ax, ay] = pts[i];
                let [bx, by] = pts[i + 1];
                verts.push(to_v(ax, ay));
                verts.push(to_v(bx, by));
            }
            if m >= 3 {
                let [cx, cy] = pts[m - 1];
                let [dx, dy] = pts[0];
                verts.push(to_v(cx, cy));
                verts.push(to_v(dx, dy));
            }
            self.lasso_buf.write_at(&self.queue, 0, bytemuck::cast_slice(&verts));
            self.lasso_vert_count = verts.len() as u32;
        } else if new_n == 2 {
            // First segment [pt0 → pt1].
            let [ax, ay] = self.lasso_pts[0];
            let [bx, by] = self.lasso_pts[1];
            let verts = [to_v(ax, ay), to_v(bx, by)];
            self.lasso_buf.write_at(&self.queue, 0, bytemuck::cast_slice(&verts));
            self.lasso_vert_count = 2;
        } else {
            // n >= 2: overwrite old close segment with new open segment,
            // then append new close segment.
            // Buffer layout: [seg(0→1), ..., seg(n-2→n-1), close(n-1→0)]
            // close_offset = 2*(n-1) verts * SV  (works for n==2: appends, no overwrite)
            let close_offset = 2 * (n as u64 - 1) * SV;
            let [ax, ay] = self.lasso_pts[n - 1];  // prev last pt
            let [bx, by] = self.lasso_pts[n];       // new last pt
            let [cx, cy] = self.lasso_pts[0];        // first pt
            let new_segs = [to_v(ax, ay), to_v(bx, by), to_v(bx, by), to_v(cx, cy)];
            self.lasso_buf.write_at(&self.queue, close_offset, bytemuck::cast_slice(&new_segs));
            self.lasso_vert_count = (2 * new_n) as u32;
        }

        self.lasso_visible = true;
    }

    /// Finish the lasso: run polygon picking against the recorded path,
    /// clear the visual overlay, and return matching `(actor_id, point_index)` pairs.
    /// Returns an empty vec when fewer than 3 points were recorded.
    pub fn lasso_end(&mut self) -> Vec<(u32, u32)> {
        let pts = std::mem::take(&mut self.lasso_pts);
        self.lasso_visible = false;
        self.lasso_vert_count = 0;
        if pts.len() < 3 {
            return vec![];
        }
        self.pick_polygon(&pts)
    }

    /// Cancel the active lasso without picking — hides the overlay.
    pub fn lasso_cancel(&mut self) {
        self.lasso_pts.clear();
        self.lasso_visible = false;
        self.lasso_vert_count = 0;
    }

    /// Return `(actor_id, point_index)` for all visible points inside a
    /// screen-space polygon defined by `screen_verts` (pixels).
    pub fn pick_polygon(&mut self, screen_verts: &[[f32; 2]]) -> Vec<(u32, u32)> {
        if screen_verts.len() < 3 {
            return vec![];
        }
        let vp = self.camera.view_proj();
        let (w, h) = (self.width as f32, self.height as f32);
        let mut result = Vec::new();
        // Compute bounding box of the polygon to limit grid cell traversal.
        let bb_x0 = screen_verts.iter().map(|v| v[0]).fold(f32::MAX, f32::min);
        let bb_x1 = screen_verts.iter().map(|v| v[0]).fold(f32::MIN, f32::max);
        let bb_y0 = screen_verts.iter().map(|v| v[1]).fold(f32::MAX, f32::min);
        let bb_y1 = screen_verts.iter().map(|v| v[1]).fold(f32::MIN, f32::max);

        for actor in &mut self.actors {
            if !actor.visible { continue; }
            actor.ensure_pick_cache(vp, w, h);
            let (cache, actor_id) = match actor.pick_cache.as_ref() {
                Some(c) => (c, actor.id),
                None => continue,
            };
            cache.for_each_in_rect(bb_x0, bb_y0, bb_x1, bb_y1, |i, [sx, sy]| {
                if point_in_polygon(sx, sy, screen_verts) {
                    result.push((actor_id, i));
                }
            });
        }
        result
    }

    // ── Chart2D rendering path ────────────────────────────────────────────────

    /// Enable chart2D mode.  All subsequent renders use a dedicated 2D transform
    /// instead of the 3D camera.  Call again whenever xlim/ylim/labels change.
    #[allow(clippy::too_many_arguments)]
    pub fn set_chart2d(
        &mut self,
        plot_left:   f32, plot_right: f32,
        plot_top:    f32, plot_bottom: f32,
        x0: f32, x1: f32,
        y0: f32, y1: f32,
        x_label: String, y_label: String,
        y_tick_step_override: Option<f32>,
        title: String,
        x_tick_step_override: Option<f32>,
    ) {
        // Preserve format/log-scale settings from previous state across reconfigurations.
        let (x_fmt, y_fmt, x_log, y_log) = match self.chart2d_state.as_ref() {
            Some(old) => (old.x_tick_format.clone(), old.y_tick_format.clone(),
                          old.x_log_scale, old.y_log_scale),
            None => ("default".to_string(), "default".to_string(), false, false),
        };
        let x_step = if x_log { 1.0 } else {
            x_tick_step_override
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or_else(|| tick_step(x0, x1, 5))
        };
        let y_step = if y_log { 1.0 } else {
            y_tick_step_override
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or_else(|| tick_step(y0, y1, 5))
        };
        let mut state = Chart2DState {
            plot_left, plot_right, plot_top, plot_bottom,
            x0, x1, y0, y1, x_label, y_label, title,
            x_tick_cache: Vec::new(),
            x_tick_step: x_step, y_tick_step: y_step,
            x_tick_format: x_fmt, y_tick_format: y_fmt,
            x_log_scale: x_log, y_log_scale: y_log,
        };

        // Update the chart2d uniform buffer with the new affine transform.
        let u = state.as_uniforms(self.width as f32, self.height as f32);
        self.queue.write_buffer(&self.chart2d_uniform_buf, 0, bytemuck::bytes_of(&u));

        // Rebuild NDC-space axis geometry (frame, Y grid, tick marks).
        let verts = build_chart2d_axis_verts(&state, self.width, self.height);
        self.chart2d_axis_count = verts.len() as u32;
        if !verts.is_empty() {
            self.chart2d_axis_buf.upload(
                &self.device, &self.queue, bytemuck::cast_slice(&verts));
        }

        // Rebuild pre-shaped text labels for ticks + axis titles.
        self.chart2d_text_labels.clear();
        let sz_tick  = 11.0_f32;
        let sz_title = 13.0_f32;
        let sx = state.scale_x();
        let ox = state.offset_x();
        let sy = state.scale_y();
        let oy = state.offset_y();

        // X tick labels — save values in cache for future xlim-only updates.
        let x_ticks = chart2d_x_ticks(&state);
        state.x_tick_cache = x_ticks.clone();
        for &xt in &x_ticks {
            let ndc_x = sx * state.chart_x(xt) + ox;
            if ndc_x < -1.05 || ndc_x > 1.05 { continue; }
            let buf = build_label_buffer(&mut self.font_system, &format_tick_with_fmt(xt, &state.x_tick_format), sz_tick);
            self.chart2d_text_labels.push(Chart2DLabel { ndc_val: ndc_x, glyph_buf: buf, kind: Chart2DLabelKind::XTick });
        }
        // Y tick labels
        for &yt in &chart2d_y_ticks(&state) {
            let ndc_y = sy * state.chart_y(yt) + oy;
            if ndc_y < -1.05 || ndc_y > 1.05 { continue; }
            let buf = build_label_buffer(&mut self.font_system, &format_tick_with_fmt(yt, &state.y_tick_format), sz_tick);
            self.chart2d_text_labels.push(Chart2DLabel { ndc_val: ndc_y, glyph_buf: buf, kind: Chart2DLabelKind::YTick });
        }
        // X axis title
        if !state.x_label.is_empty() {
            let buf = build_label_buffer(&mut self.font_system, &state.x_label, sz_title);
            let mid_chart = (state.chart_x(x0) + state.chart_x(x1)) * 0.5;
            let mid_ndc = (sx * mid_chart + ox).clamp(-1.0, 1.0);
            self.chart2d_text_labels.push(Chart2DLabel { ndc_val: mid_ndc, glyph_buf: buf, kind: Chart2DLabelKind::XTitle });
        }
        // Y axis title
        if !state.y_label.is_empty() {
            let buf = build_label_buffer(&mut self.font_system, &state.y_label, sz_title);
            let mid_chart = (state.chart_y(y0) + state.chart_y(y1)) * 0.5;
            let mid_ndc = (sy * mid_chart + oy).clamp(-1.0, 1.0);
            self.chart2d_text_labels.push(Chart2DLabel { ndc_val: mid_ndc, glyph_buf: buf, kind: Chart2DLabelKind::YTitle });
        }
        // Chart title — centered horizontally on the plot area.
        if !state.title.is_empty() {
            let sz_chart_title = 14.0_f32;
            let buf = build_label_buffer(&mut self.font_system, &state.title, sz_chart_title);
            let plot_center_ndc = (state.plot_left + state.plot_right) - 1.0;
            self.chart2d_text_labels.push(Chart2DLabel {
                ndc_val: plot_center_ndc, glyph_buf: buf, kind: Chart2DLabelKind::ChartTitle,
            });
        }

        self.chart2d_state = Some(state);
        self.rebuild_all_chart2d_lines();
    }

    /// Fast path: slide xlim without touching y-tick / title glyphs.
    /// Called at animation rate (≈60 fps) while a time-series stream is scrolling.
    ///
    /// Cost breakdown:
    ///   Always:   uniform buffer write + chart chrome rebuild (cheap).
    ///   When tick values change: x-tick glyph reshape.
    pub fn chart2d_update_xlim(&mut self, x0: f32, x1: f32) {
        if self.chart2d_state.is_none() { return; }

        // Update state x range.
        {
            let s = self.chart2d_state.as_mut().unwrap();
            s.x0 = x0;
            s.x1 = x1;
        }
        let new_ticks = chart2d_x_ticks(self.chart2d_state.as_ref().unwrap());
        let ticks_changed = new_ticks != self.chart2d_state.as_ref().unwrap().x_tick_cache;
        self.chart2d_state.as_mut().unwrap().x_tick_cache = new_ticks.clone();

        // Always: update the affine transform.
        let u = self.chart2d_state.as_ref().unwrap()
            .as_uniforms(self.width as f32, self.height as f32);
        self.queue.write_buffer(&self.chart2d_uniform_buf, 0, bytemuck::bytes_of(&u));

        // Always: rebuild axis geometry so X tick marks track the new xlim.
        let frame_verts = build_chart2d_axis_verts(
            self.chart2d_state.as_ref().unwrap(), self.width, self.height);
        self.chart2d_axis_count = frame_verts.len() as u32;
        if !frame_verts.is_empty() {
            self.chart2d_axis_buf.upload(
                &self.device, &self.queue, bytemuck::cast_slice(&frame_verts));
        }

        // Only when tick values change: reshape glyphs.
        if ticks_changed {
            self.chart2d_text_labels.retain(|l| !matches!(l.kind, Chart2DLabelKind::XTick));
            let s = self.chart2d_state.as_ref().unwrap();
            let sx = s.scale_x(); let ox = s.offset_x();
            let fmt = s.x_tick_format.clone();
            let sz_tick = 11.0_f32;
            for &xt in &new_ticks {
                let ndc_x = sx * s.chart_x(xt) + ox;
                if ndc_x < -1.05 || ndc_x > 1.05 { continue; }
                let buf = build_label_buffer(&mut self.font_system, &format_tick_with_fmt(xt, &fmt), sz_tick);
                self.chart2d_text_labels.push(Chart2DLabel {
                    ndc_val: ndc_x, glyph_buf: buf, kind: Chart2DLabelKind::XTick,
                });
            }
        } else {
            // Tick values unchanged — just recompute NDC positions in-place.
            let s = self.chart2d_state.as_ref().unwrap();
            let sx = s.scale_x(); let ox = s.offset_x();
            let mut i = 0;
            for label in self.chart2d_text_labels.iter_mut() {
                if !matches!(label.kind, Chart2DLabelKind::XTick) { continue; }
                if i < new_ticks.len() {
                    label.ndc_val = sx * s.chart_x(new_ticks[i]) + ox;
                    i += 1;
                }
            }
        }
        // GPU-side transform handles line vertex data; only spans, ref lines, and
        // markers need geometry rebuilds on an axis-limit change.
        self.rebuild_chart2d_axis_dependent();
    }

    /// Fast path: update only the y data-range while keeping the Y tick interval fixed.
    pub fn chart2d_update_ylim(&mut self, y0: f32, y1: f32) {
        if self.chart2d_state.is_none() { return; }
        {
            let s = self.chart2d_state.as_mut().unwrap();
            s.y0 = y0;
            s.y1 = y1;
        }
        let new_ticks = chart2d_y_ticks(self.chart2d_state.as_ref().unwrap());

        let u = self.chart2d_state.as_ref().unwrap()
            .as_uniforms(self.width as f32, self.height as f32);
        self.queue.write_buffer(&self.chart2d_uniform_buf, 0, bytemuck::bytes_of(&u));

        let frame_verts = build_chart2d_axis_verts(
            self.chart2d_state.as_ref().unwrap(), self.width, self.height);
        self.chart2d_axis_count = frame_verts.len() as u32;
        if !frame_verts.is_empty() {
            self.chart2d_axis_buf.upload(
                &self.device, &self.queue, bytemuck::cast_slice(&frame_verts));
        }

        self.chart2d_text_labels.retain(|l| !matches!(l.kind, Chart2DLabelKind::YTick));
        let s = self.chart2d_state.as_ref().unwrap();
        let sy = s.scale_y(); let oy = s.offset_y();
        let fmt = s.y_tick_format.clone();
        let sz_tick = 11.0_f32;
        for &yt in &new_ticks {
            let ndc_y = sy * s.chart_y(yt) + oy;
            if ndc_y < -1.05 || ndc_y > 1.05 { continue; }
            let buf = build_label_buffer(&mut self.font_system, &format_tick_with_fmt(yt, &fmt), sz_tick);
            self.chart2d_text_labels.push(Chart2DLabel {
                ndc_val: ndc_y, glyph_buf: buf, kind: Chart2DLabelKind::YTick,
            });
        }

        let s = self.chart2d_state.as_ref().unwrap();
        let mid_chart = (s.chart_y(y0) + s.chart_y(y1)) * 0.5;
        let mid_ndc = (s.scale_y() * mid_chart + s.offset_y()).clamp(-1.0, 1.0);
        for label in self.chart2d_text_labels.iter_mut() {
            if matches!(label.kind, Chart2DLabelKind::YTitle) { label.ndc_val = mid_ndc; }
        }
        // GPU-side transform handles line vertex data; only spans, ref lines, and
        // markers need geometry rebuilds on an axis-limit change.
        self.rebuild_chart2d_axis_dependent();
    }

    /// Set the tick label format for one axis.
    ///
    /// `axis` is `"x"` or `"y"`.
    /// `fmt` is one of `"default"`, `"sci"`, `"time"`, `"int"`.
    pub fn chart2d_set_tick_format(&mut self, axis: &str, fmt: &str) {
        let Some(state) = self.chart2d_state.as_mut() else { return; };
        match axis {
            "x" => state.x_tick_format = fmt.to_string(),
            "y" => state.y_tick_format = fmt.to_string(),
            _ => return,
        }
        // Reshape all tick-label glyphs for the affected axis.
        self.rebuild_chart2d_tick_labels(axis);
    }

    /// Enable or disable log₁₀ scale on one axis.
    ///
    /// `axis` is `"x"` or `"y"`.  Triggers a full geometry rebuild.
    pub fn chart2d_set_log_scale(&mut self, axis: &str, enabled: bool) {
        let Some(state) = self.chart2d_state.as_mut() else { return; };
        match axis {
            "x" => {
                state.x_log_scale = enabled;
                // When switching to linear, recompute a sensible tick step from the
                // current range; log mode doesn't use tick_step so its stored value
                // may be stale (or was forced to 1.0 on the prior enable).
                state.x_tick_step = if enabled {
                    1.0  // log mode ignores tick_step
                } else {
                    tick_step(state.x0, state.x1, 5)
                };
            }
            "y" => {
                state.y_log_scale = enabled;
                state.y_tick_step = if enabled {
                    1.0
                } else {
                    tick_step(state.y0, state.y1, 5)
                };
            }
            _ => return,
        }
        // Full rebuild: uniform, axis geometry, labels, and all line/span geometry.
        let u = self.chart2d_state.as_ref().unwrap()
            .as_uniforms(self.width as f32, self.height as f32);
        self.queue.write_buffer(&self.chart2d_uniform_buf, 0, bytemuck::bytes_of(&u));
        let verts = build_chart2d_axis_verts(
            self.chart2d_state.as_ref().unwrap(), self.width, self.height);
        self.chart2d_axis_count = verts.len() as u32;
        if !verts.is_empty() {
            self.chart2d_axis_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
        }
        self.rebuild_chart2d_tick_labels("x");
        self.rebuild_chart2d_tick_labels("y");
        self.rebuild_all_chart2d_lines();
    }

    /// Rebuild tick-label glyphs for `"x"` or `"y"` from the current chart state.
    fn rebuild_chart2d_tick_labels(&mut self, axis: &str) {
        let Some(state) = self.chart2d_state.as_ref() else { return; };
        let sz_tick = 11.0_f32;
        match axis {
            "x" => {
                self.chart2d_text_labels.retain(|l| !matches!(l.kind, Chart2DLabelKind::XTick));
                let ticks = chart2d_x_ticks(state);
                let sx = state.scale_x(); let ox = state.offset_x();
                let fmt = state.x_tick_format.clone();
                let cx_fn: Vec<f32> = ticks.iter().map(|&v| state.chart_x(v)).collect();
                for (i, &xt) in ticks.iter().enumerate() {
                    let ndc_x = sx * cx_fn[i] + ox;
                    if ndc_x < -1.05 || ndc_x > 1.05 { continue; }
                    let buf = build_label_buffer(&mut self.font_system, &format_tick_with_fmt(xt, &fmt), sz_tick);
                    self.chart2d_text_labels.push(Chart2DLabel {
                        ndc_val: ndc_x, glyph_buf: buf, kind: Chart2DLabelKind::XTick,
                    });
                }
            }
            "y" => {
                self.chart2d_text_labels.retain(|l| !matches!(l.kind, Chart2DLabelKind::YTick));
                let ticks = chart2d_y_ticks(state);
                let sy = state.scale_y(); let oy = state.offset_y();
                let fmt = state.y_tick_format.clone();
                let cy_fn: Vec<f32> = ticks.iter().map(|&v| state.chart_y(v)).collect();
                for (i, &yt) in ticks.iter().enumerate() {
                    let ndc_y = sy * cy_fn[i] + oy;
                    if ndc_y < -1.05 || ndc_y > 1.05 { continue; }
                    let buf = build_label_buffer(&mut self.font_system, &format_tick_with_fmt(yt, &fmt), sz_tick);
                    self.chart2d_text_labels.push(Chart2DLabel {
                        ndc_val: ndc_y, glyph_buf: buf, kind: Chart2DLabelKind::YTick,
                    });
                }
            }
            _ => {}
        }
    }

    fn rebuild_chart2d_line_actor(&self, actor: &mut Chart2DLineActor) {
        let verts = xy_to_gpu_thick_verts(
            &actor.x,
            &actor.y,
            actor.color,
            actor.line_width,
            self.chart2d_state.as_ref(),
        );
        actor.vertex_count = verts.len() as u32;
        if verts.is_empty() {
            actor.buf = GrowableBuffer::new(wgpu::BufferUsages::VERTEX);
        } else {
            actor.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
        }
        // Marker geometry (square dots at every Nth data point).
        if actor.marker_style > 0 && actor.marker_every > 0 {
            if let Some(state) = self.chart2d_state.as_ref() {
                let mverts = build_marker_vertices(
                    &actor.x, &actor.y,
                    actor.marker_style, actor.marker_size_px, actor.marker_every,
                    actor.color, state, self.width, self.height,
                );
                actor.marker_vertex_count = mverts.len() as u32;
                if !mverts.is_empty() {
                    actor.marker_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&mverts));
                } else {
                    actor.marker_buf = GrowableBuffer::new(wgpu::BufferUsages::VERTEX);
                }
            } else {
                actor.marker_vertex_count = 0;
            }
        } else {
            actor.marker_vertex_count = 0;
        }
    }

    fn rebuild_all_chart2d_lines(&mut self) {
        if self.chart2d_state.is_none() {
            return;
        }
        let mut actors = std::mem::take(&mut self.chart2d_line_actors);
        for actor in &mut actors {
            self.rebuild_chart2d_line_actor(actor);
        }
        self.chart2d_line_actors = actors;

        let mut spans = std::mem::take(&mut self.chart2d_span_actors);
        for span in &mut spans {
            self.rebuild_chart2d_span_actor(span);
        }
        self.chart2d_span_actors = spans;

        let mut refs = std::mem::take(&mut self.chart2d_ref_lines);
        for rl in &mut refs {
            self.rebuild_chart2d_ref_line(rl);
        }
        self.chart2d_ref_lines = refs;
    }

    /// Rebuild geometry that depends on axis limits (spans, ref lines, marker dots) but
    /// NOT the main thick-line vertex data, which uses the GPU-side transform and is
    /// updated automatically when the uniform buffer is written.
    /// Called by `chart2d_update_xlim` / `chart2d_update_ylim` instead of the full
    /// `rebuild_all_chart2d_lines` to keep those paths O(1) for vertex data.
    fn rebuild_chart2d_axis_dependent(&mut self) {
        if self.chart2d_state.is_none() {
            return;
        }
        // Marker vertices are baked in chart-space units that scale with px_per_x/y,
        // so they must be rebuilt whenever the visible range changes.
        let mut actors = std::mem::take(&mut self.chart2d_line_actors);
        for actor in &mut actors {
            if actor.marker_style > 0 && actor.marker_every > 0 {
                if let Some(state) = self.chart2d_state.as_ref() {
                    let mverts = build_marker_vertices(
                        &actor.x, &actor.y,
                        actor.marker_style, actor.marker_size_px, actor.marker_every,
                        actor.color, state, self.width, self.height,
                    );
                    actor.marker_vertex_count = mverts.len() as u32;
                    if !mverts.is_empty() {
                        actor.marker_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&mverts));
                    } else {
                        actor.marker_buf = GrowableBuffer::new(wgpu::BufferUsages::VERTEX);
                    }
                } else {
                    actor.marker_vertex_count = 0;
                }
            }
        }
        self.chart2d_line_actors = actors;

        let mut spans = std::mem::take(&mut self.chart2d_span_actors);
        for span in &mut spans {
            self.rebuild_chart2d_span_actor(span);
        }
        self.chart2d_span_actors = spans;

        let mut refs = std::mem::take(&mut self.chart2d_ref_lines);
        for rl in &mut refs {
            self.rebuild_chart2d_ref_line(rl);
        }
        self.chart2d_ref_lines = refs;
    }

    fn rebuild_chart2d_span_actor(&self, actor: &mut Chart2DSpanActor) {
        let Some(state) = self.chart2d_state.as_ref() else {
            actor.vertex_count = 0;
            return;
        };
        // Resolve infinite extents to the current data range, then map to chart space.
        let x0 = state.chart_x(if actor.x0.is_finite() { actor.x0 } else { state.x0.min(state.x1) });
        let x1 = state.chart_x(if actor.x1.is_finite() { actor.x1 } else { state.x0.max(state.x1) });
        let y0 = state.chart_y(if actor.y0.is_finite() { actor.y0 } else { state.y0.min(state.y1) });
        let y1 = state.chart_y(if actor.y1.is_finite() { actor.y1 } else { state.y0.max(state.y1) });
        let c = actor.color;
        let verts = [
            MeshVertex { position: [x0, y0, 0.0], color: c },
            MeshVertex { position: [x1, y0, 0.0], color: c },
            MeshVertex { position: [x1, y1, 0.0], color: c },
            MeshVertex { position: [x0, y0, 0.0], color: c },
            MeshVertex { position: [x1, y1, 0.0], color: c },
            MeshVertex { position: [x0, y1, 0.0], color: c },
        ];
        actor.vertex_count = verts.len() as u32;
        actor.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
    }

    fn rebuild_chart2d_ref_line(&self, actor: &mut Chart2DRefLine) {
        let Some(state) = self.chart2d_state.as_ref() else {
            actor.vertex_count = 0;
            return;
        };
        let (x, y) = if actor.is_vertical {
            (
                vec![actor.value, actor.value],
                vec![state.y0.min(state.y1), state.y0.max(state.y1)],
            )
        } else {
            (
                vec![state.x0.min(state.x1), state.x0.max(state.x1)],
                vec![actor.value, actor.value],
            )
        };
        let verts = xy_to_thick_line_vertices(
            &x, &y, actor.color, actor.line_width,
            Some(state), self.width, self.height,
        );
        actor.vertex_count = verts.len() as u32;
        if verts.is_empty() {
            actor.buf = GrowableBuffer::new(wgpu::BufferUsages::VERTEX);
        } else {
            actor.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
        }
    }

    /// Add a polyline in data space; returns a handle.  The chart2d affine
    /// transform converts data (x,y) → NDC at render time.
    pub fn chart2d_add_line(
        &mut self,
        x: &[f32],
        y: &[f32],
        color: [f32; 3],
        line_width: f32,
    ) -> u32 {
        let id = self.next_chart2d_line_id;
        self.next_chart2d_line_id += 1;
        let mut actor = Chart2DLineActor {
            id,
            buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            vertex_count: 0,
            visible: true,
            x: x.to_vec(),
            y: y.to_vec(),
            color,
            line_width,
            marker_style: 0,
            marker_size_px: 6.0,
            marker_every: 1,
            marker_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            marker_vertex_count: 0,
        };
        self.rebuild_chart2d_line_actor(&mut actor);
        self.chart2d_line_actors.push(actor);
        id
    }

    /// Replace a chart2d polyline's geometry.
    pub fn chart2d_update_line(
        &mut self,
        id: u32,
        x: &[f32],
        y: &[f32],
        color: [f32; 3],
        line_width: f32,
    ) -> bool {
        if let Some(a) = self.chart2d_line_actors.iter_mut().find(|a| a.id == id) {
            a.x.clear();
            a.x.extend_from_slice(x);
            a.y.clear();
            a.y.extend_from_slice(y);
            a.color = color;
            a.line_width = line_width;
            let verts = xy_to_gpu_thick_verts(
                &a.x,
                &a.y,
                a.color,
                a.line_width,
                self.chart2d_state.as_ref(),
            );
            a.vertex_count = verts.len() as u32;
            if verts.is_empty() {
                a.buf = GrowableBuffer::new(wgpu::BufferUsages::VERTEX);
            } else {
                a.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
            }
            true
        } else {
            false
        }
    }

    /// Remove a chart2d polyline by handle.
    pub fn chart2d_remove_line(&mut self, id: u32) -> bool {
        if let Some(pos) = self.chart2d_line_actors.iter().position(|a| a.id == id) {
            self.chart2d_line_actors.remove(pos);
            true
        } else {
            false
        }
    }

    /// Remove all chart2d polylines.
    pub fn chart2d_clear_lines(&mut self) {
        self.chart2d_line_actors.clear();
    }

    /// Toggle visibility of a chart2d polyline without removing it.
    pub fn chart2d_set_line_visible(&mut self, id: u32, visible: bool) {
        if let Some(a) = self.chart2d_line_actors.iter_mut().find(|a| a.id == id) {
            a.visible = visible;
        }
    }

    /// Set marker style/size/spacing on an existing line actor.
    /// `style` 0 = none, 1 = square.  `every` = draw one marker every N data points.
    pub fn chart2d_set_line_markers(&mut self, id: u32, style: u8, size_px: f32, every: u32) -> bool {
        let Some(a) = self.chart2d_line_actors.iter_mut().find(|a| a.id == id) else {
            return false;
        };
        a.marker_style  = style;
        a.marker_size_px = size_px;
        a.marker_every  = every;
        let (xs, ys, color, lw) = (a.x.clone(), a.y.clone(), a.color, a.line_width);
        if let Some(state) = self.chart2d_state.as_ref() {
            if style > 0 && every > 0 {
                let mverts = build_marker_vertices(
                    &xs, &ys, style, size_px, every, color, state, self.width, self.height,
                );
                let a2 = self.chart2d_line_actors.iter_mut().find(|a| a.id == id).unwrap();
                a2.marker_vertex_count = mverts.len() as u32;
                if !mverts.is_empty() {
                    a2.marker_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&mverts));
                } else {
                    a2.marker_buf = GrowableBuffer::new(wgpu::BufferUsages::VERTEX);
                }
            } else {
                let a2 = self.chart2d_line_actors.iter_mut().find(|a| a.id == id).unwrap();
                a2.marker_vertex_count = 0;
            }
            // Also rebuild the line itself (GPU thick-line format) to keep vertex data fresh.
            let verts = xy_to_gpu_thick_verts(&xs, &ys, color, lw,
                self.chart2d_state.as_ref());
            let a3 = self.chart2d_line_actors.iter_mut().find(|a| a.id == id).unwrap();
            a3.vertex_count = verts.len() as u32;
            if verts.is_empty() {
                a3.buf = GrowableBuffer::new(wgpu::BufferUsages::VERTEX);
            } else {
                a3.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
            }
        }
        true
    }

    // ── Reference overlays (axhspan / axvspan / axhline / axvline) ────────────

    /// Add a horizontal band spanning the full x extent.  `color` is RGBA.
    pub fn chart2d_add_hspan(&mut self, y0: f32, y1: f32, color: [f32; 4]) -> u32 {
        let id = self.next_chart2d_overlay_id;
        self.next_chart2d_overlay_id += 1;
        let mut actor = Chart2DSpanActor {
            id,
            x0: f32::NEG_INFINITY, x1: f32::INFINITY,
            y0, y1,
            color,
            buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            vertex_count: 0,
        };
        self.rebuild_chart2d_span_actor(&mut actor);
        self.chart2d_span_actors.push(actor);
        id
    }

    /// Add a vertical band spanning the full y extent.  `color` is RGBA.
    pub fn chart2d_add_vspan(&mut self, x0: f32, x1: f32, color: [f32; 4]) -> u32 {
        let id = self.next_chart2d_overlay_id;
        self.next_chart2d_overlay_id += 1;
        let mut actor = Chart2DSpanActor {
            id,
            x0, x1,
            y0: f32::NEG_INFINITY, y1: f32::INFINITY,
            color,
            buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            vertex_count: 0,
        };
        self.rebuild_chart2d_span_actor(&mut actor);
        self.chart2d_span_actors.push(actor);
        id
    }

    /// Add an infinite horizontal reference line at data y = `y`.
    pub fn chart2d_add_hline(&mut self, y: f32, color: [f32; 3], line_width: f32) -> u32 {
        let id = self.next_chart2d_overlay_id;
        self.next_chart2d_overlay_id += 1;
        let mut actor = Chart2DRefLine {
            id, value: y, is_vertical: false,
            color, line_width,
            buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            vertex_count: 0,
        };
        self.rebuild_chart2d_ref_line(&mut actor);
        self.chart2d_ref_lines.push(actor);
        id
    }

    /// Add an infinite vertical reference line at data x = `x`.
    pub fn chart2d_add_vline(&mut self, x: f32, color: [f32; 3], line_width: f32) -> u32 {
        let id = self.next_chart2d_overlay_id;
        self.next_chart2d_overlay_id += 1;
        let mut actor = Chart2DRefLine {
            id, value: x, is_vertical: true,
            color, line_width,
            buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            vertex_count: 0,
        };
        self.rebuild_chart2d_ref_line(&mut actor);
        self.chart2d_ref_lines.push(actor);
        id
    }

    /// Remove a span or reference line by its overlay id.
    pub fn chart2d_remove_overlay(&mut self, id: u32) -> bool {
        if let Some(pos) = self.chart2d_span_actors.iter().position(|a| a.id == id) {
            self.chart2d_span_actors.remove(pos);
            return true;
        }
        if let Some(pos) = self.chart2d_ref_lines.iter().position(|a| a.id == id) {
            self.chart2d_ref_lines.remove(pos);
            return true;
        }
        false
    }

    /// Remove all spans and reference lines.
    pub fn chart2d_clear_overlays(&mut self) {
        self.chart2d_span_actors.clear();
        self.chart2d_ref_lines.clear();
    }

    /// Show or move the chart2d crosshair cursor to `(x_data, y_data)`.
    ///
    /// The crosshair is two thin lines (horizontal + vertical) rendered inside
    /// the plot-rect scissor so they are automatically clipped to the chart area.
    /// Call with `visible = false` to hide the cursor.
    pub fn chart2d_set_cursor(&mut self, x_data: f32, y_data: f32, visible: bool) {
        self.chart2d_cursor_visible = visible;
        self.chart2d_cursor_vertex_count = 0;
        if !visible {
            return;
        }
        // Snapshot state values to avoid borrow conflicts with self.
        let (x0, x1, y0, y1, pl, pr, pt, pb) = match self.chart2d_state.as_ref() {
            Some(s) => (s.x0, s.x1, s.y0, s.y1,
                        s.plot_left, s.plot_right, s.plot_top, s.plot_bottom),
            None => return,
        };
        let w = self.width;
        let h = self.height;
        // Clamp cursor to the data domain.
        let xc = x_data.clamp(x0.min(x1), x0.max(x1));
        let yc = y_data.clamp(y0.min(y1), y0.max(y1));
        // Build a lightweight state snapshot (strings/cache not needed for vertex gen).
        let tmp = Chart2DState {
            plot_left: pl, plot_right: pr, plot_top: pt, plot_bottom: pb,
            x0, x1, y0, y1,
            x_label: String::new(), y_label: String::new(), title: String::new(),
            x_tick_cache: Vec::new(), x_tick_step: 1.0, y_tick_step: 1.0,
            x_tick_format: String::new(), y_tick_format: String::new(),
            x_log_scale: false, y_log_scale: false,
        };
        let cc = [0.72_f32, 0.72_f32, 0.80_f32];
        // H line spans the full x domain at the cursor y.
        let mut verts = xy_to_thick_line_vertices(
            &[x0, x1], &[yc, yc], cc, 1.0, Some(&tmp), w, h);
        // V line spans the full y domain at the cursor x.
        verts.extend(xy_to_thick_line_vertices(
            &[xc, xc], &[y0, y1], cc, 1.0, Some(&tmp), w, h));
        if verts.is_empty() { return; }
        let bytes: &[u8] = bytemuck::cast_slice(&verts);
        self.chart2d_cursor_buf.upload(&self.device, &self.queue, bytes);
        self.chart2d_cursor_vertex_count = verts.len() as u32;
    }

    /// Build or update the chart2d legend.
    ///
    /// `entries` is a list of `(label, [r, g, b])` pairs in display order.
    /// `position` is one of `"top-right"`, `"top-left"`, `"bottom-right"`, `"bottom-left"`.
    /// Set `visible = false` to hide without clearing the entry list.
    pub fn chart2d_set_legend(
        &mut self,
        entries: Vec<(String, [f32; 3])>,
        position: &str,
        visible: bool,
    ) {
        self.chart2d_legend_visible = visible;
        self.chart2d_legend_text_labels.clear();
        self.chart2d_legend_swatch_count = 0;
        self.chart2d_legend_buf = GrowableBuffer::new(wgpu::BufferUsages::VERTEX);

        if !visible || entries.is_empty() {
            return;
        }

        let w = self.width  as f32;
        let h = self.height as f32;
        if w < 1.0 || h < 1.0 { return; }

        const SWATCH_W: f32 = 20.0;
        const SWATCH_H: f32 = 4.0;
        const GAP:      f32 = 6.0;
        const ENTRY_H:  f32 = 20.0;
        const MARGIN:   f32 = 12.0;
        const FONT_SZ:  f32 = 11.0;

        // Shape all label texts and measure the maximum rendered width.
        let mut shaped: Vec<Buffer> = entries.iter()
            .map(|(lbl, _)| build_label_buffer(&mut self.font_system, lbl, FONT_SZ))
            .collect();

        let max_text_w: f32 = shaped.iter()
            .map(|buf| buf.layout_runs()
                .flat_map(|r| r.glyphs.iter())
                .fold(0.0_f32, |acc, g| acc.max(g.x + g.w)))
            .fold(0.0_f32, f32::max);

        let block_w = SWATCH_W + GAP + max_text_w.max(1.0);
        let block_h = entries.len() as f32 * ENTRY_H;

        // Compute top-left anchor of the legend block.
        let (anchor_x, anchor_y) = match position {
            "top-left"     => (MARGIN, MARGIN),
            "top-right"    => (w - MARGIN - block_w, MARGIN),
            "bottom-left"  => (MARGIN, h - MARGIN - block_h),
            "bottom-right" => (w - MARGIN - block_w, h - MARGIN - block_h),
            _              => (w - MARGIN - block_w, MARGIN), // default: top-right
        };

        let ndc_x = |px: f32| -> f32 { px / w * 2.0 - 1.0 };
        let ndc_y = |py: f32| -> f32 { 1.0 - py / h * 2.0 };

        let mut swatch_verts: Vec<MeshVertex> = Vec::with_capacity(entries.len() * 6);

        for (i, ((_, color), glyph_buf)) in entries.iter().zip(shaped.drain(..)).enumerate() {
            let entry_top = anchor_y + i as f32 * ENTRY_H;
            let swatch_cy = entry_top + ENTRY_H * 0.5;
            let sx0 = anchor_x;
            let sx1 = anchor_x + SWATCH_W;
            let sy0 = swatch_cy - SWATCH_H * 0.5;
            let sy1 = swatch_cy + SWATCH_H * 0.5;

            let rgba = [color[0], color[1], color[2], 1.0];
            let (ax, ay) = (ndc_x(sx0), ndc_y(sy0));
            let (bx, by) = (ndc_x(sx1), ndc_y(sy0));
            let (cx, cy) = (ndc_x(sx1), ndc_y(sy1));
            let (dx, dy) = (ndc_x(sx0), ndc_y(sy1));
            swatch_verts.extend_from_slice(&[
                MeshVertex { position: [ax, ay, 0.0], color: rgba },
                MeshVertex { position: [bx, by, 0.0], color: rgba },
                MeshVertex { position: [cx, cy, 0.0], color: rgba },
                MeshVertex { position: [ax, ay, 0.0], color: rgba },
                MeshVertex { position: [cx, cy, 0.0], color: rgba },
                MeshVertex { position: [dx, dy, 0.0], color: rgba },
            ]);

            let text_h = FONT_SZ * 1.4;
            self.chart2d_legend_text_labels.push(Chart2DLegendLabel {
                glyph_buf,
                px: anchor_x + SWATCH_W + GAP,
                py: entry_top + (ENTRY_H - text_h) * 0.5,
            });
        }

        self.chart2d_legend_swatch_count = swatch_verts.len() as u32;
        if !swatch_verts.is_empty() {
            self.chart2d_legend_buf.upload(
                &self.device, &self.queue, bytemuck::cast_slice(&swatch_verts));
        }
    }

    // ── Line overlay actors ───────────────────────────────────────────────────

    pub fn add_line_actor(&mut self, vertices: &[LineVertex]) -> u32 {
        let id = self.next_line_actor_id;
        self.next_line_actor_id += 1;
        let (data_min, data_max) = line_vertex_bounds(vertices);
        let mut actor = LineActor {
            id,
            buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            vertex_count: vertices.len() as u32,
            visible: true,
            data_min,
            data_max,
        };
        if !vertices.is_empty() {
            actor.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(vertices));
        }
        self.line_actors.push(actor);
        id
    }

    pub fn update_line_actor_data(&mut self, id: u32, vertices: &[LineVertex]) -> bool {
        if let Some(a) = self.line_actors.iter_mut().find(|a| a.id == id) {
            a.vertex_count = vertices.len() as u32;
            let (data_min, data_max) = line_vertex_bounds(vertices);
            a.data_min = data_min;
            a.data_max = data_max;
            if !vertices.is_empty() {
                a.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(vertices));
            }
            true
        } else {
            false
        }
    }

    pub fn remove_line_actor(&mut self, id: u32) -> bool {
        if let Some(pos) = self.line_actors.iter().position(|a| a.id == id) {
            self.line_actors.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn set_line_actor_visibility(&mut self, id: u32, visible: bool) -> bool {
        if let Some(a) = self.line_actors.iter_mut().find(|a| a.id == id) {
            a.visible = visible;
            true
        } else {
            false
        }
    }

    pub fn clear_line_actors(&mut self) {
        self.line_actors.clear();
    }

    // ── Rendering modes ───────────────────────────────────────────────────────

    /// Set the point rendering style: 0 = circle (soft), 1 = square, 2 = gaussian.
    pub fn set_point_style(&mut self, style: u32) {
        self.point_style = style.min(2);
    }

    /// Set the LOD divisor. When > 1 each actor draws only `count / lod_factor`
    /// instances, giving fast interaction at the cost of apparent density.
    pub fn set_lod_factor(&mut self, factor: u32) {
        self.lod_factor = factor.max(1);
    }

    // ── Visual appearance ─────────────────────────────────────────────────────

    pub fn set_grid_visible(&mut self, visible: bool) {
        self.grid_visible = visible;
    }

    pub fn set_grid_planes(&mut self, major: bool, minor: bool) {
        self.major_grid_planes = major;
        self.minor_grid_planes = minor;
        self.rebuild_grid_geometry();
    }

    pub fn set_background_color(&mut self, r: f64, g: f64, b: f64) {
        self.bg_color = [r, g, b, 1.0];
    }

    pub fn set_axis_labels(&mut self, x: String, y: String, z: String) {
        self.axis_label_texts = [x, y, z];
        self.rebuild_grid_geometry();
    }

    // ── Orientation axes ──────────────────────────────────────────────────────

    pub fn set_orientation_axes_visible(&mut self, visible: bool) {
        self.axes_visible = visible;
    }

    fn update_axes_buf(&mut self) {
        if !self.axes_visible { return; }
        // Corner center in NDC (bottom-left, accounting for wgpu Y-up NDC).
        let (cx, cy) = (-0.82_f32, -0.82_f32);
        let scale = 0.13_f32;
        let vm = self.camera.view_matrix();
        // World-axis directions in camera space (X=right, Y=up in NDC).
        let axes: [([f32; 3], [f32; 3]); 3] = [
            ([1., 0., 0.], [0.95, 0.30, 0.30]),  // X — red
            ([0., 1., 0.], [0.30, 0.90, 0.30]),  // Y — green
            ([0., 0., 1.], [0.40, 0.60, 1.00]),  // Z — blue
        ];
        let mut verts = [LineVertex { position: [0.; 3], color: [0.; 3] }; 6];
        for (i, (world_axis, color)) in axes.iter().enumerate() {
            // transform_vector3 applies only the rotation part of the view matrix.
            let d = vm.transform_vector3(Vec3::from(*world_axis));
            verts[i * 2]     = LineVertex { position: [cx, cy, 0.], color: *color };
            verts[i * 2 + 1] = LineVertex { position: [cx + d.x * scale, cy + d.y * scale, 0.], color: *color };
        }
        self.axes_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
    }

    pub fn clear_grid(&mut self) {
        self.line_count = 0;
        self.cached_labels.clear();
        self.last_grid_min = None;
        self.last_grid_max = None;
        self.last_data_min = None;
        self.last_data_max = None;
    }

    pub fn set_tick_override(&mut self, x: Option<usize>, y: Option<usize>, z: Option<usize>) {
        self.tick_override = [x, y, z];
        self.rebuild_grid_geometry();
    }

    pub fn set_axis_visible(&mut self, x: bool, y: bool, z: bool) {
        self.axis_visible = [x, y, z];
        self.rebuild_grid_geometry();
    }

    pub fn set_grid(&mut self, data_min: Vec3, data_max: Vec3, nice_min: Vec3, nice_max: Vec3) {
        // Skip rebuild when both rounded bounds AND raw data bounds are unchanged.
        // Raw bounds drive flat-axis detection in build_grid(), so a change in
        // data_min/data_max can alter which axes show ticks even when nice bounds
        // are identical (e.g. adding a flat Z axis to an existing XY scatter).
        if self.last_grid_min == Some(nice_min)
            && self.last_grid_max == Some(nice_max)
            && self.last_data_min == Some(data_min)
            && self.last_data_max == Some(data_max)
        {
            return;
        }
        self.last_grid_min = Some(nice_min);
        self.last_grid_max = Some(nice_max);
        self.last_data_min = Some(data_min);
        self.last_data_max = Some(data_max);
        // Reset face bits so the next render rebuilds with the live camera position.
        self.grid_face_bits = 0xFF;
        self.rebuild_grid_geometry();
    }

    /// Rebuild grid line geometry and pre-shaped label buffers using the current
    /// camera position.  Called from set_grid() and from render() whenever the
    /// camera crosses an axis midplane.
    fn rebuild_grid_geometry(&mut self) {
        let (dmin, dmax, nmin, nmax) = match (
            self.last_data_min, self.last_data_max,
            self.last_grid_min, self.last_grid_max,
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => return,
        };

        let eye = self.camera.position();
        let axis_texts = self.axis_label_texts.clone();
        // Pass the independent ortho extents when set_parallel_scale is active so
        // build_grid can size tick marks relative to the visible range of each axis.
        let ortho_scale = if self.camera.ortho_half_w > 0.0 && self.camera.ortho_half_h > 0.0 {
            Some((self.camera.ortho_half_w, self.camera.ortho_half_h))
        } else {
            None
        };
        let geo = build_grid(dmin, dmax, nmin, nmax,
                             self.tick_override, self.axis_visible,
                             eye, &axis_texts,
                             self.major_grid_planes, self.minor_grid_planes,
                             ortho_scale);

        self.line_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&geo.vertices));
        self.line_count = geo.vertices.len() as u32;

        // Pre-shape label glyphs.  Axis title labels use a larger font.
        self.cached_labels.clear();
        for anchor in geo.labels {
            let (size, line_h, buf_w, buf_h) = if anchor.is_axis_title {
                (14.0_f32, 18.0_f32, 200.0_f32, 24.0_f32)
            } else {
                (11.0_f32, 14.0_f32, 120.0_f32, 20.0_f32)
            };
            let mut buf = Buffer::new(&mut self.font_system, Metrics::new(size, line_h));
            buf.set_size(&mut self.font_system, Some(buf_w), Some(buf_h));
            buf.set_text(
                &mut self.font_system,
                &anchor.text,
                Attrs::new().family(Family::SansSerif),
                Shaping::Basic,
            );
            buf.shape_until_scroll(&mut self.font_system, false);
            self.cached_labels.push(CachedLabel {
                glyph_buf:    buf,
                world_pos:    anchor.world_pos,
                tick_pos:     anchor.tick_pos,
                is_axis_title: anchor.is_axis_title,
            });
        }
    }

    /// Called at the top of each frame.  If the camera has crossed an axis
    /// midplane since the last rebuild, regenerate the grid geometry so tick
    /// marks and labels always sit on the correct outward-facing face.
    fn update_grid_for_camera(&mut self) {
        if let (Some(nmin), Some(nmax)) = (self.last_grid_min, self.last_grid_max) {
            let center = (nmin + nmax) * 0.5;
            let eye    = self.camera.position();
            let bits   = face_bits(eye, center);
            if bits != self.grid_face_bits {
                self.grid_face_bits = bits;
                self.rebuild_grid_geometry();
            }
        }
    }

    pub fn fit_camera(&mut self, center: Vec3, radius: f32) {
        let aspect = self.width as f32 / self.height.max(1) as f32;
        self.camera = Camera::fit(center, radius, aspect);
        self.fit_center = center;
        self.fit_radius = radius;
    }

    // ── Resize ────────────────────────────────────────────────────────────────

    pub fn resize(&mut self, width: u32, height: u32) {
        let (w, h) = (width.max(1), height.max(1));
        // Skip if nothing changed
        if w == self.width && h == self.height {
            return;
        }
        self.width = w;
        self.height = h;
        if let RenderSurface::Windowed { ref mut surface_config, ref surface } = self.render_surface {
            surface_config.width = w;
            surface_config.height = h;
            surface.configure(&self.device, surface_config);
        }
        let (dt, dv) = make_depth_texture(&self.device, w, h);
        self.depth_texture = dt;
        self.depth_view = dv;
        self.camera.aspect = w as f32 / h as f32;
        self.screenshot_cache = None;  // dimensions changed — invalidate cached resources
        // Pixel coordinates have changed — all cached screen projections are stale.
        for actor in &mut self.actors {
            actor.pick_cache = None;
        }
    }

    // ── Camera controls ───────────────────────────────────────────────────────

    pub fn mouse_drag(&mut self, dx: f32, dy: f32, button: u8) {
        match button {
            1 => self.camera.orbit(glam::Vec2::new(dx, dy)),
            2 => self.camera.pan(glam::Vec2::new(dx, dy)),
            _ => {}
        }
    }

    pub fn scroll(&mut self, delta: f32) {
        self.camera.zoom(delta);
    }

    /// Resets to the last fitted view (center + radius from the most recent `fit_camera` call).
    pub fn reset_camera(&mut self) {
        let parallel = self.camera.parallel;
        let aspect = self.width as f32 / self.height.max(1) as f32;
        self.camera = Camera::fit(self.fit_center, self.fit_radius, aspect);
        self.camera.parallel = parallel;
    }

    pub fn get_camera_state(&self) -> CameraState {
        self.camera.state()
    }

    pub fn set_camera_state(&mut self, state: CameraState) {
        self.camera.apply_state(state);
    }

    pub fn set_parallel_projection(&mut self, on: bool) {
        self.camera.parallel = on;
    }

    /// Set independent half-extents for the orthographic projection.
    /// When called, the camera shows exactly [-half_w, half_w] × [-half_h, half_h]
    /// centred on `camera.target`, regardless of the data aspect ratio.
    /// Implicitly enables parallel projection.
    /// Cleared by any `fit_camera` call (e.g. on `add_points` with unfitted camera).
    pub fn set_parallel_scale(&mut self, half_w: f32, half_h: f32) {
        self.camera.ortho_half_w = half_w.max(1e-6);
        self.camera.ortho_half_h = half_h.max(1e-6);
        self.camera.parallel = true;
    }

    /// Reorient camera to a preset view direction, preserving target and distance.
    /// `yaw` and `pitch` are the spherical angles for the desired look direction.
    pub fn set_view_direction(&mut self, yaw: f32, pitch: f32) {
        self.camera.yaw = yaw;
        self.camera.pitch = pitch.clamp(-1.55, 1.55);
    }

    /// Fit the camera to explicit world-space bounds [min_x,min_y,min_z, max_x,max_y,max_z].
    pub fn fit_to_bounds(&mut self, bounds: [f32; 6]) {
        let bmin = glam::Vec3::from_slice(&bounds[0..3]);
        let bmax = glam::Vec3::from_slice(&bounds[3..6]);
        let center = (bmin + bmax) * 0.5;
        let radius = (bmax - bmin).length() * 0.5;
        self.fit_camera(center, radius.max(1e-6));
    }

    // ── Export ────────────────────────────────────────────────────────────────

    /// Render the current scene to an offscreen texture and return raw RGBA bytes.
    ///
    /// Returns `(width, height, rgba_bytes)`. Bytes are always RGBA regardless of
    /// the internal surface format (BGRA is swapped before returning).
    pub fn screenshot(&mut self) -> Result<(u32, u32, Vec<u8>), Box<dyn std::error::Error + Send + Sync>> {
        let (w, h) = (self.width, self.height);

        // Readback row stride must satisfy COPY_BYTES_PER_ROW_ALIGNMENT.
        let bytes_per_px = 4u32;
        let unpadded_row = w * bytes_per_px;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) & !(align - 1);

        // Reuse the offscreen color texture + readback buffer if dimensions match.
        if self.screenshot_cache.as_ref().map_or(true, |c| c.w != w || c.h != h) {
            let color_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("screenshot_color"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("screenshot_readback"),
                size: (padded_row * h) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.screenshot_cache = Some(ScreenshotCache { w, h, color_tex, color_view, readback, padded_row });
        }

        // Take the cache out of self so we can freely call self.* methods alongside
        // the borrowed cache fields (color_view, color_tex, readback).
        let cache = self.screenshot_cache.take().unwrap();

        // Update uniforms and text, exactly as in render().
        self.update_grid_for_camera();
        let view_proj = self.camera.view_proj();
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            screen_size: [w as f32, h as f32],
            style: self.point_style,
            _pad: 0.0,
        };
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        self.viewport.update(&self.queue, Resolution { width: w, height: h });
        self.prepare_text_labels(view_proj);
        self.update_axes_buf();

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("screenshot") }
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("screenshot_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &cache.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.bg_color[0], g: self.bg_color[1],
                            b: self.bg_color[2], a: self.bg_color[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            if self.chart2d_state.is_some() {
                pass.set_pipeline(&self.line_pipeline_nodepth);

                let (sc_x, sc_y, sc_w, sc_h) = if let Some(ref state) = self.chart2d_state {
                    let sx = (state.plot_left * self.width as f32).floor() as u32;
                    let sy = (state.plot_top * self.height as f32).floor() as u32;
                    let sw = ((state.plot_right - state.plot_left) * self.width as f32).ceil() as u32 + 1;
                    let sh = ((state.plot_bottom - state.plot_top) * self.height as f32).ceil() as u32 + 1;
                    (
                        sx.min(self.width.saturating_sub(1)),
                        sy.min(self.height.saturating_sub(1)),
                        sw.min(self.width.saturating_sub(sx)),
                        sh.min(self.height.saturating_sub(sy)),
                    )
                } else {
                    (0, 0, self.width, self.height)
                };

                pass.set_bind_group(0, &self.overlay_bind_group, &[]);
                if self.chart2d_axis_count > 0 {
                    if let Some(slice) = self.chart2d_axis_buf.slice() {
                        pass.set_vertex_buffer(0, slice);
                        pass.draw(0..self.chart2d_axis_count, 0..1);
                    }
                }

                // Spans and ref lines use chart-space MeshVertex (chart2d_line_pipeline).
                pass.set_pipeline(&self.chart2d_line_pipeline);
                pass.set_bind_group(0, &self.chart2d_bind_group, &[]);
                pass.set_scissor_rect(sc_x, sc_y, sc_w, sc_h);
                // Reference spans (axhspan / axvspan) — drawn first, behind everything.
                for span in &self.chart2d_span_actors {
                    if span.vertex_count > 0 {
                        if let Some(slice) = span.buf.slice() {
                            pass.set_vertex_buffer(0, slice);
                            pass.draw(0..span.vertex_count, 0..1);
                        }
                    }
                }
                // Reference lines (axhline / axvline) — above spans, below data.
                for rl in &self.chart2d_ref_lines {
                    if rl.vertex_count > 0 {
                        if let Some(slice) = rl.buf.slice() {
                            pass.set_vertex_buffer(0, slice);
                            pass.draw(0..rl.vertex_count, 0..1);
                        }
                    }
                }
                // Data lines — GPU-side miter-join shader (ThickLineVert).
                pass.set_pipeline(&self.chart2d_thick_pipeline);
                pass.set_bind_group(0, &self.chart2d_bind_group, &[]);
                for la in &self.chart2d_line_actors {
                    if la.visible && la.vertex_count > 0 {
                        if let Some(slice) = la.buf.slice() {
                            pass.set_vertex_buffer(0, slice);
                            pass.draw(0..la.vertex_count, 0..1);
                        }
                    }
                }
                // Markers and cursor use chart-space MeshVertex (chart2d_line_pipeline).
                pass.set_pipeline(&self.chart2d_line_pipeline);
                pass.set_bind_group(0, &self.chart2d_bind_group, &[]);
                for la in &self.chart2d_line_actors {
                    if la.visible && la.marker_vertex_count > 0 {
                        if let Some(slice) = la.marker_buf.slice() {
                            pass.set_vertex_buffer(0, slice);
                            pass.draw(0..la.marker_vertex_count, 0..1);
                        }
                    }
                }
                // Crosshair cursor — drawn inside plot scissor, on top of data lines.
                if self.chart2d_cursor_visible && self.chart2d_cursor_vertex_count > 0 {
                    if let Some(slice) = self.chart2d_cursor_buf.slice() {
                        pass.set_vertex_buffer(0, slice);
                        pass.draw(0..self.chart2d_cursor_vertex_count, 0..1);
                    }
                }
                pass.set_scissor_rect(0, 0, self.width, self.height);
            } else {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                if self.grid_visible {
                    if let Some(slice) = self.line_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..self.line_count, 0..1);
                    }
                }
                for la in &self.line_actors {
                    if la.visible && la.vertex_count > 0 {
                        if let Some(slice) = la.buf.slice() {
                        pass.set_vertex_buffer(0, slice);
                        pass.draw(0..la.vertex_count, 0..1);
                        }
                    }
                }
                pass.set_pipeline(&self.point_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                for actor in &self.actors {
                    if actor.visible && actor.count > 0 {
                        if let Some(slice) = actor.buf.slice() {
                        pass.set_vertex_buffer(0, slice);
                        pass.draw(0..6, 0..actor.count);
                        }
                    }
                }

            // ── Mesh actors (convex hulls, ellipsoids) ─────────────────────────
            draw_mesh_actors(&self.mesh_actors, &self.mesh_pipeline_wireframe,
                             &self.mesh_pipeline_opaque, &self.mesh_pipeline_transparent,
                             &self.uniform_bind_group, view_proj, &mut pass);

            }

            pass.set_pipeline(&self.line_pipeline);
            pass.set_bind_group(0, &self.overlay_bind_group, &[]);
            if self.scalar_bar_visible && self.scalar_bar_line_count > 0 {
                if let Some(slice) = self.scalar_bar_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..self.scalar_bar_line_count, 0..1);
                }
            }
            // Legend overlay (screen-space, like scalar bar)
            if self.legend_visible && self.legend_line_count > 0 {
                if let Some(slice) = self.legend_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..self.legend_line_count, 0..1);
                }
            }
            // Selection rect intentionally excluded from screenshots.
            if self.axes_visible {
                pass.set_pipeline(&self.line_pipeline_nodepth);
                pass.set_bind_group(0, &self.overlay_bind_group, &[]);
                if let Some(slice) = self.axes_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..6, 0..1);
                }
            }
            self.text_renderer.render(&self.text_atlas, &self.viewport, &mut pass).ok();
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &cache.color_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &cache.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(cache.padded_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // Wait for GPU and read bytes back to CPU.
        let (tx, rx) = std::sync::mpsc::channel();
        cache.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let raw = cache.readback.slice(..).get_mapped_range();
        let is_bgra = matches!(
            self.surface_format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        let bytes_per_px = 4usize;
        // Fast path: RGBA format with no row padding — single memcpy.
        let pixels = if !is_bgra && cache.padded_row == w * bytes_per_px as u32 {
            raw.to_vec()
        } else {
            let mut pixels = Vec::with_capacity(w as usize * h as usize * bytes_per_px);
            for row in 0..h as usize {
                let start = row * cache.padded_row as usize;
                let row_bytes = &raw[start..start + w as usize * bytes_per_px];
                if is_bgra {
                    for px in row_bytes.chunks_exact(4) {
                        pixels.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
                    }
                } else {
                    pixels.extend_from_slice(row_bytes);
                }
            }
            pixels
        };
        drop(raw);
        cache.readback.unmap();

        // Return the cache so future calls reuse the GPU resources.
        self.screenshot_cache = Some(cache);
        Ok((w, h, pixels))
    }

    /// Render one frame offscreen and return raw RGBA bytes (width × height × 4).
    /// Only valid when the renderer was created with `new_offscreen()`.
    /// For windowed renderers use `screenshot()` instead.
    pub fn render_offscreen(&mut self) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let (_w, _h, pixels) = self.screenshot()?;
        Ok(pixels)
    }

    // ── Render ────────────────────────────────────────────────────────────────

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let (output, view) = match &self.render_surface {
            RenderSurface::Windowed { surface, .. } => {
                let output = surface.get_current_texture()?;
                let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                (Some(output), view)
            }
            RenderSurface::Offscreen => {
                // Offscreen mode: callers should use render_offscreen() instead.
                // Silently succeed without presenting anything.
                return Ok(());
            }
        };

        // Rebuild grid geometry if the camera has crossed an axis midplane since
        // the last frame.  Skipped in chart2d mode (no world-space grid).
        if self.chart2d_state.is_none() {
            self.update_grid_for_camera();
        }

        let view_proj = self.camera.view_proj();
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            screen_size: [self.width as f32, self.height as f32],
            style: self.point_style,
            _pad: 0.0,
        };
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        self.viewport.update(&self.queue, Resolution { width: self.width, height: self.height });

        self.prepare_text_labels(view_proj);
        self.update_axes_buf();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.bg_color[0], g: self.bg_color[1],
                            b: self.bg_color[2], a: self.bg_color[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            if self.chart2d_state.is_some() {
                // ── Chart2D path ───────────────────────────────────────────────
                pass.set_pipeline(&self.line_pipeline_nodepth);

                // Compute scissor rect for plot area.
                let (sc_x, sc_y, sc_w, sc_h) = if let Some(ref state) = self.chart2d_state {
                    let sx = (state.plot_left   * self.width  as f32).floor() as u32;
                    let sy = (state.plot_top     * self.height as f32).floor() as u32;
                    let sw = ((state.plot_right  - state.plot_left)   * self.width  as f32).ceil() as u32 + 1;
                    let sh = ((state.plot_bottom - state.plot_top)    * self.height as f32).ceil() as u32 + 1;
                    (sx.min(self.width.saturating_sub(1)),
                     sy.min(self.height.saturating_sub(1)),
                     sw.min(self.width.saturating_sub(sx)),
                     sh.min(self.height.saturating_sub(sy)))
                } else { (0, 0, self.width, self.height) };

                // Frame + grid lines + tick marks (NDC space, identity bind group).
                pass.set_bind_group(0, &self.overlay_bind_group, &[]);
                if self.chart2d_axis_count > 0 {
                    if let Some(slice) = self.chart2d_axis_buf.slice() {
                        pass.set_vertex_buffer(0, slice);
                        pass.draw(0..self.chart2d_axis_count, 0..1);
                    }
                }

                // Spans and ref lines use chart-space MeshVertex (chart2d_line_pipeline).
                pass.set_pipeline(&self.chart2d_line_pipeline);
                pass.set_bind_group(0, &self.chart2d_bind_group, &[]);
                pass.set_scissor_rect(sc_x, sc_y, sc_w, sc_h);
                // Reference spans (axhspan / axvspan) — drawn first, behind everything.
                for span in &self.chart2d_span_actors {
                    if span.vertex_count > 0 {
                        if let Some(slice) = span.buf.slice() {
                            pass.set_vertex_buffer(0, slice);
                            pass.draw(0..span.vertex_count, 0..1);
                        }
                    }
                }
                // Reference lines (axhline / axvline) — above spans, below data.
                for rl in &self.chart2d_ref_lines {
                    if rl.vertex_count > 0 {
                        if let Some(slice) = rl.buf.slice() {
                            pass.set_vertex_buffer(0, slice);
                            pass.draw(0..rl.vertex_count, 0..1);
                        }
                    }
                }
                // Data lines — GPU-side miter-join shader (ThickLineVert).
                pass.set_pipeline(&self.chart2d_thick_pipeline);
                pass.set_bind_group(0, &self.chart2d_bind_group, &[]);
                for la in &self.chart2d_line_actors {
                    if la.visible && la.vertex_count > 0 {
                        if let Some(slice) = la.buf.slice() {
                            pass.set_vertex_buffer(0, slice);
                            pass.draw(0..la.vertex_count, 0..1);
                        }
                    }
                }
                // Markers and cursor use chart-space MeshVertex (chart2d_line_pipeline).
                pass.set_pipeline(&self.chart2d_line_pipeline);
                pass.set_bind_group(0, &self.chart2d_bind_group, &[]);
                for la in &self.chart2d_line_actors {
                    if la.visible && la.marker_vertex_count > 0 {
                        if let Some(slice) = la.marker_buf.slice() {
                            pass.set_vertex_buffer(0, slice);
                            pass.draw(0..la.marker_vertex_count, 0..1);
                        }
                    }
                }
                // Crosshair cursor — drawn inside plot scissor, on top of data lines.
                if self.chart2d_cursor_visible && self.chart2d_cursor_vertex_count > 0 {
                    if let Some(slice) = self.chart2d_cursor_buf.slice() {
                        pass.set_vertex_buffer(0, slice);
                        pass.draw(0..self.chart2d_cursor_vertex_count, 0..1);
                    }
                }
                pass.set_scissor_rect(0, 0, self.width, self.height);
            } else {
                // ── Normal 3D path ─────────────────────────────────────────────
                pass.set_pipeline(&self.line_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                if self.grid_visible {
                    if let Some(slice) = self.line_buf.slice() {
                        pass.set_vertex_buffer(0, slice);
                        pass.draw(0..self.line_count, 0..1);
                    }
                }
                for la in &self.line_actors {
                    if la.visible && la.vertex_count > 0 {
                        if let Some(slice) = la.buf.slice() {
                            pass.set_vertex_buffer(0, slice);
                            pass.draw(0..la.vertex_count, 0..1);
                        }
                    }
                }
                pass.set_pipeline(&self.point_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                let lod = self.lod_factor.max(1);
                for actor in &self.actors {
                    if actor.visible && actor.count > 0 {
                        if let Some(slice) = actor.buf.slice() {
                            pass.set_vertex_buffer(0, slice);
                            pass.draw(0..6, 0..(actor.count / lod).max(1));
                        }
                    }
                }
                draw_mesh_actors(&self.mesh_actors, &self.mesh_pipeline_wireframe,
                                 &self.mesh_pipeline_opaque, &self.mesh_pipeline_transparent,
                                 &self.uniform_bind_group, view_proj, &mut pass);
            }

            // ── Screen-space overlays (always drawn regardless of mode) ────────
            // Scalar bar: screen-space overlay drawn with identity view_proj.
            pass.set_pipeline(&self.line_pipeline);
            pass.set_bind_group(0, &self.overlay_bind_group, &[]);
            if self.scalar_bar_visible && self.scalar_bar_line_count > 0 {
                if let Some(slice) = self.scalar_bar_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..self.scalar_bar_line_count, 0..1);
                }
            }
            // Legend overlay (screen-space, like scalar bar)
            if self.legend_visible && self.legend_line_count > 0 {
                if let Some(slice) = self.legend_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..self.legend_line_count, 0..1);
                }
            }
            // Chart2D legend swatches (colored rectangles; MeshVertex in NDC via identity).
            if self.chart2d_legend_visible && self.chart2d_legend_swatch_count > 0 {
                pass.set_pipeline(&self.chart2d_line_pipeline);
                pass.set_bind_group(0, &self.overlay_bind_group, &[]);
                if let Some(slice) = self.chart2d_legend_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..self.chart2d_legend_swatch_count, 0..1);
                }
            }
            // Selection rectangle overlay (same pipeline, identity view_proj).
            if self.sel_rect_visible {
                if let Some(slice) = self.sel_rect_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..8, 0..1);
                }
            }
            // Lasso overlay (open or closed polyline, screen-space).
            if self.lasso_visible && self.lasso_vert_count > 0 {
                if let Some(slice) = self.lasso_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..self.lasso_vert_count, 0..1);
                }
            }
            // Orientation axes: 3 colored axis lines in the bottom-left corner.
            if self.axes_visible {
                pass.set_pipeline(&self.line_pipeline_nodepth);
                pass.set_bind_group(0, &self.overlay_bind_group, &[]);
                if let Some(slice) = self.axes_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..6, 0..1);
                }
            }

            self.text_renderer
                .render(&self.text_atlas, &self.viewport, &mut pass)
                .ok();
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        if let Some(output) = output {
            output.present();
        }

        // Trim glyph atlas every 120 frames — infrequent enough to not defeat caching
        self.atlas_trim_counter += 1;
        if self.atlas_trim_counter >= 120 {
            self.text_atlas.trim();
            self.atlas_trim_counter = 0;
        }

        Ok(())
    }

    // ── User label API ────────────────────────────────────────────────────────

    pub fn add_user_label(
        &mut self,
        x: f32, y: f32, z: f32,
        text: &str,
        color: [f32; 4],
        size: f32,
        anchor: u8,
    ) -> u64 {
        let id = self.next_user_label_id;
        self.next_user_label_id += 1;
        let glyph_buf = build_label_buffer(&mut self.font_system, text, size);
        self.user_labels.push(UserLabel {
            id,
            text: text.to_string(),
            glyph_buf,
            world_pos: Vec3::new(x, y, z),
            color,
            size,
            anchor: LabelAnchor::from_u8(anchor),
            visible: true,
        });
        id
    }

    pub fn update_user_label(
        &mut self,
        id: u64,
        pos: Option<[f32; 3]>,
        text: Option<&str>,
        color: Option<[f32; 4]>,
        size: Option<f32>,
        anchor: Option<u8>,
    ) {
        let Some(label) = self.user_labels.iter_mut().find(|l| l.id == id) else { return };
        if let Some(p) = pos   { label.world_pos = Vec3::new(p[0], p[1], p[2]); }
        if let Some(c) = color { label.color = c; }
        if let Some(a) = anchor { label.anchor = LabelAnchor::from_u8(a); }
        // Rebuild the glyph buffer when text or size changes.
        let new_text = text.map(|t| t.to_string());
        let new_size = size.filter(|&s| (s - label.size).abs() > 0.01);
        if new_text.is_some() || new_size.is_some() {
            if let Some(s) = new_size { label.size = s; }
            if let Some(t) = new_text { label.text = t; }
            // Borrow fields individually to satisfy the borrow checker.
            let buf = build_label_buffer(&mut self.font_system, &label.text.clone(), label.size);
            label.glyph_buf = buf;
        }
    }

    pub fn remove_user_label(&mut self, id: u64) {
        self.user_labels.retain(|l| l.id != id);
    }

    pub fn set_user_label_visible(&mut self, id: u64, visible: bool) {
        if let Some(label) = self.user_labels.iter_mut().find(|l| l.id == id) {
            label.visible = visible;
        }
    }

    pub fn clear_user_labels(&mut self) {
        self.user_labels.clear();
    }

    // ── Mesh actor API ────────────────────────────────────────────────────────

    pub fn add_mesh_actor(
        &mut self,
        vertices: &[[f32; 3]],
        indices: &[[u32; 3]],
        color: [f32; 4],
        wireframe: bool,
    ) -> u64 {
        let id = self.next_mesh_actor_id;
        self.next_mesh_actor_id += 1;

        let (data_min, data_max) = mesh_bounds(vertices);
        let mesh_verts: Vec<MeshVertex> = vertices.iter()
            .map(|&p| MeshVertex { position: p, color })
            .collect();

        let index_bytes: Vec<u8>;
        let index_count: u32;
        if wireframe {
            let wire = triangles_to_wireframe_indices(indices);
            index_count = wire.len() as u32;
            index_bytes = bytemuck::cast_slice::<u32, u8>(&wire).to_vec();
        } else {
            let flat: Vec<u32> = indices.iter().flat_map(|&[a, b, c]| [a, b, c]).collect();
            index_count = flat.len() as u32;
            index_bytes = bytemuck::cast_slice::<u32, u8>(&flat).to_vec();
        }

        let mut vbuf = GrowableBuffer::new(wgpu::BufferUsages::VERTEX);
        let mut ibuf = GrowableBuffer::new(wgpu::BufferUsages::INDEX);
        vbuf.upload(&self.device, &self.queue, bytemuck::cast_slice(&mesh_verts));
        ibuf.upload(&self.device, &self.queue, &index_bytes);

        self.mesh_actors.push(MeshActor {
            id, vbuf, ibuf, index_count, visible: true, wireframe, color, data_min, data_max,
            positions: vertices.to_vec(),
            triangle_indices: indices.to_vec(),
        });
        id
    }

    pub fn update_mesh_actor(
        &mut self,
        id: u64,
        vertices: &[[f32; 3]],
        indices: &[[u32; 3]],
        color: [f32; 4],
        wireframe: bool,
    ) -> bool {
        let actor = match self.mesh_actors.iter_mut().find(|a| a.id == id) {
            Some(a) => a,
            None => return false,
        };
        let (data_min, data_max) = mesh_bounds(vertices);
        let mesh_verts: Vec<MeshVertex> = vertices.iter()
            .map(|&p| MeshVertex { position: p, color })
            .collect();
        let index_bytes: Vec<u8>;
        let index_count: u32;
        if wireframe {
            let wire = triangles_to_wireframe_indices(indices);
            index_count = wire.len() as u32;
            index_bytes = bytemuck::cast_slice::<u32, u8>(&wire).to_vec();
        } else {
            let flat: Vec<u32> = indices.iter().flat_map(|&[a, b, c]| [a, b, c]).collect();
            index_count = flat.len() as u32;
            index_bytes = bytemuck::cast_slice::<u32, u8>(&flat).to_vec();
        }
        actor.data_min = data_min;
        actor.data_max = data_max;
        actor.color = color;
        actor.wireframe = wireframe;
        actor.index_count = index_count;
        actor.positions = vertices.to_vec();
        actor.triangle_indices = indices.to_vec();
        actor.vbuf.upload(&self.device, &self.queue, bytemuck::cast_slice(&mesh_verts));
        actor.ibuf.upload(&self.device, &self.queue, &index_bytes);
        true
    }

    /// Update only color and wireframe mode without retransferring geometry from Python.
    ///
    /// Uses the positions and triangle indices stored at add/update time to re-bake
    /// the vertex buffer (color change) and optionally rebuild the index buffer
    /// (solid→wireframe or wireframe→solid toggle).  Skips any re-upload when the
    /// respective field did not change.
    pub fn update_mesh_style_actor(&mut self, id: u64, color: [f32; 4], wireframe: bool) -> bool {
        let actor = match self.mesh_actors.iter_mut().find(|a| a.id == id) {
            Some(a) => a,
            None => return false,
        };
        let color_changed = actor.color != color;
        let wireframe_changed = actor.wireframe != wireframe;
        actor.color = color;
        actor.wireframe = wireframe;
        if color_changed {
            let mesh_verts: Vec<MeshVertex> = actor.positions.iter()
                .map(|&p| MeshVertex { position: p, color })
                .collect();
            actor.vbuf.upload(&self.device, &self.queue, bytemuck::cast_slice(&mesh_verts));
        }
        if wireframe_changed {
            let idxs = actor.triangle_indices.clone();
            let (index_bytes, index_count) = if wireframe {
                let wire = triangles_to_wireframe_indices(&idxs);
                let cnt = wire.len() as u32;
                (bytemuck::cast_slice::<u32, u8>(&wire).to_vec(), cnt)
            } else {
                let flat: Vec<u32> = idxs.iter().flat_map(|&[a, b, c]| [a, b, c]).collect();
                let cnt = flat.len() as u32;
                (bytemuck::cast_slice::<u32, u8>(&flat).to_vec(), cnt)
            };
            actor.index_count = index_count;
            actor.ibuf.upload(&self.device, &self.queue, &index_bytes);
        }
        true
    }

    pub fn remove_mesh_actor(&mut self, id: u64) -> bool {
        if let Some(pos) = self.mesh_actors.iter().position(|a| a.id == id) {
            self.mesh_actors.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn set_mesh_actor_visibility(&mut self, id: u64, visible: bool) -> bool {
        if let Some(a) = self.mesh_actors.iter_mut().find(|a| a.id == id) {
            a.visible = visible;
            true
        } else {
            false
        }
    }

    pub fn clear_mesh_actors(&mut self) {
        self.mesh_actors.clear();
    }

    fn prepare_text_labels(&mut self, vp: glam::Mat4) {
        // No early-return on empty: we must always call prepare() so glyphon can
        // flush stale vertices from the previous frame (e.g. after clear_grid).

        // Build TextArea list by projecting pre-shaped buffers to current screen positions
        let (w, h) = (self.width, self.height);

        // Minimum screen-space gap (pixels) between a tick mark and its label.
        // When the push direction collapses into the depth axis (e.g. orthographic
        // aligned views), the gap is near-zero and the label is suppressed instead
        // of piling up on the grid.
        const MIN_PUSH_PX: f32 = 16.0;

        let mut text_areas: Vec<TextArea> = Vec::with_capacity(self.cached_labels.len());
        if self.grid_visible {
        for label in &self.cached_labels {
            let clip = vp * label.world_pos.extend(1.0);
            if clip.w <= 0.0 { continue; }
            let ndc = clip.truncate() / clip.w;
            if ndc.x < -1.1 || ndc.x > 1.1 || ndc.y < -1.1 || ndc.y > 1.1 { continue; }

            let mut sx = (ndc.x + 1.0) * 0.5 * w as f32;
            let mut sy = (1.0 - ndc.y) * 0.5 * h as f32;

            if label.is_axis_title {
                if self.axis_visible[2] {
                    // 3D mode: titles sit slightly farther from the cube so they
                    // clear the silhouette across viewing angles.
                    let cx = w as f32 * 0.5;
                    let cy = h as f32 * 0.5;
                    let dx = sx - cx;
                    let dy = sy - cy;
                    let len = (dx * dx + dy * dy).sqrt().max(1.0);
                    sx += dx / len * 24.0;
                    sy += dy / len * 24.0;
                }
            } else {
                // Tick labels: push away from their tick mark; suppress when depth-aligned.
                let tick_clip = vp * label.tick_pos.extend(1.0);
                if tick_clip.w > 0.0 {
                    let tndc = tick_clip.truncate() / tick_clip.w;
                    let tx = (tndc.x + 1.0) * 0.5 * w as f32;
                    let ty = (1.0 - tndc.y) * 0.5 * h as f32;
                    let push = glam::Vec2::new(sx - tx, sy - ty);
                    let push_len = push.length();
                    if push_len < 1.0 {
                        continue;
                    }
                    if push_len < MIN_PUSH_PX {
                        let n = push / push_len;
                        sx = tx + n.x * MIN_PUSH_PX;
                        sy = ty + n.y * MIN_PUSH_PX;
                    }
                }
            }

            text_areas.push(TextArea {
                buffer: &label.glyph_buf,
                left: sx,
                top: sy,
                scale: 1.0,
                bounds: TextBounds::default(),
                default_color: if label.is_axis_title {
                    Color::rgb(220, 220, 240)
                } else {
                    Color::rgb(200, 200, 200)
                },
                custom_glyphs: &[],
            });
        }
        } // end grid_visible guard

        // Chart2D axis labels — pixel positions computed dynamically from state + w/h.
        if let Some(ref state) = self.chart2d_state {
            let pl_px  = state.plot_left   * w as f32;
            let pb_px  = state.plot_bottom * h as f32;
            let pt_px  = state.plot_top    * h as f32;
            let line_h = 11.0_f32 * 1.4;  // matches sz_tick * 1.4 from build_label_buffer

            // Pre-compute max Y-tick label width so the Y-axis title can be
            // placed to the left of the widest tick without overlapping.
            let max_ytick_w: f32 = self.chart2d_text_labels.iter()
                .filter(|l| matches!(l.kind, Chart2DLabelKind::YTick))
                .map(|l| l.glyph_buf.layout_runs()
                    .flat_map(|r| r.glyphs.iter())
                    .fold(0.0_f32, |acc, g| acc.max(g.x + g.w)))
                .fold(0.0_f32, f32::max);

            for label in &self.chart2d_text_labels {
                let (sx, sy, color) = match label.kind {
                    Chart2DLabelKind::XTick => {
                        let px = (label.ndc_val + 1.0) * 0.5 * w as f32;
                        let text_w: f32 = label.glyph_buf.layout_runs()
                            .flat_map(|r| r.glyphs.iter())
                            .fold(0.0_f32, |acc, g| acc.max(g.x + g.w));
                        (px - text_w * 0.5, pb_px + 6.0, Color::rgb(190, 190, 195))
                    }
                    Chart2DLabelKind::YTick => {
                        let py = (1.0 - label.ndc_val) * 0.5 * h as f32;
                        let text_w: f32 = label.glyph_buf.layout_runs()
                            .flat_map(|r| r.glyphs.iter())
                            .fold(0.0_f32, |acc, g| acc.max(g.x + g.w));
                        (pl_px - 8.0 - text_w, py - line_h * 0.5, Color::rgb(190, 190, 195))
                    }
                    Chart2DLabelKind::XTitle => {
                        let px = (label.ndc_val + 1.0) * 0.5 * w as f32;
                        let text_w: f32 = label.glyph_buf.layout_runs()
                            .flat_map(|r| r.glyphs.iter())
                            .fold(0.0_f32, |acc, g| acc.max(g.x + g.w));
                        (px - text_w * 0.5, pb_px + 28.0, Color::rgb(210, 210, 230))
                    }
                    Chart2DLabelKind::YTitle => {
                        let py = (1.0 - label.ndc_val) * 0.5 * h as f32;
                        let text_w: f32 = label.glyph_buf.layout_runs()
                            .flat_map(|r| r.glyphs.iter())
                            .fold(0.0_f32, |acc, g| acc.max(g.x + g.w));
                        let title_h = 13.0_f32 * 1.4;
                        // Place to the left of the widest tick label (8px gap + 6px gap).
                        let right_edge = pl_px - 8.0 - max_ytick_w - 6.0;
                        (right_edge - text_w, py - title_h * 0.5, Color::rgb(210, 210, 230))
                    }
                    Chart2DLabelKind::ChartTitle => {
                        // Centered horizontally on the plot area, above the frame.
                        let px = (label.ndc_val + 1.0) * 0.5 * w as f32;
                        let text_w: f32 = label.glyph_buf.layout_runs()
                            .flat_map(|r| r.glyphs.iter())
                            .fold(0.0_f32, |acc, g| acc.max(g.x + g.w));
                        let title_h = 14.0_f32 * 1.4;
                        (px - text_w * 0.5, pt_px - title_h - 4.0, Color::rgb(235, 235, 245))
                    }
                };
                // Suppress labels that would fall outside the window.
                if sx < -100.0 || sy < -100.0 || sx > w as f32 + 100.0 || sy > h as f32 + 100.0 {
                    continue;
                }
                // In 2D chart mode also suppress Y labels that are too close to
                // the top or bottom of the plot area (would overlap the frame).
                if matches!(label.kind, Chart2DLabelKind::YTick) {
                    if sy < pt_px - 4.0 || sy + line_h > pb_px + 4.0 { continue; }
                }
                text_areas.push(TextArea {
                    buffer: &label.glyph_buf,
                    left: sx,
                    top: sy,
                    scale: 1.0,
                    bounds: TextBounds::default(),
                    default_color: color,
                    custom_glyphs: &[],
                });
            }
        }

        // Scalar bar text labels (screen-space, pixel positions already known).
        for lbl in &self.scalar_bar_labels {
            text_areas.push(TextArea {
                buffer: &lbl.glyph_buf,
                left: lbl.px,
                top: lbl.py,
                scale: 1.0,
                bounds: TextBounds::default(),
                default_color: Color::rgb(200, 200, 200),
                custom_glyphs: &[],
            });
        }

        // Legend text labels (screen-space, fixed pixel positions)
        if self.legend_visible {
            for lbl in &self.legend_labels {
                text_areas.push(TextArea {
                    buffer: &lbl.glyph_buf,
                    left: lbl.px,
                    top: lbl.py,
                    scale: 1.0,
                    bounds: TextBounds::default(),
                    default_color: Color::rgb(200, 200, 200),
                    custom_glyphs: &[],
                });
            }
        }

        // Chart2D legend text labels
        if self.chart2d_legend_visible {
            for lbl in &self.chart2d_legend_text_labels {
                text_areas.push(TextArea {
                    buffer: &lbl.glyph_buf,
                    left: lbl.px,
                    top: lbl.py,
                    scale: 1.0,
                    bounds: TextBounds::default(),
                    default_color: Color::rgb(210, 210, 215),
                    custom_glyphs: &[],
                });
            }
        }

        // User-defined world-space labels
        for label in &self.user_labels {
            if !label.visible { continue; }
            let clip = vp * label.world_pos.extend(1.0);
            if clip.w <= 0.0 { continue; }
            let ndc = clip.truncate() / clip.w;
            if ndc.x < -1.1 || ndc.x > 1.1 || ndc.y < -1.1 || ndc.y > 1.1 { continue; }
            let sx = (ndc.x + 1.0) * 0.5 * w as f32;
            let sy = (1.0 - ndc.y) * 0.5 * h as f32;

            // Measure rendered text width from layout runs for anchor offset.
            let text_w: f32 = label.glyph_buf.layout_runs()
                .flat_map(|r| r.glyphs.iter())
                .fold(0.0_f32, |acc, g| acc.max(g.x + g.w));
            let text_h = label.glyph_buf.metrics().line_height;

            let (ox, oy) = match label.anchor {
                LabelAnchor::Center => (-text_w * 0.5, -text_h * 0.5),
                LabelAnchor::Left   => (4.0, -text_h * 0.5),
                LabelAnchor::Right  => (-text_w - 4.0, -text_h * 0.5),
                LabelAnchor::Top    => (-text_w * 0.5, -text_h - 4.0),
                LabelAnchor::Bottom => (-text_w * 0.5, 4.0),
            };

            let r = (label.color[0].clamp(0.0, 1.0) * 255.0) as u8;
            let g_ch = (label.color[1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (label.color[2].clamp(0.0, 1.0) * 255.0) as u8;
            let a = (label.color[3].clamp(0.0, 1.0) * 255.0) as u8;

            text_areas.push(TextArea {
                buffer: &label.glyph_buf,
                left: sx + ox,
                top: sy + oy,
                scale: 1.0,
                bounds: TextBounds::default(),
                default_color: Color::rgba(r, g_ch, b, a),
                custom_glyphs: &[],
            });
        }

        // Always call prepare — even with an empty list, this clears any
        // glyph vertices from the previous frame, preventing stale text when
        // all labels project off-screen.
        let _ = self.text_renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.font_system,
            &mut self.text_atlas,
            &self.viewport,
            text_areas,
            &mut self.swash_cache,
        );
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn push_chart2d_tri(
    out: &mut Vec<MeshVertex>,
    a: Vec2,
    b: Vec2,
    c: Vec2,
    color: [f32; 3],
) {
    let rgba = [color[0], color[1], color[2], 1.0];
    out.push(MeshVertex { position: [a.x, a.y, 0.0], color: rgba });
    out.push(MeshVertex { position: [b.x, b.y, 0.0], color: rgba });
    out.push(MeshVertex { position: [c.x, c.y, 0.0], color: rgba });
}

/// Build square marker vertices (2 triangles each) in data space.
/// `style`: 1 = filled square (only supported style currently).
fn build_marker_vertices(
    x: &[f32],
    y: &[f32],
    style: u8,
    size_px: f32,
    every: u32,
    color: [f32; 3],
    state: &Chart2DState,
    width: u32,
    height: u32,
) -> Vec<MeshVertex> {
    if style == 0 || every == 0 { return vec![]; }
    let px_per_x = (state.plot_right - state.plot_left) * width.max(1) as f32
        / (state.effective_x1() - state.effective_x0()).abs().max(1e-10);
    let px_per_y = (state.plot_bottom - state.plot_top) * height.max(1) as f32
        / (state.effective_y1() - state.effective_y0()).abs().max(1e-10);
    if px_per_x <= 0.0 || px_per_y <= 0.0 { return vec![]; }
    let hw = size_px * 0.5 / px_per_x;  // half-width in chart (log) units
    let hh = size_px * 0.5 / px_per_y;  // half-height in chart (log) units
    let rgba = [color[0], color[1], color[2], 1.0_f32];
    let n = x.len().min(y.len());
    let every = every.max(1) as usize;
    let mut out = Vec::with_capacity((n / every + 1) * 6);
    for i in (0..n).step_by(every) {
        let xi = state.chart_x(x[i]); let yi = state.chart_y(y[i]);
        if !xi.is_finite() || !yi.is_finite() { continue; }
        let bl = [xi - hw, yi - hh, 0.0_f32];
        let br = [xi + hw, yi - hh, 0.0_f32];
        let tr = [xi + hw, yi + hh, 0.0_f32];
        let tl = [xi - hw, yi + hh, 0.0_f32];
        out.push(MeshVertex { position: bl, color: rgba });
        out.push(MeshVertex { position: br, color: rgba });
        out.push(MeshVertex { position: tr, color: rgba });
        out.push(MeshVertex { position: bl, color: rgba });
        out.push(MeshVertex { position: tr, color: rgba });
        out.push(MeshVertex { position: tl, color: rgba });
    }
    out
}

fn xy_to_thick_line_vertices(
    x: &[f32],
    y: &[f32],
    color: [f32; 3],
    line_width: f32,
    state: Option<&Chart2DState>,
    width: u32,
    height: u32,
) -> Vec<MeshVertex> {
    let Some(state) = state else { return vec![]; };
    let n = x.len().min(y.len());
    if n < 2 || !line_width.is_finite() || line_width <= 0.0 {
        return vec![];
    }

    let px_per_x = (state.plot_right - state.plot_left) * width.max(1) as f32
        / (state.effective_x1() - state.effective_x0()).abs().max(1e-10);
    let px_per_y = (state.plot_bottom - state.plot_top) * height.max(1) as f32
        / (state.effective_y1() - state.effective_y0()).abs().max(1e-10);
    if px_per_x <= 0.0 || px_per_y <= 0.0 {
        return vec![];
    }

    let mut pts: Vec<Vec2> = Vec::with_capacity(n);
    for i in 0..n {
        let xi = x[i];
        let yi = y[i];
        if !xi.is_finite() || !yi.is_finite() {
            continue;
        }
        let p = Vec2::new(state.chart_x(xi), state.chart_y(yi));
        if pts.last().is_some_and(|q| (*q - p).length_squared() < 1e-12) {
            continue;
        }
        pts.push(p);
    }
    if pts.len() < 2 {
        return vec![];
    }

    let mut seg_normals: Vec<Vec2> = Vec::with_capacity(pts.len() - 1);
    for win in pts.windows(2) {
        let d = win[1] - win[0];
        let d_screen = Vec2::new(d.x * px_per_x, d.y * px_per_y);
        let len = d_screen.length();
        if len < 1e-6 {
            continue;
        }
        let dir = d_screen / len;
        seg_normals.push(Vec2::new(-dir.y, dir.x));
    }
    if seg_normals.is_empty() {
        return vec![];
    }

    let half_w = line_width * 0.5;
    let last = pts.len() - 1;
    let mut offsets_screen = vec![Vec2::ZERO; pts.len()];
    offsets_screen[0] = seg_normals[0] * half_w;
    offsets_screen[last] = seg_normals[seg_normals.len() - 1] * half_w;
    for i in 1..last {
        let n_prev = seg_normals[i - 1];
        let n_next = seg_normals[i.min(seg_normals.len() - 1)];
        let sum = n_prev + n_next;
        if sum.length_squared() < 1e-6 {
            offsets_screen[i] = n_next * half_w;
            continue;
        }
        let miter = sum.normalize();
        let denom = miter.dot(n_next).abs();
        let scale = if denom < 0.25 {
            half_w
        } else {
            (half_w / denom).min(half_w * 4.0)
        };
        offsets_screen[i] = miter * scale;
    }

    let mut out = Vec::with_capacity((pts.len() - 1) * 6);
    for i in 0..last {
        let off0 = Vec2::new(offsets_screen[i].x / px_per_x, offsets_screen[i].y / px_per_y);
        let off1 = Vec2::new(
            offsets_screen[i + 1].x / px_per_x,
            offsets_screen[i + 1].y / px_per_y,
        );
        let p0 = pts[i];
        let p1 = pts[i + 1];
        let l0 = p0 + off0;
        let r0 = p0 - off0;
        let l1 = p1 + off1;
        let r1 = p1 - off1;
        push_chart2d_tri(&mut out, l0, r0, l1, color);
        push_chart2d_tri(&mut out, l1, r0, r1, color);
    }
    out
}

/// Build NDC-space geometry for: 4-side frame, X/Y grid lines, and X/Y tick marks.
fn build_chart2d_axis_verts(state: &Chart2DState, width: u32, height: u32) -> Vec<LineVertex> {
    let (pl, pr, pt, pb) = (state.plot_left, state.plot_right, state.plot_top, state.plot_bottom);
    let ndc_l = pl * 2.0 - 1.0;
    let ndc_r = pr * 2.0 - 1.0;
    let ndc_t = 1.0 - 2.0 * pt;
    let ndc_b = 1.0 - 2.0 * pb;

    let frame_col: [f32; 3] = [0.55, 0.55, 0.60];
    let grid_col:  [f32; 3] = [0.18, 0.18, 0.22];

    let tick_y = 7.0 / height.max(1) as f32 * 2.0;
    let tick_x = 7.0 / width.max(1)  as f32 * 2.0;

    let sx = state.scale_x(); let ox = state.offset_x();
    let sy = state.scale_y(); let oy = state.offset_y();

    let mut v: Vec<LineVertex> = Vec::new();
    let mut seg = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| {
        v.push(LineVertex { position: a, color: c });
        v.push(LineVertex { position: b, color: c });
    };

    // Frame (4 edges)
    seg([ndc_l, ndc_b, 0.0], [ndc_r, ndc_b, 0.0], frame_col);
    seg([ndc_l, ndc_b, 0.0], [ndc_l, ndc_t, 0.0], frame_col);
    seg([ndc_l, ndc_t, 0.0], [ndc_r, ndc_t, 0.0], frame_col);
    seg([ndc_r, ndc_t, 0.0], [ndc_r, ndc_b, 0.0], frame_col);

    // X grid lines + tick marks
    for &xt in &chart2d_x_ticks(state) {
        let nx = sx * state.chart_x(xt) + ox;
        if nx < ndc_l - 0.002 || nx > ndc_r + 0.002 { continue; }
        seg([nx, ndc_b, 0.0], [nx, ndc_t, 0.0], grid_col);
        seg([nx, ndc_b, 0.0], [nx, ndc_b - tick_y, 0.0], frame_col);
    }

    // Y tick marks + Y grid lines
    for &yt in &chart2d_y_ticks(state) {
        let ny = sy * state.chart_y(yt) + oy;
        if ny < ndc_b - 0.002 || ny > ndc_t + 0.002 { continue; }
        seg([ndc_l, ny, 0.0], [ndc_l - tick_x, ny, 0.0], frame_col);
        seg([ndc_l, ny, 0.0], [ndc_r,  ny, 0.0], grid_col);
    }

    v
}

fn axis_ticks_fixed_step(lo: f32, hi: f32, step: f32) -> Vec<f32> {
    let range = hi - lo;
    if range < 1e-10 || step <= 0.0 {
        return vec![];
    }
    let first = (lo / step).ceil() * step;
    let mut ticks = Vec::new();
    let mut t = first;
    while t <= hi + step * 1e-4 {
        ticks.push(t);
        t += step;
    }
    ticks
}

/// Return data-space tick values for the X axis, respecting log scale.
fn chart2d_x_ticks(state: &Chart2DState) -> Vec<f32> {
    if state.x_log_scale {
        axis_ticks_log(state.x0, state.x1)
    } else {
        axis_ticks_fixed_step(state.x0, state.x1, state.x_tick_step)
    }
}

/// Return data-space tick values for the Y axis, respecting log scale.
fn chart2d_y_ticks(state: &Chart2DState) -> Vec<f32> {
    if state.y_log_scale {
        axis_ticks_log(state.y0, state.y1)
    } else {
        axis_ticks_fixed_step(state.y0, state.y1, state.y_tick_step)
    }
}

/// Ray-casting point-in-polygon test (screen space, Y-down).
fn point_in_polygon(px: f32, py: f32, poly: &[[f32; 2]]) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let [xi, yi] = poly[i];
        let [xj, yj] = poly[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn line_vertex_bounds(verts: &[LineVertex]) -> (Vec3, Vec3) {
    let mut bmin = Vec3::splat(f32::INFINITY);
    let mut bmax = Vec3::splat(f32::NEG_INFINITY);
    for v in verts {
        let p = Vec3::from(v.position);
        bmin = bmin.min(p);
        bmax = bmax.max(p);
    }
    if verts.is_empty() { (Vec3::ZERO, Vec3::ZERO) } else { (bmin, bmax) }
}

fn make_depth_texture(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn build_point_pipeline(
    device: &wgpu::Device,
    uniform_layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    wgsl: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("points_shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("point_layout"),
        bind_group_layouts: &[uniform_layout],
        push_constant_ranges: &[],
    });
    let stride = std::mem::size_of::<PointInstance>() as u64;
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("point_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: stride,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                    wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32   },
                    wgpu::VertexAttribute { offset: 16, shader_location: 2, format: wgpu::VertexFormat::Float32x3 },
                    wgpu::VertexAttribute { offset: 28, shader_location: 3, format: wgpu::VertexFormat::Float32   },
                ],
            }],
        },
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

fn build_line_pipeline(
    device: &wgpu::Device,
    uniform_layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    wgsl: &str,
    depth_test: bool,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lines_shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("line_layout"),
        bind_group_layouts: &[uniform_layout],
        push_constant_ranges: &[],
    });
    let stride = std::mem::size_of::<LineVertex>() as u64;
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("line_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                    wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
                ],
            }],
        },
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: if depth_test { wgpu::CompareFunction::Less } else { wgpu::CompareFunction::Always },
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

// ── Mesh pipeline builders ────────────────────────────────────────────────────

fn build_mesh_pipeline(
    device: &wgpu::Device,
    uniform_layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    wgsl: &str,
    transparent: bool,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh_shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mesh_layout"),
        bind_group_layouts: &[uniform_layout],
        push_constant_ranges: &[],
    });
    let stride = std::mem::size_of::<MeshVertex>() as u64;
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(if transparent { "mesh_transparent" } else { "mesh_opaque" }),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                    wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
                ],
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: !transparent,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: if transparent { Some(wgpu::BlendState::ALPHA_BLENDING) } else { None },
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

fn build_chart2d_line_pipeline(
    device: &wgpu::Device,
    uniform_layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    wgsl: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chart2d_line_shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chart2d_line_layout"),
        bind_group_layouts: &[uniform_layout],
        push_constant_ranges: &[],
    });
    let stride = std::mem::size_of::<MeshVertex>() as u64;
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("chart2d_line_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                    wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
                ],
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

/// Build the GPU-side chart2d thick-line pipeline.
/// Vertex format: `ThickLineVert` (48 bytes) — chart-space prev/curr/next positions,
/// side flag, line width, and RGBA colour.  Miter joins are computed entirely in the
/// vertex shader, so axis-limit changes require only a uniform buffer write.
fn build_chart2d_thick_pipeline(
    device: &wgpu::Device,
    uniform_layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    wgsl: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chart2d_thick_shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chart2d_thick_layout"),
        bind_group_layouts: &[uniform_layout],
        push_constant_ranges: &[],
    });
    let stride = std::mem::size_of::<ThickLineVert>() as u64;
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("chart2d_thick_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    // pos_prev: float32x2  @ offset  0
                    wgpu::VertexAttribute { offset:  0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                    // pos_curr: float32x2  @ offset  8
                    wgpu::VertexAttribute { offset:  8, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
                    // pos_next: float32x2  @ offset 16
                    wgpu::VertexAttribute { offset: 16, shader_location: 2, format: wgpu::VertexFormat::Float32x2 },
                    // side:     float32    @ offset 24
                    wgpu::VertexAttribute { offset: 24, shader_location: 3, format: wgpu::VertexFormat::Float32 },
                    // line_width: float32  @ offset 28
                    wgpu::VertexAttribute { offset: 28, shader_location: 4, format: wgpu::VertexFormat::Float32 },
                    // color:    float32x4  @ offset 32
                    wgpu::VertexAttribute { offset: 32, shader_location: 5, format: wgpu::VertexFormat::Float32x4 },
                ],
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

/// Build GPU-side thick-line vertex data for `chart2d_thick_pipeline`.
///
/// Positions are stored in chart space (log10 already applied for log scales) so
/// the view_proj uniform can transform them without any per-frame CPU work.
/// Each segment emits 6 `ThickLineVert` vertices (2 triangles); each vertex carries
/// the prev/curr/next neighbours needed for GPU miter-join computation.
fn xy_to_gpu_thick_verts(
    x: &[f32],
    y: &[f32],
    color: [f32; 3],
    line_width: f32,
    state: Option<&Chart2DState>,
) -> Vec<ThickLineVert> {
    let Some(state) = state else { return vec![]; };
    let n = x.len().min(y.len());
    if n < 2 || !line_width.is_finite() || line_width <= 0.0 {
        return vec![];
    }

    // Convert data → chart space (applying log₁₀ when log scale is active) and
    // deduplicate coincident points, matching xy_to_thick_line_vertices exactly.
    let mut pts: Vec<[f32; 2]> = Vec::with_capacity(n);
    for i in 0..n {
        let xi = x[i]; let yi = y[i];
        if !xi.is_finite() || !yi.is_finite() { continue; }
        let p = [state.chart_x(xi), state.chart_y(yi)];
        if let Some(&last) = pts.last() {
            let dx = p[0] - last[0]; let dy = p[1] - last[1];
            if dx * dx + dy * dy < 1e-12 { continue; }
        }
        pts.push(p);
    }
    let m = pts.len();
    if m < 2 { return vec![]; }

    let rgba = [color[0], color[1], color[2], 1.0_f32];
    let mut out = Vec::with_capacity((m - 1) * 6);

    for i in 0..m - 1 {
        // For endpoint caps, duplicate the endpoint as prev/next so the shader
        // detects a zero-length segment and falls back to the single-direction normal.
        let prev  = if i > 0      { pts[i - 1] } else { pts[i] };
        let curr  = pts[i];
        let next  = pts[i + 1];
        let next2 = if i + 2 < m  { pts[i + 2] } else { pts[i + 1] };

        // Left (side = +1) and right (side = -1) at the current end of this segment.
        let curr_l = ThickLineVert { pos_prev: prev,  pos_curr: curr, pos_next: next,  side:  1.0, line_width, color: rgba };
        let curr_r = ThickLineVert { pos_prev: prev,  pos_curr: curr, pos_next: next,  side: -1.0, line_width, color: rgba };
        // Left and right at the far end of this segment.
        let next_l = ThickLineVert { pos_prev: curr,  pos_curr: next, pos_next: next2, side:  1.0, line_width, color: rgba };
        let next_r = ThickLineVert { pos_prev: curr,  pos_curr: next, pos_next: next2, side: -1.0, line_width, color: rgba };

        // Triangle 1: curr_L, curr_R, next_L
        out.push(curr_l);
        out.push(curr_r);
        out.push(next_l);
        // Triangle 2: curr_R, next_R, next_L
        out.push(curr_r);
        out.push(next_r);
        out.push(next_l);
    }
    out
}

fn build_wireframe_pipeline(
    device: &wgpu::Device,
    uniform_layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    wgsl: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wireframe_shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wireframe_layout"),
        bind_group_layouts: &[uniform_layout],
        push_constant_ranges: &[],
    });
    let stride = std::mem::size_of::<MeshVertex>() as u64;
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mesh_wireframe"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                    wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
                ],
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

// ── Mesh render helper (called from both render() and screenshot()) ───────────

fn draw_mesh_actors<'a>(
    mesh_actors: &'a [MeshActor],
    pipeline_wire: &'a wgpu::RenderPipeline,
    pipeline_opaque: &'a wgpu::RenderPipeline,
    pipeline_transparent: &'a wgpu::RenderPipeline,
    bind_group: &'a wgpu::BindGroup,
    view_proj: glam::Mat4,
    pass: &mut wgpu::RenderPass<'a>,
) {
    // 1. Wireframe actors — alpha-blended, back-to-front sorted
    let mut wire_idx: Vec<(usize, f32)> = mesh_actors.iter().enumerate()
        .filter(|(_, ma)| ma.visible && ma.wireframe && ma.index_count > 0)
        .map(|(i, ma)| {
            let c = (ma.data_min + ma.data_max) * 0.5;
            let clip = view_proj * glam::Vec4::new(c.x, c.y, c.z, 1.0);
            let depth = if clip.w.abs() > 1e-7 { clip.z / clip.w } else { 0.0 };
            (i, depth)
        })
        .collect();
    wire_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pass.set_pipeline(pipeline_wire);
    pass.set_bind_group(0, bind_group, &[]);
    for (i, _) in &wire_idx {
        let ma = &mesh_actors[*i];
        if let (Some(vs), Some(is)) = (ma.vbuf.slice(), ma.ibuf.slice()) {
            pass.set_vertex_buffer(0, vs);
            pass.set_index_buffer(is, wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..ma.index_count, 0, 0..1);
        }
    }

    // Partition filled actors by alpha for depth-correct rendering.
    let mut opaque_idx: Vec<usize> = Vec::new();
    let mut transp_idx: Vec<(usize, f32)> = Vec::new();
    for (i, ma) in mesh_actors.iter().enumerate() {
        if !ma.visible || ma.wireframe || ma.index_count == 0 { continue; }
        let c = (ma.data_min + ma.data_max) * 0.5;
        let clip = view_proj * glam::Vec4::new(c.x, c.y, c.z, 1.0);
        let depth = if clip.w.abs() > 1e-7 { clip.z / clip.w } else { 0.0 };
        if ma.color[3] >= 1.0 {
            opaque_idx.push(i);
        } else {
            transp_idx.push((i, depth));
        }
    }

    // 2. Opaque filled meshes
    pass.set_pipeline(pipeline_opaque);
    pass.set_bind_group(0, bind_group, &[]);
    for i in &opaque_idx {
        let ma = &mesh_actors[*i];
        if let (Some(vs), Some(is)) = (ma.vbuf.slice(), ma.ibuf.slice()) {
            pass.set_vertex_buffer(0, vs);
            pass.set_index_buffer(is, wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..ma.index_count, 0, 0..1);
        }
    }

    // 3. Transparent filled meshes — back-to-front (highest NDC depth drawn first)
    transp_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pass.set_pipeline(pipeline_transparent);
    pass.set_bind_group(0, bind_group, &[]);
    for (i, _) in &transp_idx {
        let ma = &mesh_actors[*i];
        if let (Some(vs), Some(is)) = (ma.vbuf.slice(), ma.ibuf.slice()) {
            pass.set_vertex_buffer(0, vs);
            pass.set_index_buffer(is, wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..ma.index_count, 0, 0..1);
        }
    }
}

// ── Mesh geometry helpers ─────────────────────────────────────────────────────

fn mesh_bounds(vertices: &[[f32; 3]]) -> (Vec3, Vec3) {
    let mut bmin = Vec3::splat(f32::INFINITY);
    let mut bmax = Vec3::splat(f32::NEG_INFINITY);
    for &[x, y, z] in vertices {
        let v = Vec3::new(x, y, z);
        bmin = bmin.min(v);
        bmax = bmax.max(v);
    }
    if bmin.x > bmax.x { (Vec3::ZERO, Vec3::ZERO) } else { (bmin, bmax) }
}

fn triangles_to_wireframe_indices(indices: &[[u32; 3]]) -> Vec<u32> {
    let mut wire = Vec::with_capacity(indices.len() * 6);
    for &[a, b, c] in indices {
        wire.extend_from_slice(&[a, b, b, c, c, a]);
    }
    wire
}

// ── Unit tests (no GPU required) ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity VP on a 200×200 screen:
    ///   world [x, y, z] → screen [(x+1)*100, (1-y)*100]
    fn unit_vp() -> glam::Mat4 { glam::Mat4::IDENTITY }
    const W: f32 = 200.0;
    const H: f32 = 200.0;

    fn build(positions: &[[f32; 3]]) -> ScreenPickCache {
        ScreenPickCache::build(positions, unit_vp(), W, H)
    }

    #[test]
    fn grid_assigns_point_to_correct_cell() {
        // world (0,0,0) → screen (100,100) with unit_vp on 200×200
        let c = build(&[[0.0, 0.0, 0.0]]);
        assert_eq!(c.screen_xy[0], Some([100.0, 100.0]));
        // Cell: cx = floor(100/16) = 6, cy = floor(100/16) = 6
        // cols = ceil(200/16) = 13
        let cols = ((W / GRID_CELL_PX).ceil() as u32).max(1);
        let cell = (6 * cols + 6) as usize;
        let start = c.cell_start[cell] as usize;
        let end   = c.cell_start[cell + 1] as usize;
        assert_eq!(end - start, 1);
        assert_eq!(c.sorted_pts[start], 0);
    }

    #[test]
    fn for_each_finds_all_points_in_full_rect() {
        // Four points at screen-space corners (NDC ±0.9 → screen 10/190)
        let pos = vec![
            [-0.9_f32,  0.9, 0.0],  // screen (10, 10)
            [ 0.9_f32,  0.9, 0.0],  // screen (190, 10)
            [-0.9_f32, -0.9, 0.0],  // screen (10, 190)
            [ 0.9_f32, -0.9, 0.0],  // screen (190, 190)
        ];
        let c = build(&pos);
        let mut all = Vec::new();
        c.for_each_in_rect(0.0, 0.0, W, H, |i, _| all.push(i));
        all.sort();
        assert_eq!(all, [0, 1, 2, 3]);
    }

    #[test]
    fn for_each_restricts_to_quadrant() {
        let pos = vec![
            [-0.9_f32,  0.9, 0.0],  // screen (10, 10)   — top-left
            [ 0.9_f32, -0.9, 0.0],  // screen (190, 190) — bottom-right
        ];
        let c = build(&pos);
        let mut tl = Vec::new();
        c.for_each_in_rect(0.0, 0.0, 100.0, 100.0, |i, _| tl.push(i));
        assert!(tl.contains(&0), "top-left point must be in top-left quadrant");
        assert!(!tl.contains(&1), "bottom-right point must NOT be in top-left quadrant");
    }

    #[test]
    fn diagonal_corner_fallback_scenario() {
        // Encodes the exact geometry that exposes the pick_point contract violation
        // and verifies `best_dist_sq > R²` is the correct fallback trigger.
        //
        // Grid cell size = 16 px; R = 32 px (2 cells).
        // fast-path calls for_each_in_rect(cx-R, cy-R, cx+R, cy+R).
        //
        // With cursor C = screen (0, 0):
        //   cx0 = max(0, floor(-32/16)) = 0
        //   cx1 = min(cols-1, ceil(32/16)) = 2
        //   → cells 0..=2 are searched in x; cell 2 covers x=[32,48).
        //   → any point with screen_x < 48 is captured by the local search.
        //   → a point at screen_x ≥ 48 (cell 3+) is MISSED.
        //
        //  Point A at screen (47, 47):
        //    • Cell (2, 2) — within local search range       → found by local search
        //    • Euclidean dist from C: √(47²+47²) ≈ 66.5 px
        //
        //  Point B at screen (49, 0):
        //    • Cell (3, 0) — outside local search range      → MISSED by local search
        //    • Euclidean dist from C: 49 px                  → CLOSER than A
        //
        //  Correct result: return B (globally nearest).
        //  Old bug:  local finds A, best.is_none()=false → fallback skipped → returns A.
        //  Fixed:    local finds A, A's dist²≈4418>R²=1024 → fallback runs → finds B.

        const R: f32 = GRID_CELL_PX * 2.0;

        // Invert screen projection to find world coords (identity VP, 200×200).
        //   screen_x = (wx + 1) * W/2  →  wx = screen_x / (W/2) - 1
        //   screen_y = (1 - wy) * H/2  →  wy = 1 - screen_y / (H/2)
        let world = |sx: f32, sy: f32| -> [f32; 3] {
            [sx / (W * 0.5) - 1.0, 1.0 - sy / (H * 0.5), 0.0]
        };

        // A at screen (47, 47); B at screen (49, 0).
        let screen_a = (47.0_f32, 47.0_f32);
        let screen_b = (49.0_f32, 0.0_f32);
        let pos = vec![world(screen_a.0, screen_a.1), world(screen_b.0, screen_b.1)];
        let c = build(&pos);

        let [ax, ay] = c.screen_xy[0].expect("A must project on-screen");
        assert!((ax - screen_a.0).abs() < 0.01 && (ay - screen_a.1).abs() < 0.01,
            "A projects to ({ax},{ay}), expected {:?}", screen_a);

        let [bx, by] = c.screen_xy[1].expect("B must project on-screen");
        assert!((bx - screen_b.0).abs() < 0.01 && (by - screen_b.1).abs() < 0.01,
            "B projects to ({bx},{by}), expected {:?}", screen_b);

        // Local search around cursor (0,0): A found (cell 2), B not (cell 3).
        let mut local: Vec<u32> = Vec::new();
        c.for_each_in_rect(-R, -R, R, R, |i, _| local.push(i));
        assert!(local.contains(&0), "A (cell 2) must be in the local search range");
        assert!(!local.contains(&1), "B (cell 3) must be outside the local search range");

        // Distances from cursor (0, 0):
        let dist_a_sq = screen_a.0.powi(2) + screen_a.1.powi(2); // 47²+47² ≈ 4418
        let dist_b_sq = screen_b.0.powi(2) + screen_b.1.powi(2); // 49²     = 2401

        assert!(dist_b_sq < dist_a_sq, "B must be globally closer than A");

        // The correct fallback trigger: A's squared distance exceeds R².
        // When this holds, a closer point outside the local band (B) may exist.
        assert!(dist_a_sq > R * R,
            "A's dist² ({dist_a_sq}) must exceed R² ({}) so the fallback fires", R * R);
    }

    #[test]
    fn chart2d_fixed_step_ticks_do_not_flip_between_nice_intervals() {
        let t0 = axis_ticks_fixed_step(0.0, 10.0, 2.0);
        let t1 = axis_ticks_fixed_step(0.001, 10.001, 2.0);
        let t2 = axis_ticks_fixed_step(1.999, 11.999, 2.0);

        assert_eq!(t0, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        assert_eq!(t1, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        assert_eq!(t2, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn chart2d_fixed_step_y_ticks_do_not_flip_between_nice_intervals() {
        let t0 = axis_ticks_fixed_step(-2.0, 2.0, 1.0);
        let t1 = axis_ticks_fixed_step(-1.99, 2.01, 1.0);
        let t2 = axis_ticks_fixed_step(-1.01, 2.99, 1.0);

        assert_eq!(t0, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert_eq!(t1, vec![-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(t2, vec![-1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn chart2d_thick_line_generates_triangle_mesh() {
        let x = [0.0, 10.0];
        let y = [0.0, 0.0];
        let color = [1.0, 0.0, 0.0];
        let state = Chart2DState {
            plot_left: 0.1,
            plot_right: 0.9,
            plot_top: 0.1,
            plot_bottom: 0.9,
            x0: 0.0,
            x1: 10.0,
            y0: -1.0,
            y1: 1.0,
            x_label: String::new(),
            y_label: String::new(),
            title: String::new(),
            x_tick_cache: Vec::new(),
            x_tick_step: 2.0,
            y_tick_step: 1.0,
            x_tick_format: String::new(),
            y_tick_format: String::new(),
            x_log_scale: false,
            y_log_scale: false,
        };

        let verts = xy_to_thick_line_vertices(&x, &y, color, 4.0, Some(&state), 800, 600);

        assert_eq!(verts.len(), 6);
        let y_span = verts
            .iter()
            .map(|v| v.position[1])
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), yv| (mn.min(yv), mx.max(yv)));
        assert!(y_span.1 > y_span.0);
    }

    #[test]
    fn chart2d_wider_lines_produce_larger_mesh_span() {
        let color = [1.0, 1.0, 1.0];
        let x = [0.0, 10.0];
        let y = [0.0, 0.0];
        let state = Chart2DState {
            plot_left: 0.1,
            plot_right: 0.9,
            plot_top: 0.1,
            plot_bottom: 0.9,
            x0: 0.0,
            x1: 10.0,
            y0: -1.0,
            y1: 1.0,
            x_label: String::new(),
            y_label: String::new(),
            title: String::new(),
            x_tick_cache: Vec::new(),
            x_tick_step: 2.0,
            y_tick_step: 1.0,
            x_tick_format: String::new(),
            y_tick_format: String::new(),
            x_log_scale: false,
            y_log_scale: false,
        };

        let narrow = xy_to_thick_line_vertices(&x, &y, color, 2.0, Some(&state), 800, 600);
        let wide = xy_to_thick_line_vertices(&x, &y, color, 8.0, Some(&state), 800, 600);

        let span = |verts: &[MeshVertex]| {
            let (mn, mx) = verts
                .iter()
                .map(|v| v.position[1])
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), yv| (mn.min(yv), mx.max(yv)));
            mx - mn
        };
        assert!(span(&wide) > span(&narrow));
    }

    #[test]
    fn chart2d_line_width_in_data_units_tracks_y_scale() {
        let color = [1.0, 1.0, 1.0];
        let state0 = Chart2DState {
            plot_left: 0.1,
            plot_right: 0.9,
            plot_top: 0.1,
            plot_bottom: 0.9,
            x0: 0.0,
            x1: 10.0,
            y0: -1.0,
            y1: 1.0,
            x_label: String::new(),
            y_label: String::new(),
            title: String::new(),
            x_tick_cache: Vec::new(),
            x_tick_step: 2.0,
            y_tick_step: 1.0,
            x_tick_format: String::new(),
            y_tick_format: String::new(),
            x_log_scale: false,
            y_log_scale: false,
        };
        let state1 = Chart2DState {
            plot_left: 0.1,
            plot_right: 0.9,
            plot_top: 0.1,
            plot_bottom: 0.9,
            x0: 0.0,
            x1: 10.0,
            y0: -10.0,
            y1: 10.0,
            x_label: String::new(),
            y_label: String::new(),
            title: String::new(),
            x_tick_cache: Vec::new(),
            x_tick_step: 2.0,
            y_tick_step: 5.0,
            x_tick_format: String::new(),
            y_tick_format: String::new(),
            x_log_scale: false,
            y_log_scale: false,
        };
        let line_x = [0.0, 10.0];
        let y = [0.0, 0.0];

        let line0 = xy_to_thick_line_vertices(&line_x, &y, color, 4.0, Some(&state0), 800, 600);
        let line1 = xy_to_thick_line_vertices(&line_x, &y, color, 4.0, Some(&state1), 800, 600);

        let span = |verts: &[MeshVertex]| {
            let (mn, mx) = verts
                .iter()
                .map(|v| v.position[1])
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), yv| (mn.min(yv), mx.max(yv)));
            mx - mn
        };
        assert!(span(&line1) > span(&line0));
    }
}
