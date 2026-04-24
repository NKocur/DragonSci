// GPU-side chart2d thick line shader.
//
// Each vertex stores the chart-space positions of the *previous*, *current*, and
// *next* polyline points together with a side flag (+1 = left, -1 = right), the
// line width in pixels, and the RGBA colour.  The vertex shader computes miter-
// join geometry entirely on the GPU, so axis-limit changes require only a write
// to the uniform buffer rather than O(N) vertex rebuilds.
//
// Vertex layout — ThickLineVert, 48 bytes:
//   pos_prev   float32x2  offset  0
//   pos_curr   float32x2  offset  8
//   pos_next   float32x2  offset 16
//   side       float32    offset 24
//   line_width float32    offset 28
//   color      float32x4  offset 32
//
// Uniforms layout (80 bytes) is identical to the main Uniforms struct so the
// same bind-group layout can be reused without a separate bind-group layout:
//   view_proj    mat4x4<f32>   offset  0   (chart space → clip space, w == 1)
//   screen_size  vec2<f32>     offset 64
//   style        u32           offset 72   (unused by this shader)
//   _pad         f32           offset 76

struct Uniforms {
    view_proj:   mat4x4<f32>,
    screen_size: vec2<f32>,
    style:       u32,
    _pad:        f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertIn {
    @location(0) pos_prev:   vec2<f32>,
    @location(1) pos_curr:   vec2<f32>,
    @location(2) pos_next:   vec2<f32>,
    @location(3) side:       f32,
    @location(4) line_width: f32,
    @location(5) color:      vec4<f32>,
}

struct VertOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       color:    vec4<f32>,
}

// Transform a chart-space 2D point to a "screen-scaled NDC" space where each
// axis is scaled by half the window dimension.  Since view_proj is an orthographic
// affine transform, clip.w == 1.0 always, so ndc == clip.xy.
// The resulting space has the same orientation as NDC (y-up) but is measured in
// half-pixels, making it convenient for pixel-accurate length comparisons.
fn to_screen(p: vec2<f32>) -> vec2<f32> {
    let clip = uniforms.view_proj * vec4<f32>(p, 0.0, 1.0);
    return clip.xy * uniforms.screen_size * 0.5;
}

@vertex
fn vs_main(v: VertIn) -> VertOut {
    let s_prev = to_screen(v.pos_prev);
    let s_curr = to_screen(v.pos_curr);
    let s_next = to_screen(v.pos_next);

    // Segment direction vectors in screen space (y-up, pixel units).
    let d0 = s_curr - s_prev;   // prev → curr
    let d1 = s_next - s_curr;   // curr → next
    let len0 = length(d0);
    let len1 = length(d1);

    let half_w = v.line_width * 0.5;

    // Compute screen-space offset (left-side miter, length == half_w at minimum).
    // Mirrors the CPU logic in xy_to_thick_line_vertices exactly:
    //   n = normalize(rotate90_CCW(d))  for each segment
    //   miter = normalize(n0 + n1)
    //   scale = clamp(half_w / dot(miter, n1), half_w, half_w * 4)
    var offset: vec2<f32>;

    if len0 > 1e-4 && len1 > 1e-4 {
        // Interior point: miter join of the two adjacent segments.
        let n0 = vec2<f32>(-d0.y, d0.x) / len0;   // left normal of prev→curr
        let n1 = vec2<f32>(-d1.y, d1.x) / len1;   // left normal of curr→next
        let sum = n0 + n1;
        let sum_len = length(sum);
        if sum_len < 1e-6 {
            // Anti-parallel segments (180° hairpin): fall back to next-segment normal.
            offset = n1 * half_w;
        } else {
            let miter = sum / sum_len;
            let denom = abs(dot(miter, n1));
            // Clamp miter length: never shorter than half_w, never longer than 4×.
            let scale = select(half_w, min(half_w / denom, half_w * 4.0), denom >= 0.25);
            offset = miter * scale;
        }
    } else if len1 > 1e-4 {
        // First endpoint (or degenerate prev segment): use next-segment normal.
        offset = vec2<f32>(-d1.y, d1.x) / len1 * half_w;
    } else if len0 > 1e-4 {
        // Last endpoint (or degenerate next segment): use prev-segment normal.
        offset = vec2<f32>(-d0.y, d0.x) / len0 * half_w;
    } else {
        // Fully degenerate vertex: arbitrary small upward offset so the vertex
        // is not at the exact same clip position as all its neighbours.
        offset = vec2<f32>(0.0, half_w);
    }

    // Displace curr by side * offset in screen space, then convert back to NDC.
    // side == +1.0 → left edge; side == -1.0 → right edge.
    let s_out  = s_curr + v.side * offset;
    let ndc_out = s_out / (uniforms.screen_size * 0.5);

    var out: VertOut;
    out.clip_pos = vec4<f32>(ndc_out, 0.0, 1.0);
    out.color    = v.color;
    return out;
}

@fragment
fn fs_main(v: VertOut) -> @location(0) vec4<f32> {
    return v.color;
}
