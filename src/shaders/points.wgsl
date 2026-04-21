struct Uniforms {
    view_proj: mat4x4<f32>,
    screen_size: vec2<f32>,
    // style: 0 = circle (soft), 1 = square, 2 = gaussian
    style: u32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) alpha: f32,
}

// Two triangles forming a unit quad centered at origin
var<private> QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-0.5, -0.5),
    vec2<f32>( 0.5, -0.5),
    vec2<f32>( 0.5,  0.5),
    vec2<f32>(-0.5, -0.5),
    vec2<f32>( 0.5,  0.5),
    vec2<f32>(-0.5,  0.5),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @location(0) position: vec3<f32>,
    @location(1) size: f32,
    @location(2) color: vec3<f32>,
    @location(3) alpha: f32,
) -> VertexOutput {
    let quad = QUAD[vid % 6u];
    let clip_center = uniforms.view_proj * vec4<f32>(position, 1.0);

    // Billboard offset in clip space: keeps size constant in pixels
    let ndc_offset = quad * size / uniforms.screen_size * 2.0;
    let clip_offset = vec4<f32>(ndc_offset * clip_center.w, 0.0, 0.0);

    var out: VertexOutput;
    out.clip_position = clip_center + clip_offset;
    out.color = color;
    out.uv = quad + vec2<f32>(0.5);
    out.alpha = alpha;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv - vec2<f32>(0.5));
    var a = in.alpha;
    if uniforms.style == 1u {
        // square — full quad, no discard
    } else if uniforms.style == 2u {
        // gaussian — soft falloff, no hard edge
        a *= exp(-8.0 * dist * dist);
        if a < 0.004 { discard; }
    } else {
        // circle (default) — hard edge with 1-pixel AA
        if dist > 0.5 { discard; }
        a *= 1.0 - smoothstep(0.38, 0.50, dist);
    }
    return vec4<f32>(in.color, a);
}
