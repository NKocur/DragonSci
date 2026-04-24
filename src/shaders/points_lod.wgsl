struct Uniforms {
    view_proj: mat4x4<f32>,
    screen_size: vec2<f32>,
    // style: 0 = circle (soft), 1 = square, 2 = gaussian
    style: u32,
    lod_factor: u32,
}

struct PointInstance {
    position: vec3<f32>,
    size: f32,
    color: vec3<f32>,
    alpha: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<PointInstance>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) alpha: f32,
}

var<private> QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-0.5, -0.5),
    vec2<f32>( 0.5, -0.5),
    vec2<f32>( 0.5,  0.5),
    vec2<f32>(-0.5, -0.5),
    vec2<f32>( 0.5,  0.5),
    vec2<f32>(-0.5,  0.5),
);

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    let quad = QUAD[vid % 6u];
    let source_index = (vid / 6u) * max(uniforms.lod_factor, 1u);
    let point = points[source_index];
    let clip_center = uniforms.view_proj * vec4<f32>(point.position, 1.0);

    let ndc_offset = quad * point.size / uniforms.screen_size * 2.0;
    let clip_offset = vec4<f32>(ndc_offset * clip_center.w, 0.0, 0.0);

    var out: VertexOutput;
    out.clip_position = clip_center + clip_offset;
    out.color = point.color;
    out.uv = quad + vec2<f32>(0.5);
    out.alpha = point.alpha;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv - vec2<f32>(0.5));
    var a = in.alpha;
    if uniforms.style == 1u {
    } else if uniforms.style == 2u {
        a *= exp(-8.0 * dist * dist);
        if a < 0.004 { discard; }
    } else {
        if dist > 0.5 { discard; }
        a *= 1.0 - smoothstep(0.38, 0.50, dist);
    }
    return vec4<f32>(in.color, a);
}
