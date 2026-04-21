// Mesh overlay shader — used for convex hulls, ellipsoids, and wireframes.
// Vertex data: float32x3 position at offset 0, float32x4 color at offset 12.
// Color (including alpha) is baked per-vertex so one vertex buffer serves
// both filled and wireframe modes without extra bind groups.

struct Uniforms {
    view_proj:   mat4x4<f32>,
    screen_size: vec2<f32>,
    style:       u32,
    _pad:        f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertIn {
    @location(0) position: vec3<f32>,
    @location(1) color:    vec4<f32>,
}

struct VertOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       color:    vec4<f32>,
}

@vertex
fn vs_main(v: VertIn) -> VertOut {
    var out: VertOut;
    out.clip_pos = uniforms.view_proj * vec4<f32>(v.position, 1.0);
    out.color    = v.color;
    return out;
}

@fragment
fn fs_main(v: VertOut) -> @location(0) vec4<f32> {
    return v.color;
}
