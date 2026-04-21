// Colormaps defined by control points (r, g, b) in [0,1].
// Linear interpolation between adjacent points.

/// Resolve the control-point table for `name` once, then call `sample` in a loop.
pub fn resolve(name: &str) -> &'static [[f32; 3]] {
    control_points(name)
}

/// Sample a pre-resolved control-point table at normalised `t` in [0, 1].
#[inline(always)]
pub fn sample(cpts: &[[f32; 3]], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let n = cpts.len() - 1;
    let scaled = t * n as f32;
    let i = (scaled as usize).min(n - 1);
    let frac = scaled - i as f32;
    let a = cpts[i];
    let b = cpts[i + 1];
    [
        a[0] + (b[0] - a[0]) * frac,
        a[1] + (b[1] - a[1]) * frac,
        a[2] + (b[2] - a[2]) * frac,
    ]
}

fn control_points(name: &str) -> &'static [[f32; 3]] {
    match name {
        "plasma"   => &PLASMA,
        "inferno"  => &INFERNO,
        "magma"    => &MAGMA,
        "coolwarm" => &COOLWARM,
        "hot"      => &HOT,
        "gray" | "grey" => &GRAY,
        "turbo"    => &TURBO,
        "cividis"  => &CIVIDIS,
        "blues"    => &BLUES,
        "greens"   => &GREENS,
        "reds"     => &REDS,
        _          => &VIRIDIS,
    }
}

// Viridis: perceptually uniform, colorblind-friendly
static VIRIDIS: [[f32; 3]; 9] = [
    [0.267, 0.005, 0.329],
    [0.283, 0.141, 0.458],
    [0.254, 0.265, 0.530],
    [0.207, 0.372, 0.554],
    [0.164, 0.471, 0.558],
    [0.128, 0.566, 0.551],
    [0.200, 0.663, 0.475],
    [0.477, 0.821, 0.318],
    [0.993, 0.906, 0.144],
];

// Plasma
static PLASMA: [[f32; 3]; 9] = [
    [0.050, 0.030, 0.528],
    [0.272, 0.010, 0.630],
    [0.450, 0.004, 0.659],
    [0.616, 0.045, 0.614],
    [0.754, 0.127, 0.505],
    [0.862, 0.231, 0.381],
    [0.944, 0.349, 0.260],
    [0.989, 0.514, 0.108],
    [0.940, 0.975, 0.131],
];

// Inferno
static INFERNO: [[f32; 3]; 9] = [
    [0.001, 0.000, 0.014],
    [0.087, 0.024, 0.180],
    [0.242, 0.025, 0.353],
    [0.391, 0.061, 0.380],
    [0.545, 0.116, 0.347],
    [0.697, 0.179, 0.271],
    [0.838, 0.277, 0.170],
    [0.950, 0.449, 0.062],
    [0.988, 0.998, 0.644],
];

// Magma
static MAGMA: [[f32; 3]; 9] = [
    [0.001, 0.000, 0.014],
    [0.081, 0.028, 0.196],
    [0.224, 0.047, 0.380],
    [0.366, 0.069, 0.436],
    [0.512, 0.115, 0.431],
    [0.657, 0.175, 0.400],
    [0.803, 0.260, 0.373],
    [0.936, 0.425, 0.416],
    [0.988, 0.998, 0.645],
];

// Coolwarm: blue-white-red diverging
static COOLWARM: [[f32; 3]; 5] = [
    [0.017, 0.318, 0.824],
    [0.553, 0.714, 0.949],
    [0.881, 0.881, 0.881],
    [0.953, 0.600, 0.502],
    [0.706, 0.016, 0.149],
];

// Hot: black-red-yellow-white
static HOT: [[f32; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
];

// Gray
static GRAY: [[f32; 3]; 2] = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];

// Turbo: wide-gamut rainbow (Google 2019), good for depth/distance
static TURBO: [[f32; 3]; 9] = [
    [0.190, 0.072, 0.232],
    [0.085, 0.386, 0.830],
    [0.071, 0.679, 0.838],
    [0.156, 0.904, 0.581],
    [0.547, 0.987, 0.227],
    [0.916, 0.838, 0.104],
    [0.979, 0.531, 0.077],
    [0.840, 0.216, 0.062],
    [0.479, 0.015, 0.019],
];

// Cividis: colorblind-safe blue → yellow (perceptually uniform)
static CIVIDIS: [[f32; 3]; 9] = [
    [0.000, 0.135, 0.304],
    [0.068, 0.219, 0.409],
    [0.177, 0.296, 0.440],
    [0.284, 0.379, 0.453],
    [0.393, 0.464, 0.452],
    [0.516, 0.558, 0.422],
    [0.659, 0.659, 0.356],
    [0.810, 0.764, 0.254],
    [0.995, 0.909, 0.142],
];

// Blues: white → dark blue (sequential single-hue)
static BLUES: [[f32; 3]; 5] = [
    [0.969, 0.984, 1.000],
    [0.709, 0.855, 0.941],
    [0.420, 0.682, 0.839],
    [0.129, 0.443, 0.710],
    [0.032, 0.188, 0.420],
];

// Greens: white → dark green (sequential single-hue)
static GREENS: [[f32; 3]; 5] = [
    [0.969, 0.988, 0.961],
    [0.698, 0.867, 0.639],
    [0.412, 0.741, 0.388],
    [0.137, 0.545, 0.271],
    [0.000, 0.267, 0.106],
];

// Reds: white → dark red (sequential single-hue)
static REDS: [[f32; 3]; 5] = [
    [1.000, 0.961, 0.941],
    [0.988, 0.733, 0.631],
    [0.988, 0.416, 0.290],
    [0.796, 0.094, 0.114],
    [0.404, 0.000, 0.051],
];

pub const COLORMAP_NAMES: &[&str] = &[
    "viridis", "plasma", "inferno", "magma", "coolwarm", "hot", "gray",
    "turbo", "cividis", "blues", "greens", "reds",
];
