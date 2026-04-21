# dragonsci

GPU-accelerated scientific visualization for Python, written in Rust.

Renders large point clouds (250 k+ points) at interactive frame rates by using
[wgpu](https://github.com/gfx-rs/wgpu) (Vulkan / Metal / DirectX 12) via
[PyO3](https://github.com/PyO3/pyo3). The widget embeds directly in any Tkinter
layout — no separate window or subprocess.

## Installation

**Tkinter** (default):
```bash
pip install dragonsci
```

**JupyterLab / Jupyter Notebook**:
```bash
pip install "dragonsci[notebook]"
```

The `[notebook]` extra adds `jupyter_rfb`, `ipywidgets`, and `Pillow`.

Pre-built wheels will be published to PyPI for Python 3.9–3.12 on Windows,
macOS, and Linux once a CI release pipeline is in place. Until then, install
from source (see **Building from source** below).

## GitHub release publishing

This repo now includes automated PyPI publishing through
[release.yml](.github/workflows/release.yml).
The workflow builds:

- a source distribution
- Linux `x86_64` manylinux wheels
- Windows `x64` wheels
- macOS `x86_64` and `arm64` wheels

Publishing uses PyPI Trusted Publishing. Before the first release, configure
`dragonsci` on PyPI to trust this exact GitHub Actions workflow:

1. Repository owner: `NKocur`
2. Repository name: `dragonsci`
3. Workflow path: `.github/workflows/release.yml`

After that, publishing a GitHub Release will build artifacts and upload them to
PyPI automatically. You can also run the same workflow manually from GitHub
Actions via `workflow_dispatch`; leave `publish` unchecked for a build-only dry
run, or enable it to upload to PyPI.

## Quick start

### Tkinter

```python
import tkinter as tk
import numpy as np
from dragonsci import Scatter3D

root = tk.Tk()
widget = Scatter3D(root, width=900, height=700)
widget.pack(fill="both", expand=True)

pts = np.random.default_rng(0).standard_normal((250_000, 3)).astype(np.float32)
widget.set_points(pts, colormap="plasma")

root.mainloop()
```

### JupyterLab / Jupyter Notebook

```python
from dragonsci import scatter3d   # requires dragonsci[notebook]
import numpy as np

w = scatter3d(width=800, height=600)
w.set_points(np.random.randn(50_000, 3).astype("f4"), colormap="viridis")
w  # display in cell
```

For 2-D scatter plots pass `(N, 2)` or `(N, 3)` arrays to `JupyterScatter2D`:

```python
from dragonsci import JupyterScatter2D
import numpy as np

w = JupyterScatter2D(width=700, height=500)
w.set_points(np.random.randn(20_000, 2).astype("f4"), colormap="plasma")
w
```

Point picking is available via callback:

```python
def on_hit(result):
    print(result["actor"], result["index"], result["point"])

w.enable_point_picking(on_pick=on_hit)
```

## API

### Widget construction

```python
Scatter3D(master, *, width=800, height=600, fps=60, vsync=True, **kwargs)
```

A `tk.Frame` subclass. All standard frame keyword arguments are forwarded.

**Pre-map behaviour:** The following may be called before `mainloop()` and are
queued, then replayed when the renderer initialises on widget map:
`set_points`, `add_points`, `add_lines`, `add_box`, `scalar_bar`,
`show_orientation_axes`, `point_style`, `show_grid`, `set_background`,
`set_axes`, `set_ticks`, and the `parallel_projection` property.

Camera methods (`reset_camera`, `view_*`, `fit`, `get_camera`, `set_camera`)
and overlay-mutation methods (`update_lines`, `remove_overlay`,
`set_overlay_visibility`) are silent no-ops before the widget is mapped.

---

### Point clouds

```python
widget.set_points(positions, *, colors=None, scalars=None, colormap="viridis",
                  point_size=4.0, clim=None, nan_color=(0.3,0.3,0.3),
                  log_scale=False, opacity=1.0)
```

Replace all point actors with a single point cloud. Line overlays are not affected; call `clear()` first to reset the whole scene.

| Parameter | Type | Description |
|-----------|------|-------------|
| `positions` | `(N, 3) float32` | XYZ coordinates |
| `colors` | `(N, 3) float32`, optional | Per-point RGB in \[0, 1\]. Takes priority over `scalars`. |
| `scalars` | `(N,) float32`, optional | Mapped through `colormap`. Used when `colors` is absent. |
| `colormap` | `str` | One of the names returned by `colormap_names()`. |
| `point_size` | `float` | Point diameter in pixels (default 4). |
| `clim` | `(float, float)`, optional | Colormap range. Defaults to data min/max. |
| `nan_color` | `(r, g, b)` | Colour for NaN scalars (default dark grey). |
| `log_scale` | `bool` | Apply log₁₀ normalisation before colouring. |
| `opacity` | `float` | Per-call alpha, \[0, 1\] (default 1.0). |

```python
handle = widget.add_points(positions, *, ...)   # same kwargs as set_points
widget.update_actor(handle, positions, *, ...)  # replace in-place
widget.set_actor_visibility(handle, visible)
widget.remove_actor(handle)
widget.clear()                                   # remove all actors, overlays, and grid
```

---

### Rendering style

```python
widget.point_style = "circle"   # "circle" | "square" | "gaussian"
widget.set_ticks(x=None, y=None, z=None)
```

---

### Camera

```python
widget.reset_camera()
widget.fit(bounds=None)          # (xmin,ymin,zmin,xmax,ymax,zmax) or re-fit data
widget.view_xy(); widget.view_xz(); widget.view_yz(); widget.view_isometric()
widget.parallel_projection       # bool property
state = widget.get_camera()
widget.set_camera(state)
```

---

### Axis labels, grid and background

```python
widget.set_axes(x="X", y="Y", z="Z")   # axis title labels at grid extents
widget.show_grid(visible=True)          # toggle grid lines + tick labels
widget.set_background(color)            # (r,g,b) tuple or "#RRGGBB" hex string
```

---

### Scalar bar

```python
widget.scalar_bar(visible=True, *, vmin=0.0, vmax=1.0, log_scale=False,
                  colormap="viridis", title="")
```

---

### Line overlays

Segments are `(N, 6) float32` arrays, each row `[x0,y0,z0, x1,y1,z1]`.
Overlay bounds are included in camera fitting and grid extents.

```python
handle = widget.add_lines(segments, color=(1,1,1))
handle = widget.add_box(bounds, color=(1,1,0))   # (xmin,ymin,zmin,xmax,ymax,zmax)
widget.update_lines(handle, segments, color)
widget.set_overlay_visibility(handle, visible)
widget.remove_overlay(handle)
widget.clear_overlays()
widget.show_orientation_axes(visible=True)
```

---

### Picking

```python
widget.enable_point_picking(on_pick=None)      # fires <<PointPicked>>
widget.enable_rectangle_picking(on_select=None) # Shift+drag; fires <<SelectionChanged>>
widget.disable_picking()
# After <<PointPicked>>:  widget.picked_point, .picked_index, .picked_actor
# After <<SelectionChanged>>: widget.selected  →  [{"actor":int,"index":int},...]
```

---

### Animation export

```python
widget.open_gif(path, fps=20, loop=0)
widget.write_frame()
widget.close_gif()
# or all-in-one orbit:
widget.orbit_gif(path, n_frames=60, fps=20, loop=0, elevation=0.3, on_progress=None)
```

---

### Screenshot

```python
rgba = widget.screenshot()   # (H, W, 4) uint8 ndarray, or None before map
widget.save_png(path)
```

---

### Linked cameras

```python
from dragonsci import link_cameras, unlink_cameras
link_cameras(w1, w2, w3, ...)
unlink_cameras(w1, w2)
```

---

### Misc

```python
Scatter3D.colormap_names() → list[str]
```

## Camera controls

| Input | Action |
|-------|--------|
| Left-drag | Orbit |
| Shift + left-drag | Pan |
| Middle-drag | Pan |
| Scroll wheel | Zoom |
| Double left-click | Reset camera |

## Platform notes

### Windows

Fully supported. Uses DirectX 12 or Vulkan via wgpu. No extra dependencies.

### macOS

Fully supported. Uses Metal via wgpu. No extra dependencies.

### Linux

Requires an **X11/Xlib display** (Wayland is not supported). The widget uses
`winfo_id()` to obtain an XID and opens the display via `libX11.so.6`. Pure
Wayland sessions will fail at renderer initialisation with a warning. Running
under XWayland works.

```bash
# Confirm X11 is available:
echo $DISPLAY   # should print e.g. :0
```

## Building from source

Requires Rust (stable) and maturin ≥ 1.7.

**Development** (installs the extension in-place for local imports):

```bash
pip install maturin
maturin develop --release
pytest tests/
```

**Distribution wheel** — use the helper script, which removes the
`maturin develop` artifact from the source tree before building. Running
`maturin build` directly from a development working tree will fail because
the in-place extension collides with the freshly-compiled one:

```bash
python scripts/build_wheel.py           # produces target/wheels/*.whl
# or equivalently:
rm -f python/dragonsci/_dragonsci*.{pyd,so,dylib}
maturin build --release
```

## License

MIT
