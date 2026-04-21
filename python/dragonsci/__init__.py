"""dragonsci — GPU-accelerated 3-D scatter plot widget for Tkinter.

Quick start
-----------
::

    import tkinter as tk
    import numpy as np
    from dragonsci import Scatter3D

    root = tk.Tk()
    root.title("My Point Cloud")

    widget = Scatter3D(root, width=900, height=700)
    widget.pack(fill="both", expand=True)

    rng = np.random.default_rng(0)
    pts = rng.standard_normal((250_000, 3)).astype(np.float32)
    widget.set_points(pts, colormap="plasma")

    root.mainloop()
"""

from .widget import Scatter3D, Scatter2D, link_cameras, unlink_cameras
from .figure import Figure

__all__ = ["Scatter3D", "Scatter2D", "Figure", "link_cameras", "unlink_cameras", "scatter3d"]

try:
    from .jupyter_widget import JupyterScatter3D, JupyterScatter2D
    __all__ += ["JupyterScatter3D", "JupyterScatter2D"]
except ImportError:
    pass


def scatter3d(**kwargs) -> "JupyterScatter3D":
    """Return a :class:`JupyterScatter3D` when called inside a Jupyter kernel.

    This is the recommended entry point for notebook usage::

        from dragonsci import scatter3d
        import numpy as np

        w = scatter3d(width=800, height=600)
        w.set_points(np.random.randn(50_000, 3).astype("f4"), colormap="viridis")
        w  # display in cell

    Raises :exc:`RuntimeError` when called outside a Jupyter kernel.  In
    Tkinter scripts, construct :class:`Scatter3D` directly instead.
    """
    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        ip = None
    if ip is None or ip.__class__.__name__ != "ZMQInteractiveShell":
        raise RuntimeError(
            "scatter3d() must be called inside a Jupyter kernel.  "
            "In a plain IPython terminal or regular Python script, "
            "use Scatter3D(master, ...) for Tkinter or construct "
            "JupyterScatter3D(...) directly."
        )
    from .jupyter_widget import JupyterScatter3D
    return JupyterScatter3D(**kwargs)

try:
    from importlib.metadata import version
    __version__ = version("dragonsci")
except Exception:
    __version__ = "unknown"
