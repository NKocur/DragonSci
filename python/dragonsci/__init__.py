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

__all__ = ["Scatter3D", "Scatter2D", "link_cameras", "unlink_cameras"]

try:
    from importlib.metadata import version
    __version__ = version("dragonsci")
except Exception:
    __version__ = "unknown"
