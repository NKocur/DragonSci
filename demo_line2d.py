"""Interactive demo for Line2D — the dedicated 2-D chart widget.

The streaming line uses two incommensurate frequencies so the pattern
never looks stationary.  The x-window grows from 0 → WINDOW, then slides.
"""

import tkinter as tk
import numpy as np
from dragonsci import Line2D

WINDOW = 8.0   # seconds of history shown
RATE   = 30    # Hz

root = tk.Tk()
root.title("dragonsci — Line2D streaming demo")
root.geometry("900x600")
root.configure(bg="#111")

chart = Line2D(root, width=900, height=600)
chart.pack(fill="both", expand=True)

chart.set_xlabel("Time (s)")
chart.set_ylabel("Amplitude")

# Pre-set axis limits so the chart frame appears before data arrives.
chart.set_xlim(0.0, WINDOW)
chart.set_ylim(-1.5, 1.5)

stream = chart.add_line_stream(
    max_points=int(WINDOW * RATE * 2),
    mode="ring",
    color=(0.3, 1.0, 0.5),
)

_t = [0.0]

def tick():
    dt = 1.0 / RATE
    t0 = _t[0]
    # Two incommensurate frequencies → never looks stationary
    ts = np.array([t0, t0 + dt], dtype=np.float32)
    ys = (0.7 * np.sin(ts * 2.1) + 0.3 * np.sin(ts * 5.3)).astype(np.float32)
    _t[0] += 2 * dt

    chart.stream_line(stream, ts, ys)

    t_now = _t[0]
    # Grow window for the first WINDOW seconds, then slide it.
    x_lo = max(0.0, t_now - WINDOW)
    x_hi = max(WINDOW, t_now)
    chart.set_xlim(x_lo, x_hi)

    root.after(33, tick)

root.after(200, tick)
root.mainloop()
