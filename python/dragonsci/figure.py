"""Figure — multi-subplot grid of Scatter3D widgets."""
from __future__ import annotations

import tkinter as tk

from .widget import Scatter3D, link_cameras


class Figure(tk.Frame):
    """A grid of :class:`Scatter3D` subplots inside a single Tkinter frame.

    Parameters
    ----------
    master : tk.Misc
        Tkinter parent widget.
    rows, cols : int
        Grid dimensions.
    width, height : int
        Total pixel size of the figure (distributed evenly across cells).
    padding : int
        Pixel gap between subplot cells.
    equal_aspect : bool
        Force cells to be square (uses the smaller cell dimension for both axes).
    share_cameras : bool
        Link cameras of all subplots together on creation.
    **kwargs
        Forwarded to each ``Scatter3D`` constructor (e.g. ``bg=``, ``fps=``).
    """

    def __init__(
        self,
        master: tk.Misc,
        rows: int = 1,
        cols: int = 1,
        width: int = 800,
        height: int = 600,
        padding: int = 4,
        equal_aspect: bool = False,
        share_cameras: bool = False,
        **kwargs,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError(
                f"rows and cols must each be >= 1, got rows={rows}, cols={cols}"
            )
        super().__init__(master)
        self._rows = rows
        self._cols = cols
        self._padding = padding
        self._equal_aspect = equal_aspect
        self._scatter_kwargs = kwargs

        cell_w = max(1, (width - padding * (cols + 1)) // cols)
        cell_h = max(1, (height - padding * (rows + 1)) // rows)
        if equal_aspect:
            cell_w = cell_h = min(cell_w, cell_h)

        # equal_aspect: sticky="" keeps the widget at its requested size (the grid
        # cell may be larger, but the widget won't stretch to fill it).
        # Without equal_aspect: sticky="nsew" expands widgets to fill cells.
        sticky = "" if equal_aspect else "nsew"

        self._axes: list[list[Scatter3D]] = []
        for r in range(rows):
            row_list: list[Scatter3D] = []
            for c in range(cols):
                w = Scatter3D(self, width=cell_w, height=cell_h, **kwargs)
                w.grid(row=r, column=c, padx=padding // 2, pady=padding // 2,
                       sticky=sticky)
                row_list.append(w)
            self._axes.append(row_list)

        for r in range(rows):
            self.rowconfigure(r, weight=1)
        for c in range(cols):
            self.columnconfigure(c, weight=1)

        if share_cameras:
            flat = self.axes
            if len(flat) > 1:
                link_cameras(*flat)

        self.bind("<Configure>", self._on_resize)
        self._last_size = (width, height)

    # ── Subplot access ─────────────────────────────────────────────────────────

    def __getitem__(self, key: tuple[int, int]) -> Scatter3D:
        """Return the :class:`Scatter3D` at ``(row, col)``."""
        r, c = key
        return self._axes[r][c]

    @property
    def axes(self) -> list[Scatter3D]:
        """Flat list of all subplots in row-major order."""
        return [w for row in self._axes for w in row]

    # ── Camera linking ─────────────────────────────────────────────────────────

    def link_cameras(self, *cell_coords: tuple[int, int]) -> None:
        """Link cameras of the given subplot cells together.

        Parameters
        ----------
        *cell_coords
            Pairs ``(row, col)`` identifying the cells to link.
        """
        seen: set[int] = set()
        widgets = []
        for r, c in cell_coords:
            w = self._axes[r][c]
            if id(w) not in seen:
                seen.add(id(w))
                widgets.append(w)
        if len(widgets) >= 2:
            link_cameras(*widgets)

    # ── Scalar bar ─────────────────────────────────────────────────────────────

    def scalar_bar(
        self,
        row: "int | None" = None,
        col: "int | None" = None,
        *,
        vmin: float = 0.0,
        vmax: float = 1.0,
        log_scale: bool = False,
        colormap: str = "viridis",
        title: str = "",
    ) -> None:
        """Show a scalar bar in one cell per row/col and hide it from the rest.

        v1 workaround: the rightmost cell in each targeted row (or bottom cell
        of each targeted column) renders the bar; all others in the same
        row/column have it hidden.  This makes the bar appear shared without
        requiring a figure-level overlay.

        Parameters
        ----------
        row : int or None
            Target row.  ``None`` with ``col=None`` targets all rows.
        col : int or None
            Target column.  ``None`` with ``row=None`` targets all rows.
        vmin, vmax, log_scale, colormap, title
            Forwarded to each cell's :meth:`Scatter3D.scalar_bar`.
        """
        kw = dict(vmin=vmin, vmax=vmax, log_scale=log_scale,
                  colormap=colormap, title=title)

        # Build the set of cells that should show the bar.
        if row is None and col is None:
            visible: set[tuple[int, int]] = {
                (r, self._cols - 1) for r in range(self._rows)
            }
        elif row is not None and col is None:
            visible = {(row, self._cols - 1)}
        elif col is not None and row is None:
            visible = {(self._rows - 1, col)}
        else:
            visible = {(row, col)}

        # Always touch every cell so stale bars from previous calls are cleared.
        for r in range(self._rows):
            for c in range(self._cols):
                self._axes[r][c].scalar_bar((r, c) in visible, **kw)

    # ── Resize ─────────────────────────────────────────────────────────────────

    def _on_resize(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        new_w, new_h = event.width, event.height
        if (new_w, new_h) == self._last_size:
            return
        self._last_size = (new_w, new_h)

        cell_w = max(1, (new_w - self._padding * (self._cols + 1)) // self._cols)
        cell_h = max(1, (new_h - self._padding * (self._rows + 1)) // self._rows)
        if self._equal_aspect:
            cell_w = cell_h = min(cell_w, cell_h)

        for row in self._axes:
            for w in row:
                w.configure(width=cell_w, height=cell_h)
