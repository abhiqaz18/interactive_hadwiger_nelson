# Hadwiger–Nelson GUI — Developer Guide

This document targets developers extending or customizing the interactive GUI in `hadwiger_nelson_gui.py`.

## Overview

- Entrypoint: `python -m hadwiger_nelson_gui`
- Core modules:
  - `GraphModel`: nodes/edges, serialization, geometry helpers
  - `ChromaticSolver`: DSATUR + backtracking coloring
  - `HNApp`: GUI, layout, exact solving, stochastic optimize (Muon), templates
- Templates saved under `../hn_graphs/` as JSON

## Architecture

- `HNApp.__init__()` sets up:
  - Matplotlib figure and main axes (left)
  - Right-side control panel (two columns)
  - Widgets via `_build_widgets()`
  - Event bindings (click/scroll/key)
  - Timer for dynamic layout `_on_timer()` that calls `_layout_step()`
- Layout/back-end functions:
  - `_spring_repulsion_solve()`: static solver (gradient descent on energy)
  - `_solve_least_squares()`: SciPy exact unit-edge solve with small anchors
  - `_stochastic_optimize()`: Muon optimizer with temperature schedule (T0→T1)

## Customization knobs

- Solver parameters are collected in `@dataclass SolverParams`:
  - `spring_k`: spring strength for unit edges
  - `repulsion_c`: repulsion coefficient (non-edges only)
  - `step`: step size for static solver
  - `max_iter`: iterations in static solver
  - `tol`: layout tolerance used by timer settle check and solvers
  - `restarts`: static solver restarts for robustness

- Dynamic layout (`HNApp` fields):
  - `_dyn_step`: per-step integration step in `_layout_step()`
  - Velocity buffer `self.vel` and damping inside `_layout_step()` (see code)
  - `_on_timer()`: number of steps per tick; settle logic when `err < tol`

- Exact solver (SciPy):
  - Tight tolerances, residuals only for unit edges and small anchors
  - No repulsion or symmetry-bias terms

- Stochastic/Muon:
  - UI fields `lr`, `steps`, `T0`, `T1`
  - Only Muon uses annealing (noise amplitude from `T0` to `T1`)

## UI layout and styling

- Two columns in `_build_widgets()`:
  - Left: layout/physics (Solve, Exact, Muon, Shake, Freeze, rep, Check)
  - Right: colors/templates (Color, Clear, save/insert/List/New)
- Column outlines drawn by `_add_group_box()` (transparent, behind widgets)
- Column headers drawn by `_add_column_header()`
- Compact labels and fonts set by `_shrink_widget_fonts()`

### Adjusting layout metrics

In `_build_widgets()` tune these figure-relative values:
- `right1`, `right2`: x positions for the two columns
- `bw`, `bh`: button/textbox width and height
- `pad`: vertical spacing
- You can also change figure size and main axes position in `HNApp.__init__()`:
  - `plt.subplots(figsize=(10.5, 6))`
  - `self.ax.set_position([0.06, 0.06, 0.66, 0.88])`

### Adding a new control

1. Decide column (`right1` or `right2`) and insert an axes slot:
   ```python
   self.ax_mybtn = self.fig.add_axes([right1, y1, bw, bh]); y1 -= (bh + pad)
   ```
2. Create the widget and wire it:
   ```python
   self.btn_my = Button(self.ax_mybtn, "My Action")
   self.btn_my.on_clicked(self.on_my_action)
   ```
3. Implement `on_my_action(self, event)` in `HNApp`.

### Styling tips

- Use short labels to avoid overlap, e.g., `lr`, `T0`, `T1`, `rep`.
- Use `_shrink_widget_fonts()` or customize per widget with `widget.label.set_fontsize(8)`.
- Keep group boxes at deep negative z-order to avoid covering widgets.

## Repulsion behavior

- Global repulsion annealing was removed. Repulsion is constant and can be changed live via the **rep** TextBox.
- Exact solver uses no repulsion; it only enforces unit edges with small anchor constraints.
- Muon is the only component that anneals (noise temperature schedule).

## Templates (I/O)

- Save: TextBox `save` + button "Save Template"
- Insert: TextBox `insert` + button "Insert Template"
- Files are written/read as `../hn_graphs/<name>.json`

## Extending the solver

- To change energies/forces in the static solver, see `_spring_repulsion_solve()`:
  - Spring energy ~ `(||d|| - 1)^2` on edges
  - Repulsion ~ `c / ||d||` on non-edge pairs
  - Gradient accumulation uses `np.add.at` for scatter-add into force buffer
- Consider adding cutoff radii or spatial hashing for large graphs.

## Debugging tips

- Add temporary `print()` or status messages via `_set_status()` to the on-canvas status box.
- Use fewer steps per tick in `_on_timer()` if UI lags.
- If SciPy solve fails to converge, try "Shake Layout" then run it again.

## Roadmap ideas

- Optional drag-to-move nodes; rubber-band edge drawing
- Better large-graph scalability (neighbor lists, spatial index)
- Export images/SVG; richer template metadata
- Alternative optimizers (LBFGS), exact distance-geometry formulations
