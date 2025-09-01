# Hadwiger–Nelson Interactive GUI — User Guide

This tool lets you build and explore graphs for the Hadwiger–Nelson problem interactively.
File: `hadwiger_nelson_gui.py`
Run from repo root:

```bash
python -m hadwiger_nelson_gui
```

## Core interactions

- **Add point**: Left-click empty space in the plane to add a point at the cursor.
- **Add unit-length edge**: Left-click a point to select it, then left-click another point to add a unit edge between them. The layout will relax and pull/push points toward unit length while keeping everything well-spaced.
- **Select/Clear selection**: Click a point to select; click it again to clear.
- **Scroll to zoom**: Use mouse wheel over the plane to zoom in/out near the cursor.
- **Auto zoom**: The view auto-fits after edits and occasionally during relaxation.

## Buttons and actions (right panel)

Two columns with simple group outlines:

- Left column (Layout / Physics)
  - **Solve Layout**: Starts continuous relaxation (springs on unit edges + constant repulsion on non-edges).
  - **Exact Solve (SciPy)**: Nonlinear least squares enforcing unit edge lengths with light anchors. No repulsion or side-bias.
  - **Muon Optimize** with fields: `lr`, `steps`, `T0`, `T1` (only Muon uses annealing of noise).
  - **Shake Layout** + `amp`: Adds small random jitter and restarts layout.
  - **Freeze** toggle: Temporarily stops dynamics and edits.
  - **rep**: TextBox to set repulsion coefficient live.
  - **Check Coincident**: Reports nearly coincident node pairs.

- Right column (Colors / Templates)
  - **Color Graph**: Compute coloring (DSATUR + backtracking) and color nodes.
  - **Clear Colors**: Remove coloring.
  - **save / Save Template**: Save current graph as `../hn_graphs/<name>.json`.
  - **insert / Insert Template**: Insert a saved template at the current scene.
  - **List Templates**: List available templates.
  - **New Graph**: Start a fresh graph.

## Node deletion

- **Right-click a node** to delete it immediately (removes incident edges and reindexes remaining nodes).
- **Delete/Backspace**: Deletes the currently selected node.

## Tips for good layouts

- **Add a first edge** early: The app rescales the graph so the first unit edge has length 1, giving a good starting scale for subsequent edges.
- **If nodes cluster too closely**: Click "Solve Layout" or increase repulsion (ask us to tune `repulsion_c`).
- **If the graph seems stuck**: Use "Shake Layout" or "Exact Solve (SciPy)" to escape local minima.
- **Performance**: For very large graphs, real-time relaxation may slow down. Consider pausing interactions while solving or reduce the number of relaxation steps per timer tick.

## Behind the scenes (short)

- **Relaxation**: Velocity-based integration using springs on unit edges and constant repulsion on non-edge pairs (no global annealing). Edges are excluded from repulsion to avoid fighting constraints. Damping is applied to avoid oscillations.
- **Exact solve**: `scipy.optimize.least_squares` with residuals that enforce unit edge lengths and light anchoring of a couple of nodes (no repulsion, no side-bias).
- **Muon optimize**: Stochastic optimizer with a temperature schedule from `T0` to `T1` controlling noise amplitude during the run.

## Known limitations

- **Exact global feasibility** is not guaranteed (some unit-distance constraints may be impossible in 2D). The app reports when it can’t satisfy constraints to tolerance.
- **Chromatic number**: Exact search is exponential; practical for small graphs.

## Customization knobs (quick)

- UI: Adjust repulsion live via the **rep** TextBox.
- In code: `SolverParams.spring_k`, `repulsion_c`, `step`, `max_iter`, `tol`, `restarts`.
- Dynamics: `_on_timer()` calls `_layout_step()` multiple times per tick for smoothness.
- Exact solve: tight tolerances, unit-edge + small anchors; no repulsion or side-bias.

## File layout

- Main app: `hadwiger_nelson_gui.py`
- Templates: `../hn_graphs/` (JSON)

If you want more features (dragging nodes, Qt/GL UI, custom constraint sets), let us know!
