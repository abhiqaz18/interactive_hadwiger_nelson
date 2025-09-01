"""
Hadwiger–Nelson interactive GUI

Features (no extra dependencies beyond matplotlib, numpy, scipy in requirements):
- Left click on empty space to add a new point (untethered).
- Left click on an existing point then another to add a unit-length edge.
  If constraints cannot be satisfied, the edge is reverted and a message is shown.
- Save the current graph as a named template and insert templates later.
- Compute the chromatic number (exact for small graphs) and color nodes.
- Non-rigid graphs are laid out with spring (unit-length) + repulsion forces to keep points well-spaced.
- Auto-zoom adjusts the view as the graph grows.

Notes:
- This uses a simple spring/repulsion iterative solver. It is not a rigorous geometric constraint solver,
  but is robust for interactive use. If it cannot satisfy constraints to a tolerance after several restarts,
  it reports impossibility.
- Chromatic number search uses DSATUR with backtracking. Works well for small graphs.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, TextBox
from matplotlib.patches import FancyBboxPatch
# from matplotlib.patches import FancyBboxPatch  # removed to simplify UI (no group boxes)
from scipy.optimize import least_squares


# ---------------------------- Graph Model ---------------------------- #


class GraphModel:
    def __init__(self) -> None:
        self.pos: np.ndarray = np.zeros((0, 2), dtype=float)  # shape (N,2)
        self.edges: Set[Tuple[int, int]] = set()  # undirected edges with i<j

    # ---- basic ops ---- #
    def add_node(self, x: float, y: float) -> int:
        if self.pos.size == 0:
            self.pos = np.array([[x, y]], dtype=float)
        else:
            self.pos = np.vstack([self.pos, [x, y]])
        return self.pos.shape[0] - 1

    def remove_node(self, i: int) -> None:
        n = self.n_nodes
        if not (0 <= i < n):
            return
        # remove edges incident to i
        self.edges = {(u, v) for (u, v) in self.edges if u != i and v != i}
        # reindex remaining edges after removing node
        mapping = {}
        idx = 0
        for k in range(n):
            if k == i:
                continue
            mapping[k] = idx
            idx += 1
        self.edges = {
            (min(mapping[u], mapping[v]), max(mapping[u], mapping[v])) for (u, v) in self.edges
        }
        # remove position
        self.pos = np.delete(self.pos, i, axis=0)

    def add_edge(self, i: int, j: int) -> bool:
        if i == j:
            return False
        if i > j:
            i, j = j, i
        if not (0 <= i < self.n_nodes and 0 <= j < self.n_nodes):
            return False
        if (i, j) in self.edges:
            return False
        self.edges.add((i, j))
        return True

    def remove_edge(self, i: int, j: int) -> None:
        if i > j:
            i, j = j, i
        self.edges.discard((i, j))

    @property
    def n_nodes(self) -> int:
        return self.pos.shape[0]

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def adjacency_lists(self) -> List[List[int]]:
        n = self.n_nodes
        adj = [[] for _ in range(n)]
        for (i, j) in self.edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def edges_array(self) -> np.ndarray:
        if not self.edges:
            return np.zeros((0, 2), dtype=int)
        arr = np.array(sorted(list(self.edges)), dtype=int)
        return arr

    # ---- geometry helpers ---- #
    def bounding_box(self) -> Tuple[float, float, float, float]:
        if self.n_nodes == 0:
            return -1.0, 1.0, -1.0, 1.0
        xs = self.pos[:, 0]
        ys = self.pos[:, 1]
        return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())

    def center(self) -> Tuple[float, float]:
        if self.n_nodes == 0:
            return 0.0, 0.0
        c = self.pos.mean(axis=0)
        return float(c[0]), float(c[1])

    def translate(self, dx: float, dy: float) -> None:
        if self.n_nodes:
            self.pos += np.array([dx, dy], dtype=float)

    def scale(self, s: float) -> None:
        if self.n_nodes:
            self.pos *= float(s)

    def normalize_for_template(self) -> None:
        # Center at origin and scale so mean edge length is 1 (if edges). Otherwise keep as-is.
        if self.n_nodes == 0:
            return
        cx, cy = self.center()
        self.translate(-cx, -cy)
        if self.n_edges > 0:
            e = self.edges_array()
            if e.shape[0] > 0:
                d = np.linalg.norm(self.pos[e[:, 0]] - self.pos[e[:, 1]], axis=1)
                mean_len = float(np.mean(d)) if d.size else 1.0
                if mean_len > 1e-12:
                    self.scale(1.0 / mean_len)

    # ---- serialization ---- #
    def to_dict(self) -> Dict:
        return {
            "nodes": self.pos.tolist(),
            "edges": [list(e) for e in sorted(list(self.edges))],
        }

    @staticmethod
    def from_dict(data: Dict) -> "GraphModel":
        g = GraphModel()
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        if nodes:
            g.pos = np.array(nodes, dtype=float)
        for (i, j) in edges:
            if i > j:
                i, j = j, i
            g.edges.add((int(i), int(j)))
        return g


# ---------------------------- Chromatic Number Solver ---------------------------- #


class ChromaticSolver:
    def __init__(self, adj: List[List[int]]):
        self.adj = [list(nei) for nei in adj]
        self.n = len(adj)
        self.deg = [len(nei) for nei in self.adj]

    def greedy_dsatur_upper_bound(self) -> Tuple[int, List[int]]:
        n = self.n
        colors = [-1] * n
        # DSATUR heuristic ordering by saturation, tie by degree
        sat = [0] * n
        colored = 0
        used_colors = 0
        neighbor_colors: List[Set[int]] = [set() for _ in range(n)]
        while colored < n:
            # choose vertex with max saturation (ties by degree)
            candidates = [i for i in range(n) if colors[i] == -1]
            i = max(candidates, key=lambda v: (len(neighbor_colors[v]), self.deg[v]))
            unavailable = neighbor_colors[i]
            c = 0
            while c in unavailable:
                c += 1
            colors[i] = c
            used_colors = max(used_colors, c + 1)
            colored += 1
            for j in self.adj[i]:
                if colors[j] == -1:
                    neighbor_colors[j].add(c)
        return used_colors, colors

    def is_k_colorable(self, k: int) -> Tuple[bool, Optional[List[int]]]:
        n = self.n
        colors = [-1] * n
        neighbor_colors: List[Set[int]] = [set() for _ in range(n)]

        def choose_next() -> int:
            # DSATUR choice: max saturation, then degree
            candidates = [i for i in range(n) if colors[i] == -1]
            return max(candidates, key=lambda v: (len(neighbor_colors[v]), self.deg[v]))

        def backtrack(colored_count: int) -> bool:
            if colored_count == n:
                return True
            v = choose_next()
            unavailable = neighbor_colors[v]
            # try available colors, smallest first
            for c in range(k):
                if c in unavailable:
                    continue
                colors[v] = c
                changed = []
                for u in self.adj[v]:
                    if colors[u] == -1 and c not in neighbor_colors[u]:
                        neighbor_colors[u].add(c)
                        changed.append(u)
                if backtrack(colored_count + 1):
                    return True
                # undo
                for u in changed:
                    neighbor_colors[u].remove(c)
                colors[v] = -1
            return False

        ok = backtrack(0)
        return ok, (colors if ok else None)

    def chromatic_number(self) -> Tuple[int, List[int]]:
        ub, greedy_colors = self.greedy_dsatur_upper_bound()
        # trivial lower bound: max clique size is hard; use max degree lower bound of 1 + min? No.
        # We'll use a simple bound: max clique approx via greedy heuristic
        lb = self._greedy_clique_lower_bound()
        for k in range(max(1, lb), ub + 1):
            ok, col = self.is_k_colorable(k)
            if ok and col is not None:
                return k, col
        # Fallback
        return ub, greedy_colors

    def _greedy_clique_lower_bound(self) -> int:
        # heuristic: repeatedly pick vertex with highest degree, intersect neighborhoods
        n = self.n
        if n == 0:
            return 0
        remaining = set(range(n))
        best = 1
        # try a few random starts for robustness
        starts = list(range(n))
        random.shuffle(starts)
        starts = starts[: min(10, n)]
        for s in starts:
            clique = [s]
            cand = set(self.adj[s])
            while cand:
                v = max(cand, key=lambda x: self.deg[x])
                clique.append(v)
                cand = cand.intersection(self.adj[v])
            best = max(best, len(clique))
        return best


# ---------------------------- Interactive App ---------------------------- #


@dataclass
class SolverParams:
    spring_k: float = 6.0
    repulsion_c: float = 0.06
    step: float = 0.05
    max_iter: int = 1200
    tol: float = 1e-3
    restarts: int = 3


class HNApp:
    def __init__(self) -> None:
        self.model = GraphModel()
        self.fig, self.ax = plt.subplots(figsize=(10.5, 6))
        self.fig.canvas.manager.set_window_title("Hadwiger–Nelson interactive GUI")
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_title("Hadwiger–Nelson: click empty space to add point; click two points to add unit edge")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True, alpha=0.2)
        # Reserve right margin for controls to avoid overlap
        self.ax.set_position([0.06, 0.06, 0.66, 0.88])

        # plotting state
        self.node_size = 160
        self.pick_radius_px = 10  # pixels
        self.node_colors_rgba = np.array([[0.2, 0.6, 1.0, 1.0]])  # default color
        self.color_assignment: Optional[List[int]] = None
        self.selected_idx: Optional[int] = None

        # artists
        self.node_scatter = self.ax.scatter([], [], s=self.node_size, c=[self.node_colors_rgba[0]], edgecolors="k", zorder=3)
        self.edge_collection = LineCollection([], colors="k", linewidths=1.5, zorder=2)
        self.ax.add_collection(self.edge_collection)
        # selection highlight
        self.sel_scatter = self.ax.scatter([], [], s=self.node_size * 1.8, facecolors="none", edgecolors="orange", linewidths=2.0, zorder=4)
        # labels for node colors
        self.node_labels: List[plt.Text] = []

        # status text
        self.status_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va="top", ha="left", fontsize=9, color="black",
                                        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="#999999", alpha=0.8))

        # solver parameters (must be before building widgets that read params)
        self.params = SolverParams()

        # UI widgets on the right
        self._build_widgets()

        # events
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        # scroll zoom handler (optional)
        self.cid_scroll = self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        # key events for deletion
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # templates dir
        self.templates_dir = os.path.join(os.path.dirname(__file__), "hn_graphs")
        os.makedirs(self.templates_dir, exist_ok=True)

        # dynamic layout (timer-driven)
        self.layout_running: bool = False
        self._dyn_step: float = 0.05
        self._tick_count: int = 0
        self.timer = self.fig.canvas.new_timer(interval=20)
        self.timer.add_callback(self._on_timer)
        self.timer.start()
        self.vel = np.zeros_like(self.model.pos)
        self.settled_ticks: int = 0
        self.frozen: bool = False

        # initial limits
        self._auto_zoom()
        self._redraw()

    def on_stochastic_opt_clicked(self, _):
        if self.frozen:
            self._set_status("Frozen: unfreeze to run Muon Optimize.")
            return
        method = "Muon"
        try:
            steps = int(float(self.tb_steps.text.strip()))
        except Exception:
            steps = 1200
        try:
            lr = float(self.tb_lr.text.strip())
        except Exception:
            lr = 0.04
        try:
            t0 = float(self.tb_tstart.text.strip())
        except Exception:
            t0 = 0.04
        try:
            t1 = float(self.tb_tend.text.strip())
        except Exception:
            t1 = 0.0
        steps = max(1, steps)
        lr = max(1e-5, min(1.0, lr))
        t0 = max(0.0, min(1.0, t0))
        t1 = max(0.0, min(1.0, t1))
        ok = self._stochastic_optimize(method=method, steps=steps, lr=lr, t0=t0, t1=t1)
        if ok:
            self._set_status(f"Stochastic optimize ({method}) done.")
        else:
            self._set_status(f"Stochastic optimize ({method}) finished; constraints not fully satisfied.")
        self._start_layout()
        self._auto_zoom()
        self._redraw()

    # ---------------- UI ---------------- #
    def _build_widgets(self) -> None:
        # Two columns with group boxes (non-overlapping)
        right1 = 0.74
        right2 = 0.87
        pad = 0.006
        bw = 0.12
        bh = 0.035
        # Column 1: Solve/Optimize/Physics
        y1 = 0.92
        self.ax_solve = self.fig.add_axes([right1, y1, bw, bh]); y1 -= (bh + pad)
        self.ax_exact = self.fig.add_axes([right1, y1, bw, bh]); y1 -= (bh + pad)
        self.ax_stoch = self.fig.add_axes([right1, y1, bw, bh]); y1 -= (bh + pad)
        self.ax_opt_lr = self.fig.add_axes([right1, y1, bw, bh]); y1 -= (bh + pad)
        self.ax_opt_steps = self.fig.add_axes([right1, y1, bw, bh]); y1 -= (bh + pad)
        self.ax_opt_tstart = self.fig.add_axes([right1, y1, bw, bh]); y1 -= (bh + pad)
        self.ax_opt_tend = self.fig.add_axes([right1, y1, bw, bh]); y1 -= (bh + pad)
        self.ax_shake = self.fig.add_axes([right1, y1, bw, bh]); y1 -= (bh + pad)
        self.ax_shake_tb = self.fig.add_axes([right1, y1, bw, bh]); y1 -= (bh + pad)
        self.ax_freeze = self.fig.add_axes([right1, y1, bw, bh]); y1 -= (bh + pad)
        self.ax_rep_tb = self.fig.add_axes([right1, y1, bw, bh]); y1 -= (bh + pad)
        self.ax_check = self.fig.add_axes([right1, y1, bw, bh]); y1 -= (bh + pad)
        # Column 2: Colors/Templates/New
        y2 = 0.92
        self.ax_color = self.fig.add_axes([right2, y2, bw, bh]); y2 -= (bh + pad)
        self.ax_clear_color = self.fig.add_axes([right2, y2, bw, bh]); y2 -= (bh + pad)
        self.ax_save_tb = self.fig.add_axes([right2, y2, bw, bh]); y2 -= (bh + pad)
        self.ax_save_btn = self.fig.add_axes([right2, y2, bw, bh]); y2 -= (bh + pad)
        self.ax_ins_tb = self.fig.add_axes([right2, y2, bw, bh]); y2 -= (bh + pad)
        self.ax_ins_btn = self.fig.add_axes([right2, y2, bw, bh]); y2 -= (bh + pad)
        self.ax_list_btn = self.fig.add_axes([right2, y2, bw, bh]); y2 -= (bh + pad)
        self.ax_new = self.fig.add_axes([right2, y2, bw, bh]); y2 -= (bh + pad)
        # Group boxes (drawn behind; outline only, no label text)
        self._add_group_box(right1 - 0.005, y1, bw + 0.01, 0.92 - y1, "")
        self._add_group_box(right2 - 0.005, y2, bw + 0.01, 0.92 - y2, "")
        # Column headers
        self._add_column_header(right1 + bw * 0.5, 0.955, "Layout / Physics")
        self._add_column_header(right2 + bw * 0.5, 0.955, "Colors / Templates")

        # (Removed semi-transparent group boxes to avoid overlap/opacity issues)

        # Hook up widgets
        self.btn_color = Button(self.ax_color, "Color Graph", color="#e1efff", hovercolor="#cfe4ff")
        self.btn_color.on_clicked(self.on_color_clicked)

        self.btn_clear_color = Button(self.ax_clear_color, "Clear Colors", color="#f3f3f3", hovercolor="#e7e7e7")
        self.btn_clear_color.on_clicked(self.on_clear_colors)

        self.btn_solve = Button(self.ax_solve, "Solve Layout", color="#e8ffe8", hovercolor="#d7ffd7")
        self.btn_solve.on_clicked(self.on_solve_clicked)

        self.btn_exact = Button(self.ax_exact, "Exact Solve (SciPy)", color="#e6fff7", hovercolor="#d6ffef")
        self.btn_exact.on_clicked(self.on_exact_solve_clicked)

        # Muon optimizer widgets with sensible defaults (short labels)
        self.tb_lr = TextBox(self.ax_opt_lr, "lr", initial="0.04")
        self.tb_steps = TextBox(self.ax_opt_steps, "steps", initial="1200")
        self.tb_tstart = TextBox(self.ax_opt_tstart, "T0", initial="0.04")
        self.tb_tend = TextBox(self.ax_opt_tend, "T1", initial="0.0")
        self.btn_stoch = Button(self.ax_stoch, "Muon Optimize", color="#e8f7ff", hovercolor="#d9f1ff")
        self.btn_stoch.on_clicked(self.on_stochastic_opt_clicked)

        self.tb_shake = TextBox(self.ax_shake_tb, "amp", initial="0.05")
        self.btn_shake = Button(self.ax_shake, "Shake Layout", color="#fff6e6", hovercolor="#ffedd1")
        self.btn_shake.on_clicked(self.on_shake_clicked)

        # coincidence check
        self.btn_check = Button(self.ax_check, "Check Coincident", color="#ffe8f7", hovercolor="#ffd9f0")
        self.btn_check.on_clicked(self.on_check_coincident)

        self.btn_new = Button(self.ax_new, "New Graph", color="#fff2e6", hovercolor="#ffe4cc")
        self.btn_new.on_clicked(self.on_new_graph)

        self.tb_save = TextBox(self.ax_save_tb, "save", initial="my_template")
        self.btn_save = Button(self.ax_save_btn, "Save Template", color="#f0f0ff", hovercolor="#e5e5ff")
        self.btn_save.on_clicked(self.on_save_template)

        self.tb_insert = TextBox(self.ax_ins_tb, "insert", initial="my_template")
        self.btn_insert = Button(self.ax_ins_btn, "Insert Template", color="#f0fff0", hovercolor="#e3ffe3")
        self.btn_insert.on_clicked(self.on_insert_template)

        self.btn_list = Button(self.ax_list_btn, "List Templates", color="#f9f9f9", hovercolor="#ececec")
        self.btn_list.on_clicked(self.on_list_templates)

        # Freeze and repulsion controls
        self.btn_freeze = Button(self.ax_freeze, "Freeze: OFF", color="#ffecec", hovercolor="#ffdcdc")
        self.btn_freeze.on_clicked(self.on_toggle_freeze)
        self.tb_rep = TextBox(self.ax_rep_tb, "rep", initial=f"{self.params.repulsion_c}")
        self.tb_rep.on_submit(self.on_repulsion_changed)

        # shrink widget label fonts to avoid overlap
        self._shrink_widget_fonts()

    # ---------------- Events ---------------- #
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        # determine if clicked on a node
        idx = self._node_at_display(event.x, event.y)
        # right-click to delete node
        if event.button == 3:
            if idx is not None:
                self._delete_node(idx)
            return
        # only handle left click beyond this point
        if event.button != 1:
            return
        if idx is None:
            # add new node where clicked
            self.model.add_node(float(event.xdata), float(event.ydata))
            self._set_status(f"Added node #{self.model.n_nodes - 1}")
            if self.model.n_edges > 0:
                self._start_layout()
            self._auto_zoom()
            self._redraw()
            return
        # clicked on a node
        if self.selected_idx is None:
            self.selected_idx = idx
            self._set_status(f"Selected node {idx}. Now click another node to add unit edge.")
            self._update_selection_artist()
            self._redraw()
        else:
            if idx == self.selected_idx:
                self._set_status("Selection cleared.")
                self.selected_idx = None
                self._update_selection_artist()
                self._redraw()
                return
            i, j = self.selected_idx, idx
            self.selected_idx = None
            self._update_selection_artist()
            # toggle: remove edge if it exists, else add
            if (min(i, j), max(i, j)) in self.model.edges:
                self.model.remove_edge(i, j)
                self._set_status(f"Removed edge ({min(i,j)}, {max(i,j)}).")
                if not self.frozen:
                    self._start_layout()
                self._auto_zoom()
                self._redraw()
                return
            added = self.model.add_edge(i, j)
            if not added:
                self._set_status("Invalid edge indices.")
                self._redraw()
                return
            if self.frozen:
                # Do not enforce constraints or start layout when frozen
                self._set_status(f"Added edge ({i}, {j}) while frozen; constraints not enforced.")
            else:
                # try to enforce constraints
                ok = self._enforce_unit_constraints_after_edge(i, j)
                if ok:
                    self._set_status(f"Added unit edge ({i}, {j}). Adjusting layout...")
                else:
                    self._set_status("Could not fully satisfy constraints immediately; continuing to adjust...")
                self._start_layout()
            self._auto_zoom()
            self._redraw()

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        # zoom relative to mouse position
        base_scale = 1.2
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return
        if event.button == "up":
            scale_factor = 1 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            scale_factor = 1.0
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
        self.ax.set_xlim([xdata - relx * new_width, xdata + (1 - relx) * new_width])
        self.ax.set_ylim([ydata - rely * new_height, ydata + (1 - rely) * new_height])
        self.fig.canvas.draw_idle()

    def _resolve_two_circle_ambiguities_dynamic(self) -> bool:
        return False

    # ---------------- Buttons ---------------- #
    def on_key_press(self, event):
        # Delete selected node with Delete/Backspace
        if event.key in ("delete", "backspace"):
            if self.selected_idx is not None and 0 <= self.selected_idx < self.model.n_nodes:
                self._delete_node(self.selected_idx)
            else:
                self._set_status("No node selected to delete.")

    def on_solve_clicked(self, _):
        if self.model.n_edges == 0:
            self._set_status("No edges; nothing to solve.")
            return
        if self.frozen:
            self._set_status("Frozen: unfreeze to run Solve.")
            return
        self._start_layout()
        self._set_status("Adjusting layout...")
        self._auto_zoom()
        self._redraw()

    def on_exact_solve_clicked(self, _):
        if self.frozen:
            self._set_status("Frozen: unfreeze to run Exact Solve.")
            return
        if self.model.n_edges == 0 or self.model.n_nodes <= 1:
            self._set_status("Not enough constraints for exact solve.")
            return
        ok = self._solve_least_squares()
        if ok:
            self._set_status("Exact solve converged.")
        else:
            self._set_status("Exact solve did not converge; try Shake or add constraints.")
        # Do not restart dynamic layout here; keep LS result intact
        self._auto_zoom()
        self._redraw()

    def on_shake_clicked(self, _):
        if self.frozen:
            self._set_status("Frozen: unfreeze to apply Shake.")
            return
        try:
            amp = float(self.tb_shake.text.strip())
        except Exception:
            amp = 0.05
        amp = max(0.0, min(1.0, amp))
        self._shake(amp=amp)
        self._set_status("Applied shake to escape local minimum.")
        self._start_layout()
        self._auto_zoom()
        self._redraw()

    def on_toggle_freeze(self, _):
        self.frozen = not self.frozen
        if self.frozen:
            self._stop_layout()
            self.btn_freeze.label.set_text("Freeze: ON")
            self.btn_freeze.ax.set_facecolor("#ffd0d0")
            self._set_status("Frozen: layout and auto-zoom disabled.")
        else:
            self.btn_freeze.label.set_text("Freeze: OFF")
            self.btn_freeze.ax.set_facecolor("#ffecec")
            self._set_status("Unfrozen: you can Solve or Optimize to move points.")
        self.fig.canvas.draw_idle()

    def on_color_clicked(self, _):
        adj = self.model.adjacency_lists()
        solver = ChromaticSolver(adj)
        k, colors = solver.chromatic_number()
        self.color_assignment = colors
        self._apply_colors_from_assignment(colors)
        self._set_status(f"Chromatic number = {k}")
        self._redraw()

    def on_clear_colors(self, _):
        self.color_assignment = None
        self._apply_colors_from_assignment(None)
        self._set_status("Cleared coloring.")
        self._redraw()

    def on_new_graph(self, _):
        self.model = GraphModel()
        self._stop_layout()
        self.vel = np.zeros_like(self.model.pos)
        self.color_assignment = None
        self.selected_idx = None
        self._set_status("New graph.")
        self._auto_zoom()
        self._redraw()

    def on_save_template(self, _):
        name = self.tb_save.text.strip()
        if not name:
            self._set_status("Enter a template name.")
            return
        data = GraphModel.from_dict(self.model.to_dict())  # copy
        data.normalize_for_template()
        path = os.path.join(self.templates_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data.to_dict(), f, indent=2)
        self._set_status(f"Saved template '{name}' at {path}")

    def on_insert_template(self, _):
        name = self.tb_insert.text.strip()
        if not name:
            self._set_status("Enter a template name to insert.")
            return
        path = os.path.join(self.templates_dir, f"{name}.json")
        if not os.path.exists(path):
            self._set_status(f"Template '{name}' not found.")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        templ = GraphModel.from_dict(data)
        # place to the right of current bounding box
        xmin, xmax, ymin, ymax = self.model.bounding_box()
        if self.model.n_nodes == 0:
            offset_x = 0.0
            offset_y = 0.0
        else:
            span = max(xmax - xmin, ymax - ymin)
            offset_x = xmax + 0.5 * max(1.0, span)
            offset_y = (ymin + ymax) / 2.0
        # append nodes
        base_idx = self.model.n_nodes
        for p in templ.pos:
            self.model.add_node(p[0] + offset_x, p[1] + offset_y)
        for (i, j) in templ.edges:
            self.model.add_edge(base_idx + i, base_idx + j)
        self._set_status(f"Inserted template '{name}'.")
        self._start_layout()
        self._auto_zoom()
        self._redraw()

    def on_list_templates(self, _):
        files = [f for f in os.listdir(self.templates_dir) if f.endswith('.json')]
        if not files:
            self._set_status("No templates saved yet.")
        else:
            self._set_status("Templates: " + ", ".join(sorted(f[:-5] for f in files)))

    # ---------------- Solver / Layout ---------------- #
    def _enforce_unit_constraints_after_edge(self, i: int, j: int) -> bool:
        # If this is the first edge in an empty/loose graph, rescale so that |xi-xj|=1 exactly to start
        if self.model.n_edges == 1:
            pi = self.model.pos[i]
            pj = self.model.pos[j]
            d = float(np.linalg.norm(pi - pj))
            if d > 1e-9:
                s = 1.0 / d
                self.model.scale(s)
        ok = self._solve_layout_multi_start()
        return ok

    def _solve_layout_multi_start(self) -> bool:
        # Try from current, then a few jittered restarts
        if self.model.n_edges == 0 or self.model.n_nodes <= 1:
            return True
        base_pos = self.model.pos.copy()
        for attempt in range(self.params.restarts + 1):
            if attempt > 0:
                # restart with small jitter
                self.model.pos = base_pos + 0.05 * np.random.randn(*base_pos.shape)
            ok = self._spring_repulsion_solve()
            if ok:
                return True
        # restore best (closest) among attempts? For simplicity keep last
        return False

    def _spring_repulsion_solve(self) -> bool:
        # energy: sum_edges k*(|d|-1)^2 + sum_pairs c/(|d|+eps)
        params = self.params
        X = self.model.pos.copy()
        edges = self.model.edges_array()
        n = X.shape[0]
        if edges.shape[0] == 0:
            self.model.pos = X
            return True
        step = params.step
        k = params.spring_k
        # constant repulsion (no schedule)
        c = params.repulsion_c
        eps = 1e-6

        # Compute repulsion pairs within components and excluding isolates
        rep_pairs = self._compute_nonedge_pairs(n, edges)

        def edge_error(Xc: np.ndarray) -> float:
            if edges.shape[0] == 0:
                return 0.0
            d = np.linalg.norm(Xc[edges[:, 0]] - Xc[edges[:, 1]], axis=1)
            return float(np.max(np.abs(d - 1.0)))

        prev_E = float("inf")
        for it in range(params.max_iter):
            F = np.zeros_like(X)
            # spring forces on edges
            if edges.shape[0] > 0:
                Dij = X[edges[:, 0]] - X[edges[:, 1]]
                dist = np.linalg.norm(Dij, axis=1) + eps
                # force magnitude along the edge direction
                mag = k * (dist - 1.0) / dist
                f = (Dij.T * mag).T  # (m,2)
                # accumulate gradients (dE/dx)
                np.add.at(F, edges[:, 0], f)
                np.add.at(F, edges[:, 1], -f)
            # repulsion between all pairs (constant)
            if rep_pairs.shape[0] > 0 and c > 0.0:
                D = X[rep_pairs[:, 0]] - X[rep_pairs[:, 1]]
                dist2 = np.sum(D * D, axis=1) + eps
                dist = np.sqrt(dist2)
                mag_rep = c / dist2  # ~ c / r^2, gradient uses c / r^3 along D
                f_rep = (D.T * (mag_rep / dist)).T  # equals D * c / r^3
                # gradient of c/r is -D * c / r^3
                np.add.at(F, rep_pairs[:, 0], -f_rep)
                np.add.at(F, rep_pairs[:, 1], f_rep)

            # gradient descent step with damping
            X = X - step * F
            # recentre connected nodes only (avoid moving isolates)
            deg_now = self._degrees(n, edges)
            if np.any(deg_now > 0):
                mask = deg_now > 0
                center = X[mask].mean(axis=0, keepdims=True)
                X[mask] = X[mask] - center

            # basic line search: if energy increases, shrink step
            # compute energy
            E = 0.0
            if edges.shape[0] > 0:
                d = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)
                E += float(np.sum(0.5 * k * (d - 1.0) ** 2))
            if rep_pairs.shape[0] > 0 and c > 0.0:
                D = X[rep_pairs[:, 0]] - X[rep_pairs[:, 1]]
                dist = np.linalg.norm(D, axis=1) + eps
                E += float(np.sum(c / dist))
            if E > prev_E + 1e-10:
                step *= 0.7
            else:
                step *= 1.02
            step = float(np.clip(step, 1e-4, 0.2))
            prev_E = E

            # stopping criterion on max edge error
            if edge_error(X) < params.tol:
                self.model.pos = X
                return True
        # final check
        if edge_error(X) < 5 * params.tol:
            self.model.pos = X
            return True
        return False

    # ---------------- Dynamic layout (timer) ---------------- #
    def _start_layout(self) -> None:
        if self.frozen:
            # Do not start layout when frozen
            return
        self.layout_running = True
        self._tick_count = 0
        self.settled_ticks = 0
        self._ensure_vel_shape()

    def _stop_layout(self) -> None:
        self.layout_running = False
        self.settled_ticks = 0

    def _on_timer(self) -> None:
        if self.frozen or not self.layout_running:
            return
        # perform a small number of steps per tick for smoothness
        err = self._layout_step(steps=3)
        self._tick_count += 1
        # update drawing
        if self._tick_count % 2 == 0:
            self._redraw()
        # auto-zoom occasionally
        if self._tick_count % 15 == 0:
            self._auto_zoom()
        # convergence check
        if err < self.params.tol:
            self.settled_ticks += 1
        else:
            self.settled_ticks = 0
        if self.settled_ticks >= 6:
            self._stop_layout()
            self._set_status("Layout settled.")
            self._redraw()

    def _ensure_vel_shape(self) -> None:
        n = self.model.n_nodes
        if getattr(self, 'vel', None) is None:
            self.vel = np.zeros((n, 2), dtype=float)
            return
        if self.vel.shape[0] != n:
            new_v = np.zeros((n, 2), dtype=float)
            m = min(n, self.vel.shape[0])
            if m > 0:
                new_v[:m] = self.vel[:m]
            self.vel = new_v

    def _degrees(self, n: int, edges: np.ndarray) -> np.ndarray:
        deg = np.zeros(n, dtype=int)
        if edges.size:
            np.add.at(deg, edges[:, 0], 1)
            np.add.at(deg, edges[:, 1], 1)
        return deg

    def _components(self, n: int, edges: np.ndarray) -> np.ndarray:
        # simple DFS-based connected components
        adj = [[] for _ in range(n)]
        if edges.size:
            for a, b in edges:
                a = int(a); b = int(b)
                adj[a].append(b)
                adj[b].append(a)
        comp = -np.ones(n, dtype=int)
        cid = 0
        for i in range(n):
            if comp[i] != -1:
                continue
            stack = [i]
            comp[i] = cid
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if comp[v] == -1:
                        comp[v] = cid
                        stack.append(v)
            cid += 1
        return comp

    def _compute_nonedge_pairs(self, n: int, edges: np.ndarray) -> np.ndarray:
        if n <= 1:
            return np.zeros((0, 2), dtype=int)
        deg = self._degrees(n, edges)
        comp = self._components(n, edges)
        edge_set = set((int(a), int(b)) for a, b in edges)
        pairs = []
        for i in range(n):
            if deg[i] == 0:
                continue
            for j in range(i + 1, n):
                if deg[j] == 0:
                    continue
                if comp[i] != comp[j]:
                    continue
                if (i, j) in edge_set:
                    continue
                pairs.append((i, j))
        return np.array(pairs, dtype=int) if pairs else np.zeros((0, 2), dtype=int)

    def _edge_error(self, X: np.ndarray, edges: np.ndarray) -> float:
        if edges.size == 0:
            return 0.0
        d = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)
        return float(np.max(np.abs(d - 1.0)))

    def _layout_step(self, steps: int = 1) -> float:
        # Velocity-based integration for smooth drift
        params = self.params
        k = params.spring_k
        c = params.repulsion_c
        eps = 1e-6
        dt = self._dyn_step
        gamma = 0.25  # damping

        X = self.model.pos
        E = self.model.edges_array()
        n = self.model.n_nodes
        self._ensure_vel_shape()
        repel_pairs = self._compute_nonedge_pairs(n, E)

        for _ in range(max(1, int(steps))):
            F = np.zeros_like(X)
            if E.shape[0] > 0:
                Dij = X[E[:, 0]] - X[E[:, 1]]
                dist = np.linalg.norm(Dij, axis=1) + eps
                mag = k * (dist - 1.0) / dist
                f = (Dij.T * mag).T
                # accumulate gradients (dE/dx) for spring energy 0.5*k*(|d|-1)^2
                np.add.at(F, E[:, 0], f)
                np.add.at(F, E[:, 1], -f)
            if repel_pairs.shape[0] > 0 and c > 0:
                D = X[repel_pairs[:, 0]] - X[repel_pairs[:, 1]]
                dist2 = np.sum(D * D, axis=1) + eps
                dist = np.sqrt(dist2)
                mag_rep = c / dist2
                f_rep = (D.T * (mag_rep / dist)).T
                np.add.at(F, repel_pairs[:, 0], -f_rep)
                np.add.at(F, repel_pairs[:, 1], f_rep)
            # integrate
            self.vel = (1.0 - gamma) * self.vel - dt * F
            # clamp velocity
            vnorm = np.linalg.norm(self.vel, axis=1, keepdims=True) + 1e-12
            vmax = 0.5
            self.vel = self.vel * np.minimum(1.0, vmax / vnorm)
            X[...] = X + dt * self.vel
            # recenter only connected nodes (avoid moving isolates)
            deg_now = self._degrees(n, E)
            if np.any(deg_now > 0):
                mask = deg_now > 0
                center = X[mask].mean(axis=0, keepdims=True)
                X[mask] = X[mask] - center

        err = self._edge_error(X, E)
        self.model.pos = X
        return err

    def _stochastic_optimize(self, method: str = "Adam", steps: int = 800, lr: float = 0.03, t0: float = 0.05, t1: float = 0.0) -> bool:
        # Optimize edge unit constraints with SGD/Adam/Muon + simulated annealing
        E = self.model.edges_array()
        n = self.model.n_nodes
        if n <= 1 or E.size == 0:
            return True
        X = self.model.pos.copy()
        params = self.params
        k = params.spring_k
        c0 = params.repulsion_c
        eps = 1e-6
        repel_pairs = self._compute_nonedge_pairs(n, E)
        deg = self._degrees(n, E)
        mask = deg > 0

        # Adam / momentum buffers
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        mu = 0.9  # momentum for Muon/SGD-momentum
        beta1, beta2 = 0.9, 0.999

        for t in range(1, steps + 1):
            F = np.zeros_like(X)
            # springs
            if E.shape[0] > 0:
                Dij = X[E[:, 0]] - X[E[:, 1]]
                dist = np.linalg.norm(Dij, axis=1) + eps
                mag = k * (dist - 1.0) / dist
                f = (Dij.T * mag).T
                np.add.at(F, E[:, 0], f)
                np.add.at(F, E[:, 1], -f)
            # repulsion within components, exclude edges and isolates (annealed)
            c = c0 * max(0.0, 1.0 - t / float(steps))
            if repel_pairs.shape[0] > 0 and c > 0:
                D = X[repel_pairs[:, 0]] - X[repel_pairs[:, 1]]
                dist2 = np.sum(D * D, axis=1) + eps
                dist = np.sqrt(dist2)
                mag_rep = c / dist2
                f_rep = (D.T * (mag_rep / dist)).T
                np.add.at(F, repel_pairs[:, 0], -f_rep)
                np.add.at(F, repel_pairs[:, 1], f_rep)
            # zero out isolated nodes' gradients
            F[~mask] = 0.0

            if method == "SGD":
                X = X - lr * F
            elif method == "Adam":
                m = beta1 * m + (1 - beta1) * F
                v = beta2 * v + (1 - beta2) * (F * F)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                X = X - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            else:  # "Muon" (heavy-ball + Nesterov-like)
                v = mu * v + lr * F
                X = X - (mu * v + lr * F)  # Nesterov-style lookahead step

            # simulated annealing noise on connected nodes
            if t0 > 0.0 or t1 > 0.0:
                T = t0 + (t1 - t0) * (t / steps)
                if T > 0:
                    noise = T * np.random.randn(*X.shape)
                    noise[~mask] = 0.0
                    X += noise

            # recenter connected nodes only
            if np.any(mask):
                center = X[mask].mean(axis=0, keepdims=True)
                X[mask] = X[mask] - center

            # early stop
            d = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
            err = float(np.max(np.abs(d - 1.0))) if d.size else 0.0
            if err < self.params.tol:
                break

        self.model.pos = X
        self.vel *= 0.0
        # final check
        d = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
        err = float(np.max(np.abs(d - 1.0))) if d.size else 0.0
        return err < 5 * self.params.tol

    # ---------------- Utility: group boxes ---------------- #
    def _add_column_header(self, cx: float, y: float, label: str) -> None:
        if not label:
            return
        # Place a small, unobtrusive header in figure coordinates
        # Keep behind widgets but above group box outline
        self.fig.text(cx, y, label, fontsize=9, color="#333333", ha="center", va="center", zorder=-5.0)

    def _add_group_box(self, x: float, y: float, w: float, h: float, label: str) -> None:
        if h <= 0:
            return
        # Draw only an outline, fully transparent fill, and very low zorder (behind widgets)
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.01",
            fc=(1.0, 1.0, 1.0, 0.0),  # fully transparent fill
            ec=(0.65, 0.65, 0.65, 1.0),
            lw=1.0,
            zorder=-10.0,
        )
        self.fig.add_artist(rect)
        if label:
            self.fig.text(x + 0.005, y + h + 0.005, label, fontsize=9, color="#444444", zorder=-10.0)

    def _shrink_widget_fonts(self) -> None:
        # Best-effort reduce label font sizes for compact UI
        widgets = [
            getattr(self, name, None) for name in [
                'btn_color','btn_clear_color','btn_solve','btn_exact','btn_stoch','btn_shake','btn_freeze',
                'btn_new','btn_save','btn_insert','btn_list','btn_check'
            ]
        ]
        for w in widgets:
            try:
                if w is not None and hasattr(w, 'label'):
                    w.label.set_fontsize(8)
            except Exception:
                pass
        tboxes = [
            getattr(self, name, None) for name in [
                'tb_lr','tb_steps','tb_tstart','tb_tend','tb_shake','tb_save','tb_insert','tb_rep'
            ]
        ]
        for tb in tboxes:
            try:
                if tb is not None and hasattr(tb, 'label'):
                    tb.label.set_fontsize(8)
                if tb is not None and hasattr(tb, 'text_disp'):
                    tb.text_disp.set_fontsize(8)
            except Exception:
                pass

    def on_repulsion_changed(self, text: str) -> None:
        try:
            val = float(text.strip())
        except Exception:
            self._set_status("Invalid repulsion value.")
            return
        val = max(0.0, min(10.0, val))
        self.params.repulsion_c = val
        self._set_status(f"Repulsion set to {val}")

    # ---------------- Coincidence check ---------------- #
    def on_check_coincident(self, _):
        eps = 1e-3
        n = self.model.n_nodes
        if n <= 1:
            self._set_status("No coincident pairs (<=1 node).")
            return
        X = self.model.pos
        # Efficient sweep by sorting x
        order = np.argsort(X[:, 0], kind='mergesort')
        Xs = X[order]
        coinc_pairs = []
        j_start = 0
        for i in range(n):
            # advance j_start to maintain window where x diff <= eps
            while j_start < n and Xs[i, 0] - Xs[j_start, 0] > eps:
                j_start += 1
            j = max(i + 1, j_start)
            while j < n and Xs[j, 0] - Xs[i, 0] <= eps:
                if abs(Xs[j, 1] - Xs[i, 1]) <= eps:
                    a = int(order[i]); b = int(order[j])
                    if a > b:
                        a, b = b, a
                    coinc_pairs.append((a, b))
                j += 1
        if coinc_pairs:
            pairs_str = ", ".join(f"({a},{b})" for (a, b) in coinc_pairs[:10])
            more = "" if len(coinc_pairs) <= 10 else f" and {len(coinc_pairs)-10} more"
            self._set_status(f"Coincident pairs (<{eps:.0e}): {pairs_str}{more}")
        else:
            self._set_status(f"No coincident pairs found (threshold {eps:.0e}).")

    # ---------------- Local minima helpers ---------------- #
    def _shake(self, amp: float = 0.05) -> None:
        if self.model.n_nodes == 0:
            return
        self.model.pos += amp * np.random.randn(*self.model.pos.shape)
        # recenter and reset velocity for stability
        self.model.pos -= self.model.pos.mean(axis=0, keepdims=True)
        if getattr(self, 'vel', None) is not None and self.vel.size:
            self.vel *= 0.0

    def _solve_least_squares(self) -> bool:
        # Attempt exact solve for unit edges with soft repulsion and small anchors
        E = self.model.edges_array()
        n = self.model.n_nodes
        if n <= 1 or E.size == 0:
            return True
        X0 = self.model.pos.copy()
        w_edge = 10.0
        w_anchor = 0.02

        def make_fun(x0_ref: np.ndarray):
            anchors = list(range(min(2, n)))  # anchor first up to 2 nodes

            def fun(vec: np.ndarray) -> np.ndarray:
                X = vec.reshape(n, 2)
                res = []
                if E.shape[0] > 0:
                    d = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
                    res_edges = np.sqrt(w_edge) * (d - 1.0)
                    res.append(res_edges)
                if anchors:
                    res_anchor = []
                    for a in anchors:
                        res_anchor.extend([np.sqrt(w_anchor) * (X[a, 0] - x0_ref[a, 0]),
                                           np.sqrt(w_anchor) * (X[a, 1] - x0_ref[a, 1])])
                    res.append(np.array(res_anchor))
                return np.concatenate(res) if res else np.zeros(0)

            return fun

        best_err = float('inf')
        best_X = None
        rng = np.random.default_rng(0)
        for attempt in range(5):
            if attempt == 0:
                x0 = X0.reshape(-1)
                ref = X0
            else:
                jitter = 0.05 * (attempt) * rng.standard_normal(X0.shape)
                ref = X0
                x0 = (X0 + jitter).reshape(-1)
            try:
                res = least_squares(
                    make_fun(ref), x0, method='trf', loss='huber', f_scale=0.2,
                    max_nfev=3000, ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=0
                )
            except Exception:
                continue
            X = res.x.reshape(n, 2)
            # evaluate max edge error
            d = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
            err = float(np.max(np.abs(d - 1.0))) if d.size else 0.0
            if err < best_err:
                best_err = err
                best_X = X.copy()

        if best_X is not None and best_err < max(2 * self.params.tol, 1e-4):
            self.model.pos = best_X
            self.model.pos -= self.model.pos.mean(axis=0, keepdims=True)
            self.vel *= 0.0
            return True
        return False

    def _try_fix_two_circle_branch(self, E: np.ndarray, X: np.ndarray, ref: np.ndarray) -> np.ndarray:
        return X

    # ---------------- Edit ops ---------------- #
    def _delete_node(self, idx: int) -> None:
        if not (0 <= idx < self.model.n_nodes):
            return
        self.model.remove_node(idx)
        self.selected_idx = None
        self.color_assignment = None  # invalidate coloring
        self._ensure_vel_shape()
        if self.model.n_edges > 0:
            self._start_layout()
        self._set_status(f"Deleted node {idx}.")
        self._auto_zoom()
        self._redraw()

    # ---------------- Drawing helpers ---------------- #
    def _node_at_display(self, x_pix: float, y_pix: float) -> Optional[int]:
        if self.model.n_nodes == 0:
            return None
        # compute nearest node in display coordinates
        xy_disp = self.ax.transData.transform(self.model.pos)
        d2 = (xy_disp[:, 0] - x_pix) ** 2 + (xy_disp[:, 1] - y_pix) ** 2
        idx = int(np.argmin(d2))
        if d2[idx] <= self.pick_radius_px ** 2:
            return idx
        return None

    def _apply_colors_from_assignment(self, colors: Optional[Sequence[int]]):
        n = self.model.n_nodes
        if colors is None:
            c = np.tile(np.array([[0.2, 0.6, 1.0, 1.0]]), (n, 1))
            self.node_colors_rgba = c
            # clear labels
            for t in self.node_labels:
                t.remove()
            self.node_labels = []
            return
        k = max(colors) + 1 if colors else 1
        # color palette
        rng = np.random.default_rng(12345)
        palette = rng.random((k, 3)) * 0.7 + 0.2  # avoid too dark/bright
        rgba = np.ones((n, 4))
        for i in range(n):
            rgba[i, :3] = palette[colors[i] % k]
        self.node_colors_rgba = rgba
        # update labels with color indices
        for t in self.node_labels:
            t.remove()
        self.node_labels = []
        if n > 0 and colors is not None:
            for i in range(n):
                txt = self.ax.text(self.model.pos[i, 0], self.model.pos[i, 1], str(colors[i]),
                                   color='black', fontsize=10, ha='center', va='center', zorder=5,
                                   bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='#888888', alpha=0.85))
                self.node_labels.append(txt)

    def _update_selection_artist(self):
        if self.selected_idx is None or self.model.n_nodes == 0:
            self.sel_scatter.set_offsets(np.empty((0, 2)))
        else:
            p = self.model.pos[self.selected_idx : self.selected_idx + 1]
            self.sel_scatter.set_offsets(p)

    def _redraw(self) -> None:
        # nodes
        if self.model.n_nodes > 0:
            self.node_scatter.set_offsets(self.model.pos)
            if self.node_colors_rgba.shape[0] != self.model.n_nodes:
                # update to match size
                if self.color_assignment is not None:
                    self._apply_colors_from_assignment(self.color_assignment)
                else:
                    self._apply_colors_from_assignment(None)
            self.node_scatter.set_facecolors(self.node_colors_rgba)
        else:
            self.node_scatter.set_offsets(np.empty((0, 2)))
        # edges
        if self.model.n_edges > 0:
            segs = [
                [self.model.pos[i], self.model.pos[j]]
                for (i, j) in sorted(list(self.model.edges))
            ]
            self.edge_collection.set_segments(segs)
        else:
            self.edge_collection.set_segments([])
        # move labels with nodes
        if self.node_labels:
            for i, t in enumerate(self.node_labels):
                if i < self.model.n_nodes:
                    t.set_position((self.model.pos[i, 0], self.model.pos[i, 1]))
        self._update_selection_artist()
        self.fig.canvas.draw_idle()

    def _auto_zoom(self) -> None:
        if self.frozen:
            return
        xmin, xmax, ymin, ymax = self.model.bounding_box()
        if self.model.n_nodes == 0:
            self.ax.set_xlim(-3, 3)
            self.ax.set_ylim(-3, 3)
            return
        span_x = xmax - xmin
        span_y = ymax - ymin
        span = max(span_x, span_y, 1.0)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        m = 0.3 * span
        self.ax.set_xlim(cx - span / 2 - m, cx + span / 2 + m)
        self.ax.set_ylim(cy - span / 2 - m, cy + span / 2 + m)

    def _set_status(self, msg: str) -> None:
        self.status_text.set_text(msg)
        self.fig.canvas.draw_idle()

    # ---------------- Run ---------------- #
    def run(self) -> None:
        plt.show()


def main():
    app = HNApp()
    app.run()


if __name__ == "__main__":
    main()
