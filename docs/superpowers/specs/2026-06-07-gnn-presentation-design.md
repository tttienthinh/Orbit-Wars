# GNN Modelisations for Orbit Wars — Beamer Presentation Design

**Date:** 2026-06-07  
**Output:** `101-GNN_presentation/main.tex`  
**Audience:** Personal reference (dense, no game-context preamble)  
**Format:** LaTeX Beamer, `metropolis` theme, TikZ graphs

---

## Slide Structure (8 slides)

### Slide 1 — Title
- Title: "GNN Modelisations for Orbit Wars"
- Subtitle: brief tagline (e.g. "Five graph representations for fleet decision-making")

### Slide 2 — Overview / TOC
- Numbered list of all 5 modelizations with one-liner descriptions
- Serves as a quick navigation reference

### Slide 3 — Modelisation 1: Edge Prediction on Planet Graph
**Representation:**
- Nodes: planets — features: `production/5`, nature one-hot (fix/moving/comet), `{x/100, y/100, owner_oh, log(ships)/log(1024)}` × 10 timesteps → ~22 dims
- Edges: one directed edge per (src_planet, dst_planet, ETA) attack opportunity — features: `log(min_ships)/log(1024)`, `log(max_ships)/log(1024)`, `ETA/9`

**Task:** Edge classification — predict which edges to activate (multi-label binary)

**TikZ graph:** 3 planet nodes (circle), 2 directed labeled edges showing `ships, ETA` annotation

---

### Slide 4 — Modelisation 2: Heterogeneous Attack Nodes
**Representation:**
- Planet nodes: same 22-dim features as M1
- Attack nodes: features `log(min_ships)/log(1024)`, `log(max_ships)/log(1024)`, `ETA/9`
- Edges: `planet → spawns → attack`, `attack → attacks → planet`
- Optional: master node connected to all planets (global context)

**Task:** Node classification on attack nodes — predict which attack nodes to activate

**Note:** This matches the current `97-library.py` `build_hetero_data_97()` implementation (HeteroData with planet/action/master node types)

**TikZ graph:** Tripartite layout — src planet (left) → attack node (center) → dst planet (right)

---

### Slide 5 — Modelisation 3: Attack-Only Graph
**Representation:**
- Nodes: attack opportunities only — features: `ships (normalised)`, `ETA/9`
- Edges: connect attacks sharing the same source planet (same-src clique) or same target planet (same-dst clique) — no planet nodes

**Task:** Node classification on attack nodes

**TikZ graph:** Two groups of attack nodes, intra-group edges labeled "same src" and "same dst"

---

### Slide 6 — Modelisation 4: Step Nodes (ETA Lifted)
**Representation:**
- Step nodes: one node per ETA value (1–9) — no features
- Attack nodes: features `ships (normalised)` only (ETA removed — captured by structure)
- Edges: attack node → step node (its ETA), plus same-src/same-dst attack edges

**Task:** Node classification on attack nodes

**TikZ graph:** Row of step nodes (rectangles, ETA=1..3), attack nodes (circles) connected to their step

---

### Slide 7 — Modelisation 5: Temporal Unrolling
**Representation:**
- 10 copies of each planet node (one per timestep t = 0..9) — each node has single-timestep features: `production/5`, nature one-hot, `x/100`, `y/100`, `owner_oh`, `log(ships)/log(1024)`
- Edges: directed attack edge from `planet_src[t=i]` to `planet_dst[t=i+ETA]`

**Task:** Edge classification — predict which temporal attack edges to activate

**TikZ graph:** 3 columns of planet nodes (t, t+1, t+2), cross-time attack edges

---

### Slide 8 — Add-ons / Extensions
Bullet list of optional enhancements applicable to any modelization:
- **Master node:** global features — `step/500`, `(angular_velocity - 0.025)/(0.05-0.025)`, ship proportion per player (4 values); connected to all planet/attack nodes
- **Comet-aware features:** comet one-hot already in nature; path index as additional feature
- **Multi-step lookahead labels:** label attacks by forward-simulated outcome (delta ships) rather than heuristic selection
- **Fleet nodes:** represent in-flight fleets as nodes with `x, y, angle, ships, ETA_to_each_planet`

---

## Visual / Layout Conventions

- **Two-column layout per modelization slide:** left = TikZ graph, right = feature table + task description
- **Node colors:** planet = `blue!20`, attack = `orange!20`, step = `gray!20`, master = `green!20`
- **Node shapes:** planet = circle, attack = circle (smaller), step = rectangle, master = diamond
- **Feature table:** two columns — "Node/Edge type" | "Features"
- **Task line:** bold, bottom of right column

## File Layout

```
101-GNN_presentation/
  main.tex       ← single self-contained file
```

No external images; all graphics via TikZ inline.
