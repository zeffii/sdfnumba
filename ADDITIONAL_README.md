# sdfnumba: additional notes

## SDF Viewer settings (N-panel)

**Step scale** (default 1.0). How far each raymarching step moves, as a
fraction of the distance the shape reports. Keep it at 1.0 for shapes built
from primitives, booleans, `translate`, `rotate`, `scale`, `revolve`, `arc`,
`dilate` and `shell`: their distances are safe to step by in full, and full
steps are fastest. Lower it (try 0.5) only if a shape shows holes or missing
pieces. That happens with operations that bend space, whose distances can
overshoot: `twist`, `bend`, `blend`, `repeat` and `circular_array`.

**Padding** (default 0.05). Margin around the shape's estimated bounds, as a
fraction of its size. The shape is only drawn inside this box. 0.05 is
enough; larger values only make the box bigger and every pixel in it more
expensive. Raise it slightly if the edge of a shape gets clipped.

**Steps** (default 160). Most raymarching steps per pixel. Raise it if thin
or distant parts disappear; lower it for speed.

**Line width** (default 1.5). Width in pixels of every line type: seams,
creases and outline.

**Smooth edges** (default on). Anti-aliases the silhouette by casting four
rays in pixels on the shape's edge. Turn off for speed.

**Creases from shading** (default off). Also draws creases where the
shading turns sharply, for edges no primitive reports, e.g. on twisted or
bent shapes. Lines found this way are rougher; *Crease* sets how sharp a
turn counts.

## SDF Remesh: starting mesh

SDF Remesh needs only the shape. Without anything on its Vertices and
Polygons inputs it makes its own starting mesh (*Start resolution*, sharp
edges) and remeshes that, which is a quick way to see what remeshing does.

For more control, refine first with an SDF Mesh node, then remesh: connect
SDF Mesh's Vertices and Polygons to SDF Remesh as well as the shape. You can
then pick the starting resolution and Sharp/smooth yourself, and look at
the mesh before and after remeshing side by side.

## SDF Remesh: artifacts, warnings and Seed

The remesher avoids **needles** (one edge much shorter than the other two)
and **caps** (nearly flat triangles), keeps edges shorter than the shape is
thin, and never accepts a change that turns a triangle over. What it can't
avoid, it reports: the node then shows a warning saying what went wrong and
what to try.

- *The starting mesh is too coarse*: parts of the shape are thinner than
  the starting mesh's grid cells, so it arrives already folded. Raise
  Resolution on SDF Mesh, or Start resolution. Remeshing stops early in this
  case rather than growing out of control.
- *Triangles are turned over*: usually where the shape narrows to a sharp
  edge (e.g. where two equal-radius holes cross). Try another Seed, or a
  smaller Edge length.
- *Thin triangles*: try another Seed, or a higher Detail.

**Seed** (default 0) only changes the order the remesher visits edges in.
Every value gives an equally valid mesh, and the same seed always gives the
same mesh; another seed is worth a try when a warning appears.

## SDF Revolve: profiles

The profile is closed loops of vertices, given one of three ways:

- **Vertices** alone: one loop, the vertices in order. It closes itself; a
  repeated first vertex at the end is fine.
- **Vertices + Edges**: any order. The edges are chained into loops and then
  dropped. Every vertex used needs exactly two edges, so open strips and
  branches are refused with a message.
- **Vertices + Polygons**: the outline of the polygons. Edges shared by two
  polygons are inside the profile and dropped, so a profile cut into several
  faces still counts as one region.

A loop inside another makes a hole; separate loops make islands, each swept
into its own ring. Every object on the sockets adds to the same profile.

All vertices must lie in one plane, which should contain the axis (XZ or YZ
for Z). The node fits the plane and moves stray vertices onto it, warning
only when one was off by more than float32 rounding (1e-5 of the profile's
size). A plane within about 1° of a world plane is taken as that plane. A
tilted plane works too: it is revolved around the axis's projection onto it.
A profile may lie on either side of the axis, or across it. An edge drawn
along the axis is not a surface.

From Python: `shape, info = sn.revolve_profile(sn.profile_loops(verts, edges, polygons), 'Z')`.

## SDF Mesh and SDF Remesh: Settle

While values keep changing (a slider being dragged), SDF Mesh outputs a
coarse mesh, and SDF Remesh outputs its starting mesh (a coarse one if it
makes its own). Once nothing has changed for *Settle* seconds they build the
full result. 0 builds it on every change.

**Build in background** (N-panel, default off). Builds the full result on a
separate thread, so Blender stays responsive while it works, and drops a
result if the values changed while it was being built (a pause mid-drag
then never flashes a stale mesh). Off, the full result is built on Blender's
main thread, which freezes the UI briefly. sdfnumba falls back to the main
thread by itself if numba's threading layer can't safely run its kernels
alongside other numba code (the 'workqueue' layer).
