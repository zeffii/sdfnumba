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

## SDF Mesh and SDF Remesh: Settle

While values keep changing (a slider being dragged), SDF Mesh outputs a
coarse mesh and SDF Remesh passes its input through. Once nothing has
changed for *Settle* seconds they build the full result. 0 builds it on
every change.
