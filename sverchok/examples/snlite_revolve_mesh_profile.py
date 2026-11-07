"""
in verts v d=[[]] n=0
in edges s d=[[]] n=0
in polys s d=[[]] n=0
in axis s d=2 n=2
in cuts s d=0 n=2
in cut_size s d=0.1 n=2
out sdf s
out moved s
"""
# Revolve any flat mesh: draw a profile in Blender (in the XZ plane for Z),
# feed it in with Get Objects Data, and connect `sdf` to the SDF Viewer.
#   verts / edges / polys  the profile: polygons are used if connected, else
#                          edges, else the vertices in order.  Several
#                          objects, loops or polygons give holes and islands
#   axis      0, 1, 2 = X, Y, Z
#   cuts      round holes drilled through the solid around the axis (0: none)
#   cut_size  radius of those holes
#   moved     how many vertices were off the profile's plane and moved onto it
import numpy as np
import sdfnumba as sn

loops = []
for i, v in enumerate(verts):
    if len(v):
        d = {'verts': v,
             'edges': edges[min(i, len(edges) - 1)] if edges and edges[0] else None,
             'polygons': polys[min(i, len(polys) - 1)] if polys and polys[0] else None}
        loops += sn.profile_loops(**d)

sdf, moved = [], [[0]]
if loops:
    a = 'XYZ'[int(axis)]
    shape, info = sn.revolve_profile(loops, a)
    if cuts:
        k = 'XYZ'.index(a)
        lo, hi = sn.estimate_bounds(shape.program())
        reach = 2 * max(map(abs, (*lo, *hi)))
        mid = (lo[k] + hi[k]) / 2                 # halfway along the axis
        drill = sn.capped_cylinder((0, 0, 0), (reach, 0, 0), cut_size)
        drill = drill.circular_array(int(cuts)).translate((0, 0, mid))
        drill = drill.orient((sn.X, sn.Y, sn.Z)[k])
        shape = sn.difference(shape, drill)
    sdf, moved = [shape], [[info.moved]]
