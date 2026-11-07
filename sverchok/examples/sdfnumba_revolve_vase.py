"""
in height s d=2.0 n=2
in belly s d=0.35 n=2
in neck s d=0.45 n=2
in wall s d=0.06 n=2
in dimples s d=12 n=2
in dimple_size s d=0.1 n=2
out sdf s
"""
# A vase from a profile of vertices (one loop, in order), revolved around Z,
# with a ring of dimples pressed into its belly.
#   height       height of the vase
#   belly        how far the belly swells out
#   neck         radius of the neck
#   wall         wall thickness (the profile goes up the outside, down the inside)
#   dimples      how many dimples around the belly (0: none)
#   dimple_size  radius of each dimple
# Dimples deeper than the wall become holes: a pierced vase.
# The profile starts and ends on the axis: edges along the axis are dropped,
# so there is no seam down the middle.
import numpy as np
import sdfnumba as sn

t = np.linspace(0, 1, 48)
z = t * height
outer = neck + belly * np.sin(np.pi * t) ** 1.5 + 0.25 * (1 - t) ** 3
lip = 0.08 * np.exp(-((1 - t) / 0.04) ** 2)                 # a small rolled lip
r_out = outer + lip
floor = 3 * wall
inside = z >= floor
r_in = np.maximum(outer[inside] - wall, 0.05)

loop = ([(0, 0, 0)] + [(r, 0, h) for r, h in zip(r_out, z)]
        + [(r, 0, h) for r, h in zip(r_in[::-1], z[inside][::-1])]
        + [(0, 0, floor)])
vase, info = sn.revolve_profile([np.array(loop)], 'Z')

if dimples:
    k = int(np.argmax(r_out))                             # the widest point
    dent = sn.sphere(dimple_size).translate((0, 0, z[k]))
    vase = sn.difference(vase, dent.circular_array(int(dimples), r_out[k] + 0.65 * dimple_size),
                         k=0.03)
sdf = [vase]
