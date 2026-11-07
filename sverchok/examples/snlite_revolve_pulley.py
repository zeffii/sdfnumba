"""
in radius s d=1.0 n=2
in width s d=0.3 n=2
in groove s d=0.08 n=2
in bore s d=0.12 n=2
in spokes s d=5 n=2
in fillet s d=0.03 n=2
out sdf s
"""
# A V-belt pulley: a profile of three loops, revolved around Z.
#   loop 1  the hub, away from the axis, so the bore is a hole through it
#   loop 2  the rim with its V groove, a separate island
# Spokes join them; a keyway is cut into the bore.
#   radius  outer radius            groove  depth of the V groove
#   width   width of the rim        bore    radius of the bore
#   spokes  how many spokes         fillet  blend where spokes meet hub and rim
import numpy as np
import sdfnumba as sn

hub_r, hub_w = bore + 0.15, 0.6 * width
rim_in = 0.75 * radius
w = width / 2
hub = [(bore, 0, -hub_w), (hub_r, 0, -hub_w), (hub_r, 0, hub_w), (bore, 0, hub_w)]
rim = [(rim_in, 0, -w), (radius, 0, -w), (radius, 0, -0.35 * w),
       (radius - groove, 0, -0.1 * w), (radius - groove, 0, 0.1 * w),
       (radius, 0, 0.35 * w), (radius, 0, w), (rim_in, 0, w)]
wheel, info = sn.revolve_profile([np.array(hub), np.array(rim)], 'Z')

spoke = sn.capsule((hub_r * 0.8, 0, 0), (rim_in + 0.02, 0, 0), 0.25 * width)
wheel = sn.union(wheel, spoke.circular_array(int(spokes)), k=fillet or None)
key = sn.box((0.12, bore, 2 * width), center=(bore, 0, 0))
sdf = [sn.difference(wheel, key)]
