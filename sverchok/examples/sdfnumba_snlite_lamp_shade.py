"""
in radius s d=1.0 n=2
in sweep s d=0.35 n=2
in offset s d=0.2 n=2
in thickness s d=0.02 n=2
in tilt s d=0.314159 n=2
out sdf s
"""
# A lamp shade: an arc revolved into a thin shell (after spomso's example).
# Connect `sdf` to the SDF Viewer, or to SDF Mesh for geometry.
#   radius     radius of the arc
#   sweep      how much of a half-turn the arc covers (0.35 = 63 degrees)
#   offset     extra distance of the arc from the axis
#   thickness  half the wall thickness
#   tilt       turns the profile before revolving (radians)
import numpy as np
import sdfnumba as sn

profile = sn.arc(radius, np.pi, np.pi + sweep * np.pi)
profile = profile.translate((radius + offset, 0, 0)).rotate(tilt, sn.Z)
shade = profile.revolve().dilate(thickness).translate((0, 0, 0.3))
sdf = [shade]
