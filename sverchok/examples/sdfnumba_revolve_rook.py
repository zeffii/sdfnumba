"""
in height s d=2.0 n=2
in waist s d=0.32 n=2
in merlons s d=6 n=2
in cup s d=0.15 n=2
out sdf s
"""
# A chess rook: a stepped base, a curved waist and a cupped top, drawn as one
# profile loop and revolved, then the battlements cut out of the top.
#   height   overall height        merlons  how many teeth on the top
#   waist    radius of the waist   cup      depth of the hollow on top
import numpy as np
import sdfnumba as sn

h = height
t = np.linspace(0, 1, 24)
body_z = (0.22 + 0.55 * t) * h
body_r = waist + 0.18 * (1 - np.sin(np.pi * t)) * np.where(t < 0.5, 1.4, 0.8)
top = h * 0.77
loop = ([(0, 0, 0), (0.62, 0, 0), (0.62, 0, 0.06 * h), (0.55, 0, 0.09 * h),
         (0.58, 0, 0.13 * h), (0.48, 0, 0.18 * h)]
        + [(r, 0, z) for r, z in zip(body_r, body_z)]
        + [(0.52, 0, top + 0.02 * h), (0.52, 0, h), (0.38, 0, h),
           (0.38, 0, h - cup), (0, 0, h - cup)])
rook, info = sn.revolve_profile([np.array(loop)], 'Z')

gap = sn.box((1.2, 0.16, 0.12 * h), center=(0.6, 0, h))
sdf = [sn.difference(rook, gap.circular_array(int(merlons)), k=0.01)]
