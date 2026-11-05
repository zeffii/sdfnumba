"""
in radius s d=1.0 n=2
in hole s d=0.5 n=2
in blend s d=0.0 n=2
out sdf s
"""
# Paste into a Scripted Node Lite (SNLite) and connect `sdf` to the SDF Viewer.
# Until the SDF primitive nodes exist, this is how a shape gets into the tree.
import sdfnumba as sn

f = sn.sphere(radius) & sn.box(1.5 * radius)
c = sn.cylinder(hole)
f = sn.difference(f, c.orient(sn.X) | c.orient(sn.Y) | c, k=blend or None)
sdf = [f]
