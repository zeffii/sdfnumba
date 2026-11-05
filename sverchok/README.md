# sdfnumba nodes for Sverchok

1. Install sdfnumba into Blender's Python (from the Python console):

       from sverchok.utils.pip_utils import install_package
       install_package(['-e', r"path\to\sdfnumba"])

   then restart Blender.
2. Copy `nodes/sdf_numba/` into `sverchok/nodes/`.
3. Add the nodes to Sverchok's menu, in `sverchok/menus/full_by_data_type.yaml`,
   under `- Viz:`:

       - SvSDFViewerNode
       - SvSDFMeshNode
       - SvSDFRemeshNode

4. Restart Blender (or reload Sverchok).

The first time a node runs, sdfnumba compiles (about 25 s, Blender freezes;
once per machine).

## Trying the viewer

`examples/snlite_csg.py` builds a shape in a Scripted Node Lite. Load it into
an SNLite node and connect its `sdf` output to the SDF Viewer.

## Nodes

- **SDF Viewer** draws the shape on the GPU by raymarching it, with optional
  seam, crease and outline lines, colour per primitive, and a distance slice.
  Nothing is meshed. If twisted or bent shapes show holes, lower *Step scale*
  in the N-panel.
- **SDF Mesh** makes a closed mesh (quads, or triangles), smooth or *Sharp*.
- **SDF Remesh** remeshes that mesh: triangles follow curvature and get finer
  along creases. Feed it the shape plus SDF Mesh's vertices and polygons.

While values keep changing (a slider being dragged), SDF Mesh outputs a coarse
mesh and SDF Remesh passes its input through; once nothing has changed for
*Settle* seconds (N-panel) they build the full result.
