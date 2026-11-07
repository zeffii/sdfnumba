# sdfnumba nodes for Sverchok

1. Install sdfnumba into Blender's Python (from the Python console):

       from sverchok.utils.pip_utils import install_package
       install_package(['-e', r"path\to\sdfnumba"])

   then restart Blender.
2. Link (or copy) into your Sverchok folder:
   - `nodes/sdf_numba/` → `sverchok/nodes/sdf_numba/`
   - `utils/sdfnumba_helper.py` → `sverchok/utils/sdfnumba_helper.py`
3. Add the nodes to Sverchok's menu, in `sverchok/menus/full_by_data_type.yaml`:
   the viewer under `- Viz:`

       - SvSDFViewerNode

   and the others in a section of their own, e.g. after `- Viz:`

       - SDF:
           - icon_name: MESH_ICOSPHERE
           - SvSDFRevolveNode
           - SvSDFMeshNode
           - SvSDFRemeshNode

4. Restart Blender (or reload Sverchok).

The first time a node runs, sdfnumba compiles (about 25 s, Blender freezes;
once per machine).

## Trying the viewer

`examples/sdfnumba_csg.py` builds a shape in a Scripted Node Lite. Load it into
an SNLite node and connect its `sdf` output to the SDF Viewer.

More in `examples/`: a lamp shade, and revolved profiles (`sdfnumba_revolve_*`):
a vase, a V-belt pulley (holes and islands), a chess rook, and one that
revolves any flat mesh you feed it.

## Nodes

- **SDF Viewer** draws the shape on the GPU by raymarching it, with optional
  seam, crease and outline lines, colour per primitive, and a distance slice.
  Nothing is meshed. If twisted or bent shapes show holes, lower *Step scale*
  in the N-panel.
- **SDF Revolve** spins a profile (vertices, edges or polygons) around the
  X, Y or Z axis into a shape; holes and islands work. See
  [ADDITIONAL_README.md](../ADDITIONAL_README.md#sdf-revolve-profiles).
- **SDF Mesh** makes a closed mesh (quads, or triangles), smooth or *Sharp*.
- **SDF Remesh** remeshes the shape: triangles follow curvature and get finer
  along creases. Only the shape is needed; see
  [ADDITIONAL_README.md](../ADDITIONAL_README.md) for its optional mesh inputs.

While values keep changing (a slider being dragged), SDF Mesh and SDF Remesh
output a coarse stand-in; once nothing has changed for
*Settle* seconds (N-panel) they build the full result.
