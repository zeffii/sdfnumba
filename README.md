# sdfnumba

Fast signed distance fields → closed meshes, using numba.
Dual contouring for sharp edges, Surface Nets for smooth.

## Acknowledgement
Built on the ideas and API of Michael Fogleman's
[sdf](https://github.com/fogleman/sdf) (MIT). Portions are ported
from it; his copyright notice is retained in LICENSE.

## Install
    pip install -e .

## Sverchok
Copy `sverchok/nodes/sdf_numba/` into `sverchok/nodes/`
and add the menu entry listed in `sverchok/README.md`.
Requires numba in Blender's Python.

## License
MIT. See LICENSE.
