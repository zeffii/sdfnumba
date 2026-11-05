# sdfnumba

Fast signed distance fields → closed meshes, using numba.
Dual contouring for sharp edges, Surface Nets for smooth.
Optional adaptive remeshing that keeps creases.

## Acknowledgement
Built on the ideas and API of Michael Fogleman's
[sdf](https://github.com/fogleman/sdf) (MIT). Portions are ported
from it; his copyright notice is retained in LICENSE.

## Install
    pip install -e .

**First use is slow.** numba compiles everything the first time it runs
(around 15–30 s, once per machine). The result is cached in
`~/.cache/sdfnumba` and numba's own cache, so later sessions start instantly.
Installing a new version triggers one more compile.

## Sverchok
Copy `sverchok/nodes/sdf_numba/` into `sverchok/nodes/`
and add the menu entry listed in `sverchok/README.md`.
Requires numba in Blender's Python.

## License
MIT. See LICENSE.
