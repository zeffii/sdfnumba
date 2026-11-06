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

**First use is slow.** numba compiles everything the first time
`sdfnumba` is imported (around 25 s, once per machine). The result is cached
in `~/.cache/sdfnumba` and numba's own cache, so later sessions start
instantly. Installing a new version triggers one more compile.

Each new shape structure also gets its own compiled code, built in a
background process in a second or two; until it is ready the shape runs on
a slower interpreter. Changing only parameter values never recompiles.

## Sverchok
Link `sverchok/nodes/sdf_numba/` and `sverchok/utils/sdfnumba_helper.py`
into Sverchok and add the menu entries; see `sverchok/README.md`.
Requires numba in Blender's Python.

More on the nodes' settings: [ADDITIONAL_README.md](ADDITIONAL_README.md).

## License
MIT. See LICENSE.
