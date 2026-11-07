"""sdfnumba: signed distance fields to closed meshes, using numba.

Importing the package is cheap: its parts load (and numba compiles them, the
first time ever) when a name is first used, e.g. `sdfnumba.sphere`.  That
keeps Blender's startup quick when Sverchok imports the node modules.
"""

import importlib
import importlib.util
import types

__version__ = '0.1.0'

# the public names: everything public in api, plus these
_EXPORTS = {
    'mesh': ('generate', 'surface_nets', 'triangulate'),
    'remesh': ('remesh', 'quality'),
    'grid': ('evaluate', 'primitive_ids', 'sample_grid', 'estimate_bounds',
             'background_safe'),
}


def _public(module):
    """Names defined in one of this package's modules (not ones it imported)."""
    out = []
    for n in dir(module):
        value = getattr(module, n)
        if n.startswith('_') or isinstance(value, types.ModuleType):
            continue
        if getattr(value, '__module__', '').startswith(__name__):
            out.append(n)
    return out


_ALL = None


def _load():
    """Import the parts and bind the public names.  Done in one go so that
    e.g. `remesh` is the function, not the submodule of the same name."""
    global _ALL
    if _ALL is not None:
        return
    api = importlib.import_module('.api', __name__)
    names = _public(api) + ['ORIGIN', 'X', 'Y', 'Z', 'UP']
    for n in names:
        globals()[n] = getattr(api, n)
    for part, exports in _EXPORTS.items():
        module = importlib.import_module(f'.{part}', __name__)
        for n in exports:
            globals()[n] = getattr(module, n)
        names += exports
    _ALL = sorted(set(names))


def __getattr__(name):
    if name.startswith('__') and name != '__all__':
        raise AttributeError(name)
    # a submodule (sdfnumba.ops, sdfnumba.codegen ...) loads directly; going
    # through _load() here would make `from . import ops` circular.  Exported
    # functions win over submodules of the same name (remesh)
    exported = any(name in names for names in _EXPORTS.values())
    if not exported and importlib.util.find_spec(f'{__name__}.{name}') is not None:
        return importlib.import_module(f'.{name}', __name__)
    _load()
    if name == '__all__':
        return _ALL
    if name in globals():
        return globals()[name]
    raise AttributeError(f"module 'sdfnumba' has no attribute '{name}'")


def __dir__():
    _load()
    return sorted(set(globals()) | set(_ALL))
