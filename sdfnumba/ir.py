"""Turn a shape tree into bytecode.

The tree is made of the node classes in api.py; each node knows how to emit
itself through an Emitter.  The result is a Program:

    code     int64 (n, 2)   (opcode, param offset) per instruction
    params   float64 (m,)   all parameter values
    key      bytes          identifies the code alone, so two trees that only
                            differ in parameter values share a key (and a
                            compiled kernel, see codegen.py)

`program.run` / `program.run_id` are the best functions available for it:
its generated code once compiled, the interpreter until then.
"""

import numpy as np

from . import ops
from .interp import STACK_DEPTH


class Emitter:
    def __init__(self):
        self.code = []
        self.params = []

    def op(self, name, *values):
        info = ops.REGISTRY[name]
        values = [float(v) for v in values]
        if len(values) != info.nparams:
            raise ValueError(f'{name} takes {info.nparams} params, got {len(values)}')
        self.code.append((info.code, len(self.params)))
        self.params.extend(values)

    def push_point(self):
        self.code.append((ops.PUSHP, 0))

    def pop_point(self):
        self.code.append((ops.POPP, 0))


class Program:
    def __init__(self, code, params):
        self.code = np.ascontiguousarray(code, dtype=np.int64).reshape(-1, 2)
        self.params = np.ascontiguousarray(params, dtype=np.float64)
        if self.params.size == 0:
            self.params = np.zeros(1)
        self.key = self.code.tobytes()
        _check_depth(self.code)
        # run_id reports a primitive by its row in `code`; this numbers the
        # primitives 0, 1, 2 ... in the order the shape built them
        prims = np.array([ops.REGISTRY_BY_CODE[c].kind == ops.PRIM
                          if c >= ops.FIRST_OP_CODE else False for c in self.code[:, 0]])
        self.prim_number = np.full(len(self.code), -1, dtype=np.int64)
        self.prim_number[prims] = np.arange(prims.sum())

    @property
    def run(self):
        from .codegen import functions
        return functions(self)[0]

    @property
    def run_id(self):
        from .codegen import functions
        return functions(self)[1]

    def compile(self):
        """Compile this shape's generated code now (it otherwise compiles in
        the background on first use) and wait for it."""
        from .codegen import compile
        compile(self)
        return self


def compile_tree(node):
    e = Emitter()
    node.emit(e)
    return Program(e.code, e.params)


def _check_depth(code):
    """Walk the code once and make sure no stack overflows."""
    kinds = {op.code: op.kind for op in ops.REGISTRY.values()}
    pt = dt = at = 0
    worst = 0
    for c, _ in code:
        if c == ops.PUSHP:
            pt += 1
        elif c == ops.POPP:
            pt -= 1
        else:
            k = kinds[c]
            if k == ops.PRIM:
                dt += 1
            elif k == ops.XFORM_AUX:
                at += 1
            elif k == ops.DUNARY_AUX:
                at -= 1
            elif k == ops.DBINARY:
                dt -= 1
        worst = max(worst, pt, dt, at)
    if dt != 1 or pt != 0 or at != 0:
        raise ValueError('malformed program')
    if worst > STACK_DEPTH:
        raise ValueError(f'shape nests {worst} deep; the limit is {STACK_DEPTH}')
