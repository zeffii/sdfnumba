"""numba types shared by everything that evaluates a shape.

Every SDF function, whether the bytecode interpreter or code generated for
one particular shape, has the signature RUN:

    f(code, P, x, y, z, ps, ds, aux) -> distance

(generated code simply ignores code and the stacks).  RUN_ID is the same but
returns the index of the primitive that "won" at that point.

The heavy kernels take such a function as their first argument, typed as the
function-pointer type FN / FN_ID.  That way each kernel is compiled once and
works with any shape's function, instead of being recompiled per shape.
"""

from numba import types

f64 = types.float64
i64 = types.int64
i32 = types.int32
b1 = types.boolean

A1 = types.Array(f64, 1, 'C')
A2 = types.Array(f64, 2, 'C')
A3f = types.Array(types.float32, 3, 'C')
A1i = types.Array(i64, 1, 'C')
A2i = types.Array(i64, 2, 'C')
A3i32 = types.Array(i32, 3, 'C')
A1b = types.Array(b1, 1, 'C')

RUN = f64(A2i, A1, f64, f64, f64, A2, A1, A1)
RUN_ID = i64(A2i, A1, f64, f64, f64, A2, A1, A1, A1i)

FN = types.FunctionType(RUN)
FN_ID = types.FunctionType(RUN_ID)
