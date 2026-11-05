"""Time sdfnumba against fogleman's sdf on the same shapes and grid.

    python bench/bench_vs_fogleman.py [samples]

Both libraries share the API, so each shape is built once per library.
sdfnumba is warmed up first (numba compiles on first use and caches it).
"""

import sys
import time

import numpy as np

import sdf as fg
import sdfnumba as sn

SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 2 ** 22


def csg(m):
    f = m.sphere(1) & m.box(1.5)
    c = m.cylinder(0.5)
    return f - (c.orient(m.X) | c.orient(m.Y) | c)


def smooth_blobs(m):
    f = m.sphere(0.6)
    for i in range(12):
        a = i * 2 * np.pi / 12
        f = f | m.sphere(0.25).translate((np.cos(a), np.sin(a), 0.3 * np.sin(3 * a))).k(0.2)
    return f


def twisted_array(m):
    bar = m.rounded_box((0.3, 0.3, 2), 0.05).twist(1.5)
    return bar.translate((0.8, 0, 0)).circular_array(8) | m.torus(0.8, 0.1).k(0.1)


SHAPES = {'csg': csg, 'smooth_blobs': smooth_blobs, 'twisted_array': twisted_array}


def timed(fn):
    t = time.perf_counter()
    result = fn()
    return time.perf_counter() - t, result


def main():
    print(f'samples {SAMPLES}')
    print(f'{"shape":16} {"fogleman":>10} {"sdfnumba":>10} {"speedup":>8} {"sharp":>10}')
    for name, build in SHAPES.items():
        f_sn, f_fg = build(sn), build(fg)
        bounds = sn.estimate_bounds(f_sn.program())
        f_sn.generate(bounds=bounds, samples=2 ** 12)            # warm up
        f_sn.generate(bounds=bounds, samples=2 ** 12, sharp=True)
        t_fg, _ = timed(lambda: f_fg.generate(bounds=bounds, samples=SAMPLES,
                                              verbose=False))
        t_sn, _ = timed(lambda: f_sn.generate(bounds=bounds, samples=SAMPLES))
        t_dc, _ = timed(lambda: f_sn.generate(bounds=bounds, samples=SAMPLES, sharp=True))
        print(f'{name:16} {t_fg:9.2f}s {t_sn:9.2f}s {t_fg / t_sn:7.1f}x {t_dc:9.2f}s')


if __name__ == '__main__':
    main()
