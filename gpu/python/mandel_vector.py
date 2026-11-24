"""mandel_vector.py -- four pixels per machine: the vector Mandelbrot.

    python3 mandel_vector.py [--width 64] [--height 24] [--cap 40]
                             [--left -2.0] [--right 0.6] [--bottom -1.05]
                             [--top 1.05]

Builds two array jobs over the same window -- the scalar kernel (one pixel
per machine, from make_job.py) and a vector kernel (four pixels per machine,
one lane per pixel) -- runs both on the GPU, and checks that the pictures
agree while counting what each one cost.

The vector loop carries zx, zy, cx, cy, the count and the predicate as four
lanes: zx in d0..d3, zy in d4..d7, n in d8..d11, cx in d12..d15, cy in
d16..d19, scratch d20..d27, and the frozen predicate in d28..d31.  There is
no per-lane escape: every lane iterates the full cap, and the count freezes
because the predicate -- MIN(previous, dist < 4), four scalar LTs against
4.0 -- goes to zero when the lane first escapes and stays there even if its
orbit wanders back inside |z| < 2.  The 4.0 lives in d23 for the four LTs,
borrowed after zx' has moved out; a lane stores the iteration it escaped on,
exactly what the scalar kernel stores.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mantissa import asm  # noqa: E402

FMT_DOUBLE = 2
RAMP = " .:-=+*#%@"


def vec_program(base, iterations):
    """The four-pixel kernel.  The escape check reads the distance of the
    CURRENT z before each update -- since an escaped orbit never comes back,
    no frozen predicate is needed.  The count starts at 1 (the initial z is
    inside) and gains 1 per iteration whose distance is still below 4, so a
    lane stores the iteration it escaped on and interior lanes store
    `iterations + 1`.  The builder passes cap - 1 as the iteration count.

    Data layout, from `base`:
        base + 0,  8, 16, 24   cx0..cx3     patched
        base + 32, 40, 48, 56  cy0..cy3     patched
        base + 64, 72, 80, 88  out counts
    """
    return f"""
        LDI.D  d28, 4           ; the 4.0 the LTs compare against
        LDI.D  d8, 0
        MOV.D  d9, d8
        MOV.D  d10, d8
        MOV.D  d11, d8          ; n = 0: the loop's 39 checks cover z0..z38
        LDI.H  h9, {iterations}
        LDI.H  h10, 1
        LDI.F  f1, {base}
        LD.D   d12, [f1]        ; cx0
        LDI.F  f1, {base + 8}
        LD.D   d13, [f1]        ; cx1
        LDI.F  f1, {base + 16}
        LD.D   d14, [f1]        ; cx2
        LDI.F  f1, {base + 24}
        LD.D   d15, [f1]        ; cx3
        LDI.F  f1, {base + 32}
        LD.D   d16, [f1]        ; cy0
        LDI.F  f1, {base + 40}
        LD.D   d17, [f1]        ; cy1
        LDI.F  f1, {base + 48}
        LD.D   d18, [f1]        ; cy2
        LDI.F  f1, {base + 56}
        LD.D   d19, [f1]        ; cy3
        LDI.D  d0, 0
        MOV.D  d1, d0
        MOV.D  d2, d0
        MOV.D  d3, d0           ; zx = 0, 0, 0, 0
        LDI.D  d4, 0
        MOV.D  d5, d4
        MOV.D  d6, d4
        MOV.D  d7, d4           ; zy = 0
iter:   VMUL4  d20, d0, d0      ; zx * zx
        VMUL4  d24, d4, d4      ; zy * zy
        VSUB4  d20, d20, d24    ; u = zx*zx - zy*zy
        VADD4  d24, d24, d24    ; 2 * zy * zy
        VADD4  d24, d20, d24    ; dist = u + 2*zy*zy = zx*zx + zy*zy
        LT.D   d24, d24, d28    ; the predicate per lane: dist < 4, in place
        LT.D   d25, d25, d28
        LT.D   d26, d26, d28
        LT.D   d27, d27, d28
        VADD4  d8, d8, d24      ; n += the predicate (0 once escaped: monotone)
        VADD4  d20, d20, d12    ; zx' = u + cx
        VMUL4  d24, d0, d4      ; zx * zy, the old lanes
        VADD4  d24, d24, d24    ; 2 * zx * zy, by self-add
        VADD4  d4, d24, d16     ; zy' = 2*zx*zy + cy
        MOV.D  d0, d20          ; zx = zx'
        MOV.D  d1, d21
        MOV.D  d2, d22
        MOV.D  d3, d23
        SUB.H  h9, h9, h10
        BNZ.H  h9, iter
        LDI.F  f1, {base + 64}
        ST.D   [f1], d8
        LDI.F  f1, {base + 72}
        ST.D   [f1], d9
        LDI.F  f1, {base + 80}
        ST.D   [f1], d10
        LDI.F  f1, {base + 88}
        ST.D   [f1], d11
        HALT
"""


def sized_program(cap):
    """The program assembled at its real data base: assemble once to learn the
    length, then again with the base just past the code."""
    guess = vec_program(256, cap)
    base = (len(asm.assemble(guess).cells) + 7) // 8 * 8
    text = vec_program(base, cap)
    program = asm.assemble(text)
    return program, base, base + 64


def pack_double(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def job(instances, inputs, outputs, image, memory, max_steps):
    n_inputs = len(inputs[0]) if inputs else 0
    n_outputs = len(outputs)
    patches = b"".join(
        struct.pack("<IIQ", cell, FMT_DOUBLE, pack_double(value))
        for row in inputs for cell, value in row)
    picks = b"".join(struct.pack("<II", cell, FMT_DOUBLE)
                     for _ in range(instances) for cell, _ in outputs)
    header = b"M1AJ" + struct.pack("<IIIQIIIIIII", 1, instances, memory, max_steps,
                                   n_inputs, n_outputs, len(image.cells), 0, 0, memory, 0)
    return (header + patches + picks
            + bytes(struct.pack("<%dH" % len(image.cells), *image.cells)))


def build_vector_job(args):
    program, base, out = sized_program(args.cap)
    cells = list(program.cells)
    while len(cells) < base + 96:
        cells.append(0)
    memory = base + 96
    groups_x, groups_y = args.width // 4, args.height
    instances = groups_x * groups_y
    inputs = []
    for row in range(groups_y):
        for gx in range(groups_x):
            one = []
            for lane in range(4):
                col = gx * 4 + lane
                cx = args.left + (args.right - args.left) * col / (args.width - 1)
                cy = args.bottom + (args.top - args.bottom) * row / (args.height - 1)
                one.append((base + 8 * lane, cx))
                one.append((base + 32 + 8 * lane, cy))
            inputs.append(one)
    outputs = [(out + 8 * k, 0) for k in range(4)]
    job_bytes = job(instances, inputs, outputs, program, memory, args.steps)
    return job_bytes, instances


def run_gpu(job_path, result_path):
    done = subprocess.run(["gpu/build/mantissa-gpu", "array", job_path, "--out", result_path],
                          capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"array run failed: {done.stderr}")
    for line in done.stdout.splitlines():
        if "instructions ran" in line:
            print("  ", line.strip())


def read_result(path, instances, outputs):
    data = Path(path).read_bytes()
    rows, at = [], 16
    for _ in range(instances):
        at += 8 + 16
        values = struct.unpack(f"<{outputs}d", data[at:at + 8 * outputs])
        at += 8 * outputs
        rows.append(values)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--cap", type=int, default=40)
    parser.add_argument("--left", type=float, default=-2.0)
    parser.add_argument("--right", type=float, default=0.6)
    parser.add_argument("--bottom", type=float, default=-1.05)
    parser.add_argument("--top", type=float, default=1.05)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--memory", type=int, default=256)
    args = parser.parse_args()
    if args.width % 4:
        sys.exit("the width must be a multiple of 4")

    sys.path.insert(0, ".")
    import make_job

    scalar_job, scalar_instances = make_job.build_job(args)
    Path("/tmp/mv-scalar.bin").write_bytes(scalar_job)
    print(f"scalar: one pixel per machine ({scalar_instances} machines)")
    run_gpu("/tmp/mv-scalar.bin", "/tmp/mv-scalar.result")
    scalar = read_result("/tmp/mv-scalar.result", scalar_instances, 1)
    scalar_counts = [int(round(v[0])) for v in scalar]

    vector_job, vector_instances = build_vector_job(args)
    Path("/tmp/mv-vector.bin").write_bytes(vector_job)
    print(f"vector: four pixels per machine ({vector_instances} machines)")
    run_gpu("/tmp/mv-vector.bin", "/tmp/mv-vector.result")
    vector = read_result("/tmp/mv-vector.result", vector_instances, 4)
    vector_counts = [int(round(v)) for row in vector for v in row]

    same = sum(1 for a, b in zip(scalar_counts, vector_counts) if a == b)
    print(f"pictures: {same} of {len(scalar_counts)} pixels agree "
          f"({len(scalar_counts) - same} differ)")

    for row in range(args.height):
        line_s = "".join(RAMP[min(scalar_counts[row * args.width + c], 9)]
                         for c in range(args.width))
        line_v = "".join(RAMP[min(vector_counts[row * args.width + c], 9)]
                         for c in range(args.width))
        mark = "" if line_s == line_v else "   <-- differs"
        print(line_s + ("  |  " + line_v + mark if mark else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
