#!/usr/bin/env python3
"""make_job.py -- build an array job for mantissa-gpu and check one pixel.

    python3 make_job.py OUT.bin [--width 64] [--height 24] [--cap 40]
                         [--left -2.0] [--right 0.6] [--bottom -1.05] [--top 1.05]
                         [--memory 256] [--steps 400]

Every machine of the job is one pixel: the host patches cx and cy into its
private memory, the machine iterates in Mantissa-1 instructions and stores
the escape count.  The job's program and layout come from PROGRAM below;
kernel/mandel_pixel.mant documents the same program.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo copy's root: mantissa/

from mantissa import asm  # noqa: E402

FMT_HALF, FMT_FLOAT, FMT_DOUBLE = 0, 1, 2  # the port's format indices, not cell widths

PROGRAM = """
        LDI.F  f1, {cx}
        LDI.F  f2, {cy}
        LDI.F  f3, {cap}
        LDI.F  f4, {out}
        LD.D   d0, [f1]
        LD.D   d1, [f2]
        LD.D   d2, [f3]
        LDI.D  d3, 0
        LDI.D  d4, 0
        LDI.D  d5, 0
        LDI.D  d6, 4
        LDI.D  d7, 1
        LDI.D  d13, 2
loop:   MUL.D  d8, d3, d3
        MUL.D  d9, d4, d4
        ADD.D  d10, d8, d9
        BLE.D  d6, d10, escaped
        BLE.D  d2, d5, escaped
        SUB.D  d11, d8, d9
        MUL.D  d12, d3, d4
        MUL.D  d12, d12, d13
        ADD.D  d4, d12, d1
        ADD.D  d3, d11, d0
        ADD.D  d5, d5, d7
        BR     loop
escaped: ST.D   [f4], d5
        HALT
"""


def build_job(args):
    """(job bytes, layout) for the window and size in args."""
    image = asm.assemble(PROGRAM.format(cx=160, cy=164, cap=168, out=172))
    assert len(image.cells) <= 160, len(image.cells)
    cells = list(image.cells)
    cap = fp_bits(args.cap)
    while len(cells) < 172:
        cells.append(0)
    # a double in memory is its cells most significant first, as DATA.D would lay it out
    cells[168:172] = list(struct.unpack(">4H", struct.pack(">Q", cap)))
    width, height = args.width, args.height
    instances = width * height
    patches = b""
    for row in range(height):
        for column in range(width):
            cx = args.left + (args.right - args.left) * column / (width - 1)
            cy = args.bottom + (args.top - args.bottom) * row / (height - 1)
            patches += struct.pack("<IIQ", 160, FMT_DOUBLE, fp_bits(cx))
            patches += struct.pack("<IIQ", 164, FMT_DOUBLE, fp_bits(cy))
    picks = struct.pack("<II", 172, FMT_DOUBLE) * instances
    header = b"M1AJ" + struct.pack(
        "<IIIQIIIIIII", 1, instances, args.memory, args.steps, 2, 1, len(cells), 0, 0, args.memory, 0
    )
    # version, instances, cells, max_steps (64 bits), n_inputs, n_outputs,
    # image_cells, org, entry, sp, spare: 52 bytes
    assert len(header) == 52, len(header)
    return header + patches + picks + bytes(struct.pack("<%dH" % len(cells), *cells)), instances


def fp_bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--cap", type=int, default=40)
    parser.add_argument("--left", type=float, default=-2.0)
    parser.add_argument("--right", type=float, default=0.6)
    parser.add_argument("--bottom", type=float, default=-1.05)
    parser.add_argument("--top", type=float, default=1.05)
    parser.add_argument("--memory", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    job, instances = build_job(args)
    Path(args.out).write_bytes(job)
    print(f"{args.out}: {instances} machines of {args.memory} cells, {len(job)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
