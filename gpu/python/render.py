#!/usr/bin/env python3
"""render.py -- turn an array result into a picture, and cross-check a sample.

    python3 render.py RESULT.bin --width 64 --height 24 [--crosscheck 60]

Reads the M1AR file mantissa-gpu array writes, prints the Mandelbrot picture
the machines drew, and replays --crosscheck instances in the Python reference
to confirm the GPU's counts.
"""
from __future__ import annotations

import argparse
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mantissa import asm, fp  # noqa: E402
from mantissa.machine import Cpu, run  # noqa: E402
from mantissa.memory import Memory  # noqa: E402

RAMP = " .:-=+*#%@"
WINDOW = (-2.0, 0.6, -1.05, 1.05)  # left, right, bottom, top: as make_job.py defaults


def read_result(path, instances, outputs):
    """(steps, status, values) per instance, from the M1AR file."""
    data = Path(path).read_bytes()
    if data[:4] != b"M1AR":
        sys.exit(f"{path}: not an array result file")
    rows, at = [], 16
    for _ in range(instances):
        steps = struct.unpack("<Q", data[at:at + 8])[0]
        at += 8
        status = struct.unpack("<III", data[at:at + 12])
        at += 12
        _pad = struct.unpack("<I", data[at:at + 4])
        at += 4
        values = struct.unpack(f"<{outputs}d", data[at:at + 8 * outputs])
        at += 8 * outputs
        rows.append((steps, status, values))
    return rows


def pixel_job(width, height, cap, steps):
    """The per-pixel program, assembled, with the memory layout make_job.py uses."""
    source = (Path(__file__).with_name("make_job.py")).read_text()
    program = source.split('PROGRAM = """')[1].split('"""')[0]
    program = program.format(cx=160, cy=164, cap=168, out=172)
    image = asm.assemble(program)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--cap", type=int, default=40)
    parser.add_argument("--crosscheck", type=int, default=60)
    args = parser.parse_args()
    instances = args.width * args.height
    rows = read_result(args.result, instances, 1)
    counts = [min(int(round(row[2][0])), 9) for row in rows]
    for row in range(args.height):
        print("".join(RAMP[c] for c in counts[row * args.width:(row + 1) * args.width]))
    inside = sum(1 for row in rows if int(round(row[2][0])) >= args.cap)
    halted = sum(1 for row in rows if row[1][0] == 1)
    print(f"picture: {args.width}x{args.height}, {inside} pixels never escaped, {halted} of {instances} halted")

    if not args.crosscheck:
        return 0
    image = pixel_job(args.width, args.height, args.cap, 1000)
    rng = random.Random(1)
    bad = 0
    left, right, bottom, top = WINDOW
    for _ in range(args.crosscheck):
        k = rng.randrange(instances)
        row, col = divmod(k, args.width)
        cx = left + (right - left) * col / (args.width - 1)
        cy = bottom + (top - bottom) * row / (args.height - 1)
        memory = Memory(256)
        memory.write(image.cells)
        memory.store(fp.D, 160.0, cx)
        memory.store(fp.D, 164.0, cy)
        memory.store(fp.D, 168.0, float(args.cap))
        run(Cpu(pc=image.entry, f=(0.0,) * 31 + (256.0,)), memory, max_steps=1000)
        want = int(memory.load(fp.D, 172.0))
        if want != int(round(rows[k][2][0])):
            bad += 1
            print(f"MISMATCH instance {k} at ({cx!r}, {cy!r}): GPU {rows[k][2][0]}, reference {want}")
    print(f"cross-check: {args.crosscheck} instances replayed in the reference, "
          f"{args.crosscheck - bad} agree, {bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
