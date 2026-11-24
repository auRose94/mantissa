#!/usr/bin/env python3
"""Show where a simulator and the reference disagree on a trace file.

    python3 tools/show_trace.py TRACE [STEP]      (STEP: the step to look around; default: the first that differs)

Runs build/<objdir>/m1_sim --first on the trace (any simulator built from this
directory), reads its report, and disassembles the instructions around the
first step whose lines differ, with the registers the reference shows for them.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mantissa import disasm  # noqa: E402
from mantissa.memory import Memory  # noqa: E402


def load(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = dict(line.split(" ", 1) for line in lines[:6])
    count = int(header["image"])
    cells = [int(c, 16) for line in lines[6:6 + (count + 15) // 16] for c in line.split()]
    return header, cells, lines[lines.index("run") + 1:]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    trace = sys.argv[1]
    sim = sys.argv[2] if len(sys.argv) > 2 else "build/obj_model/m1_sim"
    header, cells, run = load(trace)
    memory = Memory(int(header["cells"]))
    memory.write(cells)
    proc = subprocess.run([sim, trace, "--out", "/tmp/mantissa-fpga/show.got"], capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    got = Path("/tmp/mantissa-fpga/show.got").read_text(encoding="utf-8").splitlines()
    got = got[got.index("run") + 1:]
    where = None
    for i in range(max(len(run), len(got))):
        a = run[i] if i < len(run) else "<none>"
        b = got[i] if i < len(got) else "<none>"
        if a != b:
            where = i
            break
    if where is None:
        return
    # the step the differing line belongs to
    k = where
    while k >= 0 and not run[k].startswith("S "):
        k -= 1
    first = max(0, k - 12)
    print(f"\\nreference, from run line {first + 1} (the first difference is at line {where + 1}):")
    regs = {}
    for i in range(first, min(len(run), k + 6)):
        line = run[i]
        print("  ", line)
        if line.startswith("S "):
            pc = int(line.split()[2])
            try:
                text = disasm.disassemble(memory, pc)[0]
            except Exception as error:  # noqa: BLE001
                text = f"<{type(error).__name__}: {error}>"
            print(f"       {pc:6d}  {text}")
            for r, v in regs.items():
                pass
        elif line.startswith("R "):
            _, _, _, reg, bits = line.split()
            regs[reg] = int.from_bytes(bytes.fromhex(bits), "little")
    print("\\nwhat the reference had in the registers those instructions read:")
    for name in sorted(regs):
        import struct
        value = struct.unpack("<d", regs[name].to_bytes(8, "little"))[0]
        print(f"   {name} = {value!r}")


if __name__ == "__main__":
    main()
