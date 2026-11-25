"""Golden traces: every effect a program has on the machine, one line at a time.

A port, the GPU or the FPGA core, replays the same image and must produce the
same lines.  The format:

    # mantissa-1 trace
    cells 65536            memory size in cells
    org 0                  where the image was loaded
    entry 0                the first pc
    sp 65536               f31 starts here
    image 120              how many cells the image has
    0000 0000 ...          the image, four hex digits per cell, most significant first
    run
    S 0 0                        step 0 begins at pc 0
    R 0 0 f5 3ff0000000000000    a register changed
    M 4 16 32 4 3ff0000000000000  a store
    P 5 20 4024000000000000       PUT handed a value over
    C 6 24 33                     PUTC printed a character
    X 3 12                       a step that changed nothing
    H                      halted.  F <kind> for a fault, L for the step limit.

Every value is the 64 bits of the double that holds it.

Stores alone tell memory apart from the image, so memory is not listed again.
Flags are not listed: they are not part of what a program can observe.  Nor is
the spelling of a printed value: PUT hands a value over, and how a console
writes it is the console's business.
"""
from __future__ import annotations

import math
import struct

from . import fp
from .machine import FAULT, HALTED, RUNNING, step

FILES = ("h", "f", "d", "q")  # registers are listed file by file, in this order


def bits(value):
    """The bits holding `value`, as hex digits: sixteen for the formats a
    double carries, thirty-two for quad."""
    from . import fp
    if isinstance(value, (fp.Quad, fp.Qnan)):
        return fp.to_bits(fp.Q, value).to_bytes(16, "little").hex()
    return struct.pack("<d", value).hex()


class TracedMemory:
    """A memory that remembers what was stored during the last step."""

    def __init__(self, memory):
        self._memory = memory
        self.stores = []

    def __getattr__(self, name):
        return getattr(self._memory, name)

    def store(self, fmt, addr, value):
        self._memory.store(fmt, addr, value)
        self.stores.append((addr, fmt.cells, value))


def _same(a, b):
    if isinstance(a, (fp.Quad, fp.Qnan)) or isinstance(b, (fp.Quad, fp.Qnan)):
        return a == b          # quads carry their own sign; dataclass equality is it
    return a is b or (a == b and math.copysign(1.0, a) == math.copysign(1.0, b))


def header(image, memory, cpu):
    yield "# mantissa-1 trace"
    yield f"cells {memory.size}"
    yield f"org {image.org}"
    yield f"entry {int(cpu.pc)}"
    yield f"sp {int(cpu.f[31])}"
    yield f"image {len(image.cells)}"
    for at in range(0, len(image.cells), 16):
        yield " ".join(f"{cell:04x}" for cell in image.cells[at:at + 16])


def effects(memory, cpu, max_steps):
    """Yield the run section: every step with its effects, then how it ended."""
    mem = TracedMemory(memory)
    handed = []
    steps = 0
    while cpu.status == RUNNING and steps < max_steps:
        mem.stores.clear()
        handed.clear()
        before = cpu
        cpu = step(cpu, mem, handed.append)
        pc = int(before.pc)
        changed = False
        yield f"S {steps} {pc}"
        for letter in FILES:
            old, new = getattr(before, letter), getattr(cpu, letter)
            for index, (a, b) in enumerate(zip(old, new)):
                if not _same(a, b):
                    yield f"R {steps} {pc} {letter}{index} {bits(b)}"
                    changed = True
        for addr, cells, value in mem.stores:
            yield f"M {steps} {pc} {int(addr)} {cells} {bits(value)}"
            changed = True
        for item in handed:
            if isinstance(item, str):
                yield f"C {steps} {pc} {ord(item)}"  # PUTC: a character
            else:
                yield f"P {steps} {pc} {bits(item)}"  # PUT: a value
            changed = True
        if not changed:
            yield f"X {steps} {pc}"
        steps += 1
    if cpu.status == HALTED:
        yield "H"
    elif cpu.status == FAULT:
        yield f"F {cpu.fault.kind}"
    else:
        yield "L"


def lines(image, memory, cpu, max_steps):
    """The whole trace as a list of lines."""
    return [*header(image, memory, cpu), "run", *effects(memory, cpu, max_steps)]
