"""Command line:

    python -m mantissa run prog.mant [--steps N] [--cells N] [--trace] [--panel]
    python -m mantissa disasm prog.mant
    python -m mantissa trace prog.mant --out prefix    (golden files for the ports)

Program output goes to stdout; notes from the simulator go to stderr.
"""
from __future__ import annotations

import argparse
import sys

from . import asm, fp, machine, trace
from .disasm import disassemble
from .fault import Fault
from .isa import BY_NUMBER, decode, name_of
from .machine import Cpu, FAULT, HALTED
from .memory import Memory

EXIT_CODES = {"halt": 0, "fault": 1, "step limit": 2}


def _load(args):
    try:
        with open(args.file, encoding="utf-8") as handle:
            source = handle.read()
    except OSError as error:
        sys.exit(str(error))
    try:
        image = asm.assemble(source, args.org)
    except asm.AsmError as error:
        sys.exit(f"{args.file}:{error.lineno}: {error.message}")
    memory = Memory(args.cells)
    floats = [0.0] * 32
    floats[31] = float(args.cells)  # the stack pointer starts at the top of memory
    try:
        memory.write(image.cells, args.org)
    except ValueError as error:
        sys.exit(str(error))
    return image, memory, Cpu(f=tuple(floats), pc=image.entry)


def _note(text):
    print(f"; {text}", file=sys.stderr, flush=True)


def _run(args):
    image, memory, cpu = _load(args)

    def show(pc, text):
        _note(f"{pc:6.0f}  {text}")

    last = [""]  # the last character the program printed

    def write(text):
        sys.stdout.write(text)
        last[0] = text[-1] if text else last[0]

    result = machine.run(cpu, memory, out=machine.console(write), max_steps=args.steps,
                         trace=show if args.trace else None)
    sys.stdout.flush()
    if last[0] not in ("", "\n"):  # a fault mid-line: start the report on its own line
        print(file=sys.stderr)
    cpu = result.cpu
    if cpu.status == HALTED:
        _note(f"halted: HALT at {cpu.pc:.0f} after {result.steps} steps")
    elif cpu.status == FAULT:
        try:
            at = disassemble(memory, cpu.pc)[0]
        except Fault:  # the pc itself is not a place an instruction could be
            at = ""
        where = f" ({at})" if at and not at.startswith(".double") else ""
        _note(f"fault: {cpu.fault.kind} at {cpu.pc:.0f}{where}: {cpu.fault.detail}")
    else:
        _note(f"stopped: step limit {args.steps} reached while still running at {cpu.pc:.0f}")
    if args.panel:
        for line in machine.describe(cpu):
            _note(line)
        _note(f"pc: {cpu.pc:.0f}  flags: {' '.join(fp.flag_names(cpu.flags)) or 'none'}")
    return EXIT_CODES[result.reason]


def _grouped(number):
    """20000102 -> 20 00 01 02, so the digit fields are easy to tell apart."""
    return " ".join(f"{number // 10 ** k % 100:02d}" for k in (6, 4, 2, 0))


def _listing(args):
    image, memory, _ = _load(args)
    symbols = {name: address for name, address in image.symbols.items() if name not in asm.PREDEFINED}
    addr = args.org
    end = args.org + len(image.cells)
    while addr < end:
        names = [name for name, address in symbols.items() if address == addr]
        ins = decode(memory.operand(fp.D, addr)) if addr % 4 == 0 else None
        if ins is not None:
            text, width = disassemble(memory, addr, symbols)
            number = _grouped(int(memory.operand(fp.D, addr)))
        else:
            value = fp.from_bits(fp.H, memory.cells[addr])  # any whole cell reads as a half
            char = f"  {chr(int(value))!r}" if value.is_integer() and 32 <= value < 127 else ""
            number, text, width = "         ", f".half {value!r}{char}", 1
        print(f"{addr:6d}  {number}  {' '.join(names + [text])}")
        addr += width


def _trace(args):
    image, memory, cpu = _load(args)
    lines = trace.lines(image, memory, cpu, args.steps)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    steps = sum(line.startswith("S ") for line in lines)
    _note(f"{args.out}: {steps} steps, {len(image.cells)} cells of image, ends '{lines[-1]}'")
    return 0  # a run that faults was recorded as well as one that halts


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mantissa",
        description="Simulator for the Mantissa-1, a computer with no integers and no bytes.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("run", "assemble a program and run it"),
                            ("disasm", "assemble a program and list it"),
                            ("trace", "assemble a program and record a golden trace of it")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("file", help="assembly source, usually .mant")
        sub.add_argument("--org", type=int, default=0,
                         help="cell address the program is loaded at (default 0)")
        sub.add_argument("--cells", type=int, default=65536, help="memory size in cells (default 65536)")
        sub.add_argument("--steps", type=int, default=1_000_000,
                         help="give up after this many instructions (default 1000000)")
    runner = commands.choices["run"]
    runner.add_argument("--trace", action="store_true", help="show each instruction as it runs")
    runner.add_argument("--panel", action="store_true", help="show registers and flags at the end")
    commands.choices["trace"].add_argument("--out", required=True, help="file to write the trace to")
    args = parser.parse_args(argv)
    return {"run": _run, "disasm": _listing, "trace": _trace}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
