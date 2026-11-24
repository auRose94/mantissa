"""Turning instruction numbers back into text: traces, listings and fault reports."""
from __future__ import annotations

from . import fp
from .isa import BY_NUMBER, decode, name_of

_THREE = {"ADD", "SUB", "MUL", "DIV", "FMA", "REM", "MOD", "MIN", "MAX", "CPSGN", "SCALB", "NEXT",
          "LT", "LE", "EQ", "CMP", "SPLIT", "JOIN", "MANT",
          "FLOOR_TO", "CEIL_TO", "CHOP_TO", "ROUND_TO",
          "VADD2", "VSUB2", "VMUL2", "VDIV2", "VFMA2", "VMIN2", "VMAX2",
          "VADD4", "VSUB4", "VMUL4", "VDIV4", "VFMA4", "VMIN4", "VMAX4"}
_TWO = {"MOV", "SQRT", "NEG", "ABS", "LOGB", "CLASS", "FLOOR", "CEIL", "CHOP", "ROUND"}
_RELATION = {"BEQ", "BNE", "BLT", "BLE"}


def _target(value, symbols):
    if symbols:
        for name, address in symbols.items():
            if address == value:
                return name
    return repr(value)


def disassemble(mem, addr, symbols=None):
    """(text, cells) for the instruction, or the stray number, at addr."""
    word = mem.operand(fp.D, addr)
    ins = decode(word)
    if ins is None:
        return f".double {word!r}", 4
    op, a, b, c = ins
    spec = BY_NUMBER[op]
    name = spec.name
    r = name_of

    def operand(at):
        return mem.operand(fp.D, at)

    if name.endswith("_TO"):
        text = f"{name[:-3]} {r(a)}, {r(b)}, {r(c)}"
    elif name == "LDI":
        text = f"LDI {r(a)}, {operand(addr + 4)!r}"
    elif name == "LD":
        text = f"LD {r(a)}, [{r(b)}]"
    elif name == "ST":
        text = f"ST [{r(a)}], {r(b)}"
    elif name == "CVT":
        text = f"CVT {r(a)}, {r(b)}"
    elif name in _THREE:
        text = f"{name} {r(a)}, {r(b)}, {r(c)}"
    elif name in _TWO:
        text = f"{name} {r(a)}, {r(b)}"
    elif name in ("PUT", "PUTC", "JMP"):
        text = f"{name} {r(a)}"
    elif name in ("BZ", "BNZ"):
        text = f"{name} {r(a)}, {_target(operand(addr + 4), symbols)}"
    elif name in _RELATION:
        text = f"{name} {r(a)}, {r(b)}, {_target(operand(addr + 4), symbols)}"
    elif name == "BSGN":
        text = f"BSGN {r(a)}, {_target(operand(addr + 4), symbols)}, {_target(operand(addr + 8), symbols)}"
    elif name == "JAL":
        text = f"JAL {r(a)}, {_target(operand(addr + 4), symbols)}"
    else:
        text = f"BR {_target(operand(addr + 4), symbols)}" if name == "BR" else name
    return text, spec.cells
