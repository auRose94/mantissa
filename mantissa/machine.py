"""The processor: register files, the step function and the run loop."""
from __future__ import annotations

from dataclasses import dataclass

from . import fp
from .disasm import disassemble
from .fault import Fault
from .isa import BY_NUMBER, OP, decode, file_of

RUNNING, HALTED, FAULT = "running", "halted", "fault"
ILLEGAL = "illegal-instruction"
NOT_A_CHARACTER = "not-a-character"
MAX_CODE_POINT = 0x10FFFF
HALF, FLOAT, DOUBLE = 1, 2, 3  # register files, as numbered in operand fields

_ZERO = (0.0,) * 32


@dataclass(frozen=True, slots=True)
class Cpu:
    h: tuple = _ZERO
    f: tuple = _ZERO
    d: tuple = _ZERO
    pc: float = 0.0
    flags: int = 0
    status: str = RUNNING
    fault: Fault | None = None


def _file(cpu, number):
    if number == HALF:
        return cpu.h
    if number == FLOAT:
        return cpu.f
    return cpu.d


def _read(cpu, code):
    return _file(cpu, file_of(code))[code % 32]


def _write(cpu, code, value, pc, flags):
    number = file_of(code)
    index = code % 32
    regs = _file(cpu, number)
    regs = regs[:index] + (value,) + regs[index + 1:]
    return Cpu(regs if number == HALF else cpu.h, regs if number == FLOAT else cpu.f,
               regs if number == DOUBLE else cpu.d, pc, flags, RUNNING, None)


def _with_pc(cpu, pc):
    return Cpu(cpu.h, cpu.f, cpu.d, pc, cpu.flags, RUNNING, None)


def _format(code):
    return fp.FORMAT_BY_CODE[file_of(code)]


def _need(condition, message):
    if not condition:
        raise Fault(ILLEGAL, message)


# Two-register arithmetic: fn(fmt, x, y) -> (value, flags), keyed by opcode number.
_BINARY = {
    OP["ADD"]: fp.add, OP["SUB"]: fp.sub, OP["MUL"]: fp.mul, OP["DIV"]: fp.div,
    OP["REM"]: fp.remainder, OP["MOD"]: fp.fmod, OP["MIN"]: fp.fmin, OP["MAX"]: fp.fmax,
    OP["CPSGN"]: fp.copysign, OP["SCALB"]: fp.scaleb, OP["NEXT"]: fp.nextafter,
    OP["LT"]: fp.lt, OP["LE"]: fp.le, OP["EQ"]: fp.eq, OP["CMP"]: fp.compare,
    OP["FLOOR_TO"]: fp.floor_to, OP["CEIL_TO"]: fp.ceil_to,
    OP["CHOP_TO"]: fp.trunc_to, OP["ROUND_TO"]: fp.round_to,
}
# Vector arithmetic: fn(fmt, x, y) per lane, over consecutive registers.
# The destination may overlap a source: each lane reads only its own lanes.
_VECTOR = {
    OP["VADD2"]: (fp.add, 2), OP["VSUB2"]: (fp.sub, 2), OP["VMUL2"]: (fp.mul, 2),
    OP["VDIV2"]: (fp.div, 2), OP["VMIN2"]: (fp.fmin, 2), OP["VMAX2"]: (fp.fmax, 2),
    OP["VADD4"]: (fp.add, 4), OP["VSUB4"]: (fp.sub, 4), OP["VMUL4"]: (fp.mul, 4),
    OP["VDIV4"]: (fp.div, 4), OP["VMIN4"]: (fp.fmin, 4), OP["VMAX4"]: (fp.fmax, 4),
}

# One-register arithmetic: fn(fmt, x) -> (value, flags), keyed by opcode number.
_UNARY = {
    OP["SQRT"]: fp.sqrt, OP["NEG"]: fp.neg, OP["ABS"]: fp.fabs, OP["LOGB"]: fp.logb,
    OP["FLOOR"]: fp.floor, OP["CEIL"]: fp.ceil, OP["CHOP"]: fp.trunc,
    OP["ROUND"]: fp.round_even, OP["CLASS"]: fp.classify,
}
_RELATION = {OP["BEQ"]: 0, OP["BNE"]: 1, OP["BLT"]: 2, OP["BLE"]: 3}


def step(cpu, mem, out=None):
    """Advance the machine by one instruction.

    A fault ends the run and is recorded on the returned Cpu; registers keep
    their state from before the faulting instruction.
    """
    try:
        return _execute(cpu, mem, out)
    except Fault as fault:
        return Cpu(cpu.h, cpu.f, cpu.d, cpu.pc, cpu.flags, FAULT, fault)


def _execute(cpu, mem, out):
    pc = cpu.pc
    number = mem.fetch(pc)
    ins = decode(number)
    if ins is None:
        raise Fault(ILLEGAL, f"{number!r} is not an instruction")
    op, a, b, c = ins
    spec = BY_NUMBER[op]
    flags = cpu.flags
    if spec.same_type:
        kinds = {file_of(code) for place, code in zip("abc", (a, b, c)) if place in spec.used}
        _need(len(kinds) == 1, f"{spec.name} needs operands of one type")

    fn = _BINARY.get(op)
    if fn is not None:
        value, fl = fn(_format(a), _read(cpu, b), _read(cpu, c))
        return _write(cpu, a, value, pc + 4, flags | fl)

    if op in _VECTOR:
        fn, lanes = _VECTOR[op]
        for code in (a, b, c):
            if (code % 32) + lanes > 32:
                raise Fault(ILLEGAL, "a vector group runs past the end of its register file")
        cpu2 = cpu
        for lane in range(lanes):
            value, fl = fn(_format(a), _read(cpu2, b + lane), _read(cpu2, c + lane))
            cpu2 = _write(cpu2, a + lane, value, pc + 4, flags | fl)
            flags |= fl
        return cpu2

    if op in (OP["VFMA2"], OP["VFMA4"]):
        lanes = 2 if op == OP["VFMA2"] else 4
        for code in (a, b, c):
            if (code % 32) + lanes > 32:
                raise Fault(ILLEGAL, "a vector group runs past the end of its register file")
        cpu2 = cpu
        for lane in range(lanes):
            value, fl = fp.fma(_format(a), _read(cpu2, b + lane), _read(cpu2, c + lane),
                               _read(cpu2, a + lane))
            cpu2 = _write(cpu2, a + lane, value, pc + 4, flags | fl)
            flags |= fl
        return cpu2

    fn = _UNARY.get(op)
    if fn is not None:
        value, fl = fn(_format(a), _read(cpu, b))
        return _write(cpu, a, value, pc + 4, flags | fl)

    if op == OP["FMA"]:
        value, fl = fp.fma(_format(a), _read(cpu, b), _read(cpu, c), _read(cpu, a))
        return _write(cpu, a, value, pc + 4, flags | fl)

    if op == OP["MOV"]:
        return _write(cpu, a, _read(cpu, b), pc + 4, flags)

    if op == OP["LDI"]:
        value, fl = fp.convert(_format(a), mem.operand(fp.D, pc + 4))
        return _write(cpu, a, value, pc + spec.cells, flags | fl)

    if op == OP["LD"]:
        _need(file_of(b) == FLOAT, "LD reads its address from a float register")
        return _write(cpu, a, mem.load(_format(a), _read(cpu, b)), pc + 4, flags)

    if op == OP["ST"]:
        _need(file_of(a) == FLOAT, "ST reads its address from a float register")
        mem.store(_format(b), _read(cpu, a), _read(cpu, b))
        return _with_pc(cpu, pc + 4)

    if op == OP["CVT"]:
        _need(file_of(a) != file_of(b), "CVT changes type, so its registers must differ in type")
        value, fl = fp.convert(_format(a), _read(cpu, b))
        return _write(cpu, a, value, pc + 4, flags | fl)

    if op == OP["SPLIT"]:
        _need(file_of(a) == file_of(b), "SPLIT writes its two halves to registers of one type")
        high, low, fl = fp.split(_format(a), _read(cpu, c))
        return _write(_write(cpu, a, high, pc + 4, flags | fl), b, low, pc + 4, flags | fl)

    if op == OP["JOIN"]:
        _need(file_of(b) == file_of(c), "JOIN reads its two parts from registers of one type")
        value, fl = fp.join(_format(a), _read(cpu, b), _read(cpu, c))
        return _write(cpu, a, value, pc + 4, flags | fl)

    if op == OP["MANT"]:
        m, e = fp.mantissa(_read(cpu, c))
        m, fl = fp.convert(_format(a), m)
        e, fl2 = fp.convert(_format(b), e)
        return _write(_write(cpu, a, m, pc + 4, flags | fl | fl2), b, e, pc + 4, flags | fl | fl2)

    if op == OP["PUT"]:
        if out is not None:
            out(_read(cpu, a))
        return _with_pc(cpu, pc + 4)

    if op == OP["PUTC"]:
        _putc(_read(cpu, a), out)
        return _with_pc(cpu, pc + 4)

    if op == OP["BZ"] or op == OP["BNZ"]:
        target = mem.operand(fp.D, pc + 4)
        if (_read(cpu, a) == 0.0) == (op == OP["BZ"]):
            mem.check_fetch(target)
            return _with_pc(cpu, target)
        return _with_pc(cpu, pc + spec.cells)

    return _system(cpu, mem, pc, op, a, b, flags)


def _putc(value, out):
    if not value.is_integer() or not 0 <= value <= MAX_CODE_POINT \
            or 0xD800 <= value <= 0xDFFF:
        raise Fault(NOT_A_CHARACTER, f"{value!r} is not a Unicode code point")
    if out is not None:
        out(chr(int(value)))


def console(write):
    """An `out` for run() that prints the way the command line does: PUT prints
    its value as Python prints a float, the shortest text that reads back as
    the same number, and PUTC prints its character.  How a value is spelled
    is the console's business; the machine only hands the value over."""
    def out(item):
        write(f"{item!r}\n" if isinstance(item, float) else item)
    return out


def _system(cpu, mem, pc, op, a, b, flags):
    if op == OP["NOP"]:
        return _with_pc(cpu, pc + 4)
    if op == OP["HALT"]:
        return Cpu(cpu.h, cpu.f, cpu.d, pc, flags, HALTED, None)
    if op == OP["JMP"]:
        _need(file_of(a) == FLOAT, "JMP reads its target from a float register")
        target = _read(cpu, a)
        mem.check_fetch(target)
        return _with_pc(cpu, target)
    if op == OP["JAL"]:
        _need(file_of(a) == FLOAT, "JAL leaves its return address in a float register")
        target = mem.operand(fp.D, pc + 4)
        mem.check_fetch(target)
        return _write(cpu, a, pc + 8, target, flags)
    if op == OP["BR"]:
        target = mem.operand(fp.D, pc + 4)
        mem.check_fetch(target)
        return _with_pc(cpu, target)
    if op == OP["BSGN"]:
        # Ternary: negative jumps to the first target, positive to the second,
        # zero (or NaN, which is neither) falls through.
        value = _read(cpu, a)
        if value < 0:
            target = mem.operand(fp.D, pc + 4)
        elif value > 0:
            target = mem.operand(fp.D, pc + 8)
        else:
            return _with_pc(cpu, pc + 12)
        mem.check_fetch(target)
        return _with_pc(cpu, target)

    # the remaining instructions are the compare-and-branch ones
    x, y = _read(cpu, a), _read(cpu, b)
    taken = (x == y, x != y, x < y, x <= y)[_RELATION[op]]
    if taken:
        target = mem.operand(fp.D, pc + 4)
        mem.check_fetch(target)
        return _with_pc(cpu, target)
    return _with_pc(cpu, pc + 8)


@dataclass(frozen=True)
class RunResult:
    cpu: Cpu
    steps: int
    reason: str  # "halt", "fault" or "step limit"


def run(cpu, mem, out=None, max_steps=1_000_000, trace=None):
    """Step until the machine stops on its own or reaches max_steps.

    `out` receives the program's output as it is produced: a float from each
    PUT, a one-character string from each PUTC.  `trace`, if given, is called
    with (pc, text) before each instruction.
    """
    steps = 0
    while cpu.status == RUNNING and steps < max_steps:
        if trace is not None:
            trace(cpu.pc, disassemble(mem, cpu.pc)[0])
        cpu = step(cpu, mem, out)
        steps += 1
    reason = {HALTED: "halt", FAULT: "fault", RUNNING: "step limit"}[cpu.status]
    return RunResult(cpu, steps, reason)


def describe(cpu):
    """A few lines for a front panel: the registers that hold something, and the flags."""
    lines = []
    for number, letter in ((HALF, "h"), (FLOAT, "f"), (DOUBLE, "d")):
        regs = _file(cpu, number)
        lines.append(f"{letter}: " + " ".join(f"{letter}{i}={v!r}" for i, v in enumerate(regs) if v))
    return lines
