"""The assembler: source text in, a memory image out.

    label:  LDI   f0, buffer      ; an instruction: an eight-digit number
            DATA.H 1, 2, 3        ; half constants, one cell each
            TEXT   "hello\\n"     ; one code point per cell
            SPACE  16             ; cells held in reserve, all zero

A literal is rounded to its register's format the way C rounds one.  The
suffix on an instruction (.H, .F, .D or .Q) is optional, because every register
already says what type it holds; if you write one, every register must match.
"""
from __future__ import annotations

import re
from array import array
from dataclasses import dataclass

from . import fp
from .isa import VECTOR_LANES, BY_NAME, WITH_STEP, code_of, encode

FORMAT_BY_SUFFIX = {"H": fp.H, "F": fp.F, "D": fp.D, "Q": fp.Q}
KIND_BY_FILE = {1: "half", 2: "float", 3: "double", 4: "quad"}
ALIASES = {"sp": 31, "lr": 30}  # by convention, in float registers: stack pointer and link
PREDEFINED = {"inf": fp.INF, "nan": fp.NAN}

# Operand kinds.  R: a register.  A: a float register holding an address.
# M: "[fN]".  E: a number or label expression, which becomes an inline operand.
# An instruction with several forms takes the one with the right number of operands.
_SAME3 = ("ADD SUB MUL DIV FMA REM MOD MIN MAX CPSGN SCALB NEXT LT LE EQ CMP "
          "VADD2 VSUB2 VMUL2 VDIV2 VFMA2 VMIN2 VMAX2 "
          "VADD4 VSUB4 VMUL4 VDIV4 VFMA4 VMIN4 VMAX4").split()
FORMS = {
    "HALT": [""], "NOP": [""],
    "MOV": ["RR"], "LDI": ["RE"], "LD": ["RM"], "ST": ["MR"], "CVT": ["RR"],
    "SPLIT": ["RRR"], "JOIN": ["RRR"], "MANT": ["RRR"],
    "SQRT": ["RR"], "NEG": ["RR"], "ABS": ["RR"], "LOGB": ["RR"], "CLASS": ["RR"],
    "FLOOR": ["RR", "RRR"], "CEIL": ["RR", "RRR"], "CHOP": ["RR", "RRR"], "ROUND": ["RR", "RRR"],
    "BR": ["E"], "BSGN": ["REE"], "BEQ": ["RRE"], "BNE": ["RRE"], "BLT": ["RRE"], "BLE": ["RRE"],
    "BZ": ["RE"], "BNZ": ["RE"], "JAL": ["AE"], "JMP": ["A"],
    "PUT": ["R"], "PUTC": ["R"],
}
for _name in _SAME3:
    FORMS[_name] = ["RRR"]
# the form with a step register is written with the same mnemonic
STEP_FORMS = {"FLOOR": "FLOOR_TO", "CEIL": "CEIL_TO", "CHOP": "CHOP_TO", "ROUND": "ROUND_TO"}

_LABEL = re.compile(r"\s*([A-Za-z_]\w*)\s*:")
_STATEMENT = re.compile(r"([A-Za-z][A-Za-z0-9]*)(?:\.([A-Za-z]+))?\s*(.*)$")
_REGISTER = re.compile(r"([hfdq])(\d{1,2})")
_MEMORY = re.compile(r"\[\s*([A-Za-z_]\w*)\s*\]")
_TOKEN = re.compile(r"[+\-]|[A-Za-z_]\w*|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_STRING = re.compile(r'"((?:\\.|[^"\\])*)"$')
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"'}


class AsmError(Exception):
    def __init__(self, lineno, message):
        super().__init__(message)
        self.lineno = lineno
        self.message = message


class _Undefined(Exception):
    pass


@dataclass
class Image:
    org: int
    cells: array
    symbols: dict
    start: float  # address of the first instruction

    @property
    def entry(self):
        """Where execution starts: `main` if there is one, else the first instruction."""
        return self.symbols.get("main", self.start)


def _strip_comment(line):
    out = []
    quoted = False
    i = 0
    while i < len(line):
        char = line[i]
        if quoted:
            if char == "\\":
                out.append(line[i:i + 2])
                i += 2
                continue
            if char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == ";":
            break
        out.append(char)
        i += 1
    return "".join(out)


def _operands(text, lineno):
    parts = [part.strip() for part in text.split(",")]
    if parts == [""]:
        return []
    if not all(parts):
        raise AsmError(lineno, "an operand is missing")
    return parts


def _expression(text, lineno):
    """Parse 'buf + 4' into signed terms: [(1.0, 'buf'), (1.0, '4')]."""
    text = text.strip()
    items = []
    sign, need_term, pos = 1.0, True, 0
    for match in _TOKEN.finditer(text):
        if text[pos:match.start()].strip():
            break
        token = match[0]
        if token == "+" or token == "-":
            if not need_term:
                sign = 1.0
            if token == "-":
                sign = -sign
            need_term = True
        else:
            if not need_term:
                raise AsmError(lineno, f"missing operator in '{text}'")
            items.append((sign, token))
            sign, need_term = 1.0, False
        pos = match.end()
    if text[pos:].strip() or need_term or not items:
        raise AsmError(lineno, f"cannot read '{text}' as a number or a label")
    return items


def _evaluate(items, symbols):
    def term(text):
        if text[0].isdigit() or text[0] == ".":
            return float(text)
        if text in symbols:
            return symbols[text]
        raise _Undefined(text)

    # Start from the first term, not from 0.0, so that "-0" stays a negative zero.
    total = items[0][0] * term(items[0][1])
    for sign, text in items[1:]:
        total += sign * term(text)
    return total


def _unescape(body, lineno):
    out = []
    i = 0
    while i < len(body):
        if body[i] != "\\":
            out.append(body[i])
            i += 1
            continue
        if i + 1 == len(body):
            raise AsmError(lineno, "a string cannot end in a lone backslash")
        esc = body[i + 1]
        if esc in _ESCAPES:
            out.append(_ESCAPES[esc])
            i += 2
        elif esc in "xu":
            digits = 2 if esc == "x" else 4
            try:
                out.append(chr(int(body[i + 2:i + 2 + digits], 16)))
            except ValueError:
                raise AsmError(lineno, f"cannot read the \\{esc} escape") from None
            i += 2 + digits
        else:
            raise AsmError(lineno, f"unknown escape \\{esc}")
    return "".join(out)


def _register_name(file_number, index):
    if file_number == 4:
        return f"q{index}"
    return "hfd"[file_number - 1] + str(index)


class _Assembler:
    def __init__(self, org):
        self.org = org
        self.cells = array("H")
        self.symbols = dict(PREDEFINED)
        self.pending = []  # labels waiting for the next thing that occupies cells
        self.fixups = []   # values to write once a forward label is known
        self.start = None

    # -- space ---------------------------------------------------------------

    def here(self):
        return self.org + len(self.cells)

    def align(self, n):
        while self.here() % n:
            self.cells.append(0)

    def bind(self):
        for name, lineno in self.pending:
            if name in self.symbols:
                raise AsmError(lineno, f"'{name}' is defined twice")
            self.symbols[name] = float(self.here())
        self.pending = []

    def label(self, name, lineno):
        if _REGISTER.fullmatch(name.lower()) or name.lower() in ALIASES or name in PREDEFINED:
            raise AsmError(lineno, f"'{name}' names a register or a constant, so it cannot be a label")
        self.pending.append((name, lineno))

    # -- directives ------------------------------------------------------------

    def data(self, suffix, text, lineno):
        if suffix not in FORMAT_BY_SUFFIX:
            raise AsmError(lineno, "write DATA.H, DATA.F or DATA.D")
        fmt = FORMAT_BY_SUFFIX[suffix]
        for operand in _operands(text, lineno):
            self.align(fmt.cells)
            self.bind()
            self.value(fmt, _expression(operand, lineno), lineno, "data")

    def text(self, text, lineno):
        match = _STRING.fullmatch(text.strip())
        if match is None:
            raise AsmError(lineno, "TEXT needs one double-quoted string")
        self.bind()
        for char in _unescape(match[1], lineno):
            if ord(char) > 2047:
                raise AsmError(lineno, f"code point U+{ord(char):04X} does not fit in a half")
            self.cells.extend(fp.pack(fp.H, float(ord(char))))  # a cell holds a half value, not a byte

    def space(self, text, lineno):
        try:
            n = _evaluate(_expression(text, lineno), self.symbols)
        except _Undefined as term:
            raise AsmError(lineno, f"'{term}' is not defined") from None
        if n < 0 or not n.is_integer():
            raise AsmError(lineno, f"SPACE needs a whole number of cells, not {n!r}")
        self.bind()
        self.cells.extend((0,) * int(n))

    def value(self, fmt, items, lineno, role, narrow=None):
        """Emit one value; if it depends on a label not seen yet, patch it later.

        `narrow` rounds the value to that format first, the way a literal is
        rounded to the register it is loaded into."""
        start = len(self.cells)
        self.cells.extend((0,) * fmt.cells)
        try:
            x = _evaluate(items, self.symbols)
        except _Undefined:
            self.fixups.append((start, fmt, items, lineno, role, narrow))
            return
        self._place(start, fmt, x, lineno, role, narrow)

    def _place(self, start, fmt, x, lineno, role, narrow=None):
        if role == "target":
            if x < 0 or not x.is_integer() or x % 4 or x != fp.round_literal(fp.F, x):
                raise AsmError(lineno, f"{x!r} is not an instruction address: a whole cell number "
                                       f"on a four-cell boundary")
        if narrow is not None:
            x = fp.round_literal(narrow, x)
        cells = fp.pack(fmt, fp.round_literal(fmt, x))
        self.cells[start:start + len(cells)] = array("H", cells)

    # -- instructions ------------------------------------------------------------

    def instruction(self, name, suffix, text, lineno):
        forms = FORMS[name]
        operands = _operands(text, lineno)
        form = next((f for f in forms if len(f) == len(operands)), None)
        if form is None:
            wanted = " or ".join(str(len(f)) for f in forms)
            raise AsmError(lineno, f"{name} takes {wanted} operands, not {len(operands)}")
        if suffix is not None and suffix not in FORMAT_BY_SUFFIX:
            raise AsmError(lineno, f"there is no .{suffix}: registers are .H, .F, .D or .Q")
        if suffix is not None and "R" not in form and "A" not in form:
            raise AsmError(lineno, f"{name} works on no register, so it takes no suffix")
        parts = [self._operand(kind, item, lineno) for kind, item in zip(form, operands)]
        registers = [part for kind, part in zip(form, parts) if kind != "E"]
        expressions = [part for kind, part in zip(form, parts) if kind == "E"]

        if len(registers) == 3 and name in STEP_FORMS:
            name = STEP_FORMS[name]
        spec = BY_NAME[name]
        files = [file_number for file_number, _ in registers]
        self._own_type(name.replace("_TO", ""), suffix, registers, files,
                       [kind for kind, part in zip(form, parts) if kind != "E"], lineno)

        self.align(4)
        self.bind()
        if self.start is None:
            self.start = float(self.here())
        fields = [code_of(*register) for register in registers] + [0] * (3 - len(registers))
        lanes = VECTOR_LANES.get(spec.name)
        if lanes is not None:
            for file_number, index in registers:
                if index + lanes > 32:
                    raise AsmError(lineno, f"{spec.name} needs {lanes} registers from "
                                           f"{_register_name(file_number, index)} onwards, and the "
                                           f"{KIND_BY_FILE[file_number]} file only reaches 31")
        self.cells.extend(fp.pack(fp.D, float(encode(spec.number, *fields))))
        for items in expressions:
            # an inline operand is a double in memory, whatever reads it: a
            # quad LDI narrows its literal to double and widens at runtime
            narrow = fp.FORMAT_BY_CODE[files[0]] if spec.name == "LDI" else None
            if narrow is fp.Q:
                narrow = fp.D
            self.value(fp.D, items, lineno, "value" if spec.name == "LDI" else "target",
                       narrow=narrow)

    def _operand(self, kind, text, lineno):
        if kind == "E":
            return _expression(text, lineno)
        if kind == "M":
            match = _MEMORY.fullmatch(text)
            if match is None:
                raise AsmError(lineno, f"'{text}' should be a memory operand such as [f3]")
            text = match[1]
        file_number, index = self._register(text, lineno)
        if kind == "A" and file_number != 2:
            # a jump target stays a float: a half link that rounds a return
            # address is a crash, not a pointer.  Data addresses take any
            # file -- a pointer's type is its reach
            raise AsmError(lineno, f"a jump target lives in a float register, and '{text}' is not one")
        return file_number, index

    @staticmethod
    def _register(text, lineno):
        match = _REGISTER.fullmatch(text.lower())
        if match and int(match[2]) < 32:
            file_number = "hfdq".index(match[1]) + 1
            index = int(match[2])
            if file_number == 4 and index > 3:
                raise AsmError(lineno, f"the machine holds four quad registers, "
                                       f"and '{text}' is past them")
            return file_number, index
        if text.lower() in ALIASES:
            return 2, ALIASES[text.lower()]
        raise AsmError(lineno, f"'{text}' is not a register")

    def _own_type(self, name, suffix, registers, files, kinds, lineno):
        def show(position):
            file_number, index = registers[position]
            return f"{_register_name(file_number, index)} holds a {KIND_BY_FILE[file_number]}"

        # A suffix names the type of the value the instruction works on: every
        # register but address registers, and for a conversion the destination.
        if suffix is not None:
            wanted = FORMAT_BY_SUFFIX[suffix].code
            checked = [0] if name == "CVT" else [p for p, kind in enumerate(kinds) if kind == "R"]
            for position in checked:
                if files[position] != wanted:
                    raise AsmError(lineno, f"{name}.{suffix} is written for {KIND_BY_FILE[wanted]}s here, "
                                           f"but {show(position)}")
        if name == "CVT":
            if files[0] == files[1]:
                raise AsmError(lineno, f"CVT changes type, but both of its registers hold "
                                       f"{KIND_BY_FILE[files[0]]}s")
        elif name in ("SPLIT", "JOIN"):
            first, second = (0, 1) if name == "SPLIT" else (1, 2)
            if files[first] != files[second]:
                raise AsmError(lineno, f"{name} needs two registers of one type, "
                                       f"but {show(first)} and {show(second)}")
        elif BY_NAME[name].same_type and len(set(files)) != 1:
            raise AsmError(lineno, f"{name} needs operands of one type, but {show(0)} and {show(1)}")

    # -- wrap-up ----------------------------------------------------------------

    def finish(self):
        self.bind()
        for start, fmt, items, lineno, role, narrow in self.fixups:
            try:
                x = _evaluate(items, self.symbols)
            except _Undefined as term:
                raise AsmError(lineno, f"'{term}' is not defined") from None
            self._place(start, fmt, x, lineno, role, narrow)
        return Image(self.org, self.cells, self.symbols, float(self.org) if self.start is None else self.start)


def assemble(source, org=0):
    """Turn assembly text into an Image.  AsmError carries the offending line."""
    state = _Assembler(org)
    for lineno, raw in enumerate(source.splitlines(), 1):
        line = _strip_comment(raw)
        match = _LABEL.match(line)
        while match:
            state.label(match[1], lineno)
            line = line[match.end():]
            match = _LABEL.match(line)
        line = line.strip()
        if not line:
            continue
        match = _STATEMENT.fullmatch(line)
        if match is None:
            raise AsmError(lineno, f"cannot read '{line}'")
        name = match[1].upper()
        suffix = match[2].upper() if match[2] else None
        if name == "DATA":
            state.data(suffix, match[3], lineno)
        elif name == "TEXT":
            state.text(match[3], lineno)
        elif name == "SPACE":
            state.space(match[3], lineno)
        elif name == "ALIGN":
            try:
                n = float(match[3])
            except ValueError:
                n = 0.0
            if n <= 0 or n != int(n) or n > 65536:
                raise AsmError(lineno, f"ALIGN needs a whole number of cells, not {match[3]!r}")
            state.align(int(n))
        elif name in FORMS:
            state.instruction(name, suffix, match[3], lineno)
        else:
            raise AsmError(lineno, f"unknown instruction or directive '{match[1]}'")
    return state.finish()
