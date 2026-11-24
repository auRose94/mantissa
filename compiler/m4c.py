"""m4c.py -- the Mantissa compiler: a small float-only language to Mantissa-1 assembly.

    python3 m4c.py program.m4            print the assembly
    python3 m4c.py program.m4 --out a.mant

The language (v0):

    var x = 0.0                 every variable is a double, forever
    var i = 0.0
    while i < 10.0 { ... }      while, if, else, with { } blocks
    x = x + 1.0
    print x                     the console spells the value; the machine hands it over

    expressions: + - * / < <= > >= == !=, numbers, variables, and
    sqrt(x) fma(a, b, c) floor(x) ceil(x) chop(x) round(x) min(a, b) max(a, b)
    mod(a, b) rem(a, b) scalb(x, n) logb(x) abs(x) neg(x) sign(a, b) next(x, y)
    classify(x) compare(a, b) lt(a, b) le(a, b) eq(a, b)

There are no integers in the language because there are none in the machine.
Counters count while their format keeps step: a double counts to 2^53.  The
compiler warns when a comparison uses a bound in the awkward band above half a
counting wall, where the programmer is probably assuming the wrong format.

How it compiles: the machine's arithmetic is three-address, so an expression
evaluates onto a stack of temporary registers and a variable's own register is
returned directly, never copied -- but every register is tracked as owned (a
temporary) or borrowed (a variable), and only owned ones go back to the free
list.  Control flow is BZ and BR over labels.  Three optimisations run before
codegen, each one tested in tests/test_compiler.py:

    folding      an expression of only literals becomes the literal the
                 machine would round them to -- folded once, not per run
    identities   x * 1 -> x and x - 0 -> x (both exact for every IEEE value,
                 NaN payloads included; x + 0 is NOT, it turns -0 into +0)
    literal pool a constant used more than once is loaded into one register,
                 before the code, instead of riding along as an inline operand
                 at every use -- inside a loop that is one instruction saved
                 per iteration
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mantissa import asm, fp  # noqa: E402

TEMP_LOW, TEMP_HIGH = 8, 28   # d8..d27 are the expression stack; d0..d7 are variables
MAX_VARIABLES = TEMP_LOW
WALLS = {2048.0: "half", 16777216.0: "float", 9007199254740992.0: "double"}

BINARY_OPS = {"add": "ADD", "sub": "SUB", "mul": "MUL", "div": "DIV",
              "lt": "LT", "le": "LE", "eq2": "EQ"}
ONE_ARG = {"sqrt": "SQRT", "abs": "ABS", "neg": "NEG", "floor": "FLOOR",
           "ceil": "CEIL", "chop": "CHOP", "round": "ROUND", "logb": "LOGB",
           "classify": "CLASS"}
TWO_ARG = {"min": "MIN", "max": "MAX", "mod": "MOD", "rem": "REM", "scalb": "SCALB",
           "sign": "CPSGN", "next": "NEXT", "compare": "CMP"}
BUILTINS = set(ONE_ARG) | set(TWO_ARG) | {"fma", "lt", "le", "eq"}


class CompileError(Exception):
    def __init__(self, line, message):
        super().__init__(f"line {line}: {message}")
        self.line = line
        self.message = message


# --- lexer ---------------------------------------------------------------------------


def lex(source):
    tokens = []
    line = 1
    i = 0
    while i < len(source):
        c = source[i]
        if c == "\n":
            line += 1
            i += 1
        elif c in " \t":
            i += 1
        elif c == "#":
            while i < len(source) and source[i] != "\n":
                i += 1
        elif c.isalpha() or c == "_":
            j = i
            while j < len(source) and (source[j].isalnum() or source[j] == "_"):
                j += 1
            tokens.append((line, "name", source[i:j]))
            i = j
        elif c.isdigit() or (c == "." and i + 1 < len(source) and source[i + 1].isdigit()):
            j = i
            while j < len(source) and (source[j].isdigit() or source[j] in ".eE+-"):
                if source[j] in "+-" and source[j - 1] not in "eE":
                    break
                j += 1
            text = source[i:j]
            try:
                tokens.append((line, "num", float(text)))
            except ValueError:
                raise CompileError(line, f"cannot read the number '{text}'")
            i = j
        elif source[i:i + 2] in ("<=", ">=", "==", "!="):
            tokens.append((line, "op", source[i:i + 2]))
            i += 2
        elif c in "+-*/<>=(){},;":
            tokens.append((line, "op", c))
            i += 1
        else:
            raise CompileError(line, f"unexpected character '{c}'")
    tokens.append((line, "end", ""))
    return tokens


# --- parser: statements to a list, expressions to tuples -------------------------------


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.at = 0

    def peek(self):
        return self.tokens[self.at]

    def take(self):
        token = self.tokens[self.at]
        self.at += 1
        return token

    def expect(self, kind, text=None):
        line, kind2, value = self.take()
        if kind2 != kind or (text is not None and value != text):
            want = text if text is not None else kind
            raise CompileError(line, f"expected '{want}', found '{value or kind2}'")
        return line, value

    def semicolon(self):
        if self.peek()[1] == "op" and self.peek()[2] == ";":
            self.take()

    def statements_until_close(self, into):
        self.expect("op", "{")
        while not (self.peek()[1] == "op" and self.peek()[2] == "}"):
            self.statement(into)
        self.expect("op", "}")

    def statement(self, into):
        line, kind, value = self.peek()
        if kind == "name" and value == "var":
            self.take()
            while True:
                line2, name = self.expect("name")
                into.append(["var", line2, name])
                if self.peek()[1] == "op" and self.peek()[2] == "=":
                    self.take()
                    into.append(["assign", line2, name, self.expression()])
                if self.peek()[1] == "op" and self.peek()[2] == ",":
                    self.take()
                    continue
                break
            self.semicolon()
        elif kind == "name" and value == "while":
            self.take()
            condition = self.expression()
            body = []
            self.statements_until_close(body)
            into.append(["while", line, condition, body])
        elif kind == "name" and value == "if":
            self.take()
            condition = self.expression()
            body = []
            self.statements_until_close(body)
            otherwise = []
            if self.peek()[1] == "name" and self.peek()[2] == "else":
                self.take()
                if self.peek()[1] == "name" and self.peek()[2] == "if":
                    nested = []
                    self.statement(nested)
                    otherwise = nested
                else:
                    self.statements_until_close(otherwise)
            into.append(["if", line, condition, body, otherwise])
        elif kind == "name" and value == "print":
            self.take()
            into.append(["print", line, self.expression()])
        elif kind == "name":
            line2, _, name = self.take()
            self.expect("op", "=")
            into.append(["assign", line2, name, self.expression()])
            self.semicolon()
        else:
            raise CompileError(line, f"expected a statement, found '{value or kind}'")

    PRECEDENCE = (("lt2", "le2", "gt2", "ge2", "eq2", "ne2"),
                  ("add", "sub"), ("mul", "div"))

    def expression(self):
        return self.binary(0)

    def binary(self, level):
        if level == len(self.PRECEDENCE):
            return self.unary()
        node = self.binary(level + 1)
        while True:
            line, kind, value = self.peek()
            if kind != "op":
                break
            op = {"<": "lt2", "<=": "le2", ">": "gt2", ">=": "ge2",
                  "==": "eq2", "!=": "ne2", "+": "add", "-": "sub",
                  "*": "mul", "/": "div"}.get(value, value)
            if op not in self.PRECEDENCE[level]:
                break
            self.take()
            node = ("bin", line, op, node, self.binary(level + 1))
        return node

    def unary(self):
        line, kind, value = self.peek()
        if kind == "op" and value == "-":
            self.take()
            return ("neg", line, self.unary())
        if kind == "op" and value == "+":
            self.take()
            return self.unary()
        return self.primary()

    def primary(self):
        line, kind, value = self.take()
        if kind == "num":
            return ("num", line, value)
        if kind == "name":
            if value in BUILTINS:
                self.expect("op", "(")
                args = [self.expression()]
                while self.peek()[1] == "op" and self.peek()[2] == ",":
                    self.take()
                    args.append(self.expression())
                self.expect("op", ")")
                return ("call", line, value, args)
            if value in ("inf", "nan"):
                return ("num", line, float(value))
            return ("var", line, value)
        if kind == "op" and value == "(":
            inner = self.expression()
            self.expect("op", ")")
            return inner
        raise CompileError(line, f"expected an expression, found '{value or kind}'")


# --- the optimisation passes over the AST -----------------------------------------------


def fold(node):
    """An expression of only literals becomes one literal, rounded once."""
    kind = node[0]
    if kind == "bin":
        _, line, op, left, right = node
        left, right = fold(left), fold(right)
        if left[0] == "num" and right[0] == "num":
            a, b = left[2], right[2]
            if op == "add":
                return ("num", line, fp.round_literal(fp.D, a + b))
            if op == "sub":
                return ("num", line, fp.round_literal(fp.D, a - b))
            if op == "mul":
                return ("num", line, fp.round_literal(fp.D, a * b))
            if op == "div" and b != 0.0:
                return ("num", line, fp.round_literal(fp.D, a / b))
            if op == "lt2":
                return ("num", line, float(a < b))
            if op == "le2":
                return ("num", line, float(a <= b))
            if op == "gt2":
                return ("num", line, float(a > b))
            if op == "ge2":
                return ("num", line, float(a >= b))
            if op == "eq2":
                return ("num", line, float(a == b))
            if op == "ne2":
                return ("num", line, float(a != b))
        return ("bin", line, op, left, right)
    if kind == "neg":
        _, line, inner = node
        inner = fold(inner)
        if inner[0] == "num":
            return ("num", line, -inner[2])
        return ("neg", line, inner)
    if kind == "call":
        _, line, name, args = node
        args = [fold(a) for a in args]
        if name == "sqrt" and len(args) == 1 and args[0][0] == "num" and args[0][2] >= 0:
            root = args[0][2] ** 0.5
            if root.is_integer():
                return ("num", line, float(root))  # only exact roots: 4 -> 2, not 2 -> 1.41...
        return ("call", line, name, args)
    return node


def identities(node):
    """x * 1 -> x and x - 0 -> x.  Both exact for every IEEE value, signed zeros
    and NaN payloads included.  x + 0 is not (it turns -0 into +0), so it stays."""
    kind = node[0]
    if kind == "bin":
        _, line, op, left, right = node
        left, right = identities(left), identities(right)
        if op == "mul" and right[0] == "num" and right[2] == 1.0:
            return left
        if op == "mul" and left[0] == "num" and left[2] == 1.0:
            return right
        if op == "sub" and right[0] == "num" and right[2] == 0.0:
            return left
        return ("bin", line, op, left, right)
    if kind == "neg":
        return ("neg", node[1], identities(node[2]))
    if kind == "call":
        return ("call", node[1], node[2], [identities(a) for a in node[3]])
    return node


def collect_literals(node, pool):
    kind = node[0]
    if kind == "num":
        pool[node[2]] = pool.get(node[2], 0) + 1
    elif kind == "bin":
        collect_literals(node[3], pool)
        collect_literals(node[4], pool)
    elif kind == "neg":
        collect_literals(node[2], pool)
    elif kind == "call":
        for a in node[3]:
            collect_literals(a, pool)


def simplify_all(nodes):
    for node in nodes:
        kind = node[0]
        if kind == "assign":
            node[3] = identities(fold(node[3]))
        elif kind == "print":
            node[2] = identities(fold(node[2]))
        elif kind == "while":
            node[2] = identities(fold(node[2]))
            simplify_all(node[3])
        elif kind == "if":
            node[2] = identities(fold(node[2]))
            simplify_all(node[3])
            simplify_all(node[4])


def collect_all(nodes, pool):
    for node in nodes:
        kind = node[0]
        if kind == "assign":
            collect_literals(node[3], pool)
        elif kind == "print":
            collect_literals(node[2], pool)
        elif kind == "while":
            collect_literals(node[2], pool)
            collect_all(node[3], pool)
        elif kind == "if":
            collect_literals(node[2], pool)
            collect_all(node[3], pool)
            collect_all(node[4], pool)


def check_walls(nodes, warnings):
    """A comparison against a bound in the awkward band above half a counting
    wall: the counter probably cannot get there in the format the programmer
    is imagining."""
    for node in nodes:
        kind = node[0]
        if kind == "assign":
            check_walls([node], warnings) if False else check_walls_expr(node[3], warnings)
        elif kind == "print":
            check_walls_expr(node[2], warnings)
        elif kind == "while":
            check_walls_expr(node[2], warnings)
            check_walls(node[3], warnings)
        elif kind == "if":
            check_walls_expr(node[2], warnings)
            check_walls(node[3], warnings)
            check_walls(node[4], warnings)


def check_walls_expr(node, warnings):
    kind = node[0]
    if kind == "bin" and node[2] in ("lt2", "le2") and node[4][0] == "num":
        bound = abs(node[4][2])
        for wall, fmt in sorted(WALLS.items()):
            if bound > wall / 2:
                warnings.append(f"a bound of {bound:.0f} is at or above half of the {fmt} counting "
                                f"wall ({wall:.0f}): a {fmt} counter freezes before it gets there")
    elif kind == "bin":
        check_walls_expr(node[3], warnings)
        check_walls_expr(node[4], warnings)
    elif kind == "neg":
        check_walls_expr(node[2], warnings)
    elif kind == "call":
        for a in node[3]:
            check_walls_expr(a, warnings)


# --- codegen ---------------------------------------------------------------------------

BINARY_OPS = {"add": "ADD", "sub": "SUB", "mul": "MUL", "div": "DIV",
              "lt2": "LT", "le2": "LE", "eq2": "EQ"}


class Compiler:
    def __init__(self):
        self.variables = {}
        self.variable_count = 0
        self.lines = []
        self.labels = 0
        self.pool = {}
        self.free_temps = []
        self.pending_zero = None

    def label(self):
        self.labels += 1
        return f"L{self.labels}"

    def emit(self, text):
        self.lines.append("        " + text)

    # the temporary register stack --------------------------------------------------
    # expr() returns (register, owned): owned registers are temporaries and go
    # back to the free list after use; a variable's register is borrowed and
    # never freed.  Getting this wrong silently corrupts variables.

    def take_temp(self):
        if not self.free_temps:
            raise CompileError(0, "the expression is too deep for the temporary registers")
        return f"d{self.free_temps.pop()}"

    def give_temp(self, register):
        self.free_temps.append(int(register[1:]))

    def expr(self, node):
        kind = node[0]
        if kind == "num":
            value = node[2]
            key = fp.to_bits(fp.D, value)
            if key in self.pool:
                return self.pool[key], False
            register = self.take_temp()
            self.emit(f"LDI.D  {register}, {self.pool_values.get(key, value)!r}")
            return register, True
        if kind == "var":
            _, line, name = node
            if name not in self.variables:
                raise CompileError(line, f"'{name}' was never declared")
            return self.variables[name], False
        if kind == "neg":
            register, owned = self.expr(node[2])
            if owned:
                self.emit(f"NEG.D  {register}, {register}")
                return register, True
            out = self.take_temp()
            self.emit(f"NEG.D  {out}, {register}")
            return out, True
        if kind == "call":
            return self.call(node)
        if kind == "bin":
            return self.bin(node)
        raise CompileError(0, f"cannot compile {kind}")

    def bin(self, node):
        _, line, op, left, right = node
        if op in ("gt2", "ge2"):  # no GT or GE in the ISA: swap the operands
            op = {"gt2": "lt2", "ge2": "le2"}[op]
            left, right = right, left
        left_reg, left_owned = self.expr(left)
        right_reg, right_owned = self.expr(right)
        out = self.take_temp()
        self.emit(f"{BINARY_OPS[op]}.D {out}, {left_reg}, {right_reg}")
        if left_owned:
            self.give_temp(left_reg)
        if right_owned:
            self.give_temp(right_reg)
        return out, True

    def call(self, node):
        _, line, name, args = node
        registers = []
        for a in args:
            register, owned = self.expr(a)
            registers.append((register, owned))
        out = self.take_temp()
        if name == "fma":
            self.emit(f"MOV.D  {out}, {registers[2][0]}")
            self.emit(f"FMA.D  {out}, {registers[0][0]}, {registers[1][0]}")
        elif name in ("lt", "le", "eq"):
            self.emit(f"{name.upper()}.D {out}, {registers[0][0]}, {registers[1][0]}")
        elif name in ONE_ARG:
            self.emit(f"{ONE_ARG[name]}.D {out}, {registers[0][0]}")
        else:
            self.emit(f"{TWO_ARG[name]}.D {out}, {registers[0][0]}, {registers[1][0]}")
        for register, owned in registers:
            if owned:
                self.give_temp(register)
        return out, True

    # statements ------------------------------------------------------------------

    def statement(self, node):
        kind = node[0]
        if kind == "assign" and self.pending_zero == self.variables.get(node[2]):
            self.pending_zero = None      # the assign makes the zero-init dead
        elif self.pending_zero is not None:
            self.emit(f"LDI.D  {self.pending_zero}, 0.0")
            self.pending_zero = None
        if kind == "var":
            _, line, name = node
            if name in self.variables:
                raise CompileError(line, f"'{name}' is declared twice")
            if self.variable_count >= MAX_VARIABLES:
                raise CompileError(line, f"no register left for '{name}': {MAX_VARIABLES} variables maximum")
            self.variables[name] = f"d{self.variable_count}"
            self.variable_count += 1
            if self.pending_zero is not None:
                self.emit(f"LDI.D  {self.pending_zero}, 0.0")
                self.pending_zero = None
            # the zero-initialisation is a dead store when the very next
            # statement assigns the variable: skip it
            self.pending_zero = self.variables[name]
        elif kind == "assign":
            _, line, name, value = node
            if name not in self.variables:
                raise CompileError(line, f"'{name}' was never declared")
            register, owned = self.expr(value)
            target = self.variables[name]
            if register != target:
                self.emit(f"MOV.D  {target}, {register}")
            if owned:
                self.give_temp(register)
            self.pending_zero = None
        elif kind == "print":
            register, owned = self.expr(node[2])
            self.emit(f"PUT.D  {register}")
            if owned:
                self.give_temp(register)
        elif kind == "while":
            _, line, condition, body = node
            top, end = self.label(), self.label()
            self.emit(f"{top}:")
            register, owned = self.expr(condition)
            self.emit(f"BZ.D   {register}, {end}")
            if owned:
                self.give_temp(register)
            for statement2 in body:
                self.statement(statement2)
            self.emit(f"BR     {top}")
            self.emit(f"{end}:")
        elif kind == "if":
            _, line, condition, body, otherwise = node
            end = self.label()
            register, owned = self.expr(condition)
            if otherwise:
                else_label = self.label()
                self.emit(f"BZ.D   {register}, {else_label}")
                if owned:
                    self.give_temp(register)
                for statement2 in body:
                    self.statement(statement2)
                self.emit(f"BR     {end}")
                self.emit(f"{else_label}:")
                for statement2 in otherwise:
                    self.statement(statement2)
            else:
                self.emit(f"BZ.D   {register}, {end}")
                if owned:
                    self.give_temp(register)
                for statement2 in body:
                    self.statement(statement2)
            self.emit(f"{end}:")
        else:
            raise CompileError(0, f"cannot compile {kind}")


def compile_source(source):
    """(assembly text, image, warnings).  CompileError carries the line."""
    statements = []
    parser = Parser(lex(source))
    while parser.peek()[1] != "end":
        parser.statement(statements)
    simplify_all(statements)
    pool = {}
    collect_all(statements, pool)
    warnings = []
    check_walls(statements, warnings)
    compiler = Compiler()
    compiler.pool = {fp.to_bits(fp.D, value): f"d{TEMP_HIGH - 1 - k}"
                     for k, (value, uses) in enumerate(
                         sorted(pool.items(), key=lambda kv: (-kv[1], kv[0])))
                     if uses > 1}
    compiler.pool_values = {fp.to_bits(fp.D, value): value
                            for value, uses in pool.items() if fp.to_bits(fp.D, value) in compiler.pool}
    compiler.free_temps = list(range(TEMP_HIGH - 1 - len(compiler.pool), TEMP_LOW - 1, -1))
    for key, register in sorted(compiler.pool.items(), key=lambda kv: kv[1]):
        compiler.emit(f"LDI.D  {register}, {compiler.pool_values[key]!r}      ; the literal pool")
    for node in statements:
        compiler.statement(node)
    if compiler.pending_zero is not None:
        compiler.emit(f"LDI.D  {compiler.pending_zero}, 0.0")
    compiler.emit("HALT")
    text = "\n".join(compiler.lines)
    return text, asm.assemble(text), warnings


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 0
    source = Path(sys.argv[1]).read_text(encoding="utf-8")
    text, image, warnings = compile_source(source)
    for warning in warnings:
        print(f"note: {warning}", file=sys.stderr)
    if len(sys.argv) > 3 and sys.argv[2] == "--out":
        Path(sys.argv[3]).write_bytes(bytes(struct_bundle(image)))
        return 0
    print(text)
    return 0


def struct_bundle(image):
    return image.cells


if __name__ == "__main__":
    sys.exit(main())
