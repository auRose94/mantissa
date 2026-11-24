"""m4c.py -- the Mantissa compiler: a small float-only language to Mantissa-1 assembly.

    python3 m4c.py program.m4            print the assembly
    python3 m4c.py program.m4 --out a.mant

The language (v1):

    var x = 0.0                 every variable is a double, forever
    array a[10]                 arrays live in memory, doubles, zero-filled
    a[0] = 1.5
    print a[0] + x              the console spells the value; the machine hands it over
    while i < 10.0 { ... }      while, if, else, with { } blocks
    fn area(r) { return 3.0 * r * r }   functions at the top level; return, or
    print area(2.0)                     fall off the end for 0.0.  Recursion works.

    expressions: + - * / < <= > >= == !=, numbers, variables, calls, and
    sqrt(x) fma(a, b, c) floor(x) ceil(x) chop(x) round(x) min(a, b) max(a, b)
    mod(a, b) rem(a, b) scalb(x, n) logb(x) abs(x) neg(x) sign(a, b) next(x, y)
    classify(x) compare(a, b) lt(a, b) le(a, b) eq(a, b)

Functions see their own variables only -- a function and the main code each
get the variable registers to themselves.  The calling convention is on the
machine's stack, sp being f31 as the runner starts it: the caller pushes its
variable registers and any expression temporaries it still holds, then the
arguments, and calls with JAL into f30; the callee pops its arguments into
d0.., computes, and leaves the answer in d0; the caller parks the answer in a
fresh temporary before restoring what it pushed.  Every frame's variables are
saved by its caller, which is what makes recursion work.

There are no integers in the language because there are none in the machine.
Counters count while their format keeps step: a double counts to 2^53.  The
compiler warns when a comparison uses a bound in the awkward band above half a
counting wall, where the programmer is probably assuming the wrong format.

Arrays are 1D blocks of doubles in memory: `array a[16]` reserves 64 cells
(zero-filled) at a 4-cell-aligned label, and `a[i]` compiles to the address
`a + 4*i` in a float register plus one LD or ST.  The index is not chopped:
a fractional index makes a fractional cell address, and the machine faults
on it -- exactly the fault a whole-number machine would give.  There are no
bounds checks: an address past the end of memory faults the same way.

How it compiles: the machine's arithmetic is three-address, so an expression
evaluates onto a stack of temporary registers and a variable's own register is
returned directly, never copied -- but every register is tracked as owned (a
temporary) or borrowed (a variable), and only owned ones go back to the free
list.  Control flow is BZ and BR over labels.  Four optimisations run before
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

TEMP_LOW, TEMP_HIGH = 8, 28   # d8..d27 are the expression stack; d0..d5 are variables
SIZE_REG = "d6"              # holds the element size 4.0, for array addressing
LINK_SHUTTLE = "d7"          # carries f30 to the stack: no scope ever holds d7
STACK_STEP = "f2"            # holds 4.0 in a float register, for sp moves
LINK_REG, SP = "f30", "f31"  # JAL leaves pc+8 in f30; the stack pointer is f31
CELLS_PER_DOUBLE = 4.0       # a stack slot is one double, four cells
MAX_VARIABLES = 6            # per scope: main's, and each function's params+locals
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
        elif c in "+-*/<>=(){},;[]":
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
        if kind == "name" and value == "fn":
            raise CompileError(line, "a fn is declared at the top level only")
        if kind == "name" and value == "return":
            self.take()
            nxt = self.peek()
            if nxt[1] == "end" or (nxt[1] == "op" and nxt[2] in (";", "}")):
                into.append(["return", line, None])
            else:
                into.append(["return", line, self.expression()])
            self.semicolon()
        elif kind == "name" and value == "var":
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
        elif kind == "name" and value == "array":
            self.take()
            line2, name = self.expect("name")
            self.expect("op", "[")
            line3, size_kind, size = self.take()
            if size_kind != "num" or size != int(size) or size <= 0:
                raise CompileError(line2, f"the array size must be a positive whole number, "
                                          f"not {size!r}")
            self.expect("op", "]")
            self.semicolon()
            into.append(["arraydecl", line2, name, int(size)])
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
            if self.peek()[1] == "op" and self.peek()[2] == "[":
                self.take()
                index = self.expression()
                self.expect("op", "]")
                self.expect("op", "=")
                into.append(["astore", line2, name, index, self.expression()])
                self.semicolon()
            else:
                self.expect("op", "=")
                into.append(["assign", line2, name, self.expression()])
                self.semicolon()
        else:
            raise CompileError(line, f"expected a statement, found '{value or kind}'")

    def parse_fn(self, into):
        """A top-level `fn name(params) { body }`.  Called by compile_source
        when the next token is `fn`; statement() refuses it anywhere deeper."""
        line, _, _ = self.take()
        line2, name = self.expect("name")
        self.expect("op", "(")
        params = []
        if not (self.peek()[1] == "op" and self.peek()[2] == ")"):
            while True:
                line3, param = self.expect("name")
                params.append(param)
                if self.peek()[1] == "op" and self.peek()[2] == ",":
                    self.take()
                    continue
                break
        self.expect("op", ")")
        body = []
        self.statements_until_close(body)
        into.append(["fn", line, name, params, body])

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
            if self.peek()[1] == "op" and self.peek()[2] == "(" and value not in BUILTINS:
                self.take()
                args = []
                if not (self.peek()[1] == "op" and self.peek()[2] == ")"):
                    args.append(self.expression())
                    while self.peek()[1] == "op" and self.peek()[2] == ",":
                        self.take()
                        args.append(self.expression())
                self.expect("op", ")")
                return ("usercall", line, value, args)
            if value in BUILTINS:
                self.expect("op", "(")
                args = [self.expression()]
                while self.peek()[1] == "op" and self.peek()[2] == ",":
                    self.take()
                    args.append(self.expression())
                self.expect("op", ")")
                return ("call", line, value, args)
            if self.peek()[1] == "op" and self.peek()[2] == "[":
                self.take()
                index = self.expression()
                self.expect("op", "]")
                return ("index", line, value, index)
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
    if kind == "usercall":
        return ("usercall", node[1], node[2], [fold(a) for a in node[3]])
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
    if kind == "usercall":
        return ("usercall", node[1], node[2], [identities(a) for a in node[3]])
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
    elif kind in ("call", "usercall"):
        for a in node[3]:
            collect_literals(a, pool)


def simplify_all(nodes):
    for node in nodes:
        kind = node[0]
        if kind == "assign":
            node[3] = identities(fold(node[3]))
        elif kind == "print":
            node[2] = identities(fold(node[2]))
        elif kind == "return":
            if node[2] is not None:
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
        elif kind == "return":
            if node[2] is not None:
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
            check_walls_expr(node[3], warnings)
        elif kind == "astore":
            check_walls_expr(node[3], warnings)
            check_walls_expr(node[4], warnings)
        elif kind == "print":
            check_walls_expr(node[2], warnings)
        elif kind == "return":
            if node[2] is not None:
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
    elif kind in ("call", "usercall"):
        for a in node[3]:
            check_walls_expr(a, warnings)


# --- codegen ---------------------------------------------------------------------------

BINARY_OPS = {"add": "ADD", "sub": "SUB", "mul": "MUL", "div": "DIV",
              "lt2": "LT", "le2": "LE", "eq2": "EQ"}


class Compiler:
    def __init__(self):
        self.variables = {}
        self.variable_count = 0
        self.arrays = {}          # name -> elements
        self.functions = {}       # name -> arity
        self.in_function = False
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

    def live_temps(self):
        """The temporaries currently checked out: exactly what a call has to
        save, because the callee spends them freely.  The pool's registers are
        read-only and every scope shares them, so they stay put."""
        managed = range(TEMP_LOW, TEMP_HIGH - len(self.pool))
        return sorted(set(managed) - set(self.free_temps))

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
        if kind == "index":
            _, line, name, index = node
            if name not in self.arrays:
                what = "is declared as a scalar" if name in self.variables else "was never declared"
                raise CompileError(line, f"'{name}' {what}; it cannot be indexed")
            r_i, owned_i = self.expr(index)
            scratch = self.take_temp()
            self.emit(f"MUL.D  {scratch}, {r_i}, {SIZE_REG}")
            self.emit(f"CVT.F  f1, {scratch}")
            self.emit(f"LDI.F  f0, {name}")
            self.emit(f"ADD.F  f0, f0, f1")
            out = self.take_temp()
            self.emit(f"LD.D   {out}, [f0]")
            if owned_i:
                self.give_temp(r_i)
            self.give_temp(scratch)
            return out, True
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
        if kind == "usercall":
            return self.usercall(node)
        if kind == "bin":
            return self.bin(node)
        raise CompileError(0, f"cannot compile {kind}")

    def usercall(self, node):
        """A call to a fn the program declared.  The caller pushes its variable
        registers and every temporary it still holds, then the arguments, and
        calls; the callee pops its arguments and leaves the answer in d0; the
        caller parks the answer in a fresh temporary before restoring what it
        pushed.  Every frame is saved by its caller, which is what makes
        recursion work."""
        _, line, name, args = node
        saved = self.live_temps()
        # the link goes in deepest: the callee pops its arguments off the top,
        # and an inner call would otherwise overwrite f30, making every frame
        # below the innermost return twice, popping slots that are not its own.
        # f30 cannot itself be stored as a double, so the shuttle carries it --
        # and every slot on the stack stays one double
        self.emit(f"CVT.D  {LINK_SHUTTLE}, {LINK_REG}")
        self.emit(f"SUB.F  {SP}, {SP}, {STACK_STEP}")
        self.emit(f"ST.D   [{SP}], {LINK_SHUTTLE}")
        for register in range(self.variable_count):
            self.emit(f"SUB.F  {SP}, {SP}, {STACK_STEP}")
            self.emit(f"ST.D   [{SP}], d{register}")
        for register in saved:
            self.emit(f"SUB.F  {SP}, {SP}, {STACK_STEP}")
            self.emit(f"ST.D   [{SP}], d{register}")
        for a in args:
            register, owned = self.expr(a)
            self.emit(f"SUB.F  {SP}, {SP}, {STACK_STEP}")
            self.emit(f"ST.D   [{SP}], {register}")
            if owned:
                self.give_temp(register)
        self.emit(f"JAL    {LINK_REG}, {name}")
        out = self.take_temp()
        self.emit(f"MOV.D  {out}, d0")
        for register in reversed(saved):
            self.emit(f"LD.D   d{register}, [{SP}]")
            self.emit(f"ADD.F  {SP}, {SP}, {STACK_STEP}")
        for register in reversed(range(self.variable_count)):
            self.emit(f"LD.D   d{register}, [{SP}]")
            self.emit(f"ADD.F  {SP}, {SP}, {STACK_STEP}")
        self.emit(f"LD.D   {LINK_SHUTTLE}, [{SP}]")
        self.emit(f"ADD.F  {SP}, {SP}, {STACK_STEP}")
        self.emit(f"CVT.F  {LINK_REG}, {LINK_SHUTTLE}")
        return out, True

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
        if kind == "arraydecl":
            _, line, name, elements = node
            if self.in_function:
                raise CompileError(line, "arrays are declared at the top level only")
            self.arrays[name] = elements   # registered in the pre-pass; SPACE emitted first
        elif kind == "astore":
            _, line, name, index, value = node
            if name not in self.arrays:
                what = "is declared as a scalar" if name in self.variables else "was never declared"
                raise CompileError(line, f"'{name}' {what}; it cannot be indexed")
            r_v, owned_v = self.expr(value)
            r_i, owned_i = self.expr(index)
            scratch = self.take_temp()
            self.emit(f"MUL.D  {scratch}, {r_i}, {SIZE_REG}")
            self.emit(f"CVT.F  f1, {scratch}")
            self.emit(f"LDI.F  f0, {name}")
            self.emit(f"ADD.F  f0, f0, f1")
            self.emit(f"ST.D   [f0], {r_v}")
            if owned_v:
                self.give_temp(r_v)
            if owned_i:
                self.give_temp(r_i)
            self.give_temp(scratch)
        elif kind == "var":
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
        elif kind == "return":
            _, line, value = node
            if not self.in_function:
                raise CompileError(line, "return outside a function")
            if value is None:
                self.emit("LDI.D  d0, 0.0")
            else:
                register, owned = self.expr(value)
                if register != "d0":
                    self.emit(f"MOV.D  d0, {register}")
                if owned:
                    self.give_temp(register)
            self.emit(f"JMP    {LINK_REG}")
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

    def function(self, node):
        """A fn's scope: the parameters arrive in d0.., its own variables take
        the registers from there, and falling off the end returns 0.0."""
        _, line, name, params, body = node
        if len(params) > MAX_VARIABLES:
            raise CompileError(line, f"{name} takes too many arguments: a frame is "
                                     f"{MAX_VARIABLES} registers wide")
        self.in_function = True
        outer_variables, outer_count = self.variables, self.variable_count
        self.variables = {}
        self.variable_count = len(params)
        self.pending_zero = None
        for k, param in enumerate(params):
            if param in self.variables:
                raise CompileError(line, f"'{param}' is declared twice")
            self.variables[param] = f"d{k}"
        self.emit(f"{name}:")
        for register in reversed(range(len(params))):
            self.emit(f"LD.D   d{register}, [{SP}]")
            self.emit(f"ADD.F  {SP}, {SP}, {STACK_STEP}")
        for statement2 in body:
            self.statement(statement2)
        if self.pending_zero is not None:
            self.emit(f"LDI.D  {self.pending_zero}, 0.0")
            self.pending_zero = None
        self.emit("LDI.D  d0, 0.0")
        self.emit(f"JMP    {LINK_REG}")
        self.in_function = False
        self.variables, self.variable_count = outer_variables, outer_count


def check_calls(nodes, signatures, inside):
    """Every call names a declared fn with the right arity, return appears
    only inside a fn, and arrays are declared only at the top level.  Running
    this over the AST is what lets a fn be called before its definition."""
    for node in nodes:
        kind = node[0]
        if kind == "arraydecl":
            if inside:
                raise CompileError(node[1], "arrays are declared at the top level only")
        elif kind == "return":
            if not inside:
                raise CompileError(node[1], "return outside a function")
            if node[2] is not None:
                check_calls_expr(node[2], signatures)
        elif kind == "assign":
            check_calls_expr(node[3], signatures)
        elif kind == "print":
            check_calls_expr(node[2], signatures)
        elif kind == "astore":
            check_calls_expr(node[3], signatures)
            check_calls_expr(node[4], signatures)
        elif kind == "while":
            check_calls_expr(node[2], signatures)
            check_calls(node[3], signatures, inside)
        elif kind == "if":
            check_calls_expr(node[2], signatures)
            check_calls(node[3], signatures, inside)
            check_calls(node[4], signatures, inside)


def check_calls_expr(node, signatures):
    kind = node[0]
    if kind == "usercall":
        _, line, name, args = node
        if name not in signatures:
            raise CompileError(line, f"'{name}' is not a fn this program declares")
        if len(args) != signatures[name]:
            raise CompileError(line, f"{name} takes {signatures[name]} argument(s), "
                                     f"not {len(args)}")
        for a in args:
            check_calls_expr(a, signatures)
    elif kind == "bin":
        check_calls_expr(node[3], signatures)
        check_calls_expr(node[4], signatures)
    elif kind == "neg":
        check_calls_expr(node[2], signatures)
    elif kind == "index":
        check_calls_expr(node[3], signatures)
    elif kind == "call":
        for a in node[3]:
            check_calls_expr(a, signatures)


def compile_source(source):
    """(assembly text, image, warnings).  CompileError carries the line."""
    items = []
    parser = Parser(lex(source))
    while parser.peek()[1] != "end":
        if parser.peek()[1] == "name" and parser.peek()[2] == "fn":
            parser.parse_fn(items)
        else:
            parser.statement(items)
    functions = [node for node in items if node[0] == "fn"]
    main = [node for node in items if node[0] != "fn"]
    signatures = {}
    for node in functions:
        _, line, name, params, _ = node
        if name in signatures:
            raise CompileError(line, f"'{name}' is declared twice")
        if name in BUILTINS:
            raise CompileError(line, f"'{name}' is a builtin; a fn cannot take its name")
        if name == "main":
            raise CompileError(line, "a fn cannot be named 'main': the machine starts there")
        signatures[name] = len(params)
    for node in functions:
        check_calls(node[4], signatures, True)
    check_calls(main, signatures, False)
    simplify_all(main)
    for node in functions:
        simplify_all(node[4])
    pool = {}
    collect_all(main, pool)
    for node in functions:
        collect_all(node[4], pool)
    warnings = []
    check_walls(main, warnings)
    for node in functions:
        check_walls(node[4], warnings)
    compiler = Compiler()
    compiler.functions = signatures
    compiler.pool = {fp.to_bits(fp.D, value): f"d{TEMP_HIGH - 1 - k}"
                     for k, (value, uses) in enumerate(
                         sorted(pool.items(), key=lambda kv: (-kv[1], kv[0])))
                     if uses > 1}
    compiler.pool_values = {fp.to_bits(fp.D, value): value
                            for value, uses in pool.items() if fp.to_bits(fp.D, value) in compiler.pool}
    compiler.free_temps = list(range(TEMP_HIGH - 1 - len(compiler.pool), TEMP_LOW - 1, -1))
    # the arrays' space comes first in the image: position 0 is 4-cell aligned,
    # and every array's size is a multiple of 4 cells, so all bases stay aligned
    array_lines = []
    for node in main:
        if node[0] == "arraydecl":
            if node[2] in compiler.arrays or node[2] in compiler.variables or node[2] in signatures:
                raise CompileError(node[1], f"'{node[2]}' is declared twice")
            compiler.arrays[node[2]] = node[3]
            array_lines.append(f"{node[2]}:     SPACE {4 * node[3]}")
    for line in array_lines:
        compiler.emit(line)
    if compiler.arrays:
        compiler.emit(f"LDI.D  {SIZE_REG}, 4.0      ; the element size, for array addressing")
    for key, register in sorted(compiler.pool.items(), key=lambda kv: kv[1]):
        compiler.emit(f"LDI.D  {register}, {compiler.pool_values[key]!r}      ; the literal pool")
    if signatures:
        compiler.emit(f"LDI.F  {STACK_STEP}, {CELLS_PER_DOUBLE!r}   ; the stack step, one double")
        over = compiler.label()
        compiler.emit(f"BR     {over}")
        for node in functions:
            compiler.function(node)
        compiler.emit(f"{over}:")
    for node in main:
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
