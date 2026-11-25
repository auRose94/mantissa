"""evolve.py -- a generational search over Mantissa-1 programs, on the fleet.

    python3 evolve.py --demo                    # the fleet, if gpu/build/mantissa-gpu exists
    python3 evolve.py --demo --cpu              # the same loop on the reference simulator
    python3 evolve.py --demo --population 256 --generations 900 --seed 6

The task, stated precisely: the skeleton below is a case loop.  Per machine it
reads `--cases` inputs x (patched doubles at `cases`), runs each through the
evolved body, and stores the answer from d4 to `outs`, one pick per case.  The
body is `--slots` instruction slots between the load of x and the store of the
answer, each slot exactly one instruction: four cells, no inline operands, so
the layout never moves.  A candidate is `--slots` whole instruction numbers,
one d0 -> d4 function.  Fitness is the sum of squared error of d4 against the
reference's exact answer for f(x) = x*x + x (the quadratic x^2 + x; partial
fits exist, so the landscape has a gradient).  0.0 is a perfect program and
the search stops at `--threshold`.

The population rides to the fleet as the patch block of one array job: one
candidate is one machine, its genome IS its input patches, and a generation
is one `mantissa-gpu array` run over the whole population.  The host scores,
keeps the elites, mutates children of tournament winners, adds a few fresh
random bodies, and sends the next generation as a new job.  If nothing
improves for `--restart` generations, the bottom of the population is
redrawn -- selection has collapsed onto one stuck elite by then, and the
fleet makes fresh bodies cheaper than the patience to wait for them.

Mutation is the SMC arithmetic of programs/smc_edit.mant, done on the host:

    +1000000*k  retargets the opcode      +10000*k  the first operand field
    +100*k      the second operand field  +k        the third

and the ISA's decimal layout makes small steps semantic ones: the arithmetic
opcodes are consecutive (20 ADD, 21 SUB, 22 MUL), and registers sit beside
their neighbours (d0 is 64, d1 is 65).

The float wall: an instruction number such as 20010203 is past 16,777,216,
where a float stops counting by ones -- in a float the number is not even
representable (it rounds to 20010204, and fp.convert flags it inexact), so
the mutation arithmetic would stick and slide.  The host therefore mutates
on the integer the whole double names and stores back as a double, exact for
any eight-digit number (doubles count by ones to 2^53).  edit() below is the
whole story; test_evolve.py asserts it against the naive float path.  A
program doing this to itself needs .D registers, as smc_edit.mant does.
"""
from __future__ import annotations

import argparse
import array
import math
import os
import random
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo copy's root: mantissa/

from mantissa import asm, fp, isa, machine  # noqa: E402
from mantissa.disasm import disassemble  # noqa: E402
from mantissa.machine import Cpu  # noqa: E402
from mantissa.memory import Memory  # noqa: E402

FMT_DOUBLE = 2  # the port's format index, as in make_job.py

TARGET_TEXT = "f(x) = x*x + x"


def target(x):
    """What a perfect body computes: the quadratic x^2 + x, exact on the cases."""
    return x * x + x


ERROR_CAP = 1e100  # clamped errors square to 1e200: finite, and ranking enough


def sse(outputs, cases):
    """Sum of squared error of the machine's outputs against target() over the
    cases.  A body can output anything -- a garbage double, an infinity, a
    NaN -- and the host still owes the fleet a ranking, so each error is
    clamped into [-ERROR_CAP, ERROR_CAP] first; without that, squaring an
    overflowing double raises on the host."""
    total = 0.0
    for out, x in zip(outputs, cases):
        e = out - target(x)
        if not -ERROR_CAP <= e <= ERROR_CAP:  # also false for NaN
            e = math.copysign(ERROR_CAP, e) if e == e else ERROR_CAP
        total += e * e
    return total


# --- mutation as arithmetic -------------------------------------------------------------

POOL = {
    "ADD": 10, "SUB": 8, "MUL": 12, "DIV": 4, "FMA": 6, "MOV": 10, "NOP": 6,
    "MIN": 3, "MAX": 3, "NEG": 3, "ABS": 3, "REM": 2, "MOD": 2, "CPSGN": 2,
    "SQRT": 2, "FLOOR": 2, "CEIL": 2, "CHOP": 2, "ROUND": 2, "FLOOR_TO": 2,
    "CEIL_TO": 2, "CHOP_TO": 2, "ROUND_TO": 2, "SCALB": 2, "LOGB": 1, "NEXT": 1,
    "CLASS": 1, "LT": 3, "LE": 2, "EQ": 2, "CMP": 2, "CVT": 4, "SPLIT": 1,
    "JOIN": 1, "MANT": 1,
}  # every one of these is four cells: no inline operands, so a slot stays a slot
assert all(isa.BY_NAME[name].cells == 4 for name in POOL)

SPLICE, MACRO, RATE = 0.15, 0.05, 0.5  # a slot: take the donor's, redraw, or one SMC edit


def edit(word, delta):
    """word + delta, exactly, or None if the result stops being a four-cell
    instruction.

    The arithmetic happens on the integer the whole double names.  It could
    not happen in a float: 20010203 is past the wall at 2^24, where a float
    counts by twos and this number rounds to 20010204 before anything is
    added.  A double counts by ones to 2^53, so float(int) is exact for every
    eight-digit instruction.  None means an opcode off the table, a field
    naming a register the fleet's machine cannot run, a field the instruction
    does not use, or an op that would move the layout (LDI, the branches);
    the caller keeps the word or tries another delta."""
    m = int(word) + delta
    if not 0 <= m < isa.LIMIT:
        return None
    w = float(m)
    ins = isa.decode(w)
    if ins is None or isa.BY_NUMBER[ins[0]].cells != 4:
        return None
    for code in ins[1:]:
        if code >= 96:
            return None     # q0..q3 are legal ISA, but gpu/src/machine.h has no quad yet
    return w


def mutate_word(word, rng, tries=8):
    """One neighbour of `word`: a whole number of fields or opcodes away.

    Most edits die in edit() -- a field walked into 96..99, an opcode off the
    table -- so each try redraws the delta.  If nothing legal turns up the
    word rides unchanged; the population's immigrants carry novelty instead."""
    for _ in range(tries):
        r = rng.random()
        if r < 0.20:      # the opcode: 20 ADD to 22 MUL is one step
            delta = 10 ** 6 * rng.choice((-3, -2, -1, 1, 2, 3))
        elif r < 0.60:    # the first operand: 64 d0 to 65 d1 is one step
            delta = 10 ** 4 * rng.choice((-3, -2, -1, 1, 2, 3))
        elif r < 0.80:
            delta = 100 * rng.choice((-3, -2, -1, 1, 2, 3))
        else:
            delta = rng.choice((-3, -2, -1, 1, 2, 3))
        w = edit(word, delta)
        if w is not None:
            return w
    return word


def _reg(rng, file=None):
    """A register code.  The double file is drawn four to one: the body's work
    is a d0 -> d4 function, and a body of halves mostly scores nothing."""
    f = file if file is not None else rng.choices((1, 2, 3), weights=(1, 1, 4))[0]
    return isa.code_of(f, rng.randrange(32))


def random_word(rng):
    """A legal four-cell instruction: a weighted opcode from the pool with
    well-formed operand fields.  The type rules the decode does not check
    (one type across the operands, LD and ST through a float register) are
    kept here, so a fresh body runs; mutation is freer and may deal a fault,
    which is then just the worst kind of fitness."""
    name = rng.choices(list(POOL), weights=list(POOL.values()))[0]
    spec = isa.BY_NAME[name]
    a = b = c = 0
    if spec.same_type:
        f = rng.choices((1, 2, 3), weights=(1, 1, 4))[0]
        a = isa.code_of(f, rng.randrange(32))
        if "b" in spec.used:
            b = isa.code_of(f, rng.randrange(32))
        if "c" in spec.used:
            c = isa.code_of(f, rng.randrange(32))
    elif name == "LD":
        a, b = _reg(rng), _reg(rng, 2)
    elif name == "ST":
        a, b = _reg(rng, 2), _reg(rng)
    elif name == "CVT":
        fa, fb = rng.sample((1, 2, 3), 2)
        a, b = isa.code_of(fa, rng.randrange(32)), isa.code_of(fb, rng.randrange(32))
    elif name == "SPLIT":
        f = rng.choices((1, 2, 3), weights=(1, 1, 4))[0]
        a, b, c = isa.code_of(f, rng.randrange(32)), isa.code_of(f, rng.randrange(32)), _reg(rng)
    elif name == "JOIN":
        f = rng.choices((1, 2, 3), weights=(1, 1, 4))[0]
        a, b, c = _reg(rng), isa.code_of(f, rng.randrange(32)), isa.code_of(f, rng.randrange(32))
    elif name == "MANT":
        a, b, c = _reg(rng), _reg(rng), _reg(rng)
    return float(isa.encode(spec.number, a, b, c))


def random_body(rng, slots):
    return [random_word(rng) for _ in range(slots)]


def seeded_body(slots):
    """The template start: d4 = x*x, d4 = d4 + 1.0, then blanks.  It scores
    5.0 on the default cases -- wrong by exactly one register in the add --
    so a small population has a graded hill to descend and the search proper
    has something to do from its first generation.  The demo starts random;
    this is for small runs and for tests."""
    slots = max(slots, 2)
    return [float(isa.encode(22, 68, 64, 64)),    # MUL.D  d4, d0, d0
            float(isa.encode(20, 68, 68, 65)),    # ADD.D  d4, d4, d1
            *([float(isa.encode(1, 0, 0, 0))] * (slots - 2))]


def mutate_body(parent, donor, rng):
    """One child of `parent`: per slot, splice the donor's slot, redraw the
    slot from the pool, or take one SMC edit.  The splice is what assembles
    multi-slot answers -- one parent that found the multiply, another that
    found the add, a child that carries both."""
    child = list(parent)
    for i in range(len(child)):
        r = rng.random()
        if r < SPLICE:
            child[i] = donor[rng.randrange(len(donor))]
        elif r < SPLICE + MACRO:
            child[i] = random_word(rng)
        elif r < SPLICE + MACRO + RATE:
            child[i] = mutate_word(child[i], rng)
    return child


def next_generation(population, scores, rng, elite, immigrants):
    """Elites ride over unchanged; then mutated children of tournament
    winners; then fresh random bodies.  The best fitness never worsens: an
    elite is only displaced by something better or equal."""
    n = len(population)
    order = sorted(range(n), key=lambda i: scores[i])
    pool = order[:max(2, n // 2)]  # the fitter half can parent

    def winner():
        picks = rng.sample(pool, min(3, len(pool)))
        return population[min(picks, key=lambda i: scores[i])]

    children = [[*population[i]] for i in order[:elite]]
    while len(children) < n - immigrants:
        children.append(mutate_body(winner(), winner(), rng))
    children += [random_body(rng, len(population[0])) for _ in range(immigrants)]
    return children


# --- the skeleton and its layout --------------------------------------------------------

# One shared image; a machine's body and its cases arrive as patches.  The
# loop keeps small constants on hand in d1 = 1.0, d2 = 2.0 and d3 = 0.25
# (re-loaded each pass, so the body may clobber them), reads x into d0, and
# stores d4.  Everything between is the evolved body.
def _template(ncases, slots):
    lines = ["        LDI.F  f5, cases",
             "        LDI.F  f6, outs",
             "        LDI.F  f7, 4",
             f"        LDI.H  h1, {ncases}",
             "        LDI.H  h2, 1",
             "loop:   LD.D   d0, [f5]",
             "        LDI.D  d1, 1",
             "        LDI.D  d2, 2",
             "        LDI.D  d3, 0.25"]
    lines += [f"slot{k}:  DATA.D 0" for k in range(1, slots + 1)]
    lines += ["        ST.D   [f6], d4",
              "        ADD.F  f5, f5, f7",
              "        ADD.F  f6, f6, f7",
              "        SUB.H  h1, h1, h2",
              "        BNZ.H  h1, loop",
              "        HALT",
              f"cases:  SPACE {4 * ncases}",
              f"outs:   SPACE {4 * ncases}"]
    return "\n".join(lines) + "\n"


class Layout:
    def __init__(self, ncases, slots):
        image = asm.assemble(_template(ncases, slots))
        self.cells = image.cells
        self.entry = float(image.start)
        self.slots = [int(image.symbols[f"slot{k}"]) for k in range(1, slots + 1)]
        self.cases = int(image.symbols["cases"])
        self.outs = int(image.symbols["outs"])
        self.memory = max(256, (self.outs + 4 * ncases + 15) // 16 * 16)
        self.max_steps = ncases * (13 + slots) + 64


def write_double(mem, addr, value):
    mem.cells[addr:addr + 4] = array.array("H", fp.pack(fp.D, value))


def pack_double(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


# --- evaluation: the fleet, or the reference --------------------------------------------

def job_bytes(layout, population, cases, max_steps):
    """One array job: one machine per candidate, its genome and its cases as
    patches, one pick per case."""
    n_in = len(cases) + len(population[0])
    n_out = len(cases)
    patches = b""
    for body in population:
        for k, word in enumerate(body):
            patches += struct.pack("<IIQ", layout.slots[k], FMT_DOUBLE, pack_double(float(word)))
        for k, x in enumerate(cases):
            patches += struct.pack("<IIQ", layout.cases + 4 * k, FMT_DOUBLE, pack_double(x))
    picks = b"".join(struct.pack("<II", layout.outs + 4 * k, FMT_DOUBLE)
                     for _ in population for k in range(n_out))
    header = b"M1AJ" + struct.pack("<IIIQIIIIIII", 1, len(population), layout.memory, max_steps,
                                   n_in, n_out, len(layout.cells), 0, 0, layout.memory, 0)
    return header + patches + picks + bytes(struct.pack("<%dH" % len(layout.cells), *layout.cells))


def read_result(path, instances, n_outputs):
    """(steps, status, fault, values) per machine, as main.cpp writes them:
    M1AR, version, instances, n_outputs, then a row per machine."""
    data = Path(path).read_bytes()
    if data[:4] != b"M1AR":
        raise ValueError(f"{path}: not an array result file")
    version, count, nout = struct.unpack_from("<III", data, 4)
    if version != 1 or count != instances or nout != n_outputs:
        raise ValueError(f"{path}: a result of version {version}, {count} machines, "
                         f"{nout} outputs; expected 1, {instances}, {n_outputs}")
    rows = []
    at = 16
    for _ in range(instances):
        steps, = struct.unpack_from("<Q", data, at)
        status, fault, flags, spare = struct.unpack_from("<IIII", data, at + 8)
        values = struct.unpack_from("<%dd" % n_outputs, data, at + 24)
        rows.append((steps, status, fault, values))
        at += 24 + 8 * n_outputs
    return rows


HALTED, FAULTED, RUNNING = 1, 2, 0  # status values of gpu/src/machine.h


def evaluate_fleet(gpu, layout, population, cases, max_steps, work):
    """One array run per call: the whole population, one machine each."""
    job, result = Path(work) / "job.bin", Path(work) / "result.bin"
    job.write_bytes(job_bytes(layout, population, cases, max_steps))
    done = subprocess.run([gpu, "array", str(job), "--out", str(result)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"array run failed: {done.stderr.strip()}")
    timing = re.search(r"kernel ([0-9.]+) ms", done.stdout)
    rows = read_result(result, len(population), len(cases))
    return _score_rows(rows, cases, float(timing.group(1)) if timing else 0.0)


def evaluate_reference(layout, population, cases, max_steps):
    """The same evaluation on mantissa.machine, one run per candidate.  Some
    operands the port survives stop the reference with a host exception (DIV
    by infinity: see gpu/NOTES.md and the fuzzer's `refused` counter), so a
    run that raises is scored as the fault it would be on the fleet."""
    rows = []
    for body in population:
        mem = Memory(layout.memory)
        mem.write(layout.cells, 0)
        for k, word in enumerate(body):
            write_double(mem, layout.slots[k], float(word))
        for k, x in enumerate(cases):
            write_double(mem, layout.cases + 4 * k, x)
        regs = [0.0] * 32
        regs[31] = float(layout.memory)
        try:
            result = machine.run(Cpu(f=tuple(regs), pc=layout.entry), mem, max_steps=max_steps)
            values = tuple(mem.load(fp.D, layout.outs + 4 * k) for k in range(len(cases)))
            status = {"halt": HALTED, "fault": FAULTED, "step limit": RUNNING}[result.reason]
            steps = result.steps
        except (OverflowError, ValueError, ZeroDivisionError):
            values, status, steps = (0.0,) * len(cases), FAULTED, 0
        rows.append((steps, status, 0, values))
    return _score_rows(rows, cases, None)


def _score_rows(rows, cases, kernel_ms):
    scores = [sse(values, cases) for _, _, _, values in rows]
    info = {"faulted": sum(1 for r in rows if r[1] == FAULTED),
            "limited": sum(1 for r in rows if r[1] == RUNNING),
            "kernel_ms": kernel_ms,
            "values": [r[3] for r in rows]}
    return scores, info


def find_binary(path):
    """The GPU binary, as given or from the repo root; None if it is not there."""
    for candidate in (Path(path), Path(__file__).resolve().parents[2] / path):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


# --- the loop ---------------------------------------------------------------------------

def search(evaluate, population, rng, generations, elite, immigrants, threshold, report, restart=75):
    """The generational loop.  Returns (body, score, generation, converged):
    the champion as last evaluated, which generation it was found, and
    whether the threshold was met before the generation budget ran out.

    On a stall -- `restart` generations without an improvement -- the bottom
    of the population is redrawn and the top `elite` ride over.  Selection
    has collapsed onto one stuck elite by then, and cloning the champion
    alone would also wipe the near-champion cloud the population has
    accumulated -- often the part that holds the last mile -- so the redraw
    keeps the top as they are and replaces the rest with fresh bodies."""
    champion, best_score, found = population[0], float("inf"), 0
    stalled = 0
    for generation in range(generations):
        scores, info = evaluate(population)
        order = sorted(range(len(population)), key=lambda i: scores[i])
        best, mean = scores[order[0]], sum(scores) / len(scores)
        if best < best_score:
            champion, best_score, found = population[order[0]], best, generation
            stalled = 0
        else:
            stalled += 1
        report(generation, best, mean, population[order[0]], info)
        if best <= threshold:
            return champion, best_score, generation, True
        if restart and stalled >= restart:
            keep = min(elite, len(population))
            population = ([[*population[i]] for i in order[:keep]]
                          + [random_body(rng, len(champion)) for _ in range(len(population) - keep)])
            stalled = 0
            report(generation, best, mean, champion, {**info, "restart": True})
        else:
            population = next_generation(population, scores, rng, elite, immigrants)
    return champion, best_score, found, False


# --- listings ---------------------------------------------------------------------------

def disasm_word(word):
    """One instruction number, as text."""
    mem = Memory(4)
    write_double(mem, 0, word)
    return disassemble(mem, 0)[0]


def disasm_program(layout, body):
    """The skeleton with `body` in its slots, from entry to the HALT."""
    mem = Memory(layout.memory)
    mem.write(layout.cells, 0)
    for k, word in enumerate(body):
        write_double(mem, layout.slots[k], float(word))
    lines, addr = [], layout.entry
    while addr < len(layout.cells):
        text, cells = disassemble(mem, addr)
        lines.append(f"{int(addr):4d}  {text}")
        if text == "HALT":
            break
        addr += cells
    return "\n".join(lines)


# --- the demo ---------------------------------------------------------------------------

CASES = (-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
         5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5)  # exact quarters: targets are exact too


def demo(args):
    rng = random.Random(args.seed)
    cases = CASES[:args.cases]
    layout = Layout(len(cases), args.slots)
    max_steps = args.max_steps or layout.max_steps
    gpu = None if args.cpu else find_binary(args.gpu)

    if gpu:
        print(f"evolve: fleet {gpu} array, one run of {args.population} machines per generation")
        work = Path("/tmp/m1-evolve")
        work.mkdir(exist_ok=True)
        base = lambda pop: evaluate_fleet(gpu, layout, pop, cases, max_steps, work)
    else:
        why = "asked (--cpu)" if args.cpu else f"no GPU binary at {args.gpu}"
        print(f"evolve: {why}; evaluating on the reference simulator, one run per candidate")
        base = lambda pop: evaluate_reference(layout, pop, cases, max_steps)

    walls, kernels = [], []

    def evaluate(population):
        began = time.time()
        scores, info = base(population)
        walls.append(time.time() - began)
        if info["kernel_ms"]:
            kernels.append(info["kernel_ms"])
        return scores, info

    seen = [float("inf")]

    def report(generation, best, mean, body, info):
        marks = []
        if info["faulted"]:
            marks.append(f"{info['faulted']} faulted")
        if info["limited"]:
            marks.append(f"{info['limited']} at the limit")
        if info.get("restart"):
            marks.append("redrawn around the champion")
        note = f" [{', '.join(marks)}]" if marks else ""
        if best < seen[0]:
            seen[0] = best
            listing = " | ".join(disasm_word(w) for w in body)
            print(f"gen {generation:4d}: best {best:.6g}  mean {mean:.6g}{note}  {listing}")
        elif generation % 25 == 0 or info.get("restart"):
            print(f"gen {generation:4d}: best {best:.6g}  mean {mean:.6g}{note}")

    print(f"evolve: target {TARGET_TEXT} over the cases {list(cases)}")
    print(f"evolve: population {args.population}, {args.slots} slots, "
          f"elite {args.elite}, immigrants {args.immigrants}, threshold {args.threshold:g}")
    print(f"evolve: {len(layout.cells)} cells of skeleton, slots at {layout.slots}, "
          f"cases at {layout.cases}, outs at {layout.outs}, memory {layout.memory} cells, "
          f"step limit {max_steps}")

    began = time.time()
    champion, score, found, converged = search(
        evaluate, [random_body(rng, args.slots) for _ in range(args.population)], rng,
        args.generations, args.elite, args.immigrants, args.threshold, report, args.restart)
    wall = time.time() - began
    generations_ran = len(walls)  # the cross-check below evaluates once more; keep it out of the timing

    print()
    print(disasm_program(layout, champion))
    print()
    if converged:
        print(f"converged: generation {found}, best sum of squared error {score:.3g}")
    else:
        print(f"not converged at generation {args.generations}: best sum of squared "
              f"error {score:.3g}, found at generation {found}")

    # the reference reads the champion back, and where the fleet ran, its own
    # picks are checked against the reference case by case
    _, check = evaluate_reference(layout, [champion], cases, max_steps)
    mine = check["values"][0]
    if gpu:
        _, info = evaluate([champion])
        theirs = info["values"][0]
        agree = sum(1 for a, b in zip(theirs, mine) if a == b)
        print(f"fleet against reference: {agree} of {len(cases)} outputs agree")

    exact = 0
    for x, out in zip(cases, mine):
        want = target(x)
        exact += out == want
        print(f"  x = {x:5g}   out {out!r:24}   want {want!r}")
    print(f"{exact} of {len(cases)} outputs exact")

    per_gen = wall / max(1, generations_ran)
    print(f"{('fleet' if gpu else 'reference')}: {generations_ran} generations in {wall:.1f} s, "
          f"{per_gen * 1000:.0f} ms a generation", end="")
    if kernels:
        print(f" (kernel {sum(kernels) / len(kernels):.2f} ms)", end="")
    print()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true", help="run the evolution demo")
    parser.add_argument("--cpu", action="store_true", help="evaluate on the reference simulator")
    parser.add_argument("--gpu", default="gpu/build/mantissa-gpu", help="path of the GPU binary")
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--generations", type=int, default=900)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=1e-9)
    parser.add_argument("--elite", type=int, default=16)
    parser.add_argument("--immigrants", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = sized from the cases and slots")
    parser.add_argument("--restart", type=int, default=75,
                        help="generations without improvement before the population is redrawn; 0 disables")
    args = parser.parse_args()
    if not args.demo:
        parser.print_help()
        return 2
    if not 2 <= args.cases <= len(CASES):
        parser.error(f"--cases must be between 2 and {len(CASES)}")
    if not 2 <= args.slots <= 8:
        parser.error("--slots must be between 2 and 8")
    if args.elite + args.immigrants >= args.population:
        parser.error("the population must exceed elites plus immigrants")
    return demo(args)


if __name__ == "__main__":
    sys.exit(main())
