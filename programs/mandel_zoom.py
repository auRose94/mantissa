"""The Mandelbrot program in all three precisions, at any zoom.

    python programs/mandel_zoom.py [WIDTH ...]

Half, float and double get the same program and the same window.  Each WIDTH
is a window centred on the seahorse valley.  With no arguments the tour starts
with the whole set and zooms in, and the formats run out of digits in turn.

This is the premise of the machine in one picture: the program never changes,
only the type it is written for, and what breaks is arithmetic itself.
"""
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from mantissa import asm  # noqa: E402
from mantissa.machine import Cpu, console, run  # noqa: E402
from mantissa.memory import Memory  # noqa: E402

SOURCE = Path(__file__).with_name("mandel.mant")
CENTER = (0.0, 1.0)  # the point i: a Misiurewicz point, detailed at every scale
ASPECT = 2.2 / 2.6   # window height over width, as in mandel.mant
TOUR = (2.6, 0.02, 0.0008, 0.000008, 0.0000002)

VARIANTS = (
    ("half", "H", lambda n: f"h{n}", False),
    ("float", "F", lambda n: f"f{n + 15}", True),  # f0 to f14 are taken by counters and addresses
    ("double", "D", lambda n: f"d{n}", False),
)


def retyped(source, suffix, register, counters_too):
    """mandel.mant with its double arithmetic written for another format.

    `register(n)` names the stand-in for dN.  When the arithmetic shares a
    format with the loop counters, converting to or from a counter is a copy.
    """
    lines = []
    for line in source.splitlines():
        code, sep, comment = line.partition(";")
        to_arithmetic = re.match(r"(\s*)CVT\.D\s+d(\d+),\s*(f\d+)\s*$", code)
        from_arithmetic = re.match(r"(\s*)CVT\.F\s+(f\d+),\s*d(\d+)\s*$", code)
        if to_arithmetic:
            code = (f"{to_arithmetic[1]}{'MOV' if counters_too else 'CVT'}.{suffix} "
                    f"{register(int(to_arithmetic[2]))}, {to_arithmetic[3]}")
        elif from_arithmetic:
            code = (f"{from_arithmetic[1]}{'MOV.F' if counters_too else 'CVT.F'} "
                    f"{from_arithmetic[2]}, {register(int(from_arithmetic[3]))}")
        else:
            code = re.sub(r"\bd(\d+)", lambda m: register(int(m[1])), code.replace(".D", "." + suffix))
        lines.append(f"{code.rstrip()}  {sep}{comment}" if sep else code.rstrip())
    return "\n".join(lines)


def windowed(source, left, right, bottom, top):
    """Point the program's four edge literals at another window."""
    edges = iter((left, right, bottom, top))
    return re.sub(r"^(\s*LDI\.[HFD]\s+[hfd]\d+,)\s*[^\s;]+",
                  lambda m: f"{m[1]} {next(edges)!r}", source, count=4, flags=re.M)


def render(source):
    """(picture, steps) for one variant of the program."""
    image = asm.assemble(source)
    memory = Memory(65536)
    memory.write(image.cells)
    out = io.StringIO()
    result = run(Cpu(pc=image.entry), memory, out=console(out.write), max_steps=5_000_000)
    return out.getvalue(), result.steps


def main(argv=None):
    widths = [float(a) for a in (sys.argv[1:] if argv is None else argv)]
    windows = []
    for width in widths or TOUR:
        half = width / 2
        # the tour opens on the whole set; every other window is centred on CENTER
        center = (-0.7, 0.0) if width == 2.6 else CENTER
        windows.append((center[0] - half, center[0] + half,
                        center[1] - half * ASPECT, center[1] + half * ASPECT))
    source = SOURCE.read_text(encoding="utf-8")
    for left, right, bottom, top in windows:
        print(f"; window: x from {left:.9g} to {right:.9g}, y from {bottom:.9g} to {top:.9g}")
        seen = {}
        for label, suffix, register, counters_too in VARIANTS:
            program = windowed(retyped(source, suffix, register, counters_too), left, right, bottom, top)
            picture, steps = render(program)
            print(f"; {label}, {steps} steps" + (f": the same picture as the {seen[picture]}" if picture in seen else ""))
            if picture not in seen:
                seen[picture] = label
                print(picture, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
