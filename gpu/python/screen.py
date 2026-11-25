"""screen.py -- the watching console: a framebuffer in memory, a window beside it.

    python3 screen.py programs/plasma.mant --play            a live window (ffplay)
    python3 screen.py programs/plasma.mant --record out.mp4  a video
    python3 screen.py programs/plasma.mant --pngs frames     snapshots as PNG

The convention (see EXTENSIONS.md): the framebuffer is a row of W*H doubles
at the label `fb`, one pixel per double, row major, top left first, the
colour packed as r*65536 + g*256 + b with r, g and b whole numbers of at most
255 -- exact in a double, and deliberately just under the float counting wall
at 16,777,216.  The program paints; this console watches: it runs the program
in step bursts, reads the framebuffer cells, and hands RGB frames to ffmpeg
(a window through ffplay, a video through libx264, or PNG snapshots).  No
instruction ever touches a window.

The machine computes; the console watches.  What the console must not do is
interpret: every colour it reads was computed by the machine.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mantissa import asm, fp
from mantissa.machine import Cpu, run
from mantissa.memory import Memory

try:
    from PIL import Image
except ImportError:  # snapshots and scaling want PIL; a plain pipe does not
    Image = None


def decode_colour(value):
    """(r, g, b) from the packed colour, by the host, which may use integers."""
    c = int(round(value)) % 16_777_216
    return (c // 65536, c // 256 % 256, c % 256)


def pack_colour(r, g, b):
    """The double a pixel paints: r*65536 + g*256 + b, exact below the wall."""
    for part in (r, g, b):
        if not (0 <= part <= 255 and part == int(part)):
            raise ValueError(f"a colour part must be a whole number 0..255, not {part!r}")
    return float(r * 65536 + g * 256 + b)


def image_from_memory(memory, fb_addr, width, height, scale=1):
    """The framebuffer cells, as a PIL image.  Row major, top left first."""
    if Image is None:
        raise RuntimeError("PIL is needed for images: pip install pillow")
    pixels = bytearray()
    for row in range(height):
        for col in range(width):
            r, g, b = decode_colour(memory.load(fp.D, fb_addr + 4 * (row * width + col)))
            pixels += bytes((r, g, b))
    image = Image.frombytes("RGB", (width, height), bytes(pixels))
    if scale != 1:
        image = image.resize((width * scale, height * scale), Image.NEAREST)
    return image


class Console:
    """Runs a painting program and watches its framebuffer."""

    def __init__(self, source_path, width=64, height=48, memory_cells=65536):
        source = Path(source_path).read_text(encoding="utf-8")
        self.image = asm.assemble(source)
        self.width, self.height = width, height
        self.fb = self.image.symbols["fb"]        # the label is a cell address
        self.input_addr = self.image.symbols.get("input")  # key cells, if the program asks
        self.memory = Memory(memory_cells)
        self.memory.write(self.image.cells)
        self.cpu = Cpu(f=(0.0,) * 31 + (float(memory_cells),), pc=self.image.entry)

    KEY_NAMES = ("up", "down", "left", "right")   # input cell k is key k, 1.0 pressed

    def inject_keys(self, pressed):
        """Write the key cells: 1.0 for a pressed key, 0.0 for a released one.
        This is the whole input path -- the console writes memory, the
        machine reads it, and neither breaks the rule."""
        if self.input_addr is None:
            return
        from mantissa import fp
        for k, name in enumerate(self.KEY_NAMES):
            self.memory.store(fp.D, self.input_addr + 4 * k,
                              1.0 if name in pressed else 0.0)

    def step(self, steps):
        """Run a burst of steps; a step limit is how a frame ends here, not a fault."""
        result = run(self.cpu, self.memory, max_steps=steps)
        self.cpu = result.cpu

    def frame(self, scale=1):
        return image_from_memory(self.memory, self.fb, self.width, self.height, scale)

    def frame_bytes(self):
        """The frame as rgb24 bytes, ready for a rawvideo pipe."""
        image = self.frame()
        return image.tobytes()

    def play(self, frames, steps_per_frame, fps=30, scale=4, record=None, pngs=None, keys=False):
        """Run and watch.  One of: an ffplay window (default), --record path
        (libx264 video), --pngs directory (snapshots every frames//6 frames).
        With keys=True the terminal steers: arrows or wasd write the key
        cells, q quits."""
        if record:
            pipe = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{self.width * scale}x{self.height * scale}", "-r", str(fps), "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", record], stdin=subprocess.PIPE)
        elif pngs:
            Path(pngs).mkdir(parents=True, exist_ok=True)
            pipe = None
        else:
            pipe = subprocess.Popen(
                ["ffplay", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{self.width * scale}x{self.height * scale}", "-i", "-"],
                stdin=subprocess.PIPE)
        if keys:
            import termios
            import tty
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        try:
            buf = []
            quit_now = False
            for frame in range(frames):
                if keys:
                    pressed, buf = read_keys(buf)
                    if "quit" in pressed:
                        break
                    self.inject_keys(pressed)
                self.step(steps_per_frame)
                image = self.frame(scale)
                if pipe is not None:
                    pipe.stdin.write(image.tobytes())
                if pngs and frame % max(1, frames // 6) == 0:
                    image.save(f"{pngs}/frame{frame:04d}.png")
        finally:
            if keys:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            if pipe is not None:
                pipe.stdin.close()
                pipe.wait(timeout=10)


KEY_CHARS = {"w": "up", "s": "down", "a": "left", "d": "right",
             "A": "up", "B": "down", "D": "left", "C": "right"}


def read_keys(buf):
    """Every key ready in the terminal, as names: wasd, or the arrows as the
    three-byte escape sequences A/B/C/D.  'q' comes back as quit."""
    import select
    out = set()
    while select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        buf.append(ch)
        if ch == "q":
            out.add("quit")
            buf.clear()
        elif ch in "wasd":
            out.add(KEY_CHARS[ch])
            buf.clear()
        elif ch == "\x1b":
            if len(buf) >= 3 and buf[-3] == "\x1b" and buf[-2] == "[" and buf[-1] in "ABCD":
                out.add(KEY_CHARS[buf[-1]])
                buf.clear()
            # a lone escape: quit
            elif len(buf) == 1:
                out.add("quit")
                buf.clear()
        elif ch not in "[":
            buf.clear()
    return out, buf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program")
    parser.add_argument("--keys", action="store_true",
                        help="steer from the terminal: arrows or wasd, q quits")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--steps-per-frame", type=int, default=50_000,
                        help="how many machine steps make one frame (default 50000)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--record", metavar="PATH")
    parser.add_argument("--pngs", metavar="DIR")
    args = parser.parse_args()

    console = Console(args.program, args.width, args.height)
    console.play(args.frames, args.steps_per_frame, args.fps, args.scale,
                 record=args.record, pngs=args.pngs, keys=args.keys)
    return 0


if __name__ == "__main__":
    sys.exit(main())
