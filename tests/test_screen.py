"""The screen device: the framebuffer convention, and the watching console.

The machine paints; the console watches.  These tests check the contract
between them: the packed colour, the cell-to-image conversion, and a real
painting program observed through the console's own machinery.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, ".")

from mantissa.machine import Cpu, run
from mantissa.memory import Memory

from gpu.python.screen import Console, decode_colour, image_from_memory, pack_colour

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class PackedColours(unittest.TestCase):
    def test_pack_and_decode_round_trip(self):
        for r, g, b in [(0, 0, 0), (255, 255, 255), (1, 0, 1), (12, 34, 56), (255, 128, 1)]:
            packed = pack_colour(r, g, b)
            self.assertTrue(packed < 16_777_216.0, "a colour rides under the float wall")
            self.assertEqual(decode_colour(packed), (r, g, b))

    def test_packing_is_exact_in_a_double(self):
        packed = pack_colour(255, 255, 255)      # 16,777,215: one below the wall
        self.assertEqual(packed, 255 * 65536 + 255 * 256 + 255)

    def test_pack_refuses_what_a_pixel_cannot_hold(self):
        for parts in [(256, 0, 0), (-1, 0, 0), (0.5, 0, 0), (0, 0, 1.5)]:
            with self.assertRaises(ValueError):
                pack_colour(*parts)


class WatchingConsole(unittest.TestCase):
    @unittest.skipUnless(HAS_PIL, "PIL is not installed")
    def test_image_from_memory_reads_the_convention(self):
        memory = Memory(4096)
        fb = 1024
        # the four pixels of a 2x2 framebuffer, row major
        values = [pack_colour(255, 0, 0), pack_colour(0, 255, 0),
                  pack_colour(0, 0, 255), pack_colour(9, 8, 7)]
        from mantissa import fp
        for k, v in enumerate(values):
            memory.store(fp.D, fb + 4 * k, v)
        image = image_from_memory(memory, fb, 2, 2)
        self.assertEqual(image.size, (2, 2))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(image.getpixel((1, 0)), (0, 255, 0))
        self.assertEqual(image.getpixel((0, 1)), (0, 0, 255))
        self.assertEqual(image.getpixel((1, 1)), (9, 8, 7))

    @unittest.skipUnless(HAS_PIL, "PIL is not installed")
    def test_the_painter_observed_through_the_console(self):
        here = Path(__file__).resolve().parents[1]
        console = Console(here / "programs" / "plasma.mant", width=64, height=48)
        console.step(10_000)  # the top-left band of frame 0 is painted by now
        image = console.frame()
        self.assertEqual(image.size, (64, 48))
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(image.getpixel((1, 0)), (1, 0, 1))
        self.assertEqual(image.getpixel((3, 5)), (3, 5, 8))
        # the far corner has not been painted at 10,000 steps: still zeroed memory
        self.assertEqual(image.getpixel((63, 47)), (0, 0, 0))

    @unittest.skipUnless(HAS_PIL, "PIL is not installed")
    def test_frames_move_the_stripes(self):
        here = Path(__file__).resolve().parents[1]
        console = Console(here / "programs" / "plasma.mant", width=64, height=48)
        console.step(200_000)  # four frames complete (about 49,500 steps each)
        image = console.frame()
        first = image.getpixel((0, 0))
        self.assertEqual(first[0], 4, "four finished frames: t counted to 4, r = t at the corner")


class InteractiveInput(unittest.TestCase):
    """Input streams in through the convention too: the console writes the
    key cells, the machine reads them.  These tests feed inject_keys the way
    the terminal would."""

    HERE = Path(__file__).resolve().parents[1]

    def console(self):
        return Console(self.HERE / "programs" / "pong.mant", width=64, height=48)

    def cell(self, console, label, which):
        return console.image.symbols[label] + 4 * which

    def read(self, console, label, which):
        from mantissa import fp
        return console.memory.load(fp.D, self.cell(console, label, which))

    def test_a_paddle_steers_by_the_key_cells(self):
        console = self.console()
        pressed = {"left"}
        console.inject_keys(pressed)
        before = self.read(console, "pad", 0)
        console.step(5_000)     # the physics runs at the frame's start
        after = self.read(console, "pad", 0)
        self.assertEqual(after, before - 3.0)   # left: px -= 3
        console.inject_keys({"right"})
        console.step(130_000)   # a whole frame: the next physics hides deep in the paint
        self.assertEqual(self.read(console, "pad", 0), before)  # and back again

    def test_the_ball_bounces_off_the_left_wall(self):
        console = self.console()
        from mantissa import fp
        # park the ball at the left edge, drifting out
        ball = console.image.symbols["ball"]
        console.memory.store(fp.D, ball + 4, 24.0)      # by
        console.memory.store(fp.D, ball, 0.5)           # bx
        console.memory.store(fp.D, ball + 8, -1.0)      # vx: heading out
        console.step(130_000)   # two frames: plenty for the ball to leave and return
        self.assertEqual(self.read(console, "ball", 2), 1.0, "the left wall flips vx to +1")
        self.assertGreaterEqual(self.read(console, "ball", 0), 0.0)

    def test_the_paddle_bounces_the_ball(self):
        console = self.console()
        from mantissa import fp
        ball = console.image.symbols["ball"]
        pad = console.image.symbols["pad"]
        console.memory.store(fp.D, pad, 32.0)           # paddle centred
        console.memory.store(fp.D, ball, 32.0)          # bx under the paddle
        console.memory.store(fp.D, ball + 4, 44.0)      # by just below the bounce line
        console.memory.store(fp.D, ball + 8, 1.0)
        console.memory.store(fp.D, ball + 12, 1.0)      # vy heading down
        console.step(130_000)
        self.assertEqual(self.read(console, "ball", 3), -1.0, "the paddle sends vy up")

    def test_key_cells_reset_between_frames(self):
        console = self.console()
        console.inject_keys({"left"})
        self.assertEqual(self.read(console, "input", 2), 1.0)
        console.inject_keys(set())                       # a frame with nothing pressed
        self.assertEqual(self.read(console, "input", 2), 0.0)
