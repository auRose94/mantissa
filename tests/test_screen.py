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

    def test_frames_move_the_stripes(self):
        here = Path(__file__).resolve().parents[1]
        console = Console(here / "programs" / "plasma.mant", width=64, height=48)
        console.step(200_000)  # four frames complete (about 49,500 steps each)
        image = console.frame()
        first = image.getpixel((0, 0))
        self.assertEqual(first[0], 4, "four finished frames: t counted to 4, r = t at the corner")
