"""Tests for gpu/python/evolve.py: the mutation arithmetic, the fitness, the
job and result shapes of the fleet path, and a tiny evolution run on the
reference simulator.  No GPU is required here; the fleet demo is
`python3 gpu/python/evolve.py --demo`."""
from __future__ import annotations

import random
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gpu" / "python"))

from mantissa import fp, isa  # noqa: E402
import evolve  # noqa: E402


def word(opcode, a=0, b=0, c=0):
    return float(isa.encode(opcode, a, b, c))


class TestMutationArithmetic(unittest.TestCase):
    def test_edit_walks_the_isa(self):
        # the three edits the README shows for 20010203, ADD h1, h2, h3
        self.assertEqual(evolve.edit(20010203.0, 10000), 20020203.0)     # h1 -> h2
        self.assertEqual(evolve.edit(20010203.0, 320000), 20330203.0)    # h file -> f file
        self.assertEqual(evolve.edit(20010203.0, 5000000), 25010203.0)   # ADD -> FMA
        # and each result still decodes to what it says
        self.assertEqual(isa.decode(evolve.edit(20010203.0, 10000)), (20, 2, 2, 3))
        self.assertEqual(isa.decode(evolve.edit(20010203.0, 5000000))[0], 25)

    def test_edit_refuses_the_illegal(self):
        self.assertIsNone(evolve.edit(20010203.0, -20000000))   # negative is not a number
        self.assertIsNone(evolve.edit(85999999.0, 1000000))     # opcode 86 is off the table
        self.assertIsNone(evolve.edit(20109500.0, 100))         # field b walked into 96..99
        self.assertIsNone(evolve.edit(word(70), 100))           # PUT's unused field must stay 00
        self.assertIsNone(evolve.edit(word(10), 1000000))       # landing on LDI would move the layout
        self.assertIsNone(evolve.edit(word(2), 100))            # so would HALT growing a field

    def test_edit_is_exact_whatever_the_size(self):
        # every eight-digit result survives the round trip through the double
        rng = random.Random(11)
        for _ in range(500):
            w = evolve.random_word(rng)
            delta = rng.randrange(-10 ** 6, 10 ** 6)
            edited = evolve.edit(w, delta)
            if edited is not None:
                self.assertEqual(int(edited), int(w) + delta)
                self.assertTrue(edited.is_integer())

    def test_mutate_word_stays_legal(self):
        rng = random.Random(5)
        legal = [evolve.random_word(rng) for _ in range(30)]
        for _ in range(200):
            w = evolve.mutate_word(rng.choice(legal), rng)
            ins = isa.decode(w)
            self.assertIsNotNone(ins, f"{w!r} does not decode")
            self.assertEqual(isa.BY_NUMBER[ins[0]].cells, 4, f"{w!r} would move the layout")
            legal.append(w)

    def test_mutate_moves_whole_fields(self):
        rng = random.Random(6)
        w = evolve.random_word(rng)
        for _ in range(300):
            w2 = evolve.mutate_word(w, rng)
            delta = abs(int(w2) - int(w))
            self.assertTrue(delta == 0 or delta % 10 ** 6 == 0 or delta % 10 ** 4 == 0
                            or delta % 100 == 0 or delta < 4,
                            f"edit by {int(w2) - int(w)} is not whole fields")
            w = w2

    def test_mutate_is_deterministic(self):
        first = [evolve.mutate_word(20010203.0, random.Random(9)) for _ in range(50)]
        again = [evolve.mutate_word(20010203.0, random.Random(9)) for _ in range(50)]
        self.assertEqual(first, again)


class TestFloatWall(unittest.TestCase):
    def test_an_instruction_number_is_not_a_float(self):
        # 20010203 is past the wall at 16,777,216: a float rounds it before
        # anything is added, and the inexact flag is the only trace
        naive, flags = fp.convert(fp.F, 20010203.0)
        self.assertEqual(naive, 20010204.0)
        self.assertEqual(flags, fp.INEXACT)
        self.assertNotEqual(struct.unpack("<f", struct.pack("<f", 20010203.0))[0], 20010203.0)
        exact, flags = fp.convert(fp.D, 20010203.0)
        self.assertEqual((exact, flags), (20010203.0, 0))

    def test_the_mutation_path_is_exact_where_a_float_would_slide(self):
        # in float32 the sum is wrong by one register before the add happens:
        # the stored instruction is 20010204, so +10000 lands on 20020204
        stored = struct.unpack("<f", struct.pack("<f", 20010203.0))[0]
        self.assertEqual(stored + 10000.0, 20020204.0)
        self.assertNotEqual(stored + 10000.0, 20020203.0)
        # on the integer the whole double names, it is the right answer
        self.assertEqual(evolve.edit(20010203.0, 10000), 20020203.0)

    def test_a_float_stops_counting_at_the_wall_and_edit_does_not(self):
        stuck, flags = fp.convert(fp.F, 16777216.0 + 1.0)
        self.assertEqual(stuck, 16777216.0)
        self.assertEqual(flags, fp.INEXACT)
        self.assertEqual(evolve.edit(16777215.0, 1), 16777216.0)
        self.assertEqual(evolve.edit(16777216.0, 1), 16777217.0)  # past the wall, still by ones


class TestFitness(unittest.TestCase):
    CASES = (-1.5, -0.5, 0.5, 1.5, 2.5, 3.5)

    def test_target_by_hand(self):
        self.assertEqual(evolve.target(3.0), 12.0)
        self.assertEqual(evolve.target(2.5), 8.75)
        self.assertEqual(evolve.target(-1.5), 0.75)
        self.assertEqual(evolve.target(0.0), 0.0)

    def test_sse_by_hand(self):
        self.assertEqual(evolve.sse([2.0, 6.0, 0.0], (1.0, 2.0, 3.0)), 144.0)
        self.assertEqual(evolve.sse([evolve.target(x) for x in self.CASES], self.CASES), 0.0)
        self.assertEqual(evolve.sse([0.0] * len(self.CASES), self.CASES),
                         0.75 ** 2 + 0.25 ** 2 + 0.75 ** 2 + 3.75 ** 2 + 8.75 ** 2 + 15.75 ** 2)

    def test_known_body_is_perfect_on_the_reference(self):
        layout = evolve.Layout(len(self.CASES), 4)
        body = [word(22, 68, 64, 64),   # MUL.D  d4, d0, d0
                word(20, 68, 68, 64),   # ADD.D  d4, d4, d0
                word(1), word(1)]       # NOP, NOP
        scores, info = evolve.evaluate_reference(layout, [body], self.CASES, layout.max_steps)
        self.assertEqual(scores[0], 0.0)
        self.assertEqual(info["faulted"], 0)
        self.assertEqual(list(info["values"][0]), [evolve.target(x) for x in self.CASES])

    def test_a_silent_body_scores_the_zero_floor(self):
        layout = evolve.Layout(len(self.CASES), 4)
        body = [word(1)] * 4
        scores, _ = evolve.evaluate_reference(layout, [body], self.CASES, layout.max_steps)
        self.assertGreater(scores[0], 0.0)
        self.assertEqual(scores[0], evolve.sse([0.0] * len(self.CASES), self.CASES))


class TestFleetShapes(unittest.TestCase):
    POP = [(word(22, 68, 64, 64), word(1), word(1), word(1)),
           (word(10, 68, 64, 0), word(1), word(1), word(1))]

    def setUp(self):
        self.cases = (-1.5, 0.5, 2.5)
        self.layout = evolve.Layout(len(self.cases), 4)
        self.job = evolve.job_bytes(self.layout, self.POP, self.cases, 400)

    def test_job_header_and_size(self):
        self.assertEqual(self.job[:4], b"M1AJ")
        fields = struct.unpack_from("<IIIQIIIIIII", self.job, 4)
        version, instances, cells, steps, n_in, n_out, image_cells, org, entry, sp, spare = fields
        self.assertEqual(version, 1)
        self.assertEqual((instances, cells, steps), (len(self.POP), self.layout.memory, 400))
        self.assertEqual((n_in, n_out), (len(self.cases) + 4, len(self.cases)))
        self.assertEqual((image_cells, org, entry, sp), (len(self.layout.cells), 0, 0, self.layout.memory))
        need = 52 + instances * n_in * 16 + instances * n_out * 8 + image_cells * 2
        self.assertEqual(len(self.job), need)

    def test_job_patches_carry_the_genomes_and_the_cases(self):
        at, patch = 52, []
        for _ in range(len(self.cases) + 4):
            cell, fmt, bits = struct.unpack_from("<IIQ", self.job, at)
            patch.append((cell, fmt, struct.unpack("<d", struct.pack("<Q", bits))[0]))
            at += 16
        # the first machine's first four patches are its slots, then its cases
        for k, w in enumerate(self.POP[0]):
            self.assertEqual((patch[k][0], patch[k][1], patch[k][2]),
                             (self.layout.slots[k], evolve.FMT_DOUBLE, w))
        for k, x in enumerate(self.cases):
            self.assertEqual(patch[4 + k][0], self.layout.cases + 4 * k)
            self.assertEqual(patch[4 + k][2], x)
        # the second machine's slots differ, and are its own
        offset = 52 + (len(self.cases) + 4) * 16
        cell, fmt, bits = struct.unpack_from("<IIQ", self.job, offset)
        self.assertEqual((cell, fmt, struct.unpack("<d", struct.pack("<Q", bits))[0]),
                         (self.layout.slots[0], evolve.FMT_DOUBLE, self.POP[1][0]))

    def test_result_reader_against_the_port_format(self):
        values = [(3.0, -0.25, 8.75), (0.0, 0.0, 0.0)]
        out = bytearray(b"M1AR" + struct.pack("<III", 1, len(self.POP), len(self.cases)))
        for i, triple in enumerate(values):
            out += struct.pack("<Q", 30 + i)          # steps
            out += struct.pack("<IIII", evolve.HALTED, 0, 0, 0)
            out += struct.pack("<3d", *triple)
        path = Path("/tmp/m1-evolve-test.result")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(out)
        rows = evolve.read_result(path, len(self.POP), len(self.cases))
        self.assertEqual([r[0] for r in rows], [30, 31])
        self.assertEqual([r[1] for r in rows], [evolve.HALTED, evolve.HALTED])
        self.assertEqual([r[3] for r in rows], values)


class TestTinyEvolution(unittest.TestCase):
    def test_a_small_run_improves(self):
        cases = (-0.5, 0.5, 1.5, 2.5)
        layout = evolve.Layout(len(cases), 4)
        # the template start: x*x + 1.0, one register short of the target, so
        # the first generation has a hill in sight (score 5.0) and one field
        # edit at a time can walk it down to zero.  Seed 4 converges at
        # generation 6 of 24; all thirty seeds tried improved.
        rng = random.Random(4)
        population = [evolve.seeded_body(4) for _ in range(4)]
        population += [evolve.random_body(rng, 4) for _ in range(12)]
        history = []

        def report(generation, best, mean, body, info):
            history.append(best)

        champion, score, found, converged = evolve.search(
            lambda pop: evolve.evaluate_reference(layout, pop, cases, layout.max_steps),
            population, rng, generations=24, elite=4, immigrants=1,
            threshold=1e-9, report=report)
        self.assertEqual(history[0], 5.0)                  # the template's own error
        self.assertLess(score, history[0])                 # the run went down the hill
        self.assertTrue(converged, f"stuck at {score}, generation {found}")
        self.assertEqual(score, 0.0)
        self.assertLessEqual(score, min(history))          # and never came back up
        # the champion decodes, runs, and does what the search thinks it does
        scores, _ = evolve.evaluate_reference(layout, [champion], cases, layout.max_steps)
        self.assertEqual(scores[0], score)
        for w in champion:
            self.assertIsNotNone(isa.decode(w))


if __name__ == "__main__":
    unittest.main()
