"""make_game.py -- the programs and job builders of Mantissa Dodge.

    python3 make_game.py --check     assemble both programs and print sizes

One pixel machine per pixel, and a referee machine that advances the
obstacles and tests collision.  The programs the machines run are written out
here in Mantissa-1 assembly; kernel/game.mant documents them.  The host
(run_game.py) is only the console: it patches the frame state in, picks the
pixels and the new state out, and carries the save file between frames,
because an array run always starts from its image and its patches.

Colours are packed into one double as r*65536 + g*256 + b, which is exact in
every format the machine has (the brightest, 240-white, is 15,790,320 -- just
under the float wall at 16,777,216, which is a rule of the machine, not a
choice made here).

A pixel machine's memory (double addresses):

    160 player x            patched each frame
    168, 176 obstacle 1 x, z   patched each frame
    184, 192 obstacle 2 x, z   "
    200, 208 obstacle 3 x, z   "
    240 this pixel's column    patched when the job is built
    248 this pixel's row       "
    256 out: the colour, packed

The referee machine's memory:

    160 frame number        patched each frame
    168 player x            "
    176 alive (1 or 0)      "
    184 score so far        "
    256 out: score
    264 out: alive
    272, 280 lane 1 x, z    out
    288, 296 lane 2 x, z    out
    304, 312 lane 3 x, z    out
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mantissa import asm  # noqa: E402

FMT_DOUBLE = 2  # the port's format index

W, H = 64, 48
LANES = (-0.8, 0.0, 0.8)      # obstacle x, one per lane
SPEEDS = (0.9, 1.1, 1.3)      # z cells per frame
OFFSETS = (0.0, 10.0, 20.0)   # so the lanes are out of step
ZMAX = 30.0                   # obstacles spawn at +ZMAX and travel down to -ZMAX
PLAYER_Z = 2.0                # where the player sits on the track
PLAYER_R = 0.45
OBSTACLE_R = 0.45
PIXEL_MEMORY = 320
REF_MEMORY = 320


def packed(r, g, b):
    return r * 65536.0 + g * 256.0 + b


SKY = packed(100, 150, 230)
CHECKER_LIGHT = packed(150, 150, 150)
CHECKER_DARK = packed(70, 70, 70)
PLAYER_COLOR = packed(240, 240, 240)
OBSTACLE_COLOR = packed(220, 60, 60)

RAY_SETUP = """
; rd = normalise(u, v, 1); the camera sits at (0, 0, -3) looking along +z
        LDI.F  f1, {col}
        LD.D   d0, [f1]          ; col
        LDI.F  f2, {row}
        LD.D   d1, [f2]          ; row
        LDI.D  d2, {W}
        DIV.D  d0, d0, d2
        LDI.D  d2, {H}
        DIV.D  d1, d1, d2
        LDI.D  d3, 0.5
        SUB.D  d0, d0, d3        ; -0.5 .. 0.5
        SUB.D  d1, d3, d1        ; row 0 is the top of the screen
        LDI.D  d2, 0.9
        MUL.D  d0, d0, d2
        MUL.D  d1, d1, d2
        LDI.D  d4, 1
        MUL.D  d5, d0, d0
        MUL.D  d6, d1, d1
        ADD.D  d5, d5, d6
        ADD.D  d5, d5, d4
        SQRT.D d5, d5
        DIV.D  d6, d0, d5        ; rd x
        DIV.D  d7, d1, d5        ; rd y
        DIV.D  d8, d4, d5        ; rd z
; nearest hit so far: the sky, far away
        LDI.D  d9, 1000000       ; best t
        LDI.D  d10, {sky!r}      ; best colour, packed
"""

FLOOR_TEST = f"""
; the floor, y = -1
        LDI.D  d11, 0
        BLT.D  d7, d11, floor_yes
        BR     floor_after
floor_yes:
        LDI.D  d12, -1
        DIV.D  d12, d12, d7      ; t
        MUL.D  d13, d12, d6      ; hit x
        MUL.D  d0, d12, d8       ; t * rd z
        LDI.D  d11, -3
        ADD.D  d0, d0, d11       ; hit z
        CHOP.D d11, d13          ; floor of hit x
        CHOP.D d0, d0            ; floor of hit z
        ADD.D  d11, d11, d0
        LDI.D  d0, 2
        MOD.D  d11, d11, d0      ; the checker bit: 0 or 1
        MOV.D  d13, d12          ; keep t safe
        LDI.D  d12, 0
        BLE.D  d11, d12, checker_dark
        LDI.D  d12, {CHECKER_LIGHT!r}
        BR     checker_done
checker_dark:
        LDI.D  d12, {CHECKER_DARK!r}
checker_done:
        BLT.D  d13, d9, floor_near
        BR     floor_after
floor_near:
        MOV.D  d9, d13           ; best t = t
        MOV.D  d10, d12          ; best colour = the checker
floor_after:
"""


def ball_test(index, x_cell, z_cell, colour, radius):
    """The ray-sphere test for one ball whose centre is patched in memory.

    Centre (sx, -0.5, sz); camera (0, 0, -3); oc = ro - centre;
    t = -b - sqrt(disc), disc = b*b - cc, b = oc . rd, cc = oc . oc - r*r.
    """
    k = 0.25 - radius * radius
    return f"""
; ball {index}: centre (sx, -0.5, sz), patched at cells {x_cell} and {z_cell}
        LDI.F  f4, {x_cell}
        LD.D   d11, [f4]         ; sx
        LDI.F  f4, {z_cell}
        LD.D   d12, [f4]         ; sz
        LDI.D  d13, -3
        SUB.D  d13, d13, d12     ; -3 - sz
        MUL.D  d13, d13, d8      ; (-3 - sz) * rd z
        MUL.D  d0, d11, d6       ; sx * rd x
        SUB.D  d13, d13, d0
        LDI.D  d0, 0.5
        MUL.D  d0, d0, d7        ; 0.5 * rd y
        ADD.D  d13, d13, d0      ; b = oc . rd
        MUL.D  d0, d11, d11      ; sx * sx
        LDI.D  d1, 3
        ADD.D  d1, d1, d12       ; 3 + sz
        MUL.D  d1, d1, d1        ; (3 + sz) * (3 + sz)
        LDI.D  d2, {k!r}         ; 0.25 - r * r
        ADD.D  d1, d1, d2
        ADD.D  d1, d1, d0        ; cc
        MUL.D  d2, d13, d13      ; b * b
        SUB.D  d2, d2, d1        ; disc
        LDI.D  d0, 0
        BLT.D  d0, d2, disc{index}
        BR     done{index}
disc{index}:
        SQRT.D d2, d2
        LDI.D  d3, 0
        SUB.D  d3, d3, d13       ; -b
        SUB.D  d3, d3, d2        ; t = -b - sqrt(disc)
        LDI.D  d0, 0
        BLT.D  d0, d3, tpos{index}
        BR     done{index}
tpos{index}:
        BLE.D  d3, d9, nearer{index}
        BR     done{index}
nearer{index}:
        MOV.D  d9, d3            ; best t = t
        LDI.D  d10, {colour!r}   ; best colour = this ball
done{index}:
"""


def pixel_program(base):
    parts = [RAY_SETUP.format(col=base, row=base + 8, sky=SKY, W=W, H=H),
             FLOOR_TEST.format(checker_dark=CHECKER_DARK, checker_light=CHECKER_LIGHT)]
    data = base + 80
    parts.append(ball_test(0, data, data + 8, PLAYER_COLOR, PLAYER_R))          # the player
    for lane in range(3):
        parts.append(ball_test(lane + 1, data + 16 + 16 * lane, data + 24 + 16 * lane,
                               OBSTACLE_COLOR, OBSTACLE_R))
    parts.append(f"""
; store the colour
        LDI.F  f5, {data + 64}
        ST.D   [f5], d10
        HALT
""")
    return "\n".join(parts), base, data + 64  # the cell the store above writes


def lane_code(lane, base):
    """One lane of the referee: advance the obstacle, store it, test collision."""
    x_cell, z_cell = base + 112 + 16 * lane, base + 120 + 16 * lane
    return f"""
        LDI.D  d8, {SPEEDS[lane]!r}
        MUL.D  d8, d8, d0        ; frame * speed
        LDI.D  d9, {OFFSETS[lane]!r}
        ADD.D  d8, d8, d9
        MOD.D  d8, d8, d6        ; travel, 0 .. 2 * ZMAX
        SUB.D  d8, d5, d8        ; z = ZMAX - travel
        LDI.D  d9, {LANES[lane]!r}
        LDI.F  f2, {x_cell}
        ST.D   [f2], d9
        LDI.F  f2, {z_cell}
        ST.D   [f2], d8
        SUB.D  d10, d1, d9       ; player x - lane x
        ABS.D  d10, d10
        LDI.D  d11, 0.9
        LT.D   d10, d10, d11     ; the lanes are close in x
        LDI.D  d12, {PLAYER_Z!r}
        SUB.D  d12, d8, d12
        ABS.D  d12, d12
        LT.D   d12, d12, d11     ; and close in z
        MUL.D  d10, d10, d12     ; both: a hit
        ADD.D  d4, d4, d10
"""


REF_HEADER = """
; the referee: inputs patched in, obstacle positions and the score out
        LDI.F  f1, {frame}
        LD.D   d0, [f1]          ; frame
        LDI.F  f1, {px}
        LD.D   d1, [f1]          ; player x
        LDI.F  f1, {alive}
        LD.D   d2, [f1]          ; alive
        LDI.F  f1, {score}
        LD.D   d3, [f1]          ; score
        LDI.D  d4, 0             ; collided
        LDI.D  d5, {zmax!r}      ; ZMAX
        LDI.D  d6, {two}         ; 2 * ZMAX
        LDI.D  d7, 1
"""

REF_FOOTER = """
        LDI.D  d11, 1
        MIN.D  d4, d4, d11       ; collided is 0 or 1
        SUB.D  d12, d11, d4      ; 1 - collided
        MUL.D  d5, d2, d12       ; alive' = alive * (1 - collided)
        MUL.D  d6, d2, d0        ; alive * frame
        SUB.D  d8, d11, d2       ; 1 - alive
        MUL.D  d8, d8, d3        ; (1 - alive) * score
        ADD.D  d6, d6, d8        ; the score freezes when the run ends
        LDI.F  f2, {out_score}
        ST.D   [f2], d6
        LDI.F  f2, {out_alive}
        ST.D   [f2], d5
        HALT
"""


def referee_program(base):
    outs = {"out_score": base + 96, "out_alive": base + 104}
    parts = [REF_HEADER.format(zmax=ZMAX, two=2 * ZMAX, frame=base, px=base + 8,
                               alive=base + 16, score=base + 24)]
    for lane in range(3):
        parts.append(lane_code(lane, base))
    parts.append(REF_FOOTER.format(**outs))
    return "\n".join(parts), base, base + 96  # the score: the first of the eight outputs


def job(instances, inputs, outputs, image, memory, max_steps):
    """An array job file: header, patches, picks, image."""
    n_inputs = len(inputs[0]) if inputs else 0
    n_outputs = len(outputs)
    patches = b"".join(
        struct.pack("<IIQ", cell, FMT_DOUBLE, pack_double(value))
        for row in inputs for cell, value in row)
    picks = b"".join(struct.pack("<II", cell, FMT_DOUBLE)
                     for _ in range(instances) for cell, _ in outputs)
    header = b"M1AJ" + struct.pack("<IIIQIIIIIII", 1, instances, memory, max_steps,
                                   n_inputs, n_outputs, len(image.cells), 0, 0, memory, 0)
    assert len(header) == 52
    return header + patches + picks + bytes(struct.pack("<%dH" % len(image.cells), *image.cells))


def pack_double(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def build_referee_job(frame, player_x, alive, score, max_steps=200):
    _, base, out = referee_program(512)   # a guess, only to learn the program's length
    program = asm.assemble(referee_program(base)[0])
    base = (len(program.cells) + 7) // 8 * 8
    text, base, out = referee_program(base)
    program = asm.assemble(text)
    inputs = [[(base, float(frame)), (base + 8, player_x), (base + 16, float(alive)),
               (base + 24, float(score))]]
    outputs = [(out + k, 0) for k in range(0, 64, 8)]
    return job(1, inputs, outputs, program, base + 176, max_steps)


def build_pixel_job(player_x, obstacles, max_steps=1000):
    """obstacles: three (x, z) pairs, as the referee picked them."""
    _, col, out = pixel_program(1104)   # a guess, only to learn the program's length
    program = asm.assemble(pixel_program(col)[0])
    base = (len(program.cells) + 7) // 8 * 8
    image, col, out = pixel_program(base)   # now with the real data base
    program = asm.assemble(pixel_program(base)[0])
    inputs = []
    for row in range(H):
        for col in range(W):
            row_patches = [(base, float(col)), (base + 8, float(row)),
                           (base + 80, player_x), (base + 88, PLAYER_Z)]
            for lane, (ox, oz) in enumerate(obstacles):
                row_patches.append((base + 96 + 16 * lane, ox))
                row_patches.append((base + 104 + 16 * lane, oz))
            inputs.append(row_patches)
    outputs = [(out, 0)]
    memory = base + 176
    return job(W * H, inputs, outputs, program, memory, max_steps)


def read_result(path, instances, outputs):
    """(steps, status, values) per instance, from the M1AR file."""
    data = Path(path).read_bytes()
    if data[:4] != b"M1AR":
        sys.exit(f"{path}: not an array result file")
    rows, at = [], 16
    for _ in range(instances):
        steps = struct.unpack("<Q", data[at:at + 8])[0]
        at += 8
        status = struct.unpack("<III", data[at:at + 12])
        at += 12
        _pad = struct.unpack("<I", data[at:at + 4])
        at += 4
        values = struct.unpack(f"<{outputs}d", data[at:at + 8 * outputs])
        at += 8 * outputs
        rows.append((steps, status, values))
    return rows


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        def sized(make, seed):
            """A program's real size: assemble with a guess, then with the true base."""
            _, guess, _ = make(seed)
            text, base, out = make((len(asm.assemble(text if False else make(seed)[0]).cells) + 7) // 8 * 8)
            return asm.assemble(text), base, out
        program, pbase, pout = sized(pixel_program, 1104)
        print(f"pixel program: {len(program.cells)} cells, data at {pbase}")
        program, rbase, rout = sized(referee_program, 512)
        print(f"referee program: {len(program.cells)} cells, data at {rbase}")
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
