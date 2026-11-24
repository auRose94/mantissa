"""run_game.py -- the console of Mantissa Dodge.

    python3 run_game.py --frames 180 --record dodge.mp4     (scripted dodge)
    python3 run_game.py --keys                              (arrow keys: steer)
    python3 run_game.py --frames 3 --pngs frames            (just a few, as PNGs)

Each frame: patch the frame state into the referee (one machine), run it,
patch what it says into every pixel machine, run them, decode the colours.
The host carries the save file -- alive and score -- between frames, because
an array run always starts fresh.  The machines do everything else.

Speed: the machines render a frame in about 2 milliseconds of kernel time;
the console (Python, and a subprocess per array run) is the slow part, at
roughly 20-40 frames a second.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_game  # noqa: E402
from make_game import H, LANES, PLAYER_Z, W, read_result  # noqa: E402

HERE = Path(__file__).resolve().parent
GPU = HERE.parent / "build" / "mantissa-gpu"


def run_array(job_bytes, instances, outputs, tag):
    job_path = Path(f"/tmp/mantissa-game-{tag}.bin")
    result_path = Path(f"/tmp/mantissa-game-{tag}.result")
    job_path.write_bytes(job_bytes)
    done = subprocess.run([str(GPU), "array", str(job_path), "--out", str(result_path)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"array run failed: {done.stderr}")
    return read_result(result_path, instances, outputs)


def decode_colour(value):
    """(r, g, b) from the packed colour, by the host, which may use integers."""
    c = int(round(value)) % 16_777_216
    return (c // 65536, c // 256 % 256, c % 256)


def demo_steering(frame, player_x):
    """The scripted player: slide toward the lane whose obstacle is furthest away."""
    safe = LANES[max(range(3), key=lambda i: -abs(LANES[i] - player_x))]  # nearest lane by default
    danger = [abs(0) for _ in LANES]
    return safe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--record", metavar="MP4")
    parser.add_argument("--pngs", metavar="DIR")
    parser.add_argument("--keys", action="store_true", help="steer with the arrow keys")
    parser.add_argument("--width", type=int, default=W)
    parser.add_argument("--height", type=int, default=H)
    args = parser.parse_args()
    if args.width != W or args.height != H:
        sys.exit("the kernel program is written for 64x48; change make_game.py to move")

    player_x, alive, score = 0.0, 1, 0
    obstacles = [(lane, 30.0) for lane in LANES]
    png_dir = None
    if args.pngs:
        png_dir = Path(args.pngs)
        png_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    encoder = None
    if args.record:
        encoder = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             args.record], stdin=subprocess.PIPE)

    if args.keys:
        import termios
        import tty
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        keys = {}

        def press():
            import select
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            while ready:
                keys[sys.stdin.read(1)] = 1
                ready, _, _ = select.select([sys.stdin], [], [], 0)

    frames = []
    start = time.perf_counter()
    try:
        for frame in range(args.frames):
            if args.keys:
                press()
                if "q" in keys:
                    break
                if "\x1b[C" in keys:  # right
                    player_x = min(player_x + 0.12, 2.2)
                if "\x1b[D" in keys:  # left
                    player_x = max(player_x - 0.12, -2.2)
            else:
                # the scripted player: steer toward the lane whose obstacle is
                # furthest from the player's z; step 0.12 cells a frame
                danger = []
                for lane in range(3):
                    z = obstacles[lane][1] if frame else make_game.ZMAX
                    danger.append(abs(z - PLAYER_Z))
                target = LANES[danger.index(max(danger))]
                if player_x < target:
                    player_x = min(player_x + 0.12, target)
                elif player_x > target:
                    player_x = max(player_x - 0.12, target)

            referee = run_array(make_game.build_referee_job(frame, player_x, alive, score),
                                1, 8, "ref")
            _, status, values = referee[0]
            score, alive = int(round(values[0])), int(round(values[1]))
            obstacles = [(values[2 + 2 * k], values[3 + 2 * k]) for k in range(3)]

            pixels = run_array(make_game.build_pixel_job(player_x, obstacles),
                               W * H, 1, "pix")
            image = Image.new("RGB", (W, H))
            for k in range(W * H):
                colour = decode_colour(pixels[k][2][0])
                image.putpixel((k % W, k // W), colour)
            if encoder:
                encoder.stdin.write(image.tobytes())
            if png_dir and frame % max(1, args.frames // 6) == 0:
                image.resize((W * 4, H * 4), Image.NEAREST).save(png_dir / f"frame{frame:04d}.png")
            frames.append(image)

        seconds = time.perf_counter() - start
        if encoder:
            encoder.stdin.close()
            encoder.wait()
        state = "alive" if alive else "out at frame " + str(score)
        print(f"{args.frames} frames in {seconds:.2f} s ({args.frames / seconds:.1f} fps, "
              f"console-bound); score {score}, {state}")
    finally:
        if args.keys:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
