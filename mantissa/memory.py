"""Memory: a row of 16-bit cells, and nothing smaller."""
from __future__ import annotations

import math
from array import array

from . import fp
from .fault import Fault


def _show(addr):
    if 0 <= addr < 1e15 and addr == int(addr):
        return str(int(addr))
    return repr(float(addr))


class Memory:
    """`size` cells of 16 bits.  A half is one cell, a float two, a double four,
    most significant cell first.  Data must be naturally aligned; the operands
    that trail an instruction sit on four-cell boundaries anyway."""

    def __init__(self, size):
        self.size = size
        self.cells = array("H", bytes(2 * size))

    def _index(self, width, addr, aligned):
        if isinstance(addr, fp.Quad):
            if addr.inf:
                addr = math.inf
            else:
                exact = fp.q_fraction(addr)
                # a quad address names a cell exactly when its value is a whole
                # number: a double address stops at the float wall, a quad
                # address does not have to
                addr = exact if exact.denominator == 1 else float(exact)
        elif isinstance(addr, fp.Qnan):
            addr = math.nan
        else:
            addr = float(addr)  # the machine always hands over floats; tools may pass ints
        if addr != addr:
            reason = "NaN address"
        elif addr < 0:
            reason = "negative address"
        elif addr + width > self.size:
            reason = "past the end of memory"
        elif not addr.is_integer():
            reason = "not a whole number of cells"
        elif aligned and addr % width:
            reason = f"misaligned, needs a multiple of {width}"
        else:
            return int(addr)
        raise Fault("address", f"{_show(addr)}: {reason}")

    @staticmethod
    def _bits(cells, start, n):
        bits = 0
        for k in range(n):
            bits = bits << 16 | cells[start + k]
        return bits

    def check_fetch(self, addr):
        """Raise an address fault unless an instruction could be fetched at addr."""
        self._index(4, addr, True)

    def fetch(self, addr):
        """The double at addr, which must be a whole cell on a four-cell boundary:
        the instruction word."""
        return fp.from_bits(fp.D, self._bits(self.cells, self._index(4, addr, True), 4))

    def load(self, fmt, addr):
        """The value of format `fmt` stored at the aligned address addr."""
        return fp.from_bits(fmt, self._bits(self.cells, self._index(fmt.cells, addr, True), fmt.cells))

    def store(self, fmt, addr, value):
        """Place a value of format `fmt` at the aligned address addr."""
        start = self._index(fmt.cells, addr, True)
        bits = fp.to_bits(fmt, value)
        for k in range(fmt.cells):
            self.cells[start + k] = bits >> 16 * (fmt.cells - 1 - k) & 0xFFFF

    def operand(self, fmt, addr):
        """Load an inline operand of the instruction stream."""
        return fp.from_bits(fmt, self._bits(self.cells, self._index(fmt.cells, addr, False), fmt.cells))

    def write(self, cells, org=0):
        """Copy an image into memory at address org."""
        if org < 0 or org + len(cells) > self.size:
            raise ValueError(f"an image of {len(cells)} cells at {org} does not fit in {self.size}")
        self.cells[org:org + len(cells)] = array("H", cells)
