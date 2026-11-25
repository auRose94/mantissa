// Verilator testbench for the floating point unit on its own.
//
//   tb_fpu <vectors> [--max N] [--verbose]
//
// <vectors> is what tools/gen_fpu_vectors.py writes: one request per line
// with the answer mantissa/fp.py gives.  Each request goes in with `start`,
// the bench waits for `done` (at most 100000 cycles), and y, y2 and the flags
// are compared with the expected ones.  Exit status is 0 when every request
// matches.
#include "Vmantissa_fpu.h"
#include "verilated.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

uint64_t unhex(const std::string& s) { return std::strtoull(s.c_str(), nullptr, 16); }

// a value field: sixteen hex digits for a narrow format, thirty-two for a
// quad (most significant word first) -- set a 128-bit port from either
void set128(uint32_t* port, const std::string& h) {
  std::string hi_part, lo_part;
  if (h.size() > 16) {
    hi_part = h.substr(0, h.size() - 16);
    lo_part = h.substr(h.size() - 16);
  } else {
    hi_part = "0000000000000000";
    lo_part = h;
  }
  const uint64_t hi = unhex(hi_part), lo = unhex(lo_part);
  port[0] = static_cast<uint32_t>(lo);
  port[1] = static_cast<uint32_t>(lo >> 32);
  port[2] = static_cast<uint32_t>(hi);
  port[3] = static_cast<uint32_t>(hi >> 32);
}

std::string get128(const uint32_t* port) {
  const uint64_t lo = (uint64_t)port[0] | ((uint64_t)port[1] << 32);
  const uint64_t hi = (uint64_t)port[2] | ((uint64_t)port[3] << 32);
  char buf[33];
  std::snprintf(buf, sizeof buf, "%016llx%016llx", (unsigned long long)hi, (unsigned long long)lo);
  return buf;
}

std::vector<std::string> fields(const std::string& line) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() && line[i] == ' ') i++;
    size_t j = i;
    while (j < line.size() && line[j] != ' ') j++;
    if (j > i) out.push_back(line.substr(i, j - i));
    i = j;
  }
  return out;
}

const char* OP_NAMES[] = {"ADD", "SUB", "MUL", "DIV", "SQRT", "FMA", "REM", "MOD", "MIN", "MAX",
                          "NEG", "ABS", "CPSGN", "FLOOR", "CEIL", "CHOP", "ROUND", "FLOOR_TO", "CEIL_TO",
                          "CHOP_TO", "ROUND_TO", "SCALB", "LOGB", "NEXT", "CLASS", "LT", "LE", "EQ", "CMP",
                          "CVT", "FROMBITS", "TOBITS", "SPLIT", "JOIN", "MANT"};

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  std::string path;
  long max = -1;
  bool verbose = false;
  for (int i = 1; i < argc; i++) {
    const std::string arg(argv[i]);
    if (arg == "--max" && i + 1 < argc) max = std::atol(argv[++i]);
    else if (arg == "--verbose") verbose = true;
    else if (path.empty()) path = arg;
  }
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f) { std::fprintf(stderr, "tb_fpu: cannot read %s\n", path.c_str()); return 2; }

  Vmantissa_fpu* dut = new Vmantissa_fpu;
  dut->clk = 0;
  dut->rst_n = 0;
  dut->start = 0;
  dut->op = 0; dut->fmt = 0; dut->fmt2 = 0; dut->sel = 0;
  dut->a[0] = 0; dut->a[1] = 0; dut->a[2] = 0; dut->a[3] = 0;
  dut->b[0] = 0; dut->b[1] = 0; dut->b[2] = 0; dut->b[3] = 0;
  dut->c[0] = 0; dut->c[1] = 0; dut->c[2] = 0; dut->c[3] = 0;
  auto tick = [&]() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); };
  tick(); tick();
  dut->rst_n = 1;

  char buf[512];
  long line_no = 0, tested = 0, failed = 0, cycles = 0;
  long failed_by_op[35] = {};
  const auto t0 = std::chrono::steady_clock::now();
  while (std::fgets(buf, sizeof buf, f)) {
    line_no++;
    std::vector<std::string> v = fields(buf);
    if (v.size() != 10) continue;
    if (max >= 0 && tested >= max) break;
    const int op = std::atoi(v[0].c_str());
    dut->op = op;
    dut->fmt = std::strtoul(v[1].c_str(), nullptr, 10);
    dut->fmt2 = std::strtoul(v[2].c_str(), nullptr, 10);
    dut->sel = std::strtoul(v[3].c_str(), nullptr, 10);
    set128(&dut->a[0], v[4]);
    set128(&dut->b[0], v[5]);
    set128(&dut->c[0], v[6]);
    uint32_t want_y_port[4], want_y2_port[4];
    set128(want_y_port, v[7]);
    set128(want_y2_port, v[8]);
    const std::string want_y = get128(want_y_port), want_y2 = get128(want_y2_port);
    const unsigned want_flags = static_cast<unsigned>(std::strtoul(v[9].c_str(), nullptr, 16));
    dut->start = 1;
    tick(); cycles++;
    dut->start = 0;
    long wait = 0;
    while (!dut->done && wait < 100000) { tick(); cycles++; wait++; }
    tested++;
    const std::string got_y = get128(&dut->y[0]), got_y2 = get128(&dut->y2[0]);
    const bool ok = dut->done && got_y == want_y && got_y2 == want_y2
                    && (dut->flags & 0x1f) == want_flags;
    if (!ok) {
      failed++;
      if (op >= 0 && op < 35) failed_by_op[op]++;
      if (failed <= 40 || verbose)
        std::printf("line %ld: %-9s fmt %s sel %s  a=%s b=%s c=%s  y %s y2 %s flags %02x, expected y %s y2 %s flags %02x%s\n",
                    line_no, (op >= 0 && op < 35) ? OP_NAMES[op] : "?", v[1].c_str(), v[3].c_str(),
                    v[4].c_str(), v[5].c_str(), v[6].c_str(), got_y.c_str(),
                    got_y2.c_str(), dut->flags & 0x1f, want_y.c_str(),
                    want_y2.c_str(), want_flags, dut->done ? "" : "  (no done)");
    }
  }
  std::fclose(f);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s: %ld requests, %ld failed, %ld cycles, %.3f s\n", failed ? "FAIL" : "PASS", tested,
              failed, cycles, seconds);
  if (failed)
    for (int op = 0; op < 35; op++)
      if (failed_by_op[op]) std::printf("  %-9s %ld failed\n", OP_NAMES[op], failed_by_op[op]);
  dut->final();
  delete dut;
  return failed ? 1 : 0;
}
