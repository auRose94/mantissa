// Verilator testbench for the Mantissa-1 core.
//
//   m1_sim <trace-file> [--out <file>] [--max-steps N] [--quiet] [--no-compare]
//
// Reads a trace file: the header says how many cells the machine has, where
// the image goes, the entry, what f31 starts as, and the image itself.  Loads
// the image, runs the core until HALT, a fault or the step limit, and writes
// the run's own trace in the same format: S, then R, M, P, C or X lines for
// what the step changed, then H, F <kind> or L.  With an expected run in the
// file, the two are compared line for line.
//
// The step limit is not in the header.  --max-steps sets it; otherwise a file
// that ends with L is replayed to as many steps as it has, and any other file
// to 1,000,000, the reference's default.
//
// Which registers a step changed is decided here, not in the core, the way
// the reference decides it in trace.py rather than in machine.py: a register
// has changed when the value in it is not the same value.  For every value
// but a NaN that is "other bits".  The reference is Python, and for a NaN it
// is "another object": the NaN that arithmetic produces is one shared object,
// a NaN loaded from memory is a new object every time, and MOV and SPLIT hand
// the object on.  So the core says where each NaN it writes came from, and
// this keeps a tag per register to tell the objects apart.
#include "Vmantissa_top.h"
#include "verilated.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int EV_RETIRE = 0, EV_HALT = 1, EV_FAULT = 2;
constexpr int FAULT_ADDRESS = 0, FAULT_ILLEGAL = 1, FAULT_NOT_A_CHARACTER = 2;
constexpr int PROV_CANON = 0, PROV_INHERIT = 1, PROV_FRESH = 2;
constexpr int NREGS = 96;
constexpr int R_SP = 63;       // f31
constexpr int NTAGS = 97;   // tags 1..97 name loaded NaNs; tag 0 is the shared one, or not a NaN

struct Header {
  long long cells = 0, org = 0, entry = 0, sp = 0, image = 0;
  std::vector<uint16_t> image_cells;
};

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

bool read_lines(const std::string& path, std::vector<std::string>& lines) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return false;
  char buf[8192];
  while (std::fgets(buf, sizeof buf, f)) {
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    lines.push_back(s);
  }
  std::fclose(f);
  return true;
}

bool parse_trace(const std::string& path, Header& h, std::vector<std::string>& expected) {
  std::vector<std::string> lines;
  if (!read_lines(path, lines)) {
    std::fprintf(stderr, "m1_sim: cannot read %s\n", path.c_str());
    return false;
  }
  size_t at = 0;
  auto header_value = [&](const char* key, long long& into) {
    if (at >= lines.size()) return false;
    auto f = fields(lines[at]);
    if (f.size() != 2 || f[0] != key) return false;
    into = std::strtoll(f[1].c_str(), nullptr, 10);
    at++;
    return true;
  };
  if (at >= lines.size() || lines[at] != "# mantissa-1 trace") {
    std::fprintf(stderr, "m1_sim: %s is not a mantissa-1 trace file\n", path.c_str());
    return false;
  }
  at++;
  if (!header_value("cells", h.cells) || !header_value("org", h.org) || !header_value("entry", h.entry)
      || !header_value("sp", h.sp) || !header_value("image", h.image)) {
    std::fprintf(stderr, "m1_sim: %s: cannot read the header\n", path.c_str());
    return false;
  }
  long long left = h.image;
  while (left > 0) {
    if (at >= lines.size()) {
      std::fprintf(stderr, "m1_sim: %s: the image is short\n", path.c_str());
      return false;
    }
    for (const std::string& f : fields(lines[at])) h.image_cells.push_back(
        static_cast<uint16_t>(std::strtoul(f.c_str(), nullptr, 16)));
    left -= 16;
    at++;
  }
  if (static_cast<long long>(h.image_cells.size()) != h.image) {
    std::fprintf(stderr, "m1_sim: %s: the image is not %lld cells\n", path.c_str(), h.image);
    return false;
  }
  if (at >= lines.size() || lines[at] != "run") {
    std::fprintf(stderr, "m1_sim: %s: no run section\n", path.c_str());
    return false;
  }
  at++;
  expected.assign(lines.begin() + static_cast<long>(at), lines.end());
  return true;
}

double bits_to_dbl(uint64_t bits) {
  double d;
  std::memcpy(&d, &bits, sizeof d);
  return d;
}

uint64_t dbl_to_bits(double d) {
  uint64_t bits;
  std::memcpy(&bits, &d, sizeof bits);
  return bits;
}

// A value in a trace is written the way the reference writes it,
// struct.pack("<d", value).hex(): the double's eight bytes, least
// significant first.
std::string hex16(uint64_t bits) {
  char buf[32];
  for (int i = 0; i < 8; i++)
    std::snprintf(buf + 2 * i, 3, "%02x", static_cast<unsigned>((bits >> (8 * i)) & 0xff));
  return buf;
}

bool is_nan_bits(uint64_t bits) {
  return ((bits >> 52) & 0x7ff) == 0x7ff && (bits & ((1ull << 52) - 1)) != 0;
}

std::string reg_name(int code) {
  static const char kFiles[3] = {'h', 'f', 'd'};
  return std::string(1, kFiles[code >> 5]) + std::to_string(code & 31);
}

struct RegState {
  uint64_t bits;
  int tag;
};

// trace._same: the same object, or equal with the same sign of zero
bool same(const RegState& a, const RegState& b) {
  if (is_nan_bits(a.bits) || is_nan_bits(b.bits))
    return is_nan_bits(a.bits) && is_nan_bits(b.bits) && a.tag == b.tag;
  return a.bits == b.bits;
}

const char* fault_name(int kind) {
  switch (kind) {
    case FAULT_ADDRESS: return "address";
    case FAULT_ILLEGAL: return "illegal-instruction";
    default: return "not-a-character";
  }
}

void usage() {
  std::fprintf(stderr,
               "usage: m1_sim <trace-file> [--out <file>] [--max-steps N] [--quiet] [--no-compare]\n"
               "       m1_sim --batch <list-file> [--quiet]   one trace path per line\n");
}

}  // namespace

struct Options {
  std::string out_path;
  long long max_steps = -1;
  bool quiet = false;
  bool no_compare = false;
};

// Load one trace file's image, run it, compare.  Returns 0 on a match.
int run_one(const std::string& in_path, const Options& opt) {
  Header h;
  std::vector<std::string> expected;
  if (!parse_trace(in_path, h, expected)) return 2;

  long long limit;
  if (opt.max_steps >= 0) {
    limit = opt.max_steps;
  } else if (!expected.empty() && expected.back() == "L") {
    limit = static_cast<long long>(std::count_if(expected.begin(), expected.end(),
                                                 [](const std::string& s) { return !s.compare(0, 2, "S ", 2); }));
  } else {
    limit = 1000000;
  }

  Vmantissa_top* dut = new Vmantissa_top;
  long long cycles = 0;
  auto tick = [&]() {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
    cycles++;
  };

  dut->rst_n = 0;
  dut->cfg_we = 0;
  dut->cfg_addr = 0;
  dut->cfg_data = 0;
  dut->boot_pc = dbl_to_bits(static_cast<double>(h.entry));
  dut->boot_sp = dbl_to_bits(static_cast<double>(h.sp));
  dut->max_steps = limit > 0 && limit <= 0xffffffffll ? static_cast<uint32_t>(limit) : 0u;
  dut->eval();

  // the build has 2**AW cells of memory at most; the machine it is asked to be has `cells`
  const long long capacity = static_cast<long long>(std::llround(bits_to_dbl(dut->capacity)));
  if (h.cells <= 0 || capacity < h.cells || h.org + h.image > h.cells) {
    std::fprintf(stderr, "m1_sim: %s needs %lld cells and this build has %lld\n", in_path.c_str(),
                 std::max(h.cells, h.org + h.image), capacity);
    delete dut;
    return 2;
  }
  dut->cells = static_cast<uint32_t>(h.cells);

  tick();
  tick();
  for (long long i = 0; i < h.image; i++) {
    dut->cfg_we = 1;
    dut->cfg_addr = static_cast<uint32_t>(h.org + i);
    dut->cfg_data = h.image_cells[static_cast<size_t>(i)];
    tick();
  }
  dut->cfg_we = 0;
  dut->cfg_addr = 0;
  dut->cfg_data = 0;
  dut->rst_n = 1;

  // the trace, and what the machine looks like to the trace
  std::vector<std::string> out;
  RegState shadow[NREGS];
  for (int i = 0; i < NREGS; i++) shadow[i] = {0, 0};
  shadow[R_SP] = {dbl_to_bits(static_cast<double>(h.sp)), 0};   // f31 starts at the stack pointer

  long long steps = 0;
  const auto t0 = std::chrono::steady_clock::now();
  const long long cycle_bound = 4096ll * (limit > 0 ? limit : 1000000) + 100000ll;
  while (cycles < cycle_bound) {
    tick();
    // the end of the run is announced with the last step's event, so that comes first
    if (dut->ev_valid) {
      const std::string where = std::to_string(steps) + " " + std::to_string(
          static_cast<long long>(std::llround(bits_to_dbl(dut->ev_pc))));
      out.push_back("S " + where);
      bool changed = false;

      // register writes: the new tags come from the state before the step
      const int nwr = dut->ev_nwr;
      int code[4] = {dut->ev_wcode0, dut->ev_wcode1, dut->ev_wcode2, dut->ev_wcode3};
      uint64_t val[4] = {dut->ev_wval0, dut->ev_wval1, dut->ev_wval2, dut->ev_wval3};
      int prov[4] = {dut->ev_wprov0, dut->ev_wprov1, dut->ev_wprov2, dut->ev_wprov3};
      int src[4] = {dut->ev_wsrc0, dut->ev_wsrc1, dut->ev_wsrc2, dut->ev_wsrc3};
      int tag[2] = {0, 0};
      for (int i = 0; i < nwr && i < 4; i++) {
        if (code[i] >= NREGS) { std::fprintf(stderr, "m1_sim: bad register code %d\n", code[i]); return 1; }
        if (!is_nan_bits(val[i])) continue;
        if (prov[i] == PROV_INHERIT) {
          tag[i] = shadow[src[i]].tag;
        } else if (prov[i] == PROV_FRESH) {
          // a new object: a tag no register holds, the one being written included,
          // since the new NaN is never the NaN that was there (96 registers, 97 tags)
          bool used[NTAGS + 1] = {};
          for (int r = 0; r < NREGS; r++) used[shadow[r].tag] = true;
          tag[i] = 0;
          for (int t = 1; t <= NTAGS; t++)
            if (!used[t]) { tag[i] = t; break; }
        }
      }
      RegState before[NREGS];
      bool touched[NREGS] = {};
      for (int i = 0; i < nwr && i < 4; i++) {
        if (!touched[code[i]]) {
          touched[code[i]] = true;
          before[code[i]] = shadow[code[i]];
        }
        shadow[code[i]] = {val[i], tag[i]};
      }
      // file by file, by register number within a file: the register code's order
      for (int r = 0; r < NREGS; r++) {
        if (!touched[r]) continue;
        if (!same(before[r], shadow[r])) {
          out.push_back("R " + where + " " + reg_name(r) + " " + hex16(shadow[r].bits));
          changed = true;
        }
      }

      if (dut->ev_store) {
        out.push_back("M " + where + " " + std::to_string(dut->ev_saddr) + " "
                      + std::to_string(1 << dut->ev_sfmt) + " " + hex16(dut->ev_sval));
        changed = true;
      }
      if (dut->ev_put) {
        out.push_back("P " + where + " " + hex16(dut->ev_putval));
        changed = true;
      }
      if (dut->ev_putc) {
        out.push_back("C " + where + " " + std::to_string(dut->ev_putcode));
        changed = true;
      }
      if (!changed) out.push_back("X " + where);

      if (dut->ev_end == EV_HALT) out.push_back("H");
      else if (dut->ev_end == EV_FAULT) out.push_back(std::string("F ") + fault_name(dut->ev_fault));
      steps++;
    }
    if (dut->halted || dut->fault || dut->step_limit) break;
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(t1 - t0).count();

  std::string ending = out.empty() ? "" : out.back();
  if (dut->step_limit && ending != "L") { out.push_back("L"); ending = "L"; }
  const bool hung = !(dut->halted || dut->fault || dut->step_limit);

  // the whole trace file, if wanted
  if (!opt.out_path.empty()) {
    FILE* f = std::fopen(opt.out_path.c_str(), "w");
    if (f) {
      std::fprintf(f, "# mantissa-1 trace\ncells %lld\norg %lld\nentry %lld\nsp %lld\nimage %lld\n", h.cells,
                   h.org, h.entry, h.sp, h.image);
      for (long long at = 0; at < h.image; at += 16) {
        for (long long k = 0; k < 16 && at + k < h.image; k++)
          std::fprintf(f, "%s%04x", k ? " " : "", h.image_cells[static_cast<size_t>(at + k)]);
        std::fputc('\n', f);
      }
      std::fputs("run\n", f);
      for (const std::string& line : out) std::fprintf(f, "%s\n", line.c_str());
      std::fclose(f);
    }
  }

  // the comparison, line for line
  bool pass = !hung;
  size_t shown = 0;
  if (!opt.no_compare && !expected.empty()) {
    const size_t n = std::max(out.size(), expected.size());
    for (size_t i = 0; i < n; i++) {
      const std::string got = i < out.size() ? out[i] : "<none>";
      const std::string want = i < expected.size() ? expected[i] : "<none>";
      if (got != want) {
        pass = false;
        if (shown < 8) {
          std::printf("  run line %zu: expected '%s', got '%s'\n", i + 1, want.c_str(), got.c_str());
          shown++;
        }
      }
    }
  }

  const double cpi = steps ? static_cast<double>(cycles) / static_cast<double>(steps) : 0.0;
  const char* name = std::strrchr(in_path.c_str(), '/');
  name = name ? name + 1 : in_path.c_str();
  if (hung) std::printf("FAIL %s: the core stopped making progress after %lld steps\n", name, steps);
  else if (opt.quiet) std::printf("%s %s: %lld steps, ends '%s'\n", pass ? "PASS" : "FAIL", name, steps, ending.c_str());
  else std::printf("%s %s: %lld steps, ends '%s', %lld cycles (%.2f cycles per instruction), %.3f s (%.0f thousand instructions per second)\n",
                   pass ? "PASS" : "FAIL", name, steps, ending.c_str(), cycles, cpi, seconds,
                   seconds > 0 ? steps / seconds / 1000.0 : 0.0);

  dut->final();
  delete dut;
  return pass ? 0 : 1;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  std::string in_path, batch_path;
  Options opt;
  for (int i = 1; i < argc; i++) {
    const std::string arg(argv[i]);
    if (arg == "--out" && i + 1 < argc) opt.out_path = argv[++i];
    else if (arg == "--max-steps" && i + 1 < argc) opt.max_steps = std::atoll(argv[++i]);
    else if (arg == "--quiet") opt.quiet = true;
    else if (arg == "--no-compare") opt.no_compare = true;
    else if (arg == "--batch" && i + 1 < argc) batch_path = argv[++i];
    else if (!arg.empty() && arg[0] != '-' && in_path.empty()) in_path = arg;
    else { usage(); return 2; }
  }
  if (!batch_path.empty()) {
    // many traces in one process: the fuzzers feed it thousands of small programs
    std::vector<std::string> lines;
    if (!read_lines(batch_path, lines)) { std::fprintf(stderr, "m1_sim: cannot read %s\n", batch_path.c_str()); return 2; }
    long passed = 0, failed = 0;
    Options q = opt;
    q.quiet = true;
    q.out_path.clear();
    for (const std::string& line : lines) {
      if (line.empty() || line[0] == '#') continue;
      if (run_one(line, q) == 0) passed++;
      else failed++;
    }
    std::printf("batch: %ld passed, %ld failed\n", passed, failed);
    return failed ? 1 : 0;
  }
  if (in_path.empty()) { usage(); return 2; }
  return run_one(in_path, opt);
}
