// main.cpp -- the host program of the GPU port of the Mantissa-1.
//
//   mantissa-gpu cosim TRACE.trace [--out OUT.trace] [--steps N] [--panel]
//   mantissa-gpu array JOB.bin [--out RESULT.bin]
//   mantissa-gpu info
//
// cosim runs one machine on the GPU from the image in a trace file and writes
// the trace of what it did in the reference's format.  Given a reference run,
// it says whether the two agree line for line.  array runs many machines at
// once from a job file; gpu/python writes those.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "align.h"
#include "fp.h"
#include "kernels.h"
#include "machine.h"

void* operator new(std::size_t n) { return mantissa_aligned_new(n); }

namespace {

#define CK(call)                                                                                                          \
    do {                                                                                                                  \
        hipError_t error_ = (call);                                                                                       \
        if (error_ != hipSuccess) {                                                                                       \
            std::fprintf(stderr, "mantissa-gpu: %s: %s\n", #call, hipGetErrorString(error_));                             \
            return 2;                                                                                                     \
        }                                                                                                                 \
    } while (0)

int usage() {
    std::fprintf(stderr, "usage: mantissa-gpu cosim TRACE.trace [--out OUT.trace] [--steps N] [--panel]\n"
                         "       mantissa-gpu array JOB.bin [--out RESULT.bin]\n"
                         "       mantissa-gpu info\n");
    return 1;
}

// --- files -----------------------------------------------------------------------------

struct Image {
    uint32_t cells = 0, org = 0, entry = 0, sp = 0;
    std::vector<uint16_t> data;
};

bool split_lines(const std::string& path, std::vector<std::string>& lines) {
    FILE* handle = std::fopen(path.c_str(), "rb");
    if (!handle) {
        std::perror(path.c_str());
        return false;
    }
    std::string text;
    char buffer[1 << 16];
    size_t got;
    while ((got = std::fread(buffer, 1, sizeof buffer, handle)) > 0) text.append(buffer, got);
    std::fclose(handle);
    size_t at = 0;
    while (at <= text.size()) {
        size_t end = text.find('\n', at);
        if (end == std::string::npos) {
            if (at < text.size()) lines.push_back(text.substr(at));
            break;
        }
        lines.push_back(text.substr(at, end - at));
        at = end + 1;
    }
    if (!lines.empty() && lines.back().empty()) lines.pop_back();
    return true;
}

bool parse_u32(const std::string& line, const char* key, uint32_t& out) {
    if (line.compare(0, std::strlen(key), key) != 0) return false;
    char* end;
    unsigned long value = std::strtoul(line.c_str() + std::strlen(key), &end, 10);
    if (end == line.c_str() + std::strlen(key)) return false;
    out = (uint32_t)value;
    return true;
}

// The header and image of a trace file, then its run section if it has one.
bool read_trace(const std::string& path, Image& image, std::string& header, std::vector<std::string>& run,
                bool& has_run) {
    std::vector<std::string> lines;
    if (!split_lines(path, lines)) return false;
    size_t at = 0;
    auto next = [&](const char* key, uint32_t& out) -> bool {
        while (at < lines.size() && !lines[at].empty() && lines[at][0] == '#') at++;
        return at < lines.size() && parse_u32(lines[at++], key, out);
    };
    if (!next("cells ", image.cells) || !next("org ", image.org) || !next("entry ", image.entry) ||
        !next("sp ", image.sp)) {
        std::fprintf(stderr, "%s: not a mantissa-1 trace: no cells, org, entry and sp lines\n", path.c_str());
        return false;
    }
    uint32_t count = 0;
    if (!next("image ", count)) {
        std::fprintf(stderr, "%s: no image line\n", path.c_str());
        return false;
    }
    header.clear();
    for (size_t i = 0; i < at; i++) header += lines[i] + "\n";
    image.data.assign(count, 0);
    size_t placed = 0;
    for (uint32_t row = 0; row * 16 < count && at < lines.size(); row++) {
        header += lines[at] + "\n";
        const std::string& line = lines[at++];
        size_t pos = 0;
        while (pos < line.size() && placed < count) {
            while (pos < line.size() && line[pos] == ' ') pos++;
            char* end;
            unsigned long value = std::strtoul(line.c_str() + pos, &end, 16);
            if (end == line.c_str() + pos) break;
            image.data[placed++] = (uint16_t)value;
            pos = (size_t)(end - line.c_str());
        }
    }
    if (placed != count) {
        std::fprintf(stderr, "%s: the image says %u cells but %zu were found\n", path.c_str(), count, placed);
        return false;
    }
    has_run = at < lines.size() && lines[at] == "run";
    run.clear();
    if (has_run) run.assign(lines.begin() + (long)at + 1, lines.end());
    return true;
}

bool read_bytes(const std::string& path, std::vector<uint8_t>& data) {
    FILE* handle = std::fopen(path.c_str(), "rb");
    if (!handle) {
        std::perror(path.c_str());
        return false;
    }
    uint8_t buffer[1 << 16];
    size_t got;
    while ((got = std::fread(buffer, 1, sizeof buffer, handle)) > 0) data.insert(data.end(), buffer, buffer + got);
    std::fclose(handle);
    return true;
}

bool write_bytes(const std::string& path, const void* data, size_t size) {
    FILE* handle = std::fopen(path.c_str(), "wb");
    if (!handle) {
        std::perror(path.c_str());
        return false;
    }
    bool ok = std::fwrite(data, 1, size, handle) == size;
    if (std::fclose(handle) != 0) ok = false;
    return ok;
}

std::string hex64(uint64_t value) {
    // The reference writes struct.pack("<d", value).hex(): the double's eight
    // bytes in little-endian order, so the value itself reversed byte by byte.
    char out[17];
    for (int k = 0; k < 8; k++)
        std::snprintf(out + 2 * k, 3, "%02x", (unsigned)(value >> (8 * k)) & 0xFF);
    return out;
}

// --- cosim ------------------------------------------------------------------------------

int cosim(int argc, char** argv) {
    if (argc < 3) return usage();
    std::string out_path, in_path = argv[2];
    long long steps_arg = -1;
    bool panel = false;
    for (int i = 3; i < argc; i++) {
        if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) steps_arg = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--panel")) panel = true;
        else return usage();
    }
    Image image;
    std::string header;
    std::vector<std::string> reference;
    bool has_run;
    if (!read_trace(in_path, image, header, reference, has_run)) return 2;

    uint64_t steps = (uint64_t)std::max(0LL, steps_arg);
    if (steps_arg < 0) {
        if (!has_run) {
            std::fprintf(stderr, "%s has no run to take a step limit from; give --steps\n", in_path.c_str());
            return 2;
        }
        for (const std::string& line : reference) {
            if (line.compare(0, 2, "S ") == 0) steps++;
        }
    }
    uint64_t capacity = 3 * steps + 8;  // a step writes at most S and two registers
    const size_t event_bytes = (size_t)capacity * sizeof(m1::Event);
    const size_t max_bytes = (size_t)6 << 30;
    if (event_bytes > max_bytes) {
        std::fprintf(stderr, "%llu steps would need a %zu byte trace buffer; the limit is %zu\n",
                     (unsigned long long)steps, event_bytes, max_bytes);
        return 2;
    }

    std::vector<uint8_t> memory((size_t)image.cells * 2, 0);
    if ((size_t)image.org + image.data.size() > image.cells) {
        std::fprintf(stderr, "the image does not fit: %zu cells at %u in %u\n", image.data.size(), image.org,
                     image.cells);
        return 2;
    }
    std::memcpy(memory.data() + (size_t)image.org * 2, image.data.data(), image.data.size() * 2);

    uint8_t* d_memory; CK(hipMalloc(&d_memory, memory.size()));
    CK(hipMemcpy(d_memory, memory.data(), memory.size(), hipMemcpyHostToDevice));
    m1::Event* d_events; CK(hipMalloc(&d_events, event_bytes));
    uint32_t* d_result; CK(hipMalloc(&d_result, 8 * 4));
    CK(hipMemset(d_result, 0, 8 * 4));

    auto wall0 = std::chrono::steady_clock::now();
    hipEvent_t began, ended;
    CK(hipEventCreate(&began));
    CK(hipEventCreate(&ended));
    CK(hipEventRecord(began));
    hipLaunchKernelGGL(m1::cosim_kernel, dim3(1), dim3(32), m1::COSIM_LDS, 0, image.entry, image.cells, image.sp, steps,
                       d_memory, d_events, (uint32_t)capacity, d_result);
    CK(hipGetLastError());
    CK(hipEventRecord(ended));
    uint32_t result[8];
    CK(hipMemcpy(result, d_result, sizeof result, hipMemcpyDeviceToHost));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, began, ended));
    std::vector<m1::Event> events(result[0]);
    if (result[0]) CK(hipMemcpy(events.data(), d_events, (size_t)result[0] * sizeof(m1::Event), hipMemcpyDeviceToHost));
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    CK(hipFree(d_memory));
    CK(hipFree(d_events));
    CK(hipFree(d_result));
    if (result[1]) {
        std::fprintf(stderr, "the trace buffer overflowed at %llu events\n", (unsigned long long)capacity);
        return 2;
    }

    // The events become the text of the reference's format.
    std::string text;
    text.reserve(events.size() * 40 + header.size() + 16);
    text += header;
    text += "run\n";
    std::vector<std::string> mine;
    mine.reserve(events.size());
    char line[96];
    for (const m1::Event& e : events) {
        switch (e.kind) {
            case m1::E_STEP: std::snprintf(line, sizeof line, "S %u %u", e.step, e.pc); break;
            case m1::E_REG:
                std::snprintf(line, sizeof line, "R %u %u %c%u %s", e.step, e.pc, "hfd"[e.c >> 5], e.c & 31,
                              hex64(e.d).c_str());
                break;
            case m1::E_STORE:
                std::snprintf(line, sizeof line, "M %u %u %u %u %s", e.step, e.pc, e.c >> 8, e.c & 255,
                              hex64(e.d).c_str());
                break;
            case m1::E_PUT: std::snprintf(line, sizeof line, "P %u %u %s", e.step, e.pc, hex64(e.d).c_str()); break;
            case m1::E_PUTC: std::snprintf(line, sizeof line, "C %u %u %u", e.step, e.pc, e.c); break;
            case m1::E_NOTHING: std::snprintf(line, sizeof line, "X %u %u", e.step, e.pc); break;
            case m1::E_HALT: std::snprintf(line, sizeof line, "H"); break;
            case m1::E_FAULT: std::snprintf(line, sizeof line, "F %s", m1::FAULT_NAMES[e.c]); break;
            default: std::snprintf(line, sizeof line, "L"); break;
        }
        text += line;
        text += '\n';
        mine.push_back(line);
    }
    if (!out_path.empty() && !write_bytes(out_path, text.data(), text.size())) return 2;

    uint64_t ran = *(uint64_t*)(result + 4);
    std::printf("cosim %s: %llu steps on one GPU machine in %.3f ms (%.0f instructions a second)\n", in_path.c_str(),
                (unsigned long long)ran, ms, ms > 0 ? ran * 1000.0 / ms : 0.0);
    if (panel) {
        std::printf("  ended %s\n", result[2] == m1::HALTED ? "halted" : result[2] == m1::FAULTED ? "faulted" : "at the step limit");
        if (result[2] == m1::FAULTED) std::printf("  fault: %s\n", m1::FAULT_NAMES[result[3]]);
    }
    if (!has_run) {
        std::printf("cosim: %s has no reference run to compare with\n", in_path.c_str());
        return 0;
    }
    size_t at = 0;
    while (at < mine.size() && at < reference.size() && mine[at] == reference[at]) at++;
    if (at == mine.size() && at == reference.size()) {
        std::printf("cosim PASS: %zu run lines agree with %s\n", mine.size(), in_path.c_str());
        return 0;
    }
    std::printf("cosim FAIL: %s differs at run line %zu of %zu\n", in_path.c_str(), at + 1, reference.size());
    std::printf("  reference: %s\n", at < reference.size() ? reference[at].c_str() : "<end>");
    std::printf("  GPU:       %s\n", at < mine.size() ? mine[at].c_str() : "<end>");
    if (out_path.empty()) {
        std::printf("  run again with --out to keep the GPU's trace\n");
    } else {
        std::printf("  the GPU's trace is in %s\n", out_path.c_str());
    }
    return 1;
}

// --- array mode ---------------------------------------------------------------------------

int array(int argc, char** argv) {
    if (argc < 3) return usage();
    std::string out_path, job_path = argv[2];
    for (int i = 3; i < argc; i++) {
        if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else return usage();
    }
    std::vector<uint8_t> job;
    if (!read_bytes(job_path, job)) return 2;
    if (job.size() < 52 || std::memcmp(job.data(), "M1AJ", 4) != 0) {
        std::fprintf(stderr, "%s: not an array job file\n", job_path.c_str());
        return 2;
    }
    auto u32 = [&](size_t at) { return (uint32_t)job[at] | (uint32_t)job[at + 1] << 8 | (uint32_t)job[at + 2] << 16 | (uint32_t)job[at + 3] << 24; };
    auto u64 = [&](size_t at) { return (uint64_t)u32(at) | (uint64_t)u32(at + 4) << 32; };
    m1::ArrayHead h;
    h.instances = u32(8);
    h.cells = u32(12);
    h.max_steps = u64(16);
    h.n_inputs = u32(24);
    h.n_outputs = u32(28);
    h.image_cells = u32(32);
    h.org = u32(36);
    h.entry = u32(40);
    h.sp = u32(44);
    const size_t patches_bytes = (size_t)h.instances * h.n_inputs * 16;
    const size_t picks_bytes = (size_t)h.instances * h.n_outputs * 8;
    const size_t image_bytes = (size_t)h.image_cells * 2;
    const size_t need = 52 + patches_bytes + picks_bytes + image_bytes;
    if (u32(4) != 1 || job.size() != need) {
        std::fprintf(stderr, "%s: a job of version %u and %zu bytes; expected version 1 and %zu bytes\n", job_path.c_str(),
                     u32(4), job.size(), need);
        return 2;
    }
    if (h.instances == 0 || h.n_inputs > 4096 || h.n_outputs > 4096) return usage();
    const uint8_t* patches = job.data() + 52;
    const uint8_t* picks = patches + patches_bytes;
    const uint16_t* image = (const uint16_t*)(picks + picks_bytes);
    const size_t memory_bytes = (size_t)h.instances * h.cells * 2;
    size_t free_bytes = 0, total_bytes = 0;
    CK(hipMemGetInfo(&free_bytes, &total_bytes));
    const size_t results_bytes = (size_t)h.instances * (16 + (size_t)h.n_outputs * 8);
    if (memory_bytes + results_bytes + image_bytes + patches_bytes + picks_bytes > free_bytes) {
        std::fprintf(stderr, "%u machines of %u cells need %zu bytes; the device has %zu free\n", h.instances, h.cells,
                     memory_bytes + results_bytes, free_bytes);
        return 2;
    }

    uint16_t* d_image; CK(hipMalloc(&d_image, image_bytes));
    CK(hipMemcpy(d_image, image, image_bytes, hipMemcpyHostToDevice));
    m1::Patch* d_patches; CK(hipMalloc(&d_patches, patches_bytes ? patches_bytes : 16));
    if (patches_bytes) CK(hipMemcpy(d_patches, patches, patches_bytes, hipMemcpyHostToDevice));
    m1::Pick* d_picks; CK(hipMalloc(&d_picks, picks_bytes ? picks_bytes : 16));
    if (picks_bytes) CK(hipMemcpy(d_picks, picks, picks_bytes, hipMemcpyHostToDevice));
    uint8_t* d_memory; CK(hipMalloc(&d_memory, memory_bytes));
    uint64_t* d_steps; CK(hipMalloc(&d_steps, (size_t)h.instances * 8));
    uint8_t* d_status; CK(hipMalloc(&d_status, h.instances));
    uint64_t* d_values; CK(hipMalloc(&d_values, (size_t)h.instances * h.n_outputs * 8 + 8));

    uint32_t block = m1::ARRAY_BLOCK;
    uint32_t groups = (h.instances + block - 1) / block;
    hipEvent_t began, ended;
    CK(hipEventCreate(&began));
    CK(hipEventCreate(&ended));
    auto wall0 = std::chrono::steady_clock::now();
    CK(hipEventRecord(began));
    hipLaunchKernelGGL(m1::array_kernel, dim3(groups), dim3(block), block * m1::REG_STRIDE, 0, h, d_image, d_patches,
                       d_picks, d_memory, d_steps, d_status, d_values);
    CK(hipGetLastError());
    CK(hipEventRecord(ended));
    std::vector<uint64_t> steps(h.instances);
    std::vector<uint8_t> status(h.instances);
    std::vector<uint64_t> values((size_t)h.instances * h.n_outputs);
    CK(hipMemcpy(steps.data(), d_steps, steps.size() * 8, hipMemcpyDeviceToHost));
    CK(hipMemcpy(status.data(), d_status, status.size(), hipMemcpyDeviceToHost));
    if (h.n_outputs) CK(hipMemcpy(values.data(), d_values, values.size() * 8, hipMemcpyDeviceToHost));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, began, ended));
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    CK(hipFree(d_image));
    CK(hipFree(d_patches));
    CK(hipFree(d_picks));
    CK(hipFree(d_memory));
    CK(hipFree(d_steps));
    CK(hipFree(d_status));
    CK(hipFree(d_values));

    if (!out_path.empty()) {
        std::vector<uint8_t> out;
        out.reserve(results_bytes + 16);
        auto put32 = [&](uint32_t v) { for (int k = 0; k < 4; k++) out.push_back((uint8_t)(v >> (8 * k))); };
        auto put64 = [&](uint64_t v) { put32((uint32_t)v); put32((uint32_t)(v >> 32)); };
        out.insert(out.end(), {'M', '1', 'A', 'R'});
        put32(1);
        put32(h.instances);
        put32(h.n_outputs);
        for (uint32_t i = 0; i < h.instances; i++) {
            put64(steps[i]);
            put32(status[i] & 3);            // status
            put32((status[i] >> 2) & 3);     // fault kind
            put32((status[i] >> 4) & 31);    // flags
            put32(0);
            for (uint32_t k = 0; k < h.n_outputs; k++) put64(values[(size_t)i * h.n_outputs + k]);
        }
        if (!write_bytes(out_path, out.data(), out.size())) return 2;
    }

    uint64_t total_steps = 0, halted = 0, faults[4] = {0, 0, 0, 0}, limited = 0;
    for (uint32_t i = 0; i < h.instances; i++) {
        total_steps += steps[i];
        uint32_t state = status[i] & 3;
        if (state == m1::HALTED) halted++;
        else if (state == m1::FAULTED) faults[(status[i] >> 2) & 3]++;
        else limited++;
    }
    double seconds = ms / 1000.0;
    std::printf("array: %u machines, %u cells each, step limit %llu\n", h.instances, h.cells,
                (unsigned long long)h.max_steps);
    std::printf("array: kernel %.3f ms, host %.3f s including setup and copy back\n", ms, wall);
    std::printf("array: %llu instructions ran in all (%.0f a machine on average); %.3f giga machine-instructions a second in aggregate\n",
                (unsigned long long)total_steps, h.instances ? (double)total_steps / h.instances : 0.0,
                seconds > 0 ? (double)total_steps / seconds / 1e9 : 0.0);
    std::printf("array: %llu halted, %llu faulted (%llu address, %llu illegal instruction, %llu not a character), %llu at the limit\n",
                (unsigned long long)halted, (unsigned long long)(faults[1] + faults[2] + faults[3]),
                (unsigned long long)faults[1], (unsigned long long)faults[2], (unsigned long long)faults[3],
                (unsigned long long)limited);
    return 0;
}

int info() {
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, 0));
    std::printf("device 0: %s, %s\n", p.name, p.gcnArchName);
    std::printf("  %d compute units, wavefront %d, %d kHz, %d KiB of shared memory a workgroup\n", p.multiProcessorCount,
                p.warpSize, p.clockRate, (int)(p.sharedMemPerBlock >> 10));
    int runtime = 0, driver = 0;
    hipDriverGetVersion(&driver);
    hipRuntimeGetVersion(&runtime);
    std::printf("  HIP runtime %d, driver %d\n", runtime, driver);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    if (!std::strcmp(argv[1], "cosim")) return cosim(argc, argv);
    if (!std::strcmp(argv[1], "array")) return array(argc, argv);
    if (!std::strcmp(argv[1], "info")) return info();
    return usage();
}
