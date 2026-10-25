// main.cpp - pelias_probe: one program, one subcommand per probe.
//   pelias_probe <selftest|state|list|h8|h1|h2|h3|h10|h4|h5|h6|h12|h7|m3|m4|h9|h13> [--quick] [--out DIR] [--tag NAME]
//                [--machine-sha SHA] [probe options]
// Every probe writes one JSON file into --out (never overwrites). Exit code: 0 ok, 2 error, 3 probe failed its threshold.
#include <cstdio>

#include "probes_b.h"

namespace {

using namespace pp;

struct Check {
  std::string name;
  bool ok = false;
  bool info_only = false;
  std::string detail;
};

std::string upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  return s;
}

void write_failure(const Ctx& c, const std::string& sub, const std::string& what) {
  Result r;
  r.probe = "P-" + upper(sub);
  r.status = "error";
  r.error = what;
  r.headline = what;
  r.notes = "The probe stopped with an error before finishing; nothing here is a measurement.";
  write_result(c, r, sub + "_error");
}

// ---------------------------------------------------------------- machine state
int cmd_state(Ctx& c) {
  Result r;
  r.probe = "STATE";
  JW host;
  const HostMem m = host_mem();
  host.kv("platform", platform_name()).kv("cpu", cpu_model()).kv("os", os_text()).kv("hardware_threads", hw_threads()).kv("avx2", cpu_has_avx2());
  host.kv("mem_total_gib", m.total_gib()).kv("mem_available_gib", m.avail_gib()).kv("swap_total_mib", static_cast<double>(m.swap_total_kib) / 1024.0).kv("swap_free_mib", static_cast<double>(m.swap_free_kib) / 1024.0);
  host.kv("rlimit_memlock_mib_0_means_unlimited", static_cast<double>(memlock_limit_bytes()) / kMiB).kv("anon_huge_mib", static_cast<double>(m.anon_huge_kib) / 1024.0);
  JW gpu;
  const bool nv = c.nvml.open();
  gpu.kv("nvml_available", nv);
  if (nv) {
    gpu.kv("name", c.nvml.name()).kv("driver", c.nvml.driver());
    unsigned v = 0;
    if (c.nvml.clock(1, v)) gpu.kv("sm_clock_now_mhz", v);
    if (c.nvml.clock(2, v)) gpu.kv("mem_clock_now_mhz", v);
    if (c.nvml.max_clock(1, v)) gpu.kv("sm_clock_max_mhz", v);
    if (c.nvml.max_clock(2, v)) gpu.kv("mem_clock_max_mhz", v);
    double w = 0;
    if (c.nvml.power_limit_w(w)) gpu.kv("power_limit_w", w);
    unsigned g = 0, wd = 0, gm = 0, wm = 0;
    if (c.nvml.pcie(g, wd, gm, wm)) gpu.kv("pcie_gen_now", g).kv("pcie_width_now", wd).kv("pcie_gen_max", gm).kv("pcie_width_max", wm);
    NvmlMemory mem{};
    if (c.nvml.mem(mem)) gpu.kv("vram_total_mib", static_cast<double>(mem.total) / kMiB).kv("vram_used_mib", static_cast<double>(mem.used) / kMiB);
    const Sample s = c.nvml.sample();
    if (s.temp_c) gpu.kv("temp_c", s.temp_c);
    if (s.power_w >= 0) gpu.kv("power_w_now", s.power_w);
  }
  int rt = 0, drv = 0;
  const cudaError_t e = cudaRuntimeGetVersion(&rt);
  (void)cudaDriverGetVersion(&drv);
  gpu.kv("cuda_runtime_version", e == cudaSuccess ? rt : -1).kv("cuda_driver_version", drv);
  r.values.kv_obj("host", host).kv_obj("gpu", gpu);
  r.headline = platform_name() + ", " + std::to_string(m.total_gib()).substr(0, 5) + " GiB RAM, driver " + (nv ? c.nvml.driver() : std::string("unknown"));
  r.notes = "Snapshot taken at the start of a run. Clocks and temperature are 'now' values at an unknown load.";
  write_result(c, r, "state");
  return 0;
}

// ---------------------------------------------------------------- self-test
int cmd_selftest(Ctx& c) {
  std::vector<Check> checks;
  auto add = [&](const std::string& name, bool ok, const std::string& detail, bool info = false) {
    checks.push_back({name, ok, info, detail});
    std::printf("  [%s] %s%s%s\n", ok ? "ok  " : (info ? "info" : "FAIL"), name.c_str(), detail.empty() ? "" : " - ", detail.c_str());
    std::fflush(stdout);
  };
  auto guarded = [&](const std::string& name, const std::function<void()>& fn) {
    try {
      fn();
    } catch (const std::exception& e) {
      add(name, false, std::string("exception: ") + e.what());
    }
  };

  guarded("cuda_init", [&] {
    init_cuda(c);
    char b[160];
    std::snprintf(b, sizeof b, "%s, sm_%d%d, %d SMs, %.2f GiB", c.dev_name.c_str(), c.cc_major, c.cc_minor, c.sm_count, static_cast<double>(c.vram_total) / kGiB);
    add("cuda_init", true, b);
  });
  if (c.sm_count == 0) {
    add("abort", false, "no CUDA device: the remaining checks need one");
  } else {
    Stream s;
    guarded("empty_kernel", [&] {
      PP_CK(pk::launch_empty(s));
      s.sync();
      add("empty_kernel", true, "");
    });
    guarded("tensor_kernels", [&] {
      const int blocks = gpu_blocks(c, 4);
      DevBuf sink(static_cast<size_t>(blocks) * pk::kMmaThreads * sizeof(float));
      for (const pk::Mma k : {pk::Mma::BF16, pk::Mma::FP16, pk::Mma::INT8, pk::Mma::FP8}) {
        if (k == pk::Mma::FP8 && (c.cc_major * 10 + c.cc_minor) < 89) {
          add(std::string("mma_") + pk::mma_name(k), true, "skipped: needs sm_89 or newer", true);
          continue;
        }
        const unsigned it = calibrate_mma_iters(k, blocks, sink.as<float>(), s, 20.0);
        const double ms = time_stream_ms(s, [&] { PP_CK(pk::launch_mma(k, blocks, it, sink.as<float>(), s)); });
        const double tf = pk::mma_launch_flop(k, blocks, it) / (ms * 1e-3) / 1e12;
        char b[120];
        std::snprintf(b, sizeof b, "%.1f T(FL)OPS in a %.0f ms burst (not a benchmark)", tf, ms);
        add(std::string("mma_") + pk::mma_name(k), tf > 1.0 && ms > 1.0, b);
      }
    });
    guarded("canary_gemm", [&] {
      Canary can(2);
      can.fill(s);
      can.enqueue(0, s);
      can.enqueue(1, s);
      s.sync();
      unsigned long long h[2] = {0, 0};
      float sm[2] = {0, 0};
      PP_CK(cudaMemcpy(h, can.hash_out.as(), sizeof h, cudaMemcpyDeviceToHost));
      PP_CK(cudaMemcpy(sm, can.sum_out.as(), sizeof sm, cudaMemcpyDeviceToHost));
      const double ratio = can.check_cpu();
      char b[160];
      std::snprintf(b, sizeof b, "cpu-check error/tolerance %.3f (must be <= 1), two runs identical: %s", ratio, (h[0] == h[1] && std::memcmp(&sm[0], &sm[1], 4) == 0) ? "yes" : "NO");
      add("canary_gemm", ratio <= 1.0 && h[0] == h[1] && h[0] != 0 && std::memcmp(&sm[0], &sm[1], 4) == 0, b);
    });
    guarded("streaming_kernels", [&] {
      const size_t bytes = size_t{64} << 20;
      DevBuf A(bytes), B(bytes), sink(8);
      PP_CK(cudaMemset(A.as(), 0x11, bytes));
      PP_CK(cudaMemset(B.as(), 0x22, bytes));
      PP_CK(cudaMemset(sink.as(), 0, 8));
      const int blocks = gpu_blocks(c, 32);
      PP_CK(pk::launch_read(A.as(), bytes / 16, sink.as<unsigned long long>(), blocks, 256, s));
      PP_CK(pk::launch_write(B.as(), bytes / 16, blocks, 256, s));
      s.sync();
      uint32_t w[4] = {0, 0, 0, 0};
      PP_CK(cudaMemcpy(w, B.as(), 16, cudaMemcpyDeviceToHost));
      const bool wr = w[0] == 0xA5A5A5A5u && w[1] == 0x5A5A5A5Au && w[2] == 0x0F0F0F0Fu && w[3] == 0xF0F0F0F0u;
      PP_CK(pk::launch_copy(A.as(), B.as(), bytes / 16, blocks, 256, s));
      s.sync();
      uint32_t cw[4] = {0, 0, 0, 0};
      PP_CK(cudaMemcpy(cw, A.as<unsigned char>() + bytes - 16, 16, cudaMemcpyDeviceToHost));
      const bool cp = cw[0] == 0xA5A5A5A5u && cw[3] == 0xF0F0F0F0u;
      add("streaming_kernels", wr && cp, std::string("write pattern ") + (wr ? "ok" : "WRONG") + ", copy " + (cp ? "ok" : "WRONG"));
    });
    guarded("spin_kernel", [&] {
      DevBuf sink(static_cast<size_t>(c.sm_count) * 8 * 4);
      PP_CK(cudaMemset(sink.as(), 0, sink.bytes()));
      const double t0 = now_s();
      PP_CK(pk::launch_spin(100000000ULL, sink.as<unsigned long long>(), c.sm_count, s));
      s.sync();
      const double ms = (now_s() - t0) * 1000.0;
      char b[100];
      std::snprintf(b, sizeof b, "asked 100 ms, took %.0f ms", ms);
      add("spin_kernel", ms >= 90.0 && ms <= 500.0, b);
    });
    guarded("graph_replay", [&] {
      DevBuf ctr(8);
      PP_CK(cudaMemset(ctr.as(), 0, 8));
      cudaGraph_t g = nullptr;
      cudaGraphExec_t ex = nullptr;
      PP_CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      PP_CK(pk::launch_bump(ctr.as<unsigned long long>(), 1, s));
      PP_CK(pk::launch_bump(ctr.as<unsigned long long>(), 2, s));
      PP_CK(pk::launch_bump(ctr.as<unsigned long long>(), 3, s));
      PP_CK(cudaStreamEndCapture(s, &g));
      PP_CK(cudaGraphInstantiate(&ex, g, 0ULL));
      for (int i = 0; i < 5; ++i) PP_CK(cudaGraphLaunch(ex, s));
      s.sync();
      unsigned long long v = 0;
      PP_CK(cudaMemcpy(&v, ctr.as(), 8, cudaMemcpyDeviceToHost));
      cudaGraphExecDestroy(ex);
      cudaGraphDestroy(g);
      add("graph_replay", v == 30, "counter " + std::to_string(v) + " (expected 30)");
    });
    guarded("pinned_roundtrip", [&] {
      const size_t bytes = size_t{64} << 20;
      PinnedBuf p;
      PP_CK(p.try_alloc(bytes));
      AlignedBuf back(4096, bytes);
      if (!back.p) throw std::runtime_error("host allocation failed");
      fill_pattern(p.ptr(), bytes, 9);
      std::memset(back.p, 0, bytes);
      DevBuf d(bytes);
      PP_CK(cudaMemcpy(d.as(), p.ptr(), bytes, cudaMemcpyHostToDevice));
      PP_CK(cudaMemcpy(back.p, d.as(), bytes, cudaMemcpyDeviceToHost));
      add("pinned_roundtrip", std::memcmp(p.ptr(), back.p, bytes) == 0, "64 MiB host->device->host compared byte for byte");
    });
    guarded("shared_memory_optin", [&] {
      DevBuf out(sizeof(int));
      const cudaError_t e = pk::smem_probe(99 * 1024, out.as<int>());
      add("shared_memory_optin", e == cudaSuccess, "99 KiB dynamic shared memory per block: " + (e == cudaSuccess ? std::string("ok") : soft_err(e)));
    });
  }
  guarded("cpu_checksum", [&] {
    if (!cpu_has_avx2()) {
      add("cpu_checksum", true, "skipped: no AVX2", true);
      return;
    }
    AlignedBuf b(4096, size_t{1} << 20);
    fill_pattern(b.p, b.bytes, 1);
    uint64_t ref = 0;
    const auto* w = static_cast<const uint64_t*>(b.p);
    for (size_t i = 0; i < b.bytes / 8; ++i) ref += w[i];
    add("cpu_checksum", cpu_read_sum_avx2(b.p, b.bytes) == ref, "AVX2 sum equals the plain 64-bit word sum");
  });
  add("nvml", c.nvml.open(), c.nvml.ok() ? c.nvml.name() + ", driver " + c.nvml.driver() : "not available: clocks/temperature/throttle reasons will be missing from results", true);

  // Tensor-related failures (mma_*, canary_gemm) only block the tensor probes; anything else blocks everything.
  bool fatal = false, tensor = false;
  std::vector<JW> list;
  for (const auto& k : checks) {
    JW j;
    j.kv("name", k.name).kv("ok", k.ok).kv("informational", k.info_only).kv("detail", k.detail);
    list.push_back(j);
    if (!k.ok && !k.info_only) {
      if (k.name.rfind("mma_", 0) == 0 || k.name == "canary_gemm" || k.name == "tensor_kernels") tensor = true;
      else fatal = true;
    }
  }
  Result r;
  r.probe = "SELFTEST";
  r.values.arr_obj("checks", list).kv("fatal_failure", fatal).kv("tensor_failure_only", tensor && !fatal);
  r.pass = (!fatal && !tensor) ? 1 : 0;
  r.headline = fatal ? "FAILED - do not trust the probe numbers until this is fixed"
                     : (tensor ? "tensor-core checks FAILED (everything else passed): tensor probes P-H3, P-H10, P-H4 are blocked" : "all checks passed");
  r.notes = "This only proves the program's own kernels and plumbing work. It is not a performance measurement. Exit code: 0 all ok, 3 fatal failure, 4 only tensor-related failures.";
  write_result(c, r, "selftest");
  return fatal ? 3 : (tensor ? 4 : 0);
}

void usage() {
  std::puts(
      "pelias_probe <command> [--quick] [--out DIR] [--tag NAME] [--machine-sha SHA]\n"
      "  selftest   check the program's own kernels and plumbing (run this first)\n"
      "  state      record machine facts\n"
      "  h8 h1 h2 h3 h10 h4     batch A: VRAM, device limits, bandwidth, tensor rates, canary, soak\n"
      "  h5 h6 h12 h7 m3 m4 h9  batch B: PCIe transfers, contention, overlap, pinned limit, long hold, model file, graphs\n"
      "  h13 --stage 2s|5s|30s|queued|graph   display-timeout test (may reset the display driver)\n"
      "options: h3 --seconds N | h4 --minutes N | h10 --iters N | h7 --max-gib N | m3 --pinned-gib N --pageable-gib N --minutes N | m4 --gguf PATH (or env PELIAS_GGUF)");
}

}  // namespace

int main(int argc, char** argv) {
  const Args a(argc, argv);
  if (a.pos().empty() || a.has("help")) {
    usage();
    return a.pos().empty() ? 1 : 0;
  }
  Ctx c;
  const std::string sub = a.pos()[0];
  c.out_dir = a.get("out", "./pelias_probe_out");
  c.tag = a.get("tag", "");
  c.machine_sha = a.get("machine-sha", "unrecorded");
  c.quick = a.has("quick");
  for (int i = 0; i < argc; ++i) c.cmd += std::string(i ? " " : "") + argv[i];
  if (sub != "m4" && sub != "list") (void)c.nvml.open();

  struct Entry { const char* name; int (*fn)(Ctx&, const Args&); };
  static const Entry table[] = {{"h8", p_h8},   {"h1", p_h1},   {"h2", p_h2},   {"h3", p_h3},   {"h10", p_h10}, {"h4", p_h4},   {"h5", p_h5},
                                {"h6", p_h6},   {"h12", p_h12}, {"h7", p_h7},   {"m3", p_m3},   {"m4", p_m4},   {"h9", p_h9},   {"h13", p_h13}};
  try {
    if (sub == "list") {
      for (const auto& e : table) std::printf("%s\n", e.name);
      return 0;
    }
    if (sub == "state") return cmd_state(c);
    if (sub == "selftest") return cmd_selftest(c);
    for (const auto& e : table)
      if (sub == e.name) return e.fn(c, a);
    std::fprintf(stderr, "unknown command '%s'\n", sub.c_str());
    usage();
    return 1;
  } catch (const CudaError& e) {
    std::fprintf(stderr, "CUDA error: %s\n", e.what());
    (void)cudaGetLastError();
    write_failure(c, sub, std::string("CUDA error: ") + e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    write_failure(c, sub, e.what());
    return 2;
  }
}
