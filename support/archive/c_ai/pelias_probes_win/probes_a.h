// probes_a.h - Batch A (platform): P-H8, P-H1, P-H2, P-H3, P-H10, P-H4.
// Definitions are the master's section 11.2; each function writes one JSON through write_result().
#pragma once

#include "cudautil.h"

namespace pp {

// ================================================================ P-H8: free VRAM before / after context
inline int p_h8(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-H8";
  NvmlMemory nb{}, na{};
  NvmlBar1 b1b{}, b1a{};
  const bool have_nb = c.nvml.mem(nb);
  const bool have_b1b = c.nvml.bar1(b1b);
  init_cuda(c);  // creates the CUDA context: the cost of the context is the difference
  size_t f = 0, t = 0;
  PP_CK(cudaMemGetInfo(&f, &t));
  const bool have_na = c.nvml.mem(na);
  const bool have_b1a = c.nvml.bar1(b1a);

  // Ceiling 10.7 GiB: author ruling 2026-10-23 22:21 (it was 10.9 with a 0.3 margin until the first measured readings:
  // CUDA free after the context was 10.83 GiB in three runs, NVML free after the context 10.72 to 10.88 GiB).
  const double ceiling = a.num("ceiling-gib", 10.7), margin = a.num("margin-gib", 0.0), need = ceiling + margin;
  const double cuda_free_gib = static_cast<double>(f) / kGiB;
  const double nvml_free_gib = have_na ? static_cast<double>(na.free) / kGiB : cuda_free_gib;
  const double judged = std::min(cuda_free_gib, nvml_free_gib);  // the smaller of the two readings, after the context exists
  JW v;
  v.kv("cuda_free_gib", cuda_free_gib).kv("cuda_total_gib", static_cast<double>(t) / kGiB);
  v.kv("nvml_after_ctx_free_gib", nvml_free_gib).kv("judged_free_gib_smaller_of_cuda_and_nvml_after_ctx", judged).kv("margin_over_required_gib", judged - need);
  if (have_nb) v.kv("nvml_before_ctx_used_mib", static_cast<double>(nb.used) / kMiB).kv("nvml_before_ctx_free_mib", static_cast<double>(nb.free) / kMiB).kv("nvml_total_mib", static_cast<double>(nb.total) / kMiB);
  else v.kv_null("nvml_before_ctx_used_mib");
  if (have_na) v.kv("nvml_after_ctx_used_mib", static_cast<double>(na.used) / kMiB).kv("nvml_after_ctx_free_mib", static_cast<double>(na.free) / kMiB);
  if (have_nb && have_na) v.kv("context_cost_mib", (static_cast<double>(na.used) - static_cast<double>(nb.used)) / kMiB);
  if (have_b1b) v.kv("bar1_before_total_mib", static_cast<double>(b1b.total) / kMiB).kv("bar1_before_used_mib", static_cast<double>(b1b.used) / kMiB);
  if (have_b1a) v.kv("bar1_after_total_mib", static_cast<double>(b1a.total) / kMiB).kv("bar1_after_used_mib", static_cast<double>(b1a.used) / kMiB);
  r.values = v;
  r.thresholds.kv("vram_ceiling_gib", ceiling).kv("margin_gib", margin).kv("required_free_gib", need).kv("ceiling_source", "author ruling 2026-10-23 22:21 (was 10.9 + 0.3 margin before the first measured readings)");
  r.pass = (judged >= need) ? 1 : 0;
  char b[220];
  std::snprintf(b, sizeof b, "free after context: CUDA %.2f / NVML %.2f of %.2f GiB; need >= %.2f (margin %+.2f)", cuda_free_gib, nvml_free_gib, static_cast<double>(t) / kGiB, need, judged - need);
  r.headline = b;
  r.notes = "Close browsers/games before the run for a clean reading. Windows-side per-app VRAM use is in the PowerShell facts file.";
  write_result(c, r, "h8_vram");
  return r.pass == 1 ? 0 : 3;
}

// ================================================================ P-H1: device limits + driver-surface census
struct AttrRow {
  const char* name;
  cudaDeviceAttr id;
};
#define PP_ATTR(x) {#x, x}

inline int p_h1(Ctx& c, const Args&) {
  Result r;
  r.probe = "P-H1";
  init_cuda(c);

  static const AttrRow rows[] = {
      PP_ATTR(cudaDevAttrMultiProcessorCount),
      PP_ATTR(cudaDevAttrComputeCapabilityMajor),
      PP_ATTR(cudaDevAttrComputeCapabilityMinor),
      PP_ATTR(cudaDevAttrWarpSize),
      PP_ATTR(cudaDevAttrMaxThreadsPerBlock),
      PP_ATTR(cudaDevAttrMaxThreadsPerMultiProcessor),
      PP_ATTR(cudaDevAttrMaxSharedMemoryPerBlock),
      PP_ATTR(cudaDevAttrMaxSharedMemoryPerBlockOptin),
      PP_ATTR(cudaDevAttrMaxSharedMemoryPerMultiprocessor),
      PP_ATTR(cudaDevAttrReservedSharedMemoryPerBlock),
      PP_ATTR(cudaDevAttrMaxRegistersPerBlock),
      PP_ATTR(cudaDevAttrMaxRegistersPerMultiprocessor),
      PP_ATTR(cudaDevAttrL2CacheSize),
      PP_ATTR(cudaDevAttrGlobalMemoryBusWidth),
      PP_ATTR(cudaDevAttrClockRate),
      PP_ATTR(cudaDevAttrMemoryClockRate),
      PP_ATTR(cudaDevAttrAsyncEngineCount),
      PP_ATTR(cudaDevAttrConcurrentKernels),
      PP_ATTR(cudaDevAttrEccEnabled),
      PP_ATTR(cudaDevAttrKernelExecTimeout),
      PP_ATTR(cudaDevAttrIntegrated),
      PP_ATTR(cudaDevAttrTccDriver),
      PP_ATTR(cudaDevAttrComputeMode),
      PP_ATTR(cudaDevAttrUnifiedAddressing),
      PP_ATTR(cudaDevAttrCanMapHostMemory),
      PP_ATTR(cudaDevAttrHostRegisterSupported),
      PP_ATTR(cudaDevAttrHostRegisterReadOnlySupported),
      PP_ATTR(cudaDevAttrCanUseHostPointerForRegisteredMem),
      PP_ATTR(cudaDevAttrMaxPersistingL2CacheSize),
      PP_ATTR(cudaDevAttrHostNativeAtomicSupported),
      PP_ATTR(cudaDevAttrPageableMemoryAccess),
      PP_ATTR(cudaDevAttrPageableMemoryAccessUsesHostPageTables),
      PP_ATTR(cudaDevAttrDirectManagedMemAccessFromHost),
      PP_ATTR(cudaDevAttrManagedMemory),
      PP_ATTR(cudaDevAttrConcurrentManagedAccess),
      PP_ATTR(cudaDevAttrMemoryPoolsSupported),
      PP_ATTR(cudaDevAttrMemoryPoolSupportedHandleTypes),
      PP_ATTR(cudaDevAttrGPUDirectRDMASupported),
      PP_ATTR(cudaDevAttrGPUDirectRDMAFlushWritesOptions),
      PP_ATTR(cudaDevAttrGPUDirectRDMAWritesOrdering),
      PP_ATTR(cudaDevAttrIpcEventSupport),
      PP_ATTR(cudaDevAttrCooperativeLaunch),
      PP_ATTR(cudaDevAttrHostNumaId),
  };
  JW attrs;
  int opt_in = 0, sm_count = 0;
  for (const auto& a : rows) {
    int val = 0;
    const cudaError_t e = cudaDeviceGetAttribute(&val, a.id, 0);
    if (e == cudaSuccess) {
      attrs.kv(a.name, val);
      if (a.id == cudaDevAttrMaxSharedMemoryPerBlockOptin) opt_in = val;
      if (a.id == cudaDevAttrMultiProcessorCount) sm_count = val;
    } else {
      (void)cudaGetLastError();
      attrs.kv(a.name, "unsupported_by_this_runtime_or_driver");
    }
  }

  // Ramp the dynamic shared-memory opt-in until the driver refuses.
  DevBuf d_out(sizeof(int));
  int ok_max = 0;
  std::string first_fail;
  for (int kib = 48; kib <= 256; ++kib) {
    const int bytes = kib * 1024;
    const cudaError_t e = pk::smem_probe(bytes, d_out.as<int>());
    if (e == cudaSuccess) {
      ok_max = bytes;
    } else {
      first_fail = std::to_string(kib) + " KiB: " + soft_err(e);
      break;
    }
  }

  cudaFuncAttributes fa{};
  int occ = 0;
  const std::string fe = soft_err(pk::mma_func_info(&fa, &occ));
  int rt = 0, drv = 0;
  (void)cudaRuntimeGetVersion(&rt);
  (void)cudaDriverGetVersion(&drv);

  JW dev;
  dev.kv("name", c.dev_name).kv("vram_total_gib", static_cast<double>(c.vram_total) / kGiB).kv("cuda_runtime_version", rt).kv("cuda_driver_version", drv);
  JW sm;
  sm.kv("opt_in_attribute_bytes", opt_in).kv("ramp_largest_ok_bytes", ok_max).kv("ramp_first_failure", first_fail);
  JW kf;
  if (fe.empty()) kf.kv("registers_per_thread", fa.numRegs).kv("static_shared_bytes", fa.sharedSizeBytes).kv("max_threads_per_block", fa.maxThreadsPerBlock).kv("occupancy_blocks_per_sm_128_threads", occ);
  else kf.kv("error", fe);
  r.values.kv_obj("device", dev).kv_obj("shared_memory", sm).kv_obj("tensor_kernel", kf).kv_obj("attributes", attrs);
  r.thresholds.kv("opt_in_shared_bytes_min", 99 * 1024).kv("sm_count_expected", 46);
  r.pass = (opt_in >= 99 * 1024 && sm_count == 46) ? 1 : 0;
  char b[220];
  std::snprintf(b, sizeof b, "%s sm=%d opt-in smem=%d B ramp-ok=%d B regs=%d", c.dev_name.c_str(), sm_count, opt_in, ok_max, fe.empty() ? fa.numRegs : -1);
  r.headline = b;
  r.notes = "attributes block = driver-surface census: what this driver stack exposes (RDMA, pageable access, managed memory, host register...).";
  write_result(c, r, "h1_device");
  return 0;
}

// ================================================================ P-H2: VRAM and host DRAM bandwidth
inline std::vector<double> bw_reps(cudaStream_t s, int reps, double bytes_moved, const std::function<void()>& launch) {
  std::vector<double> v;
  for (int i = 0; i < 2; ++i) {  // warm-up
    launch();
    PP_CK(cudaStreamSynchronize(s));
  }
  for (int i = 0; i < reps; ++i) v.push_back(gbps(bytes_moved, time_stream_ms(s, launch)));
  return v;
}

inline int p_h2(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-H2";
  init_cuda(c);
  const size_t bytes = static_cast<size_t>(qnum(c, a, "gib", 1.0, 2.0) * kGiB);
  const int reps = c.quick ? 5 : 20;
  const size_t n16 = bytes / 16;
  DevBuf A(bytes), B(bytes), sink(sizeof(unsigned long long));
  PP_CK(cudaMemset(A.as(), 0x11, bytes));
  PP_CK(cudaMemset(B.as(), 0x22, bytes));
  PP_CK(cudaMemset(sink.as(), 0, sizeof(unsigned long long)));
  Stream s;
  const int blocks = gpu_blocks(c, 32), threads = 256;

  const auto rd = bw_reps(s, reps, static_cast<double>(bytes), [&] { PP_CK(pk::launch_read(A.as(), n16, sink.as<unsigned long long>(), blocks, threads, s)); });
  const auto wr = bw_reps(s, reps, static_cast<double>(bytes), [&] { PP_CK(pk::launch_write(B.as(), n16, blocks, threads, s)); });
  const auto cp = bw_reps(s, reps, 2.0 * static_cast<double>(bytes), [&] { PP_CK(pk::launch_copy(B.as(), A.as(), n16, blocks, threads, s)); });
  const auto d2d = bw_reps(s, reps, 2.0 * static_cast<double>(bytes), [&] { PP_CK(cudaMemcpyAsync(B.as(), A.as(), bytes, cudaMemcpyDeviceToDevice, s)); });

  JW vram;
  vram.kv_obj("read_GBps", stat_obj(rd)).kv_obj("write_GBps", stat_obj(wr)).kv_obj("copy_kernel_GBps_read_plus_write", stat_obj(cp)).kv_obj("memcpy_d2d_GBps_read_plus_write", stat_obj(d2d));

  // Host DRAM read, AVX2 loads, 1/4/8/16 threads.
  JW cpu;
  std::string cpu_note;
  if (cpu_has_avx2()) {
    const size_t cb = static_cast<size_t>(qnum(c, a, "cpu-gib", 1.0, 4.0) * kGiB);
    AlignedBuf buf(4096, cb);
    if (!buf.p) {
      cpu_note = "host buffer allocation failed";
    } else {
      fill_pattern(buf.p, cb, 7);
      for (const int nt : {1, 4, 8, 16}) {
        std::vector<double> g;
        for (int rep = 0; rep < (c.quick ? 2 : 4); ++rep) g.push_back(cpu_read_threads(buf.u8(), cb, nt, 2, nullptr).gbps());
        cpu.kv_obj("threads_" + std::to_string(nt) + "_GBps", stat_obj(g));
      }
    }
  } else {
    cpu_note = "CPU has no AVX2: host read test skipped";
  }
  r.values.kv_obj("vram", vram).kv_obj("host_dram_read", cpu).kv("buffer_gib", static_cast<double>(bytes) / kGiB);
  r.thresholds.kv("vram_spec_GBps_PUBLISHED", 504.0).kv("cpu_read_GBps_engine_MEASURED", 45.9);
  char b[200];
  std::snprintf(b, sizeof b, "VRAM read %.0f / write %.0f / copy %.0f GB/s (median)", median(rd), median(wr), median(cp));
  r.headline = b;
  r.notes = cpu_note + " GB = 1e9 bytes. Copy counts bytes read plus bytes written.";
  write_result(c, r, "h2_bandwidth");
  return 0;
}

// ================================================================ P-H3: sustained tensor rates
struct MmaRun {
  pk::Mma kind;
  double seconds = 0;
  unsigned iters = 0;
  std::vector<double> t_rel, tflops;
  double t_begin = 0, t_end = 0;
};

inline MmaRun run_mma_for(pk::Mma k, double seconds, double launch_ms, int blocks, float* sink, cudaStream_t s) {
  MmaRun m;
  m.kind = k;
  m.iters = calibrate_mma_iters(k, blocks, sink, s, launch_ms);
  m.t_begin = now_s();
  while (now_s() - m.t_begin < seconds && !g_abort.load()) {
    const double ms = time_stream_ms(s, [&] { PP_CK(pk::launch_mma(k, blocks, m.iters, sink, s)); });
    m.t_rel.push_back(now_s() - m.t_begin);
    m.tflops.push_back(pk::mma_launch_flop(k, blocks, m.iters) / (ms * 1e-3) / 1e12);
  }
  m.t_end = now_s();
  m.seconds = m.t_end - m.t_begin;
  return m;
}

inline int p_h3(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-H3";
  init_cuda(c);
  const double secs = qnum(c, a, "seconds", 3.0, 60.0);
  const int blocks = gpu_blocks(c, 4);
  DevBuf sink(static_cast<size_t>(blocks) * pk::kMmaThreads * sizeof(float));
  Stream s;
  Sampler sampler(c.nvml);
  sampler.start(100);
  const double t_ref = now_s();

  JW kinds;
  bool all_ok = true;
  std::string headline;
  struct Derived { pk::Mma k; double stock, oc; };
  const Derived derived[] = {{pk::Mma::BF16, 58.3, 65.4}, {pk::Mma::FP16, -1, -1}, {pk::Mma::INT8, 233.2, 261.4}, {pk::Mma::FP8, 116.6, -1}};
  JW thr;
  for (const pk::Mma k : {pk::Mma::BF16, pk::Mma::INT8, pk::Mma::FP16, pk::Mma::FP8}) {
    const MmaRun m = run_mma_for(k, secs, 250.0, blocks, sink.as<float>(), s);
    const std::vector<Sample> win = window(sampler.snapshot(), m.t_begin, m.t_end);
    const double tail_from = std::max(m.seconds - 20.0, m.seconds * 2.0 / 3.0);
    std::vector<double> early, late;
    for (size_t i = 0; i < m.tflops.size(); ++i) {
      if (m.t_rel[i] < 5.0) early.push_back(m.tflops[i]);
      if (m.t_rel[i] >= tail_from) late.push_back(m.tflops[i]);
    }
    const double burst = vmax(early), sustained = vmean(late);
    const double ratio = burst > 0 ? sustained / burst : std::nan("");
    const std::vector<Sample> late_s = window(sampler.snapshot(), m.t_begin + tail_from, m.t_end);
    std::vector<double> clk;
    for (const auto& x : late_s)
      if (x.sm_mhz) clk.push_back(x.sm_mhz);
    const double mean_clk = vmean(clk);
    JW kj;
    kj.kv("seconds", m.seconds).kv("launches", m.tflops.size()).kv("iters_per_launch", m.iters).kv("burst_tflops", burst).kv("sustained_tflops", sustained).kv("sustained_over_burst", ratio);
    if (!clk.empty()) kj.kv("sustained_window_sm_clock_mhz_mean", mean_clk).kv("tflops_per_sm_per_ghz", sustained / (static_cast<double>(c.sm_count) * mean_clk / 1000.0));
    kj.kv_obj("nvml_window", summarize(win));
    std::vector<double> ts, tf;
    const size_t step = std::max<size_t>(1, m.tflops.size() / 60);
    for (size_t i = 0; i < m.tflops.size(); i += step) { ts.push_back(m.t_rel[i]); tf.push_back(m.tflops[i]); }
    kj.arr_num("series_t_s", ts).arr_num("series_tflops", tf);
    kinds.kv_obj(pk::mma_name(k), kj);
    if (!(ratio >= 0.85)) all_ok = false;
    char b[100];
    std::snprintf(b, sizeof b, "%s %.1f ", k == pk::Mma::BF16 ? "bf16" : (k == pk::Mma::INT8 ? "int8" : (k == pk::Mma::FP16 ? "fp16" : "fp8")), sustained);
    headline += b;
    for (const auto& d : derived)
      if (d.k == k) {
        JW t;
        if (d.stock > 0) t.kv("derived_stock_tops_DERIVED", d.stock);
        if (d.oc > 0) t.kv("derived_overclock_tops_DERIVED", d.oc);
        if (!t.empty()) thr.kv_obj(pk::mma_name(k), t);
      }
    if (g_abort.load()) break;
  }
  const std::vector<Sample> all = sampler.stop();
  r.values.kv_obj("kinds", kinds).kv_obj("nvml_whole_run", summarize(all)).kv_obj("series", series(all, t_ref));
  r.inputs.kv("seconds_per_type", secs).kv("blocks", blocks).kv("threads_per_block", pk::kMmaThreads).kv("accumulator_chains_per_warp", pk::kMmaChains);
  r.thresholds.kv("sustained_over_burst_min", 0.85).kv_obj("derived_reference", thr);
  r.pass = all_ok ? 1 : 0;
  r.headline = "sustained TFLOPS/TOPS: " + headline;
  r.notes = "Register-resident-operand microbenchmark (no memory traffic). Run again after resetting Afterburner to stock with --tag stock. " + (g_abort.load() ? "ABORTED: " + abort_reason() : std::string());
  write_result(c, r, "h3_peak");
  return 0;
}

// ================================================================ canary GEMM (P-H10, P-H4)
class Canary {
 public:
  static constexpr int M = 2048, N = 2048, K = 2048;
  DevBuf A, Bt, C, acc, partial, hash_out, sum_out;
  size_t slots;

  explicit Canary(size_t nslots)
      : A(static_cast<size_t>(M) * K * 2), Bt(static_cast<size_t>(N) * K * 2), C(static_cast<size_t>(M) * N * 4),
        acc(sizeof(unsigned long long)), partial(pk::kHashBlocks * sizeof(float)),
        hash_out(nslots * sizeof(unsigned long long)), sum_out(nslots * sizeof(float)), slots(nslots) {}

  void fill(cudaStream_t s) {
    PP_CK(pk::launch_fill_bf16(A.as<uint16_t>(), static_cast<size_t>(M) * K, 12345u, s));
    PP_CK(pk::launch_fill_bf16(Bt.as<uint16_t>(), static_cast<size_t>(N) * K, 67890u, s));
  }
  void enqueue(size_t slot, cudaStream_t s) {
    PP_CK(cudaMemsetAsync(acc.as(), 0, sizeof(unsigned long long), s));
    PP_CK(pk::launch_gemm_bf16(A.as<uint16_t>(), Bt.as<uint16_t>(), C.as<float>(), M, N, K, s));
    PP_CK(pk::launch_hash(C.as<float>(), static_cast<size_t>(M) * N, acc.as<unsigned long long>(), partial.as<float>(), s));
    PP_CK(pk::launch_final(partial.as<float>(), acc.as<unsigned long long>(), hash_out.as<unsigned long long>(), sum_out.as<float>(), static_cast<int>(slot), s));
  }
  // Compare 256 sampled outputs with a double-precision CPU reference. Returns the worst error / tolerance ratio.
  double check_cpu() {
    std::vector<uint16_t> ha(static_cast<size_t>(M) * K), hb(static_cast<size_t>(N) * K);
    std::vector<float> hc(static_cast<size_t>(M) * N);
    PP_CK(cudaMemcpy(ha.data(), A.as(), ha.size() * 2, cudaMemcpyDeviceToHost));
    PP_CK(cudaMemcpy(hb.data(), Bt.as(), hb.size() * 2, cudaMemcpyDeviceToHost));
    PP_CK(cudaMemcpy(hc.data(), C.as(), hc.size() * 4, cudaMemcpyDeviceToHost));
    auto f = [](uint16_t h) {
      const uint32_t u = static_cast<uint32_t>(h) << 16;
      float x;
      std::memcpy(&x, &u, 4);
      return static_cast<double>(x);
    };
    double worst = 0;
    uint64_t x = 88172645463325252ull;
    for (int t = 0; t < 256; ++t) {
      x ^= x << 13; x ^= x >> 7; x ^= x << 17;
      const size_t i = static_cast<size_t>(x % M), j = static_cast<size_t>((x >> 20) % N);
      double ref = 0, mag = 0;
      for (size_t k = 0; k < K; ++k) {
        const double p = f(ha[i * K + k]) * f(hb[j * K + k]);
        ref += p;
        mag += std::fabs(p);
      }
      const double err = std::fabs(ref - static_cast<double>(hc[i * N + j]));
      worst = std::max(worst, err / (1e-4 * mag + 1e-6));
    }
    return worst;  // <= 1 means within tolerance
  }
};

inline int p_h10(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-H10";
  init_cuda(c);
  const size_t iters = static_cast<size_t>(a.has("iters") ? a.inum("iters", 10000) : (c.quick ? 1000 : 10000));
  Stream s;
  Canary can(iters);
  can.fill(s);
  can.enqueue(0, s);
  s.sync();
  const double cpu_ratio = can.check_cpu();
  Sampler sampler(c.nvml);
  sampler.start(200);
  const double t_ref = now_s();
  for (size_t done = 1; done < iters && !g_abort.load();) {
    const size_t end = std::min(iters, done + 500);
    for (size_t i = done; i < end; ++i) can.enqueue(i, s);
    s.sync();
    done = end;
  }
  const auto smp = sampler.stop();
  std::vector<unsigned long long> h(iters);
  std::vector<float> sm(iters);
  PP_CK(cudaMemcpy(h.data(), can.hash_out.as(), iters * 8, cudaMemcpyDeviceToHost));
  PP_CK(cudaMemcpy(sm.data(), can.sum_out.as(), iters * 4, cudaMemcpyDeviceToHost));
  size_t mism = 0, ran = 0;
  std::set<unsigned long long> distinct;
  std::vector<double> bad;
  for (size_t i = 0; i < iters; ++i) {
    if (h[i] == 0 && i > 0 && sm[i] == 0.f) continue;  // slot never ran (aborted early)
    ++ran;
    distinct.insert(h[i]);
    if (h[i] != h[0] || std::memcmp(&sm[i], &sm[0], 4) != 0) {
      ++mism;
      if (bad.size() < 20) bad.push_back(static_cast<double>(i));
    }
  }
  r.values.kv("iterations_requested", iters).kv("iterations_run", ran).kv("mismatches", mism).kv("distinct_hashes", distinct.size()).arr_num("first_mismatch_indices", bad)
      .kv("cpu_reference_worst_error_over_tolerance", cpu_ratio).kv("gemm_shape", std::to_string(Canary::M) + "x" + std::to_string(Canary::N) + "x" + std::to_string(Canary::K))
      .kv_obj("nvml", summarize(smp)).kv_obj("series", series(smp, t_ref));
  r.thresholds.kv("mismatches_max", 0).kv("cpu_error_over_tolerance_max", 1.0);
  r.pass = (mism == 0 && cpu_ratio <= 1.0 && ran == iters) ? 1 : 0;
  char b[160];
  std::snprintf(b, sizeof b, "%zu runs, %zu mismatches, %zu distinct hashes, cpu-check ratio %.3f", ran, mism, distinct.size(), cpu_ratio);
  r.headline = b;
  r.notes = "Same fixed GEMM + hash + fixed-order reduction every time at the current (run) clocks. Any difference marks the clock setting unsafe for training. " + (g_abort.load() ? "ABORTED: " + abort_reason() : std::string());
  write_result(c, r, "h10_canary");
  return r.pass == 1 ? 0 : 3;
}

// ================================================================ P-H4: overclock soak
inline int p_h4(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-H4";
  init_cuda(c);
  const double minutes = qnum(c, a, "minutes", 1.0, 30.0);
  const int blocks = gpu_blocks(c, 4);
  DevBuf sink(static_cast<size_t>(blocks) * pk::kMmaThreads * sizeof(float));
  Stream s;
  Canary can(1);
  can.fill(s);
  can.enqueue(0, s);
  s.sync();
  const double cpu_ratio = can.check_cpu();
  unsigned long long ref_hash = 0;
  float ref_sum = 0;
  PP_CK(cudaMemcpy(&ref_hash, can.hash_out.as(), 8, cudaMemcpyDeviceToHost));
  PP_CK(cudaMemcpy(&ref_sum, can.sum_out.as(), 4, cudaMemcpyDeviceToHost));

  const pk::Mma order[] = {pk::Mma::BF16, pk::Mma::INT8, pk::Mma::FP16, pk::Mma::FP8};
  unsigned iters[4];
  for (int i = 0; i < 4; ++i) iters[i] = calibrate_mma_iters(order[i], blocks, sink.as<float>(), s, 100.0);

  Sampler sampler(c.nvml);
  sampler.start(250);
  const double t_ref = now_s();
  const double t_end = t_ref + minutes * 60.0;
  size_t cycles = 0, mism = 0;
  std::vector<JW> events;
  while (now_s() < t_end && !g_abort.load()) {
    const int k = static_cast<int>(cycles % 4);
    PP_CK(pk::launch_mma(order[k], blocks, iters[k], sink.as<float>(), s));
    can.enqueue(0, s);
    unsigned long long h = 0;
    float sm = 0;
    PP_CK(cudaMemcpyAsync(&h, can.hash_out.as(), 8, cudaMemcpyDeviceToHost, s));
    PP_CK(cudaMemcpyAsync(&sm, can.sum_out.as(), 4, cudaMemcpyDeviceToHost, s));
    s.sync();
    if (h != ref_hash || std::memcmp(&sm, &ref_sum, 4) != 0) {
      ++mism;
      if (events.size() < 50) {
        JW e;
        e.kv("cycle", cycles).kv("t_s", now_s() - t_ref).kv("tensor_kernel_before", pk::mma_name(order[k])).kv("hash_got", std::to_string(h)).kv("hash_expected", std::to_string(ref_hash));
        events.push_back(e);
      }
    }
    ++cycles;
  }
  const auto smp = sampler.stop();
  const JW sum = summarize(smp);
  // Per-minute clock/temperature bins.
  std::vector<double> bin_clk, bin_temp, bin_pw;
  for (double t0 = t_ref; t0 < now_s(); t0 += 60.0) {
    std::vector<double> clk, tmp, pw;
    for (const auto& x : window(smp, t0, t0 + 60.0)) {
      if (x.sm_mhz) clk.push_back(x.sm_mhz);
      if (x.temp_c) tmp.push_back(x.temp_c);
      if (x.power_w >= 0) pw.push_back(x.power_w);
    }
    bin_clk.push_back(vmean(clk)); bin_temp.push_back(vmax(tmp)); bin_pw.push_back(vmean(pw));
  }
  unsigned long long orm = 0;
  size_t thermal = 0;
  for (const auto& x : smp) { orm |= x.throttle; if (x.throttle & kThermalBits) ++thermal; }
  const double elapsed = now_s() - t_ref;
  const bool completed = elapsed >= minutes * 60.0 - 1.0;
  r.values.kv("minutes_requested", minutes).kv("minutes_run", elapsed / 60.0).kv("cycles", cycles).kv("mismatches", mism).arr_obj("mismatch_events", events)
      .kv("cpu_reference_worst_error_over_tolerance", cpu_ratio).kv("thermal_throttle_samples", thermal).kv("throttle_reasons_seen", throttle_names(orm))
      .arr_num("per_minute_sm_clock_mhz_mean", bin_clk).arr_num("per_minute_temp_c_max", bin_temp).arr_num("per_minute_power_w_mean", bin_pw)
      .kv_obj("nvml", sum).kv_obj("series", series(smp, t_ref));
  r.thresholds.kv("mismatches_max", 0).kv("thermal_throttle_samples_max", 0).kv("abort_temp_c", static_cast<int>(kAbortTempC));
  r.pass = (mism == 0 && thermal == 0 && completed && cpu_ratio <= 1.0) ? 1 : 0;
  char b[200];
  std::snprintf(b, sizeof b, "%.1f min, %zu cycles, %zu mismatches, thermal samples %zu, throttle: %s", elapsed / 60.0, cycles, mism, thermal, throttle_names(orm).c_str());
  r.headline = b;
  r.notes = "Xid / display-driver errors cannot be read from inside WSL2 (the driver lives in Windows): the PowerShell facts script lists the Windows event log for the same time window. " + (g_abort.load() ? "ABORTED: " + abort_reason() : std::string());
  write_result(c, r, "h4_soak");
  return r.pass == 1 ? 0 : 3;
}

}  // namespace pp
