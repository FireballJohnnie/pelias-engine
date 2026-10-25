// probes_b.h - Batch B (transfers, host memory, files, graphs, display timeout):
//   P-H5, P-H6, P-H12, P-H7, P-M3, P-M4, P-H9, P-H13.
// Definitions are the master's section 11.2; each function writes one JSON through write_result().
#pragma once

#include "probes_a.h"

namespace pp {

using std::chrono::milliseconds;

// "Started" marker, written before anything risky, so a kill / OOM / driver reset still leaves evidence on disk.
inline std::string write_started(const Ctx& c, const std::string& probe, const std::string& stem, const std::string& what) {
  JW j;
  j.kv("probe", probe).kv("date", iso_now()).kv("machine_state_sha", c.machine_sha).kv("status", "started_not_finished");
  j.kv("platform", platform_name()).kv("tag", c.tag).kv("quick", c.quick).kv("about_to_run", what);
  const std::string path = unique_path(c.out_dir, stem + "_started" + (c.tag.empty() ? "" : "_" + c.tag), ".json");
  std::ofstream f(path, std::ios::binary);
  f << j.obj() << "\n";
  std::printf("[START] %-6s %s\n", probe.c_str(), what.c_str());
  std::fflush(stdout);
  return path;
}

inline std::string fmt_g(double v) {
  char b[40];
  std::snprintf(b, sizeof b, "%g", v);
  return b;
}

inline std::string mib_key(size_t mib) { return "mib_" + std::to_string(mib); }

// ================================================================ P-H5: host <-> device transfer rates
inline int p_h5(Ctx& c, const Args&) {
  Result r;
  r.probe = "P-H5";
  init_cuda(c);
  const std::vector<size_t> sizes = c.quick ? std::vector<size_t>{16, 256} : std::vector<size_t>{1, 4, 16, 64, 256, 512};
  const int reps = c.quick ? 3 : 10;
  const size_t maxb = sizes.back() << 20;
  const size_t pass_mib = 256;

  DevBuf dev(maxb);
  PP_CK(cudaMemset(dev.as(), 0x5a, maxb));
  Stream s;
  AlignedBuf pageable(4096, maxb);
  if (!pageable.p) throw std::runtime_error("host allocation of the pageable test buffer failed");
  fill_pattern(pageable.p, maxb, 3);

  PinnedBuf pin, wc, mapped;
  RegisteredBuf reg;
  JW alloc_err;
  const std::string e_pin = soft_err(pin.try_alloc(maxb));
  const std::string e_wc = soft_err(wc.try_alloc(maxb, cudaHostAllocWriteCombined));
  const std::string e_map = soft_err(mapped.try_alloc(maxb, cudaHostAllocMapped));
  const std::string e_reg = soft_err(reg.try_alloc(maxb));
  alloc_err.kv("pinned", e_pin).kv("write_combined", e_wc).kv("mapped_zero_copy", e_map).kv("registered", e_reg);
  if (pin.ptr()) fill_pattern(pin.ptr(), maxb, 4);
  if (wc.ptr()) std::memset(wc.ptr(), 0x33, maxb);  // write-combined memory: write-only access from the CPU
  if (mapped.ptr()) fill_pattern(mapped.ptr(), maxb, 6);

  std::map<std::string, double> med_at_pass;
  auto run_dir = [&](const char* name, void* host, cudaMemcpyKind dir) -> JW {
    JW j;
    if (!host) {
      j.kv("unavailable", "allocation failed (see alloc_errors)");
      return j;
    }
    for (const size_t mib : sizes) {
      const size_t bytes = mib << 20;
      const auto v = bw_reps(s, reps, static_cast<double>(bytes), [&] {
        if (dir == cudaMemcpyHostToDevice) PP_CK(cudaMemcpyAsync(dev.as(), host, bytes, dir, s));
        else PP_CK(cudaMemcpyAsync(host, dev.as(), bytes, dir, s));
      });
      j.kv_obj(mib_key(mib), stat_obj(v));
      if (mib == pass_mib) med_at_pass[std::string(name)] = median(v);
    }
    return j;
  };

  JW h2d, d2h;
  h2d.kv_obj("pageable_GBps", run_dir("h2d_pageable", pageable.p, cudaMemcpyHostToDevice));
  h2d.kv_obj("pinned_GBps", run_dir("h2d_pinned", pin.ptr(), cudaMemcpyHostToDevice));
  h2d.kv_obj("write_combined_GBps", run_dir("h2d_wc", wc.ptr(), cudaMemcpyHostToDevice));
  h2d.kv_obj("registered_GBps", run_dir("h2d_registered", reg.ptr(), cudaMemcpyHostToDevice));
  d2h.kv_obj("pageable_GBps", run_dir("d2h_pageable", pageable.p, cudaMemcpyDeviceToHost));
  d2h.kv_obj("pinned_GBps", run_dir("d2h_pinned", pin.ptr(), cudaMemcpyDeviceToHost));
  d2h.kv_obj("registered_GBps", run_dir("d2h_registered", reg.ptr(), cudaMemcpyDeviceToHost));

  // Zero-copy: kernel reads / writes host memory straight over PCIe.
  JW zc;
  void* dptr = nullptr;
  if (mapped.ptr() && soft_err(cudaHostGetDevicePointer(&dptr, mapped.ptr(), 0)).empty() && dptr) {
    DevBuf sink(sizeof(unsigned long long));
    PP_CK(cudaMemset(sink.as(), 0, sizeof(unsigned long long)));
    const int blocks = gpu_blocks(c, 32);
    JW rd, wr;
    for (const size_t mib : sizes) {
      const size_t bytes = mib << 20;
      rd.kv_obj(mib_key(mib), stat_obj(bw_reps(s, reps, static_cast<double>(bytes), [&] { PP_CK(pk::launch_read(dptr, bytes / 16, sink.as<unsigned long long>(), blocks, 256, s)); })));
      wr.kv_obj(mib_key(mib), stat_obj(bw_reps(s, reps, static_cast<double>(bytes), [&] { PP_CK(pk::launch_write(dptr, bytes / 16, blocks, 256, s)); })));
    }
    zc.kv_obj("kernel_read_GBps", rd).kv_obj("kernel_write_GBps", wr).kv("device_pointer_differs_from_host_pointer", dptr != mapped.ptr());
  } else {
    zc.kv("unavailable", "mapped allocation or cudaHostGetDevicePointer failed");
  }

  // Three streams at once (pinned, host to device), largest size up to 256 MiB.
  JW multi;
  if (pin.ptr()) {
    Stream st[3];
    const size_t part = (std::min<size_t>(pass_mib, sizes.back()) << 20) / 3 / 4096 * 4096;
    std::vector<double> v;
    for (int i = 0; i < reps; ++i) {
      PP_CK(cudaDeviceSynchronize());
      const double t0 = now_s();
      for (size_t k = 0; k < 3; ++k) PP_CK(cudaMemcpyAsync(dev.as<unsigned char>() + k * part, pin.u8() + k * part, part, cudaMemcpyHostToDevice, st[k]));
      for (auto& x : st) x.sync();
      v.push_back(gbps(static_cast<double>(part * 3), (now_s() - t0) * 1000.0));
    }
    multi.kv_obj("three_streams_pinned_h2d_GBps", stat_obj(v)).kv("bytes_per_stream", part);
  }

  // Small-transfer latency (pinned, copy plus stream synchronize).
  JW lat;
  if (pin.ptr()) {
    for (const size_t kib : {size_t{4}, size_t{64}, size_t{1024}}) {
      std::vector<double> us;
      for (int i = 0; i < (c.quick ? 50 : 300); ++i) {
        const double t0 = now_s();
        PP_CK(cudaMemcpyAsync(dev.as(), pin.ptr(), kib << 10, cudaMemcpyHostToDevice, s));
        s.sync();
        us.push_back((now_s() - t0) * 1e6);
      }
      lat.kv_obj("kib_" + std::to_string(kib) + "_microseconds", stat_obj(us));
    }
  }

  r.values.kv_obj("host_to_device", h2d).kv_obj("device_to_host", d2h).kv_obj("zero_copy_kernel", zc).kv_obj("concurrent", multi).kv_obj("small_transfer_latency", lat).kv_obj("alloc_errors", alloc_err);
  r.inputs.kv("sizes_mib_max", maxb >> 20).kv("repetitions", reps).kv("pass_size_mib", pass_mib);
  r.thresholds.kv("pinned_h2d_GBps_min_at_pass_size", 20.0).kv("pcie4_x16_theoretical_GBps_PUBLISHED", 31.5);
  const double pin_h2d = med_at_pass.count("h2d_pinned") ? med_at_pass["h2d_pinned"] : std::nan("");
  const double pin_d2h = med_at_pass.count("d2h_pinned") ? med_at_pass["d2h_pinned"] : std::nan("");
  const double pg_h2d = med_at_pass.count("h2d_pageable") ? med_at_pass["h2d_pageable"] : std::nan("");
  r.pass = (pin_h2d >= 20.0) ? 1 : 0;
  char b[220];
  std::snprintf(b, sizeof b, "at %zu MiB: pinned H2D %.1f / pinned D2H %.1f / pageable H2D %.1f GB/s", pass_mib, pin_h2d, pin_d2h, pg_h2d);
  r.headline = b;
  r.notes = "GB = 1e9 bytes. Medians of repeated timed copies after 2 warm-ups. Registered = ordinary memory plus cudaHostRegister.";
  write_result(c, r, "h5_transfer");
  return r.pass == 1 ? 0 : 3;
}

// ================================================================ P-H6: transfer rate while the CPU streams DRAM
class CpuLoad {
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> prog_{0};
  std::thread th_;

 public:
  CpuLoad() = default;
  CpuLoad(const CpuLoad&) = delete;
  CpuLoad& operator=(const CpuLoad&) = delete;
  ~CpuLoad() { halt(); }
  void start(const unsigned char* base, size_t bytes, int n) {
    halt();
    stop_ = false;
    prog_ = 0;
    th_ = std::thread([this, base, bytes, n] { (void)cpu_read_threads(base, bytes, n, 0, &stop_, &prog_); });
  }
  [[nodiscard]] uint64_t progress() const { return prog_.load(std::memory_order_relaxed); }
  void halt() {
    stop_ = true;
    if (th_.joinable()) th_.join();
  }
};

inline int p_h6(Ctx& c, const Args&) {
  Result r;
  r.probe = "P-H6";
  init_cuda(c);
  if (!cpu_has_avx2()) throw std::runtime_error("CPU has no AVX2: the host read load cannot run");
  const double win = c.quick ? 1.0 : 3.0;
  const size_t cb = (c.quick ? size_t{1} : size_t{4}) << 30;
  const size_t xfer = size_t{256} << 20;
  AlignedBuf cpu_buf(4096, cb);
  if (!cpu_buf.p) throw std::runtime_error("host allocation for the CPU load buffer failed");
  fill_pattern(cpu_buf.p, cb, 11);
  DevBuf dev(xfer), sink(sizeof(unsigned long long));
  PP_CK(cudaMemset(sink.as(), 0, sizeof(unsigned long long)));
  PinnedBuf pin, mapped;
  PP_CK(pin.try_alloc(xfer));
  PP_CK(mapped.try_alloc(xfer, cudaHostAllocMapped));
  void* dptr = nullptr;
  PP_CK(cudaHostGetDevicePointer(&dptr, mapped.ptr(), 0));
  fill_pattern(pin.ptr(), xfer, 5);
  fill_pattern(mapped.ptr(), xfer, 6);
  Stream s;
  const int blocks = gpu_blocks(c, 32);
  CpuLoad load;

  // Runs transfers for `seconds`; returns their GB/s and (through cpu_rate) what the readers achieved meanwhile.
  auto run_xfer = [&](bool zero_copy, double seconds, double* cpu_rate) -> double {
    uint64_t moved = 0;
    const uint64_t p0 = load.progress();
    const double t0 = now_s();
    while (now_s() - t0 < seconds && !g_abort.load()) {
      if (zero_copy) PP_CK(pk::launch_read(dptr, xfer / 16, sink.as<unsigned long long>(), blocks, 256, s));
      else PP_CK(cudaMemcpyAsync(dev.as(), pin.ptr(), xfer, cudaMemcpyHostToDevice, s));
      s.sync();
      moved += xfer;
    }
    const double t1 = now_s();
    const uint64_t p1 = load.progress();
    if (cpu_rate) *cpu_rate = t1 > t0 ? static_cast<double>(p1 - p0) / (t1 - t0) / 1e9 : 0.0;
    return t1 > t0 ? static_cast<double>(moved) / (t1 - t0) / 1e9 : 0.0;
  };
  auto settle = [&] { std::this_thread::sleep_for(milliseconds(500)); };

  std::vector<JW> rows;
  double base_h2d = 0, base_zc = 0;
  double worst_h2d_ratio = 1.0;
  for (const int nt : {0, 4, 8, 16}) {
    double cpu_alone = 0, cpu_h = 0, cpu_z = 0;
    if (nt > 0) {
      load.start(cpu_buf.u8(), cb, nt);
      settle();
      const uint64_t p0 = load.progress();
      const double t0 = now_s();
      std::this_thread::sleep_for(std::chrono::duration<double>(win));
      const double t1 = now_s();
      cpu_alone = static_cast<double>(load.progress() - p0) / (t1 - t0) / 1e9;
      load.halt();
      load.start(cpu_buf.u8(), cb, nt);
      settle();
    }
    const double h = run_xfer(false, win, &cpu_h);
    if (nt > 0) {
      load.halt();
      load.start(cpu_buf.u8(), cb, nt);
      settle();
    }
    const double z = run_xfer(true, win, &cpu_z);
    load.halt();
    if (nt == 0) { base_h2d = h; base_zc = z; }
    JW row;
    row.kv("reader_threads", nt).kv("h2d_pinned_GBps", h).kv("zero_copy_read_GBps", z);
    if (nt > 0) {
      row.kv("cpu_read_alone_GBps", cpu_alone).kv("cpu_read_during_h2d_GBps", cpu_h).kv("cpu_read_during_zero_copy_GBps", cpu_z);
      row.kv("h2d_slowdown_factor", base_h2d > 0 ? h / base_h2d : std::nan(""));
      row.kv("zero_copy_slowdown_factor", base_zc > 0 ? z / base_zc : std::nan(""));
      row.kv("cpu_slowdown_factor_during_h2d", cpu_alone > 0 ? cpu_h / cpu_alone : std::nan(""));
      row.kv("cpu_slowdown_factor_during_zero_copy", cpu_alone > 0 ? cpu_z / cpu_alone : std::nan(""));
      row.kv("combined_cpu_plus_h2d_GBps", cpu_h + h);
      if (base_h2d > 0) worst_h2d_ratio = std::min(worst_h2d_ratio, h / base_h2d);
    }
    rows.push_back(row);
    std::printf("  readers=%2d  h2d %.1f GB/s  zero-copy %.1f GB/s  cpu alone %.1f / during h2d %.1f GB/s\n", nt, h, z, cpu_alone, cpu_h);
    std::fflush(stdout);
    if (g_abort.load()) break;
  }
  r.values.arr_obj("rows", rows).kv("worst_h2d_over_unloaded", worst_h2d_ratio);
  r.inputs.kv("window_seconds", win).kv("cpu_buffer_gib", static_cast<double>(cb) / kGiB).kv("transfer_bytes", xfer);
  r.thresholds.kv("engine_prior_cpu_alone_GBps_MEASURED", 45.9).kv("engine_prior_cpu_plus_transfer_GBps_MEASURED", 47.6);
  char b[200];
  std::snprintf(b, sizeof b, "H2D %.1f GB/s unloaded; worst loaded/unloaded ratio %.2f", base_h2d, worst_h2d_ratio);
  r.headline = b;
  r.notes = "Record only: shows how much the host-DRAM readers and the PCIe copies slow each other. Compare the combined column with the engine's prior numbers.";
  write_result(c, r, "h6_contention");
  return 0;
}

// ================================================================ P-H12: do copy engines and compute overlap?
// Every combination of {H2D copies, D2H copies, compute} is timed alone and together, for three kinds of compute
// (one long tensor launch / many short tensor launches / many short VRAM-read launches) and four ways of submitting
// the work. Efficiency = (sum of the parts - time together) / (sum of the parts - longest part):
// 1 = perfect overlap, 0 = fully serial. The GPU-side start and end of each stream's work is also recorded with events,
// so "did they really run at the same time" does not rest on host timing alone.
inline int p_h12(Ctx& c, const Args&) {
  Result r;
  r.probe = "P-H12";
  init_cuda(c);
  const size_t chunk = (c.quick ? size_t{64} : size_t{256}) << 20;
  const double target_ms = c.quick ? 50.0 : 100.0;
  const int nrep = c.quick ? 3 : 7;
  PinnedBuf hin, hout;
  PP_CK(hin.try_alloc(chunk));
  PP_CK(hout.try_alloc(chunk));
  fill_pattern(hin.ptr(), chunk, 1);
  DevBuf din(chunk), dout(chunk);
  const size_t vbytes = (c.quick ? size_t{256} : size_t{1024}) << 20;
  DevBuf vbuf(vbytes), vsink(sizeof(unsigned long long));
  PP_CK(cudaMemset(dout.as(), 0, chunk));
  PP_CK(cudaMemset(vbuf.as(), 0x11, vbytes));
  PP_CK(cudaMemset(vsink.as(), 0, sizeof(unsigned long long)));
  const int blocks = gpu_blocks(c, 4);
  DevBuf msink(static_cast<size_t>(blocks) * pk::kMmaThreads * sizeof(float));
  Stream sh, sd, sc;
  const int vblocks = gpu_blocks(c, 32);
  int async_engines = -1;
  (void)cudaDeviceGetAttribute(&async_engines, cudaDevAttrAsyncEngineCount, 0);

  auto h2d_one = [&] { PP_CK(cudaMemcpyAsync(din.as(), hin.ptr(), chunk, cudaMemcpyHostToDevice, sh)); };
  auto d2h_one = [&] { PP_CK(cudaMemcpyAsync(hout.ptr(), dout.as(), chunk, cudaMemcpyDeviceToHost, sd)); };
  auto vread_one = [&] { PP_CK(pk::launch_read(vbuf.as(), vbytes / 16, vsink.as<unsigned long long>(), vblocks, 256, sc)); };
  auto calib_reps = [&](const std::function<void()>& one) -> int {
    one();
    PP_CK(cudaDeviceSynchronize());
    const double t0 = now_s();
    one();
    one();
    PP_CK(cudaDeviceSynchronize());
    const double ms = (now_s() - t0) * 1000.0 / 2.0;
    return std::max(1, static_cast<int>(std::ceil(target_ms / std::max(ms, 1e-3))));
  };
  const int n_h = calib_reps(h2d_one), n_d = calib_reps(d2h_one), n_v = calib_reps(vread_one);
  const unsigned iters_long = calibrate_mma_iters(pk::Mma::BF16, blocks, msink.as<float>(), sc, target_ms);
  const unsigned iters_short = calibrate_mma_iters(pk::Mma::BF16, blocks, msink.as<float>(), sc, 0.5);
  const double short_ms = time_stream_ms(sc, [&] {
                            for (int i = 0; i < 20; ++i) PP_CK(pk::launch_mma(pk::Mma::BF16, blocks, iters_short, msink.as<float>(), sc));
                          }) / 20.0;
  const int n_ts = std::max(1, static_cast<int>(std::ceil(target_ms / std::max(short_ms, 1e-3))));

  enum class CK { TensorLong = 0, TensorShort = 1, VramShort = 2 };
  struct Pat {
    const char* name;
    bool compute_first;  // enqueue the compute before the copies
    bool interleave;     // alternate H2D and D2H copies instead of all H2D then all D2H
    bool flush;          // cudaStreamQuery after each enqueue (pushes batched work to the GPU on WDDM-style stacks)
  };
  static const Pat kPats[4] = {
      {"default_copies_first", false, false, false},
      {"compute_first", true, false, false},
      {"compute_first_interleaved", true, true, false},
      {"compute_first_interleaved_flushed", true, true, true},
  };
  static const char* const kKindName[3] = {"tensor_one_long_launch", "tensor_many_short_launches", "vram_read_many_short_launches"};
  static const char* const kName[8] = {"none", "h2d", "d2h", "h2d+d2h", "compute", "h2d+compute", "d2h+compute", "h2d+d2h+compute"};

  Event ev_s[3], ev_e[3];  // 0 = H2D stream, 1 = D2H stream, 2 = compute stream
  // mask bits: 1 = host-to-device copies, 2 = device-to-host copies, 4 = compute.
  auto run_set = [&](int mask, CK kind, const Pat& pat, JW* timeline) -> double {
    PP_CK(cudaDeviceSynchronize());
    const bool tl = timeline != nullptr;
    const double t0 = now_s();
    auto flush = [&](cudaStream_t s) {
      if (pat.flush) {
        (void)cudaStreamQuery(s);  // "not ready" is the normal answer here
        (void)cudaGetLastError();
      }
    };
    auto do_compute = [&] {
      if (!(mask & 4)) return;
      if (tl) ev_s[2].record(sc);
      switch (kind) {
        case CK::TensorLong: PP_CK(pk::launch_mma(pk::Mma::BF16, blocks, iters_long, msink.as<float>(), sc)); break;
        case CK::TensorShort:
          for (int i = 0; i < n_ts; ++i) PP_CK(pk::launch_mma(pk::Mma::BF16, blocks, iters_short, msink.as<float>(), sc));
          break;
        case CK::VramShort:
          for (int i = 0; i < n_v; ++i) vread_one();
          break;
      }
      if (tl) ev_e[2].record(sc);
      flush(sc);
    };
    auto do_copies = [&] {
      int ih = (mask & 1) ? n_h : 0, id = (mask & 2) ? n_d : 0;
      if (tl) {
        if (ih) ev_s[0].record(sh);
        if (id) ev_s[1].record(sd);
      }
      while (ih > 0 || id > 0) {
        if (pat.interleave) {
          if (ih > 0) { h2d_one(); --ih; flush(sh); }
          if (id > 0) { d2h_one(); --id; flush(sd); }
        } else if (ih > 0) {
          h2d_one();
          --ih;
        } else {
          d2h_one();
          --id;
        }
      }
      if (tl) {
        if (mask & 1) ev_e[0].record(sh);
        if (mask & 2) ev_e[1].record(sd);
      }
      flush(sh);
      flush(sd);
    };
    if (pat.compute_first) { do_compute(); do_copies(); } else { do_copies(); do_compute(); }
    if (mask & 1) sh.sync();
    if (mask & 2) sd.sync();
    if (mask & 4) sc.sync();
    const double ms = (now_s() - t0) * 1000.0;
    if (tl) {
      // Start/end of each active stream on the GPU clock, relative to the earliest start.
      double s[3] = {0, 0, 0}, e[3] = {0, 0, 0};
      bool act[3] = {(mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0};
      int ref = -1;
      for (int i = 0; i < 3; ++i)
        if (act[i]) { ref = i; break; }
      double lo = 0;
      for (int i = 0; i < 3; ++i)
        if (act[i]) {
          s[i] = Event::ms(ev_s[ref], ev_s[i]);
          e[i] = Event::ms(ev_s[ref], ev_e[i]);
          lo = std::min(lo, s[i]);
        }
      for (int i = 0; i < 3; ++i) { s[i] -= lo; e[i] -= lo; }
      static const char* const nm[3] = {"h2d", "d2h", "compute"};
      for (int i = 0; i < 3; ++i)
        if (act[i]) timeline->arr_num(nm[i], {s[i], e[i]});
      auto ov = [&](int a, int b) { return act[a] && act[b] ? std::max(0.0, std::min(e[a], e[b]) - std::max(s[a], s[b])) : std::nan(""); };
      timeline->kv("overlap_ms_h2d_d2h", ov(0, 1)).kv("overlap_ms_h2d_compute", ov(0, 2)).kv("overlap_ms_d2h_compute", ov(1, 2));
    }
    return ms;
  };

  JW kinds_j;
  double eff_short_default = std::nan(""), eff_short_best = std::nan(""), eff_long_default = std::nan(""), eff_vram_default = std::nan("");
  double eff_pair_copies_best = std::nan("");
  std::string best_pat = "none";
  for (int ki = 0; ki < 3; ++ki) {
    const CK kind = static_cast<CK>(ki);
    const int npat = (kind == CK::VramShort) ? 1 : 4;  // the variants are only needed for the tensor cases
    JW pats_j;
    for (int pi = 0; pi < npat; ++pi) {
      const Pat& pat = kPats[pi];
      std::vector<double> t(8, 0.0);
      for (int mask = 1; mask < 8; ++mask) {
        run_set(mask, kind, pat, nullptr);  // warm-up
        std::vector<double> v;
        for (int i = 0; i < nrep; ++i) v.push_back(run_set(mask, kind, pat, nullptr));
        t[static_cast<size_t>(mask)] = median(v);
      }
      JW tj, ej;
      for (int mask = 1; mask < 8; ++mask) tj.kv(kName[mask], t[static_cast<size_t>(mask)]);
      double eff7 = std::nan(""), eff3 = std::nan("");
      for (int mask = 3; mask < 8; ++mask) {
        if ((mask & (mask - 1)) == 0) continue;  // single op
        double sum = 0, mx = 0;
        for (int bit = 1; bit < 8; bit <<= 1)
          if (mask & bit) {
            sum += t[static_cast<size_t>(bit)];
            mx = std::max(mx, t[static_cast<size_t>(bit)]);
          }
        const double eff = (sum - mx) > 1e-9 ? (sum - t[static_cast<size_t>(mask)]) / (sum - mx) : std::nan("");
        ej.kv(kName[mask], eff);
        if (mask == 7) eff7 = eff;
        if (mask == 3) eff3 = eff;
      }
      JW tlj;
      (void)run_set(7, kind, pat, &tlj);  // one more all-three run, with GPU-side timestamps
      JW pj;
      pj.kv_obj("median_ms_per_set", tj).kv_obj("overlap_efficiency", ej).kv_obj("gpu_timeline_ms_one_all_three_run", tlj);
      pats_j.kv_obj(pat.name, pj);
      if (pi == 0 && kind == CK::TensorShort) eff_short_default = eff7;
      if (pi == 0 && kind == CK::TensorLong) eff_long_default = eff7;
      if (pi == 0 && kind == CK::VramShort) eff_vram_default = eff7;
      if (kind == CK::TensorShort && !(eff7 <= eff_short_best)) {  // also true while eff_short_best is NaN
        eff_short_best = eff7;
        best_pat = pat.name;
      }
      if (!(eff3 <= eff_pair_copies_best)) eff_pair_copies_best = eff3;
    }
    kinds_j.kv_obj(kKindName[ki], pats_j);
  }
  r.values.kv_obj("sets", kinds_j)
      .kv("async_engine_count_reported_by_driver", async_engines)
      .kv("copy_chunk_bytes", chunk)
      .kv("h2d_repeats", n_h)
      .kv("d2h_repeats", n_d)
      .kv("vram_read_launches", n_v)
      .kv("tensor_long_iters", iters_long)
      .kv("tensor_short_iters", iters_short)
      .kv("tensor_short_launches", n_ts)
      .kv("tensor_short_ms_each", short_ms)
      .kv("best_pattern_for_tensor_short", best_pat)
      .kv("h2d_d2h_pair_efficiency_best_seen", eff_pair_copies_best);
  r.inputs.kv("target_ms_per_op", target_ms).kv("repetitions", nrep);
  r.thresholds.kv("all_three_overlap_efficiency_min", 0.90)
      .kv("judged_on", "many short tensor launches, best of the submission patterns tried")
      .kv("meaning", "1 = perfect overlap, 0 = fully serial: (sum - t_set) / (sum - max)");
  r.pass = (eff_short_best >= 0.90) ? 1 : 0;
  char b[330];
  std::snprintf(b, sizeof b,
                "all-three overlap: tensor many-short default %.2f best %.2f (%s); tensor one-long default %.2f; vram-read default %.2f; "
                "copy engines reported %d; best H2D+D2H %.2f",
                eff_short_default, eff_short_best, best_pat.c_str(), eff_long_default, eff_vram_default, async_engines, eff_pair_copies_best);
  r.headline = b;
  r.notes = "Pass is judged on the tensor case with many short launches (how the trainer will submit work), best submission pattern. "
            "The patterns say how to submit work so it overlaps; the GPU timeline shows whether the streams really ran at the same time.";
  write_result(c, r, "h12_overlap");
  return r.pass == 1 ? 0 : 3;
}

// ================================================================ P-H7: how much pinned memory can be held?
inline int p_h7(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-H7";
  init_cuda(c);
  if (!cpu_has_avx2()) throw std::runtime_error("CPU has no AVX2: the checksum routine needs it");
  const size_t target = static_cast<size_t>(qnum(c, a, "max-gib", 6, 32));
  const double stop_avail_gib = 4.0;
  const size_t gib = size_t{1} << 30;
  write_started(c, "P-H7", "h7_pinned_limit", "growing cudaHostAlloc in 1 GiB steps up to " + std::to_string(target) + " GiB");

  const HostMem m0 = host_mem();
  std::vector<PinnedBuf> blocks;
  std::vector<uint64_t> sums;
  std::vector<JW> steps;
  std::string stop_reason = "reached_target";
  std::string alloc_error;
  double max_alloc_s = 0;
  std::vector<double> avail_drop;
  for (size_t g = 1; g <= target && !g_abort.load(); ++g) {
    const HostMem before = host_mem();
    if (before.avail_gib() < stop_avail_gib + 1.0) {
      stop_reason = "safety_stop: host MemAvailable below 5 GiB, not allocating the next GiB";
      break;
    }
    PinnedBuf b;
    const double t0 = now_s();
    const cudaError_t e = b.try_alloc(gib);
    const double t_alloc = now_s() - t0;
    if (e != cudaSuccess) {
      alloc_error = soft_err(e);
      stop_reason = "cudaHostAlloc failed at " + std::to_string(g) + " GiB";
      break;
    }
    fill_pattern(b.ptr(), gib, g);
    sums.push_back(cpu_read_sum_avx2(b.ptr(), gib));
    const HostMem after = host_mem();
    max_alloc_s = std::max(max_alloc_s, t_alloc);
    avail_drop.push_back(before.avail_gib() - after.avail_gib());
    JW st;
    st.kv("pinned_gib", g).kv("alloc_seconds", t_alloc).kv("host_avail_gib", after.avail_gib()).kv("swap_free_mib", static_cast<double>(after.swap_free_kib) / 1024.0).kv("swapin_pages_since_start", after.swapin_pages - m0.swapin_pages);
    steps.push_back(st);
    std::printf("  pinned %2zu GiB ok  alloc %.2f s  host avail %.1f GiB\n", g, t_alloc, after.avail_gib());
    std::fflush(stdout);
    blocks.push_back(std::move(b));
  }

  // Integrity: every block still holds what was written, and a DMA round trip through the GPU is intact.
  size_t bad = 0;
  for (size_t i = 0; i < blocks.size(); ++i)
    if (cpu_read_sum_avx2(blocks[i].ptr(), gib) != sums[i]) ++bad;
  JW rt;
  std::string rt_note;
  if (!blocks.empty() && host_mem().avail_gib() > stop_avail_gib + 0.5) {
    const size_t piece = size_t{256} << 20;
    DevBuf dev(piece);
    AlignedBuf back(4096, piece);
    if (back.p) {
      for (const size_t idx : {size_t{0}, blocks.size() - 1}) {
        const unsigned char* src = blocks[idx].u8() + (idx == 0 ? 0 : gib - piece);  // first block: its start; last block: its end
        const uint64_t want = cpu_read_sum_avx2(src, piece);
        PP_CK(cudaMemcpy(dev.as(), src, piece, cudaMemcpyHostToDevice));
        std::memset(back.p, 0, piece);
        PP_CK(cudaMemcpy(back.p, dev.as(), piece, cudaMemcpyDeviceToHost));
        const bool ok = cpu_read_sum_avx2(back.p, piece) == want;
        rt.kv(idx == 0 ? "first_block_roundtrip_ok" : "last_block_roundtrip_ok", ok);
        if (!ok) ++bad;
      }
    } else {
      rt_note = "round trip skipped: could not allocate the 256 MiB pageable check buffer. ";
    }
  } else {
    rt_note = "round trip skipped: not enough free host memory or no blocks. ";
  }
  const HostMem m1 = host_mem();
  const double swapin_mib = static_cast<double>(m1.swapin_pages - m0.swapin_pages) * 4096.0 / kMiB;
  const size_t reached = blocks.size();
  const double lock_mib = static_cast<double>(memlock_limit_bytes()) / kMiB;

  r.values.kv("reached_gib", reached).kv("target_gib", target).kv("stop_reason", stop_reason).kv("alloc_error", alloc_error).kv("max_single_gib_alloc_seconds", max_alloc_s);
  r.values.kv("integrity_failures", bad).kv_obj("dma_roundtrip", rt).kv("swapin_mib_during_run", swapin_mib);
  r.values.kv("host_total_gib", m0.total_gib()).kv("host_avail_gib_at_start", m0.avail_gib()).kv("host_avail_gib_at_end", m1.avail_gib());
  r.values.kv("median_avail_drop_per_pinned_gib", median(avail_drop)).kv("rlimit_memlock_mib_0_means_unlimited", lock_mib).arr_obj("steps", steps);
  r.inputs.kv("step_gib", 1).kv("safety_stop_host_avail_gib", stop_avail_gib);
  r.thresholds.kv("reached_gib_min", 24).kv("swapin_mib_max", 100).kv("integrity_failures_max", 0);
  if (c.quick) r.pass = -1;
  else r.pass = (reached >= 24 && swapin_mib < 100.0 && bad == 0) ? 1 : 0;
  char b[220];
  std::snprintf(b, sizeof b, "held %zu GiB pinned (target %zu), swap-in %.0f MiB, integrity failures %zu; %s", reached, target, swapin_mib, bad, stop_reason.c_str());
  r.headline = b;
  r.notes = rt_note + "median_avail_drop_per_pinned_gib near 1 means the pages are really held in this OS; near 0 means they are accounted elsewhere (e.g. by the Windows host under WSL2). " + (c.quick ? "Quick mode: record only, not a pass/fail." : "");
  write_result(c, r, "h7_pinned_limit");
  return (c.quick || r.pass == 1) ? 0 : 3;
}

// ================================================================ P-M3: hold pinned memory for a long time
class Jitter {
  std::atomic<bool> run_{false};
  std::thread th_;
  std::vector<double> over_ms_;

 public:
  Jitter() = default;
  Jitter(const Jitter&) = delete;
  Jitter& operator=(const Jitter&) = delete;
  ~Jitter() { (void)stop(); }
  void start() {
    run_ = true;
    th_ = std::thread([this] {
      while (run_.load()) {
        const double t0 = now_s();
        std::this_thread::sleep_for(milliseconds(10));
        over_ms_.push_back(std::max(0.0, (now_s() - t0) * 1000.0 - 10.0));
      }
    });
  }
  std::vector<double> stop() {
    run_ = false;
    if (th_.joinable()) th_.join();
    return over_ms_;
  }
};

inline int p_m3(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-M3";
  init_cuda(c);
  if (!cpu_has_avx2()) throw std::runtime_error("CPU has no AVX2: the checksum routine needs it");
  const size_t pinned_gib = static_cast<size_t>(qnum(c, a, "pinned-gib", 6, 24));
  const size_t pageable_gib = static_cast<size_t>(qnum(c, a, "pageable-gib", 2, 4));
  const double minutes = qnum(c, a, "minutes", 1.0, 30.0);
  const size_t gib = size_t{1} << 30;
  const double need = static_cast<double>(pinned_gib + pageable_gib) + 4.0;
  write_started(c, "P-M3", "m3_hold", "holding " + std::to_string(pinned_gib) + " GiB pinned + " + std::to_string(pageable_gib) + " GiB pageable for " + fmt_g(minutes) + " min");
  const HostMem m0 = host_mem();
  if (m0.avail_gib() < need) {
    r.status = "skipped";
    r.error = "not enough free host memory: need " + std::to_string(need) + " GiB (hold + 4 GiB margin), available " + std::to_string(m0.avail_gib()) + " GiB";
    r.headline = r.error;
    write_result(c, r, "m3_hold");
    return 0;
  }
  std::vector<PinnedBuf> pin;
  std::vector<uint64_t> pin_sum;
  for (size_t g = 0; g < pinned_gib; ++g) {
    PinnedBuf b;
    const cudaError_t e = b.try_alloc(gib);
    if (e != cudaSuccess) {
      r.status = "error";
      r.error = "cudaHostAlloc failed at " + std::to_string(g + 1) + " GiB: " + soft_err(e);
      break;
    }
    fill_pattern(b.ptr(), gib, 100 + g);
    pin_sum.push_back(cpu_read_sum_avx2(b.ptr(), gib));
    pin.push_back(std::move(b));
  }
  std::vector<AlignedBuf> pg;
  std::vector<uint64_t> pg_sum;
  for (size_t g = 0; g < pageable_gib && r.status == "ok"; ++g) {
    AlignedBuf b(4096, gib);
    if (!b.p) {
      r.status = "error";
      r.error = "pageable allocation failed at " + std::to_string(g + 1) + " GiB";
      break;
    }
    fill_pattern(b.p, gib, 200 + g);
    pg_sum.push_back(cpu_read_sum_avx2(b.p, gib));
    pg.push_back(std::move(b));
  }
  if (r.status != "ok") {
    r.headline = r.error;
    write_result(c, r, "m3_hold");
    return 3;
  }

  const size_t xfer = size_t{256} << 20;
  DevBuf dev(xfer);
  Stream s;
  Jitter jit;
  jit.start();
  const HostMem m_hold = host_mem();
  const double t_ref = now_s(), t_end = t_ref + minutes * 60.0, cycle_s = 10.0;
  size_t cycles = 0, mismatches = 0, rot = 0;
  std::string stop_reason = "completed";
  std::vector<double> ts, avail, swapin, h2d;
  std::vector<double> all_h2d;
  while (now_s() < t_end && !g_abort.load()) {
    const double tc = now_s();
    for (size_t i = 0; i < pg.size(); ++i)
      if (cpu_read_sum_avx2(pg[i].p, gib) != pg_sum[i]) ++mismatches;
    for (size_t k = 0; k < std::min<size_t>(4, pin.size()); ++k) {
      const size_t i = (rot + k) % pin.size();
      if (cpu_read_sum_avx2(pin[i].ptr(), gib) != pin_sum[i]) ++mismatches;
    }
    double gbps_h2d = std::nan("");
    if (!pin.empty()) {
      const size_t i = rot % pin.size();
      const size_t off = ((cycles * xfer) % gib);
      const double t0 = now_s();
      PP_CK(cudaMemcpyAsync(dev.as(), pin[i].u8() + off, std::min(xfer, gib - off), cudaMemcpyHostToDevice, s));
      s.sync();
      gbps_h2d = gbps(static_cast<double>(std::min(xfer, gib - off)), (now_s() - t0) * 1000.0);
      all_h2d.push_back(gbps_h2d);
    }
    ++rot;
    ++cycles;
    const HostMem hm = host_mem();
    ts.push_back(now_s() - t_ref);
    avail.push_back(hm.avail_gib());
    swapin.push_back(static_cast<double>(hm.swapin_pages - m_hold.swapin_pages) * 4096.0 / kMiB);
    h2d.push_back(gbps_h2d);
    if (hm.avail_gib() < 2.0) {
      stop_reason = "safety_stop: host MemAvailable below 2 GiB while holding";
      break;
    }
    while (now_s() - tc < cycle_s && now_s() < t_end && !g_abort.load()) std::this_thread::sleep_for(milliseconds(100));
  }
  const double held_s = now_s() - t_ref;
  const std::vector<double> over = jit.stop();
  // final full verification
  size_t final_bad = 0;
  for (size_t i = 0; i < pin.size(); ++i)
    if (cpu_read_sum_avx2(pin[i].ptr(), gib) != pin_sum[i]) ++final_bad;
  for (size_t i = 0; i < pg.size(); ++i)
    if (cpu_read_sum_avx2(pg[i].p, gib) != pg_sum[i]) ++final_bad;
  const HostMem m1 = host_mem();
  const double swapin_mib = static_cast<double>(m1.swapin_pages - m_hold.swapin_pages) * 4096.0 / kMiB;
  size_t stalls = 0;
  for (const double x : over)
    if (x > 50.0) ++stalls;
  JW jj;
  jj.kv("samples", over.size()).kv("median_ms", median(over)).kv("p99_ms", percentile(over, 99.0)).kv("p999_ms", percentile(over, 99.9)).kv("max_ms", vmax(over)).kv("stalls_over_50ms", stalls);
  const size_t step = std::max<size_t>(1, ts.size() / 300);
  std::vector<double> tsd, avd, swd, hd;
  for (size_t i = 0; i < ts.size(); i += step) { tsd.push_back(ts[i]); avd.push_back(avail[i]); swd.push_back(swapin[i]); hd.push_back(h2d[i]); }
  JW ser;
  ser.arr_num("t_s", tsd).arr_num("host_avail_gib", avd).arr_num("swapin_mib", swd).arr_num("h2d_GBps", hd);
  const bool completed = held_s >= minutes * 60.0 - cycle_s;
  r.values.kv("held_seconds", held_s).kv("cycles", cycles).kv("checksum_mismatches_in_loop", mismatches).kv("checksum_mismatches_final", final_bad).kv("swapin_mib_while_holding", swapin_mib);
  r.values.kv("host_avail_gib_at_hold_start", m_hold.avail_gib()).kv("host_avail_gib_at_end", m1.avail_gib()).kv("stop_reason", stop_reason).kv_obj("heartbeat_jitter_10ms_sleep", jj);
  r.values.kv_obj("h2d_pinned_GBps_during_hold", stat_obj(all_h2d)).kv_obj("series", ser);
  r.inputs.kv("pinned_gib", pinned_gib).kv("pageable_gib", pageable_gib).kv("minutes", minutes).kv("cycle_seconds", cycle_s);
  r.thresholds.kv("checksum_mismatches_max", 0).kv("swapin_mib_max", 100).kv("jitter_threshold", "none yet: unmeasured; compare with an idle run (--pinned-gib 0 --pageable-gib 0)");
  if (c.quick) r.pass = -1;
  else r.pass = (mismatches == 0 && final_bad == 0 && swapin_mib < 100.0 && completed && stop_reason == "completed") ? 1 : 0;
  char b[220];
  std::snprintf(b, sizeof b, "held %zu+%zu GiB for %.1f min, mismatches %zu, swap-in %.0f MiB, jitter p99 %.1f ms", pinned_gib, pageable_gib, held_s / 60.0, mismatches + final_bad, swapin_mib, percentile(over, 99.0));
  r.headline = b;
  r.notes = "Windows timers tick at 15.6 ms by default, so compare jitter with an idle run of the same probe on the same platform. " + std::string(c.quick ? "Quick mode: record only." : "") + (g_abort.load() ? " ABORTED: " + abort_reason() : std::string());
  write_result(c, r, "m3_hold");
  return (c.quick || r.pass == 1) ? 0 : 3;
}

// ================================================================ P-M4: reading the model file
inline int p_m4(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-M4";
  std::string path = a.get("gguf");
  if (path.empty()) {
    const char* e = std::getenv("PELIAS_GGUF");
    if (e) path = e;
  }
  std::error_code ec;
  if (path.empty() || !std::filesystem::is_regular_file(path, ec)) {
    r.status = "skipped";
    r.error = "no model file given: set PELIAS_GGUF or pass --gguf PATH (the path is never written to the results)";
    r.headline = r.error;
    r.notes = "Nothing was read.";
    write_result(c, r, "m4_modelfile");
    return 0;
  }
  const uint64_t size = static_cast<uint64_t>(std::filesystem::file_size(path, ec));
  const FsInfo fi = fs_info(path);
  const uint64_t limit = c.quick ? (uint64_t{4} << 30) : 0;
  const bool root = is_root();
  const bool dropped = drop_caches_if_root();
  struct Case { const char* name; size_t chunk; bool mmap; bool cold; };
  const Case cases[] = {
      {"read_1MiB_cold", size_t{1} << 20, false, true},     {"read_1MiB_warm", size_t{1} << 20, false, false},
      {"read_16MiB_cold", size_t{16} << 20, false, true},   {"read_16MiB_warm", size_t{16} << 20, false, false},
      {"mmap_cold", 0, true, true},                         {"mmap_warm", 0, true, false},
  };
  JW res;
  std::vector<std::string> errs;
  std::set<uint64_t> sums;
  double best_cold = 0;
  for (const auto& k : cases) {
    if (!k.cold) (void)(k.mmap ? read_file_mmap(path, limit, false) : read_file_seq(path, k.chunk, limit, false));  // untimed priming pass: makes "warm" really warm (Windows cold reads bypass the cache)
    const ReadResult rr = k.mmap ? read_file_mmap(path, limit, k.cold) : read_file_seq(path, k.chunk, limit, k.cold);
    JW j;
    j.kv("ok", rr.ok).kv("GBps", rr.gbps()).kv("seconds", rr.secs).kv("bytes", rr.bytes);
    if (!rr.ok) errs.push_back(std::string(k.name) + ": " + rr.err);
    else sums.insert(rr.sum);
    res.kv_obj(k.name, j);
    if (rr.ok && k.cold && !k.mmap) best_cold = std::max(best_cold, rr.gbps());
    std::printf("  %-16s %.2f GB/s\n", k.name, rr.gbps());
    std::fflush(stdout);
  }
  r.inputs.kv("file", "test-model").kv("size_bytes", size).kv("fs_type", fi.fstype).kv("mount_device", fi.device).kv("mount_point", fi.mount).kv("limit_bytes_0_is_whole_file", limit);
  r.values.kv_obj("cases", res).kv("checksum_variants_across_cases_expected_1", sums.size()).kv("best_cold_read_GBps", best_cold).kv("page_cache_dropped_by_root", dropped).kv("running_as_root", root);
  r.raw = errs;
  r.thresholds.kv("earlier_observed_cold_GBps_unattributed", "1.9-2.1 (drive not identified)");
  char b[200];
  std::snprintf(b, sizeof b, "model file %.1f GiB on %s (%s): best cold read %.2f GB/s", static_cast<double>(size) / kGiB, fi.device.c_str(), fi.fstype.c_str(), best_cold);
  r.headline = b;
  r.notes = "Record only. 'Cold' asks the OS to drop this file's cached pages first (Linux fadvise / Windows unbuffered I/O); on /mnt/c under WSL2 the Windows file cache is outside WSL's control, so treat that as warm-ish. The file name is not recorded.";
  write_result(c, r, "m4_modelfile");
  return 0;
}

// ================================================================ P-H9: CUDA graph vs eager launches
inline int p_h9(Ctx& c, const Args&) {
  Result r;
  r.probe = "P-H9";
  init_cuda(c);
  constexpr int kn = 100;
  const int eager_iters = c.quick ? 200 : 2000;
  const int replays = c.quick ? 5000 : 100000;
  const double cap_s = c.quick ? 30.0 : 150.0;
  DevBuf ctr(sizeof(unsigned long long));
  Stream s;
  unsigned long long per_iter = 0;
  std::vector<unsigned> inc(kn);
  for (int k = 0; k < kn; ++k) { inc[static_cast<size_t>(k)] = 1u + static_cast<unsigned>(k % 3); per_iter += inc[static_cast<size_t>(k)]; }
  auto reset = [&] {
    PP_CK(cudaMemset(ctr.as(), 0, sizeof(unsigned long long)));
    PP_CK(cudaDeviceSynchronize());
  };
  auto read_ctr = [&]() -> unsigned long long {
    unsigned long long v = 0;
    PP_CK(cudaMemcpy(&v, ctr.as(), sizeof v, cudaMemcpyDeviceToHost));
    return v;
  };
  auto enqueue_set = [&](cudaStream_t st) {
    for (int k = 0; k < kn; ++k) PP_CK(pk::launch_bump(ctr.as<unsigned long long>(), inc[static_cast<size_t>(k)], st));
  };

  // Eager launches.
  for (int i = 0; i < 5; ++i) enqueue_set(s);
  s.sync();
  reset();
  int e_done = 0;
  const double te0 = now_s();
  for (; e_done < eager_iters && now_s() - te0 < cap_s; ++e_done) {
    enqueue_set(s);
    if ((e_done & 63) == 63) s.sync();
  }
  s.sync();
  const double te = now_s() - te0;
  const unsigned long long eager_count = read_ctr();
  const bool eager_exact = eager_count == per_iter * static_cast<unsigned long long>(e_done);

  // Graph: capture once, replay many times.
  cudaGraph_t g = nullptr;
  cudaGraphExec_t ex = nullptr;
  PP_CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  enqueue_set(s);
  PP_CK(cudaStreamEndCapture(s, &g));
  PP_CK(cudaGraphInstantiate(&ex, g, 0ULL));
  PP_CK(cudaGraphLaunch(ex, s));
  s.sync();
  reset();
  int g_done = 0;
  const double tg0 = now_s();
  for (; g_done < replays; ++g_done) {
    PP_CK(cudaGraphLaunch(ex, s));
    if ((g_done & 255) == 255) {
      s.sync();
      if (now_s() - tg0 > cap_s) { ++g_done; break; }
    }
  }
  s.sync();
  const double tg = now_s() - tg0;
  const unsigned long long graph_count = read_ctr();
  const bool graph_exact = graph_count == per_iter * static_cast<unsigned long long>(g_done);

  // Parameter update on the instantiated graph (no re-capture).
  bool upd_ok = false;
  std::string upd_note;
  unsigned long long upd_expect = 0, upd_got = 0;
  {
    size_t nn = 0;
    PP_CK(cudaGraphGetNodes(g, nullptr, &nn));
    std::vector<cudaGraphNode_t> nodes(nn);
    PP_CK(cudaGraphGetNodes(g, nodes.data(), &nn));
    cudaGraphNode_t kn0 = nullptr;
    for (size_t i = 0; i < nn && !kn0; ++i) {
      cudaGraphNodeType t{};
      if (cudaGraphNodeGetType(nodes[i], &t) == cudaSuccess && t == cudaGraphNodeTypeKernel) kn0 = nodes[i];
    }
    if (!kn0) {
      upd_note = "no kernel node found in the captured graph";
    } else {
      cudaKernelNodeParams kp{};
      const std::string e1 = soft_err(cudaGraphKernelNodeGetParams(kn0, &kp));
      if (!e1.empty() || !kp.kernelParams) {
        upd_note = "cudaGraphKernelNodeGetParams failed: " + e1;
      } else {
        const unsigned old_inc = *static_cast<unsigned*>(kp.kernelParams[1]);
        unsigned long long* dptr = ctr.as<unsigned long long>();
        unsigned new_inc = 1000u;
        void* args[2] = {&dptr, &new_inc};
        kp.kernelParams = args;
        const std::string e2 = soft_err(cudaGraphExecKernelNodeSetParams(ex, kn0, &kp));
        if (!e2.empty()) {
          upd_note = "cudaGraphExecKernelNodeSetParams failed: " + e2;
        } else {
          reset();
          PP_CK(cudaGraphLaunch(ex, s));
          s.sync();
          upd_got = read_ctr();
          upd_expect = per_iter - old_inc + new_inc;
          upd_ok = upd_got == upd_expect;
          upd_note = "changed one node's increment from " + std::to_string(old_inc) + " to 1000 without re-capture";
        }
      }
    }
  }
  if (ex) cudaGraphExecDestroy(ex);
  if (g) cudaGraphDestroy(g);

  // Small fixed costs.
  s.sync();
  const int nsync = c.quick ? 2000 : 20000;
  const double ts0 = now_s();
  for (int i = 0; i < nsync; ++i) PP_CK(cudaStreamSynchronize(s));
  const double idle_sync_us = (now_s() - ts0) / nsync * 1e6;
  const int nround = c.quick ? 200 : 2000;
  const double tr0 = now_s();
  for (int i = 0; i < nround; ++i) {
    PP_CK(pk::launch_empty(s));
    s.sync();
  }
  const double empty_roundtrip_us = (now_s() - tr0) / nround * 1e6;
  const int npipe = c.quick ? 2000 : 20000;
  const double tp0 = now_s();
  for (int i = 0; i < npipe; ++i) PP_CK(pk::launch_empty(s));
  s.sync();
  const double empty_pipelined_us = (now_s() - tp0) / npipe * 1e6;

  // Cost of a "flush" (cudaStreamQuery, the call P-H12 found necessary on this stack): on an idle stream, on a busy stream,
  // and added to every enqueue of a pipelined loop (compare with empty_pipelined_us above).
  const int nq = c.quick ? 2000 : 20000;
  s.sync();
  double tq0 = now_s();
  for (int i = 0; i < nq; ++i) (void)cudaStreamQuery(s);
  (void)cudaGetLastError();
  const double query_idle_us = (now_s() - tq0) / nq * 1e6;
  tq0 = now_s();
  for (int i = 0; i < nq; ++i) {
    PP_CK(pk::launch_empty(s));
    (void)cudaStreamQuery(s);
    (void)cudaGetLastError();
  }
  s.sync();
  const double launch_plus_query_us = (now_s() - tq0) / nq * 1e6;
  std::vector<double> query_busy;
  for (int i = 0; i < 200; ++i) {
    for (int k = 0; k < 50; ++k) PP_CK(pk::launch_empty(s));
    const double qa = now_s();
    (void)cudaStreamQuery(s);
    const double qb = now_s();
    (void)cudaGetLastError();
    query_busy.push_back((qb - qa) * 1e6);
    s.sync();
  }
  const double query_busy_us = median(query_busy);

  const double eager_us = e_done > 0 ? te / e_done / kn * 1e6 : std::nan("");
  const double graph_us = g_done > 0 ? tg / g_done / kn * 1e6 : std::nan("");
  const double saving = eager_us > 0 ? 1.0 - graph_us / eager_us : std::nan("");
  r.values.kv("kernels_per_graph", kn).kv("eager_iterations_done", e_done).kv("graph_replays_done", g_done).kv("eager_us_per_kernel", eager_us).kv("graph_us_per_kernel", graph_us).kv("saving_fraction", saving);
  r.values.kv("eager_seconds", te).kv("graph_seconds", tg).kv("eager_counter_exact", eager_exact).kv("graph_counter_exact", graph_exact).kv("expected_counter_per_iteration", per_iter);
  r.values.kv("param_update_ok", upd_ok).kv("param_update_expected", upd_expect).kv("param_update_got", upd_got).kv("param_update_note", upd_note);
  r.values.kv("idle_stream_synchronize_us", idle_sync_us).kv("empty_kernel_launch_plus_sync_us", empty_roundtrip_us).kv("empty_kernel_launch_pipelined_us", empty_pipelined_us);
  r.values.kv("flush_cudaStreamQuery_idle_stream_us", query_idle_us).kv("flush_cudaStreamQuery_busy_stream_median_us", query_busy_us).kv("empty_kernel_launch_plus_flush_pipelined_us", launch_plus_query_us);
  r.inputs.kv("eager_iterations_requested", eager_iters).kv("graph_replays_requested", replays).kv("time_cap_seconds_per_phase", cap_s);
  r.thresholds.kv("saving_fraction_min", 0.30).kv("counters_exact", true).kv("param_update_ok", true);
  r.pass = (saving >= 0.30 && eager_exact && graph_exact && upd_ok) ? 1 : 0;
  char b[220];
  std::snprintf(b, sizeof b, "eager %.3f vs graph %.3f us/kernel (saving %.0f%%), exact=%s, param-update=%s", eager_us, graph_us, saving * 100.0, (eager_exact && graph_exact) ? "yes" : "NO", upd_ok ? "ok" : "FAILED");
  r.headline = b;
  r.notes = "Kernels are tiny on purpose: the numbers are launch overhead, not compute. Counter must come out exactly right after all replays.";
  write_result(c, r, "h9_graphs");
  return r.pass == 1 ? 0 : 3;
}

// ================================================================ P-H13: long-running kernel and the display timeout
// One stage per process. Stages: 2s, 5s, 30s (one spin kernel), queued (10 x 1 s launches in a row), graph (a 10-node graph of 1 s kernels).
inline int p_h13(Ctx& c, const Args& a) {
  Result r;
  r.probe = "P-H13";
  const std::string stage = a.get("stage", "2s");
  double req_s = 0;
  if (stage == "2s") req_s = 2;
  else if (stage == "5s") req_s = 5;
  else if (stage == "30s") req_s = 30;
  else if (stage == "queued" || stage == "graph") req_s = 10;
  else throw std::runtime_error("unknown --stage '" + stage + "' (use 2s, 5s, 30s, queued, graph)");
  const std::string stem = "h13_" + stage;
  init_cuda(c);
  int exec_timeout_attr = -1;
  (void)cudaDeviceGetAttribute(&exec_timeout_attr, cudaDevAttrKernelExecTimeout, 0);
  write_started(c, "P-H13", stem, "running a GPU kernel for about " + fmt_g(req_s) + " s total (stage " + stage + ") - the display may flicker or the driver may reset");

  DevBuf sink(static_cast<size_t>(c.sm_count) * sizeof(unsigned long long) * 4);
  PP_CK(cudaMemset(sink.as(), 0, sink.bytes()));
  Stream s;
  const int blocks = c.sm_count;
  Sampler sampler(c.nvml);
  sampler.start(250);
  std::string launch_err, sync_err;
  const double t0 = now_s();
  if (stage == "2s" || stage == "5s" || stage == "30s") {
    launch_err = soft_err(pk::launch_spin(static_cast<unsigned long long>(req_s * 1e9), sink.as<unsigned long long>(), blocks, s));
  } else if (stage == "queued") {
    for (int i = 0; i < 10 && launch_err.empty(); ++i) launch_err = soft_err(pk::launch_spin(1000000000ULL, sink.as<unsigned long long>(), blocks, s));
  } else {
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ex = nullptr;
    PP_CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < 10; ++i) PP_CK(pk::launch_spin(1000000000ULL, sink.as<unsigned long long>(), blocks, s));
    PP_CK(cudaStreamEndCapture(s, &g));
    PP_CK(cudaGraphInstantiate(&ex, g, 0ULL));
    launch_err = soft_err(cudaGraphLaunch(ex, s));
    sync_err = soft_err(cudaStreamSynchronize(s));
    cudaGraphExecDestroy(ex);
    cudaGraphDestroy(g);
  }
  if (sync_err.empty()) sync_err = soft_err(cudaStreamSynchronize(s));
  const double wall = now_s() - t0;
  const std::string last_err = soft_err(cudaGetLastError());
  // Is the device still alive and usable?
  const std::string alive_err = soft_err(pk::launch_empty(s));
  const std::string alive_sync = alive_err.empty() ? soft_err(cudaStreamSynchronize(s)) : alive_err;
  const bool alive = alive_sync.empty();
  const auto smp = sampler.stop();
  NvmlMemory nm{};
  const bool nvml_after = c.nvml.mem(nm);
  const bool completed = launch_err.empty() && sync_err.empty();
  const bool timing_ok = wall >= req_s * 0.9 && wall <= req_s * 1.5 + 1.0;
  r.values.kv("stage", stage).kv("requested_seconds_total", req_s).kv("wall_seconds", wall).kv("launch_error", launch_err).kv("sync_error", sync_err).kv("last_error", last_err);
  r.values.kv("completed", completed).kv("device_usable_afterwards", alive).kv("post_check_error", alive_sync).kv("nvml_responds_afterwards", nvml_after).kv("kernel_exec_timeout_attribute", exec_timeout_attr);
  r.values.kv_obj("nvml", summarize(smp));
  r.inputs.kv("stage", stage).kv("blocks", blocks);
  r.thresholds.kv("wall_seconds_between", "0.9x and 1.5x+1 of requested");
  r.pass = (completed && alive && timing_ok) ? 1 : 0;
  char b[220];
  std::snprintf(b, sizeof b, "stage %s: wall %.2f s (asked %.0f), completed=%s, device usable after=%s%s%s", stage.c_str(), wall, req_s, completed ? "yes" : "NO", alive ? "yes" : "NO", sync_err.empty() ? "" : ", error: ", sync_err.c_str());
  r.headline = b;
  r.notes = "A GPU kernel that runs longer than the display timeout (about 2 s on a stock Windows driver) can be killed and the driver reset. Run stages in increasing order and stop at the first failure.";
  write_result(c, r, stem);
  return r.pass == 1 ? 0 : 3;
}

}  // namespace pp
