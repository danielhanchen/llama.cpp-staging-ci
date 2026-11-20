// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// k19-autoprof: the self-learned profile files (src/llama-autoprof.h, the PLE row recorder of src/llama-row-cache.*).
//   1. the directory (the CPU-miss calibration's: LLAMA_MOE_CPU_MISS_CAL_DIR, LLAMA_CACHE, XDG_CACHE_HOME, HOME; LLAMA_MOE_AUTOPROF_DIR
//      overrides), the switch LLAMA_MOE_AUTOPROF (0 off, 1 record + use, 2 record only, 3 use only), the file names and the model key;
//   2. the atomic write: the file holds the new body, no temporary is left, a write that cannot complete leaves the old file;
//   3. the MoE routing profile body (as written by the cache) reads back with llama_moe_profile_load; no file / a cut file reads nothing;
//      llama_moe_profile_pick (k19-mgpu2, resident device prefill): hottest pairs of a group first, then the lowest ids in turn;
//   4. the PLE recorder: a record-only cache (no slots) gathers bit-identical rows, counts every read and keeps each row's bytes; its
//      profile preloads exactly the recorded rows in a later cache (gathers still bit-identical, static hits); a second session adds its
//      reads to the halved loaded counts; the memory cap holds; a missing profile preloads nothing.
// usage: test-autoprof

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../src/llama-autoprof.h"
#include "../src/llama-moe-cache.h"
#include "../src/llama-row-cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
static int setenv(const char * name, const char * value, int /*overwrite*/) { return _putenv_s(name, value); }
static int unsetenv(const char * name) { return _putenv_s(name, ""); }
#endif

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

namespace fs = std::filesystem;

static std::string slurp(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static size_t n_tmp_left(const fs::path & d) {
    size_t n = 0;
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(d, ec)) {
        n += e.path().filename().string().find(".tmp") != std::string::npos ? 1 : 0;
    }
    return n;
}

static void test_names(const fs::path & root) {
    unsetenv("LLAMA_MOE_CPU_MISS_CAL_DIR");
    unsetenv("LLAMA_CACHE");
    unsetenv("LLAMA_MOE_AUTOPROF_DIR");
    setenv("XDG_CACHE_HOME", (root / "xdg").string().c_str(), 1);
#if !defined(_WIN32)
    CHECK(llama_autoprof::cal_dir() == (root / "xdg").string() + "/llama.cpp", "XDG dir: %s", llama_autoprof::cal_dir().c_str());
#endif
    setenv("LLAMA_CACHE", (root / "lc").string().c_str(), 1);
    CHECK(llama_autoprof::cal_dir() == (root / "lc").string(), "LLAMA_CACHE dir: %s", llama_autoprof::cal_dir().c_str());
    setenv("LLAMA_MOE_CPU_MISS_CAL_DIR", (root / "cal").string().c_str(), 1);
    CHECK(llama_autoprof::cal_dir() == (root / "cal").string(), "calibration dir: %s", llama_autoprof::cal_dir().c_str());
    CHECK(llama_autoprof::dir() == llama_autoprof::cal_dir(), "the profile directory is the calibration directory by default");
    setenv("LLAMA_MOE_AUTOPROF_DIR", (root / "ap").string().c_str(), 1);
    CHECK(llama_autoprof::dir() == (root / "ap").string(), "LLAMA_MOE_AUTOPROF_DIR: %s", llama_autoprof::dir().c_str());

    struct { const char * v; int m; } modes[] = { { nullptr, 1 }, { "", 1 }, { "0", 0 }, { "1", 1 }, { "2", 2 }, { "3", 3 }, { "7", 1 } };
    for (const auto & t : modes) {
        if (t.v) {
            setenv("LLAMA_MOE_AUTOPROF", t.v, 1);
        } else {
            unsetenv("LLAMA_MOE_AUTOPROF");
        }
        CHECK(llama_autoprof::mode() == t.m, "LLAMA_MOE_AUTOPROF=%s -> %d", t.v ? t.v : "(unset)", llama_autoprof::mode());
    }
    unsetenv("LLAMA_MOE_AUTOPROF");
    CHECK(!llama_autoprof::records(0) && !llama_autoprof::uses(0), "0 = off");
    CHECK(llama_autoprof::records(1) && llama_autoprof::uses(1), "1 = record + use");
    CHECK(llama_autoprof::records(2) && !llama_autoprof::uses(2), "2 = record only");
    CHECK(!llama_autoprof::records(3) && llama_autoprof::uses(3), "3 = use only");

    const uint64_t h1 = llama_autoprof::model_hash("qwen 35B IQ4_XS", 1000, 2000);
    CHECK(h1 == llama_autoprof::model_hash("qwen 35B IQ4_XS", 1000, 2000), "model key is deterministic");
    CHECK(h1 != llama_autoprof::model_hash("qwen 35B IQ4_XS", 1000, 2001), "model key names the size");
    CHECK(h1 != llama_autoprof::model_hash("qwen 35B Q4_K", 1000, 2000), "model key names the description");
    const std::string p = llama_autoprof::path("/d", "moe", 0x1234, ".48.512");
#if !defined(_WIN32)
    CHECK(p == "/d/autoprof_moe_0000000000001234.48.512", "path: %s", p.c_str());
#endif
    CHECK(llama_autoprof::path("", "moe", 1, "").empty(), "no directory, no path");
}

static void test_write(const fs::path & root) {
    const fs::path d = root / "w" / "sub";
    const std::string f = (d / "x.txt").string();
    CHECK(llama_autoprof::write_atomic(f, "first\n"), "write into a new directory");
    CHECK(slurp(f) == "first\n", "content");
    std::string big(1 << 20, 'a');
    CHECK(llama_autoprof::write_atomic(f, big), "overwrite");
    CHECK(slurp(f) == big, "overwritten content");
    CHECK(n_tmp_left(d) == 0, "no temporary left");
    // the target is a directory: the rename fails, the temporary is removed, the directory untouched
    const fs::path dd = d / "isdir";
    fs::create_directories(dd / "keep");
    CHECK(!llama_autoprof::write_atomic(dd.string(), "y"), "a write over a directory fails");
    CHECK(fs::is_directory(dd / "keep"), "the directory is untouched");
    CHECK(n_tmp_left(d) == 0, "no temporary left after a failed write");
    // a directory that cannot be made (a file in the way): false, the old file kept
    CHECK(!llama_autoprof::write_atomic((fs::path(f) / "z").string(), "z"), "a path under a file fails");
    CHECK(slurp(f) == big, "the old file is kept");
    CHECK(!llama_autoprof::write_atomic("", "z"), "no path, no write");
}

static void test_moe_body(const fs::path & root) {
    // the body ap_save writes (heat descending), read with the shared reader
    const std::string f = (root / "autoprof_moe_test.3.8").string();
    std::vector<int32_t> ne = { 8, 8, 8 };
    std::vector<llama_moe_profile_entry> out;
    CHECK(!llama_moe_profile_load(f, ne, out) && out.empty(), "no file: nothing read");
    std::string body = "moe-cache-profile 1 3\n";
    const int il[] = { 1, 0, 2, 0 }, e[] = { 3, 7, 0, 1 };
    const float h[] = { 96.5f, 40.0f, 3.0f, 0.5f };
    char buf[64];
    for (int i = 0; i < 4; ++i) {
        snprintf(buf, sizeof(buf), "%d %d %.6g\n", il[i], e[i], h[i]);
        body += buf;
    }
    CHECK(llama_autoprof::write_atomic(f, body), "write");
    CHECK(llama_moe_profile_load(f, ne, out) && out.size() == 4, "read back %zu", out.size());
    for (size_t i = 0; i < out.size() && i < 4; ++i) {
        CHECK(out[i].il == il[i] && out[i].e == e[i] && out[i].heat == h[i], "entry %zu", i);
    }
    CHECK(!llama_moe_profile_load(f, { 8, 8 }, out), "another layer count is refused");
    CHECK(llama_autoprof::write_atomic(f, body.substr(0, body.size() - 6)), "write a cut body");
    CHECK(!llama_moe_profile_load(f, ne, out), "a cut file is refused");
}

// k19-mgpu2: llama_moe_profile_pick, the resident device prefill from a profile: the hottest pairs of the group's layers over all of them
// (heat descending, zero heat and other layers ignored), then the lowest ids not taken, one layer after the other, up to the slot count
static void test_pick() {
    // layers 4 and 6 form the group (6 experts each), layer 5 is another group's
    const std::vector<llama_moe_profile_entry> prof = {
        { 6, 5, 2.0f }, { 4, 3, 9.0f }, { 5, 0, 50.0f }, { 6, 1, 7.5f }, { 4, 3, 1.0f }, { 4, 2, 0.0f }, { 6, 4, 2.0f },
    };
    size_t np = 0, nd = 0;
    // 3 slots: the three hottest of layers 4 / 6 (9.0, 7.5, then the first 2.0 in file order), nothing by id
    auto p = llama_moe_profile_pick(prof, { 4, 6 }, 6, 3, &np, &nd);
    CHECK(p.size() == 2 && p[0] == std::vector<int32_t>({ 3 }) && p[1] == std::vector<int32_t>({ 1, 5 }) && np == 3 && nd == 0,
        "3 slots: %zu / %zu, %zu + %zu", p.size() > 0 ? p[0].size() : 0, p.size() > 1 ? p[1].size() : 0, np, nd);
    // 8 slots: 4 from the profile (zero heat and the duplicate skipped), then by id in turn: 4:0, 6:0, 4:1, 6:2
    p = llama_moe_profile_pick(prof, { 4, 6 }, 6, 8, &np, &nd);
    CHECK(np == 4 && nd == 4, "8 slots: %zu + %zu", np, nd);
    CHECK(p[0] == std::vector<int32_t>({ 0, 1, 3 }) && p[1] == std::vector<int32_t>({ 0, 1, 2, 4, 5 }), "8 slots: the picked ids");
    // more slots than experts: every expert once
    p = llama_moe_profile_pick(prof, { 4, 6 }, 6, 20, &np, &nd);
    CHECK(p[0].size() == 6 && p[1].size() == 6 && np + nd == 12, "20 slots: %zu / %zu", p[0].size(), p[1].size());
    // no profile: the lowest ids per layer in turn (the prefill by id)
    p = llama_moe_profile_pick({}, { 4, 6 }, 6, 5, &np, &nd);
    CHECK(np == 0 && nd == 5 && p[0] == std::vector<int32_t>({ 0, 1, 2 }) && p[1] == std::vector<int32_t>({ 0, 1 }), "no profile");
    // no slots
    p = llama_moe_profile_pick(prof, { 4, 6 }, 6, 0, &np, &nd);
    CHECK(p[0].empty() && p[1].empty() && np == 0 && nd == 0, "0 slots");
}

// ------------------------------------------------------------------------------------------------------------------------------------
// PLE recorder

struct gather_env {
    ggml_backend_t        be  = nullptr;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         tab = nullptr;
    ggml_tensor *         ids = nullptr;
    ggml_tensor *         out = nullptr;
    ggml_cgraph *         gf  = nullptr;
};

static bool gather_init(gather_env & g, ggml_type type, int64_t ncols, int64_t nrows, int64_t nids, std::mt19937 & rng) {
    g.be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(g.be, 4);
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    g.tab = ggml_new_tensor_2d(g.ctx, type, ncols, nrows);
    g.ids = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, nids);
    g.out = ggml_get_rows(g.ctx, g.tab, g.ids);
    g.gf  = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, g.be);
    if (!g.buf) {
        return false;
    }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> f(ncols*nrows);
    for (auto & x : f) {
        x = nd(rng);
    }
    std::vector<uint8_t> q(ggml_nbytes(g.tab));
    ggml_quantize_chunk(type, f.data(), q.data(), 0, nrows, ncols, nullptr);
    ggml_backend_tensor_set(g.tab, q.data(), 0, q.size());
    return true;
}

static std::vector<float> gather_run(gather_env & g, const std::vector<int32_t> & ids) {
    ggml_backend_tensor_set(g.ids, ids.data(), 0, ids.size()*sizeof(int32_t));
    ggml_backend_graph_compute(g.be, g.gf);
    std::vector<float> out(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, out.data(), 0, out.size()*sizeof(float));
    return out;
}

static uint64_t prof_entries(const std::string & p) {
    const std::string s = slurp(p);
    uint64_t n = UINT64_MAX;
    if (s.size() >= 48 && memcmp(s.data(), "LLPLERC2", 8) == 0) {
        memcpy(&n, s.data() + 40, 8);
    }
    return n;
}

static void test_recorder(const fs::path & root) {
    std::mt19937 rng(77);
    const int64_t ncols = 256, nrows = 5000, nids = 2048;
    const ggml_type type = GGML_TYPE_Q8_0;
    gather_env g;
    if (!gather_init(g, type, ncols, nrows, nids, rng)) {
        CHECK(false, "alloc failed");
        return;
    }
    const size_t row_b = ggml_row_size(type, ncols);
    std::vector<std::vector<int32_t>> sets;
    std::set<int32_t> seen;
    for (int i = 0; i < 4; ++i) {
        std::vector<int32_t> v(nids);
        for (auto & x : v) {
            x = (rng() & 1) ? (int32_t) (rng() % 200) * 13 % nrows : (int32_t) (rng() % nrows);
        }
        seen.insert(v.begin(), v.end());
        sets.push_back(v);
    }
    std::vector<std::vector<float>> ref;
    for (const auto & s : sets) {
        ref.push_back(gather_run(g, s));
    }
    const std::string prof = (root / "autoprof_ple_test.5000").string();

    // session 1: record only (no slots, part below row 1000: those rows are recorded too)
    {
        llama_row_cache rc(g.tab, 0, 0, 1000);
        size_t n0 = rc.load_profile(fs::exists(prof) ? prof : (root / "missing.prof").string());
        CHECK(n0 == 0, "a missing profile preloads nothing");
        rc.enable_recorder(64u << 20);
        CHECK(rc.capacity() == 0, "record only: no slots (%zu)", rc.capacity());
        CHECK(llama_row_cache_attach(&rc), "attach");
        for (size_t i = 0; i < sets.size(); ++i) {
            const auto out = gather_run(g, sets[i]);
            CHECK(memcmp(out.data(), ref[i].data(), out.size()*sizeof(float)) == 0, "session 1 set %zu: gather differs", i);
        }
        llama_row_cache_detach(g.tab);
        CHECK(rc.st.n_recorded.load() == (uint64_t) (nids*sets.size()), "every read recorded: %llu", (unsigned long long) rc.st.n_recorded.load());
        CHECK(rc.n_recorded_rows() == seen.size(), "distinct rows %zu vs %zu", rc.n_recorded_rows(), seen.size());
        CHECK(rc.st.n_hits.load() == 0, "record only: no hits");
        CHECK(rc.save_profile(prof), "save");
        CHECK(prof_entries(prof) == seen.size(), "profile rows %llu", (unsigned long long) prof_entries(prof));
        CHECK(n_tmp_left(root) == 0, "no temporary left");
    }
    // session 2: the profile preloads the recorded rows above the part; gathers identical; the recorder adds this session's reads
    {
        const size_t slot = llama_row_cache::slot_bytes(row_b);
        llama_row_cache rc(g.tab, 200*slot, 100000*slot, 1000);
        const size_t n = rc.load_profile(prof);
        size_t above = 0;
        for (int32_t r : seen) {
            above += r >= 1000 ? 1 : 0;
        }
        CHECK(n == above, "preloaded %zu of %zu recorded rows above the part", n, above);
        rc.enable_recorder(64u << 20);
        rc.verify = true;
        CHECK(llama_row_cache_attach(&rc), "attach");
        for (size_t i = 0; i < sets.size(); ++i) {
            const auto out = gather_run(g, sets[i]);
            CHECK(memcmp(out.data(), ref[i].data(), out.size()*sizeof(float)) == 0, "session 2 set %zu: gather differs", i);
        }
        llama_row_cache_detach(g.tab);
        CHECK(rc.st.n_hits_static.load() > 0 && rc.st.n_verify_bad.load() == 0, "static hits %llu, bad %llu",
            (unsigned long long) rc.st.n_hits_static.load(), (unsigned long long) rc.st.n_verify_bad.load());
        CHECK(rc.save_profile(prof), "save 2");
        CHECK(prof_entries(prof) == seen.size(), "profile rows after session 2: %llu", (unsigned long long) prof_entries(prof));
        // k19-autoprof2: a profile directory that does not exist yet is created (the first save on a host-lookup tier)
        const fs::path sub = root / "new_dir" / "deeper";
        CHECK(!fs::exists(sub), "fresh directory");
        CHECK(rc.save_profile((sub / "autoprof_ple_test.5000").string()), "save into a missing directory");
        CHECK(prof_entries((sub / "autoprof_ple_test.5000").string()) == seen.size(), "profile rows in the new directory");
        CHECK(n_tmp_left(sub) == 0, "no temporary left in the new directory");
    }
    // a third cache: the rows' bytes in the file are the table's (load checks every row's checksum and spot-checks the mapping)
    {
        llama_row_cache rc(g.tab, 0, 1u << 30, 0);
        CHECK(rc.load_profile(prof) == seen.size(), "every recorded row loads");
        std::vector<uint8_t> tmp(row_b);
        size_t bad = 0;
        for (int32_t r : seen) {
            bad += !rc.peek(r, tmp.data()) || memcmp(tmp.data(), (const uint8_t *) g.tab->data + (size_t) r*g.tab->nb[1], row_b) != 0;
        }
        CHECK(bad == 0, "%zu rows differ from the table", bad);
    }
    // the memory cap: about 100 rows
    {
        llama_row_cache rc(g.tab, 0, 0, 0);
        rc.enable_recorder(100*(row_b + 24));
        CHECK(llama_row_cache_attach(&rc), "attach");
        gather_run(g, sets[0]);
        llama_row_cache_detach(g.tab);
        CHECK(rc.n_recorded_rows() <= 100 && rc.n_recorded_rows() >= 90 && rc.st.n_rec_full.load() > 0, "cap: %zu rows, %llu refused",
            rc.n_recorded_rows(), (unsigned long long) rc.st.n_rec_full.load());
    }
    ggml_backend_buffer_free(g.buf);
    ggml_free(g.ctx);
    ggml_backend_free(g.be);
}

// shipped profiles: read-only lookup beside the model file, then <cal dir>/shipped; LLAMA_MOE_AUTOPROF_SHIPPED names the one
// directory or turns the lookup off
static void test_shipped(const fs::path & root) {
    const fs::path md = root / "models";
    const fs::path cal = root / "shipcal";
    fs::create_directories(md);
    fs::create_directories(cal / "shipped");
    unsetenv("LLAMA_MOE_AUTOPROF_SHIPPED");
    setenv("LLAMA_MOE_CPU_MISS_CAL_DIR", cal.string().c_str(), 1);
    const std::string model = (md / "m.gguf").string();
    const uint64_t h = 0xabcdef;
    CHECK(llama_autoprof::find_shipped(model, "moe", h, ".48.512").empty(), "nothing shipped");
    const std::string in_cal = llama_autoprof::path((cal / "shipped").string(), "moe", h, ".48.512");
    CHECK(llama_autoprof::write_atomic(in_cal, "moe-cache-profile 1 48\n"), "write the cal-dir file");
    CHECK(llama_autoprof::find_shipped(model, "moe", h, ".48.512") == in_cal, "found in <cal dir>/shipped");
    CHECK(llama_autoprof::find_shipped("", "moe", h, ".48.512") == in_cal, "no model path: <cal dir>/shipped");
    const std::string beside = llama_autoprof::path(md.string(), "moe", h, ".48.512");
    CHECK(llama_autoprof::write_atomic(beside, "moe-cache-profile 1 48\n"), "write the file beside the model");
    CHECK(llama_autoprof::find_shipped(model, "moe", h, ".48.512") == beside, "beside the model wins");
    CHECK(llama_autoprof::find_shipped(model, "moe", h, ".48.256").empty(), "another suffix (expert count) is not found");
    CHECK(llama_autoprof::find_shipped(model, "moe", h + 1, ".48.512").empty(), "another model hash is not found");
    CHECK(llama_autoprof::find_shipped(model, "ple", h, ".48.512").empty(), "another kind is not found");
    setenv("LLAMA_MOE_AUTOPROF_SHIPPED", (cal / "shipped").string().c_str(), 1);
    CHECK(llama_autoprof::find_shipped(model, "moe", h, ".48.512") == in_cal, "the named directory only");
    setenv("LLAMA_MOE_AUTOPROF_SHIPPED", root.string().c_str(), 1);
    CHECK(llama_autoprof::find_shipped(model, "moe", h, ".48.512").empty(), "a named directory without the file: none");
    setenv("LLAMA_MOE_AUTOPROF_SHIPPED", "0", 1);
    CHECK(llama_autoprof::find_shipped(model, "moe", h, ".48.512").empty(), "0: off");
    CHECK(llama_autoprof::shipped_dirs(model).empty(), "0: no directory");
    fs::create_directories(md / ("autoprof_moe_" + std::string("0000000000abcdef") + ".48.1"));
    unsetenv("LLAMA_MOE_AUTOPROF_SHIPPED");
    CHECK(llama_autoprof::find_shipped(model, "moe", h, ".48.1").empty(), "a directory of that name is not a profile");
    unsetenv("LLAMA_MOE_CPU_MISS_CAL_DIR");
}

int main() {
    const fs::path root = fs::absolute("test-autoprof.tmp");
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    test_names(root);
    test_write(root);
    test_moe_body(root);
    test_pick();
    test_recorder(root);
    test_shipped(root);
    fs::remove_all(root, ec);
    if (n_fail == 0) {
        printf("test-autoprof: ALL OK\n");
        return 0;
    }
    printf("test-autoprof: %d FAILED\n", n_fail);
    return 1;
}
