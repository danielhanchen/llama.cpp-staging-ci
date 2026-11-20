// test-moe-trial-persist: k19-cmtpersist, the CPU-miss timing trials' verdicts kept in the calibration file (src/llama-moe-trial-persist.h).
// The file is the k19-determinism calibration file: the key on line 1, the four calibration numbers on line 2, then one line of trial
// verdicts per tag (cache size, layer count, small-batch limit). Checked here, on the CPU:
//   - a verdict line round-trips (format / parse), malformed lines are refused;
//   - write needs the calibration file with the same key (a missing file or another key writes nothing);
//   - write keeps the calibration numbers and the other tags' lines, and replaces its own tag's line;
//   - the calibration reader's form (key line, then fscanf of four numbers) still reads a file with trial lines;
//   - a later "launch" reads the verdicts back in order (replay), and appended verdicts are read by the one after it;
//   - decide: the first verdict uses the plain margin; later ones keep the previous verdict inside the hysteresis band, and
//     hyst = 0 (LLAMA_MOE_CPU_MISS_PERSIST=0) is the old rule exactly, including the tc_k1 Kaggle T4 numbers that split the launches.

#include "../src/llama-moe-trial-persist.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int fails = 0;

#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

namespace tp = llama_moe_tp;

static void write_file(const std::string & path, const std::string & body) {
    FILE * f = fopen(path.c_str(), "wb");
    if (f != nullptr) {
        fwrite(body.data(), 1, body.size(), f);
        fclose(f);
    }
}

static std::string slurp(const std::string & path) {
    std::vector<std::string> l;
    std::string s;
    if (tp::read_lines(path, l)) {
        for (auto & x : l) {
            s += x + "\n";
        }
    }
    return s;
}

// the calibration reader of llama-moe-cache.cpp (cm_cal_read), reduced: the key line, then four numbers
static bool cal_read(const std::string & path, const std::string & key, double v[4]) {
    FILE * f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    char buf[4096];
    bool ok = fgets(buf, sizeof(buf), f) != nullptr;
    if (ok) {
        std::string line = buf;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        ok = line == key && fscanf(f, "%lf %lf %lf %lf", &v[0], &v[1], &v[2], &v[3]) == 4;
    }
    fclose(f);
    return ok;
}

static bool same(const std::vector<tp::rec> & a, const std::vector<tp::rec> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].on != b[i].on || a[i].host_ok != b[i].host_ok || a[i].gap != b[i].gap || a[i].len != b[i].len ||
                std::abs(a[i].ratio - b[i].ratio) > 1e-6 || a[i].firm != b[i].firm || a[i].tries != b[i].tries) {
            return false;
        }
    }
    return true;
}

int main() {
    const char * td = getenv("TMPDIR");
#if defined(_WIN32)
    if (td == nullptr || td[0] == '\0') {
        td = getenv("TEMP"); // no /tmp on Windows
    }
#endif
    const std::string dir  = std::string(td != nullptr && td[0] != '\0' ? td : "/tmp");
    const std::string path = dir + "/test_moe_trial_persist_" + std::to_string((long long) std::rand()) + ".txt";
    const std::string key  = "moe-cpu-miss-cal v2|build=test|dev=Tesla T4|threads=4|ncpu=4|cache=main|n_expert=256|n_used=8|bytes=1966080|q4_K:2048x512";
    const std::string tag12 = tp::tag_of(4241, 40, 32);
    const std::string tag8  = tp::tag_of(2800, 40, 32);

    // format / parse
    {
        std::vector<tp::rec> r(2);
        r[0].on = 0; r[0].host_ok = 1; r[0].gap = 128; r[0].len = 40; r[0].ratio = 0.977227;
        r[1].on = 1; r[1].host_ok = 1; r[1].gap = 256; r[1].len = 29; r[1].ratio = 0.961197;
        const std::string line = tp::format_line(tag12, r);
        std::vector<tp::rec> back;
        CHECK(tp::parse_line(line, tag12, back) && same(r, back), "round trip of %s", line.c_str());
        CHECK(!tp::parse_line(line, tag8, back), "another tag must not parse");
        CHECK(!tp::parse_line(tag12 + " 2 0:1:128:40:0.97", tag12, back), "a short line must not parse");
        CHECK(!tp::parse_line(tag12 + " 1 0:1:0:40:0.97", tag12, back), "gap 0 must not parse");
        CHECK(!tp::parse_line(tag12 + "x 0", tag12, back), "a tag prefix must not parse");
        std::vector<tp::rec> none;
        CHECK(tp::parse_line(tag12 + " 0", tag12, none) && none.empty(), "an empty record parses");
    }

    // write needs the calibration file with this key
    {
        std::remove(path.c_str());
        std::vector<tp::rec> r(1);
        r[0].on = 1; r[0].len = 40; r[0].gap = 128;
        CHECK(!tp::write(path, key, tag12, r, 1), "no calibration file: nothing may be written");
        FILE * f = fopen(path.c_str(), "r");
        CHECK(f == nullptr, "write created a file without a calibration");
        if (f) fclose(f);
        write_file(path, "another key\n1 2 3 4\n");
        CHECK(!tp::write(path, key, tag12, r, 2), "another key: nothing may be written");
        CHECK(slurp(path) == "another key\n1 2 3 4\n", "a file of another key was changed");
    }

    // a first launch: the calibration exists (k19-determinism wrote it), no verdicts yet
    write_file(path, key + "\n154.7 129.55 188.0 47.0\n");
    std::vector<tp::rec> got;
    CHECK(!tp::read(path, key, tag12, got) && got.empty(), "no verdicts may be read before any were written");

    // it times two trials (tc_k1 launch 2: -3.6% on, then on) and records them
    std::vector<tp::rec> l1(2);
    l1[0].on = 1; l1[0].host_ok = 1; l1[0].gap = 128; l1[0].len = 40; l1[0].ratio = 32.223/33.442;
    l1[1].on = 1; l1[1].host_ok = 1; l1[1].gap = 256; l1[1].len = 40; l1[1].ratio = 31.014/34.861;
    CHECK(tp::write(path, key, tag12, std::vector<tp::rec>(l1.begin(), l1.begin() + 1), 3), "first verdict not written");
    CHECK(tp::write(path, key, tag12, l1, 4), "second verdict not written");
    {
        double v[4];
        CHECK(cal_read(path, key, v) && v[0] == 154.7 && v[2] == 188.0, "the calibration reader no longer reads the file");
    }
    // another tag (another tier) in the same file, then this tag again: both kept, one line each
    std::vector<tp::rec> o(1);
    o[0].on = 0; o[0].gap = 512; o[0].len = 33; o[0].ratio = 1.31;
    CHECK(tp::write(path, key, tag8, o, 5), "the other tag's verdict not written");
    CHECK(tp::read(path, key, tag12, got) && same(got, l1), "launch 2 does not read launch 1's verdicts in order");
    CHECK(tp::read(path, key, tag8, got) && same(got, o), "the other tag's verdicts were lost");

    // launch 2 replays both and times a third (hysteresis keeps it on), appended
    std::vector<tp::rec> l2 = l1;
    tp::rec t3; t3.on = 1; t3.host_ok = 1; t3.gap = 512; t3.len = 40; t3.ratio = 0.985;
    l2.push_back(t3);
    CHECK(tp::write(path, key, tag12, l2, 6), "appended verdict not written");
    CHECK(tp::read(path, key, tag12, got) && same(got, l2), "launch 3 does not read the appended record");
    {
        std::vector<std::string> lines;
        tp::read_lines(path, lines);
        int n12 = 0, n8 = 0;
        for (auto & x : lines) {
            n12 += x.compare(0, tag12.size() + 1, tag12 + " ") == 0;
            n8  += x.compare(0, tag8.size() + 1, tag8 + " ") == 0;
        }
        CHECK(lines.size() == 4 && lines[0] == key && lines[1] == "154.7 129.55 188.0 47.0" && n12 == 1 && n8 == 1,
            "file layout: %zu lines, %d / %d tag lines", lines.size(), n12, n8);
    }
    // a reader with another key reads nothing
    CHECK(!tp::read(path, key + "x", tag12, got), "verdicts read under another key");
    std::remove(path.c_str());

    // decide
    const double m = 0.03, h = 0.03;
    // hyst 0 = the old rule (t_on < 0.97 t_off), with the tc_k1 T4 12 GB trial 1 numbers of both launches
    CHECK(!tp::decide(32.657, 33.419, 13, 13, false, false, m, 0.0), "launch 1 trial 1 (-2.3%%) must be off");
    CHECK( tp::decide(32.223, 33.442, 13, 13, false, false, m, 0.0), "launch 2 trial 1 (-3.6%%) must be on");
    CHECK( tp::decide(32.223, 33.442, 13, 13, true,  false, m, 0.0), "hyst 0 ignores the previous verdict");
    // the first verdict with hysteresis on is the plain margin too
    CHECK(!tp::decide(32.657, 33.419, 13, 13, false, true, m, h), "first verdict must use the plain margin");
    // later: on stays on unless the host path is no longer (m - h) = 0% faster
    CHECK( tp::decide(32.9, 33.0, 13, 13, true, true, m, h), "on must stay on at -0.3%%");
    CHECK(!tp::decide(33.1, 33.0, 13, 13, true, true, m, h), "on must turn off when slower");
    // off stays off unless (m + h) = 6% faster
    CHECK(!tp::decide(31.3, 33.0, 13, 13, true, false, m, h), "off must stay off at -5.2%%");
    CHECK( tp::decide(30.9, 33.0, 13, 13, true, false, m, h), "off must turn on at -6.4%%");
    // too few samples: off, as before
    CHECK(!tp::decide(10.0, 33.0, 2, 13, false, false, m, h), "fewer than 3 samples must be off");

    // k19-calvpersist: a pinned run (LLAMA_MOE_CPU_MISS_CAL_VALUES) has its own file, keyed by calv_key; the library creates it (key +
    // the pinned numbers, the form of cm_cal_write) before the first trial, so the verdicts persist and a second launch replays them
    {
        const double pv[4] = { 20.7, 11.19, 46.7, 41.0 };
        const double pw[4] = { 20.7, 11.19, 46.7, 41.5 };
        const std::string kp = tp::calv_key(key, pv);
        CHECK(kp != key && kp.compare(0, key.size(), key) == 0, "the pinned key extends the machine key");
        CHECK(kp == tp::calv_key(key, pv) && kp != tp::calv_key(key, pw), "the pinned key is stable and names the values");
        const std::string pp = dir + "/test_moe_trial_persist_calv_" + std::to_string((long long) std::rand()) + ".txt";
        std::remove(pp.c_str());
        std::vector<tp::rec> r1(1);
        r1[0].on = 0; r1[0].host_ok = 0; r1[0].gap = 64; r1[0].len = 13; r1[0].ratio = 1.78;
        // before: no file under the pinned run, the verdict write fails (every pinned launch timed its trials again)
        CHECK(!tp::write(pp, kp, tag12, r1, 1), "no file: nothing written");
        char line2[160];
        snprintf(line2, sizeof(line2), "%.17g %.17g %.17g %.17g\n", pv[0], pv[1], pv[2], pv[3]);
        write_file(pp, kp + "\n" + line2);
        // launch 1 records its verdict; launch 2 (same key and tag) replays it; the pinned numbers stay readable
        CHECK(tp::write(pp, kp, tag12, r1, 2), "pinned file: verdict written");
        std::vector<tp::rec> back;
        CHECK(tp::read(pp, kp, tag12, back) && back.size() == 1 && back[0].on == 0 && back[0].host_ok == 0 && back[0].len == 13,
            "launch 2 replays launch 1's verdict");
        double cv[4];
        CHECK(cal_read(pp, kp, cv) && cv[0] == pv[0] && cv[3] == pv[3], "the pinned numbers stay readable");
        // a launch without the variable (machine key) does not read the pinned file
        CHECK(!tp::read(pp, key, tag12, back) && !cal_read(pp, key, cv), "the machine key does not read the pinned file");
        std::remove(pp.c_str());
    }

    // k19-calvmargin: a marginal "on" is not locked in. The A100 40 Flash-Next numbers (37.3 ms without, 35.7 with: -4.3%, first trial,
    // switch point 0.97) are an "on" below the 10% gain a persisted "on" needs: off, open, timed again by the next launches, firm "off"
    // after 3 timings without a clear one; a clear timing, or any "off", is firm at once; a line written before (no :firm:tries) reads
    // as firm
    {
        const double thr1 = tp::threshold(false, false, m, h);
        CHECK(std::abs(thr1 - 0.97) < 1e-12, "first trial switch point %.3f", thr1);
        CHECK(std::abs(tp::threshold(true, true, m, h) - 1.0) < 1e-12 && std::abs(tp::threshold(true, false, m, h) - 0.94) < 1e-12 &&
            std::abs(tp::threshold(true, true, m, 0.0) - 0.97) < 1e-12, "switch points with hysteresis");
        for (int k = 0; k < 4; ++k) {
            const bool hp = k >= 2, pv = k % 2 == 1;
            for (double r = 0.80; r < 1.20; r += 0.001) {
                const bool d = tp::decide(r*33.0, 33.0, 13, 13, hp, pv, m, h);
                const double t = tp::threshold(hp, pv, m, h);
                CHECK(d == (r < t) || std::abs(r - t) < 1e-9, "threshold disagrees with decide at %.3f (case %d)", r, k);
            }
        }
        // k19-calvmargin: a persisted "on" needs a 10% gain
        const double g = 0.10;
        CHECK( tp::marginal_on(true, 35.7, 37.3, g), "the A100 -4.3%% on must be marginal");
        CHECK( tp::marginal_on(true, 27.4, 29.4, g), "the A100 -6.8%% on (replayed 11%% slower later) must be marginal");
        CHECK( tp::marginal_on(true, 26.9, 29.5, g), "the A100 -8.8%% on (0.912) must be marginal");
        CHECK(!tp::marginal_on(true, 27.0, 30.0, g), "-10%% on must be clear");
        CHECK(!tp::marginal_on(false, 35.7, 37.3, g), "an off is never marginal");
        CHECK(!tp::marginal_on(true, 35.7, 37.3, 0.0), "min gain 0: never marginal (the k19-calvpersist rule)");
        tp::rec a;
        CHECK(!tp::settle(nullptr, true, true, 3, a) && a.on == 0 && a.firm == 0 && a.tries == 1, "marginal on: off and open");
        tp::rec b;
        CHECK(!tp::settle(&a, true, true, 3, b) && b.on == 0 && b.firm == 0 && b.tries == 2, "second marginal: still open");
        tp::rec c;
        CHECK(!tp::settle(&b, true, true, 3, c) && c.on == 0 && c.firm == 1 && c.tries == 3, "third marginal on: firm off");
        tp::rec c2;
        CHECK(!tp::settle(&a, false, true, 3, c2) && c2.on == 0 && c2.firm == 1 && c2.tries == 2, "a marginal off re-timing: firm off");
        tp::rec c3;
        CHECK(!tp::settle(nullptr, false, true, 3, c3) && c3.on == 0 && c3.firm == 1 && c3.tries == 1,
            "a marginal off first timing (T4 12 GB: 0.992 against 1.000): firm off at once");
        tp::rec d;
        CHECK( tp::settle(&a, true, false, 3, d) && d.on == 1 && d.firm == 1 && d.tries == 2, "a clear re-timing: firm with its verdict");
        tp::rec e;
        CHECK( tp::settle(nullptr, true, false, 3, e) && e.on == 1 && e.firm == 1 && e.tries == 1, "a clear first timing: firm");
        // trusted marginal "on" (12..31-CPU mailbox hosts): taken in the timing launch, open; three in a row: firm on; an off re-timing: firm off
        tp::rec ta;
        CHECK( tp::settle(nullptr, true, true, 3, ta, true) && ta.on == 1 && ta.firm == 0 && ta.tries == 1, "trusted marginal on: on and open");
        tp::rec tb;
        CHECK( tp::settle(&ta, true, true, 3, tb, true) && tb.on == 1 && tb.firm == 0 && tb.tries == 2, "second trusted marginal: on, open");
        tp::rec tc;
        CHECK( tp::settle(&tb, true, true, 3, tc, true) && tc.on == 1 && tc.firm == 1 && tc.tries == 3, "third trusted marginal on: firm on");
        tp::rec td;
        CHECK(!tp::settle(&ta, false, false, 3, td, true) && td.on == 0 && td.firm == 1 && td.tries == 2, "a trusted open on timed off: firm off");
        tp::rec te;
        CHECK( tp::settle(&ta, true, false, 3, te, true) && te.on == 1 && te.firm == 1 && te.tries == 2, "a trusted open on timed clear: firm on");

        // the firm "on" skip: a first trial made firm "on" by agreeing marginal timings skips the warm phase and the trials in later
        // launches, only on such a record, only on 12..31-CPU hosts and only while LLAMA_MOE_CPU_MISS_FIRM_SKIP is not 0
        const double mg = 0.10;                                             // the default LLAMA_MOE_CPU_MISS_PERSIST_MIN
        int calls = 0;
        tp::rec fo = tc; fo.ratio = 0.935;                                  // three agreeing marginal "on" timings, recorded firm
        auto first = [&](const tp::rec * r) { return [&calls, r]() { calls++; return r; }; };
        CHECK( tp::firm_marginal_on(fo, 3, mg), "firm marginal on record");
        CHECK( tp::firm_skip(true, true, first(&fo), 3, mg) && calls == 1, "(a) a firm marginal on record skips the trials");
        tp::rec fo4 = fo; fo4.tries = 4;
        CHECK( tp::firm_skip(true, true, first(&fo4), 3, mg), "(a) more tries than needed still skip");
        tp::rec nf = tb; nf.ratio = 0.935;                                  // open: two trusted marginal timings
        CHECK(!tp::firm_skip(true, true, first(&nf), 3, mg), "(b) an open record does not skip");
        tp::rec nf3 = fo; nf3.firm = 0;                                     // open with enough tries (an earlier, larger PERSIST_TRIES)
        CHECK(!tp::firm_skip(true, true, first(&nf3), 3, mg), "(b) an open on record with three tries does not skip");
        tp::rec fx = c; fx.ratio = 0.935;                                   // firm off after three untrusted marginal timings
        CHECK(!tp::firm_skip(true, true, first(&fx), 3, mg), "(b) a firm off record does not skip");
        tp::rec fx2 = td; fx2.ratio = 1.02;                                 // firm off by an off re-timing
        CHECK(!tp::firm_skip(true, true, first(&fx2), 3, mg), "(b) a firm off re-timing does not skip");
        tp::rec cl = e; cl.ratio = 0.85;                                    // a clear first "on": firm, but not marginal
        CHECK(!tp::firm_skip(true, true, first(&cl), 3, mg), "(b) a clear firm on replays as recorded");
        tp::rec few = fo; few.tries = 2;
        CHECK(!tp::firm_skip(true, true, first(&few), 3, mg), "(b) firm on with fewer tries than needed does not skip");
        tp::rec r1 = fo; r1.ratio = 1.0;
        CHECK(!tp::firm_skip(true, true, first(&r1), 3, mg), "(b) ratio 1 is not on");
        tp::rec r95 = fo; r95.ratio = 1.0 - mg;
        CHECK(!tp::firm_skip(true, true, first(&r95), 3, mg), "(b) ratio at the margin is not marginal");
        CHECK(!tp::firm_skip(true, true, first(nullptr), 3, mg), "(b) no file record does not skip");
        calls = 0;
        CHECK(!tp::firm_skip(tp::env_on("0"), true, first(&fo), 3, mg) && calls == 0, "(c) LLAMA_MOE_CPU_MISS_FIRM_SKIP=0 disables the skip");
        CHECK( tp::env_on(nullptr) && tp::env_on("1") && !tp::env_on("0"), "(c) the switch is on unless 0");
        CHECK( tp::firm_skip(tp::env_on(nullptr), true, first(&fo), 3, mg), "(c) unset: the skip is on");
        // (d) the trusted scope needs a 12..31-CPU host (defaults: wide from 32, mid from 12)
        auto mid = [](int n) { return tp::wide_mid_host(n, tp::wide_host(n, nullptr, nullptr), nullptr, nullptr); };
        CHECK(!mid(4) && !mid(8) && !mid(11) && mid(12) && mid(16) && mid(31) && !mid(32) && !mid(64) && !mid(192), "(d) 12..31 CPUs only");
        calls = 0;
        for (int n : { 1, 4, 8, 11, 32, 48, 64, 192 }) {
            CHECK(!tp::firm_skip(true, mid(n), first(&fo), 3, mg), "(d) %d CPUs must not skip", n);
        }
        CHECK(calls == 0, "(d) the record is not read outside the scope");
        for (int n : { 12, 20, 31 }) {
            CHECK( tp::firm_skip(true, mid(n), first(&fo), 3, mg), "(d) %d CPUs skip", n);
        }
        CHECK(!tp::wide_mid_host(16, tp::wide_host(16, nullptr, nullptr), "0", nullptr), "(d) LLAMA_MOE_SINGLE_WIDE_MID=0: not a mid host");
        CHECK(!tp::wide_mid_host(16, tp::wide_host(16, nullptr, "16"), nullptr, nullptr), "(d) a 16-CPU wide host is not a mid host");
        CHECK( tp::wide_mid_host(40, tp::wide_host(40, "0", nullptr), nullptr, nullptr), "(d) LLAMA_MOE_SINGLE_WIDE=0: 40 CPUs are mid");

        // round trip of an open record; a line in the old form reads as firm
        std::vector<tp::rec> r(2);
        r[0].on = 0; r[0].gap = 128; r[0].len = 40; r[0].ratio = 35.7/37.3; r[0].firm = 0; r[0].tries = 1;
        r[1].on = 0; r[1].gap = 512; r[1].len = 26; r[1].ratio = 1.31;
        std::vector<tp::rec> back;
        CHECK(tp::parse_line(tp::format_line(tag12, r), tag12, back) && same(r, back) && back[0].firm == 0 && back[1].firm == 1,
            "an open record must round-trip");
        CHECK(tp::parse_line(tag12 + " 2 0:1:128:40:0.957105 1:1:256:29:0.961197", tag12, back) && back.size() == 2 &&
            back[0].firm == 1 && back[0].tries == 1 && back[1].on == 1 && back[1].firm == 1, "an old line must read as firm");
        CHECK(!tp::parse_line(tag12 + " 1 0:1:128:40:0.95:0:0", tag12, back), "tries 0 must not parse");

        // launch 1 writes the open record; launch 2 reads it back open (to time it again), launch 3 reads the firm one
        const std::string pm = dir + "/test_moe_trial_persist_marg_" + std::to_string((long long) std::rand()) + ".txt";
        write_file(pm, key + "\n20.7 11.19 46.7 41.0\n");
        std::vector<tp::rec> l(1);
        tp::settle(nullptr, true, true, 3, l[0]); l[0].gap = 128; l[0].len = 40; l[0].ratio = 35.7/37.3;
        CHECK(tp::write(pm, key, tag12, l, 1) && tp::read(pm, key, tag12, back) && back.size() == 1 && back[0].firm == 0,
            "launch 2 must see the open record");
        tp::rec n2; tp::settle(&back[0], false, false, 3, n2); n2.gap = back[0].gap; n2.len = back[0].len; n2.ratio = 1.36;
        l[0] = n2;
        CHECK(tp::write(pm, key, tag12, l, 2) && tp::read(pm, key, tag12, back) && back.size() == 1 && back[0].firm == 1 &&
            back[0].on == 0 && back[0].tries == 2, "launch 3 must replay the firm verdict");
        std::remove(pm.c_str());
    }

    if (fails == 0) {
        printf("test-moe-trial-persist: OK\n");
    }
    return fails == 0 ? 0 : 1;
}
