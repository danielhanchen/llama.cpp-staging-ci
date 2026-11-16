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
                std::abs(a[i].ratio - b[i].ratio) > 1e-6) {
            return false;
        }
    }
    return true;
}

int main() {
    const char * td = getenv("TMPDIR");
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

    if (fails == 0) {
        printf("test-moe-trial-persist: OK\n");
    }
    return fails == 0 ? 0 : 1;
}
