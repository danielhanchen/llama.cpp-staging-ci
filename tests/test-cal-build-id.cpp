// test-cal-build-id: the build identity of the MoE CPU-miss calibration key (src/llama-build-id.h).
// A build without git metadata reports commit "unknown"; two such builds must get different identities (else an upgraded binary reuses the
// older build's calibration), and the same binary relaunched must get the same one (else the calibration is never reused).
// Two "builds" here are two module files of different content under one "unknown" commit; the relaunch is a fresh hash of the same file.

#include "../src/llama-build-id.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;

#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)

static std::string write_file(const std::string & path, const std::string & body) {
    FILE * f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return "";
    }
    fwrite(body.data(), 1, body.size(), f);
    fclose(f);
    return path;
}

int main() {
    // this executable stands in for a loaded module: its path must resolve, and hashing it twice (a relaunch) must agree
    const std::string self = llama_build_id_module_path((const void *) &write_file);
    CHECK(!self.empty(), "the module path of this executable did not resolve");
    std::string m_self[1] = { self };
    const std::string id_a = llama_build_id("unknown", m_self, 1);
    const std::string id_b = llama_build_id("unknown", m_self, 1);
    CHECK(id_a == id_b, "same binary, two identities:\n  %s\n  %s", id_a.c_str(), id_b.c_str());
    CHECK(id_a.find("|m0=0000000000000000") == std::string::npos, "the module hash is 0 (file not read): %s", id_a.c_str());

    // two builds with an unknown commit: modules of different content -> different identities, also when only one byte differs
    const char * tmp = getenv("TMPDIR");
    const std::string dir = tmp != nullptr && tmp[0] != '\0' ? tmp : ".";
    std::string body(1 << 20, 'x');
    const std::string p1 = write_file(dir + "/test-cal-build-id.1.bin", body);
    body[body.size() / 2] = 'y';
    const std::string p2 = write_file(dir + "/test-cal-build-id.2.bin", body);
    CHECK(!p1.empty() && !p2.empty(), "could not write the module stand-ins in %s", dir.c_str());
    std::string m1[2] = { p1, self };
    std::string m2[2] = { p2, self };
    const std::string id1 = llama_build_id("unknown", m1, 2);
    const std::string id2 = llama_build_id("unknown", m2, 2);
    CHECK(id1 != id2, "two unknown-commit builds share one identity: %s", id1.c_str());
    CHECK(id1 == llama_build_id("unknown", m1, 2), "the same build relaunched changed its identity");
    // a dirty commit is treated like an unknown one (local edits do not change the commit)
    CHECK(llama_build_id("abc1234-dirty", m1, 2) != llama_build_id("abc1234-dirty", m2, 2), "two dirty builds of one commit share one identity");
    // a clean commit is the identity by itself, independent of the files (unchanged keys for builds with git metadata)
    CHECK(llama_build_id("abc1234", m1, 2) == "abc1234" && llama_build_id("abc1234", m2, 2) == "abc1234", "a clean commit is not used as is");
    remove(p1.c_str());
    remove(p2.c_str());

    printf("ids: unknown self %s\n     unknown m1 %s\n     unknown m2 %s\n", id_a.c_str(), id1.c_str(), id2.c_str());
    printf("%s\n", fails == 0 ? "OK" : "FAILED");
    return fails == 0 ? 0 : 1;
}
