// k19-ramprof: the routing profile of the MoE cache (llama_moe_profile_*): write / load round trip, the checks a load makes (layer count,
// ranges, duplicates, truncation, missing file), the RAM-tier fill order (heat descending, ties in file order) and the ordered reservation
// of RAM-tier slots (per layout up to its slots, never evicting, skipping held / unwanted experts and uncached layers)
// usage: test-moe-ram-profile [scratch file path]

#include "../src/llama-moe-cache.h"
#include "../src/llama-moe-host-cache.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static void write_text(const std::string & path, const char * text) {
    FILE * f = fopen(path.c_str(), "w");
    CHECK(f != nullptr);
    if (f != nullptr) {
        fputs(text, f);
        fclose(f);
    }
}

static void test_file(const std::string & path) {
    // 3 layers: 0 and 2 cache 8 experts, layer 1 is not cached
    const std::vector<int32_t> n_expert = { 8, 0, 8 };
    const std::vector<llama_moe_profile_entry> in = { { 2, 5, 9.5f }, { 0, 1, 3.25f }, { 0, 7, 3.25f }, { 2, 0, 0.125f }, { 0, 0, 12.0f } };
    CHECK(llama_moe_profile_write(path, 3, in));
    std::vector<llama_moe_profile_entry> out;
    CHECK(llama_moe_profile_load(path, n_expert, out));
    CHECK(out.size() == in.size());
    for (size_t i = 0; i < out.size() && i < in.size(); ++i) {
        CHECK(out[i].il == in[i].il && out[i].e == in[i].e && out[i].heat == in[i].heat);
    }
    // no temporary left behind
    CHECK(fopen((path + ".tmp").c_str(), "r") == nullptr);

    // order: heat descending, ties (0 1 / 0 7) in file order
    const std::vector<llama_moe_profile_entry> o = llama_moe_profile_ram_order(out);
    const int want[5][2] = { { 0, 0 }, { 2, 5 }, { 0, 1 }, { 0, 7 }, { 2, 0 } };
    CHECK(o.size() == 5);
    for (size_t i = 0; i < o.size() && i < 5; ++i) {
        CHECK(o[i].il == want[i][0] && o[i].e == want[i][1]);
    }

    // refused: another layer count, an uncached layer, an expert out of range, a duplicate, a negative id, a cut line, no entries, no file
    const char * bad[] = {
        "moe-cache-profile 1 4\n0 1 1\n",
        "moe-cache-profile 1 3\n1 0 1\n",
        "moe-cache-profile 1 3\n0 8 1\n",
        "moe-cache-profile 1 3\n0 1 1\n2 3 1\n0 1 0.5\n",
        "moe-cache-profile 1 3\n0 -1 1\n",
        "moe-cache-profile 1 3\n0 1 1\n2 3\n",
        "moe-cache-profile 1 3\n",
        "moe-cache-profile 2 3\n0 1 1\n",
        "garbage\n",
    };
    for (const char * t : bad) {
        write_text(path, t);
        out.assign(1, { 0, 0, 0.0f });
        CHECK(!llama_moe_profile_load(path, n_expert, out));
        CHECK(out.empty());
    }
    remove(path.c_str());
    CHECK(!llama_moe_profile_load(path, n_expert, out));
    CHECK(!llama_moe_profile_write(path + ".no/such/dir", 3, in));
}

static void test_reserve() {
    const int32_t n_expert = 8;
    // two layouts: layers 0 and 1 {1000, 2000, 1500} (36864 bytes per slot), layer 2 {3000, 3000} (24576), layer 3 not cached;
    // the budget split like the experts: 8 + 4 slots
    const std::vector<std::vector<size_t>> sizes = { { 1000, 2000, 1500 }, { 1000, 2000, 1500 }, { 3000, 3000 }, {} };
    llama_moe_host_cache hc(ggml_backend_cpu_buffer_type(), 393216, LLAMA_MOE_HOST_EXCLUSIVE, n_expert, sizes, 4*36864);
    CHECK(hc.n_slots() == 12);

    // held before the fill: (0, 3), skipped by the fill
    const int32_t s03 = hc.reserve(0, 3, false);
    CHECK(s03 >= 0);
    hc.publish(s03);

    std::vector<llama_moe_profile_entry> prof;
    // layer 2 (4 slots) gets its 4 hottest, layer 3 (uncached) nothing, layers 0 / 1 share 8 slots (7 after (0, 3))
    for (int32_t e = 0; e < n_expert; ++e) {
        prof.push_back({ 2, e, 100.0f - e });
        prof.push_back({ 3, e, 200.0f });
        prof.push_back({ 0, e, 50.0f - e });
        prof.push_back({ 1, e, 49.5f - e });
    }
    const std::vector<llama_moe_profile_entry> order = llama_moe_profile_ram_order(prof);
    std::set<std::pair<int32_t, int32_t>> taken;
    std::vector<int32_t> slots;
    const size_t n = llama_moe_profile_reserve(hc, order,
        [](int32_t il, int32_t e) { return !(il == 1 && e == 0); }, // the caller refuses (1, 0) (e.g. the device holds it)
        [&](size_t i, int32_t s) {
            CHECK(s >= 0);
            taken.insert({ order[i].il, order[i].e });
            slots.push_back(s);
            for (int ip = 0; ip < hc.n_tensors(s); ++ip) {
                hc.set_lead(s, ip, 0);
                memset(hc.area(s, ip), 0x40 + order[i].e, hc.expert_size(s, ip));
            }
            hc.publish(s);
        });
    CHECK(n == 11 && taken.size() == 11);
    // layer 2: experts 0..3; layers 0 / 1 by heat 50, 49.5, 49, 48.5, ... minus (0, 3) held and (1, 0) refused
    for (int32_t e = 0; e < 4; ++e) {
        CHECK(taken.count({ 2, e }) == 1);
    }
    for (int32_t e = 4; e < n_expert; ++e) {
        CHECK(taken.count({ 2, e }) == 0);
    }
    const std::set<std::pair<int32_t, int32_t>> l01 = { { 0, 0 }, { 0, 1 }, { 1, 1 }, { 0, 2 }, { 1, 2 }, { 1, 3 }, { 0, 4 } };
    for (const auto & k : l01) {
        CHECK(taken.count(k) == 1);
    }
    CHECK(taken.count({ 0, 3 }) == 0 && taken.count({ 1, 0 }) == 0 && taken.count({ 1, 4 }) == 0);
    for (int32_t e = 0; e < n_expert; ++e) {
        CHECK(taken.count({ 3, e }) == 0);
    }
    // every filled expert is found with its bytes, (0, 3) is still held (nothing was evicted)
    for (const auto & [il, e] : taken) {
        const int32_t s = hc.find(il, e, false);
        CHECK(s >= 0);
        if (s >= 0) {
            CHECK(hc.data(s, 0)[0] == 0x40 + e);
        }
    }
    CHECK(hc.find(0, 3, false) == s03);
    CHECK(hc.st.evicted == 0);

    // a second pass finds no free slot and takes nothing
    const size_t n2 = llama_moe_profile_reserve(hc, order, [](int32_t, int32_t) { return true; }, [&](size_t, int32_t) { CHECK(false); });
    CHECK(n2 == 0);
}

// k19-ramprof2: profile-filled slots are guesses (set_spec): a reservation takes an unused one as if it were free, before any CLOCK
// eviction (finished layers of a prompt pass first, else the layers before the reserving one), never a pinned one, never for the
// profile's own fill (llama_moe_profile_reserve); a use (touching find, read, write) keeps it
static void test_spec() {
    const int32_t n_expert = 8;
    const std::vector<std::vector<size_t>> sizes = { { 1000, 2000 }, { 1000, 2000 }, { 1000, 2000 }, { 1000, 2000 } };
    llama_moe_host_cache hc(ggml_backend_cpu_buffer_type(), 6*24576, LLAMA_MOE_HOST_INCLUSIVE, n_expert, sizes, 6*24576);
    CHECK(hc.n_slots() == 6);
    std::vector<llama_moe_profile_entry> prof;
    for (int32_t e = 0; e < 2; ++e) {
        for (int32_t il = 0; il < 4; ++il) {
            prof.push_back({ il, e, 100.0f - 10*e - il });
        }
    }
    const std::vector<llama_moe_profile_entry> order = llama_moe_profile_ram_order(prof);
    auto fill = [&](size_t i, int32_t s) {
        for (int ip = 0; ip < hc.n_tensors(s); ++ip) {
            hc.set_lead(s, ip, 0);
            memset(hc.area(s, ip), 0x40 + order[i].il*8 + order[i].e, hc.expert_size(s, ip));
        }
        hc.publish(s);
        hc.set_spec(s);
    };
    CHECK(llama_moe_profile_reserve(hc, order, [](int32_t, int32_t) { return true; }, fill) == 6);
    CHECK(hc.n_spec_slots() == 6 && hc.st.spec_set == 6);
    // (0,0) (1,0) (2,0) (3,0) (0,1) (1,1) held; the profile's own fill never takes a guess
    CHECK(llama_moe_profile_reserve(hc, order, [](int32_t, int32_t) { return true; }, [&](size_t, int32_t) { CHECK(false); }) == 0);
    const int32_t s00 = hc.find(0, 0, false);
    const int32_t s11 = hc.find(1, 1, false);
    const int32_t s30 = hc.find(3, 0, false);
    CHECK(s00 >= 0 && s11 >= 0 && s30 >= 0 && hc.is_spec(s00));
    // uses: a touching find and a read keep their slots
    CHECK(hc.find(0, 0, true) == s00 && !hc.is_spec(s00));
    hc.mark_read(s11);
    CHECK(!hc.is_spec(s11) && hc.n_spec_slots() == 4 && hc.st.spec_used == 2);
    // a non-evicting reservation of layer 2 takes the unused guess of layer 1 ((1,1) was used)
    const int32_t a = hc.reserve(2, 5, false);
    CHECK(a >= 0 && hc.find(1, 0, false) < 0 && hc.find(1, 1, false) == s11);
    if (a >= 0) {
        hc.abort(a);
    }
    // the freed slot is free again: the next reservation takes it, no guess
    const int32_t b = hc.reserve(2, 5, false);
    CHECK(b == a && hc.st.spec_taken == 1);
    if (b >= 0) {
        hc.publish(b);
    }
    // a prompt pass at layer 3: finished layers first, a pinned guess ((3,0), routed by this call) is skipped
    hc.set_stream_layer(3);
    hc.begin_call();
    hc.pin(s30);
    const int32_t c = hc.reserve(0, 6, false);
    CHECK(c >= 0 && hc.find(2, 0, false) < 0 && hc.find(3, 0, false) == s30);
    if (c >= 0) {
        hc.publish(c);
    }
    hc.set_stream_layer(-1);
    hc.begin_call();
    // take_spec = false: no free slot, no guess taken
    CHECK(hc.reserve(1, 7, false, 0, false) < 0);
    CHECK(hc.n_spec_slots() == 2 && hc.st.spec_taken == 2);
    // an evicting reservation also takes a guess before the CLOCK: the used (0,0) and (1,1) stay
    const int32_t d = hc.reserve(2, 6, true);
    CHECK(d >= 0 && hc.find(0, 1, false) < 0 && hc.find(0, 0, false) == s00 && hc.find(1, 1, false) == s11);
    if (d >= 0) {
        hc.publish(d);
    }
    CHECK(hc.n_spec_slots() == 1 && hc.st.spec_taken == 3 && hc.st.evicted == 3);
    // the bytes of the kept guesses and uses are unchanged
    CHECK(hc.data(s00, 0)[0] == 0x40 + 0 && hc.data(s30, 1)[0] == 0x40 + 3*8);
    // room for a request: the full tier frees its last unused guess ((3,0)), nothing else
    CHECK(hc.n_free() == 0);
    CHECK(hc.drop_spec(5) == 1 && hc.n_free() == 1 && hc.n_spec_slots() == 0 && hc.st.spec_dropped == 1);
    CHECK(hc.find(3, 0, false) < 0 && hc.find(0, 0, false) == s00 && hc.find(1, 1, false) == s11);
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const std::string path = argc > 1 ? argv[1] : "test-moe-ram-profile.tmp.prof";
    test_file(path);
    test_reserve();
    test_spec();
    if (n_fail > 0) {
        fprintf(stderr, "test-moe-ram-profile: %d checks failed\n", n_fail);
        return 1;
    }
    printf("test-moe-ram-profile: OK\n");
    return 0;
}
