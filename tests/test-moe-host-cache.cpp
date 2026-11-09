// llama_moe_host_cache (RAM tier of the MoE cache): slots, CLOCK eviction of decayed counts, pins of the routed experts, leases of a plan,
// pieces, deferred frees, allocation failure; the weight store with a tier (reads ahead skip the held bytes, uploads from the tier, offers);
// reads into slots with faults and VERIFY; a decode simulation of the exclusive and inclusive policies against a device LRU (byte-exact)
// usage: test-moe-host-cache [scratch file path]

#include "../src/llama-moe-host-cache.h"
#include "../src/llama-weight-store.h"
#include "../ggml/src/ggml-backend-impl.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#ifdef _WIN32
int main() {
    printf("test-moe-host-cache: skipped on Windows\n");
    return 0;
}
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static int n_fail = 0;

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

// a host buffer type whose allocations fail after `limit` buffers
static int g_alloc_limit = 1 << 30;
static int g_alloc_count = 0;
static ggml_backend_buffer_t failing_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    if (g_alloc_count++ >= g_alloc_limit) {
        return nullptr;
    }
    ggml_backend_buffer_type_t cpu = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_t b = cpu->iface.alloc_buffer(cpu, size);
    if (b != nullptr) {
        b->buft = buft; // a buffer of this type
    }
    return b;
}

// a host buffer type whose allocations return buffers of another type (like pinned memory falling back to pageable)
static ggml_backend_buffer_t other_type_alloc(ggml_backend_buffer_type_t, size_t size) {
    ggml_backend_buffer_type_t cpu = ggml_backend_cpu_buffer_type();
    return cpu->iface.alloc_buffer(cpu, size);
}

struct fault {
    std::atomic<int> mode { 0 }; // 0 none, 4 corrupt one byte of the second read, 5 EIO always, 6 slow (2 ms per read)
    std::atomic<int> calls { 0 };
};

static int64_t read_hook(int fd, void * buf, size_t n, uint64_t off, bool, void * ud) {
    fault * f = (fault *) ud;
    const int c = f->calls.fetch_add(1);
    if (f->mode.load() == 5) {
        return -EIO;
    }
    if (f->mode.load() == 6) {
        usleep(2000);
    }
    const ssize_t r = pread(fd, buf, n, (off_t) off);
    if (r < 0) {
        return -errno;
    }
    if (f->mode.load() == 4 && r > 0 && c == 1) {
        ((uint8_t *) buf)[r/2] ^= 0x5a;
    }
    return r;
}

// the auto decision on the numbers the runs logged (MiB): e streamed experts, v device slots, mapped host weights, MemAvailable; one
// token routes 10 experts of about 1.55 MiB in each of 48 layers
static size_t mib(double x) { return (size_t) (x*1048576.0); }

static void test_decide() {
    const size_t e = mib(38000.0), mapped = mib(38341.0), tok = mib(48*10*1.55);
    auto decide = [&](double v, double avail, int64_t host_mib = -1, int pol = -1, bool force = false, bool disk_incl = false) {
        return llama_moe_host_decide(e, mib(v), mapped, mib(avail), tok, host_mib, pol, force, disk_incl);
    };
    // k19-disktier: below the minimum on a disk tier (mapped > avail): on, inclusive, the auto budget; off where RAM holds the mapped
    // weights, with a forced exclusive policy, without routing (token_bytes 0) or with the kill switch (disk_incl false, the cases below)
    {
        const llama_moe_host_plan p = decide(1020.4, 29212.6, -1, -1, false, true);
        CHECK(p.on && p.policy == 0 && !p.spare);
        CHECK(p.budget > mib(22790) && p.budget < mib(22810));
        CHECK( decide(LLAMA_MOE_HOST_MIN_TOKENS*744.0 - 8, 29000, -1, -1, false, true).on);
        CHECK(!decide(LLAMA_MOE_HOST_MIN_TOKENS*744.0 - 8, 60000, -1, -1, false, true).on);
        CHECK(!decide(1020.4, 29212.6, -1, 1, false, true).on);
        CHECK( decide(1020.4, 29212.6, -1, 0, false, true).on);
        CHECK(!llama_moe_host_decide(e, mib(1020.4), mapped, mib(29212.6), 0, -1, -1, false, true).on);
        // above the minimum: unchanged (exclusive)
        const llama_moe_host_plan l = decide(5114.7, 29299.5, -1, -1, false, true);
        CHECK(l.on && l.policy == 1 && l.budget > mib(22858) && l.budget < mib(22878));
        // the default argument is the new rule
        CHECK(llama_moe_host_decide(e, mib(1020.4), mapped, mib(29212.6), tok, -1, -1, false).on);
        // a budget below LLAMA_MOE_HOST_DISK_MIN_FRAC of the experts: off (22.8 of 38 GiB = 60%: on at 0.6, off at 0.61; the Kaggle
        // 33% tier: off; tc_n2's L4 budget, 35.6 of 57.6 GiB: on)
        CHECK( llama_moe_host_decide(e, mib(1020.4), mapped, mib(29212.6), tok, -1, -1, false, true, 0.59).on);
        CHECK(!llama_moe_host_decide(e, mib(1020.4), mapped, mib(29212.6), tok, -1, -1, false, true, 0.61).on);
        const size_t ek = mib(18726.0), mk = mib(19705.3), tk = mib(590.0);
        CHECK(!llama_moe_host_decide(ek, mib(637.2), mk, mib(17376.7 - 6144.0), tk, -1, -1, false).on);
        CHECK( llama_moe_host_decide(ek, mib(637.2), mk, mib(17376.7 - 2048.0), tk, -1, -1, false).on);
        CHECK( llama_moe_host_decide(mib(57587.5), mib(3754.4), mib(57909.6), mib(51559.3 - 6144.0), mib(1137.0), -1, -1, false).on);
    }
    // L4 8 GB: the device slots hold the experts of 1.4 tokens (measured with the exclusive tier: no gain, decode after 8k -14%) -> off
    // under the kill switch
    {
        const llama_moe_host_plan p = decide(1020.4, 29212.6);
        CHECK(!p.on);
        CHECK(p.tokens > 1.3 && p.tokens < 1.5);
        CHECK(p.policy == 1);
        CHECK(p.budget > mib(22790) && p.budget < mib(22810)); // the logged auto budget 22800.3 MiB
        // forced, an explicit budget: on with it
        const llama_moe_host_plan f = decide(1020.4, 29212.6, -1, -1, true);
        CHECK(f.on && f.budget == p.budget && f.policy == 1);
        const llama_moe_host_plan q = decide(1020.4, 29212.6, 4096);
        CHECK(q.on && q.budget == mib(4096) && q.policy == 1);
    }
    // L4 12 GB (6.9 tokens, forced on: 8k prompt -27%), G4 16 GB (12.4 tokens, -40%), Kaggle T4 (10.7 tokens, -48%) -> on, exclusive, the
    // logged budgets (the path the measured runs took)
    {
        const llama_moe_host_plan l = decide(5114.7, 29299.5);
        CHECK(l.on && l.policy == 1 && l.budget > mib(22858) && l.budget < mib(22878)); // logged 22868.1
        const llama_moe_host_plan g = decide(9210.8, 29058.2);
        CHECK(g.on && g.policy == 1 && g.budget > mib(22670) && g.budget < mib(22690)); // logged 22679.8
        const llama_moe_host_plan k = decide(7930.4, 29895.2);
        CHECK(k.on && k.policy == 1 && k.budget > mib(23323) && k.budget < mib(23343)); // logged 23333.1
    }
    // the boundary is the minimum number of tokens, whatever the RAM
    {
        CHECK(!decide(LLAMA_MOE_HOST_MIN_TOKENS*744.0 - 8, 29000).on);
        CHECK( decide(LLAMA_MOE_HOST_MIN_TOKENS*744.0 + 8, 29000).on);
        CHECK(!decide(LLAMA_MOE_HOST_MIN_TOKENS*744.0 - 8, 60000).on);
        CHECK( decide(LLAMA_MOE_HOST_MIN_TOKENS*744.0 + 8, 16000).on);
    }
    // the device slots hold every streamed expert (--fit): off, also forced; an explicit budget still goes on
    {
        CHECK(!decide(38000.0, 29000).on);
        CHECK(!decide(38000.0, 29000, -1, -1, true).on);
        CHECK( decide(38000.0, 29000, 1024).on);
    }
    // RAM for every expert: inclusive, the budget capped at the experts
    {
        const llama_moe_host_plan p = decide(4000, 64000);
        CHECK(p.on && p.policy == 0 && p.budget == e + mib(64));
    }
    // exclusive: the budget is capped at the experts too (the spare slots keep copies of experts the device holds, the first to go)
    CHECK(decide(20000, 64000, -1, 1).budget == e + mib(64));
    CHECK(decide(20000, 64000, -1, 1).spare);
    // V16 / R29 (9.2 GiB of slots, 22.2 GiB budget) and V32 / R29 (25.6 GiB, 22.6 GiB): spare room only at V32
    CHECK(!decide(9210.8, 29058.2).spare && decide(9210.8, 29058.2).policy == 1);
    CHECK( decide(25598.0, 29039.9).spare && decide(25598.0, 29039.9).policy == 1);
    // no RAM to spare, off by 0 (also forced)
    CHECK(!decide(9000, 4000).on);                      // below the room: no budget
    CHECK( decide(9000, 5000).on && decide(9000, 5000).budget < mib(1024));
    CHECK(!decide(9000, 29000, 0, -1, true).on);
    // a forced policy
    CHECK(decide(9210.8, 29058.2, -1, 0).policy == 0);
    // no expert routing known (token_bytes 0): auto off, forced on
    CHECK(!llama_moe_host_decide(e, mib(9000), mapped, mib(29000), 0, -1, -1, false, false).on);
    CHECK( llama_moe_host_decide(e, mib(9000), mapped, mib(29000), 0, -1, -1, true, false).on);
}

static void test_slots(ggml_backend_t backend) {
    const int32_t n_expert = 8;
    // two layouts: layers 0 and 1 {1000, 2000, 1500} (36864 bytes per slot), layer 2 {3000, 3000} (24576), layer 3 not cached
    const std::vector<std::vector<size_t>> sizes = { { 1000, 2000, 1500 }, { 1000, 2000, 1500 }, { 3000, 3000 }, {} };
    {
        llama_moe_host_cache hc(ggml_backend_cpu_buffer_type(), 393216, LLAMA_MOE_HOST_EXCLUSIVE, n_expert, sizes, 4*36864);
        CHECK(hc.n_slots() == 12); // the budget split like the experts: 8 + 4
        CHECK(hc.exclusive());
        CHECK(hc.caches(0) && hc.caches(2) && !hc.caches(3));
        CHECK(hc.pinned_bytes() == 0); // memory on first use
        printf("slots: %s\n", hc.describe().c_str());

        // reserve, fill, publish, find
        std::vector<int32_t> s0(8);
        for (int32_t e = 0; e < 8; ++e) {
            s0[e] = hc.reserve(0, e);
            CHECK(s0[e] >= 0);
            CHECK(hc.find(0, e) < 0 && hc.contains(0, e)); // being filled: not found
            for (int ip = 0; ip < 3; ++ip) {
                CHECK(((uintptr_t) hc.area(s0[e], ip)) % 4096 == 0);
                CHECK(hc.area_size(s0[e], ip) >= hc.expert_size(s0[e], ip) + 8192);
                hc.set_lead(s0[e], ip, 7*ip);
                memset(hc.area(s0[e], ip) + 7*ip, 0x10*e + ip, hc.expert_size(s0[e], ip));
            }
            hc.publish(s0[e]);
            CHECK(hc.find(0, e) == s0[e]);
            CHECK(hc.data(s0[e], 2)[0] == 0x10*e + 2);
        }
        CHECK(hc.pinned_bytes() >= 8*36864);
        CHECK(hc.reserve(0, 3) < 0); // held already

        // CLOCK over decayed counts: the experts used more survive
        for (int k = 0; k < 3; ++k) {
            for (int32_t e = 0; e < 4; ++e) {
                hc.find(0, e);
            }
        }
        for (int32_t e = 0; e < 4; ++e) {
            const int32_t s = hc.reserve(1, e);
            CHECK(s >= 0);
            hc.publish(s);
        }
        for (int32_t e = 0; e < 4; ++e) {
            CHECK(hc.find(0, e, false) >= 0);
            CHECK(hc.find(0, e + 4, false) < 0);
        }
        CHECK(hc.st.evicted == 4);

        // routed by the current call: pinned, never evicted
        hc.begin_call();
        for (int32_t e = 0; e < 4; ++e) {
            hc.pin(hc.find(0, e, false));
            hc.pin(hc.find(1, e, false));
        }
        const uint64_t no_slot = hc.st.no_slot;
        CHECK(hc.reserve(1, 7) < 0);
        CHECK(hc.st.no_slot == no_slot + 1);
        hc.begin_call();
        int32_t s17 = hc.reserve(1, 7);
        CHECK(s17 >= 0);
        hc.abort(s17);
        CHECK(!hc.contains(1, 7));
        s17 = hc.reserve(1, 7); // the aborted slot is free again
        CHECK(s17 >= 0);
        hc.publish(s17);

        // leased by the plan: never evicted until the next plan
        hc.new_plan();
        for (int32_t il = 0; il < 2; ++il) {
            for (int32_t e = 0; e < 8; ++e) {
                const int32_t s = hc.find(il, e, false);
                if (s >= 0) {
                    hc.lease(s);
                }
            }
        }
        CHECK(hc.reserve(1, 6) < 0);
        hc.new_plan();
        s17 = hc.reserve(1, 6);
        CHECK(s17 >= 0);

        // pieces: in order per tensor, the slot is published when every tensor is complete; others ignored; a plan drops incomplete ones
        std::vector<uint8_t> src(2000);
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = (uint8_t) (i*7 + 3);
        }
        CHECK(!hc.fill_piece(s17, 0, 0, src.data(), 600));
        CHECK(!hc.fill_piece(s17, 0, 700, src.data(), 300)); // gap: ignored
        CHECK(!hc.fill_piece(s17, 0, 600, src.data() + 600, 400));
        CHECK(!hc.fill_piece(s17, 1, 0, src.data(), 2000));
        CHECK(hc.filling(s17));
        CHECK(hc.fill_piece(s17, 2, 0, src.data(), 1500));
        CHECK(!hc.filling(s17) && hc.find(1, 6, false) == s17);
        CHECK(memcmp(hc.data(s17, 0), src.data(), 1000) == 0 && memcmp(hc.data(s17, 2), src.data(), 1500) == 0);
        hc.release(s17);
        const int32_t s15 = hc.reserve(1, 5);
        CHECK(s15 >= 0);
        CHECK(!hc.fill_piece(s15, 0, 0, src.data(), 1000));
        hc.new_plan();
        CHECK(!hc.contains(1, 5)); // incomplete: aborted

        // released while copies that read it are not committed: free only after the commit
        llama_moe_host_cache hc2(ggml_backend_cpu_buffer_type(), 4*24576, LLAMA_MOE_HOST_INCLUSIVE, n_expert, { {}, {}, { 3000, 3000 } });
        CHECK(hc2.n_slots() == 4);
        std::vector<int32_t> s2;
        for (int32_t e = 0; e < 4; ++e) {
            s2.push_back(hc2.reserve(2, e));
            hc2.publish(s2.back());
        }
        hc2.mark_read(s2[1]);
        hc2.release(s2[1]);
        CHECK(!hc2.contains(2, 1));
        CHECK(hc2.reserve(2, 5, false) < 0); // deferred
        hc2.commit(backend);
        const int32_t s25 = hc2.reserve(2, 5, false);
        CHECK(s25 == s2[1]);

        // copy back from the device (here the CPU backend): written by an async copy, found once the copy is done
        ggml_init_params ip = { /*.mem_size =*/ 4*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * bank = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, 4*3000);
        ggml_backend_buffer_t bbuf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<uint8_t> pat(4*3000);
        for (size_t i = 0; i < pat.size(); ++i) {
            pat[i] = (uint8_t) (i*13 + 1);
        }
        ggml_backend_tensor_set(bank, pat.data(), 0, pat.size());
        hc2.release(s25);
        const int32_t sw = hc2.reserve(2, 6, false);
        CHECK(sw >= 0);
        for (int t = 0; t < 2; ++t) {
            hc2.set_lead(sw, t, 0);
            ggml_backend_tensor_get_async(backend, bank, hc2.area(sw, t), (size_t) (1 + t)*3000, 3000);
        }
        hc2.mark_write(sw);
        CHECK(hc2.contains(2, 6));
        hc2.commit(backend);
        CHECK(hc2.find(2, 6) == sw);
        CHECK(memcmp(hc2.data(sw, 0), pat.data() + 3000, 3000) == 0 && memcmp(hc2.data(sw, 1), pat.data() + 6000, 3000) == 0);
        // copies back that nobody looked up again are evictable once committed (not frozen in the writing state)
        llama_moe_host_cache hc3(ggml_backend_cpu_buffer_type(), 2*24576, LLAMA_MOE_HOST_EXCLUSIVE, n_expert, { {}, {}, { 3000, 3000 } });
        for (int32_t e = 0; e < 2; ++e) {
            const int32_t s = hc3.reserve(2, e);
            CHECK(s >= 0);
            ggml_backend_tensor_get_async(backend, bank, hc3.area(s, 0), 0, 3000);
            ggml_backend_tensor_get_async(backend, bank, hc3.area(s, 1), 3000, 3000);
            hc3.mark_write(s);
        }
        CHECK(hc3.reserve(2, 2) < 0); // not committed yet
        hc3.commit(backend);
        const int32_t s32 = hc3.reserve(2, 2);
        CHECK(s32 >= 0);
        CHECK(hc3.st.evicted == 1);
        ggml_backend_buffer_free(bbuf);
        ggml_free(ctx);
    }

    // pinned memory runs out: the tier keeps the slots it has and evicts among them
    {
        ggml_backend_buffer_type fb = *ggml_backend_cpu_buffer_type();
        fb.iface.alloc_buffer = failing_alloc;
        g_alloc_limit = 1;
        g_alloc_count = 0;
        llama_moe_host_cache hc(&fb, 8*24576, LLAMA_MOE_HOST_EXCLUSIVE, n_expert, { {}, {}, { 3000, 3000 } }, 2*24576);
        CHECK(hc.n_slots() == 8);
        for (int32_t e = 0; e < 5; ++e) {
            const int32_t s = hc.reserve(2, e);
            CHECK(s >= 0);
            if (s >= 0) {
                hc.publish(s);
            }
        }
        CHECK(hc.st.alloc_fail == 1);
        CHECK(hc.pinned_bytes() < 3*24576);
        int held = 0;
        for (int32_t e = 0; e < 5; ++e) {
            held += hc.find(2, e, false) >= 0;
        }
        CHECK(held == 2);
        g_alloc_limit = 1 << 30;
    }

    // a buffer of another type (e.g. pageable memory instead of pinned) is a failure: the tier stops growing
    {
        ggml_backend_buffer_type fb = *ggml_backend_cpu_buffer_type();
        fb.iface.alloc_buffer = other_type_alloc;
        llama_moe_host_cache hc(&fb, 4*24576, LLAMA_MOE_HOST_EXCLUSIVE, n_expert, { {}, {}, { 3000, 3000 } });
        CHECK(hc.reserve(2, 0) < 0);
        CHECK(hc.st.alloc_fail == 1 && hc.pinned_bytes() == 0);
    }

    // pin guard: pinning fails -> pageable chunks (the tier keeps growing); a pin cap -> pageable past it
    {
        ggml_backend_buffer_type fb = *ggml_backend_cpu_buffer_type();
        fb.iface.alloc_buffer = other_type_alloc;
        llama_moe_host_cache hc(&fb, 4*24576, LLAMA_MOE_HOST_EXCLUSIVE, n_expert, { {}, {}, { 3000, 3000 } }, 0, ggml_backend_cpu_buffer_type());
        for (int32_t e = 0; e < 4; ++e) {
            const int32_t s = hc.reserve(2, e);
            CHECK(s >= 0);
            if (s >= 0) {
                hc.publish(s);
            }
        }
        CHECK(hc.st.alloc_fail == 0 && hc.pinned_bytes() == 0 && hc.pageable_bytes() > 0 && hc.st.pin_fallbacks >= 1);
    }
    {
        ggml_backend_buffer_type fb = *ggml_backend_cpu_buffer_type();
        fb.iface.alloc_buffer = failing_alloc;
        g_alloc_limit = 1 << 30;
        g_alloc_count = 0;
        // chunks of 2 slots (2*24576 + 4 KiB each), at most one chunk pinned
        llama_moe_host_cache hc(&fb, 8*24576, LLAMA_MOE_HOST_EXCLUSIVE, n_expert, { {}, {}, { 3000, 3000 } }, 2*24576, ggml_backend_cpu_buffer_type(),
                2*24576 + 4096);
        for (int32_t e = 0; e < 8; ++e) {
            const int32_t s = hc.reserve(2, e);
            CHECK(s >= 0);
            if (s >= 0) {
                hc.publish(s);
            }
        }
        CHECK(g_alloc_count == 1);
        CHECK(hc.pinned_bytes() == 2*24576 + 4096 && hc.pageable_bytes() == 3*(2*24576 + 4096) && hc.st.pin_fallbacks == 3);
        int held = 0;
        for (int32_t e = 0; e < 8; ++e) {
            held += hc.find(2, e, false) >= 0;
        }
        CHECK(held == 8);
        // no pageable memory either: the tier stops growing at the cap
        g_alloc_count = 0;
        llama_moe_host_cache hc2(&fb, 8*24576, LLAMA_MOE_HOST_EXCLUSIVE, n_expert, { {}, {}, { 3000, 3000 } }, 2*24576, nullptr, 2*24576 + 4096);
        for (int32_t e = 0; e < 3; ++e) {
            const int32_t s = hc2.reserve(2, e);
            if (s >= 0) {
                hc2.publish(s);
            }
        }
        CHECK(hc2.st.alloc_fail == 1 && hc2.pinned_bytes() == 2*24576 + 4096 && hc2.pageable_bytes() == 0);
    }

    // a slot filled by an asynchronous read: pieces, new_plan and eviction leave it alone; min_age keeps the slots of recent commits
    {
        llama_moe_host_cache hc(ggml_backend_cpu_buffer_type(), 2*24576, LLAMA_MOE_HOST_INCLUSIVE, n_expert, { {}, {}, { 3000, 3000 } });
        CHECK(hc.n_slots() == 2);
        const int32_t sa = hc.reserve(2, 5);
        CHECK(sa >= 0);
        hc.set_async(sa, 1);
        CHECK(hc.is_async(sa));
        uint8_t junk[3000] = {};
        CHECK(!hc.fill_piece(sa, 0, 0, junk, 3000));
        hc.new_plan();
        CHECK(hc.filling(sa) && hc.contains(2, 5) && hc.find(2, 5, false) < 0);
        const int32_t sb = hc.reserve(2, 6);
        CHECK(sb >= 0 && sb != sa);
        hc.publish(sb);
        const int32_t s7r = hc.reserve(2, 7); // evicts sb (the only valid slot), never sa
        CHECK(s7r == sb);
        CHECK(hc.contains(2, 5) && hc.filling(sa));
        hc.publish(sa);
        CHECK(!hc.is_async(sa) && hc.find(2, 5, false) == sa);
        // sa read by the last commit: min_age only skips slots whose copies are not known to be done (the CPU backend completes them at
        // the commit), so the reservation takes a slot without waiting
        hc.mark_read(sa);
        hc.commit(backend);
        const int32_t s7 = hc.slot_of_key(2, 7);
        if (s7 >= 0) {
            hc.abort(s7);
        }
        const int32_t sb2 = hc.reserve(2, 3);
        CHECK(sb2 >= 0);
        if (sb2 >= 0) {
            hc.publish(sb2);
            hc.mark_read(sb2);
            hc.commit(backend);
        }
        CHECK(hc.reserve(2, 4, true, 4) >= 0);
        CHECK(hc.st.waits == 0);
    }

    // k19-ramevict: a prompt pass streaming layer L evicts from the layers <= L first, then (none left) from any layer. The rule is
    // off by default (LLAMA_MOE_HOST_EVICT_DONE=1 on, read when the cache is built): the default is checked, then the rule with it on
    {
        const std::vector<std::vector<size_t>> lsz(6, std::vector<size_t>{ 3000, 3000 });
        const char * prev = getenv("LLAMA_MOE_HOST_EVICT_DONE");
        const std::string prev_s = prev ? prev : "";
        unsetenv("LLAMA_MOE_HOST_EVICT_DONE");
        {
            llama_moe_host_cache hd(ggml_backend_cpu_buffer_type(), 6*24576, LLAMA_MOE_HOST_EXCLUSIVE, n_expert, lsz);
            CHECK(!hd.evict_done_on());
        }
        setenv("LLAMA_MOE_HOST_EVICT_DONE", "1", 1);
        llama_moe_host_cache hc(ggml_backend_cpu_buffer_type(), 6*24576, LLAMA_MOE_HOST_EXCLUSIVE, n_expert, lsz);
        if (prev) {
            setenv("LLAMA_MOE_HOST_EVICT_DONE", prev_s.c_str(), 1);
        } else {
            unsetenv("LLAMA_MOE_HOST_EVICT_DONE");
        }
        CHECK(hc.evict_done_on());
        CHECK(hc.n_slots() == 6);
        for (int32_t il = 0; il < 6; ++il) {
            const int32_t s = hc.reserve(il, 0);
            CHECK(s >= 0);
            hc.publish(s);
        }
        // layers 3..5 used more: the plain CLOCK would take layer 0 first anyway, so make the finished layers the hot ones
        for (int k = 0; k < 4; ++k) {
            hc.find(0, 0); hc.find(1, 0); hc.find(2, 0);
        }
        hc.set_stream_layer(2);
        for (int32_t e = 1; e <= 3; ++e) {
            const int32_t s = hc.reserve(2, e);
            CHECK(s >= 0);
            hc.publish(s);
        }
        CHECK(hc.st.done_evicted == 3 && hc.st.ahead_evicted == 0 && hc.st.stream_evicted == 3);
        CHECK(hc.contains(3, 0) && hc.contains(4, 0) && hc.contains(5, 0)); // the layers still to come keep their experts
        // layer 0 streaming: nothing finished but layer 0, which holds nothing: the plain CLOCK takes an expert of a layer to come
        hc.set_stream_layer(0);
        const int32_t s = hc.reserve(0, 5);
        CHECK(s >= 0);
        CHECK(hc.st.ahead_evicted == 1 && hc.st.done_evicted == 3);
        hc.set_stream_layer(-1);
        hc.publish(s);
        const uint64_t ev = hc.st.stream_evicted;
        CHECK(hc.reserve(1, 6) >= 0);
        CHECK(hc.st.stream_evicted == ev); // decode: not counted, the plain CLOCK
    }

    // no budget
    {
        llama_moe_host_cache hc(ggml_backend_cpu_buffer_type(), 1000, LLAMA_MOE_HOST_EXCLUSIVE, n_expert, sizes);
        CHECK(hc.n_slots() == 0);
        CHECK(hc.reserve(0, 0) < 0);
        CHECK(hc.find(0, 0) < 0);
    }
}

// the tier of one MoE layer, like llama_moe_cache's adapter: expert granularity, prompts fill free slots from expert 0 of tensor 0 on
struct test_tier : llama_weight_tier {
    llama_moe_host_cache * hc;
    std::vector<const ggml_tensor *> srcs;
    std::set<int32_t> * device = nullptr; // exclusive: experts the device holds are not admitted
    std::set<int32_t> device_held;        // planning: bytes the device holds (not read ahead)
    uint64_t offered = 0;
    uint64_t marked  = 0;

    int index(const ggml_tensor * t) const {
        for (size_t i = 0; i < srcs.size(); ++i) {
            if (srcs[i] == t) {
                return (int) i;
            }
        }
        return -1;
    }
    void runs(const ggml_tensor * src, size_t off, size_t size, bool lease, std::vector<run> & out) override {
        const int ip = index(src);
        if (ip < 0) {
            out.push_back({ off, size, nullptr });
            return;
        }
        const size_t nb = src->nb[2];
        for (size_t o = off; o < off + size; ) {
            const int32_t e  = (int32_t) (o / nb);
            const size_t  e1 = std::min(off + size, (size_t) (e + 1)*nb);
            const int32_t s  = e < src->ne[2] ? hc->find(0, e, false) : -1;
            if (s >= 0) {
                if (lease) {
                    hc->lease(s);
                } else {
                    hc->mark_read(s);
                    marked++;
                }
                out.push_back({ o, e1 - o, hc->data(s, ip) + (o - (size_t) e*nb) });
            } else {
                const bool dev = lease && device_held.count(e) > 0;
                if (!out.empty() && out.back().ptr == nullptr && out.back().device == dev && out.back().off + out.back().size == o) {
                    out.back().size += e1 - o;
                } else {
                    out.push_back({ o, e1 - o, nullptr, dev });
                }
            }
            o = e1;
        }
    }
    void new_plan() override { hc->new_plan(); }
    uint64_t generation() const override { return hc->generation(); }
    void offer(const ggml_tensor * src, size_t off, size_t n, const uint8_t * p) override {
        const int ip = index(src);
        if (ip < 0) {
            return;
        }
        offered += n;
        const size_t nb = src->nb[2];
        const size_t end = std::min(off + n, (size_t) src->ne[2]*nb);
        for (size_t o = off; o < end; ) {
            const int32_t e  = (int32_t) (o / nb);
            const size_t  e1 = std::min(end, (size_t) (e + 1)*nb);
            const size_t  a  = o - (size_t) e*nb;
            int32_t s = -1;
            if (hc->contains(0, e)) {
                s = hc->find(0, e, false) >= 0 ? -1 : hc->slot_of_key(0, e);
            } else if (ip == 0 && a == 0 && (device == nullptr || !device->count(e))) {
                s = hc->reserve(0, e, false);
            }
            if (s >= 0 && hc->filling(s)) {
                hc->fill_piece(s, ip, a, p + (o - off), e1 - o);
            }
            o = e1;
        }
    }
    void uploaded(ggml_backend_t backend) override { hc->commit(backend); }
};

int main(int argc, char ** argv) {
    ggml_backend_t backend = ggml_backend_cpu_init();

    test_decide();
    test_slots(backend);

    const std::string path = argc > 1 ? argv[1] : "test-moe-host-cache.bin";
    const size_t file_size = (size_t) 8 << 20;
    std::vector<uint8_t> data(file_size);
    std::mt19937_64 rng(4321);
    for (size_t i = 0; i < file_size; i += 8) {
        const uint64_t v = rng();
        memcpy(data.data() + i, &v, std::min<size_t>(8, file_size - i));
    }
    {
        FILE * f = fopen(path.c_str(), "wb");
        if (!f || fwrite(data.data(), 1, file_size, f) != file_size) {
            fprintf(stderr, "cannot write %s\n", path.c_str());
            return 1;
        }
        fclose(f);
    }
    const int fd = open(path.c_str(), O_RDONLY);
    void * map = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        return 1;
    }
    ggml_backend_buffer_t src_buf = ggml_backend_cpu_buffer_from_ptr(map, file_size);

    // one MoE layer: 3 expert tensors of 16 experts, expert sizes that are not multiples of 4 KiB, at unaligned file offsets
    const int32_t n_expert = 16;
    struct tdesc { size_t off; int64_t ne0; };
    const std::vector<tdesc> descs = { { 4096 + 96, 37000 }, { (size_t) (1 << 20) + 40, 52011 }, { (size_t) (3 << 20) + 4000, 29999 } };
    ggml_init_params ip = { /*.mem_size =*/ 32*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx_src = ggml_init(ip);
    ggml_context * ctx_dst = ggml_init(ip);
    std::vector<ggml_tensor *> src, dst;
    for (const auto & d : descs) {
        ggml_tensor * s = ggml_new_tensor_3d(ctx_src, GGML_TYPE_I8, d.ne0, 1, n_expert);
        ggml_backend_tensor_alloc(src_buf, s, (uint8_t *) map + d.off);
        src.push_back(s);
        dst.push_back(ggml_new_tensor_3d(ctx_dst, GGML_TYPE_I8, d.ne0, 1, n_expert));
    }
    ggml_backend_buffer_t dst_buf = ggml_backend_alloc_ctx_tensors(ctx_dst, backend);
    auto ext = std::make_shared<llama_weight_extents>();
    {
        llama_weight_file f;
        f.path = path; f.map_addr = (const uint8_t *) map; f.map_size = file_size; f.size = file_size;
        ext->files.push_back(f);
        for (size_t i = 0; i < descs.size(); ++i) {
            llama_weight_extent e;
            e.file = 0; e.offs = descs[i].off; e.size = ggml_nbytes(src[i]);
            ext->tensors.emplace(src[i], e);
        }
    }
    const std::vector<std::vector<size_t>> sizes = { { src[0]->nb[2], src[1]->nb[2], src[2]->nb[2] } };
    size_t slot_bytes = 0;
    for (size_t s : sizes[0]) {
        slot_bytes += (s + 4095)/4096*4096 + 8192;
    }
    auto file_bytes = [&](int t, int32_t e) { return data.data() + descs[t].off + (size_t) e*src[t]->nb[2]; };
    auto check_dst = [&](const char * what) {
        for (int t = 0; t < 3; ++t) {
            std::vector<uint8_t> got(ggml_nbytes(src[t]));
            ggml_backend_tensor_get(dst[t], got.data(), 0, got.size());
            if (memcmp(got.data(), data.data() + descs[t].off, got.size()) != 0) {
                fprintf(stderr, "%s: tensor %d differs\n", what, t);
                n_fail++;
            }
        }
    };

    // prompts through the tier: the first fills the free slots from the ring, the next ones skip and upload the held experts
    for (int io : { LLAMA_WEIGHT_STREAM_IO_BUFFERED, LLAMA_WEIGHT_STREAM_IO_AUTO }) {
        llama_moe_host_cache hc(ggml_backend_cpu_buffer_type(), 6*slot_bytes, LLAMA_MOE_HOST_INCLUSIVE, n_expert, sizes);
        CHECK(hc.n_slots() == 6);
        test_tier tier;
        tier.hc = &hc;
        tier.srcs = { src[0], src[1], src[2] };
        llama_weight_store_params p;
        p.io = io; p.ring_size = (size_t) 4 << 20; p.n_threads = 3; p.chunk = (size_t) 256 << 10; p.ram_avail = (size_t) 1 << 30;
        llama_weight_store ws(ext, p);
        ws.set_tier(&tier);
        const ggml_tensor * plan[] = { src[0], src[1], src[2] };
        for (int g = 0; g < 3; ++g) {
            ggml_backend_buffer_clear(dst_buf, 0);
            ws.plan(plan, 3);
            for (int t = 0; t < 3; ++t) {
                CHECK(ws.upload(backend, src[t], dst[t], 0, 0, ggml_nbytes(src[t])));
            }
            ggml_backend_synchronize(backend);
            check_dst("tier prompt");
            if (g == 0) {
                for (int32_t e = 0; e < 6; ++e) {
                    const int32_t s = hc.find(0, e, false);
                    CHECK(s >= 0);
                    for (int t = 0; t < 3 && s >= 0; ++t) {
                        CHECK(memcmp(hc.data(s, t), file_bytes(t, e), src[t]->nb[2]) == 0);
                    }
                }
                CHECK(hc.find(0, 6, false) < 0); // no free slot left, nothing evicted
            }
        }
        const auto st = ws.get_stats();
        const size_t held_bytes = 6*(src[0]->nb[2] + src[1]->nb[2] + src[2]->nb[2]);
        CHECK(st.tier_uploaded == 2*held_bytes);
        CHECK(st.tier_skipped >= held_bytes); // the second plan skipped them (the third is the same plan)
        printf("tier io %d: skipped %.2f MiB, uploaded from tier %.2f MiB, ring %.2f MiB, demand %.2f MiB\n", io, st.tier_skipped/1048576.0,
                st.tier_uploaded/1048576.0, st.plan_bytes/1048576.0, st.demand_bytes/1048576.0);

        // held when planned, gone before the upload (e.g. promoted to the device by a seed): read on demand, exact
        ggml_backend_buffer_clear(dst_buf, 0);
        hc.release_key(0, 2);
        hc.release_key(0, 3);
        ws.plan(plan, 3);
        hc.release_key(0, 4);
        for (int t = 0; t < 3; ++t) {
            CHECK(ws.upload(backend, src[t], dst[t], 0, 0, ggml_nbytes(src[t])));
        }
        // runs with gaps (only some experts), like the routed experts of a ubatch
        CHECK(ws.upload(backend, src[1], dst[1], 5*src[1]->nb[2] + 17, 5*src[1]->nb[2] + 17, 3*src[1]->nb[2]));
        ggml_backend_synchronize(backend);
        check_dst("tier release after plan");
        CHECK(ws.get_stats().demand_bytes > 0);
        // the device holds experts 8..10 when planned (not read ahead), an upload asks for them after all: read on demand, exact
        ggml_backend_buffer_clear(dst_buf, 0);
        tier.device_held = { 8, 9, 10 };
        const uint64_t dem0 = ws.get_stats().demand_bytes;
        ws.plan(plan, 3);
        for (int t = 0; t < 3; ++t) {
            CHECK(ws.upload(backend, src[t], dst[t], 0, 0, ggml_nbytes(src[t])));
        }
        ggml_backend_synchronize(backend);
        check_dst("tier device held");
        CHECK(ws.get_stats().demand_bytes - dem0 >= 3*(src[0]->nb[2] + src[1]->nb[2] + src[2]->nb[2]));
        tier.device_held.clear();
        ws.plan(nullptr, 0);
        ws.set_tier(nullptr);
    }

    // reads into slots: direct and buffered, with faults; VERIFY repairs a corrupt read
    for (int fmode : { 0, 4, 5 }) {
        for (bool direct : { true, false }) {
            fault flt;
            flt.mode = fmode;
            llama_weight_store_params p;
            p.io = LLAMA_WEIGHT_STREAM_IO_AUTO; p.n_threads = 2; p.verify = fmode == 4; p.ram_avail = (size_t) 1 << 30;
            llama_weight_store ws(ext, p);
            if (fmode) {
                ws.set_read_fn(read_hook, &flt);
            }
            llama_moe_host_cache hc(ggml_backend_cpu_buffer_type(), 4*slot_bytes, LLAMA_MOE_HOST_INCLUSIVE, n_expert, sizes);
            std::vector<llama_weight_store::host_read> rd;
            std::vector<int32_t> sl;
            for (int32_t e : { 3, 9, 15 }) {
                const int32_t s = hc.reserve(0, e);
                sl.push_back(s);
                for (int t = 0; t < 3; ++t) {
                    rd.push_back({ src[t], (size_t) e*src[t]->nb[2], src[t]->nb[2], hc.area(s, t), 0 });
                }
            }
            CHECK(ws.read_host(rd.data(), rd.size(), direct));
            size_t k = 0;
            int i = 0;
            for (int32_t e : { 3, 9, 15 }) {
                for (int t = 0; t < 3; ++t, ++k) {
                    hc.set_lead(sl[i], t, rd[k].lead);
                    CHECK(rd[k].lead < 4096);
                    if (memcmp(hc.data(sl[i], t), file_bytes(t, e), src[t]->nb[2]) != 0) {
                        fprintf(stderr, "read_host fault %d direct %d: expert %d tensor %d differs\n", fmode, direct, e, t);
                        n_fail++;
                    }
                }
                hc.publish(sl[i]);
                i++;
            }
            const auto st = ws.get_stats();
            CHECK(st.host_reads == 9);
            if (fmode == 4) {
                CHECK(st.verify_mismatch == 1);
            }
            if (fmode == 5) {
                CHECK(st.fallback_bytes > 0);
            }
        }
    }

    // speculative reads into slots (reads ahead): exact with faults, VERIFY repairs, done / wait, demand reads go first, the store's
    // destructor completes the reads it was not asked to wait for
    for (int fmode : { 0, 4, 5, 6 }) {
        fault flt;
        flt.mode = fmode;
        llama_moe_host_cache hc(ggml_backend_cpu_buffer_type(), 16*slot_bytes, LLAMA_MOE_HOST_INCLUSIVE, n_expert, sizes);
        std::vector<int32_t> sl(n_expert, -1);
        std::vector<std::vector<llama_weight_store::host_read>> rd(n_expert);
        std::vector<uint64_t> tk(n_expert, 0);
        auto check_slot = [&](int32_t e, const char * what) {
            for (int t = 0; t < 3; ++t) {
                hc.set_lead(sl[e], t, rd[e][t].lead);
                if (memcmp(hc.data(sl[e], t), file_bytes(t, e), src[t]->nb[2]) != 0) {
                    fprintf(stderr, "%s fault %d: expert %d tensor %d differs\n", what, fmode, e, t);
                    n_fail++;
                }
            }
        };
        {
            llama_weight_store_params p;
            p.io = LLAMA_WEIGHT_STREAM_IO_AUTO; p.n_threads = 2; p.verify = fmode == 4; p.ram_avail = (size_t) 1 << 30;
            llama_weight_store ws(ext, p);
            if (fmode) {
                ws.set_read_fn(read_hook, &flt);
            }
            for (int32_t e = 0; e < n_expert; ++e) {
                sl[e] = hc.reserve(0, e);
                CHECK(sl[e] >= 0);
                for (int t = 0; t < 3; ++t) {
                    rd[e].push_back({ src[t], (size_t) e*src[t]->nb[2], src[t]->nb[2], hc.area(sl[e], t), 0 });
                }
            }
            for (int32_t e = 0; e < 12; ++e) {
                tk[e] = ws.read_host_async(rd[e].data(), rd[e].size(), true);
                CHECK(tk[e] != 0);
                hc.set_async(sl[e], 1);
            }
            // a demand read while the speculative ones are queued: it goes first and is exact
            CHECK(ws.read_host(rd[15].data(), rd[15].size(), true));
            check_slot(15, "demand behind speculative");
            hc.publish(sl[15]);
            // wait for the last one queued (moved in front), poll the others
            ws.host_wait(tk[11]);
            check_slot(11, "speculative waited");
            hc.publish(sl[11]);
            for (int32_t e = 0; e < 11; ++e) {
                if (e % 2 == 0) {
                    ws.host_wait(tk[e]);
                } else {
                    while (!ws.host_done(tk[e])) {
                        usleep(100);
                    }
                }
                CHECK(ws.host_done(tk[e])); // collected: done
                check_slot(e, "speculative");
                hc.publish(sl[e]);
            }
            // cancel: a queued read (slow reads, one ticket per expert, the last queued) is dropped, a completed one is not
            if (fmode == 6) {
                const uint64_t c1 = ws.read_host_async(rd[14].data(), rd[14].size(), true);
                const uint64_t c2 = ws.read_host_async(rd[14].data(), rd[14].size(), true);
                CHECK(c1 != 0 && c2 != 0);
                CHECK(ws.host_cancel(c2));    // behind c1: not started
                CHECK(!ws.host_cancel(c2));   // collected
                usleep(1000);                 // 2 IO threads, 2 ms per read: 2 of the 3 reads of c1 are in progress, 1 is queued
                CHECK(!ws.host_cancel(c1));   // partly started: it completes as issued
                ws.host_wait(c1);
                CHECK(!ws.host_cancel(c1));   // done and collected
                CHECK(ws.get_stats().spec_cancelled_bytes == src[0]->nb[2] + src[1]->nb[2] + src[2]->nb[2]);
                check_slot(14, "speculative after a cancel");
            }
            // not waited for: the store's destructor completes them
            tk[12] = ws.read_host_async(rd[12].data(), rd[12].size(), false);
            tk[13] = ws.read_host_async(rd[13].data(), rd[13].size(), true);
            CHECK(tk[12] != 0 && tk[13] != 0);
            const auto st = ws.get_stats();
            CHECK(st.spec_bytes == (fmode == 6 ? 16 : 14)*(src[0]->nb[2] + src[1]->nb[2] + src[2]->nb[2]));
            if (fmode == 6) {
                CHECK(st.spec_boosted >= 1);
            }
            if (fmode == 4) {
                CHECK(st.verify_mismatch == 1);
            }
            if (fmode == 5) {
                CHECK(st.fallback_bytes > 0);
            }
            printf("speculative fault %d: %llu reads, %.2f MiB, %llu boosted, %.1f ms waiting\n", fmode, (unsigned long long) st.spec_reads,
                    st.spec_bytes/1048576.0, (unsigned long long) st.spec_boosted, st.spec_wait_us/1000.0);
        }
        check_slot(12, "speculative at destruction");
        check_slot(13, "speculative at destruction");
        // an uncovered source: no ticket, nothing read
        {
            llama_weight_store_params p;
            p.n_threads = 1; p.ram_avail = (size_t) 1 << 30;
            llama_weight_store ws(ext, p);
            llama_weight_store::host_read bad = { dst[0], 0, 100, hc.area(sl[14], 0), 0 };
            CHECK(ws.read_host_async(&bad, 1, true) == 0);
            CHECK(ws.host_done(0));
        }
    }

    // decode simulation: a device LRU of 4 slots and a RAM tier of 6 (or 12) over 16 experts, 2 routed per step (skewed); the misses of
    // the device follow llama_moe_cache::upload_fills: RAM hit -> upload from RAM, RAM miss -> read into a RAM slot (inclusive: evicting;
    // exclusive: a free slot only) or for the device only, device evictions copied back into RAM (exclusive). Exclusive keeps the copy
    // of a promoted expert as the first to go (variant 2, the old exclusive: the RAM slot is freed, misses are never admitted)
    uint64_t reads[2][3] = { { 0, 0, 0 }, { 0, 0, 0 } };
    for (int big : { 0, 1 })
    for (int var : { 0, 1, 2 }) {
        const int pol = var == 1 ? LLAMA_MOE_HOST_INCLUSIVE : LLAMA_MOE_HOST_EXCLUSIVE;
        // like llama_moe_cache: duplicates are kept only with spare room (RAM + device slots >= experts), else the old exclusive
        const bool spare = (big ? 12 : 6) + 4 >= n_expert;
        const bool old_excl = pol == LLAMA_MOE_HOST_EXCLUSIVE && (var == 2 || !spare);
        llama_weight_store_params p;
        p.io = LLAMA_WEIGHT_STREAM_IO_BUFFERED; p.n_threads = 2; p.ram_avail = (size_t) 1 << 30;
        llama_weight_store ws(ext, p);
        llama_moe_host_cache hc(ggml_backend_cpu_buffer_type(), (big ? 12 : 6)*slot_bytes, pol, n_expert, sizes);
        const int n_dev = 4;
        std::vector<int32_t> dev_key(n_dev, -1);
        std::list<int32_t> lru; // device slots, front = least recently used
        for (int s = 0; s < n_dev; ++s) {
            lru.push_back(s);
        }
        std::mt19937 r2(77);
        std::discrete_distribution<int> zipf({ 16, 12, 9, 7, 6, 5, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1 });
        uint64_t ssd = 0, ram_hits = 0, misses = 0;
        bool exact = true, disjoint = true, pinned_ok = true;
        for (int step = 0; step < 600; ++step) {
            int32_t ids[2] = { zipf(r2), zipf(r2) };
            if (ids[1] == ids[0]) {
                ids[1] = (ids[0] + 1) % n_expert;
            }
            hc.begin_call();
            std::vector<int32_t> pinned_before;
            for (int32_t e : ids) {
                const int32_t s = hc.find(0, e, false);
                if (s >= 0) {
                    hc.pin(s);
                    pinned_before.push_back(e);
                }
            }
            // device plan: hits to the tail, misses evict the head
            std::vector<std::pair<int32_t, int32_t>> fills, evicted;
            for (int32_t e : ids) {
                auto it = std::find(dev_key.begin(), dev_key.end(), e);
                if (it != dev_key.end()) {
                    const int32_t s = (int32_t) (it - dev_key.begin());
                    lru.remove(s);
                    lru.push_back(s);
                }
            }
            for (int32_t e : ids) {
                if (std::find(dev_key.begin(), dev_key.end(), e) != dev_key.end()) {
                    continue;
                }
                const int32_t s = lru.front();
                lru.pop_front();
                lru.push_back(s);
                if (dev_key[s] >= 0) {
                    evicted.emplace_back(s, dev_key[s]);
                }
                dev_key[s] = e;
                fills.emplace_back(e, s);
            }
            // exclusive: the evicted experts go to RAM (the device slot still holds them: the "bank" is dst)
            if (hc.exclusive()) {
                for (const auto & [s, key] : evicted) {
                    if (hc.contains(0, key)) {
                        const int32_t h = hc.held(0, key);
                        if (h >= 0 && !old_excl) {
                            hc.set_dup(h, false);
                        }
                        continue;
                    }
                    const int32_t r = hc.reserve(0, key);
                    if (r < 0) {
                        continue;
                    }
                    for (int t = 0; t < 3; ++t) {
                        hc.set_lead(r, t, 0);
                        ggml_backend_tensor_get_async(backend, dst[t], hc.area(r, t), (size_t) s*src[t]->nb[2], src[t]->nb[2]);
                    }
                    hc.mark_write(r);
                }
            }
            std::vector<llama_weight_store::host_read> rd;
            std::vector<std::pair<int32_t, std::pair<int32_t, int32_t>>> pend;
            std::vector<llama_weight_store::region> rest;
            for (const auto & [e, s] : fills) {
                misses++;
                const int32_t r = hc.find(0, e);
                if (r >= 0) {
                    ram_hits++;
                    for (int t = 0; t < 3; ++t) {
                        ggml_backend_tensor_set_async(backend, dst[t], hc.data(r, t), (size_t) s*src[t]->nb[2], src[t]->nb[2]);
                    }
                    hc.mark_read(r);
                    if (hc.exclusive()) {
                        if (old_excl) {
                            hc.release(r);
                        } else {
                            hc.set_dup(r, true);
                        }
                    }
                    continue;
                }
                const int32_t q = old_excl ? -1 : hc.reserve(0, e, !hc.exclusive());
                if (q >= 0) {
                    for (int t = 0; t < 3; ++t) {
                        rd.push_back({ src[t], (size_t) e*src[t]->nb[2], src[t]->nb[2], hc.area(q, t), 0 });
                    }
                    pend.push_back({ q, { e, s } });
                    continue;
                }
                for (int t = 0; t < 3; ++t) {
                    rest.push_back({ src[t], (size_t) e*src[t]->nb[2], dst[t], (size_t) s*src[t]->nb[2], src[t]->nb[2] });
                }
            }
            if (!rd.empty()) {
                CHECK(ws.read_host(rd.data(), rd.size(), true));
                size_t k = 0;
                for (const auto & [q, es] : pend) {
                    for (int t = 0; t < 3; ++t, ++k) {
                        hc.set_lead(q, t, rd[k].lead);
                        ggml_backend_tensor_set_async(backend, dst[t], hc.data(q, t), (size_t) es.second*src[t]->nb[2], src[t]->nb[2]);
                    }
                    hc.publish(q);
                    hc.mark_read(q);
                    if (hc.exclusive()) {
                        hc.set_dup(q, true);
                    }
                    ssd++;
                }
            }
            hc.commit(backend);
            if (!rest.empty()) {
                CHECK(ws.upload_batch(backend, rest.data(), rest.size()));
                ssd += rest.size()/3;
            }
            ggml_backend_synchronize(backend);
            // invariants: the device slots hold their experts byte for byte, exclusive tiers are disjoint, routed experts kept
            for (int s = 0; s < n_dev; ++s) {
                if (dev_key[s] < 0) {
                    continue;
                }
                for (int t = 0; t < 3; ++t) {
                    std::vector<uint8_t> got(src[t]->nb[2]);
                    ggml_backend_tensor_get(dst[t], got.data(), (size_t) s*src[t]->nb[2], got.size());
                    exact = exact && memcmp(got.data(), file_bytes(t, dev_key[s]), got.size()) == 0;
                }
                if (old_excl && hc.contains(0, dev_key[s])) {
                    disjoint = false; // the old exclusive tiers are disjoint
                }
            }
            for (int32_t e = 0; e < n_expert; ++e) {
                const int32_t r = hc.find(0, e, false);
                if (r >= 0) {
                    for (int t = 0; t < 3; ++t) {
                        exact = exact && memcmp(hc.data(r, t), file_bytes(t, e), src[t]->nb[2]) == 0;
                    }
                }
            }
            if (!hc.exclusive()) {
                for (const auto & [e, s] : fills) {
                    pinned_ok = pinned_ok && hc.contains(0, e); // inclusive: what the device missed is in RAM (held or read now)
                }
                for (int32_t e : pinned_before) {
                    pinned_ok = pinned_ok && hc.contains(0, e); // the routed experts RAM held are not evicted by this call
                }
            }
        }
        CHECK(exact);
        CHECK(disjoint);
        CHECK(pinned_ok);
        reads[big][var] = ssd;
        printf("simulation %d RAM slots %s: %llu device misses, %llu RAM hits, %llu read from the file, %llu evicted, peak %zu held\n", big ? 12 : 6,
                var == 0 ? "exclusive" : var == 1 ? "inclusive" : "exclusive (old)", (unsigned long long) misses, (unsigned long long) ram_hits,
                (unsigned long long) ssd, (unsigned long long) hc.st.evicted, hc.st.peak_held);
    }
    // 4 device + 6 RAM slots over 16 experts: the exclusive tiers hold up to 10 distinct experts, the inclusive ones 6; 4 + 12 slots
    // hold all 16 exclusive (the misses fill the spare slots), the old exclusive only filled RAM by device evictions
    // (this simulation has no prompt seeding: the old exclusive refilled RAM from the many evictions of a 4-slot device; at V32 / R29
    // the device evicts rarely and the old exclusive kept 23% RAM hits)
    CHECK(reads[0][0] < reads[0][1]);
    CHECK(reads[0][0] == reads[0][2]); // no spare room: the old exclusive
    CHECK(reads[1][0] < reads[1][1]);
    CHECK(reads[1][0] <= reads[1][2]);

    ggml_backend_buffer_free(dst_buf);
    ggml_backend_buffer_free(src_buf);
    ggml_free(ctx_src);
    ggml_free(ctx_dst);
    munmap(map, file_size);
    close(fd);
    remove(path.c_str());
    ggml_backend_free(backend);

    printf("test-moe-host-cache: %s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
#endif
