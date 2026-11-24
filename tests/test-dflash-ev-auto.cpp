// LLAMA_DFLASH_EV_AUTO: each documented value keeps its own mode
#include "speculative.h"

#include <cstdio>

int main() {
    struct { const char * env; int mode; } cases[] = {
        { nullptr, 2 }, { "2", 2 }, { "0", 0 }, { "1", 1 }, { "3", 1 }, { "-1", 1 },
    };
    int n_fail = 0;
    for (const auto & c : cases) {
        const int m = common_speculative_dflash_ev_auto_mode(c.env);
        if (m != c.mode) {
            fprintf(stderr, "FAIL: LLAMA_DFLASH_EV_AUTO=%s -> %d, expected %d\n", c.env ? c.env : "(unset)", m, c.mode);
            n_fail++;
        }
    }
    if (n_fail == 0) {
        printf("OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
