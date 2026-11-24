// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

// Environment switches that were renamed: the new name is read first, the earlier name still works when the new one is unset.
// Header only, so the library and the tools share one table and one lookup.

#include <cstdlib>
#include <cstring>

struct llama_env_alias_entry {
    const char * name;     // current name
    const char * old_name; // earlier name, still accepted
};

inline constexpr llama_env_alias_entry llama_env_aliases[] = {
    // the cross-layer prefetch and deeper MTP windows beside a small placed MoE cache (compute capability 12.0)
    { "LLAMA_MOE_SMALLCACHE_PREFETCH",             "LLAMA_MOE_G4TAIL"              },
    { "LLAMA_MOE_SMALLCACHE_PREFETCH_LO",          "LLAMA_MOE_G4TAIL_LO"           },
    { "LLAMA_MOE_SMALLCACHE_PREFETCH_HI",          "LLAMA_MOE_G4TAIL_HI"           },
    { "LLAMA_MOE_SMALLCACHE_PREFETCH_FRAC_LO",     "LLAMA_MOE_G4TAIL_FRAC_LO"      },
    { "LLAMA_MOE_SMALLCACHE_PREFETCH_FRAC_HI",     "LLAMA_MOE_G4TAIL_FRAC_HI"      },
    { "LLAMA_MOE_SMALLCACHE_PREFETCH_PLACED",      "LLAMA_MOE_G4TAIL_PLACED"       },
    { "LLAMA_MOE_SMALLCACHE_PREFETCH_PLACED_GATE", "LLAMA_MOE_G4TAIL_PLACED_GATE"  },
    // the side-stream copy overlap of 5..8-row mailbox windows below a cached share
    { "LLAMA_MOE_MAILBOX_WIDE_OVERLAP",            "LLAMA_MOE_G4PASS"              },
    { "LLAMA_MOE_MAILBOX_WIDE_OVERLAP_MAX_FRAC",   "LLAMA_MOE_G4PASS_OV_MAX_FRAC"  },
    // the same overlap in a middle band of the cached share
    { "LLAMA_MOE_MAILBOX_MID_OVERLAP",             "LLAMA_MOE_G4MID"               },
    { "LLAMA_MOE_MAILBOX_MID_OVERLAP_MIN_FRAC",    "LLAMA_MOE_G4MID_OV_MIN_FRAC"   },
    { "LLAMA_MOE_MAILBOX_MID_OVERLAP_MAX_FRAC",    "LLAMA_MOE_G4MID_OV_MAX_FRAC"   },
};

// getenv(name), else getenv of its earlier name when name is in the table above (nullptr when neither is set)
inline const char * llama_getenv_alias(const char * name) {
    if (const char * v = getenv(name); v != nullptr) {
        return v;
    }
    for (const llama_env_alias_entry & a : llama_env_aliases) {
        if (strcmp(a.name, name) == 0) {
            return getenv(a.old_name);
        }
    }
    return nullptr;
}
