// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
// renamed environment switches: the earlier name and the new name each give the same parsed value, the new name wins when both are set

#include "../src/llama-env-alias.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

#ifdef _WIN32
static int setenv(const char * name, const char * value, int /*overwrite*/) { return _putenv_s(name, value); }
static int unsetenv(const char * name) { return _putenv_s(name, ""); }
#endif

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

static std::string val(const char * name) {
    const char * v = llama_getenv_alias(name);
    return v != nullptr ? std::string(v) : std::string("<unset>");
}

int main() {
    std::set<std::string> seen;
    int n = 0;
    for (const llama_env_alias_entry & a : llama_env_aliases) {
        n++;
        CHECK(seen.insert(a.name).second && seen.insert(a.old_name).second, "%s / %s listed twice", a.name, a.old_name);
        CHECK(strcmp(a.name, a.old_name) != 0, "%s aliases itself", a.name);
        unsetenv(a.name);
        unsetenv(a.old_name);

        // neither set: nullptr, so every caller keeps its default
        CHECK(llama_getenv_alias(a.name) == nullptr, "%s: set without any variable", a.name);

        // the values the callers parse: atoi for the switches, atof for the bands
        for (const char * v : { "0", "1", "0.25", "0.6" }) {
            setenv(a.old_name, v, 1);
            const std::string by_old = val(a.name);
            const int    i_old = atoi(by_old.c_str());
            const double f_old = atof(by_old.c_str());
            unsetenv(a.old_name);

            setenv(a.name, v, 1);
            const std::string by_new = val(a.name);
            unsetenv(a.name);

            CHECK(by_old == v, "%s=%s read as %s through %s", a.old_name, v, by_old.c_str(), a.name);
            CHECK(by_new == by_old, "%s=%s read as %s, %s=%s as %s", a.name, v, by_new.c_str(), a.old_name, v, by_old.c_str());
            CHECK(atoi(by_new.c_str()) == i_old && atof(by_new.c_str()) == f_old, "%s / %s parse differently for %s", a.name, a.old_name, v);
        }

        // both set: the new name wins
        setenv(a.old_name, "0", 1);
        setenv(a.name, "1", 1);
        CHECK(val(a.name) == "1", "%s: the earlier name %s won over the new one", a.name, a.old_name);
        unsetenv(a.name);
        CHECK(val(a.name) == "0", "%s: the earlier name %s not read once the new one is unset", a.name, a.old_name);
        unsetenv(a.old_name);
    }

    // a name outside the table is a plain getenv
    unsetenv("LLAMA_ENV_ALIAS_TEST_PLAIN");
    CHECK(llama_getenv_alias("LLAMA_ENV_ALIAS_TEST_PLAIN") == nullptr, "plain name set");
    setenv("LLAMA_ENV_ALIAS_TEST_PLAIN", "7", 1);
    CHECK(val("LLAMA_ENV_ALIAS_TEST_PLAIN") == "7", "plain name not read");
    unsetenv("LLAMA_ENV_ALIAS_TEST_PLAIN");

    CHECK(n == 12, "%d aliases, expected 12", n);
    if (n_fail == 0) {
        printf("test-env-alias: %d aliases OK\n", n);
    }
    return n_fail == 0 ? 0 : 1;
}
