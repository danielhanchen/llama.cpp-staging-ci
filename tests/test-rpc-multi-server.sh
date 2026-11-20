#!/usr/bin/env bash
set -euo pipefail

server=$1
client=$2
test_dir=$(mktemp -d)

cleanup() {
    kill "${pid_a:-}" "${pid_b:-}" 2>/dev/null || true
    rm -rf "$test_dir"
}
trap cleanup EXIT

# running and not a zombie (a server whose bind failed has exited)
alive() {
    local st
    st=$(ps -o stat= -p "$1" 2>/dev/null) || return 1
    [ -n "$st" ] && [ "${st#Z}" = "$st" ]
}

# the port answers and the server that should own it is still running
wait_for_server() {
    local port=$1 pid=$2
    for _ in {1..600}; do
        alive "$pid" || return 1
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            alive "$pid"
            return
        fi
        sleep 0.05
    done
    return 1
}

# ports below the usual ephemeral range, picked again when a server cannot bind its port (held by another socket on a busy host)
for attempt in 1 2 3 4 5 6 7 8; do
    port_a=$((20000 + RANDOM % 12000))
    port_b=$((port_a + 1))
    "$server" --device CPU --host 127.0.0.1 --port "$port_a" >"$test_dir/server-a.log" 2>&1 &
    pid_a=$!
    "$server" --device CPU --host 127.0.0.1 --port "$port_b" >"$test_dir/server-b.log" 2>&1 &
    pid_b=$!
    if wait_for_server "$port_a" "$pid_a" && wait_for_server "$port_b" "$pid_b"; then
        break
    fi
    kill "$pid_a" "$pid_b" 2>/dev/null || true
    wait "$pid_a" "$pid_b" 2>/dev/null || true
    if [ "$attempt" = 8 ]; then
        echo "could not start the two RPC servers"
        cat "$test_dir/server-a.log" "$test_dir/server-b.log"
        exit 1
    fi
done
endpoint_a="127.0.0.1:${port_a}"
endpoint_b="127.0.0.1:${port_b}"

"$client" "$endpoint_a" "$endpoint_b"

if grep -q "invalid data ptr" "$test_dir/server-b.log"; then
    cat "$test_dir/server-b.log"
    exit 1
fi
