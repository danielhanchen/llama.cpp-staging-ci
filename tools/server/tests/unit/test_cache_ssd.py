import glob
import os
import time

import pytest
from utils import *

# SSD tier of the prompt cache (--cache-ssd-dir): states evicted from the RAM cache, and every slot and RAM entry on a clean
# shutdown, are written to <dir>/<namespace>/<key>.lpc and reused by later requests, also by a restarted server

PROMPT_A = "Once upon a time, there was a little girl named Lily. She liked to play in the park with her friends. " * 4 + "One day"
PROMPT_B = "Tom had a big red ball. He threw the ball to his dog and the dog ran very fast after it. " * 4 + "Then"


def lpc_files(root):
    return sorted(glob.glob(os.path.join(root, "*", "*.lpc")))


def ns_dirs(root):
    return sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))


def make_server(root, preset=ServerPreset.tinyllama2):
    server = preset()
    server.temperature = 0.0
    server.n_slots = 1
    server.n_predict = 24
    server.cache_ssd_dir = str(root)
    return server


def complete(server, prompt, timeout=DEFAULT_REQUEST_TIMEOUT, **kw):
    data = {"prompt": prompt, "cache_prompt": True, "n_probs": 4}
    data.update(kw)
    res = server.make_request("POST", "/completion", data=data, timeout=timeout)
    assert res.status_code == 200, res.body
    return res.body


def probs(body):
    return [[(p["id"], p["logprob"]) for p in t["top_logprobs"]] for t in body["completion_probabilities"]]


def test_restart_survival_exact(tmp_path):
    root = tmp_path / "ssd"
    server = make_server(root)
    server.start()
    first = complete(server, PROMPT_A)
    assert first["timings"]["cache_n"] == 0
    complete(server, PROMPT_B)            # the slot moves on, PROMPT_A goes to the RAM cache
    ram = complete(server, PROMPT_A)      # a RAM cache hit: the reference for a restored state
    assert ram["timings"]["cache_n"] > 0
    server.stop()  # clean shutdown writes the slot and the RAM cache to the SSD tier

    assert len(lpc_files(root)) == 2

    server = make_server(root)
    server.start()
    again = complete(server, PROMPT_A)
    assert again["timings"]["cache_n"] == ram["timings"]["cache_n"]  # restored from the SSD tier
    assert again["timings"]["prompt_n"] == ram["timings"]["prompt_n"]
    assert again["content"] == first["content"]
    # bit-identical to the RAM cache hit: the SSD tier restores the same bytes (a hit re-evaluates the last prompt
    # token alone, so both differ from the fresh run in the last digits, as RAM hits always have)
    assert probs(again) == probs(ram)


def test_namespace_mismatch_is_a_miss(tmp_path):
    root = tmp_path / "ssd"
    server = make_server(root)
    server.start()
    first = complete(server, PROMPT_A)
    server.stop()
    assert len(lpc_files(root)) == 1

    server = make_server(root)
    server.ctk = "f32"  # a different K cache type gives a different namespace
    server.start()
    other = complete(server, PROMPT_A)
    assert other["timings"]["cache_n"] == 0
    assert other["content"] == first["content"]
    server.stop()
    assert len(ns_dirs(root)) == 2


@pytest.mark.parametrize("damage", ["truncate", "flip"])
def test_damaged_file_is_a_miss_and_deleted(tmp_path, damage):
    root = tmp_path / "ssd"
    server = make_server(root)
    server.start()
    first = complete(server, PROMPT_A)
    server.stop()
    files = lpc_files(root)
    assert len(files) == 1

    with open(files[0], "r+b") as f:
        data = f.read()
        if damage == "truncate":
            f.truncate(len(data) // 2)
        else:
            f.seek(len(data) // 2)
            f.write(bytes([data[len(data) // 2] ^ 0xff]))

    server = make_server(root)
    server.start()
    again = complete(server, PROMPT_A)
    assert again["timings"]["cache_n"] == 0
    assert again["content"] == first["content"]
    assert files[0] not in lpc_files(root)


def test_ram_eviction_goes_to_ssd(tmp_path):
    root = tmp_path / "ssd"
    server = make_server(root)
    server.cache_ram = 1  # 1 MiB: the RAM cache holds only a few of these prompts
    server.start()

    prompts = [f"Story number {i}. " + (PROMPT_A if i % 2 == 0 else PROMPT_B) for i in range(16)]
    results = [complete(server, p) for p in prompts]
    time.sleep(1.0)  # the writer runs in the background
    assert len(lpc_files(root)) > 0

    again = complete(server, prompts[0])
    assert again["timings"]["cache_n"] > 0
    assert again["content"] == results[0]["content"]


def test_concurrent_slots(tmp_path):
    root = tmp_path / "ssd"
    server = make_server(root)
    server.n_slots = 2
    server.n_ctx = 1024
    server.start()
    tasks = [(complete, (server, p)) for p in (PROMPT_A, PROMPT_B)]
    first = parallel_function_calls(tasks)
    server.stop()
    assert len(lpc_files(root)) == 2

    server = make_server(root)
    server.n_slots = 2
    server.n_ctx = 1024
    server.start()
    tasks = [(complete, (server, p)) for p in (PROMPT_A, PROMPT_B)]
    again = parallel_function_calls(tasks)
    for f, a in zip(first, again):
        assert a["timings"]["cache_n"] > 0
        assert a["content"] == f["content"]


def test_readonly(tmp_path):
    root = tmp_path / "ssd"
    server = make_server(root)
    server.start()
    first = complete(server, PROMPT_A)
    server.stop()
    before = lpc_files(root)
    mtimes = [os.path.getmtime(f) for f in before]

    server = make_server(root)
    server.cache_ssd_readonly = True
    server.start()
    again = complete(server, PROMPT_A)
    assert again["timings"]["cache_n"] > 0
    assert again["content"] == first["content"]
    complete(server, PROMPT_B)
    server.stop()
    assert lpc_files(root) == before
    assert [os.path.getmtime(f) for f in before] == mtimes


def test_kv_shifted_state_is_not_saved(tmp_path):
    root = tmp_path / "ssd"
    server = make_server(root)
    server.n_ctx = 256
    server.n_predict = 300
    server.enable_ctx_shift = True
    server.start()
    body = complete(server, PROMPT_A, n_predict=300)
    assert body["truncated"] or body["tokens_predicted"] > 0
    server.stop()
    assert lpc_files(root) == []


def test_swa_checkpoints(tmp_path):
    root = tmp_path / "ssd"
    base = "The quick brown fox jumps over the lazy dog. " * 20

    def swa_server():
        server = make_server(root, ServerPreset.tinygemma3)
        server.n_ubatch = 32  # checkpoints before the divergence point
        server.n_predict = 4
        return server

    server = swa_server()
    server.start()
    e1 = complete(server, base + "The first ending of this story is a happy one.")
    e2 = complete(server, base + "But the second ending was different and sad.")
    assert e2["timings"]["cache_n"] > 0
    server.stop()
    assert len(lpc_files(root)) == 1

    # the saved prompt ends with e2: e1 diverges inside it, which an SWA model can only resume from a checkpoint
    server = swa_server()
    server.start()
    again = complete(server, base + "The first ending of this story is a happy one.")
    assert again["timings"]["cache_n"] > 0
    assert again["timings"]["prompt_n"] < e1["timings"]["prompt_n"]
    assert again["content"] == e1["content"]


# a hybrid (attention + recurrent) model; slow, needs a local GGUF: QWEN36_GGUF=/path/to/Qwen3.6-35B-A3B-*.gguf
@pytest.mark.skipif(not os.environ.get("QWEN36_GGUF"), reason="set QWEN36_GGUF to a local Qwen3.6-35B-A3B GGUF")
def test_hybrid_restart_exact(tmp_path):
    root = tmp_path / "ssd"

    def hybrid_server():
        server = ServerProcess()
        server.offline = True
        server.model_file = os.environ["QWEN36_GGUF"]
        server.model_alias = "qwen36"
        server.n_ctx = 2048
        server.n_batch = 512
        server.n_ubatch = 128
        server.n_slots = 1
        server.n_predict = 16
        server.seed = 42
        server.temperature = 0.0
        server.cache_ssd_dir = str(root)
        return server

    base = "Explain in detail how a binary search tree keeps its keys ordered, and how insertion works. " * 6
    T = 3600  # CPU-only runs of a 35B model are slow
    server = hybrid_server()
    server.start(timeout_seconds=600)
    e1 = complete(server, base + "First, describe the search.", timeout=T)
    e2 = complete(server, base + "Now describe deletion instead.", timeout=T)
    server.stop()
    assert len(lpc_files(root)) == 1

    server = hybrid_server()
    server.start(timeout_seconds=600)
    a2 = complete(server, base + "Now describe deletion instead.", timeout=T)  # exact prefix of the saved state
    assert a2["timings"]["cache_n"] > 0
    assert a2["content"] == e2["content"]
    a1 = complete(server, base + "First, describe the search.", timeout=T)      # diverges inside: needs a checkpoint
    assert a1["content"] == e1["content"]
