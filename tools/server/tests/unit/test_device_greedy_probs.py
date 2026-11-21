import pytest
from utils import *

# Greedy requests that ask for probabilities (n_probs on /completion, logprobs / top_logprobs on /v1/chat/completions, streaming
# or not) return them while the device greedy chain is on (the default on CUDA / ROCm when every layer is offloaded: run with
# N_GPU_LAYERS=99 to exercise it), and the text and probabilities equal those of the host path (LLAMA_SERVER_DEVICE_GREEDY=0)

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2


PROMPT = "I believe the meaning of life is"
MESSAGES = [{"role": "user", "content": "I believe the meaning of life is"}]


def run_case(endpoint: str, extra: dict, stream: bool):
    if endpoint == "/completion":
        data = {"prompt": PROMPT, "temperature": 0.0, "n_predict": 6, "cache_prompt": False, **extra}
    else:
        data = {"messages": MESSAGES, "temperature": 0.0, "max_tokens": 6, **extra}
    text, probs = "", []
    if stream:
        chunks = list(server.make_stream_request("POST", endpoint, data={**data, "stream": True}))
    else:
        res = server.make_request("POST", endpoint, data=data)
        assert res.status_code == 200
        chunks = [res.body]
    for c in chunks:
        if endpoint == "/completion":
            text += c.get("content", "")
            for p in c.get("completion_probabilities", []) or []:
                top = p.get("top_logprobs", p.get("top_probs", []))
                probs.append((p["token"], p.get("logprob", p.get("prob")), [(t["token"], t.get("logprob", t.get("prob"))) for t in top]))
        else:
            ch = c["choices"][0]
            d = ch.get("delta") or ch.get("message") or {}
            text += d.get("content") or ""
            for p in ((ch.get("logprobs") or {}).get("content") or []):
                probs.append((p["token"], p["logprob"], [(t["token"], t["logprob"]) for t in p.get("top_logprobs", [])]))
    return text, probs


CASES = [
    ("/completion", {"n_probs": 10}, 10),
    ("/completion", {"n_probs": 10, "top_k": 1}, 10),
    ("/completion", {"n_probs": 3, "post_sampling_probs": True}, 1),
    ("/v1/chat/completions", {"logprobs": True}, 0),
    ("/v1/chat/completions", {"logprobs": True, "top_logprobs": 5}, 5),
]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("endpoint,extra,n_top", CASES)
def test_device_greedy_probs(endpoint: str, extra: dict, n_top: int, stream: bool):
    global server
    results = []
    for device_greedy in ("1", "0"):
        os.environ["LLAMA_SERVER_DEVICE_GREEDY"] = device_greedy
        try:
            server.start()
            results.append(run_case(endpoint, extra, stream))
            plain_text, _ = run_case(endpoint, {}, stream)
        finally:
            server.stop()
            del os.environ["LLAMA_SERVER_DEVICE_GREEDY"]
        text, probs = results[-1]
        assert text == plain_text, "asking for probabilities changed the greedy text"
        assert len(probs) > 0
        for _, lp, top in probs:
            assert lp is not None
            if "post_sampling_probs" not in extra:
                assert lp <= 0.0
            if n_top > 0:
                assert len(top) >= 1
                assert len(top) <= n_top
                if "post_sampling_probs" not in extra:
                    assert len(top) == n_top
    assert results[0] == results[1], "device greedy and host path differ"
