import pytest
from utils import *

server: ServerProcess

STORY = "Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her dog. "


@pytest.fixture(autouse=True)
def create_server(monkeypatch):
    global server
    # draft from short repeats so the tiny model accepts drafts
    monkeypatch.setenv("LLAMA_SPEC_SUFFIX_MIN_DRAFT", "3")
    monkeypatch.setenv("LLAMA_SPEC_SUFFIX_P_CUT", "0")
    monkeypatch.setenv("LLAMA_SPEC_SUFFIX_POLICY", "0")
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_batch = 64
    server.spec_type = "ngram-suffix"


def test_post_sampling_probs_with_ngram_drafter():
    # each token reports the distribution it was sampled from: the sampled token is among its top_probs with its probability
    global server
    server.start()
    n_tok = 0
    for seed in range(4):
        res = server.make_request("POST", "/completion", data={
            "prompt": STORY * 6,
            "n_predict": 48,
            "temperature": 0.5,
            "top_k": 3,
            "seed": seed,
            "n_probs": 3,
            "post_sampling_probs": True,
        })
        assert res.status_code == 200
        for tok in res.body["completion_probabilities"]:
            top = {p["id"]: p["prob"] for p in tok["top_probs"]}
            assert tok["id"] in top
            assert tok["prob"] == pytest.approx(top[tok["id"]], abs=1e-6)
            n_tok += 1
    assert n_tok > 0
    # the same requests without post-sampling probabilities do draft
    res = server.make_request("POST", "/completion", data={
        "prompt": STORY * 6,
        "n_predict": 48,
        "temperature": 0.5,
        "top_k": 3,
        "seed": 0,
        "n_probs": 3,
    })
    assert res.status_code == 200
    assert res.body["timings"].get("draft_n", 0) > 0
