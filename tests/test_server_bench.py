import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from consumer_gpu_llm import server_bench


class FakeServer:
    """Token count = characters / 4. Decode speed depends on depth and on concurrency."""

    def __init__(self):
        self.lock = threading.Lock()

    def tokenize(self, text):
        return len(text) // 4

    def complete(self, prompt, n_predict):
        n = self.tokenize(prompt)
        tps = 50.0 - n / 4000.0
        return {"prompt_n": n, "prompt_per_second": 900.0, "predicted_n": n_predict,
                "predicted_per_second": tps, "predicted_ms": n_predict / tps * 1000.0}


def test_make_prompt_hits_target():
    s = FakeServer()
    for target in (500, 16000, 78000):
        p = server_bench.make_prompt(target, s.tokenize)
        assert abs(s.tokenize(p) - target) <= 2


def test_make_prompt_grows_short_filler():
    p = server_bench.make_prompt(1000, lambda t: len(t) // 40, filler="ab")
    assert abs(len(p) // 40 - 1000) <= 2


def test_aggregate_uses_longest_decode():
    # the published 30k-depth case: 3 x 256 tokens, slowest sequence 3.26 tok/s (78.19 s)
    t = [{"predicted_n": 256, "predicted_ms": 78190.36},
         {"predicted_n": 256, "predicted_ms": 40997.01},
         {"predicted_n": 256, "predicted_ms": 9103.91}]
    assert round(server_bench.aggregate_decode_tps(t), 1) == 9.8


def test_depth_sweep_rows():
    rows = server_bench.depth_sweep(FakeServer(), targets=(500, 32000), n_predict=16)
    assert [r["target"] for r in rows] == [500, 32000]
    assert rows[0]["decode_tok_s"] > rows[1]["decode_tok_s"]


def test_parallel_depth_rows():
    rows = server_bench.parallel_depth(FakeServer(), targets=(500,), parallel=(1, 3), n_predict=8)
    assert [r["parallel"] for r in rows] == [1, 3]
    assert len(rows[1]["per_seq_decode_tok_s"]) == 3
    assert rows[1]["aggregate_decode_tok_s"] == pytest.approx(3 * rows[0]["aggregate_decode_tok_s"], rel=0.01)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            out = {"tokens": list(range(len(body["content"]) // 4))}
        else:
            assert body["cache_prompt"] is False
            out = {"timings": {"prompt_n": len(body["prompt"]) // 4, "prompt_per_second": 1000.0,
                               "predicted_n": body["n_predict"], "predicted_per_second": 50.0,
                               "predicted_ms": body["n_predict"] * 20.0}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_http_client_against_local_fake():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        s = server_bench.LlamaServer(f"http://127.0.0.1:{httpd.server_address[1]}", timeout=10)
        rows = server_bench.depth_sweep(s, targets=(200,), n_predict=4)
        assert rows[0]["decode_tok_s"] == 50.0 and abs(rows[0]["prompt_n"] - 200) <= 2
    finally:
        httpd.shutdown()
