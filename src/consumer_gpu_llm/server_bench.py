"""Long-context probes against a running llama.cpp `llama-server`.

* depth sweep: one request per prompt depth; records prefill and decode speed from the
  server's own `timings` block (`prompt_per_second`, `predicted_per_second`).
* parallel depth: k identical-length requests at once (each prefixed with its index so the
  prompt cache cannot share them); reports per-sequence decode speed, an aggregate decode
  rate and wall time.

The HTTP client is injectable so the logic is testable without a server.
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

# Filler used for the published measurements (any text works; only the token count matters).
FILLER = ("Let n be a positive integer. Prove that the sum of the first n odd numbers equals n squared, "
          "and discuss generalizations to sums of powers. ")


class LlamaServer:
    """Minimal client for llama-server's /tokenize and /completion endpoints."""

    def __init__(self, url: str = "http://127.0.0.1:8080", timeout: float = 3600.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def tokenize(self, text: str) -> int:
        return len(self._post("/tokenize", {"content": text})["tokens"])

    def complete(self, prompt: str, n_predict: int, temperature: float = 0.6) -> dict:
        r = self._post("/completion", {"prompt": prompt, "n_predict": n_predict,
                                       "temperature": temperature, "cache_prompt": False})
        return r["timings"]


def make_prompt(target_tokens: int, count_tokens: Callable[[str], int], filler: str = FILLER) -> str:
    """Repeat `filler` and cut it proportionally so it tokenizes to about `target_tokens`."""
    p = filler * (target_tokens // 20 + 1)
    n = count_tokens(p)
    while n < target_tokens:  # filler shorter than expected: grow until long enough
        p += p
        n = count_tokens(p)
    return p[: int(len(p) * target_tokens / n)]


def aggregate_decode_tps(timings: list[dict]) -> float:
    """Generated tokens of all sequences / the longest decode time (tok/s)."""
    longest_ms = max(t["predicted_ms"] for t in timings)
    return sum(t["predicted_n"] for t in timings) / longest_ms * 1000.0


def depth_sweep(server, targets=(500, 16000, 32000, 60000, 78000), n_predict: int = 256) -> list[dict]:
    rows = []
    for target in targets:
        p = make_prompt(target, server.tokenize)
        t0 = time.time()
        tm = server.complete(p, n_predict)
        rows.append({
            "target": target, "prompt_n": tm["prompt_n"],
            "prefill_tok_s": round(tm["prompt_per_second"], 1),
            "gen_n": tm["predicted_n"], "decode_tok_s": round(tm["predicted_per_second"], 1),
            "wall_s": round(time.time() - t0, 1),
        })
    return rows


def parallel_depth(server, targets=(500, 30000), parallel=(1, 3), n_predict: int = 256) -> list[dict]:
    rows = []
    for target in targets:
        p = make_prompt(target, server.tokenize)
        for k in parallel:
            prompts = [f"{i} {p}" for i in range(k)]
            t0 = time.time()
            with ThreadPoolExecutor(k) as ex:
                tms = list(ex.map(lambda q: server.complete(q, n_predict), prompts))
            rows.append({
                "depth": target, "parallel": k,
                "per_seq_decode_tok_s": [round(t["predicted_per_second"], 1) for t in tms],
                "aggregate_decode_tok_s": round(aggregate_decode_tps(tms), 1),
                "wall_s": round(time.time() - t0, 1),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Depth sweep / parallel-depth probe against llama-server.")
    ap.add_argument("mode", choices=["depth", "parallel"])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--targets", default=None, help="comma-separated prompt depths in tokens")
    ap.add_argument("--parallel", default="1,3")
    ap.add_argument("--n-predict", type=int, default=256)
    a = ap.parse_args(argv)
    s = LlamaServer(a.url)
    if a.mode == "depth":
        targets = tuple(int(x) for x in (a.targets or "500,16000,32000,60000,78000").split(","))
        rows = depth_sweep(s, targets, a.n_predict)
    else:
        targets = tuple(int(x) for x in (a.targets or "500,30000").split(","))
        rows = parallel_depth(s, targets, tuple(int(x) for x in a.parallel.split(",")), a.n_predict)
    for r in rows:
        print(json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
