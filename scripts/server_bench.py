"""Thin wrapper: python scripts/server_bench.py --help  (same as python -m consumer_gpu_llm.server_bench)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from consumer_gpu_llm.server_bench import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
