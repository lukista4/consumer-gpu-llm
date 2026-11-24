"""Thin wrapper: python scripts/gguf_block_compare.py --help  (same as python -m consumer_gpu_llm.block_compare)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from consumer_gpu_llm.block_compare import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
