#!/usr/bin/env python3
# Block size of per-block FP8 scales (ModelOpt FP8_PB_WO): padded blocks of 128, partial last block. Run: python3 tests/test-convert-fp8-block.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))

from conversion.base import fp8_scale_block_size  # noqa: E402

# exact tilings of 128
assert fp8_scale_block_size([256, 512], [2, 4]) == [128, 128]
# a 192-row weight has ceil(192 / 128) = 2 scales: block 128 (rows 128..191 use the second), not 96
assert fp8_scale_block_size([192, 128], [2, 1]) == [128, 128]
assert fp8_scale_block_size([960, 960], [8, 8]) == [128, 128]
# 576 rows (an MLA kv_a_proj) with 5 scales: block 128, no longer an error
assert fp8_scale_block_size([576, 7168], [5, 56]) == [128, 128]
# a block named by the config wins
assert fp8_scale_block_size([192, 256], [3, 4], [64, 64]) == [64, 64]
# an exact tiling with another block keeps the inferred size
assert fp8_scale_block_size([256, 256], [4, 4]) == [64, 64]
# no tiling, rank mismatch
assert fp8_scale_block_size([300, 128], [7, 1]) is None
assert fp8_scale_block_size([256, 256], [2]) is None
print("test-convert-fp8-block: OK")
