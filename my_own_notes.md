# Prerequisites
- uv clean
- rm -rf .venv uv.lock
- uv sync  # Now works! flash-attn and vllm are commented out in pyproject.toml
# Install vllm falsh-attn
- uv pip install flash-attn vllm
- or "uv sync --extra gpu"

# To Test
- uv run pytest -k test_parse_mmlu_response  # ✅ WORKING!
- uv run pytest -k test_parse_gsm8k_response  # ✅ WORKING!
- uv run pytest tests/test_metrics.py -v  # ✅ All 4 tests passing!
- uv run pytest -k test_packed_sft_dataset