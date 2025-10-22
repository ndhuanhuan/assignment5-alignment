# Prerequisites
- uv clean
- rm -rf .venv uv.lock
- uv sync  # Now works! flash-attn and vllm are commented out in pyproject.toml
# Install vllm falsh-attn
- uv pip install flash-attn vllm
- or "uv sync --extra gpu"

# QWEN Download issue
Check MODEL_DOWNLOAD_ISSUE

# To Test
- uv run pytest -k test_tokenize_prompt_and_output
- uv run pytest -k test_compute_entropy
- uv run pytest -k test_get_response_log_probs
- uv run pytest -k test_masked_normalize

- uv run pytest -k test_parse_mmlu_response  # ✅ WORKING!
- uv run pytest -k test_parse_gsm8k_response  # ✅ WORKING!
- uv run pytest tests/test_metrics.py -v  # ✅ All 4 tests passing!
- uv run pytest -k test_packed_sft_dataset
