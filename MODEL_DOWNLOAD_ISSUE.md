# Model Download Issue - Resolution Guide

## Problem Summary

The test `test_get_response_log_probs` was failing with:
```
safetensors_rust.SafetensorError: Error while deserializing header: header too large
```

## Root Cause

1. **Corrupted Model File**: The `Qwen2.5-Math-1.5B/model.safetensors` file was only 135 bytes (corrupted/placeholder)
2. **Missing Git LFS**: Git Large File Storage was not installed, so large model files weren't being downloaded properly

## Solution Steps

### Step 1: Install Git LFS
```bash
# Install via Homebrew (macOS)
brew install git-lfs

# Initialize Git LFS
git lfs install
```

### Step 2: Download the Model Properly
```bash
# Remove the corrupted model directory
rm -rf Qwen2.5-Math-1.5B

# Clone with Git LFS support
git clone https://huggingface.co/Qwen/Qwen2.5-Math-1.5B
```

### Step 3: Ensure LFS Files are Pulled
```bash
cd Qwen2.5-Math-1.5B
git lfs pull
```

## What is Git LFS?

Git LFS (Large File Storage) is a Git extension that:
- Replaces large files with text pointers inside Git
- Stores the actual file contents on a remote server
- Downloads the actual files when you clone or pull

Without Git LFS:
- Large files (like model.safetensors) appear as tiny text pointers
- You get 135-byte placeholder files instead of multi-GB model files

With Git LFS:
- Large files are automatically downloaded when you clone/pull
- Model files are complete and usable

## Verification

Check that the model file is properly downloaded:
```bash
ls -lh Qwen2.5-Math-1.5B/model.safetensors
```

Expected size: **Several GB** (not 135 bytes!)

For Qwen2.5-Math-1.5B, the model.safetensors should be approximately **3GB**.

## Alternative Solution (If Download is Too Slow)

If downloading the full 3GB model is problematic, you can modify `conftest.py` to use the tiny-gpt2 model instead:

```python
@pytest.fixture
def model_id():
    FIXTURES_PATH = Path(__file__).parent / "fixtures"
    return str(FIXTURES_PATH / "tiny-gpt2")
```

The tiny-gpt2 model (26MB) is already in `tests/fixtures/tiny-gpt2/` and works for testing purposes.

## Current Status

✅ Git LFS installed and initialized
⏳ Model cloning in progress (may take time depending on internet speed)

Once the model.safetensors file is fully downloaded, the test should pass!
