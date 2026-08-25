#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the instrumented vLLM environment on a fresh machine (pod or laptop).
# Usage: run from the directory where you want the checkout to live.

git clone --branch instrumented https://github.com/randomwish/vllm.git vllm_source
cd vllm_source
pip install -q uv
uv venv .venv
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e .
echo ""
echo "ready:"
echo "  source vllm_source/.venv/bin/activate"
echo "  python intro.py"
