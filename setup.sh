#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the instrumented vLLM environment on a pod or laptop.
# Usage: run from the directory where you want the checkout to live.

if [[ ! -d vllm_source/.git ]]; then
  git clone --branch instrumented https://github.com/randomwish/vllm.git vllm_source
else
  echo "Using existing vllm_source checkout."
fi

cd vllm_source

# Ubuntu 24.04 marks the system Python environment as externally managed
# (PEP 668), so do not install uv with the system pip. The standalone installer
# places uv in $HOME/.local/bin without modifying the system interpreter.
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to install uv." >&2
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv venv .venv
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e .
echo ""
echo "ready:"
echo "  source vllm_source/.venv/bin/activate"
echo "  python intro.py"
