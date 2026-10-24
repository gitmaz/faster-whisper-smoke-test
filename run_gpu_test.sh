#!/usr/bin/env bash
# Run the GPU transcription test with the CUDA libraries this venv installed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="$ROOT/.venv/lib/python3.14/site-packages/nvidia"
export LD_LIBRARY_PATH="$SITE/cublas/lib:$SITE/cudnn/lib:$SITE/cuda_nvrtc/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
exec python "$ROOT/test_gpu.py" "$@"
