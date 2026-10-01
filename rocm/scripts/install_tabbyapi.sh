#!/bin/bash
# Install TabbyAPI next to this repository, patched (RDNA3 GPUs, prompt lookup, stream keepalive), using the active environment.
# usage: rocm/scripts/install_tabbyapi.sh [tabby_dir] [models_dir]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
TABBY=${1:-$REPO/../tabbyAPI}
MODELS=$(cd "${2:-$REPO/models}" && pwd)
TABBY_COMMIT=f07131cd8fe34e449fe87cdd3a066b52b96d3cac   # tested revision
[ -d "$TABBY" ] || git clone https://github.com/theroyallab/tabbyAPI "$TABBY"
cd "$TABBY"
git checkout -q "$TABBY_COMMIT"
# 0001: accept RDNA3 GPUs; 0002: max_history for EXL3_MTP_LOOKUP; 0003: SSE keepalive during buffered tool calls
for p in "$REPO"/rocm/tabbyapi/*.patch; do git apply "$p"; done
# TabbyAPI's own dependencies only: torch and exllamav3 come from this repository's environment
pip install "fastapi-slim>=0.115" "pydantic>=2.11,<3" ruamel.yaml rich "uvicorn>=0.28.1" "jinja2>=3.0.0" loguru \
            "sse-starlette>=2.2.0" packaging aiofiles aiohttp async_lru psutil "httptools>=0.5.0" requests uvloop setuptools
ln -sfn "$MODELS" models
cp -n "$REPO/rocm/tabbyapi/config.dflash2-192k.yml" config.yml
echo "TabbyAPI ready in $TABBY (config.yml = DFlash2 / 192K profile)"
