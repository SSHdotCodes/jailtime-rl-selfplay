#!/usr/bin/env bash
set -euo pipefail

cd /Users/sshpro/Documents/Codex/2026-06-29/you-are-an-expert-python-package

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "Starting Gemma 4 Jailtime MPS run at $(date)"
echo "Working directory: $(pwd)"
echo

python3 -m pip install -e ".[dev,local]"
echo
jailtime devices
echo
jailtime validate-config work/gemma4_jailtime_full_run.yaml
echo

python3 work/gemma4_jailtime_full_run.py \
  --config work/gemma4_jailtime_full_run.yaml \
  --train-adapter \
  --adapter-epochs 2 \
  --max-train-examples 1024 \
  --max-length 512

echo
echo "Finished Gemma 4 Jailtime run at $(date)"
