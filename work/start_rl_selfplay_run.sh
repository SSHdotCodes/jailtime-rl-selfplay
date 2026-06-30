#!/usr/bin/env bash
cd /Users/sshpro/Documents/Codex/2026-06-29/you-are-an-expert-python-package
echo "=============================================="
echo " Jailtime real-time RL self-play (full-weight REINFORCE, no LoRA)"
echo " Started: $(date)"
echo "=============================================="
echo
python3 work/rl_selfplay_run.py "$@"
echo
echo "=============================================="
echo " Run process exited at $(date)."
echo " Hardened defender safetensors: runs/rl-selfplay/latest/defender_final"
echo "=============================================="
