#!/usr/bin/env bash
# Launch a new Terminal window that runs the real-time RL self-play loop with
# 2000 episodes on the NVIDIA RTX PRO 6000 Blackwell, logging to a timestamped
# file. The window stays open when the run finishes so you can read the summary.
#
# Usage:
#   bash work/start_rl_selfplay_cuda.sh                # 2000 episodes, CUDA config
#   bash work/start_rl_selfplay_cuda.sh --episodes 100 # override
#   CONFIG=work/rl_selfplay_cuda.yaml bash work/start_rl_selfplay_cuda.sh --episodes 500
#
set -e

REPO="/Users/sshpro/Documents/Codex/2026-06-29/you-are-an-expert-python-package"
CONFIG="${CONFIG:-$REPO/work/rl_selfplay_cuda.yaml}"
EPISODES="${EPISODES:-2000}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$REPO/work/rl_selfplay_cuda_${TS}.log"
CMDS="$REPO/work/_rl_selfplay_cuda_${TS}.sh"

cat > "$CMDS" <<EOF
#!/usr/bin/env bash
set -e
cd "$REPO"
echo "=============================================="
echo " Jailtime real-time RL self-play (CUDA Blackwell)"
echo " Config: $CONFIG"
echo " Episodes: $EPISODES"
echo " Log:     $LOG"
echo " Started: \$(date)"
echo "=============================================="
echo
python3 -u work/rl_selfplay_run.py --config "$CONFIG" --episodes "$EPISODES" --seed 20260629 "\$@" 2>&1 | tee "$LOG"
echo
echo "=============================================="
echo " Run finished at \$(date)."
echo " Hardened defender: runs/rl-selfplay-cuda/latest/defender_final/model.safetensors"
echo " Full log: $LOG"
echo " (This window stays open. Close it when done.)"
echo "=============================================="
read -n 1 -s -r -p "Press any key to close this window..."
EOF
chmod +x "$CMDS"

open -a Terminal "$CMDS"
echo "Opened a new Terminal window for the 2000-episode CUDA run."
echo "Log will be written to: $LOG"
echo "(The temp launcher script is at $CMDS)"