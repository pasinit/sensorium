#!/usr/bin/env bash
# Run the GPU smoke test on a rented box you can SSH into (Vast.ai A100 spot, RunPod ...).
# PREPARED, NOT RUN. Usage, from the repo root on your Mac, with the box's SSH address:
#
#   scripts/gpu_smoke_ssh.sh root@<host> -p <port>
#
# It copies the repo (no data/, runs/ or .venv/) and the 8-episode H1 sample, installs uv and
# the `model` extra on the box, runs the smoke test (15 minutes, batch 128), and copies
# runs/gpu-smoke/report.json back. Destroy the instance afterwards: Vast bills a stopped one.
# Never put credentials on a rented host.
set -euo pipefail
ssh_args=("$@")
port_args=()
for ((i = 0; i < ${#ssh_args[@]}; i++)); do
  if [[ "${ssh_args[$i]}" == "-p" ]]; then port_args=(-p "${ssh_args[$((i + 1))]}"); fi
done
host="${ssh_args[0]}"
rsync -az -e "ssh ${port_args[*]:-}" --exclude data --exclude runs --exclude .venv --exclude .git \
  --exclude '.claude/worktrees' ./ "$host:sensorium/"
rsync -az -e "ssh ${port_args[*]:-}" data/sample_h1/ "$host:sensorium/data/sample_h1/"
ssh "${ssh_args[@]}" 'set -e; cd sensorium; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; nvidia-smi -L; uv sync --extra model; uv run python -m sensorium.model.gpu_smoke configs/model/v0.toml --paths "data/sample_h1/humanoid/*.npz" --minutes 15 --workers 4 8'
mkdir -p runs/gpu-smoke
scp "${port_args[@]/-p/-P}" "$host:sensorium/runs/gpu-smoke/report.json" runs/gpu-smoke/report.json
