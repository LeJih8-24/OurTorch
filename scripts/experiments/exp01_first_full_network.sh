#!/usr/bin/env bash
# Experiment 01 (2026-11-03): first real training of my_torch_analyzer.
# full_network.conf (5 classes), trained on 9 000 positions of the dataset,
# evaluated on 3 000 other positions. See docs/11-journal-des-experiences.md.
#
# Usage (from anywhere, with NumPy available, e.g. after `source .venv/bin/activate`):
#     bash scripts/experiments/exp01_first_full_network.sh
# Outputs go to experiments/output/exp01/ (git-ignored).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/experiments/output/exp01"
mkdir -p "$OUT"
cd "$OUT"

make -s -C "$ROOT" re > /dev/null
rm -f full_network_1.nn trained.nn
"$ROOT/my_torch_generator" "$ROOT/configs/full_network.conf" 1 > /dev/null

# 3 000 positions of each family from the "many pieces" files for training,
# the next 1 000 of each for testing (never seen during training).
for family in check checkmate nothing; do head -n 3000 "$ROOT/dataset/$family/many_pieces.txt"; done > train.txt
for family in check checkmate nothing; do sed -n '3001,4000p' "$ROOT/dataset/$family/many_pieces.txt"; done > test.txt

"$ROOT/my_torch_analyzer" --train --save trained.nn full_network_1.nn train.txt 2> train.log | tee summary.txt
"$ROOT/my_torch_analyzer" --predict trained.nn test.txt > predictions.txt
python3 "$ROOT/scripts/evaluate.py" trained.nn predictions.txt test.txt | tee evaluation.txt
