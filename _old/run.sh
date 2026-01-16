#!/usr/bin/env bash
set -euo pipefail

# Reference inputs:
# - impact speed: 5.7 m/s
# - stroke: 10.5 cm = 0.105 m
# - total mass: 75 kg (default in spine-sim.py)
#
# auto-dt is enabled by default in analytic mode.

# for maxg in 16 24 32; do
#   ./spine-sim.py \
#     --mode analytic \
#     --impact-speed-mps 5.7 \
#     --stroke-m 0.105 \
#     --max-g "$maxg" \
#     --outdir "out_ref_g${maxg}" \
#     --debug
# done


./spine-sim.py --mode analytic --impact-speed-mps 5.7 --stroke-m 0.054 --max-g 42 --outdir koroyd

