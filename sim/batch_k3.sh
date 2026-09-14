#!/bin/bash
# K3 formal 20-trial run: final harness, effort=high, waves of 5 parallel.
cd "$(dirname "$0")"
PY=../.venv/bin/python
for wave in 0 1 2 3; do
  start=$((wave * 5))
  for s in $(seq $start $((start + 4))); do
    nohup $PY -u run_eval.py --scenes 1 --seed-start $s --effort high \
      --max-steps 2500 --log-dir logs_k3 > k3_final_$s.out 2>&1 &
  done
  wait
done
echo "K3 20-run complete: $(grep -l success_at_end k3_final_*.out | wc -l) trials logged"
grep -hoE "success_at_end': [01]" k3_final_*.out | sort | uniq -c
