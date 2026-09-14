#!/bin/bash
# Astra launcher: wait for quota reset (01:10) and for OPENAI_API_KEY in ./.env,
# then run GPT-6 Astra trials, 5 at a time.
cd "$(dirname "$0")"
PY=../.venv/bin/python

echo "[watcher] waiting until 01:10 ..."
while [ "$(date +%H%M)" -lt "0110" ]; do sleep 30; done
echo "[watcher] 01:10 reached, waiting for OPENAI_API_KEY in ./.env"

deadline=$(date -v+120M +%s 2>/dev/null || echo $(($(date +%s) + 7200)))
while true; do
  if [ -f .env ] && grep -q "OPENAI_API_KEY=sk" .env; then
    set -a; source .env; set +a
    echo "[watcher] key found, launching Astra trials"
    break
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "[watcher] no key by 03:10, giving up"
    exit 1
  fi
  sleep 60
done

for wave in 0 1 2 3; do
  start=$((wave * 5))
  for s in $(seq $start $((start + 4))); do
    nohup $PY -u run_eval.py --scenes 1 --seed-start $s \
      --model gpt-6-astra --base-url https://api.openai.com/v1 \
      --api-key-env OPENAI_API_KEY --effort high \
      --max-steps 2500 --log-dir logs_astra > astra_$s.out 2>&1 &
  done
  wait
done
echo "Astra 20-run complete"
grep -hoE "success_at_end': [01]" astra_*.out | sort | uniq -c
