#!/usr/bin/env bash
# The long Pi run (Nov 17): Qwen3-0.6B probe pass on 400 fit + 400 eval runs, all four
# prompt types. One pass gives both the single-pass score and the probe's hidden state.
# About 5 hours (estimate). Safe to rerun: it continues where it stopped.
#   nohup bash ~/pi-kit/long_run.sh > ~/lm-long.log 2>&1 &
#   tail -3 ~/lm-long.log
set -uo pipefail
cd ~/pi-kit
echo "== start $(date)"; free -m | head -2; vcgencmd measure_temp; vcgencmd get_throttled
for split in fit eval; do
  out=results/pi-0.6b-$split
  if [ -f $out/summary.json ]; then echo "already done: $out"; continue; fi
  echo "== $split $(date)"
  ~/lm-venv/bin/python pi_bench.py --repo ~/lm-detectors-micah \
      --gguf ~/models/Qwen3-0.6B-Q4_K_M.gguf --prompts prompts-$split.jsonl \
      --runs 100000 --gen-runs 0 --passes probe --out $out 2>&1 \
    | grep --line-buffered -v "embeddings required"
done
CARD=$(ls -d /media/"$USER"/*/micah-lm 2>/dev/null | head -1 || true)
if [ -d "$CARD" ]; then
  dest="$CARD/pi-long-$(date +%Y%m%d-%H%M)"; mkdir -p "$dest"
  cp -r results/pi-0.6b-fit results/pi-0.6b-eval "$dest/"; cp ~/lm-long.log "$dest/" 2>/dev/null; sync
  echo "copied to $dest"
fi
echo "== done $(date)"; vcgencmd measure_temp; vcgencmd get_throttled
