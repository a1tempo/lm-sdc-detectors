#!/usr/bin/env bash
# Set up and run Micah's LM-detector bench on the Pi. Safe to rerun: finished steps are skipped.
#
# On the Pi, with Arvin's card plugged in:
#   nohup bash ~/run_on_pi.sh > ~/lm-run.log 2>&1 &
#   tail -f ~/lm-run.log            # Ctrl-C stops watching, not the run
# nohup keeps it running if the SSH connection drops. Total: roughly 20-60 min (estimate),
# most of it the llama-cpp-python build.
set -euo pipefail

CARD=${CARD:-$(ls -d /media/"$USER"/*/micah-lm 2>/dev/null | head -1 || true)}
if [ ! -d "$CARD" ]; then
  echo "Can't find micah-lm/ on the card. Plug in Arvin's card, or run: CARD=/path/to/micah-lm bash run_on_pi.sh"
  exit 1
fi
echo "== start $(date)   card: $CARD"

echo "== 0. machine"
cat /proc/device-tree/model; echo
grep PRETTY /etc/os-release; uname -r; python3 --version
free -m; df -h ~ | tail -1
vcgencmd measure_temp || true; vcgencmd get_throttled || true

echo "== 1. copy kit and models from the card"
[ -d ~/pi-kit ] || cp -r "$CARD/pi-kit" ~/pi-kit
mkdir -p ~/models
for m in Qwen3-0.6B-Q4_K_M.gguf Qwen3-1.7B-Q4_K_M.gguf; do
  [ -f ~/models/$m ] || cp "$CARD/models/$m" ~/models/
done
(cd ~/models && sha256sum -c - <<'SUMS'
ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a  Qwen3-0.6B-Q4_K_M.gguf
b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897  Qwen3-1.7B-Q4_K_M.gguf
SUMS
)

echo "== 2. code (branch micah/lm-detectors, from the bundle on the card)"
[ -d ~/lm-detectors-micah ] || git clone -q -b micah/lm-detectors ~/pi-kit/lm-detectors.bundle ~/lm-detectors-micah
git -C ~/lm-detectors-micah log --oneline -1

echo "== 3. llama-cpp-python 0.3.36 (compiles llama.cpp; 10-30 min, estimate)"
[ -d ~/lm-venv ] || python3 -m venv ~/lm-venv
if ! ~/lm-venv/bin/python -c "import llama_cpp" 2>/dev/null; then
  start=$(date +%s)
  # Offline packages from the card first; if that fails, try the internet.
  ~/lm-venv/bin/pip install --no-index --find-links ~/pi-kit/offline llama-cpp-python==0.3.36 \
    || ~/lm-venv/bin/pip install llama-cpp-python==0.3.36
  echo "build took $(( $(date +%s) - start )) s"
fi
~/lm-venv/bin/python -c "import llama_cpp; print(llama_cpp.__version__); print(llama_cpp.llama_print_system_info().decode())"

cd ~/pi-kit
bench() {   # bench <size> <extra args...>
  local size=$1; shift
  if [ -f results/pi-$size/summary.json ]; then echo "already done: results/pi-$size"; return; fi
  free -m | head -2; vcgencmd measure_temp || true
  ~/lm-venv/bin/python pi_bench.py --repo ~/lm-detectors-micah \
      --gguf ~/models/Qwen3-${size^^}-Q4_K_M.gguf "$@" \
      --out results/pi-$size --compare results/laptop-$size 2>&1 \
    | grep -v "embeddings required" | tee results/pi-$size.log
}

echo "== 4. bench Qwen3-0.6B (2-5 min, estimate)"
bench 0.6b

echo "== 5. bench Qwen3-1.7B (5-15 min, estimate; peak memory was 2.7 GB on the laptop)"
avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3000 ]; then
  echo "SKIPPED: only $avail MB of memory available, need about 3000."
  echo "Close VS Code / the browser on the Pi, then rerun this script (it skips what's done)."
else
  bench 1.7b --gen-runs 1
fi

echo "== 6. save results to the card"
dest="$CARD/pi-results-$(date +%Y%m%d-%H%M)"   # the Pi's clock may be wrong; that's fine
mkdir -p "$dest"
cp -r ~/pi-kit/results/. "$dest/"
~/lm-venv/bin/pip freeze > "$dest/pip-freeze.txt"
[ -f ~/lm-run.log ] && cp ~/lm-run.log "$dest/"
sync
echo "== done $(date). Results: $dest  (also in ~/pi-kit/results)"
