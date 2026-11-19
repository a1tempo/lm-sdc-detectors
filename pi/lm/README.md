# Pi kit: run the LM detectors on the Raspberry Pi 5

Goal for the session, in order:
1. Build llama-cpp-python on the Pi.
2. Check that the Pi gives the same scores as the laptop on the same prompts.
3. Time single-pass, probe, and generative on the Pi, and record memory and temperature,
   for Qwen3-0.6B and, if time allows, Qwen3-1.7B (open item: which size to use).

These are first timings for choosing the model size. They are not the paper's overhead
numbers: those come from a separate session with the power meter (`scripts/09_overhead.py`).

## What goes on the USB stick

| Item | Size | What it is |
|---|---|---|
| `pi-kit/` (this folder) | ~230 MB | scripts, prompts, laptop results, offline packages, code bundle |
| `models/Qwen3-0.6B-Q4_K_M.gguf` | 397 MB | sha256 `ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a` |
| `models/Qwen3-1.7B-Q4_K_M.gguf` | 1.1 GB | sha256 `b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897` |

Both models are from unsloth on Hugging Face (`unsloth/Qwen3-0.6B-GGUF`, `unsloth/Qwen3-1.7B-GGUF`,
original hybrid-thinking release, not 2507).

In `pi-kit/`:
- `pi_bench.py`: the script that runs on the Pi. Needs only llama-cpp-python (no PyTorch).
- `prompts.jsonl`: the exact prompts for the 40 pilot runs, all 4 conditions (made by
  `make_prompts.py` on the laptop; that one needs the full team setup and does not run on the Pi).
- `results/laptop-0.6b/`, `results/laptop-1.7b/`: the same script run on the laptop, to compare against.
- `offline/`: llama-cpp-python source and ARM packages for Python 3.11 and 3.13, in case the Pi has no internet.
- `lm-detectors.bundle`: the team repo (branches `micah/lm-detectors` and `main`), in case GitHub is unreachable.

## On the Pi

### 0. Look at the machine (write the answers in the log)
```bash
cat /proc/device-tree/model; grep PRETTY /etc/os-release; python3 --version
free -h; vcgencmd measure_temp; vcgencmd get_throttled
```
`get_throttled` should say `throttled=0x0`. Is the active cooler (fan) on?

### 1. Copy from the USB stick
```bash
ls /media/$USER/                        # find the card (Arvin's shows as 4A21-0000 on the laptop)
USB=/media/$USER/<name>/micah-lm   # on Arvin's card: the 4A21-0000 volume
cp -r $USB/pi-kit ~/pi-kit
mkdir -p ~/models && cp $USB/models/*.gguf ~/models/
sha256sum ~/models/*.gguf               # compare with the table above
```

### 2. Get the code (my branch)
```bash
git clone -b micah/lm-detectors https://github.com/a1tempo/lm-sdc-detectors.git ~/lm-detectors-micah
# no internet:
git clone -b micah/lm-detectors ~/pi-kit/lm-detectors.bundle ~/lm-detectors-micah
```
A separate folder, so it does not disturb Arvin's checkout of `main` (his LiteRT runner).
The bench only uses `src/sdc/detectors/lm/llama_client.py` from it.

### 3. Build llama-cpp-python (compiles llama.cpp from source)
```bash
sudo apt install -y build-essential cmake python3-venv git    # probably already there
python3 -m venv ~/lm-venv
~/lm-venv/bin/pip install llama-cpp-python==0.3.36
# no internet:
~/lm-venv/bin/pip install --no-index --find-links ~/pi-kit/offline llama-cpp-python==0.3.36
```
On the laptop this took 3.5 min. On the Pi expect roughly 10–30 min (my estimate, not
measured). Use the same version, 0.3.36: a different llama.cpp can change the scores.

Check it:
```bash
~/lm-venv/bin/python -c "import llama_cpp; print(llama_cpp.__version__); print(llama_cpp.llama_print_system_info().decode())"
```
Expect `0.3.36` and a feature line that includes `NEON = 1` (ARM's vector instructions).

### 4. Run the bench, 0.6B (about 5–10 min, estimate)
```bash
cd ~/pi-kit
~/lm-venv/bin/python pi_bench.py --repo ~/lm-detectors-micah \
    --gguf ~/models/Qwen3-0.6B-Q4_K_M.gguf \
    --out results/pi-0.6b --compare results/laptop-0.6b 2>&1 | tee results/pi-0.6b.log
```
Default: the first 4 pilot runs (2 SDC, 2 masked) × 4 conditions for single-pass and probe,
then generative on 2 runs (white-box, zero-shot) with the 2048-token budget.
The repeated message `init: embeddings required but some input tokens were not marked as
outputs -> overriding` comes from llama.cpp during the probe pass and is expected (see below).

What to look at in the last lines:
- **`largest score difference`** vs the laptop. Laptop to laptop it is exactly 0 for single
  and gen (the bench reproduced the pilot). The Pi uses different CPU instructions, so a
  small difference (say below 0.01) is fine. A large one means something differs: stop and look.
- **`same reasoning length on k/n`**: generative uses greedy decoding, so a tiny numeric
  difference can still send the reasoning down another path. If the lengths differ, read the
  `reasoning_*.txt` files in both results folders.
- **median seconds** per pass, **peak resident memory**, and **temperature / throttled**.

### 5. Same for 1.7B (likely 15–30 min with the defaults, estimate)
```bash
~/lm-venv/bin/python pi_bench.py --repo ~/lm-detectors-micah \
    --gguf ~/models/Qwen3-1.7B-Q4_K_M.gguf \
    --out results/pi-1.7b --compare results/laptop-1.7b 2>&1 | tee results/pi-1.7b.log
```
If time is short, add `--gen-runs 1`. On the laptop, 1.7B reasoned 2–4× longer than 0.6B
(467–1,594 tokens vs 277–464) and took 36–125 s per generative run, so on the Pi one run
may take several minutes.

### 6. Bring the results back
```bash
cp -r ~/pi-kit/results $USB/pi-results-$(date +%F)
pip list --path ~/lm-venv/lib/python3*/site-packages > $USB/pi-results-$(date +%F)/pip-list.txt
```
Then log it in `results-log.md` (device, OS, Python, temperatures, the numbers, the command).

## Things to know if asked

- **Why not run `scripts/07_score_lm.py` on the Pi?** It needs PyTorch, only because it
  reads the run files through team code that imports torch (`records/io.py` and
  `observations/serialize.py` import `MONITORED_POINTS` from `model/monitor.py`, which imports
  torch). PyTorch would also add a few hundred MB to the memory we are measuring. The bench
  calls the same `QwenClient` methods with the same prompts, so it measures the same work.
  Possible fix to propose to Arvin: move `MONITORED_POINTS` into a small torch-free module.
- **Why is the probe pass slower and bigger than single-pass?** To return the hidden
  state, llama.cpp switches to "embeddings mode", which computes outputs for every prompt
  token, not just the last one. On the laptop (0.6B, longest prompt = white-box reference),
  resident memory was 1,051 MB after loading, 1,105 MB after a single-pass, and 1,546 MB
  after a probe pass; generation added nothing on top. The probe pass took about 1.3–1.4×
  the single-pass time.
  It also changes the score in the 7th decimal place (still the same scores to 6 places).
- **Threads:** the Pi 5 has 4 cores, so `--threads 4` (the default). On the laptop,
  4 vs 8 threads gave identical scores.
- **Flash attention stays off and the context is 2048 tokens**, as everywhere else.
- **If the Pi gets hot:** it slows itself down at about 80–85 °C (from memory). `vcgencmd
  get_throttled` other than `0x0` means it did. Note it; don't trust those timings.

## The long run (Nov 17–18): detection quality on the Pi, 0.6B

The bench above uses 4 runs, which is enough for speed and memory but says nothing about
detection quality. The long run scores 400 fit + 400 eval runs on the Pi.

1. **Pick runs and write prompts (laptop):** `make_subset_prompts.py --split fit` and
   `--split eval`. Per split: every SDC the range check misses (31), 169 other SDCs at
   random, 200 masked at random, seed 42. Writes `prompts-<split>.jsonl` (not in git, 3 MB
   each, rebuilt by the script) and `prompts-<split>.manifest.json` (run ids, group sizes,
   reference ranges).
2. **Run on the Pi:** copy `pi_bench.py`, `long_run.sh` and the two prompt files to
   `~/pi-kit/`, then `nohup bash ~/pi-kit/long_run.sh > ~/lm-long.log 2>&1 &`. It runs only
   the probe pass (`--passes probe`), which gives the single-pass score and the hidden
   state in one decode. It saves every 20 queries and continues where it stopped if rerun.
   Took 5.5 h on the fanless Pi (median 89 °C, throttled).
3. **Analyse (laptop):** `analyze_long_run.py <results prefix>`, e.g.
   `.../results/pi-20261118/pi-0.6b`. Needs `results.csv` and `hidden.npz` for both splits
   and the team's run files. Output of the Nov 18 run: `results/pi-20261118/analysis.txt`.

The hard SDCs are over-represented on purpose, so the script reports AUROC for them, for
the other SDCs, and a weighted combination that estimates the full-split value.
