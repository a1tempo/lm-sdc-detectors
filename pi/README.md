# Raspberry Pi 5 setup

Target: Pi 5, 8 GB, 4x Cortex-A76 @ 2.4 GHz, CPU only (no Hailo NPU: it cannot be
fault-injected or observed layer by layer). The classifier and the LM detectors
share the same 8 GB.

**Status: nothing here has been done yet.** All code so far has only run on a Mac. See
[docs/README.md](../docs/README.md) for full project context.

## Running the faulty LiteRT models (scripts/11_run_faulty_tflite.py)

The faulty models are made on a Mac by `scripts/10_export_faulty_tflite.py` (one FP32
`.tflite` per weight fault, about 43 MB each) and carried to the Pi on an SD card / USB
drive. The Pi runner needs only numpy, Pillow, PyYAML and ai-edge-litert, not PyTorch.

1. Get the code and a light Python environment (64-bit Raspberry Pi OS):
   ```bash
   git clone https://github.com/a1tempo/lm-sdc-detectors.git
   cd lm-sdc-detectors
   python3 -m venv .venv
   .venv/bin/pip install numpy pillow pyyaml ai-edge-litert
   .venv/bin/pip install --no-deps -e .
   ```
   The EuroSAT images come with the repo. If `ai-edge-litert` has no wheel for the Pi's
   Python version, use a Python version it supports and note it here.
2. Copy the clean model from the shared Drive folder: `lm-sdc-share/models/eurosat_fp32.tflite`
   to `models/eurosat_fp32.tflite`. The runner labels every faulty run against it.
3. Phone notifications (optional): `echo <your-topic> > .ntfy_topic` (see `scripts/notify.sh`).
4. Plug in the card and find it: `ls /media/$USER` (it is named `fLiteRT`). Then:
   ```bash
   source .venv/bin/activate
   python scripts/11_run_faulty_tflite.py --models-dir /media/$USER/fLiteRT/lm-sdc --count 3
   scripts/notify.sh python scripts/11_run_faulty_tflite.py --models-dir /media/$USER/fLiteRT/lm-sdc
   ```
   The first command is a quick check; the second runs the rest (it skips what is done
   and resumes after Ctrl-C).
5. Results go to `runs/pi_tflite/results.csv` (gitignored). Copy it back to the Mac / Drive.
   Each row has the PyTorch label next to the Pi label (`same_as_pytorch`).

## Overhead of the classical detectors (scripts/09_overhead.py)

1. On the Mac, build the kit (LiteRT environment): `~/.venvs/lm-sdc-tflite/bin/python
   scripts/09_prepare_overhead.py` -> `runs/overhead_kit/` (about 92 MB).
2. Copy the repo and the kit to the Pi (on the Mac; keeps the Pi's own `.venv`):
   ```bash
   rsync -av --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
       --include 'runs/' --include 'runs/overhead_kit/***' --exclude 'runs/*' \
       ./ reem@<pi>:~/lm-sdc-detectors/
   ```
3. On the Pi, with internet once: `.venv/bin/pip install scikit-learn==<kit.json scikit_learn>`.
4. Run it (keeps going after SSH disconnects; about 15-30 min fanless):
   ```bash
   nohup scripts/notify.sh .venv/bin/python scripts/09_overhead.py > ~/overhead.log 2>&1 &
   ```
   Without internet the notification waits and retries for 2 hours. Check with
   `tail -12 ~/overhead.log`.
5. Copy `runs/overhead/pi/overhead.json` back to the Mac.

The Pi has no fan and throttles at about 85 C; the LM timings were taken the same way, and
the script logs temperature and `vcgencmd get_throttled` before and after each phase.

## TODO
- [ ] OS and Python version (record here so everyone matches; the project needs 3.11 or newer)
- [ ] `pip install -e .[tflite,lm]` on the Pi; note any wheels that must be built from source
- [ ] Build llama.cpp / llama-cpp-python for ARM (OpenBLAS? NEON flags?)
- [ ] Download Qwen3 0.6B and 1.7B GGUF (4-bit) into `models/`
- [ ] Confirm yes/no next-token probabilities and final hidden states are readable
      (`sdc.detectors.lm.llama_client.QwenClient`)
- [ ] Confirm PyTorchFI runs on the Pi, or decide to inject on a workstation and
      only measure overhead on the Pi. It works on a Mac with torch 2.15
- [ ] Pin package versions so the Pi and other machines match
- [ ] If the campaign runs here, try stopping and resuming `scripts/05_inject.py`
- [ ] Power measurement: inline tester model, idle-power procedure (Shehayib & Kogge, 2026)
- [ ] Thermal: cooler/fan setup, log `/sys/class/thermal/thermal_zone0/temp`
