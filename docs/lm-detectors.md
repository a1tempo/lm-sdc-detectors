# Language-model detectors

**Last updated: 2026-11-12, branch `micah/lm-detectors` (based on `00de347`). Owner: Micah.**

This file explains the language-model (LM) detectors: what each piece does, how to run it, how to read its outputs, what was decided and why, and what the first runs showed. The project overview is in [`docs/README.md`](README.md).

## Contents
1. [In brief](#1-in-brief)
2. [The three detectors](#2-the-three-detectors)
3. [Files](#3-files)
4. [Setup](#4-setup)
5. [Running](#5-running)
6. [Output files and how to read them](#6-output-files-and-how-to-read-them)
7. [Decisions and why](#7-decisions-and-why)
8. [First results (laptop pilot, not paper numbers)](#8-first-results-laptop-pilot-not-paper-numbers)
9. [Findings that affect the classical detectors](#9-findings-that-affect-the-classical-detectors)
10. [Open questions](#10-open-questions)

---

## 1. In brief

- **Built and tested:** all three LM detectors (single-pass, generative, probe), the prompt builder, and `scripts/07_score_lm.py`. 36 new tests; the whole suite passes (497 with the model file present; without it, the 7 tests that need the real model are skipped).
- **Run so far:** a small pilot on fit runs only, on a laptop (Intel Core Ultra X7 358H, 8 threads). No eval runs have been scored. Nothing has run on the Pi yet.
- **Main open decision:** how many runs to score, and on which machine. Scoring everything on the laptop would take more than a day (estimate in [section 8](#8-first-results-laptop-pilot-not-paper-numbers)).

---

## 2. The three detectors

All three use one model, **Qwen3-0.6B, 4-bit**, through llama.cpp. Each gives a run a score, where higher means more likely an SDC. Qwen itself is never changed (it stays "frozen").

| Detector | What it does | Score |
|---|---|---|
| **Single-pass** (`lm_single_pass`) | Thinking off. One pass through Qwen, then read how likely the next word is "yes" vs "no" | P(yes) / (P(yes) + P(no)) |
| **Generative** (`lm_generative`) | Thinking on. Qwen writes its reasoning, then answers; we read P(yes) and P(no) at the answer | Same formula; 0.5 if there's no clean yes/no answer |
| **Probe** (`lm_probe`) | Same prompt and same single pass as single-pass, but we keep Qwen's 1,024 internal numbers at the answer position and train a logistic regression on them (fit runs only) | The regression's probability of SDC |

The probe and single-pass are two readouts of the same 1,024 numbers. Single-pass uses Qwen's built-in "yes minus no" weights, which were set during Qwen's general training. The probe learns its own weights from labeled fault runs. This is the "trained decision head" comparison in RQ4.

Each detector runs at both access levels (black-box, white-box) and in both prompt conditions (zero-shot, reference), so there are 4 versions of each.

---

## 3. Files

| File | What it does |
|---|---|
| `src/sdc/detectors/lm/llama_client.py` | `QwenClient`, the only code that talks to Qwen. Loads the model with pinned settings, renders Qwen's chat template (`chat_prompt`), and provides `yes_no_probs`, `yes_no_probs_and_hidden` (one pass, both outputs), and `generate_then_answer` (thinking mode) |
| `src/sdc/observations/serialize.py` | Turns a run record into text. `observation_columns(access_level)` is the one list of columns a detector may see. `serialize_observation` reads a row only through it, so `label`, fault fields, and clean-run fields can't reach a prompt. Also `output_ranges` and `serialize_reference` for the reference condition |
| `src/sdc/detectors/lm/prompts.py` | The prompt wording: task description, what the numbers mean, and the question |
| `src/sdc/detectors/lm/common.py` | `LMDetector`, shared by all three: builds each run's prompt, loops over runs, times each one. `yes_no_score` is the score formula |
| `src/sdc/detectors/lm/single_pass.py` | `SinglePassDetector` |
| `src/sdc/detectors/lm/generative.py` | `GenerativeDetector` |
| `src/sdc/detectors/lm/probe.py` | `ProbeDetector`: `extract` (internal numbers + yes/no from one pass, cached by `run_id`), `load_hidden`, `fit`, `score` |
| `scripts/07_score_lm.py` | Picks a subset of runs, scores it with any of the detectors in any of the conditions, saves one file per combination. Resumable |
| `tests/test_llama_client.py` | 6 tests against values measured on 2026-11-02 (need the model file) |
| `tests/test_serialize.py` | 13 tests, including that answer-key columns never appear in a prompt |
| `tests/test_lm_detectors.py` | 17 tests, mostly with a fake Qwen so they run in seconds; one uses the real model |

`src/sdc/detectors/lm/api_reference.py` (the optional hosted-API detector) is still a stub.

---

## 4. Setup

**Model file.** Not in the repo. Download `Qwen3-0.6B-Q4_K_M.gguf` from the Hugging Face repo `unsloth/Qwen3-0.6B-GGUF` (we used repo commit `50968a4468ef4233ed78cd7c3de230dd1d61a56b`) and check it:

```bash
sha256sum Qwen3-0.6B-Q4_K_M.gguf
# ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a   (396,705,472 bytes)
```

Qwen's own GGUF repo only has 8-bit files, so the 4-bit file comes from unsloth, built with an importance matrix (calibration data). The paper should say so.

**Packages.** `pip install -e ".[lm,dev]"` adds `llama-cpp-python`, which compiles llama.cpp from source (about 2 minutes; needs `cmake` and a C compiler). Also needs `jinja2` for the chat template.

**Environment the LM tests passed on:** Fedora Linux x86_64, Python 3.15.2, llama-cpp-python 0.3.36 (CPU build), torch 2.15.0+cpu, torchvision 0.29.2+cpu, pytorchfi 0.6.0, scikit-learn 1.9.1, pandas 3.1.1, pyarrow 26.0.1, numpy 2.5.4, jinja2 3.1.6, pytest 9.2.0. Changing llama-cpp-python's version can change scores slightly; pin it for anything reported.

**Run the tests.** The real-model tests look for `models/Qwen3-0.6B-Q4_K_M.gguf`, or the path in `SDC_GGUF`:

```bash
SDC_GGUF=/path/to/Qwen3-0.6B-Q4_K_M.gguf pytest
```

---

## 5. Running

`configs/default.yaml` has `paths.qwen_gguf: models/qwen3-0.6b-q4.gguf`, which doesn't name the exact file. Pass `--gguf` instead.

**Single-pass and generative**, 20 SDC + 20 masked fit runs, all 4 conditions:

```bash
python scripts/07_score_lm.py --gguf models/Qwen3-0.6B-Q4_K_M.gguf \
    --split fit --n-sdc 20 --n-masked 20 --threads 8 \
    --detectors single_pass generative --out runs/scores/pilot
```

**Probe, step 1: extract the internal numbers** (training happens later, from the saved file):

```bash
python scripts/07_score_lm.py --gguf models/Qwen3-0.6B-Q4_K_M.gguf \
    --split fit --n-sdc 200 --n-masked 200 --threads 8 \
    --detectors probe --access-levels whitebox --conditions zero_shot \
    --out runs/scores/probe_pilot
```

**Options:**
- `--n-sdc N`: sample N SDC runs; leave it out to take every SDC in the split.
- `--n-masked N`: sample N masked runs (required). Crashes are never scored.
- `--seed`: the sample's seed (default 42). Same seed, same runs.
- `--access-levels`, `--conditions`: run only some of the 4 conditions (default: all, from the config).
- `--max-tokens`: thinking-mode word budget (default 2048; the context size, `lm.n_ctx` = 2048, also limits it).

**Stopping and resuming.** Each detector/condition is saved to its own file, written only when complete. Stop at any time and rerun the same command: finished files are skipped, so you lose at most the one in progress. `manifest.json` in the output folder records the exact run IDs and settings. Rerunning into the same folder with different settings stops with an error, so results from different settings can't mix.

**Training and scoring the probe from a saved file:**

```python
import numpy as np
from sdc.detectors.lm.probe import ProbeDetector

z = np.load("runs/scores/probe_pilot/probe_hidden__whitebox__zero_shot.npz")
probe = ProbeDetector(client=None, condition="zero_shot", access_level="whitebox")
probe.load_hidden(z["run_ids"], z["hidden"])
probe.fit(fit_runs)            # fit runs with a "label" column: sdc or masked only
scores = probe.score(eval_runs)   # eval runs must also be in the loaded file
```

---

## 6. Output files and how to read them

**`<detector>__<access_level>__<condition>.parquet`**, one row per scored run. Join to the run records on `run_id`. Labels are deliberately not in these files.

| Column | Meaning |
|---|---|
| `run_id` | The run, as in the fault-run shards |
| `detector`, `access_level`, `prompt_condition` | Which version produced the row |
| `score` | The detector's score; higher = more likely SDC |
| `p_yes`, `p_no` | Probability that the next word is "yes" / "no" at the answer position. For generative, empty when there was no clean answer |
| `parsed` | Generative: did Qwen give a clean yes/no answer? If not, `score` is 0.5. Single-pass: always true |
| `yes_no_mass` | Single-pass only: P(yes) + P(no). Near 1 means Qwen really is choosing between the two words (it was ≥ 0.999 in the pilot) |
| `n_gen_tokens` | Generative only: how many word pieces Qwen wrote before the answer |
| `reasoning` | Generative only: the text Qwen wrote, including the `<think>` block |
| `elapsed_s` | Wall-clock seconds for this run on the machine that ran it. Laptop times are not paper numbers; overhead is measured on the Pi |

**`probe_hidden__<access_level>__<condition>.npz`**: `run_ids`, `hidden` (one row of 1,024 numbers per run, float32), `p_yes`, `p_no` (from the same pass, so the single-pass score for these runs comes free), `elapsed_s`.

**`manifest.json`**: the run IDs, the subset settings, the model file name and sha256, llama-cpp-python version, context size, threads, flash attention (off), decoding (greedy), token budget, the reference ranges shown in prompts, and the git commit (ends in `-dirty` if the code had uncommitted changes).

---

## 7. Decisions and why

| Decision | Reason |
|---|---|
| The answer words are `yes` and `no`, lowercase, no leading space (token ids 9693 and 2152) | Measured: that's what Qwen3 actually outputs at the answer position; the two together get ≥ 99.98% of the probability. `" yes"` and `"Yes"` are about e^-10 less likely |
| Thinking on/off uses the chat template's `enable_thinking` flag, not `/think` / `/no_think` | With thinking off, the template itself adds an empty `<think></think>` block, so the answer is the very next word |
| Flash attention is off everywhere | Measured: turning it on changes the single-pass score on the same prompt (0.52 vs 0.39). All fitting, evaluation, and Pi runs must use the same setting |
| `n_batch` = context size | A whole prompt goes to llama.cpp in one call, and llama.cpp aborts the process if a call has more than `n_batch` word pieces (default 512). White-box reference prompts were 508–562 in the pilot |
| Generative decoding is greedy (always the most likely next word) | Repeatable: the same prompt always gives the same reasoning and score |
| Numbers in prompts use 4 significant digits (`0.4076`, `1.103e+38`) | One format that handles both normal values and 1e38. The proposal's example uses 2 decimals, which would print 1e38 as a 39-digit number |
| The question is "Did a fault change this run's predicted class? Answer with exactly one word: yes or no." | Matches the SDC label exactly (`label_run`: argmax differs from the clean prediction). "Is this run corrupted?" would be vaguer: masked runs with huge activations are arguably "corrupted" too |
| Reference condition, white-box: per-layer [min, max] from `runs/reference_ranges.json`, plus the output ranges below | The proposal's definition: show the LM the normal ranges the classical detectors are fit on |
| Reference condition, black-box: the [min, max] of `top1_prob` and `entropy` over clean fit runs (`output_ranges`) | The proposal only defines per-layer ranges, which mean nothing at black-box level. This applies the same rule (clean fit runs, min/max) to the two numbers black-box can see. **Needs one sentence in the paper** |
| `observation_columns` is the only way a detector reads a run | Rows mix observations with the answer key (`label`, `flipped_value`, `clean_pred`, ...). One shared list keeps the LM and the learned detector on identical inputs and keeps answer columns out (tested). `feature_vector` in `stats.py` can use it too |
| Crashes are never scored by LM detectors | Detection is SDC vs masked; crashes are reported separately. Saves time |
| No clean generative answer → score 0.5, `parsed` = false | Proposal §6.3. The no-answer rate gets reported |
| The probe uses the single-pass prompt and its pass | One pass gives both the single-pass score and the probe's input, which is the "shared forward pass" in the proposal's E2 |
| The probe standardizes each of the 1,024 numbers, then fits sklearn's default logistic regression (C = 1) | "Same classifier as the learned detector" per the proposal. The learned detector isn't written yet, so these settings should match it once it is. C is not tuned yet |
| Hidden states are saved to disk, keyed by `run_id` | Extraction is the slow part; training and scoring the probe then take seconds |

---

## 8. First results (laptop pilot, not paper numbers)

Fit runs only, laptop, Qwen3-0.6B Q4_K_M, settings as in section 7. **These are sanity checks.** Small samples, no eval runs, no confidence intervals. AUROC is the chance that a random SDC run gets a higher score than a random masked run (1 = perfect, 0.5 = coin flip, below 0.5 = backwards).

**Pilot: 20 SDC + 20 masked fit runs, single-pass and generative, all 4 conditions** (`runs/scores/pilot`).

| Condition | Single-pass AUROC | Generative AUROC | Generative clean answers | Generative median s/run |
|---|---|---|---|---|
| black-box, zero-shot | 0.67 | 0.62 | **50%** | 43.2 |
| black-box, reference | 0.54 | 0.91 | 100% | 11.2 |
| white-box, zero-shot | 0.72 | 0.50 | 98% | 12.5 |
| white-box, reference | 0.44 | 0.33 | 100% | 12.4 |

- Single-pass gives high scores to everything, SDC or masked: the average per condition is 0.70–0.90 for both labels. Median 0.5–1.8 s per run.
- Generative, black-box, zero-shot: with no reference, Qwen kept reasoning until it hit the 2048 context limit on half the runs.
- Generative, white-box, reference: on 11 of 18 big-number SDCs it answered "no". Example (`fit-00013825`): its reasoning says layer3's max of 5.639e+36 is "way higher", then concludes "all the statistics ... are the same".
- Generative, black-box, reference: it compares `entropy=0` with the normal minimum (3.96e-7) and says yes. See [section 9](#9-findings-that-affect-the-classical-detectors).

**Probe: 200 SDC + 200 masked fit runs, white-box, zero-shot** (`runs/scores/probe_pilot`). 5-fold cross-validation within these runs, folds grouped by image so the same image is never in both training and test.

| Scorer | AUROC, all 400 | AUROC, the 9 SDCs whose output isn't exactly one-hot |
|---|---|---|
| Probe on Qwen's internal numbers | 0.989 | 0.75 |
| The same regression on the 32 raw white-box numbers (rough stand-in for the learned detector) | 0.962 | 0.91 |
| Single-pass score from the same pass | 0.642 | 0.28 |

- Qwen's internal numbers carry the information (probe 0.99), even though its own yes/no answer barely does (0.64).
- The raw-number stand-in is crude: plain standardization of values up to 1e38, no log. A real learned detector should do better (see section 9). So **this does not show the probe beats the learned detector.** The 9-SDC column is too small to rank anything.
- Getting the internal numbers costs about 25% more time per run (median 1.71 s vs 1.37 s), because llama.cpp computes outputs for every prompt word piece when they're requested.

**Rough cost of the full plan on the laptop (estimate from the pilot means, not measured):** about 4.4 s (single-pass, 4 conditions) + 74 s (generative, 4 conditions) per run. All 767 eval SDCs plus 767 masked would take about 33 hours, before the fit runs needed for thresholds and the probe.

---

## 9. Findings that affect the classical detectors

**Most SDC runs have entropy exactly 0.** From the stored `entropy` column (computed by `blackbox_stats`):

| | fit | eval |
|---|---|---|
| SDC with entropy exactly 0 | 720 / 771 (93%) | 699 / 767 (91%) |
| masked with entropy exactly 0 | 108 / 53,677 (0.2%) | 98 / 54,269 (0.2%) |
| clean with entropy exactly 0 | 0 / 910 | 0 / 920 |

Why: a bit-30 flip multiplies a value by about 2^128. The logits end up around ±1e35, with gaps between them of around 1e33, so softmax is exactly one-hot in float64 (anything below about e^-745 rounds to 0). Example `fit-00003718`: softmax of its stored logits is exactly `[0 0 0 1 0 0 0 0 0 0]`.

**Consequences:**
- The confidence check as specified (score = entropy, higher = more suspicious) would rank these SDCs as the *least* suspicious runs.
- Black-box access is not blind here: most SDCs are visible in the output alone.
- The hard cases for every detector are the roughly 50–70 SDCs per split that are not exactly one-hot.

**Tip for the learned detector:** white-box values reach 1e38. Plain standardization lets a few huge values swamp the rest; a signed log (for example `sign(x) * log1p(|x|)`) keeps them on a usable scale.

---

## 10. Open questions

1. **How many runs to score, and where.** The laptop can't do the full plan in time (section 8). Options: smaller masked samples, fewer generative conditions, or running detection quality on the lab's H200 server (the proposal allows detection quality off the Pi; overhead is always measured on the Pi).
2. **Qwen3-0.6B or 1.7B.** The pilot's reasoning failures argue for trying 1.7B. Decide after timing on the Pi.
3. **The probe's regularization (C).** Tune within fit runs only.
4. **Pi demo (next week).** Needs llama-cpp-python built on the Pi's ARM CPU, the model file, and a few runs from `runs/` copied over.
5. **Config.** `paths.qwen_gguf` could name the exact file (`models/Qwen3-0.6B-Q4_K_M.gguf`); the scripts take `--gguf` for now.

---

*AI disclosure: written with the help of Claude (Anthropic).*
