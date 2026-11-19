"""LAPTOP ONLY. Write the exact LM prompts for a few pilot runs to prompts.jsonl.

Why: the team scoring script (scripts/07_score_lm.py) needs PyTorch, only because it
reaches the run files through code that imports torch. The Pi bench (pi_bench.py) needs
only llama-cpp-python. So we build the prompts here with the team code, and the Pi only
has to read text.

Each line = one run, one access level, one prompt condition, with two prompts:
  "prompt_single"  thinking off (single-pass and probe detectors)
  "prompt_think"   thinking on  (generative detector)
The label is stored for display only; it never goes into a prompt.

Run from team/lm-sdc-detectors:
    ../../.venv/bin/python ../../pi-kit/make_prompts.py --gguf ../../models/Qwen3-0.6B-Q4_K_M.gguf
"""

import argparse
import json
from pathlib import Path

from sdc.detectors.lm.generative import GenerativeDetector
from sdc.detectors.lm.llama_client import QwenClient
from sdc.detectors.lm.single_pass import SinglePassDetector
from sdc.records.io import load_runs

LEVELS = ["blackbox", "whitebox"]
CONDITIONS = ["zero_shot", "reference"]


def main(args):
    pilot = json.loads(Path("runs/scores/pilot/manifest.json").read_text())
    ranges = pilot["reference_ranges"]          # the exact ranges the pilot prompts showed
    runs = load_runs("runs/faults/fit")
    runs = runs[runs["run_id"].isin(pilot["run_ids"])]

    # Interleave SDC and masked so that "the first k runs" is always balanced.
    sdc = runs[runs["label"] == "sdc"].sort_values("run_id").to_dict("records")
    masked = runs[runs["label"] == "masked"].sort_values("run_id").to_dict("records")
    ordered = [r for pair in zip(sdc, masked) for r in pair]

    # The client is only needed for the chat template (no decoding happens here).
    client = QwenClient(args.gguf, n_ctx=2048, n_threads=4, flash_attn=False)
    lines = []
    for i, row in enumerate(ordered):
        for level in LEVELS:
            for cond in CONDITIONS:
                single = SinglePassDetector(client, cond, level).fit(None, ranges)
                think = GenerativeDetector(client, cond, level).fit(None, ranges)
                lines.append({"run_index": i, "run_id": row["run_id"], "label": row["label"],
                              "access_level": level, "condition": cond,
                              "prompt_single": single.prompt(row),
                              "prompt_think": think.prompt(row)})
    out = Path(args.out)
    out.write_text("".join(json.dumps(x) + "\n" for x in lines))
    print(f"wrote {len(lines)} prompt pairs for {len(ordered)} runs to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gguf", required=True)
    p.add_argument("--out", default=str(Path(__file__).parent / "prompts.jsonl"))
    main(p.parse_args())
