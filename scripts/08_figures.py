"""Draw the paper / poster figures from the tables 08_evaluate.py wrote.

Reads <paths.results>/evaluation/*.csv and writes <paths.results>/figures/ (PNG and PDF):
    roc_classical_vs_lm      ROC curves, classical vs. LM, black-box and white-box
    auroc_classical_vs_lm    AUROC with 95% intervals, all SDCs and the 31 in-range SDCs
    data_size                AUROC vs. faults per image used for training (1, 2, 4, 8, 10)
    strata                   AUROC by fault site and by flipped bit
    layers                   AUROC by faulted layer
"""

import argparse
from pathlib import Path

import pandas as pd

from sdc.config import load_config
from sdc.eval import figures
from sdc.injection.faults import injectable_layers
from sdc.model.resnet18 import build_resnet18


def main(config_path: str | None) -> None:
    results = Path(load_config(config_path)["paths"]["results"])
    tables = results / "evaluation"
    out = results / "figures"

    comparison_path = tables / "lm_comparison.csv"
    if comparison_path.exists() and (tables / "lm_comparison_scores.csv").exists():
        comparison = pd.read_csv(comparison_path)
        figures.plot_roc_curves(pd.read_csv(tables / "lm_comparison_scores.csv"), comparison,
                                out / "roc_classical_vs_lm")
        figures.plot_auroc_summary(comparison, out / "auroc_classical_vs_lm")
    else:
        print("no LM comparison tables: run 08_evaluate.py with the LM subset options first")
    figures.plot_data_size(pd.read_csv(tables / "data_size.csv"), out / "data_size")
    figures.plot_strata({by: pd.read_csv(tables / f"strata_{by}.csv") for by in ("fault_site", "bit_group")},
                        out / "strata")
    layer_order = list(injectable_layers(build_resnet18()))
    figures.plot_layers(pd.read_csv(tables / "strata_layer.csv"), layer_order, out / "layers")
    print(f"wrote {sorted(p.name for p in out.glob('*.png'))} to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    main(parser.parse_args().config)
