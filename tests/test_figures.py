import numpy as np
import pandas as pd

from sdc.eval import figures


def comparison_tables():
    rng = np.random.default_rng(0)
    names = ["boosted", "learned", "lm_probe", "lm_single_pass"]
    detectors = [f"{n}_{lvl}_subset" if not n.startswith("lm") else f"{n}_{lvl}_{c}"
                 for lvl in ("blackbox", "whitebox") for n in names
                 for c in (("zero_shot", "reference") if n.startswith("lm") else ("",))]
    detectors = [d.replace("__", "_") for d in detectors] + ["confidence_blackbox", "range_whitebox"]
    y = np.r_[np.ones(20, bool), np.zeros(20, bool)]
    scores = pd.concat([pd.DataFrame({"detector": d, "is_sdc": y, "weight": 2.0,
                                      "score": rng.random(40) + y}) for d in detectors])
    table = pd.DataFrame({"detector": detectors, "auroc_weighted": 0.9, "auroc_low": 0.85,
                          "auroc_high": 0.95, "auroc_in_range": 0.7})
    return scores, table


def test_all_figures_render(tmp_path):
    scores, table = comparison_tables()
    figures.plot_roc_curves(scores, table, tmp_path / "roc")
    figures.plot_auroc_summary(table, tmp_path / "auroc")
    sizes = pd.DataFrame([{"faults_per_image": k, "detector": f"{e}_{lvl}", "auroc": 0.95,
                           "auroc_low": 0.94, "auroc_high": 0.96}
                          for k in (1, 2, 4, 8, 10) for e in ("boosted", "learned")
                          for lvl in ("blackbox", "whitebox")])
    figures.plot_data_size(sizes, tmp_path / "size")
    strata = {by: pd.DataFrame([{"detector": d, by: g, "n_sdc": 5, "auroc": 0.9, "auroc_low": 0.8,
                                 "auroc_high": 0.95}
                                for d in ("boosted_whitebox", "learned_whitebox", "range_whitebox")
                                for g in groups])
              for by, groups in (("fault_site", ("weight", "activation")),
                                 ("bit_group", ("exponent", "sign")))}
    figures.plot_strata(strata, tmp_path / "strata")
    layers = pd.DataFrame([{"detector": d, "layer": layer, "n_sdc": 3, "auroc": 0.9}
                           for d in ("boosted_whitebox", "learned_whitebox", "range_whitebox")
                           for layer in ("conv1", "fc")])
    figures.plot_layers(layers, ["conv1", "fc"], tmp_path / "layers")
    for name in ("roc", "auroc", "size", "strata", "layers"):
        assert (tmp_path / f"{name}.png").stat().st_size > 10_000
        assert (tmp_path / f"{name}.pdf").exists()


def test_overhead_figure_renders(tmp_path):
    rows = []
    for name, model, ms, auc in [("Range check", "", 12.0, 0.98), ("Entropy check", "", 0.05, 0.07),
                                 ("Logistic regression", "", 0.6, 0.9), ("Boosted trees", "", 1.3, 0.99),
                                 ("LM single-pass", "Qwen3-0.6B", 2000.0, 0.14),
                                 ("LM probe", "Qwen3-0.6B", 3000.0, 0.99),
                                 ("LM generative", "Qwen3-1.7B", 276000.0, None)]:
        for level in ("blackbox", "whitebox"):
            rows.append({"detector": name, "model": model, "access_level": level,
                         "added_ms": ms * (2 if level == "whitebox" else 1), "auroc": auc})
    figures.plot_overhead(pd.DataFrame(rows), 73.0, tmp_path / "overhead")
    assert (tmp_path / "overhead.png").stat().st_size > 10_000
