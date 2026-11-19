"""Figures for the poster and paper (static, light background, PNG + PDF).

Every function reads a table written by scripts/08_evaluate.py and saves one figure.
Colors follow the detector, never its rank, and are the same in every figure:
boosted trees blue, logistic regression orange, LM probe aqua, LM single-pass yellow,
range check / entropy green. LM detectors are drawn dashed, so classical vs. LM never
depends on color alone. The palette passed the colorblind checks of the dataviz method
(validate_palette.js); three colors are below 3:1 contrast, so every series is named in
a legend or a direct label and the numbers are in the CSV tables.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed entity -> color (validated categorical slots 1, 2, 3, 4, 6).
COLORS = {
    "boosted": "#2a78d6",
    "learned": "#eb6834",
    "lm_probe": "#1baf7a",
    "lm_single_pass": "#eda100",
    "rule": "#008300",
}
LABELS = {
    "boosted": "Boosted trees",
    "learned": "Logistic regression",
    "lm_probe": "LM probe (Qwen3-0.6B)",
    "lm_single_pass": "LM single-pass",
    "range": "Range check",
    "quantile": "Quantile range check",
    "confidence": "Entropy check",
}
LEVELS = {"blackbox": "Black-box (top-1 probability, entropy)",
          "whitebox": "White-box (+ per-layer min, max, mean, std)"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": TEXT_SECONDARY,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": TEXT_SECONDARY,
        "ytick.labelcolor": TEXT_SECONDARY, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
    })


def _save(fig, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _entity(detector: str) -> str:
    for key in ("lm_single_pass", "lm_probe", "boosted", "learned"):
        if detector.startswith(key):
            return key
    return "rule"


def plot_roc_curves(scored_runs: pd.DataFrame, comparison: pd.DataFrame, out_path: str) -> None:
    """ROC per detector (E1) on the LM eval subset, one panel per access level.

    `scored_runs` is lm_comparison_scores.csv (per-run scores, SDC weights that restore the
    full split's mix), `comparison` is lm_comparison.csv (for the AUROC in each label).
    Trained classical detectors are the versions trained on the same fit subset as the probe.
    The false-positive axis is logarithmic: every good detector is near the top-left corner,
    and the differences are at low false-alarm rates.
    """
    _style()
    auc = comparison.set_index("detector")["auroc_weighted"]
    panels = {
        "blackbox": ["boosted_blackbox_subset", "learned_blackbox_subset", "lm_probe_blackbox_zero_shot",
                     "lm_single_pass_blackbox_zero_shot", "confidence_blackbox"],
        "whitebox": ["boosted_whitebox_subset", "learned_whitebox_subset", "lm_probe_whitebox_zero_shot",
                     "lm_single_pass_whitebox_zero_shot", "range_whitebox"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), sharey=True)
    for ax, (level, names) in zip(axes, panels.items()):
        for name in names:
            d = scored_runs[scored_runs["detector"] == name]
            scores = np.nan_to_num(d["score"].to_numpy(dtype=float), posinf=np.finfo(float).max)
            fpr, tpr, _ = roc_curve(d["is_sdc"].astype(bool), scores,
                                    sample_weight=np.where(d["is_sdc"], d["weight"], 1.0))
            entity = _entity(name)
            base = name.split("_")[0] if entity == "rule" else entity
            label = f"{LABELS[base]}  ({auc[name]:.3f})"
            ax.step(np.clip(fpr, 1e-3, 1), tpr, where="post", color=COLORS[entity], label=label,
                    linestyle="--" if entity.startswith("lm") else "-", zorder=3)
        chance = np.logspace(-3, 0, 200)
        ax.plot(chance, chance, color=MUTED, linewidth=1, linestyle=":", zorder=1)
        ax.text(0.3, 0.22, "chance", color=MUTED, fontsize=8, rotation=0)
        ax.axvline(0.05, color=AXIS, linewidth=1, zorder=1)
        ax.text(0.054, 0.5, "5% false alarms", color=MUTED, fontsize=8, rotation=90, va="center")
        ax.set_xscale("log")
        ax.set_xlim(4e-3, 1)
        ax.set_xticks([0.005, 0.01, 0.05, 0.1, 0.5, 1], ["0.5%", "1%", "5%", "10%", "50%", "100%"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.02)
        ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
        ax.set_title(LEVELS[level])
        ax.set_xlabel("False-alarm rate on masked runs (log scale)")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.17), title="Detector (AUROC)",
                  title_fontsize=9, alignment="left", ncol=1)
    axes[0].set_ylabel("SDCs caught (true-positive rate)")
    fig.suptitle("Detecting SDCs on the 400-run LM evaluation subset (reweighted to the full split)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def plot_auroc_summary(comparison: pd.DataFrame, out_path: str) -> None:
    """AUROC with 95% interval (filled) and AUROC on the 31 in-range SDCs (hollow), per detector."""
    _style()
    rows = [
        ("whitebox", "boosted_whitebox_subset"), ("whitebox", "learned_whitebox_subset"),
        ("whitebox", "lm_probe_whitebox_zero_shot"), ("whitebox", "lm_probe_whitebox_reference"),
        ("whitebox", "range_whitebox"), ("whitebox", "lm_single_pass_whitebox_zero_shot"),
        ("whitebox", "lm_single_pass_whitebox_reference"),
        ("blackbox", "boosted_blackbox_subset"), ("blackbox", "learned_blackbox_subset"),
        ("blackbox", "lm_probe_blackbox_zero_shot"), ("blackbox", "lm_probe_blackbox_reference"),
        ("blackbox", "confidence_blackbox"), ("blackbox", "lm_single_pass_blackbox_zero_shot"),
        ("blackbox", "lm_single_pass_blackbox_reference"),
    ]
    table = comparison.set_index("detector")
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [7, 7], "hspace": 0.35})
    for ax, level in zip(axes, ("whitebox", "blackbox")):
        names = [name for lvl, name in rows if lvl == level]
        y = np.arange(len(names))[::-1]
        for yi, name in zip(y, names):
            r = table.loc[name]
            entity = _entity(name)
            color = COLORS[entity]
            ax.plot([r["auroc_low"], r["auroc_high"]], [yi, yi], color=color, linewidth=2, zorder=2)
            ax.plot(r["auroc_in_range"], yi, "o", markerfacecolor="none", markeredgecolor=color,
                    markersize=11, markeredgewidth=2, zorder=3)
            ax.plot(r["auroc_weighted"], yi, "o", color=color, markersize=6, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1)
        ax.set_yticks(y, [_row_label(n) for n in names])
        ax.set_xlim(0, 1.025)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=":", zorder=1)
        ax.grid(axis="y", visible=False)
        ax.set_title(LEVELS[level])
    axes[1].set_xlabel("AUROC (0.5 = chance)")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="-", color=TEXT_SECONDARY, markersize=6,
                   markeredgecolor=SURFACE, label="All SDCs, with 95% interval"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor=TEXT_SECONDARY, markeredgewidth=2, markersize=11,
                   label="The 31 SDCs inside the normal ranges"),
    ]
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.12), ncol=2)
    fig.suptitle("Classical vs. LM detectors on the same runs", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", y=1.02)
    _save(fig, out_path)


def _row_label(detector: str) -> str:
    entity = _entity(detector)
    if entity == "rule":
        return LABELS[detector.split("_")[0]]
    label = LABELS[entity].replace(" (Qwen3-0.6B)", "")
    if entity.startswith("lm"):
        label += ", " + ("reference" if detector.endswith("reference") else "zero-shot")
    return label


def plot_data_size(data_size: pd.DataFrame, out_path: str) -> None:
    """AUROC vs. faults per image used for training (the 1/2/4/8 experiment)."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    series = [("boosted", "whitebox"), ("learned", "whitebox"), ("boosted", "blackbox"),
              ("learned", "blackbox")]
    for entity, level in series:
        d = data_size[data_size["detector"] == f"{entity}_{level}"].sort_values("faults_per_image")
        x = d["faults_per_image"].to_numpy()
        style = "-" if level == "whitebox" else (0, (1, 2))
        ax.fill_between(x, d["auroc_low"], d["auroc_high"], color=COLORS[entity], alpha=0.12,
                        linewidth=0, zorder=1)
        ax.plot(x, d["auroc"], linestyle=style, marker="o", markersize=6, color=COLORS[entity],
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                label=f"{LABELS[entity]}, {level.replace('box', '-box')}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 10], ["1", "2", "4", "8", "10"])
    ax.minorticks_off()
    ax.set_xlim(0.85, 11.5)
    ax.set_ylim(0.9, 1.002)
    ax.legend(loc="lower right", ncol=2)
    ax.set_xlabel("Faults per image used for training (fit split)")
    ax.set_ylabel("AUROC on the full eval split")
    ax.set_title("More fault data barely helps: detectors level off at 1-2 faults per image")
    fig.text(0.01, -0.02, "Shaded: 95% interval (bootstrap over eval images). Solid: white-box, "
             "dotted: black-box. 1 fault per image = 76 training SDCs; 10 = 771.",
             color=TEXT_SECONDARY, fontsize=8)
    _save(fig, out_path)


def plot_strata(stratified_tables: dict[str, pd.DataFrame], out_path: str) -> None:
    """Detection quality by fault site and bit group (E3), white-box detectors.

    Mantissa flips caused no SDCs, so that group has no AUROC and is left out (noted).
    """
    _style()
    detectors = [("boosted_whitebox", "boosted"), ("learned_whitebox", "learned"),
                 ("range_whitebox", "rule")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, by in zip(axes, ("fault_site", "bit_group")):
        table = stratified_tables[by]
        table = table[table["n_sdc"] > 0]
        groups = sorted(table[by].unique())
        offsets = np.linspace(-0.2, 0.2, len(detectors))
        for (name, entity), dx in zip(detectors, offsets):
            d = table[table["detector"] == name].set_index(by).loc[groups]
            x = np.arange(len(groups)) + dx
            ax.vlines(x, d["auroc_low"], d["auroc_high"], color=COLORS[entity], linewidth=2, zorder=2)
            ax.plot(x, d["auroc"], "o", color=COLORS[entity], markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=3,
                    label=LABELS["range" if entity == "rule" else entity])
        n_sdc = table.drop_duplicates(by).set_index(by).loc[groups, "n_sdc"]
        ax.set_xticks(range(len(groups)), [f"{g}\n({n} SDCs)" for g, n in zip(groups, n_sdc)])
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.set_title({"fault_site": "By fault site", "bit_group": "By flipped bit"}[by])
    axes[0].set_ylabel("AUROC, white-box (95% interval)")
    axes[0].set_ylim(0.7, 1.005)
    axes[0].legend(loc="lower left")
    fig.text(0.01, -0.04, "Mantissa flips caused no SDCs, so they have no AUROC. Full eval split: "
             "767 SDCs vs. 54,269 masked runs.", color=TEXT_SECONDARY, fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def plot_layers(layer_table: pd.DataFrame, layer_order: list[str], out_path: str) -> None:
    """AUROC per faulted layer, in network order, for the white-box detectors."""
    _style()
    detectors = [("boosted_whitebox", "boosted"), ("learned_whitebox", "learned"),
                 ("range_whitebox", "rule")]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(layer_order))
    for name, entity in detectors:
        d = layer_table[layer_table["detector"] == name].set_index("layer").reindex(layer_order)
        ax.plot(x, d["auroc"], "o-", color=COLORS[entity], markersize=6, linewidth=1.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=LABELS["range" if entity == "rule" else entity], zorder=3)
    n_sdc = layer_table.drop_duplicates("layer").set_index("layer").reindex(layer_order)["n_sdc"]
    ax.set_xticks(x, [f"{name} ({n})" for name, n in zip(layer_order, n_sdc)], rotation=60,
                  ha="right", fontsize=8)
    ax.set_ylim(0.75, 1.005)
    ax.set_ylabel("AUROC, white-box")
    ax.set_title("By faulted layer (SDCs per layer in brackets)")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    _save(fig, out_path)


def plot_overhead(overhead_table, out_path: str) -> None:
    """Latency / memory / energy per detector on the Pi (E2). TODO: implement after 09_overhead."""
    raise NotImplementedError
