"""Phase 5B-3 figures: reliability diagrams and the fixed-K scenario figure.

Reliability diagrams use the frozen decade bins of predicted probability
(Addendum C8) on log-scaled axes. They show model-generated development results.

The scenario figure combines model-generated fraud counts with **assumed**
review minutes. Its title carries the hypothetical label, and its axes say which
quantity is model-generated and which is a simulated outcome of an assumed input.

Figures are written as PNG with the non-interactive ``Agg`` backend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluation.evidence_labels import HYPOTHETICAL_FIGURE_LABEL  # noqa: E402
from src.models.phase5_settings import RELIABILITY_BIN_EDGES  # noqa: E402

_FLOOR = 1e-7  # plotting floor for the log axes; bins below it are drawn at the floor


def reliability_figure(table: list[dict], title: str, subtitle: str, path: Path | None = None):
    """Log-log reliability diagram from a decade reliability table. Returns the Matplotlib figure."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.plot([_FLOOR, 1], [_FLOOR, 1], color="0.5", linestyle="--", linewidth=1, label="perfect calibration")
    for edge in RELIABILITY_BIN_EDGES[1:-1]:
        ax.axvline(edge, color="0.88", linewidth=0.8, zorder=0)
    drawn = {"interpretable": False, "sparse": False, "zero": False}
    for row in table:
        if row["n_rows"] == 0:
            continue
        x = max(row["mean_predicted_probability"], _FLOOR)
        lower, upper = row["clopper_pearson_95"]
        note = f"n={row['n_rows']:,}\nO={row['observed_frauds']}, E={row['expected_frauds']:.1f}"
        if row["observed_frauds"] == 0:
            ax.plot([x, x], [_FLOOR, max(upper, _FLOOR)], color="tab:gray", linewidth=1.2)
            ax.scatter([x], [max(upper, _FLOOR)], marker="v", color="tab:gray", zorder=3,
                       label=None if drawn["zero"] else "no observed fraud: 95% upper bound")
            drawn["zero"] = True
            ax.annotate(note, (x, max(upper, _FLOOR)), textcoords="offset points", xytext=(6, 6), fontsize=7)
            continue
        y = row["observed_rate"]
        kind = "interpretable" if row["interpretable"] else "sparse"
        style = ({"color": "tab:blue", "markerfacecolor": "tab:blue"} if row["interpretable"]
                 else {"color": "tab:orange", "markerfacecolor": "white"})
        label = None if drawn[kind] else (
            "bin with >= 5 observed and >= 5 expected frauds" if row["interpretable"]
            else "not interpretable (< 5 observed or < 5 expected)")
        drawn[kind] = True
        ax.errorbar([x], [y], yerr=[[y - max(lower, _FLOOR)], [upper - y]], fmt="o", capsize=3,
                    label=label, zorder=3, **style)
        ax.annotate(note, (x, y), textcoords="offset points", xytext=(6, -16), fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(_FLOOR, 1.5)
    ax.set_ylim(_FLOOR, 1.5)
    ax.set_xlabel("Mean calibrated probability in bin (frozen decade bins; log scale)")
    ax.set_ylabel("Observed fraud rate in bin, Clopper-Pearson 95% (log scale)")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, metadata={"Software": None})
    return fig


def fixed_k_scenario_figure(per_model_tp: dict, ks, n_frauds: int, review_minutes, partition_note: str,
                            path: Path | None = None):
    """Frauds caught at equal alert volume, with analyst-hours under assumed review minutes."""
    ks = [int(k) for k in ks]
    fig, (ax_tp, ax_hours) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    width = 0.8 / len(per_model_tp)
    positions = np.arange(len(ks))
    for i, (name, tps) in enumerate(per_model_tp.items()):
        bars = ax_tp.bar(positions + (i - (len(per_model_tp) - 1) / 2) * width, tps, width, label=name)
        ax_tp.bar_label(bars, fontsize=7)
    ax_tp.axhline(n_frauds, color="0.4", linestyle=":", linewidth=1, label=f"all frauds in the period ({n_frauds})")
    ax_tp.set_xticks(positions, [str(k) for k in ks])
    ax_tp.set_xlabel("K = alerts reviewed (equal alert volume for both models)")
    ax_tp.set_ylabel("Frauds caught in top K")
    ax_tp.set_title("MODEL-GENERATED: frauds caught at fixed K", fontsize=9)
    ax_tp.legend(fontsize=7, loc="lower right")
    for minutes in review_minutes:
        ax_hours.plot(ks, [k * minutes / 60.0 for k in ks], marker="o",
                      label=f"ASSUMED {minutes:g} min per review")
    ax_hours.set_xscale("log")
    ax_hours.set_xticks(ks, [str(k) for k in ks])
    ax_hours.set_xlabel("K = alerts reviewed")
    ax_hours.set_ylabel("SIMULATED analyst-hours (identical for both models at equal K)")
    ax_hours.set_title("SIMULATED OUTCOME of ASSUMED inputs: analyst-hours", fontsize=9)
    ax_hours.legend(fontsize=7)
    fig.suptitle(f"{HYPOTHETICAL_FIGURE_LABEL}\n{partition_note}", fontsize=9)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, metadata={"Software": None})
    return fig
