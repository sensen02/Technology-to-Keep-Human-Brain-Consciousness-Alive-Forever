"""Report paired effects on the prospectively fixed standing disturbances."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from bounded import HERE, tests
from bench import improvement, stand
from collect import read


def summarize(rows, cases):
    common = [key for key in cases if all(v["recovered"] for v in rows[key].values())]
    stats = {}
    for arm in next(iter(rows.values())):
        values = [rows[key][arm] for key in cases]
        paired = [rows[key][arm] for key in common]
        stats[arm] = dict(safe=sum(a["safe"] for a in values), recovered=sum(a["recovered"] for a in values),
            mean_recovery_ms=float(np.mean([v["recovery_ms"] for v in paired])) if common else None,
            mean_peak_body_lengths=float(np.mean([v["peak_drift_body_lengths"] for v in paired])) if common else None,
            mean_end_body_lengths=float(np.mean([v["end_drift_body_lengths"] for v in paired])) if common else None,
            improves_vs_restoring=sum(improvement(rows[key][arm], rows[key]["restoring"]) for key in cases))
    delta = {k: rows[k]["model"]["recovery_ms"] - rows[k]["restoring"]["recovery_ms"] for k in common}
    return dict(summary=stats, common=common, recovery_delta_ms=delta)


def plot(rows, cases, report, output):
    arms = {"old": ("旧站立网络", "#808080"), "restoring": ("仅站姿回正", "#205f7a"),
            "model": ("回正＋学到的修正", "#b23749")}
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.8), layout="constrained")
    fig.suptitle("学到的抗扰修正对站立恢复的影响", fontsize=17)
    for i, (arm, (label, color)) in enumerate(arms.items()):
        x = np.arange(len(cases)) + (i - 1)*.17
        valid = [rows[key][arm]["recovered"] for key in cases]
        motion = [rows[key][arm]["recovery_ms"] if ok else np.nan for key, ok in zip(cases, valid)]
        ax[0].scatter(x, motion, label=label, color=color, s=26)
        not_ok = ~np.asarray(valid)
        ax[0].scatter(x[not_ok], np.full(not_ok.sum(), 550), marker="x", color=color, s=24)
    ax[0].set(ylabel="恢复时间 (ms)", ylim=(0, 585), xticks=range(len(cases)), xticklabels=cases)
    ax[0].text(.99,.92,"× 未全程通过", transform=ax[0].transAxes, ha="right")
    ax[0].legend(loc="lower left", bbox_to_anchor=(0., 1.01), frameon=False, ncol=3, fontsize=9)
    d = report["recovery_delta_ms"]
    idx = [n for n, k in enumerate(cases) if k in d]
    vals = list(d.values())
    ax[1].bar(idx, vals, color=["#205f7a" if v < 0 else "#b23749" for v in vals], width=.55)
    ax[1].set(ylabel="加学习分支后\n恢复时间变化 (ms)",
              xlabel="独立扰动编号  ·  负值代表恢复更快", xticks=range(len(cases)),
              xticklabels=cases, xlim=(-.5,len(cases)-.5))
    ax[1].axhline(0., color="#444444", lw=.8)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%+g"))
    for panel in ax:
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="y", color="#dedede", lw=.55)
        panel.set_axisbelow(True)
    fig.savefig(output, dpi=165)


def main():
    root = HERE / "RUN"
    test = read(root / "TEST/RESULT.json")
    replay = read(root / "TEST/REPLAY.json")
    assert replay["verified"]
    rows = test["cases"]
    cases = [p.name for p in tests() if p.name.startswith("n")]
    report = summarize(rows, cases)
    stand.put(root / "SUMMARY.json", dict(**report, safety_regression={k: rows[k] for k in ("quiet", "leg_pair", "mixed")},
        steps=test["native_steps"] + replay["native_steps"], independent_replay=replay, sha256=read(root / "TRAIN.json")["sha256"]))
    plot(rows, cases, report, root / "HOLDOUT.png")
    print(report)


if __name__ == "__main__":
    main()
