"""Audit data provenance and summarize paired physical standing results."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bench import HERE, cases, eligible, improvement, live_push, stand
from collect import read
from learn import arrays, load


def collections(root):
    audit = {}
    for group in ("dev", "train", "on"):
        source = root / group
        result = read(source / "RESULT.json")
        reports = result["evaluation"]
        pushes = {p.name: p for p in cases()[group]}
        admission = None if group == "dev" else read(source / "DATASET.json")
        enrolled = []
        rows = 0
        active_rows = 0
        after_force_response = []
        after_force_end = []
        for name, reports in reports.items():
            push = pushes[name]
            ref = arrays(source / reports["old"]["file"], reports["old"]["trace_sha256"])
            for arm, report in reports.items():
                z = arrays(source / report["file"], report["trace_sha256"])
                for key in ("initial_state", "wrench", "motor_injection"):
                    np.testing.assert_array_equal(z[key], ref[key])
                applied = z["action"][np.arange(len(z["applied"])) // 5].copy()
                applied[:30] += z["motor_injection"]
                np.testing.assert_array_equal(applied, z["applied"])
                remeasured = live_push.measure(z, push.end, z["ref_target"], z["ref_joints"])
                if any(remeasured[k] != report[k] for k in remeasured):
                    raise RuntimeError("Demonstration metrics changed")
                if arm != "teacher":
                    continue
                recorded = admission[name]
                assert recorded["eligible"] == eligible(report, reports["old"])
                assert recorded["improves"] == improvement(report, reports["old"])
                switch = int(z["teacher_start_row"])
                if group == "on":
                    np.testing.assert_array_equal(z["action"][:switch], ref["action"][:switch])
                    np.testing.assert_array_equal(z["x"][:switch+1], ref["x"][:switch+1])
                    np.testing.assert_array_equal(z["com"][:switch*5+1], ref["com"][:switch*5+1])
                if recorded["eligible"]:
                    enrolled.append(name)
                    rows += len(z["x"]) - switch
                    decision_ms = np.arange(switch, len(z["x"])) * 5
                    active_rows += int(np.sum((decision_ms > push.onset_ms) & (decision_ms < push.end + 200)))
                    if switch * 5 > push.onset_ms:
                        after_force_response.append(name)
                    if switch * 5 >= push.end:
                        after_force_end.append(name)
        audit[group] = dict(attempted=len(result["evaluation"]), admitted=enrolled, rows=rows,
            after_force_active_rows=active_rows, switch_after_force_onset=after_force_response,
            switch_at_or_after_force_end=after_force_end,
            improvements=sum(r["improves"] for r in admission.values()) if admission is not None else None,
            applied_force_w=[float(np.linalg.norm(pushes[key].force_w)) for key in enrolled],
            discarded_force_w=[float(np.linalg.norm(p.force_w)) for key, p in pushes.items() if key not in enrolled] if admission is not None else [],
            native_steps=result["settle_native_steps"] + result["total_trial_native_steps"])
    return audit


def summary(result, keys):
    stats = {}
    for arm in next(iter(result.values())):
        samples = [result[k][arm] for k in keys]
        stats[arm] = dict(
            safe=sum(s["safe"] for s in samples),
            recovered=sum(s["recovered"] for s in samples),
            improvements_vs_old=sum(improvement(result[k][arm], result[k]["old"]) for k in keys),
            mean_peak_drift=float(np.mean([s["peak_drift_body_lengths"] for s in samples])),
            mean_end_drift=float(np.mean([s["end_drift_body_lengths"] for s in samples])),
        )
    common = [k for k in keys if all(s["recovered"] for s in result[k].values())]
    for arm, aggregate in stats.items():
        aggregate["mean_recovery_ms_common"] = float(np.mean([result[k][arm]["recovery_ms"] for k in common])) if common else None
        aggregate["mean_end_drift_common"] = float(np.mean([result[k][arm]["end_drift_body_lengths"] for k in common])) if common else None
    return dict(cases=keys, common_recovered=common, summary=stats)


@torch.no_grad()
def label_errors(root):
    records = read(root / "learn/DATA_1000.json")
    models = {"old": live_push.load(),
              "stage_one": load(root / "learn/step_0600.pt"),
              "final": load(root / "learn/step_1000.pt")}
    errors = {}
    for cohort, entries in records.items():
        errors[cohort] = {}
        for arm, model in models.items():
            squared = []
            for entry in entries:
                z = arrays(Path(entry["path"]), entry["sha256"])
                start = entry["teacher_start"]
                x, y = z["x"][start:start+50], z["action"][start:start+50]
                out = model(torch.from_numpy(x)).numpy()
                squared.append(np.square(out - y).mean(1))
                if cohort == "on" and arm == "stage_one":
                    prefix = model(torch.from_numpy(z["x"][:start])).numpy()
                    np.testing.assert_allclose(prefix, z["action"][:start], atol=1e-12, rtol=0.)
            errors[cohort][arm] = dict(first=float(np.mean([row[0] for row in squared])),
                                       first_250ms=float(np.concatenate(squared).mean()))
    return errors


def plot(dest, result, keys, output):
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    colors = dict(zero="#888888", old="#206c87", stage_one="#9b6d37", final="#a22f52")
    names = dict(zero="原有限 PD", old="旧站立网络", stage_one="扩充示范", final="再训练偏离状态")
    x = np.arange(len(keys))
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), layout="constrained")
    fig.suptitle("未见胸部机械推力下的站立恢复", fontsize=17, x=.52)
    for index, arm in enumerate(colors):
        y = [result[key][arm]["recovery_ms"] if result[key][arm]["recovered"] else None
             for key in keys]
        d = [result[key][arm]["end_drift_body_lengths"] for key in keys]
        axes[0, 0].plot(x, [np.nan if k is None else k for k in y], ".-", color=colors[arm], label=names[arm])
        missing = [i for i, value in enumerate(y) if value is None]
        axes[0, 0].scatter(np.asarray(missing) + .07 * (index - 1.5), [1120] * len(missing),
                           marker="x", color=colors[arm], s=20)
        axes[0, 1].plot(x, d, ".-", color=colors[arm])
    axes[0, 0].set(ylabel="恢复时间 (ms)", ylim=(0, 1180), xticks=x, xticklabels=[k[-2:] for k in keys])
    axes[0, 0].text(.99, .86, "× 未全程通过", transform=axes[0, 0].transAxes, ha="right", va="top")
    axes[0, 0].legend(loc="center left", fontsize=8, frameon=False)
    axes[0, 1].set(ylabel="末端漂移 (体长)", xticks=x, xticklabels=[k[-2:] for k in keys])
    common = [key for key in keys if all(s["recovered"] for s in result[key].values())]
    sooner = min(common, key=lambda k: result[k]["final"]["recovery_ms"] - result[k]["old"]["recovery_ms"]) if common else keys[0]
    harder = max((k for k in keys if k != sooner), key=lambda k: result[k]["old"]["end_drift_body_lengths"])
    pushes = {p.name: p for p in cases()["test"]}
    for ax, key in zip(axes[1], (sooner, harder)):
        p = pushes[key]
        for arm, color in colors.items():
            entry = result[key][arm]
            z = arrays(dest / entry["file"], entry["trace_sha256"])
            drift = np.linalg.norm(z["com"][:, :2] - z["com"][0, :2], axis=1) / stand.LENGTH
            ax.plot(np.arange(len(drift)), drift, color=color, linewidth=1.3)
        ax.axvspan(p.onset_ms, p.end, color="black", alpha=.10, lw=0)
        ax.set(title=f"扰动 {key[-2:]}   {np.linalg.norm(p.force_w):.2f} 倍体重 × {p.duration_ms} ms",
               xlabel="试验时间 (ms)", ylabel="水平位移 (体长)")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dedede", lw=.7)
    fig.savefig(output, dpi=165)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "RUN")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.run.resolve()
    data = collections(root)
    for cohort in ("train", "on"):
        registered = read(root / "learn" / f"DATA_{600 if cohort == 'train' else 1000:04}.json")[cohort]
        assert len(registered) == len(data[cohort]["admitted"])
        assert sum(row["rows"] for row in registered) == data[cohort]["rows"]
    source = root / "test"
    record = read(source / "RESULT.json")
    result = record["evaluation"]
    pushes = sorted(k for k in result if k.startswith("test_"))
    metrics = summary(result, pushes)
    replay = read(source / "REPLAY.json")
    assert replay["all_verified"]
    counts = sum(row["native_steps"] for row in data.values()) + record["settle_native_steps"] + record["trial_native_steps"] + replay["physical_native_steps"]
    stand.put(root / "AUDIT.json", dict(all_verified=True, collections=data, holdout=metrics,
               training_label_error=label_errors(root),
               native_steps=counts, regression={k: result[k] for k in ("quiet", "leg_pair", "mixed")}))
    plot(source, result, pushes, root / "HOLDOUT.png")
    print("audit", counts, metrics, flush=True)


if __name__ == "__main__":
    main()
