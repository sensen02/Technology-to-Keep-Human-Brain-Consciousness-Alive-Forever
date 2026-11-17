"""Summarize paired standing results and make the native-contact comparison figure."""
from collections import Counter
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from matrix import ARMS, FACTORS, HERE, arrays, read, stand
from aftermath import supplement
from native_contact import DT, W

START = {"nominal":"标称站姿", "lean_left":"侧扰 A", "lean_right":"侧扰 B",
         "moving_forward":"纵扰 A", "moving_backward":"纵扰 B", "uneven_feet":"非对称承重"}
CATEGORIES = ("steady_recovered", "contact_recovered", "fall_recovered", "unrecovered")
SHORT = ("稳", "接触后稳", "倒后稳", "未恢复")
ARM_LABELS = ("仅站姿回正", "站姿回正＋学习修正")
PUSH_LABELS = ("安静", "+x · 0.5 W", "+y · 0.5 W", "+x · 0.8 W", "+y · 0.8 W")


def paired_stats(result, keys):
    both = [result[k] for k in keys if all(r["event"]["recovered"] for r in result[k].values())]
    delta = [r["model"]["event"]["recovery_ms"] - r["restoring"]["event"]["recovery_ms"] for r in both]
    arms = {}
    for arm in ARMS:
        models = [result[k][arm] for k in keys]
        counter = Counter(r["event"]["behavior"] for r in models)
        arms[arm] = dict(recovered=sum(r["event"]["recovered"] for r in models),
            legacy_recovered=sum(r["legacy"]["recovered"] for r in models),
            falls=sum(bool(r["event"]["fall_intervals_steps"]) for r in models),
            events={k:counter[k] for k in CATEGORIES},
            co_recovered_time_ms=mean(r[arm]["event"]["recovery_ms"] for r in both) if both else None)
    return dict(pairs=len(keys), common_recovered=len(both), paired_delta_ms=delta,
                median_delta_ms=median(delta) if delta else None, arms=arms)


def plot_grid(result, output):
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True, layout="constrained")
    fig.suptitle("相同推力下，不同站姿的恢复结果", fontsize=17)
    cmap = ListedColormap(["#d9efe8", "#f9e8b1", "#dddcf1", "#f0c8c9"])
    for arm, title, ax in zip(ARMS, ARM_LABELS, axes):
        events = [[result[f"{key}__{p.name}"][arm]["event"] for p in FACTORS] for key in START]
        colors = [[CATEGORIES.index(x["behavior"]) for x in row] for row in events]
        ax.imshow(colors, cmap=cmap, vmin=-.5, vmax=3.5, aspect="auto")
        for i, row in enumerate(events):
            for j, item in enumerate(row):
                word = "倒地" if item["behavior"] == "unrecovered" and item["fall_intervals_steps"] else SHORT[colors[i][j]]
                recovery = item["recovery_ms"]
                if recovery is not None:
                    word = f"{word}\n{recovery} ms"
                ax.text(j, i, word, ha="center", va="center", fontsize=9)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(len(PUSH_LABELS)), PUSH_LABELS, fontsize=9)
        ax.set_yticks(range(len(START)), list(START.values()), fontsize=10)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        for i in np.arange(.5, len(PUSH_LABELS)-.5):
            ax.axvline(i, color="white", lw=2)
        for i in np.arange(.5, len(START)-.5):
            ax.axhline(i, color="white", lw=2)
    fig.savefig(output, dpi=170)


def plot_legacy(results, source, output):
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 5.9), layout="constrained")
    fig.suptitle("原案例复验的逐体段接触", fontsize=17)
    for i, key in enumerate(("n07", "n02")):
        record = results[f"nominal__{key}"]["model"]
        native = arrays((source / record["file"]).with_suffix(".native.npz"), record["native_sha256"])
        times = np.arange(len(native["foot"])) * DT * 1000
        ax = axes[0, i]
        ax.plot(times, native["nonfoot_normal"]/W, label="非足接触", color="#267080")
        ax.plot(times, native["trunk_normal"]/W, label="胸腹接触", color="#ae3b55")
        ax.set(title=key, ylabel="法向力（倍体重）")
        if i == 0:
            ax.legend(frameon=False)
        ax = axes[1, i]
        ax.plot(times, np.rad2deg(np.arccos(np.clip(native["upright"], -1., 1.))), color="#3d5966")
        ax.set(ylabel="躯干倾角（度）", xlabel="试验时间（ms）")
        for a, b in record["event"]["fall_intervals_steps"]:
            for panel in axes[:, i]:
                panel.axvspan(a*DT*1000, b*DT*1000, alpha=.10, color="#ae3b55")
        for ax in axes[:, i]:
            ax.set_xlim(0, 450 if i == 0 else len(times)*DT*1000)
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
    fig.savefig(output, dpi=160)


def main():
    root = HERE / "RUN"
    dest = root / "MATRIX"
    data = read(dest / "RESULT.json")
    replay = read(dest / "REPLAY.json")
    result = data["cases"]
    bank = read(root / "INITIAL/BANK.json")
    poses = {}
    for key, pair in result.items():
        for arm, record in pair.items():
            path = dest / record["file"]
            ms = arrays(path.with_suffix(".npz"), record["legacy"]["trace_sha256"])
            native = arrays(path.with_suffix(".native.npz"), record["native_sha256"])
            event = supplement(native, record["event"], ms)
            record["event"] = event
            if event["terminal_fallen_pose"] is not None:
                seed = dict(**event["terminal_fallen_pose"], controller=arm, case=key,
                            millisecond_trace=str(path.with_suffix(".npz")),
                            native_trace=str(path.with_suffix(".native.npz")))
                poses[f"{key}__{arm}"] = seed
    factors = {p.name: paired_stats(result, [key for key in result if key.endswith("__"+p.name)])
               for p in FACTORS}
    starts = {name:paired_stats(result, [f"{name}__{p.name}" for p in FACTORS]) for name in bank}
    legacy = {name:{arm:r["event"] for arm,r in result[f"nominal__{name}"].items()} for name in ("n07","n02")}
    total_steps = data["native_steps"] + data["worker_settle_steps"] + replay["native_replay_steps"]
    total_steps += 80000 + sum(v["native_steps"] for v in bank.values())
    stand.put(root / "EVENTS_SUPPLEMENTED.json", dict(
        cases={k:{arm:r["event"] for arm,r in pair.items()} for k,pair in result.items()},
        amendment_sha256=stand.sha(HERE / "EVENT_AMENDMENT.zh-CN.md"), physics_steps=0))
    stand.put(root / "SUMMARY.json", dict(factors=factors, starts=starts, legacy=legacy, replay=replay,
        fallen_pose_index=poses, native_steps=total_steps))
    plot_grid(result, root / "MULTI_STATE.png")
    plot_legacy(result, dest, root / "CONTACT_EVENTS.png")
    for name,row in factors.items():
        print(name, row)


if __name__ == "__main__":
    main()
