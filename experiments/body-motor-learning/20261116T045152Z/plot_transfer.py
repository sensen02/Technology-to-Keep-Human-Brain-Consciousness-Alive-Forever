"""Plot aligned source-derived swing targets and real body footpad traces."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "20261116T000536Z"))
from walking_bench import LEGS, read, stand


def heights(run):
    trace = run / "TRACE.npz"
    assert stand.sha(trace) == read(run / "RESULT.json")["trace_sha256"]
    with np.load(trace, allow_pickle=False) as z:
        return z["padz"].copy()


def summarise(reference, teacher, student, output):
    output.mkdir()
    meta = read(reference)
    baseline = Path(meta["receiver_baseline_dir"])
    with np.load(reference.with_suffix(".npz"), allow_pickle=False) as z:
        residual = z["correction_native"]
        assert stand.sha(reference.with_suffix(".npz")) == meta["ref_sha256"]
    trajectories = {"teacher": heights(teacher)}
    if student is not None:
        trajectories["student"] = heights(student)
    base = heights(baseline)
    strength = defaultdict(list)
    for event in meta["receiver_windows"]:
        a, b, leg = event["from_ms"], event["to_ms"], event["leg"]
        idx = LEGS.index(leg)
        strength[leg].append((np.square(residual[a:b+1, idx]).sum(), a, b, idx))
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(),
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.7), sharex=True)
    quant = []
    for ax, leg in zip(axs.flat, meta["source_legs"], strict=True):
        _, start, end, idx = max(strength[leg])
        selected = slice(start, end+1)
        dx = residual[selected, idx]
        phase = np.linspace(0, 1, len(dx))
        ax.axhline(0, color="#6c7580", linewidth=1, label="冻结原型")
        ax.plot(phase, dx, color="#323048", linewidth=1.9, linestyle=(0, (3, 2)), label="全量源形状目标")
        item = dict(leg=leg, from_ms=start, to_ms=end, baseline_rms=float(np.sqrt(np.mean(dx**2))))
        for key, arr in trajectories.items():
            if end >= len(arr):
                continue
            shift = arr[selected, idx] - base[selected, idx]
            shift -= np.linspace(shift[0], shift[-1], len(shift))
            color, label = ("#037c70", "同身教师") if key == "teacher" else ("#bd571b", "复训身体")
            ax.plot(phase, shift, color=color, linewidth=1.8, label=label)
            item[key] = dict(relative_shape_projection=float(shift @ dx / (dx @ dx)),
                             source_rms_error=float(np.sqrt(np.mean((dx - shift)**2))),
                             shape_rms=float(np.sqrt(np.mean(shift**2))))
        quant.append(item)
        ax.set_title(f"{leg}  |  {start/1000:.3f}–{end/1000:.3f} 秒", loc="left", pad=8)
        ax.grid(axis="y", color="#e4e4e4", linewidth=.7)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.))
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.suptitle("同一身体上的 FlyGym 摆足形状迁移", x=.10, y=.98, ha="left", fontsize=17)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.08, .933), ncol=len(labels), frameon=False)
    fig.supxlabel("预先选定摆动窗内的进度", y=.027)
    fig.supylabel("相对冻结原型的足垫高度变化 · 原生长度", x=.027, fontsize=10)
    fig.subplots_adjust(left=.11, right=.97, bottom=.11, top=.81, hspace=.35, wspace=.28)
    fig.savefig(output / "足垫高度配对.png", dpi=170, facecolor="white")
    plt.close(fig)
    stand.put(output / "FIGURE.json", dict(interval_selection="largest source-correction energy per leg; chosen before viewing student",
        units="native height relative to baseline with endpoint chord removed", comparison=quant,
        reference=str(reference), baseline=str(baseline), teacher=str(teacher), student=str(student) if student else None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--student", type=Path)
    args = parser.parse_args()
    summarise(args.reference.resolve(), args.teacher.resolve(), args.student.resolve() if args.student else None,
              args.output.resolve())
