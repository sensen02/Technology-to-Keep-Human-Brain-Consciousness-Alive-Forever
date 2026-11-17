"""Scientific top-down plot from the archived physical positions."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np

from approach_runtime import turn


def draw(course, control, output):
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    experiment = turn.read(course / "RESULT.json")
    block = turn.read(course / "OBSTACLE.json")
    center = np.asarray(block["center"][:2])
    half = np.asarray(block["half_extent"][:2])
    fig, ax = plt.subplots(figsize=(8.2,4.2), constrained_layout=True)
    ax.add_patch(Rectangle(center-half, *(2*half), facecolor="#59b56f", alpha=.62,
                           edgecolor="#2a6c38", label="实体障碍"))
    for trial, color, label in ((course,"#2874a8","开启规避"),(control,"#d7892b","关闭规避")):
        with np.load(trial/"TRACE.npz") as t, np.load(trial/"OBS.npz") as obs:
            xy = t["com"][:,:2]
            head = t["head"][:,:2]
            goal = obs["goal"]
        ax.plot(*head.T, lw=1.5, color=color, label=f"{label}头部")
        ax.plot(*xy.T, lw=1., color=color, linestyle="--", alpha=.55, label=f"{label}质心")
        ax.scatter(*head[-1], s=26, color=color, zorder=3)
        if trial == control:
            contact = int(round(turn.read(trial/"RESULT.json")["first_obstacle_force_ms"]))
            ax.scatter(*head[contact], s=38, edgecolors=color, facecolors="none", zorder=3, label="关闭组首次碰撞时的头部中心")
    ax.scatter(*goal[:2], s=36, color="#bc4133", label="红色视觉目标")
    ax.axvline(center[0]+half[0], color="#629e69", alpha=.6, lw=.9, ls=":")
    green = experiment["detour_checks"]
    status = "通过" if experiment["passed"] else "未过全部门"
    ax.set(title=f"远距离单障碍物理轨迹，{status}", xlabel="世界 X（原生长度）", ylabel="世界 Y（原生长度）")
    ax.set_aspect("equal")
    ax.grid(alpha=.18)
    ax.legend(loc="upper right", fontsize=7, frameon=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("course",type=Path)
    parser.add_argument("control",type=Path)
    a=parser.parse_args()
    draw(a.course,a.control,a.out)
