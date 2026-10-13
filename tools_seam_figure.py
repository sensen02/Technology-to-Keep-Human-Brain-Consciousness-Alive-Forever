#!/usr/bin/env python3
"""Seam diagram: the four information flows of the closed loop, and which ones exist.

Reads `outputs/interfaces.json` (written by tools_interface_doc.py) so the picture cannot disagree
with the interface document.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_seam_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
    if Path(_p).is_file():
        font_manager.fontManager.addfont(_p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_p).get_name()
        break

state = json.loads((HERE / "outputs" / "interfaces.json").read_text())
seams = state["seams"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(19.0, 8.4),
                              gridspec_kw={"width_ratios": [1.25, 1.0]})
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# ---- the loop --------------------------------------------------------------------------------
BOX = {"brain": (5.0, 8.2, "脑：12,000 BANC 神经元\n(engine/neural_cond.py, isolation_scenario)"),
       "body": (1.6, 5.0, "身体物理引擎\nFlyGym / MuJoCo\n(engine/embodied/body_backend.py)"),
       "elec": (8.4, 5.0, "电极\n7 µm 直径 / 20 µm 间距\n(engine/electrode*.py)")}
COL = {"IMPLEMENTED": "#2e7d32", "DECLARED_NOT_IMPLEMENTED": "#c0392b",
       "UNVERIFIABLE": "#e08a00"}


def box(ax, x, y, text, w=2.6, h=1.35, c="#eeeeee"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.12",
                                fc=c, ec="#333", lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=9)


for k, (x, y, t) in BOX.items():
    box(ax, x, y, t, c="#e8f0e8" if k != "elec" else "#fdecea")

# arrows: (name, from, to, label position)
ARROWS = [("SensoryDrive", "body", "brain", (3.0, 6.0)),
          ("MotorCommand", "brain", "body", (4.2, 7.0)),
          ("NeuralRecorder", "brain", "elec", (7.0, 7.0)),
          ("NeuralStimulus", "elec", "brain", (6.3, 6.0))]
POS = {"body": (1.6, 5.0), "brain": (5.0, 8.2), "elec": (8.4, 5.0)}
for name, a, b, xy in ARROWS:
    st = seams[name]["status"]
    col = COL.get(st, "#888")
    ls = "-" if st == "IMPLEMENTED" else "--"
    ax.add_patch(FancyArrowPatch(POS[a], POS[b], arrowstyle="-|>", mutation_scale=22,
                                 lw=3.0 if st == "IMPLEMENTED" else 2.0, color=col, ls=ls,
                                 shrinkA=42, shrinkB=42, connectionstyle="arc3,rad=0.16"))
    ax.text(xy[0], xy[1], f"{name}\n{st}", ha="center", va="center", fontsize=8.6, color=col,
            fontweight="bold" if st == "IMPLEMENTED" else "normal",
            bbox=dict(fc="white", ec=col, lw=0.9, alpha=0.92, pad=2.2))

ax.text(5.0, 1.15, "实线 = 已实现　虚线 = 仅有接口声明（信息流不存在）",
        ha="center", fontsize=9.5, color="#333")
ax.set_title("闭环的四条接缝：哪条真的通了\n"
             "本平台此前只有一条半；本轮补上「脑→电极」", fontsize=12)

# ---- the module index ------------------------------------------------------------------------
ax2.axis("off")
mods = state["modules"]
# The list body is ASCII-only: a monospace family has no CJK glyphs, and the first version of this
# figure rendered its own legend as empty boxes.  The legend therefore lives in the title, which
# uses the CJK font.
lines = [f"{len(mods)} modules by size", ""]
for m in mods[:22]:
    if "error" in m:
        continue
    flags = ("U" if m["declares_units"] else " ") + ("I" if m["declares_illustrative"] else " ") + \
            ("N" if m["declares_not_established"] else " ")
    lines.append(f"{flags}  {m['lines']:>5}  {m['path']}")
ax2.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=7.6, family="monospace")
ax2.set_title("接口文档覆盖的模块　U=声明单位　I=声明 ILLUSTRATIVE　N=声明未确立\n"
              "（完整签名见 INTERFACES.zh-CN.md）", fontsize=10.5)

fig.suptitle("INTERFACES：闭环接缝 + 模块索引　（由 tools_interface_doc.py 从 AST 生成，"
             "并用 --stale 保证文档与代码一致）", fontsize=12)
dest = HERE / "outputs" / "seam_figure.png"
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(dest, dpi=120)
print(f"wrote {dest}")
for k, v in seams.items():
    print(f"  {k:<16}{v['status']:<28}{v['direction']}")
