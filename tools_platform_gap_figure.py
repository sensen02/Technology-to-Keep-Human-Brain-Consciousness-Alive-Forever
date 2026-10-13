#!/usr/bin/env python3
"""How far is this from a USABLE EXPERIMENTAL PLATFORM?  A gap scorecard.

HONESTY NOTE, and it is the most important line in this file: the 0-1 scores are MY JUDGEMENT
CALLS, not measurements.  Every measured quantity behind them is written in the annotation column
and is traceable to a file in this project.  A reader who disagrees with a score can still use the
measurement next to it.

The rows are the capabilities an experiment would actually need.  Each annotation names the
SINGLE thing that blocks that row, so the gap is actionable rather than a number.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_platform_gap_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The annotations are Chinese, and matplotlib's bundled DejaVu has no CJK glyphs, which silently
# renders as empty boxes.  Register a CJK font if the system has one; fall back loudly otherwise.
from matplotlib import font_manager

_CJK = None
for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
    if Path(_p).is_file():
        font_manager.fontManager.addfont(_p)
        _CJK = font_manager.FontProperties(fname=_p).get_name()
        break
if _CJK:
    plt.rcParams["font.family"] = _CJK
    print(f"CJK font: {_CJK}")
else:
    raise SystemExit("no CJK font found: the annotations would render as boxes")

HERE = Path(__file__).resolve().parent

# (capability, readiness 0-1, measured evidence, the one blocking thing)
ROWS = [
    ("真实果蝇验证", 0.00,
     "本项目从未对真实果蝇做过任何测量；全部“真值”来自模拟器",
     "一次真实记录都没有 —— 0 %"),
    ("相机 → 力（A→B 接通）", 0.05,
     "观测链与触觉链互不相通；面积需要 ~1 µm 穿透分辨率",
     "差 6–880×（二层近相机可到 6–14×）"),
    ("行为底座：站立", 0.00,
     "FlyMimic 肌肉骨骼模型 stood=False 5/5 配置；55–68 % 体重压在胸/腹/喙",
     "被动骨架无稳定平衡点，神经闭环也没救回来"),
    ("接触力物理性", 0.15,
     "单位错已修（1e-3→1e-6，1000×）；残留每帧六腿合力 均值 4.3–12.2×、峰值 25.3–27.8× 体重",
     "残留量级未解释（已排除致动器）"),
    ("神经 → 行为因果", 0.25,
     "BANC 12,000 神经元确实改变行为：−27 % 速度、−21° 偏航（3 seed 配对）",
     "只经 2 个旋钮；步态来自动作捕捉回放 + 6 个非脉冲振子（零神经元）"),
    ("行为底座：行走", 0.30,
     "能移动 1.2–4.1 mm、5–27 mm/s（量级对）；支撑足端中位滑移 0.7–1.0 mm/s",
     "4.9–5.1/6 条腿同时着地、从不腾空 ⇒ 非三脚步态；躯干起伏 0.35–0.40 mm = 16 % 体长"),
    ("神经元 → 肌肉", 0.40,
     "唯一真链：90 个 Hill 型肌肉，消融逐腿特异 −60 %/−41 %/−16 %，其他五腿仅变 1–3 %",
     "该模型站不住也走不了"),
    ("触觉 / 接触面积", 0.45,
     "A~穿透 指数 0.869–0.917、r² 0.979–0.989（几何路线）；力路线已被证否",
     "取自模拟器而非图像；力残留未解 ⇒ 只能作相对量"),
    ("相机 → 3D 肢体运动学", 0.60,
     "8/12 根骨 ≥10 帧；股骨中位 1.67–3.69°、胫骨 0.74–1.45°；末端位置误差 0.03–0.19 mm",
     "trochanterfemur 可见率仅 6.5–33 %；顶层偏心基准点系统性低 2.03 mm；一个 seed 近乎全败"),
    ("平台可复现性", 0.70,
     "本轮修：力单位 1000×、门限按帧高归一化、重建改 PNG 重检测、外推律作废；删废弃探针与假数字",
     "工作台服务当前未运行；历史产物中仍有不可靠数字（已就地批注）"),
]

rows = sorted(ROWS, key=lambda r: r[1])
names = [r[0] for r in rows]
vals = [r[1] for r in rows]
evid = [r[2] for r in rows]
block = [r[3] for r in rows]

fig, ax = plt.subplots(figsize=(20.0, 9.4))
colors = ["#c0392b" if v < 0.2 else ("#e08a00" if v < 0.45 else "#2e7d32") for v in vals]
ax.barh(range(len(rows)), vals, color=colors, height=0.60)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(names, fontsize=11.5)
# TWO COLUMNS, so the evidence is never printed on top of the bar it describes.  The bar region is
# 0..1; the annotation column starts to its right, separated by a rule.
ax.set_xlim(0, 3.05)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.axvline(1.06, color="#999", lw=1.2)
ax.axvline(0.5, color="#666", ls="--", lw=1.1)
ax.text(0.5, len(rows) - 0.30, "half-way", color="#666", fontsize=9, ha="center")
ax.grid(axis="x", alpha=0.22)
ax.set_xlabel("readiness toward a usable experimental platform  (0 = absent, 1 = usable)"
              "        |        MEASURED evidence and the single thing blocking each row"
              "   —  score is MY JUDGEMENT, not a measurement", fontsize=10.5)

for i, (v, e, b) in enumerate(zip(vals, evid, block)):
    ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=10.5, fontweight="bold")
    ax.text(1.14, i + 0.16, "MEASURED:  " + e, va="center", fontsize=8.4, color="#111")
    ax.text(1.14, i - 0.19, "BLOCKED BY:  " + b, va="center", fontsize=8.4, color="#8e2b22")

fig.suptitle("Distance to a usable experimental platform — capability by capability\n"
             "hand-built research prototype: no real-fly measurement exists anywhere in this "
             "project, and the two strongest rows are the OBSERVATION chain, not the biology",
             fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
dest = HERE / "outputs" / "platform_gap_figure.png"
dest.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(dest, dpi=125)
print(f"wrote {dest}")
for r in rows:
    print(f"  {r[1]:.2f}  {r[0]}")
