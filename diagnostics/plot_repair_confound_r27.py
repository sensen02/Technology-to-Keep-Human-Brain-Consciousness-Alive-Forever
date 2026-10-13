"""Figure for diagnostics/diag_repair_confound_r27.py.  Agg backend, no display needed."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "outputs/repair_confound_r27.json").read_text())
rows = d["rows"]

VARIANTS = ["both repairs (canonical)", "no arm repair", "qpos0 bands"]
SHORT = ["both repairs\n(canonical)", "no arm repair\n(STAGE 2b off)", "qpos0 bands\n(STAGE 2c off)"]
CH_DIS = "disabled (no muscle force at all)"
CH_EN = "enabled, act = 0 (passive muscle force only)"


def get(ch, var, rel):
    for r in rows:
        if r["activation_channel"] == ch and r["variant"] == var and r["rf_locks_released"] == rel:
            return r
    raise KeyError((ch, var, rel))


fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
for ax, ch, title in [
    (axs[0], CH_DIS, "actuation DISABLED — no muscle force exists\n(asserted zero at every step)"),
    (axs[1], CH_EN, "actuation ENABLED, act = 0 — passive muscle force only\n(this is the channel R25 measured)"),
]:
    x = range(len(VARIANTS))
    locked = [get(ch, v, False)["final_thorax_z_mm"] for v in VARIANTS]
    free = [get(ch, v, True)["final_thorax_z_mm"] for v in VARIANTS]
    ax.bar([i - 0.2 for i in x], locked, width=0.38, color="#ad5555", label="7 RF locks active")
    ax.bar([i + 0.2 for i in x], free, width=0.38, color="#448888", label="locks released")
    for i, (a, b) in enumerate(zip(locked, free)):
        ax.text(i - 0.2, a, f"{a:.4f}", ha="center", va="bottom", fontsize=8.5)
        ax.text(i + 0.2, b, f"{b:.4f}", ha="center", va="bottom", fontsize=8.5)
        ax.annotate("", xy=(i, b), xytext=(i, a),
                    arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0))
        ax.text(i, (a + b) / 2, f"lock\n{a - b:.4f}", ha="center", va="center", fontsize=8,
                color="#333333", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    ax.set_xticks(list(x))
    ax.set_xticklabels(SHORT, fontsize=9)
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 1.85)
axs[0].set_ylabel("final thorax height after 2 s (mm)")
axs[0].legend(loc="upper center", fontsize=9)

spread_dis = max(get(CH_DIS, v, False)["final_thorax_z_mm"] - get(CH_DIS, v, True)["final_thorax_z_mm"]
                 for v in VARIANTS) - min(
    get(CH_DIS, v, False)["final_thorax_z_mm"] - get(CH_DIS, v, True)["final_thorax_z_mm"]
    for v in VARIANTS)
spread_en = max(get(CH_EN, v, False)["final_thorax_z_mm"] - get(CH_EN, v, True)["final_thorax_z_mm"]
                for v in VARIANTS) - min(
    get(CH_EN, v, False)["final_thorax_z_mm"] - get(CH_EN, v, True)["final_thorax_z_mm"]
    for v in VARIANTS)

fig.suptitle("R27: R25's \"the repairs lifted the passive body to 1.3917 mm\" is the LOCKED foreleg, not the repairs",
             fontsize=12)
fig.text(0.5, 0.055,
         "R25 reported passive settle 0.7163 -> 1.3917 mm as the effect of the two repairs.  With the locks active the\n"
         f"canonical model reproduces 1.3917 mm exactly; the arm repair's own contribution to that number is 0.0002 mm,\n"
         f"and with no muscle force all three variants are bit-identical (lock-effect spread {spread_dis:.4f} mm vs {spread_en:.4f} mm).\n"
         "Files are distinct on disk and in structure (site diff 0.414 mm, lengthrange diff 0.646): this is physics, not a stale loader.",
         ha="center", va="center", fontsize=8.6)
fig.tight_layout(rect=(0, 0.13, 1, 0.93))
out = ROOT / "outputs/repair_confound_r27.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
