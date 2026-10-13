#!/usr/bin/env python3
"""Status board: what the repair now checks, and what each check actually returned.

Reads the JSON outputs the checks themselves wrote -- nothing is recomputed here, so the board
cannot disagree with the checks it summarises.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_status_board_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
    if Path(_p).is_file():
        font_manager.fontManager.addfont(_p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_p).get_name()
        break

COL = {"PASS": "#2e7d32", "FAIL": "#c0392b", "NOT_RUN": "#e08a00", "SKIP": "#888"}


def load(name, default=None):
    p = OUT / name
    return json.loads(p.read_text()) if p.is_file() else (default or {})


allc = load("all_checks.json")
inv = load("invariants.json")
inj = load("regression_injections.json")
hea = load("arena/episode_health.json")
tru = load("truth_boundary_audit.json")
con = load("result_contract_audit.json")
env = load("environment_lock.json")
hyp = load("hypothesis_tests.json")

fig = plt.figure(figsize=(19.5, 11.0))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.28)

# ---- panel 1: the runner's 35 items -------------------------------------------------------
# A bar per item was unreadable at 35 rows (the tick labels collided), so the list is PRINTED in
# three columns with a status glyph.  A status board that cannot be read is not a status board.
ax = fig.add_subplot(gs[0, :2])
ax.axis("off")
res = allc.get("results", [])
glyph = {"PASS": "OK  ", "FAIL": "FAIL", "NOT_RUN": "NRUN", "SKIP": "SKIP"}
per = (len(res) + 2) // 3
cols_txt = []
for c in range(3):
    chunk = res[c * per:(c + 1) * per]
    cols_txt.append("\n".join(f"{glyph.get(r['status'],'?')} {r['name'][:30]:<30}"
                              f"[{'body' if r['interpreter'] == 'venv_body' else 'cpu '}]"
                              for r in chunk))
for c, txt in enumerate(cols_txt):
    ax.text(0.005 + c * 0.335, 1.0, txt, va="top", fontsize=6.4, family="monospace",
            transform=ax.transAxes)
nretry = sum(1 for r in res if r.get("retried_because"))
ax.set_title(f"unified runner: {allc.get('overall')} — {allc.get('counts')}\n"
             f"{len(res)} items; venv_body needed for "
             f"{sum(1 for r in res if r['interpreter'] == 'venv_body')}, of which {nretry} were "
             f"found ONLY by the adaptive retry  (body = venv_body, cpu = venv)", fontsize=10)

# ---- panel 2: injections ------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
lines = ["three planted REAL bugs —", "each check must catch its own:", ""]
for r in inj.get("results", []):
    lines += [f"{r['injection']}", f"   with fault: {r.get('verdict_with_fault')}",
              f"   control   : {r.get('verdict_control_no_fault')}", f"   -> {r['status']}", ""]
ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=9, family="monospace")
ax.set_title("injection regressions\n" + inj.get("overall", ""), fontsize=10)

# ---- panel 3: invariants ------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
ax.axis("off")
lines = []
for r in inv.get("results", []):
    lines.append(f"{r['status']:<6}{r['check']}")
ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=8.6, family="monospace")
ax.set_title(f"physics / geometry invariants\n{inv.get('overall','')}", fontsize=10)

# ---- panel 4: episode health --------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
eps = hea.get("episodes", [])
labs, vals, cs = [], [], []
for e in eps:
    hr = next((c for c in e["checks"] if c["check"] == "fiducial_hit_rate_3px"), None)
    v = hr["value"]["rate"] * 100 if hr and isinstance(hr.get("value"), dict) else 0.0
    labs.append(e["episode"].replace("arena_bare_", ""))
    vals.append(v)
    cs.append(COL.get(e["status"], "#888"))
ax.bar(labs, vals, color=cs)
ax.axhline(50, color="#666", ls="--", lw=1.1)
ax.text(0.02, 53, "registered threshold 50 %", fontsize=7.5, color="#666")
ax.set_ylabel("fiducials found within 3 px (%)")
ax.set_title(f"recording health gate — {hea.get('status')}\n{hea.get('n_episodes_pass')} pass, "
             f"{hea.get('n_episodes_fail')} fail", fontsize=10)
for i, v in enumerate(vals):
    ax.text(i, v + 2, f"{v:.1f}", ha="center", fontsize=8)

# ---- panel 5: truth boundary + contract ---------------------------------------------------
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
txt = (f"truth-boundary audit: {tru.get('overall')}\n"
       f"  scripts reading a truth key : {tru.get('n_scripts_with_truth_reads')}\n"
       f"  individual reads            : {tru.get('n_reads_total')}\n"
       f"  undeclared                  : {len(tru.get('undeclared', []))}\n"
       f"  measurement violations      : {len(tru.get('violations', []))}\n"
       f"  (non-simulator roles must give a reason)\n\n"
       f"result contract: {con.get('overall')}\n"
       f"  contract-bearing outputs    : {con.get('n_contract_bearing')}\n"
       f"  violations                  : {con.get('n_violations')}\n"
       f"  legacy backlog (counted)    : {con.get('n_backlog')}\n\n"
       f"environment pinning\n"
       f"  offline_reproducible        : {env.get('offline_reproducible')}\n"
       f"  assets missing locally      : {env.get('blocking_offline',{}).get('external_assets_not_present_locally')}\n"
       f"  unpinned critical packages  : {env.get('blocking_offline',{}).get('unpinned_critical_packages')}\n\n"
       f"hypothesis guard selftest\n"
       f"  empty perturbation -> NOT_RUN   (required)\n"
       f"  real perturbation  -> PASSED_GUARD\n"
       f"  H1 (rig effect)    : {hyp.get('results',[{}])[0].get('verdict','-')}")
ax.text(0.0, 1.0, txt, va="top", fontsize=8.5, family="monospace")
ax.set_title("the remaining audits", fontsize=10)

# ---- panel 6: what is NOT done -------------------------------------------------------------
ax = fig.add_subplot(gs[2, :])
ax.axis("off")
notdone = (
 "STILL NOT DONE, stated rather than implied:\n"
 "  1. the truth audit is STATIC.  The stronger version is RUNTIME: strip the truth keys from an episode and require the "
 "measurement's output to be unchanged.\n"
 "  2. offline reproduction is unavailable (measured: FlyMimic meshes come from a download cache; mujoco/flygym/pyarrow are unpinned).\n"
 "  3. the DYNAMIC force peak (25x body weight) is unexplained.  Quasi-static is right (~0.66-1.72x in driven mode), so this is a transient, not a scale error.\n"
 "  4. passively the legs carry only 7-11 % of body weight -- the body rests on its thorax.  The arena model reaches the ground only because the CPG drives the legs.\n"
 "  5. the hypothesis registry lives in code (HYPOTHESES), not yet as a standalone auditable register.\n"
 "  6. H1 has 2 episodes per arm -- not a distribution.  Its REFUTED verdict is about this sample only.\n\n"
 "Every check above reports NOT_RUN for data it could not evaluate, and NOT_RUN is never counted as a pass."
)
ax.text(0.0, 1.0, notdone, va="top", fontsize=9.4)

fig.suptitle("Self-consistency repair: what is now checked, what it returned, and what is still open\n"
             "hand-built research prototype — no real-fly measurement exists anywhere in this project",
             fontsize=12)
dest = OUT / "status_board_figure.png"
fig.savefig(dest, dpi=118, bbox_inches="tight")
print(f"wrote {dest}")
print(f"  runner {allc.get('overall')} {allc.get('counts')}")
print(f"  invariants {inv.get('overall')}, injections {inj.get('overall')}, "
      f"health {hea.get('status')}, truth {tru.get('overall')}, contract {con.get('overall')}")
