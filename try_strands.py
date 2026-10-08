"""One focused attempt to reach the strand regime (strong elongation, weak
cohesion, slow proliferation).  Saves a final-state PNG per config so the
result can be judged VISUALLY, not from metrics."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vasculogenesis import Vasculogenesis
from viz import type_image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "strand_trials")
os.makedirs(OUT, exist_ok=True)

BASE = dict(L=160, seed=5, n_fibroblast=22, mode="sprouting",
            params={"o2_consumption": 0.05, "vegf_secretion_fib": 0.03,
                    "vegf_secretion_ec": 0.001, "vegf_decay": 0.004,
                    "tip_fraction_percentile": 80.0})

CFGS = [
    dict(tag="A_base",   p={"lambda_chem_tip": 60.0, "lambda_chem_stalk": 4.0,
                            "lambda_length": 1.5, "elongation_target": 2.0, "cycle_mcs": 300.0}),
    dict(tag="B_long",   p={"lambda_chem_tip": 60.0, "lambda_chem_stalk": 2.0,
                            "lambda_length": 4.0, "elongation_target": 3.2, "cycle_mcs": 1200.0,
                            "J_ec_ec": 2.0, "J_ec_medium": 5.0}),
    dict(tag="C_noprolif", p={"lambda_chem_tip": 80.0, "lambda_chem_stalk": 1.0,
                              "lambda_length": 3.0, "elongation_target": 3.0, "cycle_mcs": 3000.0,
                              "J_ec_ec": 2.0, "J_ec_medium": 5.5}),
]

for cfg in CFGS:
    pr = dict(BASE["params"]); pr.update(cfg["p"])
    v = Vasculogenesis(**{**BASE, "params": pr})
    t0 = time.time()
    v.run(2500, record_every=500, verbose=False)
    h = v.history[-1]
    img = type_image(v.m, __import__("vasculogenesis").TYPE_COLORS,
                     state_of={"tip": [c for c, s in v.state.items()
                                       if s == "tip" and v.m.ctype[c] == 1]},
                     state_colors=__import__("vasculogenesis").STATE_COLORS)
    fp = os.path.join(OUT, f"{cfg['tag']}.png")
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=115)
    ax.imshow(img, interpolation="nearest", origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{cfg['tag']}  EC={h['n_ec']} tips={h['n_tip']} comp={h['ec_components']} "
                 f"loops={h['ec_loops']} P/A={h['ec_perimeter_over_area']:.2f}", fontsize=8)
    fig.tight_layout(); fig.savefig(fp); plt.close(fig)
    print(f"RESULT {cfg['tag']} EC={h['n_ec']} tips={h['n_tip']} comp={h['ec_components']} "
          f"loops={h['ec_loops']} reach={h['ec_max_reach_sites']:.0f} "
          f"P/A={h['ec_perimeter_over_area']:.3f} area={h['ec_area_fraction']:.3f} "
          f"wall={time.time()-t0:.0f}s -> {fp}", flush=True)
