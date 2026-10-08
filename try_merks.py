"""Quick test: does the published parameter set reproduce sprouting?"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from merks import MerksVasculogenesis, DIFFUSION_LENGTH_SITES, CELL_SITES
from viz import type_image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "merks_trials")
os.makedirs(OUT, exist_ok=True)
print("diffusion length (sites):", DIFFUSION_LENGTH_SITES, "= 10 um; cell sites:", CELL_SITES)

for ratio, tag in ((0.0, "ci_full"), (1.0, "ci_none")):
    t0 = time.time()
    v = MerksVasculogenesis(L=150, seed=1, mode="cluster", n_cells=100,
                            chem_cc_ratio=ratio)
    print(f"[{tag}] seeded cells={v.n_made} occupy={v.m.occupancy()} sites", flush=True)
    v.run(1200, record_every=200, verbose=True)
    h = v.history[-1]
    img = type_image(v.m, {1: (0.42, 0.60, 0.90)})
    fp = os.path.join(OUT, f"{tag}.png")
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=115)
    ax.imshow(img, interpolation="nearest", origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{tag} chi_cc={ratio}  EC={h['n_ec']} comp={h['n_components']} "
                 f"lacunae={h['n_lacunae']} C={h['compactness']:.3f}", fontsize=8)
    fig.tight_layout(); fig.savefig(fp); plt.close(fig)
    print(f"RESULT {tag} EC={h['n_ec']} comp={h['n_components']} lacunae={h['n_lacunae']} "
          f"C={h['compactness']:.4f} wall={time.time()-t0:.0f}s -> {fp}", flush=True)
