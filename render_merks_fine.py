"""Render the cluster morphology at the fine-scan working point (lambda_V=20,
lambda_chem=100, 6000 MCS, seed 1) so the metrics can be checked visually."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from merks import MerksVasculogenesis, TYPE_COLORS
from viz import type_image, save_panels_png

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
frames = []
for chi in (0.0, 0.25, 0.5, 1.0):
    v = MerksVasculogenesis(L=200, seed=1, mode="cluster", n_cells=128,
                            chem_cc_ratio=chi,
                            params={"lambda_chem": 100.0, "lambda_volume": 20.0})
    v.run(6000, record_every=1000, verbose=False)
    h = v.history[-1]
    img = type_image(v.m, TYPE_COLORS)
    p = os.path.join(OUT, f"merks_trials/fine_chi_{chi:.2f}.png")
    fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=110)
    ax.imshow(img, interpolation="nearest", origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"chi={chi:.2f}  C={h['compactness']:.3f}  comp={h['n_components']}",
                 fontsize=8)
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    frames.append(p)
    print(f"chi={chi:.2f} C={h['compactness']:.4f} comp={h['n_components']} "
          f"lacunae={h['n_lacunae']}", flush=True)
save_panels_png(os.path.join(OUT, "p1d_merks_morphology.png"),
                [{"img": np.asarray(__import__("PIL.Image", fromlist=["Image"]).Image.open(f)),
                  "title": ""} for f in frames],
                ncols=4, figsize=(13.2, 3.7), dpi=130,
                suptitle="Merks 2008 cluster at the point where the control direction "
                         "matches the paper (lambda_V=20, lambda_chem=100, 6000 MCS): "
                         "chi=0 (left) is LEAST compact")
print("wrote p1d_merks_morphology.png")
