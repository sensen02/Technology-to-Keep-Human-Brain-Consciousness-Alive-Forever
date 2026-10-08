import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from viz import save_panels_png
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
frames = [os.path.join(OUT, f"merks_trials/fine_chi_{c:.2f}.png") for c in (0.0,0.25,0.5,1.0)]
frames = [f for f in frames if os.path.exists(f)]
print("frames:", len(frames))
save_panels_png(os.path.join(OUT, "p1d_merks_morphology.png"),
                [{"img": np.asarray(Image.open(f)), "title": ""} for f in frames],
                ncols=4, figsize=(13.2, 3.8), dpi=130,
                suptitle="Merks 2008 cluster at the working point where the control "
                         "direction matches the paper (lambda_V=20, lambda_chem=100, "
                         "6000 MCS): chi=0 (leftmost) is the LEAST compact")
print("wrote p1d_merks_morphology.png")
