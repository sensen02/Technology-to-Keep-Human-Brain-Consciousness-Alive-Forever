"""Regenerate a multi-panel figure from saved frame PNGs, with the CORRECT
orientation (frames read back from PNG must not be flipped again).

Usage: venv/bin/python regen_panels.py <kind>   where kind in {p1,p3,p4}
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from viz import save_panels_png

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def frames_for(tag):
    fs = sorted(glob.glob(os.path.join(OUT, "p3_frames", f"{tag}_*.png")) +
                glob.glob(os.path.join(OUT, "p3_frames", f"prolif_on_*.png")))
    return fs


def panel_img(path, title=None):
    return {"img": np.asarray(Image.open(path)), "title": title or "", "origin": "upper"}


def mcs_of(path):
    base = os.path.basename(path).rsplit(".", 1)[0]
    return int(base.split("_")[-1])


def p3():
    fs = sorted(glob.glob(os.path.join(OUT, "p3_frames", "prolif_on_*.png")),
                key=mcs_of)[:6]
    if not fs:
        print("no p3 frames"); return
    panels = [panel_img(f, f"mcs {mcs_of(f)}") for f in fs]
    save_panels_png(os.path.join(OUT, "p3_wound_panels.png"), panels, ncols=3,
                    figsize=(13, 8.6), dpi=130,
                    suptitle="P3 epithelial wound closure (40 um wound, proliferation ON) "
                             "-- frames top-to-bottom, left-to-right in time")
    print("wrote p3 panels from", len(fs), "frames")


def p4():
    fs = sorted(glob.glob(os.path.join(OUT, "p4_frames", "ctrl_*.png")), key=mcs_of)
    if not fs:
        print("no p4 frames"); return
    panels = [panel_img(f, f"mcs {mcs_of(f)}") for f in fs]
    m = json.load(open(os.path.join(OUT, "p4_epidermis_metrics.json")))
    h = m["history_control"]
    fig_extra = None
    save_panels_png(os.path.join(OUT, "p4_epidermis_panels.png"), panels, ncols=3,
                    figsize=(13.5, 9.0), dpi=130,
                    suptitle="P4 self-organising stratified epidermis: basement membrane "
                             "and O2 supply at the BOTTOM; basal (dark blue) -> spinous "
                             "(light blue) -> granular (yellow) -> cornified (red) upward")
    print("wrote p4 panels from", len(fs), "frames")


def p1():
    for tag, out in (("ci_on", "p1_sprouting_panels.png"),):
        fs = sorted(glob.glob(os.path.join(OUT, "p1_frames", f"{tag}_*.png")), key=mcs_of)
        fs = [f for f in fs if "vegf" not in f and "o2" not in f][:6]
        if not fs:
            print("no p1 frames"); continue
        panels = [panel_img(f, f"mcs {mcs_of(f)}") for f in fs]
        save_panels_png(os.path.join(OUT, out), panels, ncols=3,
                        figsize=(13.5, 8.6), dpi=130,
                        suptitle="P1/P2 sprouting angiogenesis from a parent vessel "
                                 "(left vertical cord) into hypoxic parenchyma; "
                                 "red = Delta-Notch tip cells")
        print("wrote", out, "from", len(fs), "frames")


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "p4"
    {"p1": p1, "p3": p3, "p4": p4}[kind]()
