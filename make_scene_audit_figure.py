"""Build the audit figure from the measured JSON + cached arrays (no re-measure).

Reads    : outputs/embodied_body/scene_realism_audit.json
           outputs/embodied_body/_scene_audit_ground_<cond>.npz
           outputs/embodied_body/_scene_audit_eyes_<cond>.npz
Writes   : outputs/embodied_body/scene_realism_audit.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = "/run/media/sensen/Data2/cell_wound_prototype"
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from scene_realism_audit_run import (CAM_H, CAM_W, MM_PER_PX_IN,  # noqa: E402
                                    periodicity, radial_psd, spectrum2d)

OUT = os.path.join(ROOT, "outputs", "embodied_body")
CONDS = ["baseline", "blank", "natural_lit_no_objects", "natural"]
LABEL_CN = {"baseline": "baseline (scene_preset=None)",
            "blank": "blank", "natural_lit_no_objects": "natural_lit_no_objects",
            "natural": "natural"}

J = json.load(open(os.path.join(OUT, "scene_realism_audit.json")))
G = {c: np.load(os.path.join(OUT, f"_scene_audit_ground_{c}.npz")) for c in CONDS}
E = {c: np.load(os.path.join(OUT, f"_scene_audit_eyes_{c}.npz")) for c in CONDS}


def gimg(cond, cam):
    return G[cond][f"gray_{cam}"].astype(float)


def zoom_logspec(ax, img, mmpp, half_cpx=0.05, title=""):
    F, fy, fx = spectrum2d(img, window=True)
    H, W = img.shape
    cy, cx = H // 2, W // 2
    ny = int(half_cpx / (1.0 / H))
    nx = int(half_cpx / (1.0 / W))
    sub = F[cy - ny:cy + ny, cx - nx:cx + nx]
    ext = [-nx / W * mmpp * 0 + (-nx / W), nx / W, -ny / H, ny / H]
    ax.imshow(np.log10(sub + 1e-6), cmap="inferno", extent=ext, aspect="auto")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("fx (cycles/px)", fontsize=7)
    ax.set_ylabel("fy (cycles/px)", fontsize=7)
    ax.tick_params(labelsize=6)
    return F


def main():
    fig = plt.figure(figsize=(23, 27))
    gs = GridSpec(5, 4, figure=fig, hspace=0.42, wspace=0.30,
                  height_ratios=[1.0, 1.0, 1.15, 1.0, 1.0])

    # ---------------- row 0: ground renders ---------------------------------
    panels = [("baseline", "in"), ("natural", "in"),
              ("natural", "edge"), ("blank", "far")]
    for i, (cond, cam) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        im = gimg(cond, cam)
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        mmpp = J["per_condition"][cond]["ground_renders"][cam]["mm_per_px"]
        p = J["per_condition"][cond]["ground_renders"][cam]["top_peaks_labeled"][0]
        ax.set_title(f"[{LABEL_CN[cond]}] ground camera '{cam}'\n"
                     f"{mmpp:.5f} mm/px | peak/median="
                     f"{J['per_condition'][cond]['ground_renders'][cam]['periodicity']['peak_over_median']:.0f}"
                     f" | peak axis period {p['axis_wavelength_mm']:.4f} mm",
                     fontsize=8)
        ax.axis("off")
        if cam in ("in",):
            # full-resolution inset: the macroscopic look is a downscaling artefact,
            # so show a 110x110 px crop with nearest-neighbour upsampling
            cy, cx = CAM_H // 2, CAM_W // 2
            crop = im[cy - 55:cy + 55, cx - 55:cx + 55]
            axi = ax.inset_axes([0.60, 0.02, 0.38, 0.30])
            axi.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axi.set_title(f"full-res crop 1:1\n(std {im.std():.2f})", fontsize=6)
            axi.set_xticks([]); axi.set_yticks([])
        if cam == "edge":
            b = J["per_condition"][cond]["ground_renders"]["edge"]["seam"]
            xb = CAM_W // 2
            ax.axvline(xb, color="red", lw=1.4)
            ax.text(xb + 6, 40, "heightfield patch edge (y=30 mm)\n"
                                "texture 0.239 mm  <->  7.88 mm",
                    color="red", fontsize=7, va="top")

    # ---------------- row 1: 2-D log spectra --------------------------------
    for i, (cond, cam) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        img = gimg(cond, cam)
        mmpp = J["per_condition"][cond]["ground_renders"][cam]["mm_per_px"]
        F = zoom_logspec(ax, img, mmpp, half_cpx=0.05,
                         title=f"log10|F| zoom (|f|<=0.05 c/px) : {cond}/{cam}")
        pk = J["per_condition"][cond]["ground_renders"][cam]["top_peaks_labeled"]
        for q in pk[:4]:
            H, W = img.shape
            ax.add_patch(Rectangle((q["fx_cpx"] - 0.0022, q["fy_cpx"] - 0.0022),
                                   0.0044, 0.0044, fill=False, ec="cyan", lw=1.0))
        ax.text(0.02, 0.03, "cyan boxes = detected peaks\n(a checker lattice peaks at "
                            "(+/-f0,+/-f0))", transform=ax.transAxes, fontsize=6,
                color="w", va="bottom")

    # ---------------- row 2: radial PSDs + heightfield ----------------------
    ax = fig.add_subplot(gs[2, 0:2])
    colors = {"baseline": "tab:blue", "blank": "tab:cyan",
              "natural_lit_no_objects": "tab:orange", "natural": "tab:red"}
    for cond in CONDS:
        for cam, ls in (("far", "-"), ("in", "--")):
            r = J["per_condition"][cond]["ground_renders"][cam]["radial"]
            f = np.asarray(r["f_cycles_per_mm"], float)
            p = np.asarray(r["psd"], float)
            m = (f > 0.03) & (f < 8.0) & np.isfinite(p) & (p > 0)
            ax.loglog(f[m], p[m] / np.nanmax(p[m]), ls, color=colors[cond],
                      lw=1.0, alpha=0.85,
                      label=f"{cond}/{cam} slope={r['fit_windowed']['slope']:.2f} "
                            f"r2={r['fit_windowed']['r2']:.2f}")
    s = J["synthetic_1f_reference"]
    for beta, col, dy in ((1.0, "0.4", 1e-7), (2.0, "0.15", 1e-9), (3.0, "0.3", 1e-11)):
        b = s["by_beta"][f"beta_{beta}"]
        f = np.logspace(np.log10(0.05), np.log10(2.0), 50)
        ax.loglog(f, dy * (f / 0.05) ** (-beta), ":", color=col, lw=2.0,
                  label=f"synthetic 1/f^{beta:g} reference (dy={dy:g})")
    for wl, note in ((7.88, "plane checker 7.88 mm"), (2.0, "hfield 2.02 mm"),
                     (0.239, "patch checker 0.239 mm")):
        ax.axvline(1.0 / wl, color="k", lw=0.6, alpha=0.35)
        ax.text(1.0 / wl, 1e-13, note, rotation=90, fontsize=6, va="bottom")
    ax.set_xlabel("spatial frequency (cycles/mm)", fontsize=9)
    ax.set_ylabel("radial PSD (normalised per curve)", fontsize=9)
    ax.set_title("Radially averaged ground spectra, every condition and camera\n"
                 "dotted = synthetic 1/f^beta reference fields (grown to the same "
                 "physical band) ; a natural surface should be a straight dotted-like "
                 "line, the shipped ground is a COMB", fontsize=8)
    ax.legend(fontsize=6, ncol=2, loc="lower left")
    ax.grid(True, which="both", alpha=0.2)

    ax = fig.add_subplot(gs[2, 2])
    grid = np.asarray(
        np.load(os.path.join(OUT, "_scene_audit_ground_natural.npz"))["gray_in"]) * 0
    # the elevation grid itself is in the JSON only as profiles -> refit from model
    hf = J["per_condition"]["natural"]["heightfield"]
    px = np.asarray(hf["profile_x_mid"], float)
    py = np.asarray(hf["profile_y_mid"], float)
    ax.plot(np.linspace(-30, 30, px.size), px, lw=1.0, label="elevation along model X")
    ax.plot(np.linspace(-30, 30, py.size), py, lw=1.0, label="elevation along model Y")
    ax.set_title("heightfield elevation profile (normalised to [0,1])\n"
                 f"129x129 samples, cell {hf['cell_mm_x']:.4f} mm, "
                 f"p-p {hf['peak_to_peak_mm']:.4f} mm", fontsize=8)
    ax.set_xlabel("model coordinate (mm)", fontsize=8)
    ax.set_ylabel("normalised elevation", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 3])
    hr = hf["radial"]
    f = np.asarray(hr["f_cycles_per_mm"], float)
    p = np.asarray(hr["psd"], float)
    m = (f > 0) & np.isfinite(p) & (p > 0)
    ax.loglog(f[m], p[m], "-", color="tab:red", lw=1.2,
              label=f"2-D radial PSD, slope={hr['fit']['slope']:.3f} "
                    f"r2={hr['fit']['r2']:.3f}")
    for tag, col in (("transect_model_X", "tab:blue"), ("transect_model_Y", "tab:green")):
        t = hf[tag]
        k = np.asarray(t["f_cycles_per_mm"], float)
        q = np.asarray(t["psd"], float)
        mm = (k > 0) & np.isfinite(q) & (q > 0)
        ax.loglog(k[mm], q[mm] / np.nanmax(q[mm]), "--", color=col, lw=1.0,
                  label=f"1-D transect {tag[-1]}: slope={t['slope']:.3f} r2={t['r2']:.3f}")
    ax.axvline(0.5, color="k", lw=0.8)
    ax.text(0.5, 1e-3, " 2.0 mm", rotation=90, fontsize=7, va="bottom")
    for beta in (1.8, 2.0, 2.1):
        f2 = np.logspace(np.log10(0.05), np.log10(1.0), 30)
        ax.loglog(f2, 1e-4 * (f2 / 0.05) ** (-beta), ":", color="0.4", lw=1.5)
    ax.text(0.06, 1e-5, "dotted: k^-1.8, k^-2.0, k^-2.1\n(Pelletier 1997 topography)",
            fontsize=7)
    ax.set_title("HEIGHTFIELD spectrum: a delta at 2.02 mm, not a power law\n"
                 "the 'natural' preset's micro-relief is a pure sinusoid grating",
                 fontsize=8)
    ax.set_xlabel("cycles/mm", fontsize=8)
    ax.set_ylabel("PSD", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, which="both", alpha=0.2)

    # ---------------- row 3: sky occupancy maps -----------------------------
    for i, cond in enumerate(["baseline", "baseline", "natural", "natural"]):
        eye = "left" if i % 2 == 0 else "right"
        ax = fig.add_subplot(gs[3, i])
        mp = E[cond][f"sky_map_{eye}"]
        lum = mp[..., 0] + mp[..., 1] if mp.ndim == 3 else mp
        shown = np.where(lum < 0, np.nan, lum)
        im = ax.imshow(shown, cmap="gray", vmin=0, vmax=1)
        sky = np.where(lum < 0, np.nan, np.where(lum > 0.98, 1.0, 0.0))
        ax.contour(np.nan_to_num(sky), levels=[0.5], colors="red", linewidths=1.0)
        frac = J["per_condition"][cond]["eyes"]["R2_static_standing"]["per_eye"][eye][
            "frac_ommatidia_lum_sum_gt_0p98"]
        ax.set_title(f"{LABEL_CN[cond]} {eye} eye -- retinotopic ommatidia map\n"
                     f"red contour = uniform white sky (lum_sum>0.98): "
                     f"{frac * 100:.1f}% of ommatidia", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.035)

    # ---------------- row 4: eye frames + bar chart + calibration -----------
    for i, (cond, eye) in enumerate([("baseline", 0), ("baseline", 1),
                                     ("natural", 0), ("natural", 1)]):
        ax = fig.add_subplot(gs[4, i])
        fr = E[cond]["stand_frame_last"][eye]
        g = fr.astype(float).mean(axis=2)
        ax.imshow(fr)
        ax.contour(g == 255, levels=[0.5], colors="lime", linewidths=0.8)
        ax.contour(g == 0, levels=[0.5], colors="magenta", linewidths=0.8)
        pe = J["per_condition"][cond]["eyes"]["R2_static_standing"]["per_eye"][
            "left" if eye == 0 else "right"]
        ax.set_title(f"{LABEL_CN[cond]} {'left' if eye == 0 else 'right'} eye raw\n"
                     f"px==255 {pe['frac_pixels_exactly_255'] * 100:.1f}% (lime), "
                     f"px==0 {pe['frac_pixels_exactly_0'] * 100:.1f}% (magenta void)",
                     fontsize=8)
        ax.axis("off")

    fig.suptitle(
        "SCENE REALISM AUDIT -- how artificial is the current scene?  "
        "Ground periodicity, heightfield spectrum, sky occupancy\n"
        "measured from the compiled MJCF model and from rendered pixels; "
        "model units = millimetres; fly body length 2.5 mm.  "
        "EVERY ground camera shows a sharp periodic peak (peak/median 7x10^4 - 1.9x10^5);\n"
        "synthetic 1/f^beta fields of the SAME size measure peak/median 291-671 at "
        "beta=2 and 4724-12098 at beta=3, so the shipped ground is even more periodic "
        "than a beta=3 field",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    png = os.path.join(OUT, "scene_realism_audit.png")
    fig.savefig(png, dpi=95)
    plt.close(fig)
    print("wrote", png)

    # second figure: the quantitative summary bars + calibration
    fig2 = plt.figure(figsize=(20, 11))
    gs2 = GridSpec(2, 3, figure=fig2, hspace=0.42, wspace=0.28)
    ax = fig2.add_subplot(gs2[0, 0])
    x = np.arange(len(CONDS))
    w = 0.2
    for k, cam in enumerate(("in", "far", "far2x", "edge")):
        v = [J["per_condition"][c]["ground_renders"][cam]["periodicity"]
             ["peak_over_median"] for c in CONDS]
        ax.bar(x + (k - 1.5) * w, v, w, label=cam)
    b = J["synthetic_1f_reference"]["by_beta"]
    ax.axhline(b["beta_2.0"]["peak_over_median_max"], color="k", ls="--", lw=1.2,
               label=f"max for synthetic 1/f^2 = {b['beta_2.0']['peak_over_median_max']:.0f}")
    ax.axhline(b["beta_3.0"]["peak_over_median_min"], color="0.5", ls=":", lw=1.2,
               label=f"min for synthetic 1/f^3 = {b['beta_3.0']['peak_over_median_min']:.0f}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(CONDS, fontsize=8)
    ax.set_ylabel("ground periodicity  max|F| / median|F|", fontsize=9)
    ax.set_title("PERIODICITY DETECTOR: the shipped ground is far above any\n"
                 "1/f^beta field, including beta = 3", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = fig2.add_subplot(gs2[0, 1])
    betas, pm, sl = [], [], []
    for k, v in b.items():
        betas.append(v["beta"])
        pm.append(v["peak_over_median_median"])
        sl.append(v["fitted_slope_median"])
    ax.plot(betas, pm, "o-", label="peak/median (median over 6 seeds)")
    ax.errorbar(betas, pm, yerr=[np.array(pm) - np.array([v["peak_over_median_min"]
                                                          for v in b.values()]),
                                 np.array([v["peak_over_median_max"]
                                           for v in b.values()]) - np.array(pm)],
                fmt="none", capsize=3)
    ax.set_yscale("log")
    ax.set_xlabel("true exponent beta of the synthetic 1/f^beta field", fontsize=9)
    ax.set_ylabel("measured peak/median", fontsize=9)
    ax.set_title("DETECTOR CALIBRATION (the target is measured, not invented)",
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(betas, [-s for s in sl], "s--", color="tab:red",
             label="recovered slope")
    ax2.plot(betas, betas, ":", color="0.3")
    ax2.set_ylabel("recovered radial slope (should equal beta)", fontsize=9, color="tab:red")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="lower right")

    ax = fig2.add_subplot(gs2[0, 2])
    reg = ["R1_static_frozen", "R2_static_standing"]
    lab = ["R1 strictly static\n(pose pinned)", "R2 standing static\n(0.5 s window)"]
    for j, (r_, l_) in enumerate(zip(reg, lab)):
        v = [100 * J["per_condition"][c]["eyes"][r_]
             ["frac_ommatidia_exactly_constant_mean_of_eyes"] for c in CONDS]
        ax.bar(np.arange(len(CONDS)) + (j - 0.5) * 0.35, v, 0.35, label=l_)
    ax.set_xticks(np.arange(len(CONDS)))
    ax.set_xticklabels(CONDS, fontsize=8)
    ax.set_ylabel("% of ommatidia with EXACTLY constant readout", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("OMMATIDIA CARRYING NO TEMPORAL INFORMATION", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = fig2.add_subplot(gs2[1, 0])
    x = np.arange(len(CONDS))
    ax.bar(x - 0.2, [100 * J["per_condition"][c]["eyes"]["R2_static_standing"]["per_eye"]
                     ["left"]["frac_ommatidia_lum_sum_gt_0p98"] for c in CONDS], 0.4,
           label="left eye")
    ax.bar(x + 0.2, [100 * J["per_condition"][c]["eyes"]["R2_static_standing"]["per_eye"]
                     ["right"]["frac_ommatidia_lum_sum_gt_0p98"] for c in CONDS], 0.4,
           label="right eye")
    ax.set_xticks(x)
    ax.set_xticklabels(CONDS, fontsize=8)
    ax.set_ylabel("% of ommatidia", fontsize=9)
    ax.set_ylim(0, 60)
    ax.set_title("UNIFORM WHITE SKY OCCUPANCY (solid angle proxy)\n"
                 "lum_sum > 0.98", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = fig2.add_subplot(gs2[1, 1])
    for c in CONDS:
        fr = E[c]["stand_frame_last"]
        g = fr.astype(float).mean(axis=3)
        ax.bar(np.arange(2) + (CONDS.index(c) - 1.5) * 0.2,
               [100 * (g[i] == 255).mean() for i in range(2)], 0.2, label=c)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["left eye", "right eye"], fontsize=9)
    ax.set_ylabel("% of eye-frame pixels exactly 255", fontsize=9)
    ax.set_title("EYE FRAME: uniform-sky pixels", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = fig2.add_subplot(gs2[1, 2])
    ax.axis("off")
    txt = []
    for c in CONDS:
        r = J["per_condition"][c]
        txt.append(f"{c}:")
        txt.append(f"   ngeom={r['model']['ngeom']} nlight={r['model']['nlight']} "
                   f"nhfield={r['model']['nhfield']} nu={r['model']['nu']}")
        txt.append(f"   ground peak/median  in={r['ground_renders']['in']['periodicity']['peak_over_median']:.0f}"
                   f" far={r['ground_renders']['far']['periodicity']['peak_over_median']:.0f}"
                   f" edge={r['ground_renders']['edge']['periodicity']['peak_over_median']:.0f}")
        txt.append(f"   period in={r['ground_renders']['in']['top_peaks_labeled'][0]['axis_wavelength_mm']:.4f} mm"
                   f"  far={r['ground_renders']['far']['top_peaks_labeled'][0]['axis_wavelength_mm']:.4f} mm")
        txt.append(f"   fitted radial slope: in={r['ground_renders']['in']['radial']['fit_windowed']['slope']}"
                   f" r2={r['ground_renders']['in']['radial']['fit_windowed']['r2']}")
        if r["heightfield"]["present"]:
            txt.append(f"   hfield: {r['heightfield']['nrow']}x{r['heightfield']['ncol']}"
                       f" p-p {r['heightfield']['peak_to_peak_mm']:.4f} mm"
                       f" peak/median {r['heightfield']['periodicity']['peak_over_median']:.0f}"
                       f" lambda {r['heightfield']['top_peaks'][0]['radial_wavelength_mm']:.4f} mm"
                       f" slope {r['heightfield']['radial']['fit']['slope']:.3f}"
                       f" (r2 {r['heightfield']['radial']['fit']['r2']:.3f})")
    tgt = J["target_for_a_natural_scene"]
    txt.append("")
    txt.append("TARGET for a natural scene (derived, see JSON):")
    txt.append(f"   peak/median < {tgt['periodicity_peak_over_median']['value']:.0f} "
               f"(synthetic max at beta=1.5 is 159.9, at beta=2.0 is 671.3)")
    txt.append(f"   fitted radial slope in {tgt['fitted_radial_power_law_slope']['range']}"
               f" with r2 >= {tgt['fitted_radial_power_law_slope']['r2_required']}")
    txt.append("   no sharp peak with wavelength in [1.25, 5.0] mm (= 0.5-2x fly length)")
    txt.append(f"   sky occupancy < {tgt['sky_occupancy_fraction_of_ommatidia']['value']}"
               " of ommatidia (ENGINEERING threshold, not from literature)")
    txt.append(f"   exactly-constant ommatidia < 0.5 in a static case")
    ax.text(0.0, 1.0, "\n".join(txt), fontsize=7.2, va="top", family="monospace")
    fig2.suptitle("SCENE REALISM AUDIT -- summary numbers and detector calibration",
                  fontsize=13)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    png2 = os.path.join(OUT, "scene_realism_audit_summary.png")
    fig2.savefig(png2, dpi=95)
    plt.close(fig2)
    print("wrote", png2)


if __name__ == "__main__":
    main()
