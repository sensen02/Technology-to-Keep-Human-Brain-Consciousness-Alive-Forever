#!/usr/bin/env python3
"""Plot R28 all-DOF residuals and every-step release; no equilibrium inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    doc = json.loads(Path(args.json).read_text())
    if doc.get("status") != "DIAGNOSTIC_COMPLETED":
        raise ValueError("plot requires completed diagnostic")
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    ax = axes[0]
    for key, label in (("baseline", "Initial"), ("instantaneous_equilibrium", "Best evaluated")):
        a = np.abs(doc[key]["normalized_residual_signed"])
        ax.plot(np.arange(len(a)), np.maximum(a, 1e-12), ".-", label=label, alpha=.8)
    ax.axhline(1, color="black", ls="--", label="Absolute declared tolerance")
    ax.set(yscale="log", xlabel="DOF index (all nv; mapping in JSON)", ylabel="|qacc| / absolute tolerance",
           title="All-DOF acceleration: every component must be ≤1 (not a relative-decrease test)")
    ax.legend()
    ax.grid(alpha=.2)
    trace = doc["release"]["trace"]
    t = [r["time_s"] for r in trace]
    declarations = doc["declarations"]
    ax = axes[1]
    ax.plot(t, [r["height_model_length"] for r in trace], label="Thorax height")
    ax.axhline(declarations["min_height"], ls="--", color="black", label="Declared height minimum")
    ax.set(xlabel="Time (s)", ylabel="Height (model length; fly XML: mm)")
    ax.legend(loc="upper left")
    ax.grid(alpha=.2)
    right = ax.twinx()
    right.plot(t, [r["tilt_from_initial_deg"] for r in trace], color="tab:red", label="Tilt from initial")
    right.axhline(declarations["max_tilt_deg"], color="tab:red", ls=":", label="Declared tilt maximum")
    right.set_ylabel("Tilt from initial (degrees)")
    right.legend(loc="upper right")
    ax = axes[2]
    for key, threshold, label in (("body_ground_normal_bw", "max_body_bw", "Body contact normal/BW"),
                                   ("max_abs_linear_qvel", "max_linear_velocity", "Max linear speed"),
                                   ("max_abs_angular_qvel", "max_angular_velocity", "Max angular speed")):
        ax.plot(t, np.maximum([r[key] / declarations[threshold] for r in trace], 1e-12), label=label)
    ax.axhline(1, color="black", ls="--")
    ax.set(yscale="log", xlabel="Time (s)", ylabel="Value / declared maximum",
           title="Whole-interval release envelope (every timestep; not a stability proof)")
    ax.legend()
    ax.grid(alpha=.2)
    fig.suptitle("R28 manually implemented simulation diagnostic\n" + doc["equilibrium_verdict"] +
                 "\n" + doc["release"]["status"], fontsize=11)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
