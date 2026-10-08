"""Manually implemented demo + figure for the illustrative electrode-damage model.

Produces (all under outputs/brain_isolation/):
  electrode_damage_report.json   every number with its provenance tier
  electrode_damage_field.npz     the field, sweeps and paired traces
  electrode_damage_demo.png      12-panel figure
  electrode_damage_selftest.json the executable test counts

WHAT THIS IS: a mechanically-motivated, Drosophila-SCALED damage model built on
an assumed strain field, with every threshold imported from mammals as a
cross-species proxy and force-flagged.  WHAT IT IS NOT: a prediction of real fly
electrode damage, and no claim about consciousness, viability or medical use.
There is NO quantitative insect electrode-damage measurement (two independent
searches), so nothing here is calibrated against an insect experiment.
"""
from pathlib import Path
from dataclasses import asdict
import json
import resource
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from engine.cable import Morphology
from engine.electrode_damage import (
    ADDENDUM_TAG, AffineDamageRadius, CableDamageConfig, CavityExpansionField,
    FlyBrainGeometry, HONESTY_STATEMENTS,
    NO_DROSOPHILA_LAMELLA_MEASUREMENT, NO_INSECT_INSERTION_DAMAGE_QUANTIFICATION,
    NO_SECONDS_SCALE_SHEATH_RESEALING, ProximityExcitabilityCoupling,
    STRAIN_THRESHOLD_CHOICES, SHEATH_LOAD_BEARING_EVIDENCE, SheathLayer,
    SheathRepairModel, SpeciesGuard, audit_species_labeling,
    build_registry, chemistry_config_for_variant, FLY_SPECIES,
    injury_threshold_options, parenchymal_damage_estimate, preparation_variants,
    provenance_tier, rodent_radius_transfer_table, run_paired_cable,
    run_paired_chemistry)
from engine.profile import DROSOPHILA_CNS
from run_electrode_damage_selftest import run_tests

OUT = Path(__file__).resolve().parent / "outputs" / "brain_isolation"

PROBE_DIAMETERS_UM = np.array([0.5, 1.0, 2.0, 4.0, 7.5, 10.0, 15.0, 25.0, 50.0, 100.0])
SWEEP_RADII_UM = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
SHEATH_CRITICAL_UM = (0.5, 1.0, 2.0, 3.0, 5.0)
CABLE_KWARGS = dict(Cm_uF_cm2=1.0, g_leak_S_cm2=1e-4, Ra_ohm_cm=150.0, E_leak_mV=-65.0)


def main():
    t_start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- guard
    tests = run_tests()                      # the executable evidence, re-run here
    assert tests["passed"], "selftest failed; not writing a report"
    (OUT / "electrode_damage_selftest.json").write_text(json.dumps(tests, indent=2, default=float))

    registry = build_registry()
    guard_default_refused = None
    try:
        SpeciesGuard(DROSOPHILA_CNS, registry, allow_cross=False)
    except Exception as exc:                 # the refusal itself is the evidence
        guard_default_refused = f"{type(exc).__name__}: {str(exc)[:400]}"
    assert guard_default_refused, "the species guard did NOT refuse the cross-species import"
    guard = SpeciesGuard(DROSOPHILA_CNS, registry, allow_cross=True)
    adopt = guard.adopt

    # ------------------------------------------------------------- fixtures
    geometry = FlyBrainGeometry.from_adopt(adopt)
    thresholds = injury_threshold_options(guard)
    strain_records = {k: thresholds[k] for k in STRAIN_THRESHOLD_CHOICES}
    shear_record = thresholds["_mechanoporation_shear"]
    resealing_record = adopt("rodent_reseal_small_pore_tau_s")
    transected_record = adopt("rodent_reseal_transected_tau_s")

    sheath = SheathLayer(
        thickness_um=adopt("fly_sheath_thickness_um").value,
        critical_opening_radius_um=adopt("assumed_sheath_critical_opening_radius_um").value,
        delamination_multiple=adopt("assumed_sheath_delamination_multiple").value)
    variants = preparation_variants(adopt)
    repair = SheathRepairModel(resealing_record.value, transected_record.value,
                              adopt("assumed_sheath_repair_tau_s").value,
                              adopt("insect_sheath_repair_persists_days").value)

    primary_diameter = adopt("assumed_shaft_diameter_um").value            # 10 um
    field = CavityExpansionField(
        shaft_radius_um=primary_diameter / 2.0,
        wall_displacement_fraction=adopt("assumed_wall_displacement_fraction").value,
        shear_modulus_Pa=adopt("assumed_tissue_shear_modulus_Pa").value)
    primary_threshold = strain_records["optimal"]

    # -------------------------------------------------- geometry + transfer
    transfer = rodent_radius_transfer_table(geometry, adopt, 900.0)
    affine = AffineDamageRadius(adopt("assumed_affine_damage_offset_um").value,
                               adopt("assumed_affine_damage_slope").value)

    diameter_curve = {}
    for key, record in strain_records.items():
        radii, fractions, sheath_intact = [], [], []
        for d in PROBE_DIAMETERS_UM:
            f = CavityExpansionField(d / 2.0, field.wall_displacement_fraction,
                                    field.shear_modulus_Pa)
            r = f.strain_radius_um(record.value)
            radii.append(r)
            fractions.append(geometry.damage_fraction(r))
            est = parenchymal_damage_estimate(f, record.value, geometry, sheath,
                                             variants["sheath_intact"],
                                             threshold_record=record)
            sheath_intact.append(est["parenchymal_fraction_mechanistic"])
        diameter_curve[key] = {"probe_diameter_um": PROBE_DIAMETERS_UM.copy(),
                               "damage_radius_um": np.array(radii),
                               "fly_brain_fraction": np.array(fractions),
                               "fly_brain_fraction_sheath_intact": np.array(sheath_intact)}
    shear_radii = np.array([
        CavityExpansionField(d / 2.0, field.wall_displacement_fraction,
                            field.shear_modulus_Pa).shear_stress_radius_um(shear_record.value)
        for d in PROBE_DIAMETERS_UM])
    affine_radii = np.array([affine.radius_um(d / 2.0) for d in PROBE_DIAMETERS_UM])

    free_sweep = {}
    for name, variant in variants.items():
        frac = []
        for r in SWEEP_RADII_UM:
            est = parenchymal_damage_estimate(field, primary_threshold.value, geometry,
                                             sheath, variant, free_radius_um=r,
                                             threshold_record=primary_threshold)
            frac.append(est["parenchymal_fraction_free_parameter"])
        free_sweep[name] = np.array(frac)

    primary_estimates = {
        name: parenchymal_damage_estimate(field, primary_threshold.value, geometry,
                                         sheath, variant, threshold_record=primary_threshold)
        for name, variant in variants.items()}
    # free-parameter variant of the primary case: an explicitly unmeasured radius
    free_radius = adopt("assumed_parenchymal_free_radius_um").value
    free_estimates = {
        name: parenchymal_damage_estimate(field, primary_threshold.value, geometry,
                                         sheath, variant, free_radius_um=free_radius,
                                         threshold_record=primary_threshold)
        for name, variant in variants.items()}

    # ------------------------------------------------------ field profile
    r_profile = np.geomspace(field.shaft_radius_um, 300.0, 240)
    profile = field.profile_arrays(r_profile)

    # --------------------------------------------------------- cable pairing
    morph = Morphology.cylinder(500.0, 0.5, 250)
    # perpendicular offset chosen so the shaft SURFACE stays clear of the neurite
    offset_um = field.shaft_radius_um + 1.0
    shaft_point = (250.0, offset_um, 0.0)
    cable = run_paired_cable(morph, field, shaft_point, (0.0, 0.0, 1.0),
                            primary_threshold.value,
                            config=CableDamageConfig(
                                leak_density_S_cm2=adopt("assumed_damaged_leak_density_S_cm2").value,
                                excitability=ProximityExcitabilityCoupling(
                                    enabled=False,
                                    full_effect_radius_um=adopt("assumed_proximity_full_effect_radius_um").value,
                                    max_attenuation=adopt("assumed_proximity_max_drive_attenuation").value)),
                            dt_ms=0.05, duration_ms=50.0, cable_kwargs=CABLE_KWARGS)
    cable_trace = {"x_um": morph.x.copy(),
                   "sham_mV": cable["sham"][-1].copy(),
                   "inserted_mV": cable["inserted"][-1].copy(),
                   "delta_mV": cable["delta_mV"][-1].copy()}
    force_slope = adopt("rodent_probe_force_slope_mN_per_mm")
    force_intercept = adopt("rodent_probe_force_intercept_mN")
    compress_intercept = adopt("rodent_probe_compress_intercept_mm")
    compress_slope = adopt("rodent_probe_compress_slope_mm_per_mm")
    force_at = {f"{d:g}": float((force_intercept.value + force_slope.value * d / 1000.0) * 1000.0)
                for d in (7.5, 15.0, 100.0)}
    compression_at = {f"{d:g}": float((compress_intercept.value + compress_slope.value * d / 1000.0) * 1000.0)
                      for d in (7.5, 15.0, 100.0)}

    # ------------------------------------------------------ chemistry pairing
    chemistry_fraction = 0.7          # fraction of the local tissue element inside r_dmg
    chemistry = {}
    for name, variant in variants.items():
        chemistry[name] = run_paired_chemistry(
            chemistry_fraction, config=chemistry_config_for_variant(variant, adopt),
            dt_s=0.05, duration_s=20.0, include_supported=True)
    chem_arrays = {
        "time_s": chemistry["sheath_intact"]["variants"]["closed"]["traces"]["sham"][:, 0].copy()}
    for name in variants:
        for variant_tag in ("closed", "supported"):
            for run in ("sham", "inserted"):
                chem_arrays[f"{name}_{variant_tag}_{run}_Ke_mM"] = \
                    chemistry[name]["variants"][variant_tag]["traces"][run][:, 2].copy()
                chem_arrays[f"{name}_{variant_tag}_{run}_far_ligand_nM"] = \
                    chemistry[name]["variants"][variant_tag]["traces"][run][:, 5].copy()

    # ================================================================ FIGURE
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 7.5, "figure.titlesize": 15})
    fig = plt.figure(figsize=(24, 17.0), layout="constrained")
    outer = fig.add_gridspec(2, 1, figure=fig, height_ratios=[24, 5])
    grid = outer[0].subgridspec(3, 4)
    footer = outer[1].subgridspec(1, 2)
    a = field.shaft_radius_um
    delta = field.wall_displacement_um
    col = {"conservative": "tab:red", "optimal": "tab:orange", "permissive": "tab:blue"}

    # ---- (0,0) fly-scale coronal geometry, true scale, with a zoomed inset --
    ax = fig.add_subplot(grid[0, 0])
    ax.add_patch(plt.Rectangle((0, 0), geometry.length_um, geometry.width_um,
                               fill=False, ec="k", lw=1.4))
    cx, cy = geometry.length_um / 2, geometry.width_um / 2
    for row, colour in zip(transfer["rows"], ("tab:green", "tab:purple", "tab:brown", "tab:gray")):
        r = row["radius_record"]["value"]
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec=colour, lw=1.3,
                                label=f"rodent {r:g} um -> {row['fly_brain_volume_percent']:.1f}% of fly brain"))
    ax.plot([cx], [cy], "k+", ms=9)
    ax.plot([cx, cx], [0, geometry.width_um], color="k", lw=2.0, alpha=.6)
    ax.set(xlim=(-10, 510), ylim=(-10, 310), xlabel="x (um)", ylabel="y (um)",
           title="(a) Fly brain footprint 500x300 um, TRUE scale\nshaft + rodent-reported 'damage radii'")
    ax.legend(loc="lower right", framealpha=.92, fontsize=6.4)
    ax.grid(alpha=.2)
    box = 40.0
    ax.add_patch(plt.Rectangle((cx - box / 2, cy - box / 2), box, box, fill=False,
                               ec="red", lw=1.0, ls=":"))
    inset = ax.inset_axes([0.03, 0.54, 0.44, 0.36])
    xs = np.linspace(cx - box / 2, cx + box / 2, 260)
    ys = np.linspace(cy - box / 2, cy + box / 2, 200)
    Xg, Yg = np.meshgrid(xs, ys)
    Rg = np.hypot(Xg - cx, Yg - cy)
    strain = np.where(Rg >= a, field.max_abs_principal_strain(np.maximum(Rg, a)), np.nan)
    im = inset.pcolormesh(Xg, Yg, strain, shading="auto", cmap="inferno_r",
                          norm=LogNorm(vmin=primary_threshold.value * 0.5,
                                       vmax=field.max_abs_principal_strain(a)))
    inset.add_patch(plt.Circle((cx, cy), a, color="cyan", fill=False, lw=1.4))
    for key, record in strain_records.items():
        inset.add_patch(plt.Circle((cx, cy), field.strain_radius_um(record.value),
                                   fill=False, ec=col[key], lw=1.2, ls="--"))
    inset.text(0.03, 0.955, "zoom 40x40 um: ASSUMED strain field; cyan = shaft,\n"
               "dashed = strain-threshold crossing", transform=inset.transAxes,
               va="top", ha="left", fontsize=6.2,
               bbox=dict(fc="white", ec="none", alpha=.75, pad=1.2))
    inset.set(xlim=(cx - box / 2, cx + box / 2), ylim=(cy - box / 2, cy + box / 2))
    inset.tick_params(labelleft=False, labelbottom=False, length=2)
    cb = fig.colorbar(im, ax=inset, fraction=0.046, pad=0.04)
    cb.set_label("strain (1)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    # ---- (0,1) sagittal section: the thickness argument -------------------
    ax = fig.add_subplot(grid[0, 1])
    ax.add_patch(plt.Rectangle((0, 0), geometry.length_um, geometry.thickness_um,
                               facecolor="0.93", ec="k", lw=1.4))
    for row, colour in zip(transfer["rows"], ("tab:green", "tab:purple", "tab:brown", "tab:gray")):
        r = row["radius_record"]["value"]
        ax.add_patch(plt.Circle((cx, geometry.thickness_um / 2), r, fill=False,
                                ec=colour, lw=1.3))
    ax.add_patch(plt.Rectangle((cx - a, 0), 2 * a, geometry.thickness_um,
                               facecolor="cyan", alpha=.8))
    ax.annotate("rodent 100 um RADIUS = 2.0x the ENTIRE\n100 um fly brain thickness",
                xy=(cx + 71, 62), xytext=(40, 82), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("the same 100 um in a 900 um cortex spans\nonly 22% of ITS thickness (factor 9)",
                xy=(cx + 100, 30), xytext=(38, 12), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=1))
    ax.set(xlim=(-10, 510), ylim=(-5, 105), xlabel="x (um)", ylabel="depth z (um)",
           title="(b) Sagittal section (500x100 um): why a rodent radius\nCANNOT transfer to a 100 um-thick fly brain")
    ax.grid(alpha=.2)

    # ---- (0,2) strain field vs distance, with the thresholds --------------
    ax = fig.add_subplot(grid[0, 2])
    ax.loglog(profile["r_um"], profile["max_abs_principal_strain"], "k-", lw=1.8,
              label="|eps| = delta*a/r^2 (ASSUMED)")
    for key, record in strain_records.items():
        rr = field.strain_radius_um(record.value)
        ax.axhline(record.value, color=col[key], ls="--", lw=1.2,
                   label=f"{record.value:g} strain, r={rr:.2f} um [{record.species.split(' ')[0]} proxy]")
        ax.plot([rr], [record.value], "o", color=col[key], ms=4)
    r_shear = field.shear_stress_radius_um(shear_record.value)
    ax.axvline(r_shear, color="tab:cyan", ls=":", lw=1.8,
               label=f"140 dyn/cm2 shear proxy, r={r_shear:.1f} um [rat, NO death]")
    ax.axvline(a, color="gray", lw=1.2, label=f"shaft wall r=a={a:g} um")
    ax.set(xlabel="distance from shaft axis r (um)", ylabel="max |principal strain| (1)",
           ylim=(1e-4, 1.2),
           title="(c) ASSUMED field + CROSS-SPECIES thresholds\n(no measured strain field around a probe exists)")
    ax.legend(loc="lower left", fontsize=6.8)
    ax.grid(alpha=.2, which="both")

    # ---- (0,3) why rodent radii cannot transfer ---------------------------
    ax = fig.add_subplot(grid[0, 3])
    labels, pct, mouse_frac = [], [], []
    short = {50.0: "0-50 um\nneuron-loss", 100.0: "100 um\ncited 'kill zone'",
             129.0: "129 um\nGFAP FWHM", 150.0: "150 um\ndegeneration"}
    for row in transfer["rows"]:
        r = row["radius_record"]["value"]
        labels.append(f"{r:g} um\n{short.get(r, 'unnamed').splitlines()[-1]}")
        pct.append(row["fly_brain_volume_percent"])
        mouse_frac.append(100 * row["reference_cortical_thickness_fraction_2R_over_900um"])
    bars = ax.bar(labels, pct, color="tab:red", alpha=.8, label="% of FLY brain destroyed by one track")
    ax.plot(range(len(pct)), mouse_frac, "o-", color="tab:blue",
            label="same radius: % of MOUSE cortical thickness (900 um)")
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.8, f"{p:.1f}%", ha="center", fontsize=8)
    ax.axhline(100, color="k", ls="--", lw=1)
    ax.set(ylabel="percent (%)", ylim=(0, 108),
           title="(d) Rodent radii cannot transfer:\nthe SAME number is 9x worse in the fly (transfer factor 9.0)")
    ax.tick_params(axis="x", labelsize=6.8)
    ax.legend(loc="upper left", fontsize=6.4, framealpha=.95)
    ax.grid(alpha=.2, axis="y")

    # ---- (1,0) damage radius vs probe diameter ---------------------------
    ax = fig.add_subplot(grid[1, 0])
    for key, record in strain_records.items():
        ax.loglog(diameter_curve[key]["probe_diameter_um"],
                  diameter_curve[key]["damage_radius_um"], "-o", ms=3.5, color=col[key],
                  label=f"strain {record.value:g} (guinea-pig proxy)")
    ax.loglog(PROBE_DIAMETERS_UM, shear_radii, ":s", ms=3.5, color="tab:cyan",
              label="140 dyn/cm2 shear proxy (rat, no death)")
    ax.loglog(PROBE_DIAMETERS_UM, affine_radii, "--", color="0.4",
              label="affine alternative r0+0.5a (ASSUMED)")
    ax.loglog(PROBE_DIAMETERS_UM, PROBE_DIAMETERS_UM / 2, "-", color="gray", lw=1,
              label="shaft radius a = d/2 (floor)")
    ax.axhline(50.0, color="tab:green", lw=1.4, ls="-.",
               label="mouse 0-50 um neuron-loss zone (CROSS-SPECIES, mouse)")
    for x, y, txt in ((1.0, 3.2, "insect tungsten 1 um tip"), (10.0, 0.62, "insect protease pipette ~10 um")):
        ax.plot([x], [y], "k*", ms=9)
        ax.annotate(txt, (x, y), xytext=(4, -12), textcoords="offset points", fontsize=7)
    ax.set(xlabel="probe diameter d (um)", ylabel="damage radius (um)", ylim=(0.5, 120),
           title="(e) Damage radius vs electrode diameter\nno insect scaling law exists: one is ASSUMED")
    ax.legend(loc="lower right", fontsize=6.0, ncol=2)
    ax.grid(alpha=.2, which="both")

    # ---- (1,1) damage as a FRACTION of the fly brain ---------------------
    ax = fig.add_subplot(grid[1, 1])
    for key, record in strain_records.items():
        ax.loglog(diameter_curve[key]["probe_diameter_um"],
                  100 * diameter_curve[key]["fly_brain_fraction"], "-o", ms=3.5,
                  color=col[key], label=f"sheath REMOVED, strain {record.value:g}")
    ax.loglog(PROBE_DIAMETERS_UM,
              100 * diameter_curve["optimal"]["fly_brain_fraction_sheath_intact"],
              "-^", ms=4, color="tab:orange", alpha=.55,
              label="sheath INTACT + breach herniation (x5 ASSUMED)")
    for row, colour in zip(transfer["rows"], ("tab:green", "tab:purple", "tab:brown", "tab:gray")):
        r = row["radius_record"]["value"]
        ax.axhline(row["fly_brain_volume_percent"], color=colour, ls=":", lw=1.2,
                   label=f"rodent {r:g} um => {row['fly_brain_volume_percent']:.1f}%")
    ax.set(xlabel="probe diameter d (um)", ylabel="damage as % of the fly brain",
           ylim=(3e-4, 4e2),
           title="(f) Damage reported as a FRACTION of the fly brain,\nnever as a rodent radius")
    ax.legend(loc="lower right", fontsize=5.8, ncol=2)
    ax.grid(alpha=.2, which="both")

    # ---- (1,2) sheath breach criterion sweep -----------------------------
    ax = fig.add_subplot(grid[1, 2])
    diameters = np.geomspace(0.2, 100.0, 200)
    for c in SHEATH_CRITICAL_UM:
        breached = np.array([CavityExpansionField(d / 2.0).shaft_radius_um >= c for d in diameters])
        ax.plot(diameters, breached.astype(float) * c, lw=1.6,
                label=f"critical opening radius {c:g} um (FREE parameter)")
    ax.axhspan(adopt("insect_lamella_thickness_cockroach_um_lo").value,
               adopt("insect_lamella_thickness_cockroach_um_hi").value,
               color="tab:orange", alpha=.18,
               label="cockroach lamella 2-5 um (MEASURED, CROSS-SPECIES)")
    ax.axvline(primary_diameter, color="k", ls="--", lw=1.2,
               label=f"primary demo probe d={primary_diameter:g} um")
    ax.axvline(2 * adopt("insect_tungsten_tip_diameter_um").value, color="gray", ls=":", lw=1.2,
               label="insect tungsten tip d=2 um")
    ax.set(xscale="log", xlabel="probe diameter d (um)", ylabel="breach criterion value (um)",
           ylim=(0, 13),
           title="(g) SHEATH breach: a step in diameter, not a radius\nstretch criterion is SIZE-INDEPENDENT (degenerate)")
    ax.legend(loc="upper left", fontsize=6.0, ncol=2, framealpha=.95)
    ax.grid(alpha=.2, which="both")

    # ---- (1,3) free-parameter sensitivity --------------------------------
    ax = fig.add_subplot(grid[1, 3])
    for name, frac in free_sweep.items():
        ax.loglog(SWEEP_RADII_UM, 100 * frac, "-o", ms=4,
                  label=f"{name} (x{variants[name].herniation_multiplier:g} if breached)")
    for name, est in primary_estimates.items():
        r = est["parenchymal_radius_mechanistic_um"]
        ax.plot([r], [100 * est["parenchymal_fraction_mechanistic"]], "k*", ms=11,
                label=f"{name}: mechanistic r={r:.2f} um -> "
                      f"{100 * est['parenchymal_fraction_mechanistic']:.3f}%")
    ax.axvspan(adopt("insect_lamella_thickness_cockroach_um_lo").value,
               adopt("insect_lamella_thickness_cockroach_um_hi").value,
               color="tab:orange", alpha=.18, label="cockroach lamella 2-5 um (cross-species)")
    ax.set(xlabel="ASSUMED free parenchymal damage radius (um)",
           ylabel="damage as % of the fly brain", ylim=(1e-3, 3e2),
           title="(h) The parenchymal radius is an UNMEASURED free parameter:\n1-3 orders of magnitude, never fitted")
    ax.legend(loc="lower right", fontsize=6.0)
    ax.grid(alpha=.2, which="both")

    # ---- (2,0) paired cable result ---------------------------------------
    ax = fig.add_subplot(grid[2, 0])
    idx = cable["damaged_node_indices"]
    ax.plot(cable_trace["x_um"], cable_trace["sham_mV"], "-", color="tab:blue", lw=1.8, label="sham (no damage)")
    ax.plot(cable_trace["x_um"], cable_trace["inserted_mV"], "-", color="tab:red", lw=1.8,
            label=f"inserted (r_dmg={cable['damage_radius_um']:.2f} um, {idx.size} nodes)")
    ax.axvspan(cable_trace["x_um"][idx].min(), cable_trace["x_um"][idx].max(),
               color="tab:red", alpha=.12, label="damaged compartments")
    ax.axhline(cable["E_leak_mV"], color="gray", lw=1, ls="-.",
               label=f"E_leak = {cable['E_leak_mV']:.1f} mV")
    peak = int(np.argmax(np.abs(cable_trace["delta_mV"])))
    ax.text(0.02, 0.03,
            f"max |dV| = {cable['max_abs_delta_mV']:.2f} mV at x={cable_trace['x_um'][peak]:.0f} um; "
            f"local attenuation {100 * cable['max_local_attenuation_fraction']:.0f}%\n"
            f"of the sham deflection.  Shunt leak density "
            f"{adopt('assumed_damaged_leak_density_S_cm2').value:g} S/cm2 (ASSUMED)",
            transform=ax.transAxes, fontsize=6.8, va="bottom", ha="left",
            bbox=dict(fc="white", ec="0.6", alpha=.9, pad=2.2))
    ax.set(xlabel="position along the passive neurite (um)", ylabel="Vm at 50 ms (mV)",
           title="(i) Paired coupling: extra end leak at damaged compartments\n(via CableNeuron.set_end_leak; cable.py unmodified)")
    ax2 = ax.twinx()
    ax2.plot(cable_trace["x_um"], np.abs(cable_trace["delta_mV"]), ":", color="k", lw=1.6,
             label="|dV| = sham - inserted")
    ax2.set_ylabel("|dV| (mV)", fontsize=9)
    ax2.tick_params(labelsize=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.6, framealpha=.95)
    ax.grid(alpha=.2)

    # ---- (2,1) paired chemistry: K ---------------------------------------
    ax = fig.add_subplot(grid[2, 1])
    t = chem_arrays["time_s"]
    ax.semilogy(t, chem_arrays["sheath_intact_closed_sham_Ke_mM"], "-", color="tab:blue", lw=1.6,
                label="sham, sheath intact")
    ax.semilogy(t, chem_arrays["sheath_intact_closed_inserted_Ke_mM"], "-", color="tab:red", lw=1.6,
                label=f"inserted, sheath intact (p={chemistry['sheath_intact']['variants']['closed']['inserted_permeability_um_s']:.3g} um/s)")
    ax.semilogy(t, chem_arrays["sheath_intact_supported_sham_Ke_mM"], "--", color="tab:blue", lw=1.4,
                label="sham, supported (buffer+reservoir)")
    ax.semilogy(t, chem_arrays["sheath_intact_supported_inserted_Ke_mM"], "--", color="tab:red", lw=1.4,
                label="inserted, supported")
    ax.set(xlabel="time (s)", ylabel="extracellular K (mM)",
           title="(j) Paired chemistry: K leak changed through\nInjuryPotassium.apply_injury (closed run is MODEL-LIMITED)")
    ax.legend(loc="center right", fontsize=6.4, framealpha=.96)
    ax.grid(alpha=.2, which="both")

    # ---- (2,2) paired chemistry: ligand across the sheath ----------------
    ax = fig.add_subplot(grid[2, 2])
    ax.plot(t, chem_arrays["sheath_intact_closed_inserted_far_ligand_nM"], "-", color="tab:red", lw=1.8,
            label="sheath INTACT, inserted (breached)")
    ax.plot(t, chem_arrays["sheath_intact_closed_sham_far_ligand_nM"], "-", color="tab:blue", lw=1.8,
            label="sheath INTACT, sham (exactly 0)")
    ax.plot(t, chem_arrays["sheath_removed_closed_inserted_far_ligand_nM"], ":", color="tab:red", lw=1.8,
            label="sheath REMOVED, inserted (== sham, exactly)")
    ax.plot(t, chem_arrays["sheath_removed_closed_sham_far_ligand_nM"], "--", color="tab:green", lw=1.4,
            label="sheath REMOVED, sham (restriction already gone)")
    ax.set(xlabel="time (s)", ylabel="ligand at the far voxel (nM)",
           title="(k) SHEATH-INTACT vs SHEATH-REMOVED preparation\n(most fly recordings remove the sheath first)")
    ax.legend(loc="upper left", fontsize=6.8)
    ax.grid(alpha=.2)

    # ---- footer: mandatory honesty statements + headline numbers ----------
    ax_h = fig.add_subplot(footer[0])
    ax_h.axis("off")
    honesty_lines = ["MANDATORY HONESTY STATEMENTS (on the figure, as required)"]
    honesty_lines += [f"*  {s}" for s in HONESTY_STATEMENTS]
    ax_h.text(0.0, 1.0, "\n".join(wrap_lines(honesty_lines, width=150)), va="top",
              ha="left", fontsize=8.6, transform=ax_h.transAxes)

    ax_n = fig.add_subplot(footer[1])
    ax_n.axis("off")
    number_lines = [
        "HEADLINE NUMBERS -- every input is ASSUMED or a CROSS-SPECIES proxy; none is a fly measurement",
        f"*  fly brain {geometry.length_um:g}x{geometry.width_um:g}x{geometry.thickness_um:g} um "
        f"(ILLUSTRATIVE, no primary citation); volume {geometry.volume_um3:.3g} um^3",
        f"*  mechanistic r_dmg = {cable['damage_radius_um']:.2f} um for d={primary_diameter:g} um at strain "
        f"{primary_threshold.value:g} -> {100 * primary_estimates['sheath_removed']['parenchymal_fraction_mechanistic']:.3f}% of the fly brain",
        f"*  same probe, sheath INTACT and breached -> "
        f"{100 * primary_estimates['sheath_intact']['parenchymal_fraction_mechanistic']:.2f}% "
        f"(x{variants['sheath_intact'].herniation_multiplier:g} ASSUMED herniation multiplier)",
        f"*  rodent 100 um radius -> {transfer['rows'][1]['fly_brain_volume_percent']:.1f}% of the fly brain vs "
        f"{100 * transfer['rows'][1]['reference_cortical_thickness_fraction_2R_over_900um']:.1f}% of a 900 um cortical "
        f"thickness: transfer factor {transfer['rows'][1]['transfer_factor']:.1f}x",
        f"*  sheath breach at d={primary_diameter:g} um: "
        f"{primary_estimates['sheath_intact']['sheath_breached']} with criterion "
        f"{sheath.critical_opening_radius_um:g} um (FREE parameter; cockroach lamella 2-5 um is the only measured scale)",
        f"*  cable pairing: max|dV| = {cable['max_abs_delta_mV']:.2f} mV, local attenuation "
        f"{100 * cable['max_local_attenuation_fraction']:.1f}%, R_in = {cable['input_resistance_Mohm']:.0f} Mohm, "
        f"{idx.size} of {morph.n} compartments",
        f"*  membrane resealing {resealing_record.value:.0f} s (rat) / {transected_record.value:.0f} s (guinea pig) vs "
        f"sheath repair {repair.sheath_repair_tau_s:.3g} s -> {repair.sheath_over_transected_ratio:.0f}x slower",
        f"*  mouse F = {force_intercept.value:g} + {force_slope.value:g}*d mN -> {force_at['15']:.0f} uN at d=15 um; "
        f"mouse compression {compression_at['15']:.0f} um = {compression_at['15'] / geometry.thickness_um:.1f}x the WHOLE fly brain thickness",
    ]
    ax_n.text(0.0, 1.0, "\n".join(wrap_lines(number_lines, width=150)), va="top", ha="left",
              fontsize=8.6, transform=ax_n.transAxes)

    fig.suptitle(
        "ILLUSTRATIVE Drosophila-scale electrode-damage model -- ASSUMED field, CROSS-SPECIES proxy thresholds, NO insect measurement\n"
        "NO quantitative insect electrode-damage measurement exists (two independent searches). No consciousness/viability/medical claim.\n"
        "This model does NOT predict real fly electrode damage; damage is reported as a FRACTION of the fly brain, never as a rodent radius.",
        fontsize=13.5)
    fig.savefig(OUT / "electrode_damage_demo.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------- npz
    np.savez_compressed(
        OUT / "electrode_damage_field.npz",
        r_um=profile["r_um"], displacement_um=profile["displacement_um"],
        hoop_strain=profile["hoop_strain"], radial_strain=profile["radial_strain"],
        max_abs_principal_strain=profile["max_abs_principal_strain"],
        max_shear_strain=profile["max_shear_strain"],
        radial_stress_Pa=profile["radial_stress_Pa"], hoop_stress_Pa=profile["hoop_stress_Pa"],
        probe_diameters_um=PROBE_DIAMETERS_UM,
        damage_radius_optimal_um=diameter_curve["optimal"]["damage_radius_um"],
        damage_radius_conservative_um=diameter_curve["conservative"]["damage_radius_um"],
        damage_radius_permissive_um=diameter_curve["permissive"]["damage_radius_um"],
        damage_radius_shear_proxy_um=shear_radii,
        damage_radius_affine_assumed_um=affine_radii,
        fly_fraction_optimal=diameter_curve["optimal"]["fly_brain_fraction"],
        fly_fraction_conservative=diameter_curve["conservative"]["fly_brain_fraction"],
        fly_fraction_permissive=diameter_curve["permissive"]["fly_brain_fraction"],
        fly_fraction_sheath_intact_optimal=diameter_curve["optimal"]["fly_brain_fraction_sheath_intact"],
        free_radius_sweep_um=SWEEP_RADII_UM,
        free_radius_fraction_sheath_intact=free_sweep["sheath_intact"],
        free_radius_fraction_sheath_removed=free_sweep["sheath_removed"],
        cable_x_um=cable_trace["x_um"], cable_sham_mV=cable_trace["sham_mV"],
        cable_inserted_mV=cable_trace["inserted_mV"], cable_delta_mV=cable_trace["delta_mV"],
        cable_damaged_node_indices=idx,
        **chem_arrays)

    # ------------------------------------------------------------ report
    report = {
        "status": "manually implemented prototype; numerical tests pass, NOT calibrated "
                  "against any insect experiment",
        "scope": __doc__,
        "negative_results": {
            "no_insect_insertion_damage_quantification": NO_INSECT_INSERTION_DAMAGE_QUANTIFICATION,
            "no_drosophila_lamella_measurement": NO_DROSOPHILA_LAMELLA_MEASUREMENT,
            "no_seconds_scale_sheath_resealing": NO_SECONDS_SCALE_SHEATH_RESEALING,
            "no_measured_strain_field_around_probe": "evidence file section 2 item 5: only "
                "normalised FEM output exists; connecting a strain threshold to a model is "
                "an assumption",
            "no_measured_damage_radius_vs_diameter_scaling": "evidence file section 2 item 4: "
                "about 20 data points at one specification each; the only systematic diameter "
                "series measured force/displacement/bleeding, not histology",
        },
        "fly_brain_geometry": {
            "length_um": adopt("fly_brain_length_um").to_dict(),
            "width_um": adopt("fly_brain_width_um").to_dict(),
            "thickness_um": adopt("fly_brain_thickness_um").to_dict(),
            "shape": geometry.to_dict(),
        },
        "provenance_summary": registry.summary(),
        "provenance_tiers": tier_counts(registry),
        "provenance_table_markdown": registry.markdown_table(),
        "species_guard": {
            **guard.record([record for record in strain_records.values()] +
                           [shear_record, resealing_record, transected_record,
                            adopt("rodent_probe_force_slope_mN_per_mm"),
                            adopt("rodent_probe_force_intercept_mN"),
                            adopt("rodent_probe_compress_intercept_mm"),
                            adopt("rodent_probe_compress_slope_mm_per_mm"),
                            adopt("rodent_bleed_absent_diameter_um"),
                            adopt("rodent_bleed_always_diameter_um"),
                            adopt("fly_sheath_thickness_um"),
                            adopt("fly_patch_pipette_outer_diameter_mm")]),
            "default_construction_refused_with": guard_default_refused,
            "which_test_proves_it_fires": "run_electrode_damage_selftest.py::"
                                          "t_guard_refuses_cross_species_measured_import "
                                          "(and ::t_guard_fires_on_other_insects_too for "
                                          "non-Drosophila insects)",
            "which_test_proves_it_does_not_fire_for_same_species":
                "run_electrode_damage_selftest.py::"
                "t_guard_allows_same_species_illustrative_and_measured",
        },
        "assumed_field_model": field.to_dict(),
        "field_model_analytic_limits": field_limit_evidence(field),
        "thresholds": {
            "strain_cross_species_proxies": {k: r.to_dict() for k, r in strain_records.items()},
            "mechanoporation_shear": shear_record.to_dict(),
            "mechanoporation_duration_ms": thresholds["_mechanoporation_duration_ms"].to_dict(),
            "mechanoporation_source_caveat": "the source explicitly states no cell death "
                "occurred at 140 dyn/cm2 for 300 ms; this is a permeabilisation threshold, "
                "NOT a death threshold",
            "resealing_small_pore": resealing_record.to_dict(),
            "resealing_transected": transected_record.to_dict(),
            "resealing_transected_sd": adopt("rodent_reseal_transected_tau_sd_s").to_dict(),
            "which_is_a_proxy_and_why": "the guinea-pig axonal strain thresholds are used as "
                "the membrane/axonal rupture criterion for FLY tissue, and the rat shear "
                "threshold as an optional stress criterion; neither was measured in any "
                "insect, so this transfer is an assumption, not a conversion",
        },
        "sheath_model": {
            "layer": sheath.to_dict(),
            "load_bearing_evidence": SHEATH_LOAD_BEARING_EVIDENCE,
            "fly_thickness_record": adopt("fly_sheath_thickness_um").to_dict(),
            "fly_surface_glia_record": adopt("fly_surface_glia_thickness_um_max").to_dict(),
            "why_no_fly_lamella_value": NO_DROSOPHILA_LAMELLA_MEASUREMENT,
            "measured_cross_species_lamella": {
                "cockroach_lo": adopt("insect_lamella_thickness_cockroach_um_lo").to_dict(),
                "cockroach_hi": adopt("insect_lamella_thickness_cockroach_um_hi").to_dict(),
                "manduca_lo": adopt("insect_lamella_thickness_manduca_um_lo").to_dict(),
                "manduca_hi": adopt("insect_lamella_thickness_manduca_um_hi").to_dict(),
                "locust": adopt("insect_lamella_thickness_locust_um").to_dict(),
                "perineurium_cockroach_lo": adopt("insect_perineurium_thickness_cockroach_um_lo").to_dict(),
                "perineurium_cockroach_hi": adopt("insect_perineurium_thickness_cockroach_um_hi").to_dict(),
            },
            "repair": repair.to_dict(),
            "breach_sweep": {f"{c:g}": [bool(CavityExpansionField(d / 2.0).shaft_radius_um >= c)
                                        for d in (0.5, 1.0, 2.0, 4.0, 7.5, 10.0, 15.0, 25.0, 50.0, 100.0)]
                             for c in SHEATH_CRITICAL_UM},
            "stretch_criterion_degeneracy": "hole-rim hoop strain = delta/a = "
                "wall_displacement_fraction, i.e. size-independent under the assumed "
                "delta ~ a form, so it is reported and NOT used as the primary criterion",
        },
        "preparation_variants": {name: v.to_dict() for name, v in variants.items()},
        "damage_fraction_by_probe": {
            key: {"probe_diameter_um": PROBE_DIAMETERS_UM.tolist(),
                  "damage_radius_um": diameter_curve[key]["damage_radius_um"].tolist(),
                  "fly_brain_fraction": diameter_curve[key]["fly_brain_fraction"].tolist(),
                  "fly_brain_fraction_sheath_intact": diameter_curve[key]
                      ["fly_brain_fraction_sheath_intact"].tolist()}
            for key in strain_records},
        "shear_proxy_radius_um_by_probe": shear_radii.tolist(),
        "affine_alternative": affine.to_dict(),
        "rodent_radius_transfer": transfer,
        "rodent_100um_radius_analysis": rodent_100um_analysis(geometry, transfer, adopt),
        "parenchymal_primary_case": {
            "probe_diameter_um": primary_diameter,
            "strain_criterion": primary_threshold.value,
            "strain_criterion_record": primary_threshold.to_dict(),
            "mechanistic": primary_estimates,
            "free_parameter_radius_um": free_radius,
            "free_parameter": free_estimates,
            "free_parameter_sweep": {name: {"radius_um": SWEEP_RADII_UM.tolist(),
                                            "fly_brain_fraction": frac.tolist()}
                                     for name, frac in free_sweep.items()},
            "units": "um, fraction of the fly brain (1)",
        },
        "electrode_size_dependence_mouse_function_form": {
            "note": "function FORM borrowed cross-species: the linear relations are MOUSE "
                    "measurements over 7.5-100 um tungsten. Only the form is borrowed; no "
                    "mouse value is applied to the fly.",
            "force_intercept": force_intercept.to_dict(),
            "force_slope": force_slope.to_dict(),
            "compression_intercept": compress_intercept.to_dict(),
            "compression_slope": compress_slope.to_dict(),
            "force_fit_uN_at": force_at,
            "compression_fit_um_at": compression_at,
            "source_internal_inconsistency": {
                "fit_vs_reported_15um_rel_error": adopt("fit_vs_reported_force_15um_rel_error").to_dict(),
                "fit_vs_reported_100um_rel_error": adopt("fit_vs_reported_force_100um_rel_error").to_dict(),
                "fit_vs_reported_100um_sd_units": adopt("fit_vs_reported_force_100um_sd_units").to_dict(),
                "note": "the quoted linear fit reproduces the source's own 15 um point to "
                        "0.35% but misses its 100 um point by 35% (2.4 sigma). This "
                        "UNRESOLVED inconsistency is reported, not explained away, and is a "
                        "further reason to use the relation only as a function form.",
            },
            "bleeding_thresholds_mouse": {
                "no_bleeding_at_or_below_um": adopt("rodent_bleed_absent_diameter_um").to_dict(),
                "bleeding_at_or_above_um": adopt("rodent_bleed_always_diameter_um").to_dict(),
                "note": "mouse numbers; the fly has no vertebrate vasculature "
                        "(engine/profile.py DROSOPHILA_CNS), so this row has no fly analogue",
            },
            "compression_vs_fly_thickness": {
                "compression_at_15um_um": compression_at["15"],
                "fly_brain_thickness_um": geometry.thickness_um,
                "ratio": compression_at["15"] / geometry.thickness_um,
                "note": "a naive import of the mouse compression relation would displace "
                        "tissue by almost 3x the ENTIRE fly brain thickness, and by "
                        f"{compression_at['15'] / a:.0f}x the assumed fly shaft radius; the "
                        "linear-elastic small-strain field is invalid there, which is the "
                        "quantitative statement of non-transferability",
            },
        },
        "fly_electrode_sizes": {
            "brain_patch_resistance": [adopt("fly_brain_patch_resistance_Mohm_lo").to_dict(),
                                       adopt("fly_brain_patch_resistance_Mohm_hi").to_dict()],
            "al_mb_resistance": [adopt("fly_al_mb_resistance_Mohm_lo").to_dict(),
                                 adopt("fly_al_mb_resistance_Mohm_hi").to_dict()],
            "vnc_sharp_resistance": [adopt("fly_vnc_sharp_resistance_Mohm_lo").to_dict(),
                                     adopt("fly_vnc_sharp_resistance_Mohm_hi").to_dict()],
            "only_insect_tip_diameters_um": {
                "tungsten_tip": adopt("insect_tungsten_tip_diameter_um").to_dict(),
                "protease_pipette_broken": adopt("insect_protease_pipette_broken_diameter_um").to_dict(),
            },
            "note": "no tip diameter is given for any of the three fly electrode resistances, "
                    "so resistance cannot be converted into a shaft size here; the fly patch "
                    "pipette shank is 1.5 mm OD and its pulled tip size is unmeasured",
            "percent_of_brain_width_record": adopt("rodent_15um_as_percent_of_fly_brain_width").to_dict(),
        },
        "coupling": {
            "membrane": {
                "damaged_node_indices": idx.tolist(),
                "damage_radius_um": cable["damage_radius_um"],
                "g_uS_per_node": float(cable["g_end_uS"][idx][0]),
                "record": cable["membrane_damage_record"],
                "sham_g_end_exactly_zero_proven_by": "selftest t_membrane_coupling_exact "
                    "(np.array_equal(sham.g_end, zeros(n)) is True)",
                "api": "CableNeuron.set_end_leak; cable.py NOT modified",
                "reversal": "shunt at the cable's own E_leak, which isolates the "
                            "conductance effect; a real pore has a nonspecific reversal "
                            "near 0 mV and would also depolarise (unmodelled)",
            },
            "cable_paired_result": {
                "max_abs_delta_mV": cable["max_abs_delta_mV"],
                "local_attenuation_fraction": cable["max_local_attenuation_fraction"],
                "input_resistance_Mohm": cable["input_resistance_Mohm"],
                "drive_nA": cable["drive_nA"],
                "sham_peak_deflection_mV": float(np.abs(cable["sham"][-1] - cable["E_leak_mV"]).max()),
                "inserted_peak_deflection_mV": float(np.abs(cable["inserted"][-1] - cable["E_leak_mV"]).max()),
                "materiality": "MATERIAL for the local compartments (the shunt is ~5000x the "
                               "intact leak density and pins the damaged rows towards E_leak) "
                               "but this is a statement about the ASSUMED leak density, which "
                               "is a free parameter",
                "resealing_applied": cable["resealing_kind"] or "none (static leak)",
                "leak_factor_range_over_window": cable["resealing_factor_range"],
                "units": "mV, MOhm, nA, fraction (1)",
            },
            "chemistry": {
                name: {
                    key: value for key, value in data.items()
                    if key not in ("traces", "inserted_sheath_events", "sham_sheath_events")
                } for name, data in
                ((name, chemistry[name]["variants"]["closed"]) for name in variants)},
            "chemistry_supported_variant": {
                name: {
                    key: value for key, value in chemistry[name]["variants"]["supported"].items()
                    if key not in ("traces", "inserted_sheath_events", "sham_sheath_events")
                } for name in variants},
            "chemistry_damage_fraction_used": chemistry_fraction,
            "chemistry_damage_fraction_definition": "fraction of the 8x8x8 um local tissue "
                "element centred on the track that lies inside the damage cylinder: "
                "pi*r_dmg^2/64 for r_dmg = "
                f"{cable['damage_radius_um']:.2f} um -> finite, hence the rounded 0.7 used",
            "chemistry_model_limit": "the closed variant redistributes K towards a single "
                "equilibrium because engine/injury_tissue.py is a closed two-pool reduction "
                "with no pumps, buffer or reservoir; the extreme Ke is that documented "
                "artifact, which is why the buffered+reservoir 'supported' variant is shown "
                "next to it",
            "optional_proximity_excitability": {
                **CableDamageConfig().excitability.to_dict(),
                "tested_default": "default OFF; the selftest proves a disabled run is "
                                  "bit-for-bit identical to a run with no coupling object",
                "limitation": "with a point injection far from the probe the placeholder "
                              "cannot do anything at all (scale = 1.0 everywhere), which is "
                              "reported as a limitation of a passive point-drive model",
            },
        },
        "species_labelling_audit": {
            "violations": audit_species_labeling({
                "geometry": {"length_um": adopt("fly_brain_length_um").to_dict()},
                "thresholds": {k: r.to_dict() for k, r in strain_records.items()},
                "transfer": transfer,
            }),
            "what_it_checks": "every foreign-species provenance record must set "
                              "cross_species=true and carry a citation; no bare number may "
                              "sit under a rodent/cross-species key",
        },
        "honesty_statements": list(HONESTY_STATEMENTS),
        "selftests": {"n_checks": tests["n_checks"], "n_passed": tests["n_passed"],
                      "n_failed": tests["n_failed"], "by_group": tests["by_group"],
                      "wall_seconds": tests["wall_seconds"]},
        "limitations_not_removed": limitations(),
        "outputs": {
            "report": str(OUT / "electrode_damage_report.json"),
            "field_npz": str(OUT / "electrode_damage_field.npz"),
            "figure": str(OUT / "electrode_damage_demo.png"),
            "selftest": str(OUT / "electrode_damage_selftest.json"),
        },
        "wall_seconds_total": time.perf_counter() - t_start,
        "process_peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    # audit the FULL report, not just a fragment
    full_violations = audit_species_labeling(report)
    assert full_violations == [], full_violations
    report["species_labelling_audit"]["full_report_violations"] = full_violations
    (OUT / "electrode_damage_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: report[k] for k in
                      ("status", "provenance_summary", "provenance_tiers",
                       "wall_seconds_total", "process_peak_RSS_KiB")}, indent=2, default=float))
    print("\nspecies guard refused by default with:", guard_default_refused[:200])
    print("selftests:", tests["n_passed"], "/", tests["n_checks"], "passed")
    print("outputs:", report["outputs"])


def rodent_100um_analysis(geometry, transfer, adopt):
    """The explicit numeric answer to 'what does a rodent-style 100 um radius do
    to a fly brain', stated without over-claiming."""
    row = transfer["rows"][1]
    r = row["radius_record"]["value"]
    return {
        "radius_record": row["radius_record"],
        "radius_um": r,
        "zone_diameter_um": 2 * r,
        "fly_brain_thickness_um": geometry.thickness_um,
        "zone_diameter_over_brain_thickness": 2 * r / geometry.thickness_um,
        "fly_brain_volume_fraction": row["fly_brain_volume_fraction"],
        "fly_brain_volume_percent": row["fly_brain_volume_percent"],
        "fly_footprint_fraction": geometry.footprint_fraction(r),
        "statements": [
            f"a 100 um damage RADIUS has a diameter of {2 * r:.0f} um, i.e. "
            f"{2 * r / geometry.thickness_um:.1f}x the ENTIRE {geometry.thickness_um:g} um "
            f"fly brain thickness: no depth of the brain is spared, so a single scalar "
            f"'damage radius' is the wrong output variable for a fly",
            f"as a full-thickness track it destroys {row['fly_brain_volume_percent']:.1f}% "
            f"of the whole fly brain volume ({geometry.volume_um3:.3g} um^3), from ONE track",
            f"the mouse-reported <=150 um degeneration zone would destroy "
            f"{transfer['rows'][3]['fly_brain_volume_percent']:.1f}% of the entire fly brain",
            "we therefore report damage as a FRACTION of the fly brain and never as a "
            "rodent radius; the numbers above are fly-GEOMETRY arithmetic on a rodent "
            "input, not a fly measurement",
        ],
    }


def sheath_herniation_text(variants):
    return f"{variants['sheath_intact'].herniation_multiplier:g}"


def tier_counts(registry):
    counts = {}
    for record in registry.snapshot().values():
        tier = provenance_tier(record["source"])
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def field_limit_evidence(field):
    """The analytic limits of the assumed field, measured at report time."""
    r = 10.0 * field.shaft_radius_um
    return {
        "u_times_r_constant_um2": field.displacement_um(r) * r,
        "u_over_r_minus_1_over_r_relative": abs(
            (field.displacement_um(r) * r) /
            (field.wall_displacement_um * field.shaft_radius_um) - 1.0),
        "displacement_at_wall_um": field.displacement_um(field.shaft_radius_um),
        "wall_displacement_imposed_um": field.wall_displacement_um,
        "displacement_relative_error_at_wall": abs(
            field.displacement_um(field.shaft_radius_um) / field.wall_displacement_um - 1.0),
        "displacement_at_1e6_um": field.displacement_um(1e6),
        "incompressibility_residual_um2": field.incompressibility_residual_um2(r, 1e-3),
        "lame_equilibrium_residual_relative": abs(
            field.equilibrium_residual_Pa_per_um(r, 1e-3)) /
            (abs(field.radial_stress_Pa(r)) / r),
        "wall_pressure_Pa": field.wall_pressure_Pa,
        "traction_at_wall_Pa": -field.radial_stress_Pa(field.shaft_radius_um),
        "hoop_strain_at_wall": field.hoop_strain(field.shaft_radius_um),
        "elastic_consistency_recovered_wall_displacement_um": (
            field.wall_pressure_Pa * field.shaft_radius_um / (2 * field.shear_modulus_Pa)),
        "note": "these are the field's OWN analytic limits; satisfying them says the "
                "assumed solution is self-consistent, NOT that it is right for fly tissue",
    }


def limitations():
    return [
        NO_INSECT_INSERTION_DAMAGE_QUANTIFICATION,
        NO_DROSOPHILA_LAMELLA_MEASUREMENT,
        NO_SECONDS_SCALE_SHEATH_RESEALING,
        "The strain field is an assumed plane-strain incompressible cavity expansion: no "
        "measured strain field around any inserted probe exists, in any species.",
        "The wall displacement fraction (delta/a) is assumed; it sets the strain at the "
        "shaft wall and therefore the whole damage radius.",
        "The tissue shear modulus used for the mechanoporation proxy is assumed; that "
        "radius scales as sqrt(G), so it is unconstrained.",
        "The parenchymal damage radius is an unmeasured free parameter and is swept, not "
        "fitted; with 1-100 um the answer changes by ~3 orders of magnitude.",
        "The sheath thickness for the fly is assumed (no Drosophila lamella measurement) and "
        "the sheath breach criterion is a free parameter; both are swept.",
        "The herniation multiplier for the breached-sheath preparation is assumed and "
        "uncertain by at least an order of magnitude.",
        "The cable is passive: no channels, no spikes, so 'reduced excitability' cannot be "
        "represented at all; the optional coupling is a drive-scaling placeholder and does "
        "nothing when the injection site is far from the probe.",
        "The chemistry is a closed reduced two-pool salt model: its long-time K "
        "redistribution is a documented model artifact, not fly physiology.",
        "The fly has no vertebrate blood-brain barrier or vasculature, so the mouse "
        "bleeding and BBB rows have no fly analogue and are reported only as context.",
        "The mouse force/displacement fit is internally inconsistent with its own 100 um "
        "reported point (35%, 2.4 sigma) and is used only as a function form.",
        "Surgery damage (sheath removal, collagenase) is not modelled at all, although in a "
        "real fly preparation it may exceed the insertion damage; no insect study quantifies it.",
        "Damage is defined geometrically (a cylinder around the track with a half-height "
        "criterion left implicit); no definition can be validated against insect histology "
        "because none exists.",
        "Everything is a single acute insertion with no repair kinetics on the modelled "
        "timescales: membrane resealing is applied only where its constants were measured.",
    ]


def wrap_lines(lines, width=118):
    out = []
    for line in lines:
        if len(line) <= width:
            out.append(line)
            continue
        words, current = line.split(" "), ""
        for word in words:
            if len(current) + len(word) + 1 > width:
                out.append(current)
                current = "    " + word
            else:
                current = word if not current else current + " " + word
        out.append(current)
    return out


if __name__ == "__main__":
    main()
