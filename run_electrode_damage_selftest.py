"""Executable tests for the illustrative electrode-damage module.

Run:  venv/bin/python run_electrode_damage_selftest.py

What a pass means: the code does what it CLAIMS numerically -- provenance is
enforced, the species guard refuses foreign numbers, the assumed field satisfies
its own analytic limits, thresholds behave at their boundaries, and the couplings
change the existing cable/chemistry models in an exactly accounted way.

What a pass does NOT mean: that any of it is right for a real fly. There is no
insect electrode-damage measurement (two independent searches), no measured
strain field around a probe, and no measured sheath breach threshold. Every
threshold here is a cross-species proxy and every geometry value for the fly is
either illustrative or assumed.
"""
import resource
import time

from dataclasses import replace

import numpy as np

from engine.cable import CableNeuron, Morphology
from engine.electrode_damage import (
    ADDENDUM_TAG, AffineDamageRadius, CableDamageConfig, CavityExpansionField,
    ChemistryDamageConfig, FlyBrainGeometry, HONESTY_STATEMENTS,
    PreparationVariant, ProximityExcitabilityCoupling, ResealingTimescales,
    SheathLayer, SheathRepairModel, SpeciesGuard, STRAIN_THRESHOLD_CHOICES,
    affected_node_indices, apply_chemistry_coupling, apply_membrane_damage,
    audit_species_labeling, build_registry, chemistry_config_for_variant,
    circle_rectangle_overlap_area_um2, FLY_SPECIES, injury_threshold_options,
    parenchymal_damage_estimate, preparation_variants, provenance_tier,
    rodent_radius_transfer_table, run_paired_cable, run_paired_chemistry)
from engine.injury_tissue import InjuryLigand, InjuryPotassium, PotassiumConfig
from engine.params import (ASSUMED, DERIVED, ILLUSTRATIVE, MEASURED, Param,
                            ParamRegistry, ProvenanceError)
from engine.profile import DROSOPHILA_CNS, MAMMALIAN_TISSUE, ProfileViolation

# --------------------------------------------------------------------------
# shared fixtures
# --------------------------------------------------------------------------
REGISTRY = build_registry()
GUARD = SpeciesGuard(DROSOPHILA_CNS, REGISTRY, allow_cross=True)
GEOMETRY = FlyBrainGeometry.from_adopt(GUARD.adopt)
FIELD = CavityExpansionField(shaft_radius_um=5.0)
THRESHOLDS = injury_threshold_options(GUARD)
SHEATH = SheathLayer(
    thickness_um=GUARD.adopt("fly_sheath_thickness_um").value,
    critical_opening_radius_um=GUARD.adopt("assumed_sheath_critical_opening_radius_um").value,
    delamination_multiple=GUARD.adopt("assumed_sheath_delamination_multiple").value)
VARIANTS = preparation_variants(GUARD.adopt)
RESEALING = ResealingTimescales.from_adopt(GUARD.adopt)
CABLE_KWARGS = dict(Cm_uF_cm2=1.0, g_leak_S_cm2=1e-4, Ra_ohm_cm=150.0,
                    E_leak_mV=-65.0)
MORPH = Morphology.cylinder(500.0, 0.5, 250)
SHAFT_POINT = (250.0, 6.0, 0.0)
SHAFT_DIR = (0.0, 0.0, 1.0)


def _paired_cable(**kw):
    kw.setdefault("dt_ms", 0.05)
    kw.setdefault("duration_ms", 50.0)
    kw.setdefault("cable_kwargs", CABLE_KWARGS)
    return run_paired_cable(MORPH, FIELD, SHAFT_POINT, SHAFT_DIR, 0.21, **kw)


def _paired_chemistry(fraction=0.7, variant="sheath_intact", **kw):
    kw.setdefault("dt_s", 0.05)
    kw.setdefault("duration_s", 10.0)
    kw.setdefault("include_supported", True)
    cfg = chemistry_config_for_variant(VARIANTS[variant], GUARD.adopt)
    kw.setdefault("config", cfg)
    return run_paired_chemistry(fraction, **kw)


# --------------------------------------------------------------------------
# 1. provenance enforcement
# --------------------------------------------------------------------------
def t_measured_requires_citation():
    try:
        Param("unlabelled", 1.0, "1", MEASURED)
    except ProvenanceError as exc:
        msg = str(exc)
    else:
        raise AssertionError("a measured parameter without a citation was accepted")
    assert "requires a source citation" in msg
    # and a cited measured parameter is accepted
    p = Param("cited", 1.0, "1", MEASURED, source="some citation", species="Species x")
    assert p.source == "some citation"
    return {"refused_message": msg, "accepted_with_citation": p.name}


def t_derived_requires_registered_parents():
    try:
        Param("orphan", 1.0, "1", DERIVED)
    except ProvenanceError as exc:
        assert "derived_from" in str(exc)
    else:
        raise AssertionError("a derived parameter without parents was accepted")
    r = ParamRegistry("t")
    r.add(Param("root", 2.0))
    r.add(Param("child", 4.0, "1", DERIVED, derived_from=["root"]))
    try:
        r.validate()
    except ProvenanceError:
        raise AssertionError("valid derivation rejected")
    r2 = ParamRegistry("t2")
    r2.add(Param("child", 4.0, "1", DERIVED, derived_from=["never_registered"]))
    try:
        r2.validate()
    except ProvenanceError as exc:
        assert "unregistered parameter" in str(exc)
    else:
        raise AssertionError("derivation from an unregistered parent was accepted")
    return {"validated": True, "phantom_parent_refused": True}


def t_illustrative_and_range_checks():
    p = Param("guess", 0.5, "1", ASSUMED, range=(0.1, 1.0))
    assert p.provenance == ASSUMED
    try:
        Param("out_of_range", 5.0, "1", ASSUMED, range=(0.1, 1.0))
    except ProvenanceError:
        pass
    else:
        raise AssertionError("a value outside its stated range was accepted")
    try:
        Param("bad_tag", 1.0, "1", "guessed")
    except ProvenanceError:
        pass
    else:
        raise AssertionError("an unknown provenance tag was accepted")
    return {"assumed_accepted": True, "summary": REGISTRY.summary()}


def t_unregistered_parameter_refused():
    try:
        GUARD.adopt("rodent_kill_zone_radius_um")
    except KeyError as exc:
        assert "undeclared parameter" in str(exc)
    else:
        raise AssertionError("an undeclared parameter name was accepted")
    try:
        REGISTRY["rodent_kill_zone_radius_um"]
    except KeyError:
        pass
    else:
        raise AssertionError("registry returned an unregistered value")
    return {"refused": "rodent_kill_zone_radius_um"}


def t_registry_is_valid_and_tiered():
    REGISTRY.validate()
    snap = REGISTRY.snapshot()
    tiers = {}
    for d in snap.values():
        tier = provenance_tier(d["source"])
        tiers[tier] = tiers.get(tier, 0) + 1
    # every measured/derived parameter must carry a source
    bad = [n for n, d in snap.items() if d["provenance"] == MEASURED and not d["source"]]
    assert not bad, bad
    orphans = [n for n, d in snap.items() if d["provenance"] == DERIVED
               and not d["derived_from"]]
    assert not orphans, orphans
    # a derived record still reaches the report with a resolvable citation trace
    derived = GUARD.adopt("rodent_reseal_transected_tau_s")
    assert derived.source and "derived from" in derived.source
    # addendum-sourced parameters are tagged; evidence-file ones are not tagged
    addendum = sorted(n for n, d in snap.items()
                      if provenance_tier(d["source"]) == "evidence_file_section_2b")
    assert addendum, "no addendum-sourced parameters registered"
    for n in addendum:
        assert ADDENDUM_TAG in snap[n]["source"]
    evid = sorted(n for n, d in snap.items()
                  if provenance_tier(d["source"]) == "evidence_file_primary")
    assert evid and not any(ADDENDUM_TAG in snap[n]["source"] for n in evid)
    # and the addendum strings must NOT claim the data is absent from the file
    assert not any("not in EXTERNAL_EVIDENCE_TOUCH_ELECTRODE" in snap[n]["source"]
                   for n in addendum)
    return {"n_parameters": len(REGISTRY), "tiers": tiers,
            "summary": REGISTRY.summary(),
            "n_addendum_sourced": len(addendum), "n_evidence_file_sourced": len(evid)}


def t_no_drosophila_lamella_tagged_measured():
    snap = REGISTRY.snapshot()
    offenders = [n for n, d in snap.items()
                 if "lamella" in n and d["provenance"] == MEASURED
                 and d["species"] == FLY_SPECIES]
    assert not offenders, f"Drosophila lamella tagged measured: {offenders}"
    # the fly sheath thickness must be an assumption, and it must be flagged as such
    fly = GUARD.adopt("fly_sheath_thickness_um")
    assert fly.provenance == ASSUMED and "NOT measured" in fly.note
    # measured lamella values exist only for other insect species
    measured_lamella = sorted(n for n, d in snap.items()
                              if "lamella" in n and d["provenance"] == MEASURED)
    species = sorted({snap[n]["species"] for n in measured_lamella})
    assert species and all(s != FLY_SPECIES for s in species)
    return {"fly_sheath_provenance": fly.provenance, "fly_sheath_value_um": fly.value,
            "measured_lamella_species": species}


def t_report_audit_detects_unlabelled_rodent_numbers():
    clean = {"geometry": {"value": 100.0, "provenance": ILLUSTRATIVE,
                          "species": FLY_SPECIES, "source": None, "cross_species": False},
             "thresh": GUARD.adopt("rodent_axonal_strain_optimal").to_dict()}
    assert audit_species_labeling(clean) == []
    poisoned = []
    a = GUARD.adopt("rodent_axonal_strain_optimal").to_dict()
    a["cross_species"] = False
    poisoned.append(({"thresh": a}, "cross_species=False"))
    b = GUARD.adopt("rodent_axonal_strain_optimal").to_dict()
    b["source"] = None
    poisoned.append(({"thresh": b}, "source=None"))
    poisoned.append(({"rodent_strain_threshold": 0.21}, "bare number under a rodent key"))
    poisoned.append(({"reseal": {"value": 1200.0, "provenance": MEASURED,
                                 "species": "Cavia porcellus (guinea pig)"}},
                     "measured without citation"))
    detected = {}
    for report, label in poisoned:
        found = audit_species_labeling(report)
        assert found, f"audit missed {label}"
        detected[label] = found[0]
    return {"clean_report_violations": 0, "poisoned_cases_detected": detected}


# --------------------------------------------------------------------------
# 2. species guard  (THE GUARD EVIDENCE)
# --------------------------------------------------------------------------
def t_guard_refuses_cross_species_measured_import():
    try:
        SpeciesGuard(DROSOPHILA_CNS, REGISTRY, allow_cross=False)
    except ProvenanceError as exc:
        message = str(exc)
    else:
        raise AssertionError("the guard did NOT fire on a cross-species import")
    assert "another species" in message
    offenders = REGISTRY.check_species(FLY_SPECIES, allow_cross=True)
    names = sorted(n for n, _ in offenders)
    for must in ("rodent_axonal_strain_optimal", "rodent_mechanoporation_shear_Pa",
                 "rodent_probe_force_slope_mN_per_mm", "rodent_reseal_transected_tau_s"):
        assert must in names, must
    return {"test": "species_guard_refuses_cross_species_measured_import",
            "raised": "ProvenanceError",
            "n_offenders": len(names),
            "message_head": message[:160]}


def t_guard_allows_same_species_illustrative_and_measured():
    # an illustrative SAME-species parameter must not fire
    fly = GUARD.adopt("fly_brain_thickness_um")
    assert fly.cross_species is False and fly.forced_flag is False
    # a measured SAME-species parameter must not fire either
    pip = GUARD.adopt("fly_patch_pipette_outer_diameter_mm")
    assert pip.cross_species is False and pip.provenance == MEASURED and pip.source
    vnc = GUARD.adopt("fly_vnc_sharp_resistance_Mohm_hi")
    assert vnc.cross_species is False and vnc.provenance == MEASURED
    # an assumed parameter is never flagged
    assumed = GUARD.adopt("assumed_wall_displacement_fraction")
    assert assumed.cross_species is False and assumed.provenance == ASSUMED
    offenders = dict(REGISTRY.check_species(FLY_SPECIES, allow_cross=True))
    for name in ("fly_brain_thickness_um", "fly_patch_pipette_outer_diameter_mm",
                 "assumed_wall_displacement_fraction"):
        assert name not in offenders, f"{name} was wrongly flagged"
    return {"not_flagged": ["fly_brain_thickness_um (illustrative, Drosophila)",
                            "fly_patch_pipette_outer_diameter_mm (measured, Drosophila)",
                            "fly_vnc_sharp_resistance_Mohm_hi (measured, Drosophila)",
                            "assumed_wall_displacement_fraction (assumed)"],
            "n_offenders_unchanged": len(offenders)}


def t_guard_forces_flag_when_opted_in():
    adopted = [GUARD.adopt(n) for n in
               ("rodent_axonal_strain_optimal", "rodent_reseal_transected_tau_s",
                "rodent_mechanoporation_shear_Pa")]
    for a in adopted:
        assert a.cross_species and a.forced_flag and a.source
    rec = GUARD.record(adopted)
    assert rec["allow_cross"] is True and rec["guard_default_would_refuse"] is True
    assert len(rec["foreign_parameters_used"]) == 3
    assert any("跨物种" in c or "CROSS-SPECIES" in c for c in rec["caveats"])
    # an unfixed record (flag stripped) is refused by the record checker
    stripped = GUARD.adopt("rodent_axonal_strain_optimal")
    bad = replace(stripped, forced_flag=False)
    try:
        GUARD.adopted_record(bad)
    except ProvenanceError:
        pass
    else:
        raise AssertionError("a cross-species record without its forced flag passed")
    return {"forced_flags": [a.name for a in adopted],
            "record": {"foreign_parameters_used": rec["foreign_parameters_used"],
                       "caveats": rec["caveats"]}}


def t_guard_fires_on_other_insects_too():
    """'Insect' is not 'Drosophila': cockroach/Manduca/locust lamella values fire."""
    offenders = dict(REGISTRY.check_species(FLY_SPECIES, allow_cross=True))
    insect_foreign = sorted(n for n in offenders if n.startswith("insect_"))
    assert len(insect_foreign) >= 10, insect_foreign
    for name in ("insect_lamella_thickness_cockroach_um_lo",
                 "insect_lamella_thickness_manduca_um_hi",
                 "insect_lamella_thickness_locust_um"):
        a = GUARD.adopt(name)
        assert a.cross_species and a.forced_flag
        assert a.species != FLY_SPECIES
    return {"n_insect_but_not_drosophila_offenders": len(insect_foreign),
            "examples": insect_foreign[:4]}


def t_guard_rejects_non_fly_profile():
    try:
        SpeciesGuard(MAMMALIAN_TISSUE, REGISTRY, allow_cross=True)
    except ProfileViolation as exc:
        assert "Drosophila only" in str(exc)
    else:
        raise AssertionError("a mammalian profile was accepted by the fly module")
    try:
        SpeciesGuard("Drosophila melanogaster", REGISTRY, allow_cross=True)
    except TypeError:
        pass
    else:
        raise AssertionError("a non-profile object was accepted")
    return {"refused": ["MAMMALIAN_TISSUE", "a bare species string"]}


# --------------------------------------------------------------------------
# 3. geometry sanity
# --------------------------------------------------------------------------
def t_circle_area_matches_grid_crosscheck():
    for radius, (cx, cy) in ((100.0, (250.0, 150.0)), (60.0, (250.0, 150.0)),
                             (200.0, (250.0, 150.0)), (30.0, (20.0, 280.0))):
        analytic = circle_rectangle_overlap_area_um2(radius, cx, cy, 500.0, 300.0)
        step = 0.25
        xs = np.arange(step / 2, 500.0, step)
        ys = np.arange(step / 2, 300.0, step)
        inside = ((xs[:, None] - cx) ** 2 + (ys[None, :] - cy) ** 2) <= radius ** 2
        grid = inside.sum() * step * step
        assert abs(grid - analytic) / max(analytic, 1.0) < 0.01, (radius, analytic, grid)
    assert circle_rectangle_overlap_area_um2(0.0, 250, 150, 500, 300) == 0.0
    assert abs(circle_rectangle_overlap_area_um2(1e4, 250, 150, 500, 300) - 150000.0) < 1e-6
    assert circle_rectangle_overlap_area_um2(10.0, -100.0, 150.0, 500, 300) == 0.0
    return {"analytic_vs_grid_relative_errors_checked": 4,
            "full_coverage_area_um2": circle_rectangle_overlap_area_um2(1e4, 250, 150, 500, 300)}


def t_fly_scaling_and_rodent_transfer():
    assert GEOMETRY.volume_um3 == 500 * 300 * 100
    r100 = GEOMETRY.damage_fraction(100.0)
    assert abs(r100 - 0.20943951023931953) < 1e-15
    assert abs(GEOMETRY.thickness_ratio(100.0) - 2.0) < 1e-15
    table = rodent_radius_transfer_table(GEOMETRY, GUARD.adopt, 900.0)
    for row in table["rows"]:
        assert abs(row["transfer_factor"] - 9.0) < 1e-12
        assert row["radius_record"]["cross_species"] is True
        assert row["radius_record"]["source"]
    pct = [row["fly_brain_volume_percent"] for row in table["rows"]]
    assert pct == sorted(pct) and pct[1] > 20.0 and pct[-1] > 47.0
    # a naive rodent-style 100 um radius spans 2x the whole fly thickness
    assert "cannot be transferred" in table["conclusion"]
    return {"fly_brain_volume_um3": GEOMETRY.volume_um3,
            "fraction_at_100um_radius": r100,
            "thickness_ratio_at_100um": GEOMETRY.thickness_ratio(100.0),
            "fly_volume_percent_by_rodent_radius": {
                f"{row['radius_record']['value']:g}": row["fly_brain_volume_percent"]
                for row in table["rows"]},
            "transfer_factor": 9.0}


def t_damage_monotone_in_diameter():
    radii = np.linspace(0.0, 260.0, 60)
    fractions = np.array([GEOMETRY.damage_fraction(r) for r in radii])
    assert np.all(np.diff(fractions) >= -1e-15), "damage fraction is not monotone"
    volumes = np.array([GEOMETRY.track_damage_volume_um3(r) for r in radii])
    assert np.all(np.diff(volumes) >= -1e-9)
    assert volumes[0] == 0.0 and fractions[0] == 0.0
    assert fractions.max() <= 1.0
    assert abs(GEOMETRY.damage_fraction(1e4) - 1.0) < 1e-12
    # the mechanistic radius is monotone in shaft diameter too
    radii_dmg = [CavityExpansionField(shaft_radius_um=a).strain_radius_um(0.21)
                 for a in (0.5, 1.0, 2.5, 5.0, 20.0)]
    assert all(b > a for a, b in zip(radii_dmg[:-1], radii_dmg[1:]))
    # and the affine alternative is monotone but does NOT vanish at zero radius
    aff = AffineDamageRadius(offset_um=20.0, slope=0.5)
    assert aff.radius_um(0.0) == 20.0 and aff.radius_um(5.0) == 22.5
    return {"fraction_at_radii": {f"{r:g}": GEOMETRY.damage_fraction(r)
                                  for r in (0.0, 1.0, 10.0, 50.0, 100.0, 500.0)},
            "mechanistic_radius_um_by_shaft_radius": dict(zip((0.5, 1.0, 2.5, 5.0, 20.0), radii_dmg)),
            "affine_defect_at_zero_radius_um": aff.radius_um(0.0)}


def t_geometry_depth_and_axis_validation():
    full = GEOMETRY.track_damage_volume_um3(10.0, depth_um=100.0)
    part = GEOMETRY.track_damage_volume_um3(10.0, depth_um=50.0)
    assert part > full / 2.0        # hemisphere cap adds volume
    try:
        GEOMETRY.track_damage_volume_um3(10.0, depth_um=200.0)
    except ValueError as exc:
        assert "exceeds the fly brain thickness" in str(exc)
    else:
        raise AssertionError("a track longer than the tissue thickness was accepted")
    for bad in (0.0, -5.0):
        try:
            GEOMETRY.track_damage_volume_um3(10.0, depth_um=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"non-positive depth {bad} accepted")
    try:
        GEOMETRY.damage_fraction(-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative radius accepted")
    return {"full_thickness_um3": full, "partial_depth_um3": part}


# --------------------------------------------------------------------------
# 4. analytic limits of the assumed field
# --------------------------------------------------------------------------
def t_field_1_over_r_and_decay():
    r = np.array([5.0, 10.0, 25.0, 100.0, 1000.0, 1e6])
    u = FIELD.displacement_um(r)
    products = u * r
    assert np.allclose(products, products[0], rtol=1e-14, atol=0)
    assert abs(products[0] - FIELD.wall_displacement_um * FIELD.shaft_radius_um) < 1e-12
    assert np.all(np.diff(u) < 0)                     # strictly decreasing
    assert u[-1] / u[0] < 1e-5                        # -> 0 at large r (relative)
    strains = FIELD.max_abs_principal_strain(r)
    assert np.all(np.diff(strains) < 0)               # decays monotonically
    assert FIELD.max_abs_principal_strain(1e8) < 1e-13
    return {"u_times_r_um2": float(products[0]),
            "u_um_at": {f"{x:g}": float(y) for x, y in zip(r[:4], u[:4])},
            "strain_at_1e8_um": FIELD.max_abs_principal_strain(1e8)}


def t_field_wall_continuity_and_strain():
    a = FIELD.shaft_radius_um
    assert abs(FIELD.displacement_um(a) - FIELD.wall_displacement_um) \
        <= 1e-15 * FIELD.wall_displacement_um
    # continuity from both sides of the wall (outside limit is approached)
    eps = [1e-3, 1e-4, 1e-5]
    values = [FIELD.displacement_um(a * (1 + e)) for e in eps]
    assert all(abs(v - FIELD.wall_displacement_um) < 1e-2 * FIELD.wall_displacement_um
               for v in values)
    assert values[0] < values[1] < values[2]          # approaching the wall value
    # strain at the wall is exactly the assumed displacement fraction
    assert abs(FIELD.hoop_strain(a) - FIELD.wall_displacement_fraction) < 1e-14
    assert abs(FIELD.radial_strain(a) + FIELD.wall_displacement_fraction) < 1e-14
    assert abs(FIELD.max_shear_strain(a) - 2 * FIELD.wall_displacement_fraction) < 1e-13
    try:
        FIELD.displacement_um(a * 0.999)
    except ValueError as exc:
        assert "outside the shaft wall" in str(exc)
    else:
        raise AssertionError("the field was evaluated INSIDE the shaft instead of refused")
    return {"wall_displacement_um": FIELD.wall_displacement_um,
            "hoop_strain_at_wall": FIELD.hoop_strain(a),
            "inside_shaft_refused": True}


def t_field_incompressibility_and_equilibrium():
    resid_in = [abs(FIELD.incompressibility_residual_um2(r, h)) for r in (10.0, 100.0)
                for h in (1e-2, 1e-3)]
    assert max(resid_in) < 1e-10
    # Lame equilibrium d(sigma_rr)/dr + (sigma_rr - sigma_tt)/r = 0
    scale = abs(FIELD.radial_stress_Pa(10.0)) / 10.0
    coarse = abs(FIELD.equilibrium_residual_Pa_per_um(10.0, 1e-2))
    fine = abs(FIELD.equilibrium_residual_Pa_per_um(10.0, 1e-3))
    assert coarse / scale < 1e-3 and fine / scale < 1e-5
    assert fine < coarse                                     # second-order FD
    # traction at the wall equals the wall pressure, and the elastic consistency
    # delta = p*a/(2G) reproduces the imposed displacement
    traction = -FIELD.radial_stress_Pa(FIELD.shaft_radius_um)
    assert abs(traction - FIELD.wall_pressure_Pa) < 1e-12
    recovered = FIELD.wall_pressure_Pa * FIELD.shaft_radius_um / (2 * FIELD.shear_modulus_Pa)
    assert abs(recovered - FIELD.wall_displacement_um) < 1e-14
    return {"incompressibility_residual_um2_max": max(resid_in),
            "equilibrium_residual_relative": {"h=1e-2": coarse / scale, "h=1e-3": fine / scale},
            "wall_pressure_Pa": FIELD.wall_pressure_Pa,
            "recovered_wall_displacement_um": recovered}


def t_field_threshold_radius_inverts_strain():
    for threshold, field in ((0.14, FIELD),
                             (0.34, CavityExpansionField(2.0, 0.02, 500.0)),
                             (0.21, CavityExpansionField(0.5, 0.5, 500.0))):
        r = field.strain_radius_um(threshold)
        assert abs(field.max_abs_principal_strain(r) - threshold) < 1e-12 * threshold \
            or r == field.shaft_radius_um
        assert field.max_abs_principal_strain(r * 1.001) <= threshold * 1.001
        if r > field.shaft_radius_um * 1.0001:      # not floored at the shaft wall
            assert field.max_abs_principal_strain(r * 0.999) >= threshold * 0.999
    # the damage radius can never be smaller than the shaft that made the hole
    tiny = CavityExpansionField(2.0, 0.05, 500.0)
    assert tiny.strain_radius_um(0.34) == tiny.shaft_radius_um
    # shear-stress proxy radius scales as sqrt(G) and 1/sqrt(tau)
    big_g = CavityExpansionField(5.0, 0.5, 2 * FIELD.shear_modulus_Pa)
    assert abs(big_g.shear_stress_radius_um(14.0) /
               FIELD.shear_stress_radius_um(14.0) - np.sqrt(2.0)) < 1e-12
    return {"strain_radius_um": {f"{t}": FIELD.strain_radius_um(t)
                                 for t in (0.14, 0.21, 0.34)},
            "shear_radius_at_14Pa_um": FIELD.shear_stress_radius_um(14.0),
            "floored_at_shaft_radius_um": tiny.strain_radius_um(0.34)}


def t_field_invalid_inputs_rejected():
    for kwargs in (dict(shaft_radius_um=0.0), dict(shaft_radius_um=-1.0),
                   dict(shaft_radius_um=float("nan")),
                   dict(shaft_radius_um=5.0, wall_displacement_fraction=0.0),
                   dict(shaft_radius_um=5.0, wall_displacement_fraction=1.5),
                   dict(shaft_radius_um=5.0, shear_modulus_Pa=0.0)):
        try:
            CavityExpansionField(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid field config accepted: {kwargs}")
    for bad in (0.0, -0.5):
        try:
            FIELD.displacement_um(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("r < a was accepted")
    for bad in (np.nan, np.inf):
        try:
            FIELD.displacement_um(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite r accepted")
    for bad in (0.0, -0.21):
        try:
            FIELD.strain_radius_um(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive strain threshold accepted")
    return {"invalid_field_configs_refused": 6, "invalid_radii_refused": 4}


# --------------------------------------------------------------------------
# 5. thresholds and resealing
# --------------------------------------------------------------------------
def t_threshold_options_are_flagged_cross_species():
    for key, name in STRAIN_THRESHOLD_CHOICES.items():
        a = THRESHOLDS[key]
        assert a.name == name and a.value in (0.14, 0.21, 0.34)
        assert a.species == "Cavia porcellus (guinea pig)"
        assert a.cross_species and a.forced_flag and a.source == "10.1115/1.1324667 (guinea-pig optic nerve)"
    shear = THRESHOLDS["_mechanoporation_shear"]
    assert abs(shear.value - 14.0) < 1e-12 and shear.species.startswith("Rattus")
    assert abs(THRESHOLDS["_mechanoporation_shear_dyn_cm2"].value - 140.0) < 1e-12
    assert abs(THRESHOLDS["_mechanoporation_duration_ms"].value - 300.0) < 1e-12
    # the source's own no-death caveat is carried in the registry note, not dropped
    snap = REGISTRY.snapshot()["rodent_mechanoporation_shear_dyn_cm2"]
    assert "NO CELL DEATH" in snap["notes"]
    # the strain thresholds are used as a PROXY and that is stated
    assert "proxy" in REGISTRY.snapshot()["rodent_axonal_strain_conservative"]["notes"].lower()
    return {"strain_thresholds": {k: THRESHOLDS[k].value for k in STRAIN_THRESHOLD_CHOICES},
            "shear_threshold_Pa": shear.value,
            "shear_duration_ms": THRESHOLDS["_mechanoporation_duration_ms"].value}


def t_threshold_boundary_cases():
    # monotone in the threshold, and the ordering matches the source's wording
    r_con = FIELD.strain_radius_um(0.14)
    r_opt = FIELD.strain_radius_um(0.21)
    r_per = FIELD.strain_radius_um(0.34)
    assert r_con > r_opt > r_per
    # a cable lying at exactly the damage radius is inside; just outside is not
    for offset, expect in ((r_opt, True), (r_opt + 1e-6, False), (r_opt - 1e-6, True)):
        idx = affected_node_indices(MORPH, (250.0, offset, 0.0), (0, 0, 1), r_opt)
        assert (idx.size > 0) is expect, (offset, idx)
    # shrinking the damage radius can only shrink the affected set
    sets = [set(affected_node_indices(MORPH, SHAFT_POINT, SHAFT_DIR, r).tolist())
            for r in (r_per, r_opt, r_con)]
    assert sets[0] <= sets[1] <= sets[2]
    # strain threshold crossing is exact at the boundary
    assert abs(FIELD.max_abs_principal_strain(r_opt) - 0.21) < 1e-14
    # the shear proxy and the strain proxy disagree, and both are reported
    r_shear = FIELD.shear_stress_radius_um(THRESHOLDS["_mechanoporation_shear"].value)
    assert r_shear > r_opt
    return {"damage_radius_um_by_threshold": {"0.14": r_con, "0.21": r_opt, "0.34": r_per},
            "shear_proxy_radius_um": r_shear,
            "affected_nodes": {f"{r:g}": len(s) for r, s in zip((r_per, r_opt, r_con), sets)},
            "proxy_disagreement_ratio": r_shear / r_opt}


def t_resealing_constants_applied_correctly():
    assert abs(THRESHOLDS["_mechanoporation_shear"].value - 14.0) < 1e-12
    assert abs(RESEALING.small_pore_tau_s - 60.0 / np.log(4.0)) < 1e-12
    assert abs(RESEALING.transected_tau_s - 1200.0) < 1e-12
    assert abs(RESEALING.transected_tau_sd_s - 300.0) < 1e-12
    tau = RESEALING.transected_tau_s
    assert RESEALING.decay_factor(0.0) == 1.0
    assert abs(RESEALING.decay_factor(tau) - np.exp(-1.0)) < 1e-15
    assert abs(RESEALING.decay_factor(2 * tau) - np.exp(-2.0)) < 1e-15
    t = np.array([0.0, tau, 2 * tau, 1e6])
    f = RESEALING.decay_factors(t)
    assert np.all(np.diff(f) < 0) and f[-1] < 1e-300
    for bad in (-1.0, float("nan")):
        try:
            RESEALING.decay_factor(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid time accepted")
    try:
        RESEALING.tau_s("sheath")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown resealing kind was accepted")
    repair = SheathRepairModel(RESEALING.small_pore_tau_s, RESEALING.transected_tau_s,
                               GUARD.adopt("assumed_sheath_repair_tau_s").value, 17.0)
    assert abs(repair.sheath_over_transected_ratio - 1080.0) < 1e-9
    assert "no seconds-scale sheath resealing is applied" in \
        repair.to_dict()["applied"].lower()
    # and the cable leak really is scaled by exp(-t/tau) at the last update
    res = _paired_cable(config=CableDamageConfig(resealing_kind="transected"),
                        resealing=RESEALING)
    idx = res["damaged_node_indices"]
    factor = RESEALING.decay_factor(res["duration_ms"] / 1000.0, "transected")
    density = CableDamageConfig().leak_density_S_cm2
    areas = CableNeuron(MORPH, dt_ms=0.05, **CABLE_KWARGS).area_cm2[idx]
    assert np.allclose(res["g_end_uS"][idx], density * areas * 1e6 * factor,
                       rtol=1e-15, atol=0)
    assert res["resealing_factor_range"][1] < 1.0
    return {"small_pore_tau_s": RESEALING.small_pore_tau_s,
            "transected_tau_s": RESEALING.transected_tau_s,
            "sheath_over_transected_ratio": repair.sheath_over_transected_ratio,
            "sheath_over_pore_ratio": repair.sheath_over_pore_ratio,
            "cable_leak_factor_at_window_end": res["resealing_factor_range"][1]}


# --------------------------------------------------------------------------
# 6. coupling: exactness and materiality
# --------------------------------------------------------------------------
def t_membrane_coupling_exact():
    cable = CableNeuron(MORPH, dt_ms=0.05, **CABLE_KWARGS)
    sham = CableNeuron(MORPH, dt_ms=0.05, **CABLE_KWARGS)
    idx = affected_node_indices(MORPH, SHAFT_POINT, SHAFT_DIR, FIELD.strain_radius_um(0.21))
    assert idx.size == 5, idx
    rec = apply_membrane_damage(cable, idx, 0.5, cable.E_leak)
    expected = 0.5 * cable.area_cm2[idx] * 1e6
    assert np.array_equal(cable.g_end[idx], expected)          # exact, not approximate
    assert np.array_equal(sham.g_end, np.zeros(MORPH.n))       # sham exactly zero
    assert rec["n_rows_set"] == idx.size and rec["grouped_calls"] == 1
    assert rec["total_g_uS"] == float(expected.sum())
    assert cable.g_end[idx[0]] / cable.g_mem[idx[0]] > 1000.0  # a real shunt
    # the matrix diagonal moved by exactly the added conductance
    delta = cable.G.diagonal()[idx] - sham.G.diagonal()[idx]
    # g_end itself is EXACT (array_equal above); the reassembled matrix diagonal
    # only agrees to floating-point summation accuracy, which is what is claimed
    assert np.allclose(delta, expected, rtol=1e-12, atol=0)
    # duplicates / out-of-range / density of zero are rejected or exact
    for bad in (np.array([], dtype=int), np.array([0, 0]), np.array([MORPH.n])):
        try:
            apply_membrane_damage(CableNeuron(MORPH, dt_ms=0.05, **CABLE_KWARGS),
                                  bad, 0.5, -65.0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid node set accepted: {bad}")
    zero = CableNeuron(MORPH, dt_ms=0.05, **CABLE_KWARGS)
    apply_membrane_damage(zero, idx, 0.0, -65.0)
    assert np.array_equal(zero.g_end, np.zeros(MORPH.n))
    return {"damaged_nodes": idx.tolist(),
            "g_uS_per_node": float(expected[0]),
            "g_end_over_g_mem": float(cable.g_end[idx[0]] / cable.g_mem[idx[0]]),
            "matrix_diagonal_increment_exact": True}


def t_cable_pairing_changes_the_cable_and_is_material():
    base = _paired_cable()
    assert base["damage_radius_um"] > FIELD.shaft_radius_um
    assert base["max_abs_delta_mV"] > 0.5, base["max_abs_delta_mV"]
    assert base["max_local_attenuation_fraction"] > 0.5
    sham, ins = base["sham"][-1], base["inserted"][-1]
    idx = base["damaged_node_indices"]
    assert np.all(np.abs(sham[idx] - base["E_leak_mV"]) >
                  np.abs(ins[idx] - base["E_leak_mV"]))
    # the effect is reported both locally and globally
    global_frac = np.abs(base["delta_mV"][-1]).max() / \
        np.abs(sham - base["E_leak_mV"]).max()
    # and the drive is identical in both runs (only the coupling differs)
    assert np.array_equal(base["i_inject_sham_nA"], base["drive_nA"] *
                          (np.arange(MORPH.n) == base["inject_index"]))
    # a zero-diameter probe damages nothing and changes nothing bitwise
    tiny = CavityExpansionField(shaft_radius_um=0.5, wall_displacement_fraction=0.05)
    zero = run_paired_cable(MORPH, tiny, SHAFT_POINT, SHAFT_DIR, 0.34,
                            dt_ms=0.05, duration_ms=10.0, cable_kwargs=CABLE_KWARGS)
    assert zero["damaged_node_indices"].size == 0
    assert np.array_equal(zero["sham"], zero["inserted"])
    assert np.array_equal(zero["g_end_uS"], np.zeros(MORPH.n))
    return {"r_damage_um": base["damage_radius_um"],
            "damaged_nodes": idx.tolist(),
            "max_abs_delta_mV": base["max_abs_delta_mV"],
            "local_attenuation_fraction": base["max_local_attenuation_fraction"],
            "global_max_relative_change": float(global_frac),
            "input_resistance_Mohm": base["input_resistance_Mohm"],
            "drive_nA": base["drive_nA"],
            "zero_damage_case_bitwise_identical": True}


def t_chemistry_coupling_exact_and_different():
    res = _paired_chemistry(fraction=0.7, variant="sheath_intact")
    closed = res["variants"]["closed"]
    inserted = closed["inserted_coupling"]
    cfg = chemistry_config_for_variant(VARIANTS["sheath_intact"], GUARD.adopt)
    expected_p, expected_g = cfg.scaled_values(0.7)
    assert closed["inserted_permeability_um_s"] == expected_p        # exact
    assert closed["sham_permeability_um_s"] == 0.0                   # exact
    assert inserted["sheath_conductance_um3_s"] == expected_g        # exact
    assert closed["sham_coupling"]["sheath_conductance_um3_s"] == 0.0
    assert inserted["barrier_edge"] == (1, 2)
    # the ligand barrier was replaced through the public API with that value
    event = closed["inserted_sheath_events"][-1]
    assert event["barrier_edges"] == repr({(1, 2): expected_g})
    assert event["mass_jump"] == 0.0
    # sham really has zero exchange: the far side is bitwise untouched
    sham_far = closed["traces"]["sham"][-1, 5]
    assert sham_far == 0.0
    assert closed["inserted_far_ligand_nM"] > 0.0
    # the chemistry moved, and the K pools moved in the damaged run only
    assert closed["traces"]["sham"][-1, 2] != closed["traces"]["inserted"][-1, 2]
    assert abs(closed["max_potassium_mass_error"]) < 1e-6
    assert abs(closed["max_ligand_mass_error"]) < 1e-11
    return {"inserted_permeability_um_s": closed["inserted_permeability_um_s"],
            "inserted_sheath_conductance_um3_s": inserted["sheath_conductance_um3_s"],
            "sham_far_voxel_ligand_nM": sham_far,
            "inserted_far_voxel_ligand_nM": closed["inserted_far_ligand_nM"],
            "final_Ke_sham_mM": closed["final_Ke_sham_mM"],
            "final_Ke_inserted_mM": closed["final_Ke_inserted_mM"],
            "max_K_mass_error": closed["max_potassium_mass_error"],
            "max_ligand_mass_error": closed["max_ligand_mass_error"]}


def t_sheath_variants_differ_exactly_as_defined():
    intact = _paired_chemistry(fraction=0.7, variant="sheath_intact")
    removed = _paired_chemistry(fraction=0.7, variant="sheath_removed")
    cfg_removed = chemistry_config_for_variant(VARIANTS["sheath_removed"], GUARD.adopt)
    # sheath removed: the sheath term is already gone BEFORE the electrode, so the
    # damage coupling changes the sheath conductance by exactly nothing
    assert cfg_removed.sheath_conductance_intact_um3_s == \
        cfg_removed.sheath_conductance_damaged_um3_s
    a = removed["variants"]["closed"]["inserted_coupling"]["sheath_conductance_um3_s"]
    b = removed["variants"]["closed"]["sham_coupling"]["sheath_conductance_um3_s"]
    assert a == b
    diff_removed = abs(removed["variants"]["closed"]["final_Ke_inserted_mM"] -
                       removed["variants"]["closed"]["final_Ke_sham_mM"])
    diff_intact = abs(intact["variants"]["closed"]["final_Ke_inserted_mM"] -
                      intact["variants"]["closed"]["final_Ke_sham_mM"])
    assert diff_removed > 0 and diff_intact > 0
    # in the sheath-removed preparation the electrode changes the sheath term by
    # EXACTLY nothing, because that restriction was already gone before it arrived
    for variant in ("closed", "supported"):
        rv_res = removed["variants"][variant]
        assert rv_res["sham_far_ligand_nM"] == rv_res["inserted_far_ligand_nM"]
        assert rv_res["sham_coupling"]["sheath_conductance_um3_s"] == \
            rv_res["inserted_coupling"]["sheath_conductance_um3_s"]
        iv_res = intact["variants"][variant]
        assert iv_res["sham_far_ligand_nM"] == 0.0
        assert iv_res["inserted_far_ligand_nM"] > 0.0
    # a fully removed sheath is more open than a 70%-damaged intact one
    assert removed["variants"]["closed"]["inserted_far_ligand_nM"] > \
        intact["variants"]["closed"]["inserted_far_ligand_nM"]
    iv = parenchymal_damage_estimate(FIELD, 0.21, GEOMETRY, SHEATH, VARIANTS["sheath_intact"])
    rv = parenchymal_damage_estimate(FIELD, 0.21, GEOMETRY, SHEATH, VARIANTS["sheath_removed"])
    assert iv["sheath_breached"] is True and rv["sheath_breached"] is False
    assert iv["sheath_breach_area_um2"] > 0.0 and rv["sheath_breach_area_um2"] == 0.0
    assert iv["parenchymal_fraction_mechanistic"] > rv["parenchymal_fraction_mechanistic"]
    assert VARIANTS["sheath_removed"].herniation_multiplier == 1.0
    return {"sheath_intact": {k: iv[k] for k in
                              ("sheath_breached", "parenchymal_multiplier",
                               "parenchymal_fraction_mechanistic",
                               "sheath_delamination_radius_um", "sheath_breach_area_um2")},
            "sheath_removed": {k: rv[k] for k in
                               ("sheath_breached", "parenchymal_multiplier",
                                "parenchymal_fraction_mechanistic",
                                "sheath_delamination_radius_um", "sheath_breach_area_um2")},
            "Ke_difference_mM": {"intact": diff_intact, "removed": diff_removed}}


def t_sheath_breach_criterion_and_degeneracy():
    # primary criterion: opening radius, sweeps cleanly and is monotone in diameter
    for c in (0.5, 1.0, 2.0, 3.0, 5.0):
        sh = SheathLayer(thickness_um=3.0, critical_opening_radius_um=c,
                         delamination_multiple=2.0)
        states = [sh.breached(a) for a in (0.5, 1.0, 2.5, 5.0, 20.0, 50.0)]
        assert states == sorted(states)
        assert sh.breached(c) and not sh.breached(c * 0.999)
    # secondary (stretch) criterion is size-independent: report the degeneracy
    stretches = [SHEATH.hole_edge_stretch(a) for a in (0.5, 5.0, 500.0)]
    assert len(set(stretches)) == 1
    assert stretches[0] == SHEATH.wall_displacement_fraction
    assert SHEATH.stretch_criterion_breached(50.0, 0.3) is True
    assert SHEATH.stretch_criterion_breached(50.0, 0.6) is False
    # delamination radius only exists when breached
    assert SHEATH.delamination_radius_um(0.1) == 0.0
    expected = max(5.0, SHEATH.critical_opening_radius_um +
                   SHEATH.delamination_multiple * SHEATH.thickness_um)
    assert SHEATH.delamination_radius_um(5.0) == expected
    assert abs(SHEATH.breach_area_um2(5.0) - np.pi * expected ** 2) < 1e-9
    for bad in (0.0, -1.0):
        try:
            SheathLayer(thickness_um=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive sheath thickness accepted")
    return {"stretch_criterion_is_size_independent": True,
            "hole_edge_stretch": stretches[0],
            "delamination_radius_um": SHEATH.delamination_radius_um(5.0),
            "breach_area_um2": SHEATH.breach_area_um2(5.0)}


def t_free_parameter_sweep_is_reported_not_fitted():
    fractions = {}
    for r in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
        est = parenchymal_damage_estimate(FIELD, 0.21, GEOMETRY, SHEATH,
                                          VARIANTS["sheath_removed"], free_radius_um=r)
        fractions[f"{r:g}"] = est["parenchymal_fraction_free_parameter"]
    vals = list(fractions.values())
    assert vals == sorted(vals) and vals[-1] > 100 * vals[0]
    # the sweep spans 4 orders of magnitude: the answer is set by the free parameter
    assert vals[-1] / vals[0] > 1e3
    # the mechanistic radius and the free radius are reported side by side
    est = parenchymal_damage_estimate(FIELD, 0.21, GEOMETRY, SHEATH,
                                      VARIANTS["sheath_removed"], free_radius_um=25.0)
    assert abs(est["mechanistic_damage_radius_um"] - FIELD.strain_radius_um(0.21)) < 1e-12
    assert est["free_parameter_radius_um"] == 25.0
    return {"free_radius_sweep_fraction": fractions,
            "ratio_max_over_min": vals[-1] / vals[0]}


def t_optional_proximity_coupling_default_off():
    off = ProximityExcitabilityCoupling()
    assert off.enabled is False and off.to_dict()["default"] == "OFF"
    assert np.array_equal(off.scale(np.array([0.0, 1.0, 1e6])), np.ones(3))
    base = _paired_cable()
    with_off = _paired_cable(config=CableDamageConfig(excitability=off))
    assert np.array_equal(base["inserted"], with_off["inserted"])
    assert np.array_equal(base["sham"], with_off["sham"])
    assert np.array_equal(base["delta_mV"], with_off["delta_mV"])
    on = ProximityExcitabilityCoupling(enabled=True, full_effect_radius_um=5.0,
                                       max_attenuation=0.5)
    d = np.array([0.0, 2.5, 5.0, 7.5])
    expected = np.array([0.5, 0.75, 1.0, 1.0])
    assert np.array_equal(on.scale(d), expected)
    # 1) with a 5 um effect radius and every cable node >= 6 um from the shaft
    #    axis, the placeholder cannot do anything: an honest, documented limit
    #    of a point-drive model with a point injection site.
    far = _paired_cable(config=CableDamageConfig(excitability=on))
    assert np.array_equal(base["inserted"], far["inserted"])
    assert np.array_equal(base["sham"], far["sham"])
    # 2) widen the proximity radius past the shaft-to-fibre distance and the run
    #    DOES change, exactly by the scale array, in BOTH runs
    on_near = ProximityExcitabilityCoupling(enabled=True, full_effect_radius_um=10.0,
                                           max_attenuation=0.5)
    probe = int(base["damaged_node_indices"][0])
    near = _paired_cable(config=CableDamageConfig(excitability=on_near), inject_index=probe)
    near_off = _paired_cable(inject_index=probe)
    assert not np.array_equal(near["inserted"], near_off["inserted"])
    assert not np.array_equal(near["sham"], near_off["sham"])
    assert np.array_equal(near["drive_effective_nA"],
                          near["i_inject_sham_nA"] *
                          on_near.scale(near["distances_to_shaft_axis_um"]))
    expected_scale = 1.0 - 0.5 * (1.0 - near["distances_to_shaft_axis_um"][probe] / 10.0)
    assert near["drive_effective_nA"][probe] == near["drive_nA"] * expected_scale
    assert near["drive_nA"] == near_off["drive_nA"]      # only the coupling differs
    for bad in (-0.1, np.nan):
        try:
            on.scale(np.array([bad]))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid distance accepted")
    try:
        ProximityExcitabilityCoupling(max_attenuation=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("attenuation > 1 accepted")
    return {"default_enabled": off.enabled,
            "disabled_run_bitwise_identical": True,
            "enabled_scale": expected.tolist(),
            "5um_radius_effect_on_this_geometry": 0.0,
            "min_node_distance_um": float(base["distances_to_shaft_axis_um"].min()),
            "10um_radius_scale_at_probe_adjacent_node": float(expected_scale),
            "enabled_run_differs_when_radius_exceeds_shaft_distance": True}


def t_determinism_bitwise():
    a = _paired_cable()
    b = _paired_cable()
    assert np.array_equal(a["sham"], b["sham"])
    assert np.array_equal(a["inserted"], b["inserted"])
    assert np.array_equal(a["delta_mV"], b["delta_mV"])
    c = _paired_chemistry(fraction=0.7)
    d = _paired_chemistry(fraction=0.7)
    for variant in ("closed", "supported"):
        for name in ("sham", "inserted"):
            assert np.array_equal(c["variants"][variant]["traces"][name],
                                  d["variants"][variant]["traces"][name])
    assert GEOMETRY.damage_fraction(37.5) == GEOMETRY.damage_fraction(37.5)
    assert FIELD.strain_radius_um(0.21) == FIELD.strain_radius_um(0.21)
    return {"cable_pair_bitwise_identical": True, "chemistry_pair_bitwise_identical": True,
            "geometry_and_field_reproducible": True}


def t_invalid_input_rejection():
    refused = []
    cases = [
        lambda: affected_node_indices(MORPH, (0.0, 0.0), (0, 0, 1), 5.0),
        lambda: affected_node_indices(MORPH, (0.0, 0.0, 0.0), (0, 0, 0), 5.0),
        lambda: affected_node_indices(MORPH, SHAFT_POINT, (0, 0, 1), -1.0),
        lambda: affected_node_indices("not a morphology", SHAFT_POINT, (0, 0, 1), 1.0),
        lambda: run_paired_cable(MORPH, FIELD, SHAFT_POINT, SHAFT_DIR, 0.21,
                                 dt_ms=0.0, cable_kwargs=CABLE_KWARGS),
        lambda: run_paired_cable(MORPH, FIELD, SHAFT_POINT, SHAFT_DIR, 0.21,
                                 duration_ms=0.01, dt_ms=1.0, cable_kwargs=CABLE_KWARGS),
        lambda: run_paired_cable(MORPH, FIELD, SHAFT_POINT, SHAFT_DIR, 0.21,
                                 inject_index=999, cable_kwargs=CABLE_KWARGS),
        lambda: run_paired_cable(MORPH, "no field", SHAFT_POINT, SHAFT_DIR, 0.21),
        lambda: run_paired_cable(MORPH, FIELD, SHAFT_POINT, SHAFT_DIR, -0.21,
                                 cable_kwargs=CABLE_KWARGS),
        lambda: apply_membrane_damage("not a cable", [1], 0.5, -65.0),
        lambda: apply_membrane_damage(CableNeuron(MORPH, dt_ms=0.05, **CABLE_KWARGS),
                                      [1], -0.5, -65.0),
        lambda: apply_membrane_damage(CableNeuron(MORPH, dt_ms=0.05, **CABLE_KWARGS),
                                      [1], 0.5, float("nan")),
        lambda: ChemistryDamageConfig(permeability_damaged_um_s=-1.0),
        lambda: ChemistryDamageConfig().scaled_values(1.5),
        lambda: apply_chemistry_coupling("x", "y", (1, 2), 0.5),
        lambda: apply_chemistry_coupling(InjuryPotassium(PotassiumConfig()),
                                         InjuryLigand(), "not a tuple", 0.5),
        lambda: run_paired_chemistry(0.5, dt_s=0.0),
        lambda: run_paired_chemistry(0.5, duration_s=0.0, dt_s=0.1),
        lambda: preparation_variants(GUARD.adopt)["nonsense"],
        lambda: PreparationVariant(name="sheath_peeled", sheath_present=True),
        lambda: PreparationVariant(name="sheath_intact", sheath_present=True,
                                   herniation_multiplier=-2.0),
        lambda: SheathRepairModel(0.0, 1200.0, 1e6, 17.0),
        lambda: FlyBrainGeometry(length_um=0.0),
        lambda: FlyBrainGeometry(thickness_um=-100.0),
        lambda: circle_rectangle_overlap_area_um2(-1.0, 0.0, 0.0, 1.0, 1.0),
        lambda: rodent_radius_transfer_table(GEOMETRY, [1.0], 0.0),
        lambda: AffineDamageRadius(offset_um=-1.0),
        lambda: ResealingTimescales(0.0, 1200.0, 300.0),
    ]
    for i, fn in enumerate(cases):
        try:
            fn()
        except (ValueError, TypeError, KeyError, ProvenanceError) as exc:
            refused.append(f"{i}:{type(exc).__name__}")
        else:
            raise AssertionError(f"invalid input case {i} was accepted")
    assert len(refused) == len(cases)
    return {"n_invalid_cases_refused": len(refused), "kinds": sorted(set(refused))}


def t_honesty_text_present():
    blob = " ".join(HONESTY_STATEMENTS).lower()
    for must in ("no quantitative electrode-damage measurement",
                 "ASSUMED", "CROSS-SPECIES PROXY", "NO DEATH",
                 "load-bearing", "scale\nbars", "MOUSE", "FUNCTION FORM",
                 "FRACTION of the 500x300x100 um fly brain",
                 "UNMEASURED FREE PARAMETER",
                 "No consciousness, viability, survival or medical claim",
                 "does NOT predict real fly electrode damage"):
        assert must.replace("\n", " ").lower() in blob, must
    return {"n_honesty_statements": len(HONESTY_STATEMENTS),
            "all_mandatory_claims_present": True}


def t_no_rodent_number_presented_as_fly():
    """End-to-end: a report fragment built from real records audits clean."""
    report = {
        "geometry": {"length_um": GUARD.adopt("fly_brain_length_um").to_dict(),
                     "thickness_um": GUARD.adopt("fly_brain_thickness_um").to_dict()},
        "strain_thresholds": {k: THRESHOLDS[k].to_dict() for k in STRAIN_THRESHOLD_CHOICES},
        "mechanoporation": {"shear_Pa": THRESHOLDS["_mechanoporation_shear"].to_dict(),
                            "duration_ms": THRESHOLDS["_mechanoporation_duration_ms"].to_dict()},
        "resealing": {"small_pore_tau_s": GUARD.adopt("rodent_reseal_small_pore_tau_s").to_dict(),
                      "transected_tau_s": GUARD.adopt("rodent_reseal_transected_tau_s").to_dict()},
        "probe_force": {"slope_per_mm": GUARD.adopt("rodent_probe_force_slope_mN_per_mm").to_dict()},
        "bleed": {"always_diameter_um": GUARD.adopt("rodent_bleed_always_diameter_um").to_dict()},
        "rodent_radius_transfer": rodent_radius_transfer_table(
            GEOMETRY, GUARD.adopt, 900.0, radii_um=[50.0, 100.0]),
        "sheath": SHEATH.to_dict(),
        "parenchymal_sweep": {r: parenchymal_damage_estimate(
            FIELD, 0.21, GEOMETRY, SHEATH, VARIANTS["sheath_intact"], free_radius_um=r,
            threshold_record=THRESHOLDS["optimal"]) for r in (2.0, 10.0, 50.0)},
        "guard": GUARD.record([THRESHOLDS["optimal"], THRESHOLDS["_mechanoporation_shear"]]),
        "honesty": list(HONESTY_STATEMENTS),
    }
    violations = audit_species_labeling(report)
    assert violations == [], violations
    # the fly numbers in that report are the fly ones, not the rodent ones
    assert report["geometry"]["thickness_um"]["value"] == 100.0
    assert report["geometry"]["thickness_um"]["species"] == FLY_SPECIES
    assert report["strain_thresholds"]["optimal"]["cross_species"] is True
    return {"report_fragment_violations": 0,
            "n_foreign_records_used": len(GUARD.record(
                [THRESHOLDS["optimal"], THRESHOLDS["_mechanoporation_shear"]]
            )["foreign_parameters_used"])}


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
GROUPS = {
    "provenance_enforcement": [
        t_measured_requires_citation,
        t_derived_requires_registered_parents,
        t_illustrative_and_range_checks,
        t_unregistered_parameter_refused,
        t_registry_is_valid_and_tiered,
        t_no_drosophila_lamella_tagged_measured,
        t_report_audit_detects_unlabelled_rodent_numbers,
    ],
    "species_guard": [
        t_guard_refuses_cross_species_measured_import,
        t_guard_allows_same_species_illustrative_and_measured,
        t_guard_forces_flag_when_opted_in,
        t_guard_fires_on_other_insects_too,
        t_guard_rejects_non_fly_profile,
    ],
    "geometry_sanity": [
        t_circle_area_matches_grid_crosscheck,
        t_fly_scaling_and_rodent_transfer,
        t_damage_monotone_in_diameter,
        t_geometry_depth_and_axis_validation,
    ],
    "field_analytic_limits": [
        t_field_1_over_r_and_decay,
        t_field_wall_continuity_and_strain,
        t_field_incompressibility_and_equilibrium,
        t_field_threshold_radius_inverts_strain,
        t_field_invalid_inputs_rejected,
    ],
    "thresholds_and_resealing": [
        t_threshold_options_are_flagged_cross_species,
        t_threshold_boundary_cases,
        t_resealing_constants_applied_correctly,
    ],
    "coupling_exactness": [
        t_membrane_coupling_exact,
        t_cable_pairing_changes_the_cable_and_is_material,
        t_chemistry_coupling_exact_and_different,
        t_sheath_variants_differ_exactly_as_defined,
        t_sheath_breach_criterion_and_degeneracy,
        t_free_parameter_sweep_is_reported_not_fitted,
        t_optional_proximity_coupling_default_off,
    ],
    "determinism_and_validation": [
        t_determinism_bitwise,
        t_invalid_input_rejection,
    ],
    "honesty_and_labelling": [
        t_honesty_text_present,
        t_no_rodent_number_presented_as_fly,
    ],
}


def run_tests():
    start = time.perf_counter()
    checks, by_group = [], {}
    for group, functions in GROUPS.items():
        by_group.setdefault(group, {"n_checks": 0, "n_passed": 0, "n_failed": 0})
        for fn in functions:
            name = fn.__name__[2:]
            entry = {"name": name, "group": group}
            try:
                detail = fn()
                entry["status"] = "PASS"
                entry["detail"] = detail
                by_group[group]["n_passed"] += 1
            except Exception as exc:               # noqa: BLE001 - report, never hide
                entry["status"] = "FAIL"
                entry["detail"] = f"{type(exc).__name__}: {exc}"
                by_group[group]["n_failed"] += 1
            by_group[group]["n_checks"] += 1
            checks.append(entry)
    passed = sum(c["status"] == "PASS" for c in checks)
    return {
        "n_checks": len(checks),
        "n_passed": passed,
        "n_failed": len(checks) - passed,
        "passed": passed == len(checks),
        "by_group": by_group,
        "checks": checks,
        "wall_seconds": time.perf_counter() - start,
        "process_peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "what_a_pass_means": "the code does what it claims numerically: provenance is "
                             "enforced, the species guard refuses foreign numbers, the "
                             "assumed field satisfies its analytic limits, thresholds "
                             "behave at their boundaries, and the couplings change the "
                             "cable/chemistry in an exactly accounted way",
        "what_a_pass_does_not_mean": "that any of it is right for a real fly. There is "
                                     "no insect electrode-damage measurement, no measured "
                                     "strain field around a probe, and no measured sheath "
                                     "breach threshold; every threshold is a cross-species "
                                     "proxy and every fly geometry value is illustrative "
                                     "or assumed",
    }


def main():
    import json
    results = run_tests()
    print(json.dumps(results, indent=2, default=float))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
