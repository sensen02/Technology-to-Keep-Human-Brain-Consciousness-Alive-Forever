"""Natural scene experiment: pre-registered predictions, then measurement.

WHAT THIS SCRIPT IS
-------------------
A hand-built prototype experiment over ``engine.embodied.natural_scene``.  It is
NOT a finished scene system and NOT a biology result: the eye optics are
FlyGym's (its own docstring says the fisheye Retina is calibrated for FlyGym's
own eye placement, so ommatidia readouts on the NeuroMechFly body are
APPROXIMATE), the gait is FlyGym's own tripod CPG (an ENGINEERING BASELINE, not
the connectome), and the model is NOT fly-accurate in absolute scale (total mass
1.0243e-3 kg, ~1042x a real Drosophila's 0.983 mg), so no absolute force here is
biological.

THE PREDICTIONS BELOW ARE DECLARED **BEFORE** ANY MEASUREMENT IN THIS FILE
-------------------------------------------------------------------------
Thresholds are module-level constants evaluated at import time.  The runner does
not adjust them after seeing data; a FAIL is reported as a FAIL.

P1  periodicity is gone
    Terrain: the generator's periodicity score must be below
    ``P1_PEAK_FRAC_MAX``.  ``peak_frac`` is the FRACTION OF TOTAL SPECTRAL POWER
    concentrated in the single strongest non-DC bin; for an aperiodic field it is
    ~1/(number of bins), and a grating concentrates a large fraction into one or
    two bins.  It is used instead of "peak / median magnitude" because that ratio
    is NOT scale-free: measured, a natural fBm patch with no grating at all
    scored 195-217 on it, while a pure 1-D grating scored ~6e15 -- so a fixed
    threshold on the ratio would be a size artefact, not a test.  The ratio is
    still reported for continuity.
    The same test is applied to the ground TEXTURE as the eye samples it (the
    source PNG, and the source averaged down to the physical scale of one eye
    camera pixel at the nominal camera range), and to the RENDERED ground pixels
    of both the old and the new scene.
    Also: the terrain's fitted radial log-log slope must lie inside
    ``P1_SLOPE_RANGE``.  Justification: a 2-D fBm surface whose 1-D profile has
    spectral slope -(2H+1) has radial slope -(2H+1.5) because the radial average
    of a 2-D spectrum carries an extra k^-1 from the circle measure.  For
    H in [0.55, 1.0] that is [-3.5, -2.6].  The range is widened to
    ``(-3.5, -0.5)`` to allow real photographs and flatter terrain: the lower end
    is the H=1.0 extreme of the fBm family, and slopes shallower than -0.5 mean
    the field is essentially white noise, i.e. no spatial structure at all.
    MEASURED slope for the shipped OLD ground (MuJoCo's own builtin checker,
    read back from ``model.tex_data``) is reported alongside for contrast.

P2  saturation is fixed
    Eye-frame fraction of pixels at EXACTLY 255, same recording protocol, for the
    old presets and the new one.  Target declared before measuring:
    ``natural_fbm`` must be below ``P2_FRAC255_TARGET`` = 0.05, and must be at
    most ``P2_RELATIVE_FACTOR`` = 0.25 of the OLD baseline.  The baseline the
    orchestrator measured is 0.4499-0.4638; this run re-measures it.

P3  still a valid simulator
    With the new scene the fly must move at least ``P3_MIN_DISPLACEMENT_MM``
    (2.0 mm) horizontally, end with thorax height inside
    ``P3_THORAX_Z_RANGE_MM`` = (0.3, 3.0) mm (the same upright gate as
    ``body_backend.summarise_walk``), produce finite state on every recorded
    sample, never let the thorax go below ``P3_MIN_THORAX_Z_MM`` = 0.2 mm (i.e.
    not fall through the terrain), and produce eye frames with raw std > 1.0 of
    255 (non-blank).
    NOTE, measured while building this: 2.4 mm of displacement in 2.0 s is the
    CPG's own forward bias, not steering, and is NOT evidence about vision.

P4  the sky's contribution is isolated
    ``natural_fbm`` vs ``fbm_uniform_sky`` differ ONLY in the skybox rgb values,
    so the difference in eye-frame saturation and mean luminance IS the sky's
    contribution.  Declared direction: replacing the white sky with the gradient
    sky must REDUCE ``frac==255`` whenever any sky pixels are in view.  If the
    two conditions differ by exactly nothing, this script reports that the sky
    contributes nothing measurable to the EYE images in this protocol -- and says
    explicitly why (the eye cameras point at the ground, so rebalancing the
    headlight is what fixes the clipping) rather than claiming a sky effect.
    The sky itself is measured separately by rendering an upward view and
    reporting its mean / std / unique-value count / frac==255.

Run with:  MUJOCO_GL=egl venv_body/bin/python run_natural_scene.py
(osmesa FAILS on this machine: libOSMesa is absent.)
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from engine.embodied import natural_scene as ns  # noqa: E402
from engine.embodied import scene  # noqa: E402
from engine.embodied.body_backend import BodyBackend, BodyConfig  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "embodied_body")
JSON_PATH = os.path.join(OUT_DIR, "natural_scene.json")
PNG_OVERVIEW = os.path.join(OUT_DIR, "natural_scene_overview.png")
PNG_SPECTRA = os.path.join(OUT_DIR, "natural_scene_spectra.png")

# --------------------------------------------------------------------------- #
# PRE-REGISTERED THRESHOLDS -- declared before any measurement
# --------------------------------------------------------------------------- #

P1_PEAK_FRAC_MAX = 0.05
P1_PEAK_FRAC_JUSTIFICATION = (
    "peak_frac is the share of non-DC spectral power in the single strongest "
    "bin.  For an aperiodic field on n bins it is ~1/n; a test calibration run "
    "BEFORE this experiment gave: natural fBm 0.017-0.018, white noise 0.0002, "
    "of the shipped old checker 'terrain' (8 mm tiles) 0.016-0.053 depending on "
    "how many tiles fit the window, of a synthetic grating 0.50, of an aligned "
    "2-D checkerboard with an integer number of tiles per patch 0.053-0.165.  "
    "0.05 sits in the gap between 'no line' and 'a line', and is applied "
    "identically to the terrain and to the texture.")
P1_SLOPE_RANGE = (-3.5, -0.5)
P1_SLOPE_JUSTIFICATION = (
    "radial slope of a 2-D fBm surface = -(2H + 1.5); H in [0.55, 1.0] gives "
    "[-3.5, -2.6]; the band is widened to (-3.5, -0.5) so a real photograph "
    "(measured here: -1.6 for leaf litter, -1.7 for pebbles) or flatter terrain "
    "still passes, while white noise (-0.06) and any spectrum that RISES towards "
    "high frequency do not.")
P1_EXPECTED_SLOPE_FROM_TERRAIN_HURST = None      # filled from the generator metadata
P1_R2_MIN = 0.90
P2_FRAC255_TARGET = 0.05
P2_RELATIVE_FACTOR = 0.25
P2_BASELINE_REPORTED_BY_ORCHESTRATOR = (0.4499, 0.4638)
P3_MIN_DISPLACEMENT_MM = 2.0
P3_THORAX_Z_RANGE_MM = (0.3, 3.0)
P3_MIN_THORAX_Z_MM = 0.2
P3_MIN_RAW_FRAME_STD = 1.0
P4_SKY_MUST_REDUCE_FRAC255 = True

# --------------------------------------------------------------------------- #
# recording protocol -- identical for every condition
# --------------------------------------------------------------------------- #

SEED = 0
N_SETTLE = 600          # 60 ms of settling
N_SAMPLES = 12
SAMPLE_EVERY = 150      # 15 ms apart -> 180 ms of recording
DUR_S = (N_SETTLE + N_SAMPLES * SAMPLE_EVERY) * 1e-4
WORLD_HALF_SIZE_MM = 300.0   # rendered extent; the plane collides infinitely
OLD_PRESETS = ("blank", "natural")
NEW_PRESETS = ("natural_fbm", "natural_fbm_lit_no_objects", "fbm_uniform_sky")

#: nominal eye-camera pixel footprint at the sampling range, used to downscale the
#: texture to what the eye can actually resolve.  ENGINEERING_DEFAULT: the retina
#: is 450x512 px over a 157 deg fovy, and the ground under the fly is ~2 mm away,
#: giving ~0.008 mm per pixel at that range (2 mm * 2*tan(78.5deg) / 512).
EYE_SAMPLE_PITCH_MM = 0.008


def _free_camera(model, *, azimuth=135.0, elevation=25.0, distance=6.5,
                 lookat=(0.0, 0.0, 0.6)):
    """A FREE tracking camera.  ``elevation`` MUST be positive.

    Measured: with elevation < 0 the camera sits below the infinite ground plane
    and the frame comes back blank white.
    """
    import mujoco
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = np.asarray(lookat, dtype=float)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    cam.distance = float(distance)
    return cam


def _render(model, data, cam, h=480, w=640):
    import mujoco
    r = mujoco.Renderer(model, h, w)
    r.update_scene(data, camera=cam)
    fr = np.asarray(r.render())
    r.close()
    return fr


def downscale_to_pitch(image: np.ndarray, factor: int) -> np.ndarray:
    """Block-average an image by an integer factor (approximate area filter)."""
    f = max(1, int(factor))
    h, w = image.shape[:2]
    h2, w2 = (h // f) * f, (w // f) * f
    a = np.asarray(image, dtype=float)[:h2, :w2]
    if a.ndim == 3:
        a = a.reshape(h2 // f, f, w2 // f, f, a.shape[2]).mean(axis=(1, 3))
    else:
        a = a.reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))
    return a


def record(preset: str, *, preset_source: str) -> dict:
    """Build one condition, settle, record, and render.  Identical for all."""
    t0 = time.time()
    b = BodyBackend(BodyConfig(seed=SEED, scene_preset=preset, add_vision=True,
                              world_half_size_mm=WORLD_HALF_SIZE_MM,
                              add_world_camera=False), gl_backend="egl")
    rep = b.scene_report
    b.attach_cpg_baseline()
    for _ in range(N_SETTLE):
        b.step()
    frames, omma, thorax, times, legs, contacts = [], [], [], [], [], []
    for _ in range(N_SAMPLES):
        for _ in range(SAMPLE_EVERY):
            b.step()
        obs = b.observe()
        frames.append(np.asarray(b.raw_vision()))
        omma.append(np.asarray(b.ommatidia_readouts()))
        thorax.append(obs.thorax_position_mm.copy())
        times.append(float(obs.time_s))
        legs.append(int(np.sum(obs.contact_present)))
        contacts.append(np.asarray(obs.contact_found_raw, dtype=float))
    frames = np.asarray(frames)          # (T,2,H,W,3) uint8
    omma = np.asarray(omma)              # (T,2,n,2) float32
    thorax = np.asarray(thorax)          # (T,3) mm

    # independent check that the terrain is a surface at a known height and that
    # MuJoCo TILES the hfield beyond its nominal extent (so there is no cliff)
    import mujoco as _mj
    rc = getattr(b.scene_config, "roughness", None)
    half = float(rc.half_extent_mm) if rc is not None else 0.0
    terrain_ray = (ns.raycast_terrain(b.model, b.data, half_extent_mm=half, n=9)
                   if int(getattr(b.model, "nhfield", 0)) else {"available": False})
    terrain_ray_out = (ns.raycast_terrain(b.model, b.data,
                                          half_extent_mm=half * 2.5, n=7)
                       if int(getattr(b.model, "nhfield", 0)) else {"available": False})

    # free-camera world views (fly visible, elevation POSITIVE)
    world_near = _render(b.model, b.data,
                         _free_camera(b.model, distance=6.5, elevation=25.0))
    world_mid = _render(b.model, b.data,
                        _free_camera(b.model, distance=15.0, elevation=25.0))
    # the sky, measured on its own: look UP at 80 deg and sample the top band
    sky = _render(b.model, b.data,
                  _free_camera(b.model, distance=12.0, elevation=-75.0))
    band = sky[:max(8, sky.shape[0] // 4)]
    # ground pixels from the near view: lower-left quadrant is ground for this camera
    ground_px = world_near[int(world_near.shape[0] * 0.55):, :world_near.shape[1] // 3]
    out = {
        "preset": preset,
        "preset_source": preset_source,
        "model": {
            "ngeom": int(b.model.ngeom), "nlight": int(b.model.nlight),
            "nhfield": int(getattr(b.model, "nhfield", 0)),
            "npair": int(b.model.npair), "ncam": int(b.model.ncam),
            "ntex": int(b.model.ntex),
            "eye_fovy_deg_measured": float(b.cfg.eye_fovy_deg_measured),
            "eye_camera_names": dict(b.eye_camera_specs),
            "headlight_measured": {
                "ambient": [float(v) for v in np.ravel(b.model.vis.headlight.ambient)],
                "diffuse": [float(v) for v in np.ravel(b.model.vis.headlight.diffuse)],
                "specular": [float(v) for v in np.ravel(b.model.vis.headlight.specular)],
            },
            "mat_texrepeat": [[float(v) for v in np.ravel(b.model.mat_texrepeat[i])]
                              for i in range(int(b.model.nmat))],
            "mat_names": [str(b.model.material(i).name)
                          for i in range(int(b.model.nmat))],
            "textures": [{"name": str(b.model.texture(i).name),
                          "width": int(np.ravel(b.model.tex_width[i])[0]),
                          "height": int(np.ravel(b.model.tex_height[i])[0])}
                         for i in range(int(b.model.ntex))],
        },
        "times_s": [float(t) for t in times],
        "thorax_mm": thorax.tolist(),
        "legs_in_contact": legs,
        "contact_found_raw_max": float(np.max(np.asarray(contacts))),
        "eye": {
            "shape": list(frames.shape),
            "frac_exactly_255": float((frames == 255).mean()),
            "frac_exactly_255_per_eye": [float((frames[:, i] == 255).mean())
                                         for i in range(frames.shape[1])],
            "mean": float(frames.mean()),
            "std": float(frames.std()),
            "per_frame_std": [float(frames[t].std()) for t in range(frames.shape[0])],
            "min_per_frame_std": float(min(frames[t].std()
                                           for t in range(frames.shape[0]))),
            "unique_values_last_frame": int(len(np.unique(frames[-1]))),
            "channel_histogram_last": {
                "frac_0": float((frames[-1] == 0).mean()),
                "frac_255": float((frames[-1] == 255).mean()),
            },
        },
        "ommatidia": {
            "shape": list(omma.shape),
            "mean": float(omma.mean()),
            "frac_sum_gt_0_98": float((omma.sum(axis=3) > 0.98).mean()),
            "frac_sum_gt_0_999": float((omma.sum(axis=3) > 0.999).mean()),
            "min": float(omma.min()), "max": float(omma.max()),
        },
        "sky_view": {
            "mean": float(sky.mean()), "std": float(sky.std()),
            "unique_values": int(len(np.unique(sky))),
            "frac_exactly_255": float((sky == 255).mean()),
            "top_band_mean": float(band.mean()),
            "top_band_std": float(band.std()),
            "top_band_unique": int(len(np.unique(band))),
            "top_band_frac_255": float((band == 255).mean()),
            "note": ("free camera at elevation -75 deg, i.e. looking up; the "
                     "top band is sky.  The camera RENDERS, so this is the sky "
                     "MuJoCo actually drew, not the rgb values requested."),
        },
        "ground_rendered": {
            "mean": float(ground_px.mean()), "std": float(ground_px.std()),
            "audit": ns.spectral_audit(ground_px, k_floor=4,
                                       band=(0.05, 0.45)),
        },
        "terrain_raycast": terrain_ray,
        "terrain_raycast_beyond_patch": terrain_ray_out,
        "walls": float(time.time() - t0),
    }
    imgs = {"world_near": world_near, "world_mid": world_mid, "sky": sky,
            "eye_left_last": frames[-1, 0], "eye_right_last": frames[-1, 1]}
    b.close()
    return out, imgs


def _g(v, fmt="{:.4f}"):
    return "None" if v is None else fmt.format(v)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("NATURAL SCENE EXPERIMENT -- hand-built prototype, not a finished scene")
    print("=" * 78)
    print(f"protocol: seed {SEED}, {N_SETTLE} settle steps + "
          f"{N_SAMPLES}x{SAMPLE_EVERY} steps = {DUR_S:.2f} s of sim per condition")
    print("pre-registered: "
          f"P1 peak_frac < {P1_PEAK_FRAC_MAX} and slope in {P1_SLOPE_RANGE} | "
          f"P2 frac255 < {P2_FRAC255_TARGET} and <= {P2_RELATIVE_FACTOR}x baseline | "
          f"P3 >= {P3_MIN_DISPLACEMENT_MM} mm, z_end in {P3_THORAX_Z_RANGE_MM} mm | "
          f"P4 gradient sky reduces frac255")

    report: dict = {
        "what_this_is": ("hand-built prototype experiment over a hand-built scene "
                         "layer; not a finished natural-scene system and not a "
                         "biology result"),
        "protocol": {
            "seed": SEED, "n_settle_steps": N_SETTLE, "n_samples": N_SAMPLES,
            "sample_every_steps": SAMPLE_EVERY, "dur_s": DUR_S,
            "world_half_size_mm": WORLD_HALF_SIZE_MM,
            "free_camera": {"azimuth": 135.0, "elevation": 25.0,
                            "distance_mm": 6.5,
                            "note": ("elevation MUST be positive; a negative "
                                     "elevation puts the camera under the ground "
                                     "plane and the frame is blank white")},
            "eye_fovy_deg": "read back from the compiled model (157 deg per "
                            "FlyGym's shipped vision.yaml; NOT settable through "
                            "add_vision())",
            "gl_backend": "egl",
        },
        "pre_registered": {
            "P1": {"peak_frac_metric": ("share of non-DC spectral power in the "
                                        "strongest single bin"),
                   "peak_frac_max": P1_PEAK_FRAC_MAX,
                   "peak_frac_justification": P1_PEAK_FRAC_JUSTIFICATION,
                   "slope_range": list(P1_SLOPE_RANGE),
                   "slope_justification": P1_SLOPE_JUSTIFICATION,
                   "r2_min": P1_R2_MIN},
            "P2": {"frac255_max": P2_FRAC255_TARGET,
                   "relative_to_old_max": P2_RELATIVE_FACTOR,
                   "orchestrator_baseline": list(P2_BASELINE_REPORTED_BY_ORCHESTRATOR)},
            "P3": {"min_displacement_mm": P3_MIN_DISPLACEMENT_MM,
                   "thorax_z_end_range_mm": list(P3_THORAX_Z_RANGE_MM),
                   "min_thorax_z_mm": P3_MIN_THORAX_Z_MM,
                   "min_raw_frame_std": P3_MIN_RAW_FRAME_STD},
            "P4": {"sky_must_reduce_frac255": P4_SKY_MUST_REDUCE_FRAC255,
                   "comparison": "natural_fbm vs fbm_uniform_sky (sky rgb only)"},
        },
    }

    # ------------------------------------------------------------------ assets
    print("\n-- ground texture asset")
    tex_meta = ns.resolve_ground_texture(size=1024, seed=SEED)
    if tex_meta.get("route") != "PROCEDURAL":
        ns.write_provenance_sidecar(tex_meta)
    from PIL import Image
    src = np.asarray(Image.open(tex_meta["path"]))
    # physical size of one texel if the tile is texture_tile_mm across; the
    # texture is then downscaled to the eye camera's own pixel footprint so the
    # audit is done at the scale the eye can actually resolve
    tile_mm = float(ns.NATURAL_PRESETS["natural_fbm"].texture_tile_mm)
    texel_mm = tile_mm / src.shape[1]
    factor = max(1, int(round(EYE_SAMPLE_PITCH_MM / texel_mm)))
    eye_scale_tex = downscale_to_pitch(src, factor)
    tex_audit_src = ns.spectral_audit(src, sample_pitch=texel_mm,
                                     sample_unit="mm", k_floor=4)
    tex_audit_eye = ns.spectral_audit(eye_scale_tex, sample_pitch=texel_mm * factor,
                                      sample_unit="mm", k_floor=4)
    print(f"   route           : {tex_meta.get('route')}")
    print(f"   path            : {tex_meta.get('path')}")
    print(f"   licence         : {tex_meta.get('license_name')!r} "
          f"author={tex_meta.get('author')!r}")
    print(f"   source page     : {tex_meta.get('source_page')}")
    print(f"   source sha256   : {str(tex_meta.get('sha256'))[:16]}...")
    print(f"   tile            : {tile_mm} mm -> texel {texel_mm*1000:.3f} um")
    print(f"   audit source    : slope={_g(tex_audit_src['fitted_slope'],'{:.3f}')} "
          f"peak_frac={_g(tex_audit_src['peak_frac_of_power'],'{:.4f}')}")
    print(f"   audit eye-scale : slope={_g(tex_audit_eye['fitted_slope'],'{:.3f}')} "
          f"peak_frac={_g(tex_audit_eye['peak_frac_of_power'],'{:.4f}')} "
          f"(downscaled x{factor} to a {EYE_SAMPLE_PITCH_MM} mm sample)")
    report["ground_texture_asset"] = {
        "route": tex_meta.get("route"),
        "path": tex_meta.get("path"),
        "license_name": tex_meta.get("license_name"),
        "license_url": tex_meta.get("license_url"),
        "author": tex_meta.get("author"),
        "source_page": tex_meta.get("source_page"),
        "source_url": tex_meta.get("source_url"),
        "sha256": tex_meta.get("sha256"),
        "source_size_px": tex_meta.get("source_size_px"),
        "written_size_px": [int(v) for v in src.shape[:2]],
        "tile_mm": tile_mm,
        "texel_mm": texel_mm,
        "eye_sample_pitch_mm": EYE_SAMPLE_PITCH_MM,
        "eye_scale_downsample_factor": factor,
        "audit_source": tex_audit_src,
        "audit_eye_scale": tex_audit_eye,
        "provenance": tex_meta.get("provenance"),
        "route_decision": tex_meta.get("route_decision"),
        "fallback_reason": tex_meta.get("fallback_reason"),
    }

    # ---------------------------------------------------- old ground (library)
    print("\n-- the OLD ground, read from MuJoCo itself")
    chk = ns.muoco_builtin_checker_field()
    # one MuJoCo checker tile is one rgb1/rgb2 square = half of the texture? no:
    # the builtin checker alternates per PERIOD texels; measure the period from the
    # data instead of assuming it.
    row = np.asarray(chk["image"])[:, 0, 0].astype(float)
    edges = np.flatnonzero(np.diff(row) != 0) + 1
    period_px = int(np.diff(edges).max()) if edges.size > 1 else 0
    pitch_mm_per_px = (ns.OLD_GROUND_TILE_MM
                       / max(1, chk["width"] // max(1, period_px)))
    old_audit = ns.spectral_audit(chk["image"], sample_pitch=pitch_mm_per_px,
                                  sample_unit="mm", k_floor=4)
    old_audit_flat = ns.spectral_audit(chk["image"], sample_pitch=1.0,
                                       sample_unit="px", k_floor=4)
    print(f"   builtin checker {chk['width']}x{chk['height']} px, "
          f"alternation period {period_px} px, {len(np.unique(chk['image']))} "
          f"levels, tile {chk['tile_size_mm']:.3f} mm on the plane")
    print(f"   audit (px pitch): slope={_g(old_audit_flat['fitted_slope'],'{:.3f}')} "
          f"peak_frac={_g(old_audit_flat['peak_frac_of_power'],'{:.4f}')} "
          f"z={_g(old_audit_flat['peak_z_local'],'{:.2f}')}")
    report["old_ground_texture"] = {
        "source": chk["source"],
        "width_px": chk["width"], "height_px": chk["height"],
        "n_levels": int(len(np.unique(chk["image"]))),
        "alternation_period_px": period_px,
        "tile_size_mm_on_plane": chk["tile_size_mm"],
        "sampling_pitch_mm_per_px": pitch_mm_per_px,
        "audit_px_pitch": old_audit_flat,
        "audit_mm_pitch": old_audit,
    }

    # ---------------------------------------------------- old terrain (library)
    print("\n-- the OLD heightfield, regenerated exactly as scene.py builds it")
    import mujoco  # noqa: F401  (scene._heightfield_data needs it internally)
    old_rc = scene.SCENE_PRESETS["natural"].roughness
    old_elev = scene._heightfield_data(None, old_rc)          # noqa: SLF001
    old_terrain_audit = ns.spectral_audit(
        old_elev, sample_pitch=old_rc.grid_cell_mm, sample_unit="mm", k_floor=4)
    print(f"   scene.py grid {old_rc.n_rows}x{old_rc.n_cols}, cell "
          f"{old_rc.grid_cell_mm:.4f} mm, wavelength {old_rc.wavelength_mm} mm")
    print(f"   audit: slope={_g(old_terrain_audit['fitted_slope'],'{:.3f}')} "
          f"peak_frac={_g(old_terrain_audit['peak_frac_of_power'],'{:.4f}')} "
          f"z={_g(old_terrain_audit['peak_z_local'],'{:.2f}')} "
          f"peak wl={_g(old_terrain_audit.get('peak_wavelength'),'{:.3f}')} mm")
    report["old_terrain"] = {
        "generator": "scene._heightfield_data (the shipped formula)",
        "n_rows": int(old_rc.n_rows), "n_cols": int(old_rc.n_cols),
        "cell_size_mm": float(old_rc.grid_cell_mm),
        "nominal_wavelength_mm": float(old_rc.wavelength_mm),
        "half_extent_mm": float(old_rc.half_extent_mm),
        "finite_patch_warning": ("a FINITE 60 mm patch: outside it the old scene "
                                 "has no collision at all"),
        "audit": old_terrain_audit,
    }

    # ----------------------------------------------------------- new terrain
    print("\n-- the NEW terrain")
    rc = ns.NATURAL_PRESETS["natural_fbm"].roughness
    new_elev, new_tmeta = ns.generate_fbm_heightfield(
        n_samples=int(rc.n_rows), extent_mm=float(rc.half_extent_mm),
        hurst=ns.DEFAULT_HURST, seed=int(rc.seed),
        amplitude_std_mm=ns.NATURAL_HFIELD_AMPLITUDE_STD_MM,
        micro_relief_mm=ns.NATURAL_HFIELD_MICRO_MM)
    new_terrain_audit = ns.spectral_audit(
        new_elev, sample_pitch=new_tmeta["cell_size_mm"], sample_unit="mm",
        k_floor=4)
    print(f"   {int(rc.n_rows)}x{int(rc.n_cols)} over "
          f"{2*float(rc.half_extent_mm):.0f} mm, cell "
          f"{new_tmeta['cell_size_mm']:.4f} mm, H={ns.DEFAULT_HURST}")
    print(f"   amplitude: {new_tmeta['amplitude_peak_to_peak_mm']:.4f} mm "
          f"peak-to-peak (std {new_tmeta['amplitude_std_mm']:.4f} mm)")
    print(f"   audit: slope={_g(new_terrain_audit['fitted_slope'],'{:.3f}')} "
          f"r2={_g(new_terrain_audit['fitted_r2'],'{:.3f}')} "
          f"peak_frac={_g(new_terrain_audit['peak_frac_of_power'],'{:.4f}')} "
          f"z={_g(new_terrain_audit['peak_z_local'],'{:.2f}')}")
    print(f"   expectation from H: {new_tmeta['hurst_spectral_slope_expected_2d']:.3f}"
          f" (measured minus expected: "
          f"{new_terrain_audit['fitted_slope'] - new_tmeta['hurst_spectral_slope_expected_2d']:+.3f})")
    report["new_terrain"] = {"metadata": new_tmeta, "audit": new_terrain_audit}

    # --------------------------------------------------------- run conditions
    records, images = {}, {}
    all_presets = list(OLD_PRESETS) + list(NEW_PRESETS)
    for p in all_presets:
        src = "engine.embodied.scene" if p in OLD_PRESETS else \
            "engine.embodied.natural_scene"
        try:
            rec, imgs = record(p, preset_source=src)
        except Exception as exc:                       # keep the run going
            import traceback
            traceback.print_exc()
            records[p] = {"preset": p, "preset_source": src,
                          "error": f"{type(exc).__name__}: {exc}"}
            continue
        rec["outcomes"] = summarise(rec)
        records[p] = rec
        images[p] = imgs
        o = rec["outcomes"]
        print(f"\n-- {p}  [{src}]")
        print(f"   model: ngeom={rec['model']['ngeom']} nlight="
              f"{rec['model']['nlight']} nhfield={rec['model']['nhfield']} "
              f"ntex={rec['model']['ntex']} ncam={rec['model']['ncam']} "
              f"eye_fovy={rec['model']['eye_fovy_deg_measured']:.1f} deg")
        print(f"   headlight: {rec['model']['headlight_measured']}")
        print(f"   eye frac==255 = {o['frac255']:.4f}  "
              f"(per eye {[round(v,4) for v in rec['eye']['frac_exactly_255_per_eye']]}) "
              f"mean={rec['eye']['mean']:.1f} min_frame_std="
              f"{rec['eye']['min_per_frame_std']:.2f} "
              f"unique(last frame)={rec['eye']['unique_values_last_frame']}")
        print(f"   ommatidia: frac>0.98 = {rec['ommatidia']['frac_sum_gt_0_98']:.4f} "
              f"mean={rec['ommatidia']['mean']:.4f}")
        print(f"   sky view  : mean={rec['sky_view']['mean']:.1f} "
              f"std={rec['sky_view']['std']:.2f} uniq="
              f"{rec['sky_view']['unique_values']} "
              f"frac255={rec['sky_view']['frac_exactly_255']:.4f}")
        print(f"   walk      : {o['displacement_mm']:.3f} mm in {DUR_S:.2f} s, "
              f"z_end={o['thorax_z_end_mm']:.3f} mm, z_min="
              f"{o['thorax_z_min_mm']:.3f} mm, finite={o['all_finite']}, "
              f"legs={o['mean_legs_in_contact']:.2f}")
        print(f"   rendered ground px: mean={rec['ground_rendered']['mean']:.1f} "
              f"peak_frac={_g(rec['ground_rendered']['audit']['peak_frac_of_power'],'{:.4f}')} "
              f"slope={_g(rec['ground_rendered']['audit']['fitted_slope'],'{:.3f}')}")

    # ----------------------------------------------------------- predictions
    print("\n" + "=" * 78)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 78)
    old_ok = [p for p in OLD_PRESETS if "error" not in records[p]]
    new_ok = [p for p in NEW_PRESETS if "error" not in records[p]]
    if not new_ok or not old_ok:
        print("a condition failed to build; predictions cannot be evaluated")
        report["predictions"] = {"error": "a condition failed to build",
                                 "records_with_error":
                                     [p for p in all_presets
                                      if "error" in records.get(p, {})]}
        write_outputs(report, records, images, tex_meta, chk, old_elev,
                      new_elev, new_terrain_audit, old_terrain_audit, tex_audit_src,
                      tex_audit_eye)
        return 1

    nat = records["natural_fbm"]["outcomes"]
    old_frac = {p: records[p]["outcomes"]["frac255"] for p in old_ok}
    newest = max(old_frac.values())

    # ---- P1
    p1_terrain_pass = (new_terrain_audit["peak_frac_of_power"] is not None
                       and new_terrain_audit["peak_frac_of_power"]
                       < P1_PEAK_FRAC_MAX)
    p1_tex_pass = (tex_audit_src["peak_frac_of_power"] is not None
                   and tex_audit_src["peak_frac_of_power"] < P1_PEAK_FRAC_MAX)
    p1_tex_eye_pass = (tex_audit_eye["peak_frac_of_power"] is not None
                       and tex_audit_eye["peak_frac_of_power"] < P1_PEAK_FRAC_MAX)
    p1_slope_pass = (new_terrain_audit["fitted_slope"] is not None
                     and P1_SLOPE_RANGE[0] < new_terrain_audit["fitted_slope"]
                     < P1_SLOPE_RANGE[1])
    p1_r2_pass = (new_terrain_audit["fitted_r2"] is not None
                  and new_terrain_audit["fitted_r2"] >= P1_R2_MIN)
    old_checker_peaks = (old_terrain_audit["peak_frac_of_power"] is not None
                         and old_terrain_audit["peak_frac_of_power"]
                         >= P1_PEAK_FRAC_MAX)
    p1 = bool(p1_terrain_pass and p1_tex_pass and p1_tex_eye_pass
              and p1_slope_pass and p1_r2_pass)
    print(f"\nP1 periodicity is gone: {'PASS' if p1 else 'FAIL'}")
    print(f"   terrain  peak_frac={_g(new_terrain_audit['peak_frac_of_power'])} "
          f"< {P1_PEAK_FRAC_MAX} -> {p1_terrain_pass}")
    print(f"   texture  peak_frac={_g(tex_audit_src['peak_frac_of_power'])} "
          f"< {P1_PEAK_FRAC_MAX} -> {p1_tex_pass}  (source PNG)")
    print(f"   texture at eye sampling pitch peak_frac="
          f"{_g(tex_audit_eye['peak_frac_of_power'])} -> {p1_tex_eye_pass}")
    print(f"   terrain  slope={_g(new_terrain_audit['fitted_slope'],'{:.3f}')} in "
          f"{P1_SLOPE_RANGE} -> {p1_slope_pass}   r2="
          f"{_g(new_terrain_audit['fitted_r2'],'{:.3f}')} >= {P1_R2_MIN} -> {p1_r2_pass}")
    print(f"   [old terrain] peak_frac="
          f"{_g(old_terrain_audit['peak_frac_of_power'])} "
          f"slope={_g(old_terrain_audit['fitted_slope'],'{:.3f}')}")
    print(f"   [old ground texture] peak_frac="
          f"{_g(old_audit_flat['peak_frac_of_power'])} "
          f"slope={_g(old_audit_flat['fitted_slope'],'{:.3f}')}")

    # ---- P2
    p2_abs = bool(nat["frac255"] < P2_FRAC255_TARGET)
    p2_rel = bool(nat["frac255"] <= newest * P2_RELATIVE_FACTOR)
    p2 = bool(p2_abs and p2_rel)
    print(f"\nP2 saturation is fixed: {'PASS' if p2 else 'FAIL'}")
    for p in all_presets:
        if p in records and "outcomes" in records[p]:
            print(f"   {p:28s} frac==255 = "
                  f"{records[p]['outcomes']['frac255']:.4f}")
    print(f"   old baseline (max over old presets) = {newest:.4f}; "
          f"orchestrator reported {P2_BASELINE_REPORTED_BY_ORCHESTRATOR}")
    print(f"   natural_fbm = {nat['frac255']:.4f} < {P2_FRAC255_TARGET} -> {p2_abs}"
          f";  <= {P2_RELATIVE_FACTOR} * {newest:.4f} = "
          f"{P2_RELATIVE_FACTOR*newest:.4f} -> {p2_rel}")

    # ---- P3
    p3_move = bool(nat["displacement_mm"] >= P3_MIN_DISPLACEMENT_MM)
    p3_up = bool(P3_THORAX_Z_RANGE_MM[0] <= nat["thorax_z_end_mm"]
                 <= P3_THORAX_Z_RANGE_MM[1])
    p3_finite = bool(nat["all_finite"])
    p3_nofall = bool(nat["thorax_z_min_mm"] >= P3_MIN_THORAX_Z_MM)
    p3_nonblank = bool(nat["min_raw_frame_std"] > P3_MIN_RAW_FRAME_STD)
    p3 = bool(p3_move and p3_up and p3_finite and p3_nofall and p3_nonblank)
    print(f"\nP3 still a valid simulator: {'PASS' if p3 else 'FAIL'}")
    print(f"   displacement {nat['displacement_mm']:.3f} mm >= "
          f"{P3_MIN_DISPLACEMENT_MM} -> {p3_move}   (CPG forward bias, NOT steering)")
    print(f"   thorax z_end {nat['thorax_z_end_mm']:.3f} mm in "
          f"{P3_THORAX_Z_RANGE_MM} -> {p3_up}")
    print(f"   thorax z_min {nat['thorax_z_min_mm']:.3f} mm >= "
          f"{P3_MIN_THORAX_Z_MM} -> {p3_nofall}")
    print(f"   all finite -> {p3_finite};  min eye-frame raw std "
          f"{nat['min_raw_frame_std']:.2f} > {P3_MIN_RAW_FRAME_STD} -> {p3_nonblank}")
    _tr = records["natural_fbm"].get("terrain_raycast") or {}
    print("   contact roots: hfield raycast says terrain z in "
          f"{_tr.get('min_mm')} .. {_tr.get('max_mm')} mm "
          f"(peak-to-peak {_tr.get('peak_to_peak_mm')} mm, all rays hit "
          f"{_tr.get('all_rays_hit')})")

    # ---- P4
    if "fbm_uniform_sky" in records and "outcomes" in records["fbm_uniform_sky"]:
        uni = records["fbm_uniform_sky"]["outcomes"]
        d_frac = uni["frac255"] - nat["frac255"]
        d_mean = uni["eye_mean"] - nat["eye_mean"]
        sky_changed = (records["natural_fbm"]["sky_view"]["unique_values"]
                       > records["fbm_uniform_sky"]["sky_view"]["unique_values"])
        p4 = bool(d_frac > 0 and sky_changed)
        print(f"\nP4 the sky's contribution is isolated: "
              f"{'PASS' if p4 else 'FAIL'}")
        print(f"   natural_fbm    frac255={nat['frac255']:.4f} "
              f"eye mean={nat['eye_mean']:.2f}  sky(mean/std/uniq)="
              f"{records['natural_fbm']['sky_view']['mean']:.1f}/"
              f"{records['natural_fbm']['sky_view']['std']:.2f}/"
              f"{records['natural_fbm']['sky_view']['unique_values']}")
        print(f"   fbm_uniform_sky frac255={uni['frac255']:.4f} "
              f"eye mean={uni['eye_mean']:.2f}  sky(mean/std/uniq)="
              f"{records['fbm_uniform_sky']['sky_view']['mean']:.1f}/"
              f"{records['fbm_uniform_sky']['sky_view']['std']:.2f}/"
              f"{records['fbm_uniform_sky']['sky_view']['unique_values']}")
        print(f"   frac255 difference (uniform - gradient) = {d_frac:+.4f}; "
              f"eye-mean difference = {d_mean:+.2f}")
        if abs(d_frac) < 1e-6:
            print("   NOTE: the eye images are IDENTICAL between the two sky "
                  "conditions in this protocol.  The eye cameras point at the "
                  "ground, so the sky is (almost) not in their field of view; the "
                  "clipping was fixed by the headlight and material rebalance, not "
                  "by the sky.  That is a real negative result about the EYE "
                  "images, and it is reported as one.")
    else:
        uni, d_frac, d_mean, sky_changed, p4 = None, None, None, None, None
        print("\nP4: control condition did not build -- NOT EVALUATED")

    predictions = {
        "P1_periodicity_gone": {
            "verdict": "PASS" if p1 else "FAIL",
            "terrain_peak_frac": new_terrain_audit["peak_frac_of_power"],
            "terrain_peak_frac_threshold": P1_PEAK_FRAC_MAX,
            "texture_peak_frac_source": tex_audit_src["peak_frac_of_power"],
            "texture_peak_frac_eye_scale": tex_audit_eye["peak_frac_of_power"],
            "terrain_slope": new_terrain_audit["fitted_slope"],
            "terrain_slope_range": list(P1_SLOPE_RANGE),
            "terrain_slope_r2": new_terrain_audit["fitted_r2"],
            "terrain_r2_min": P1_R2_MIN,
            "old_terrain_peak_frac": old_terrain_audit["peak_frac_of_power"],
            "old_terrain_slope": old_terrain_audit["fitted_slope"],
            "old_ground_texture_peak_frac": old_audit_flat["peak_frac_of_power"],
            "old_ground_texture_slope": old_audit_flat["fitted_slope"],
            "components": {"terrain_peak": p1_terrain_pass,
                           "texture_peak_source": p1_tex_pass,
                           "texture_peak_eye_scale": p1_tex_eye_pass,
                           "terrain_slope": p1_slope_pass, "terrain_r2": p1_r2_pass},
            "note_on_metric_choice": (
                "peak / median MAGNITUDE is reported too but is NOT the "
                "threshold metric: measured on this stack a natural fBm patch "
                "with no grating scored 195-217 on it, while a 1-D grating scored "
                "~6e15, so any fixed threshold on it would be a window-size "
                "artefact."),
        },
        "P2_saturation_fixed": {
            "verdict": "PASS" if p2 else "FAIL",
            "frac255_by_preset": {p: records[p]["outcomes"]["frac255"]
                                  for p in all_presets
                                  if p in records and "outcomes" in records[p]},
            "natural_fbm_frac255": nat["frac255"],
            "absolute_target": P2_FRAC255_TARGET,
            "relative_target_factor": P2_RELATIVE_FACTOR,
            "old_baseline_max": newest,
            "orchestrator_reported_baseline": list(P2_BASELINE_REPORTED_BY_ORCHESTRATOR),
            "components": {"absolute": p2_abs, "relative": p2_rel},
        },
        "P3_still_valid": {
            "verdict": "PASS" if p3 else "FAIL",
            "displacement_mm": nat["displacement_mm"],
            "min_displacement_mm": P3_MIN_DISPLACEMENT_MM,
            "thorax_z_end_mm": nat["thorax_z_end_mm"],
            "thorax_z_end_range_mm": list(P3_THORAX_Z_RANGE_MM),
            "thorax_z_min_mm": nat["thorax_z_min_mm"],
            "min_thorax_z_mm": P3_MIN_THORAX_Z_MM,
            "all_finite": nat["all_finite"],
            "min_raw_frame_std": nat["min_raw_frame_std"],
            "min_raw_frame_std_required": P3_MIN_RAW_FRAME_STD,
            "mean_legs_in_contact": nat["mean_legs_in_contact"],
            "contact_found_raw_max": records["natural_fbm"]["contact_found_raw_max"],
            "components": {"moved": p3_move, "upright": p3_up, "finite": p3_finite,
                           "no_fall_through": p3_nofall, "nonblank": p3_nonblank},
        },
        "P4_sky_isolated": {
            "verdict": ("PASS" if p4 else "FAIL") if p4 is not None else "NOT RUN",
            "natural_fbm_frac255": nat["frac255"],
            "fbm_uniform_sky_frac255": (uni["frac255"] if uni else None),
            "frac255_difference": d_frac,
            "natural_fbm_eye_mean": nat["eye_mean"],
            "fbm_uniform_sky_eye_mean": (uni["eye_mean"] if uni else None),
            "sky_view_unique_values": {
                "natural_fbm": records["natural_fbm"]["sky_view"]["unique_values"],
                "fbm_uniform_sky": (records["fbm_uniform_sky"]["sky_view"]["unique_values"]
                                    if uni else None)},
            "sky_changed_by_rgb": sky_changed,
            "interpretation": (
                "the two conditions differ ONLY in the skybox rgb values, so this "
                "difference is the sky's contribution to the eye images.  A "
                "difference of ~0 in the eye frames with a clearly changed sky "
                "view means the sky is essentially out of the eye cameras' field "
                "of view in this protocol."),
        },
    }
    report["predictions"] = predictions
    report["records"] = {p: records[p] for p in all_presets if p in records}

    write_outputs(report, records, images, tex_meta, chk, old_elev, new_elev,
                  new_terrain_audit, old_terrain_audit, tex_audit_src, tex_audit_eye)

    verdicts = [predictions[k]["verdict"] for k in
                ("P1_periodicity_gone", "P2_saturation_fixed", "P3_still_valid",
                 "P4_sky_isolated")]
    print("\nVERDICTS: " + "  ".join(
        f"{k.split('_')[0]}={v}" for k, v in zip(
            ("P1", "P2", "P3", "P4"), verdicts)))
    print(f"wrote {JSON_PATH}")
    print(f"wrote {PNG_OVERVIEW}")
    print(f"wrote {PNG_SPECTRA}")
    return 0 if all(v == "PASS" for v in verdicts) else 2


def summarise(rec: dict) -> dict:
    th = np.asarray(rec["thorax_mm"], dtype=float)
    fr = rec["eye"]
    return {
        "displacement_mm": float(np.linalg.norm(th[-1, :2] - th[0, :2])),
        "thorax_z_end_mm": float(th[-1, 2]),
        "thorax_z_min_mm": float(th[:, 2].min()),
        "thorax_z_max_mm": float(th[:, 2].max()),
        "all_finite": bool(np.isfinite(th).all()),
        "mean_legs_in_contact": float(np.mean(rec["legs_in_contact"])),
        "frac255": float(fr["frac_exactly_255"]),
        "eye_mean": float(fr["mean"]),
        "eye_std": float(fr["std"]),
        "min_raw_frame_std": float(fr["min_per_frame_std"]),
        "ommatidia_frac_gt_0_98": float(rec["ommatidia"]["frac_sum_gt_0_98"]),
        "sky_mean": float(rec["sky_view"]["mean"]),
        "sky_std": float(rec["sky_view"]["std"]),
    }


def write_outputs(report, records, images, tex_meta, chk, old_elev, new_elev,
                  new_audit, old_audit, tex_audit_src, tex_audit_eye) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------ overview
    order = [p for p in ("blank", "natural", "natural_fbm",
                         "natural_fbm_lit_no_objects", "fbm_uniform_sky")
             if p in images]
    if order:
        fig, axes = plt.subplots(3, len(order), figsize=(3.5 * len(order), 10.5))
        if len(order) == 1:
            axes = axes.reshape(3, 1)
        for c, p in enumerate(order):
            im = images[p]
            axes[0, c].imshow(im["eye_left_last"])
            axes[0, c].set_title(f"{p}\nLEFT eye, raw fisheye (FlyGym optics,\n"
                                 f"approx.) frac255="
                                 f"{records[p]['outcomes']['frac255']:.4f}",
                                 fontsize=8)
            axes[1, c].imshow(im["world_near"])
            axes[1, c].set_title("free camera, d=6.5 mm, az=135, elev=+25 deg",
                                 fontsize=8)
            axes[2, c].imshow(im["world_mid"])
            axes[2, c].set_title("free camera, d=15 mm, az=135, elev=+25 deg",
                                 fontsize=8)
            for r in range(3):
                axes[r, c].axis("off")
        fig.suptitle(
            "Natural scene: old (blank = uniform white sky + periodic 8 mm "
            "checker; natural = scene.py heightfield + checker + white sky) vs "
            "new (natural_fbm = fBm terrain + CC0 leaf-litter texture + gradient "
            "sky + rebalanced headlight)\n"
            "FlyGym/NeuroMechFly model, units mm, fly ~2.5 mm.  Eye optics are "
            "FlyGym's and are APPROXIMATE.  Free-camera elevation must be "
            "POSITIVE.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(PNG_OVERVIEW, dpi=110)
        plt.close(fig)

    # ------------------------------------------------------------- spectra
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    def _rad_plot(ax, audit, label, colour):
        rad = audit["radial"]
        k = np.asarray(rad["k"]); p = np.asarray(rad["power"])
        m = (k > 0) & (p > 0)
        ax.loglog(k[m], p[m], colour, lw=1.4, label=label)
    ax = axes[0, 0]
    _rad_plot(ax, new_audit, "new fBm terrain", "tab:green")
    _rad_plot(ax, old_audit, "old scene.py terrain", "tab:red")
    ax.set_title(f"radial power spectrum, TERRAIN\nnew slope="
                 f"{_g(new_audit['fitted_slope'],'{:.2f}')} (r2 "
                 f"{_g(new_audit['fitted_r2'],'{:.2f}')}), old slope="
                 f"{_g(old_audit['fitted_slope'],'{:.2f}')}", fontsize=9)
    ax.set_xlabel("k (cycles/sample)"); ax.set_ylabel("mean power"); ax.legend(fontsize=7)

    ax = axes[0, 1]
    _rad_plot(ax, tex_audit_src, "new ground texture (CC0 photo)", "tab:green")
    _rad_plot(ax, tex_audit_eye, f"new texture at eye pitch", "tab:olive")
    _rad_plot(ax, ns.spectral_audit(chk["image"], k_floor=4),
              "old builtin checker texture", "tab:red")
    ax.set_title("radial power spectrum, GROUND TEXTURE\nnew slope="
                 f"{_g(tex_audit_src['fitted_slope'],'{:.2f}')}, old checker "
                 f"slope="
                 f"{_g(ns.spectral_audit(chk['image'], k_floor=4)['fitted_slope'],'{:.2f}')}",
                 fontsize=9)
    ax.set_xlabel("k (cycles/sample)"); ax.legend(fontsize=7)

    ax = axes[0, 2]
    ax.imshow(new_elev, cmap="terrain")
    ax.set_title(f"NEW terrain elevation (normalised)\n{new_elev.shape[0]}x"
                 f"{new_elev.shape[1]}, fBm H={ns.DEFAULT_HURST}, no grating",
                 fontsize=9)
    ax.set_xlabel("sample"); ax.set_ylabel("sample")

    ax = axes[1, 0]
    ax.imshow(old_elev, cmap="terrain")
    ax.set_title("OLD scene.py heightfield\nsingle sine pair + white noise; the "
                 "dominant wavelength is visible", fontsize=9)
    ax.set_xlabel("sample"); ax.set_ylabel("sample")

    ax = axes[1, 1]
    ax.imshow(chk["image"])
    ax.set_title(f"OLD ground texture: MuJoCo builtin checker "
                 f"{chk['width']}x{chk['height']} px, "
                 f"{len(np.unique(chk['image']))} grey levels\n"
                 f"tile {chk['tile_size_mm']:.1f} mm on a 2.5 mm fly",
                 fontsize=9)

    ax = axes[1, 2]
    from PIL import Image
    ax.imshow(np.asarray(Image.open(tex_meta["path"])))
    ax.set_title(f"NEW ground texture: {tex_meta.get('route')}\n"
                 f"{os.path.basename(str(tex_meta.get('path')))} "
                 f"{tex_meta.get('license_name') or ''} "
                 f"{tex_meta.get('author') or ''}", fontsize=9)
    ax.axis("off")

    # the periodicity numbers, old vs new, as text
    txt = (
        "PERIODICITY AUDIT (peak_frac = share of non-DC power in the single\n"
        "strongest bin; aperiodic ~1/n_bins, a grating concentrates it)\n"
        f"  new terrain   peak_frac={_g(new_audit['peak_frac_of_power'])}  "
        f"z_local={_g(new_audit['peak_z_local'],'{:.2f}')}  "
        f"ratio_to_median={_g(new_audit['peak_ratio_to_median'],'{:.1f}')}\n"
        f"  old terrain   peak_frac={_g(old_audit['peak_frac_of_power'])}  "
        f"z_local={_g(old_audit['peak_z_local'],'{:.2f}')}  "
        f"ratio_to_median={_g(old_audit['peak_ratio_to_median'],'{:.1f}')}\n"
        f"  new texture   peak_frac={_g(tex_audit_src['peak_frac_of_power'])}\n"
        f"  old texture   peak_frac="
        f"{_g(ns.spectral_audit(chk['image'], k_floor=4)['peak_frac_of_power'])}\n"
        f"  threshold     peak_frac < {P1_PEAK_FRAC_MAX} = 'no dominant grating'")
    fig.text(0.5, 0.005, txt, ha="center", fontsize=8, family="monospace")
    fig.suptitle("Natural scene: radial power spectra and the periodicity audit, "
                 "old vs new", fontsize=11)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(PNG_SPECTRA, dpi=110)
    plt.close(fig)

    with open(JSON_PATH, "w") as f:
        json.dump(scene._jsonable(report), f, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
