#!/usr/bin/env python3
"""RECORDING DATA-HEALTH GATE.  Can this episode's data be used at all?

WHY THIS EXISTS, and it is a measured reason rather than a good habit.  One episode in this
project (`outputs/arena/arena_bare_seed0`) was recorded while the recorder's blob-AREA filter was
hard-coded to 60 px -- a bound valid only for a 200-ROW frame.  At 640x800 a 0.30 mm fiducial
covers about 490 px, so every fiducial was rejected as "too large" and the recorded `px/` table
kept only grass specks.  Nothing crashed.  A reconstruction then fitted three of those specks and
reported a world-frame RMS of 0.119 mm, which is a PLAUSIBLE WRONG NUMBER -- the worst failure
mode there is, and it was nearly delivered as a fix.

MEASURED SEPARATION, which is where this gate's threshold comes from.  For each of the 12 declared
fiducials, in each camera, in each of the first 20 frames, take the distance to the NEAREST
recorded candidate and ask whether it is within 3 px:

    arena_bare_seed0    1.4 %   <- the broken recording
    arena_bare_seed1   93.1 %
    arena_bare_seed2   94.0 %
    arena_bare_seed3   93.8 %

So the split is not a guess: every usable episode is above 93 % and the broken one is at 1.4 %.
The registered threshold is 50 %, deliberately in the middle of an empty band rather than close to
either cluster.

WHAT THIS GATE IS NOT.  It judges whether the RECORDED DETECTIONS can support downstream work.  It
does NOT judge geometric accuracy (that needs calibration residuals) and it does NOT make an
episode's numbers right by passing -- it only refuses to let a broken one proceed silently.

CONVENTION, taken from this project's own bench protocol: exit code 0 means THE CHECK RAN, not that
it passed.  Automation must read `status` from the JSON.  `NOT_RUN` means data was missing, and
NOT_RUN is never treated as a pass.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_episode_health.py --episodes outputs/arena/arena_bare_seed0 \
        outputs/arena/arena_bare_seed3
    ./venv/bin/python tools_episode_health.py --all          # every arena episode
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ARENA = HERE / "outputs" / "arena"

#: Registered thresholds.  `provenance` records WHERE the number came from, so a reader can
#: disagree with the number without having to reverse-engineer it.
THRESHOLDS = {
    "fiducial_hit_rate_3px": {
        "value": 0.50,
        "provenance": ("measured: usable episodes 93.1-94.0 %, broken episode 1.4 %; the threshold "
                       "sits in the empty band between the two clusters"),
        "direction": ">=",
    },
    # THE CANDIDATE FLOOR IS MARKER-RELATIVE, NOT AN ABSOLUTE COUNT.  MEASURED, and pointing the
    # chain at a second body stack is what exposed it: the floor was 20, measured on stack A's
    # GRASS scene where a camera-frame contains 228-398 specks.  The newly observable stack B
    # renders a bare checkerboard floor with ONLY the 12 fiducials on it, so 11-12 candidates is
    # exactly right and the absolute floor failed a perfectly good recording.  A scene-dependent
    # number recorded as if it were universal is the same mistake as an unstated unit.  The floor is
    # now half the declared marker count: a detector that finds nothing, or finds only a couple of
    # things, still fails.
    "candidate_count_min_fraction_of_markers": {
        "value": 0.5,
        "provenance": ("replaces an absolute floor of 20 that was measured on a cluttered grass "
                       "scene; stack B's bare-floor recording has 12 fiducials and 11-12 candidates"),
        "direction": ">=",
    },
    "candidate_count_max": {
        "value": 20000,
        "provenance": "sanity bound; a value this large means the threshold collapsed to noise",
        "direction": "<=",
    },
    # THE BRIGHTNESS FLOOR IS ALSO RELATIVE TO WHAT THE SCENE CAN PRODUCE.  MEASURED: stack A's
    # fiducials saturate at 255 against grass at ~200, so 250 looked universal.  Stack B renders the
    # same emissive material against a dark floor and its markers come out WARM -- RGB
    # [255, 255, ~190] -- so the MEAN is ~242 and the 250 floor failed.  The test that matters is
    # "the detected objects are much brighter than a scene without them", not a particular value.
    "peak_grey_min": {
        "value": 200.0,
        "provenance": ("replaces an absolute floor of 250 fitted to a bright grass scene; stack B's "
                       "dark-floor markers render at a mean peak of 242 because the blue channel is "
                       "suppressed (RGB 255,255,190), and 200 still separates markers from floor"),
        "direction": ">=",
    },
    "calibration_residual_px": {
        "value": 1.0,
        "provenance": ("measured on a usable episode: median |residual| after the 2-parameter "
                       "per-camera offset fit is 0.150 px"),
        "direction": "<=",
    },
}

HIT_RATE_GATE_PX = 3.0
FRAMES_FOR_HIT_RATE = 20


def _check(name: str, ok, value, threshold_key: str | None = None, note: str = "",
           required: bool = True) -> dict:
    """One check.  `ok is None` means it could not be evaluated.

    A check that could not run for a MISSING-AND-REQUIRED reason is NOT_RUN; one that simply does
    not apply to this episode is SKIP.  Conflating the two is how "we did not measure it" quietly
    becomes "it is fine", so they are kept apart: NOT_RUN blocks a PASS, SKIP does not.
    """
    t = THRESHOLDS.get(threshold_key) if threshold_key else None
    if ok is None:
        status = "NOT_RUN" if required else "SKIP"
    else:
        status = "PASS" if ok else "FAIL"
    out = {"check": name, "status": status, "value": value, "required": bool(required)}
    if t is not None:
        out.update({"threshold": t["value"], "direction": t["direction"],
                    "threshold_provenance": t["provenance"],
                    "criterion": f"{name} {t['direction']} {t['value']}"})
    if note:
        out["note"] = note
    return out


def health(ep: Path) -> dict:
    """Every check for ONE episode.  Reads ONLY px/, frames/ and arena_geometry.json.

    It deliberately does NOT read `truth/`: this gate judges the RECORDING, and a gate that
    peeked at the simulator's own state could be fooled into passing a broken recording.
    """
    checks: list[dict] = []
    npz = ep / "episode.npz"
    if not npz.is_file():
        checks.append(_check("episode_npz_present", None, str(npz), note="missing"))
        return _summarise(ep, checks)

    from run_arena_reconstruct import (  # noqa: E402
        PinholeCamera, load_cameras_for_episode, load_projection_offsets)
    geom = json.loads((ARENA / "arena_geometry.json").read_text())
    declared = np.array([m["pos_mm"] for m in geom["markers"]["markers"]], dtype=float)
    n_markers = len(declared)

    z = np.load(npz, allow_pickle=True)
    keys = set(z.files)
    cams_present = sorted({k.split("/")[1] for k in keys if k.startswith("px/cam")})
    checks.append(_check("px_group_present", len(cams_present) > 0,
                         f"{len(cams_present)} cameras: {cams_present}"))

    # ---- frame bookkeeping ---------------------------------------------------------------
    n_frames = int(np.asarray(z["px/cam0/n_candidates"]).shape[0])
    off_len = int(np.asarray(z["px/cam0/offsets"]).shape[0])
    checks.append(_check("px_offsets_length", off_len == n_frames + 1,
                         {"offsets": off_len, "n_frames": n_frames}))

    png_counts = {}
    for c in cams_present:
        d = ep / "frames" / c
        png_counts[c] = len(list(d.glob("*.png"))) if d.is_dir() else 0
    ok_png = all(v == n_frames for v in png_counts.values()) and png_counts
    checks.append(_check("png_count_matches_frames", bool(ok_png),
                         {"n_frames": n_frames, "png_per_camera": png_counts}))

    # ---- candidate counts and peak greyness ----------------------------------------------
    med_counts, med_peaks = {}, {}
    for c in cams_present:
        nc = np.asarray(z[f"px/{c}/n_candidates"], dtype=float)
        med_counts[c] = float(np.median(nc))
        if f"px/{c}/peak_grey" in keys:
            med_peaks[c] = float(np.median(np.asarray(z[f"px/{c}/peak_grey"], dtype=float)))
    if med_counts:
        lo = min(med_counts.values())
        hi = max(med_counts.values())
        floor = THRESHOLDS["candidate_count_min_fraction_of_markers"]["value"] * n_markers
        checks.append(_check("candidate_count_lower", lo >= floor,
                             {"min_camera_median": lo, "floor": floor,
                              "n_declared_markers": n_markers},
                             "candidate_count_min_fraction_of_markers"))
        checks.append(_check("candidate_count_upper", hi <= THRESHOLDS["candidate_count_max"]["value"],
                             {"max_camera_median": hi}, "candidate_count_max"))
    if med_peaks:
        checks.append(_check("peak_grey_bright_enough",
                             min(med_peaks.values()) >= THRESHOLDS["peak_grey_min"]["value"],
                             {"per_camera_median_peak_grey": med_peaks}, "peak_grey_min"))

    # ---- THE DECISIVE ONE: are the fiducials actually in the recorded table? --------------
    # camera geometry is needed to project; use the compiled poses plus measured offsets when
    # present.  A missing offsets file is fine -- it only shifts projections by a fraction of a
    # pixel, far below the 3 px gate, so it cannot change a 1 % -vs- 94 % verdict.
    # LOAD THE MEASURED PER-CAMERA OFFSET BEFORE PROJECTING.  MEASURED DEFECT IN THIS GATE, found
    # by pointing it at a second body stack: it never called `load_projection_offsets`, so it
    # projected WITHOUT the calibration it exists to respect.  Stack A's episodes hid it -- their
    # offset is about 1.5 px at 640 rows, inside the 3 px gate -- while the newly observable stack B
    # episode has a 4.0 px offset and scored 0 %.  A gate that ignores the calibration it is given
    # cannot judge a recording's geometry, and it was one unfortunate episode away from failing
    # everything.
    off_doc = load_projection_offsets(ep / "camera_offsets.json")
    cams, _src = load_cameras_for_episode(geom, ep, ARENA / "camera_poses.json")
    offsets_note = (f"measured per-camera offsets applied "
                    f"(median |r| after {off_doc.get('median_abs_after'):.3f} px)"
                    if off_doc else "no camera_offsets.json: projections are UNoffset")
    hit = tot = 0
    n_use = min(n_frames, FRAMES_FOR_HIT_RATE)
    for f in range(n_use):
        for cam in cams:
            kr, kc, ko = f"px/{cam.name}/rows", f"px/{cam.name}/cols", f"px/{cam.name}/offsets"
            if not (kr in keys and ko in keys):
                continue
            rows = np.asarray(z[kr], dtype=float)
            cols = np.asarray(z[kc], dtype=float)
            off = np.asarray(z[ko], dtype=np.int64)
            r0, c0 = rows[off[f]:off[f + 1]], cols[off[f]:off[f + 1]]
            if r0.size == 0:
                tot += n_markers
                continue
            for k in range(n_markers):
                pr = cam.project(declared[k])[0]
                tot += 1
                if float(np.min(np.hypot(r0 - pr[0], c0 - pr[1]))) <= HIT_RATE_GATE_PX:
                    hit += 1
    # record whether the offsets were in play, so the verdict can never be read without that fact
    checks.append(_check("projection_offsets_available", off_doc is not None, offsets_note,
                         required=False))
    if tot == 0:
        checks.append(_check("fiducial_hit_rate_3px", None, None, note="no projections evaluated"))
    else:
        rate = hit / tot
        checks.append(_check("fiducial_hit_rate_3px",
                             rate >= THRESHOLDS["fiducial_hit_rate_3px"]["value"],
                             {"rate": round(rate, 4), "hits": hit, "evaluated": tot,
                              "frames": n_use, "gate_px": HIT_RATE_GATE_PX},
                             "fiducial_hit_rate_3px",
                             note=("a rate near zero means the fiducials are ABSENT from the "
                                   "recorded px/ table, whatever the images show")))

    # ---- calibration residual, when a calibration exists ---------------------------------
    cof = ep / "camera_offsets.json"
    if cof.is_file():
        doc = json.loads(cof.read_text())
        v = doc.get("median_abs_after")
        checks.append(_check("calibration_residual_px",
                             (v is not None and v <= THRESHOLDS["calibration_residual_px"]["value"]),
                             v, "calibration_residual_px"))
    else:
        # OPTIONAL, not required: a reconstruction runs without it and simply forgoes the
        # sub-pixel per-camera offset (seed3 was reconstructed exactly that way).  So its absence
        # is SKIP -- recorded, not silently treated as a pass, and not blocking the verdict.
        checks.append(_check("calibration_residual_px", None, None, required=False,
                             note="no camera_offsets.json for this episode; the sub-pixel offset "
                                  "is unavailable, which costs accuracy, not usability"))


    return _summarise(ep, checks)


def _summarise(ep: Path, checks: list[dict]) -> dict:
    """FAIL beats NOT_RUN beats PASS; SKIP is ignored, because it means "does not apply"."""
    req = [c["status"] for c in checks if c.get("required", True)]
    if "FAIL" in req:
        overall = "FAIL"
    elif "NOT_RUN" in req:
        overall = "NOT_RUN"
    else:
        overall = "PASS"
    return {"episode": ep.name, "path": str(ep), "status": overall, "checks": checks,
            "convention": ("exit code 0 means the check ran, not that it passed; read `status`. "
                           "NOT_RUN means data was missing and is never a pass."),
            "what_this_does_not_establish": (
                "geometric ACCURACY.  A PASS means the recorded detections contain the fiducials "
                "and the bookkeeping is consistent; it says nothing about how accurate a "
                "reconstruction from them will be.")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="episode directories, relative to the project root or absolute")
    ap.add_argument("--all", action="store_true", help="every outputs/arena/arena_* episode")
    ap.add_argument("--out", default=str(ARENA / "episode_health.json"))
    # IS THE GATE RIGHT, OR IS EVERY EPISODE GREEN?  Those are different questions and the aggregate
    # cannot answer the first one: `arena_bare_seed0` is a DELIBERATELY BAD fixture -- its `px/` table
    # lacks the fiducials, the failure that once fed a 2387 mm "reconstruction" downstream -- so the
    # aggregate `status` is CORRECTLY FAIL for as long as that fixture exists.  A suite entry keyed on
    # the aggregate would therefore be permanently red, and a permanently red check teaches a reader
    # to ignore red.  `--expect NAME=VERDICT` asks the question that actually matters: did the gate
    # JUDGE THE FIXTURES AS DECLARED?  It writes its verdict to --out-expect, which is what the suite
    # cross-checks.
    ap.add_argument("--expect", nargs="*", default=None, metavar="NAME=VERDICT",
                    help="assert a named episode's verdict, e.g. arena_bare_seed0=FAIL")
    ap.add_argument("--out-expect", default=str(ARENA / "episode_health_expectations.json"))
    args = ap.parse_args()

    if args.all:
        eps = sorted(p for p in ARENA.glob("arena_*") if p.is_dir())
    elif args.episodes:
        eps = [Path(p) if Path(p).is_absolute() else HERE / p for p in args.episodes]
    else:
        ap.error("give --episodes or --all")

    results = [health(p) for p in eps]
    # ---- the expectation check, if any were declared
    expect_doc = None
    if args.expect:
        got = {r["episode"]: r["status"] for r in results}
        rows, bad = [], []
        for item in args.expect:
            if "=" not in item:
                raise SystemExit(f"--expect wants NAME=VERDICT, got {item!r}")
            nm, want = item.split("=", 1)
            want = want.strip().upper()
            have = got.get(nm.strip())
            ok = (have == want)
            rows.append({"episode": nm.strip(), "expected": want, "actual": have, "ok": ok})
            if not ok:
                bad.append(rows[-1])
            print(f"   expect {nm.strip():26s} {want:8s} -> actual {have}  "
                  f"{'OK' if ok else 'MISMATCH'}")
        expect_doc = {
            "gate": "recording data health -- expectations",
            "status": "PASS" if not bad else "FAIL",
            "expectations": rows, "n_mismatched": len(bad),
            "why": ("the aggregate status of the gate is FAIL for as long as the deliberately bad "
                    "fixture exists, so the suite asks instead whether the gate JUDGED each fixture "
                    "as declared.  The bad fixture failing IS the gate working."),
            "units": "not applicable: this file records verdict matches, not values",
            "provenance": "tools_episode_health.py --expect, over the episodes it just judged",
            "what_this_does_not_establish": (
                "that the gate's THRESHOLDS are right.  It establishes that the gate produces the "
                "declared verdict on the declared fixtures; a gate with a wrong threshold would "
                "satisfy this just as well."),
        }
        ep = Path(args.out_expect)
        ep.parent.mkdir(parents=True, exist_ok=True)
        ep.write_text(json.dumps(expect_doc, indent=2, sort_keys=True))
        print(f"wrote {ep}")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # A TOP-LEVEL status as well as the per-episode ones, because the per-episode verdicts alone
    # leave "is this gate green?" ambiguous: some episodes PASSing while another FAILs is the
    # normal and useful outcome, so the aggregate is stated rather than implied.
    sts = [r["status"] for r in results]
    _overall = ("FAIL" if "FAIL" in sts else ("NOT_RUN" if "NOT_RUN" in sts else "PASS"))
    dest.write_text(json.dumps({"gate": "recording data health",
                                "status": _overall,
                                "n_episodes_pass": sum(1 for s in sts if s == "PASS"),
                                "n_episodes_fail": sum(1 for s in sts if s == "FAIL"),
                                "n_episodes_not_run": sum(1 for s in sts if s == "NOT_RUN"),
                                "units": ("dimensionless ratios and pixel counts; no physical "
                                          "quantity is reported, and the force/length units of the "
                                          "episodes themselves are outside this gate's scope"),
                                "thresholds": THRESHOLDS, "episodes": results,
                                "convention": ("exit code 0 means the check ran, not that it "
                                               "passed; read each episode's `status`")},
                               indent=2, sort_keys=True))
    print(f"{'episode':<26}{'status':>9}   decisive check (fiducial hit rate within 3 px)")
    for r in results:
        hr = next((c for c in r["checks"] if c["check"] == "fiducial_hit_rate_3px"), None)
        v = hr["value"] if hr and isinstance(hr.get("value"), dict) else None
        shown = (f"{100.0*v['rate']:5.1f} %  ({v['hits']}/{v['evaluated']})" if v
                 else str(hr.get("note") if hr else "n/a"))
        print(f"{r['episode']:<26}{r['status']:>9}   {shown}")
    print(f"\nwrote {dest}")
    print("NOTE: exit code 0 means the gate RAN.  The verdict is in `status`, per episode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
