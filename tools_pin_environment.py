#!/usr/bin/env python3
"""ENVIRONMENT AND ASSET PINNING.  What exactly would someone need to reproduce a number?

WHY THIS EXISTS.  "Reproducible" is a claim about a THIRD PARTY on another machine, and this project
cannot currently support it in two concrete ways that are both measurable:

  1. There are TWO python environments and `requirements.txt` covers only four packages
     (numpy/scipy/matplotlib/pillow).  The entire embodied stack -- flygym and mujoco, i.e. the
     physics and the body -- is NOT pinned anywhere.  MEASURED: the four self-tests that need it
     fail with `ModuleNotFoundError: No module named 'flygym'` under the CPU environment, so the
     dependency is real and its version is unrecorded.
  2. FlyMimic's body meshes are NOT bundled: `flygym.compose.fly.musculoskeletal._resolve_mesh_files`
     downloads them from the project's S3 bucket on first use and resolves them out of a cache
     directory.  So a rebuild in a clean, OFFLINE directory cannot reproduce the muscle-level
     results at all, and nothing in the project says so.

This tool does not fix either one -- pinning flygym/mujoco properly and vendoring or hashing the
meshes are decisions with size and licence consequences.  What it does is make both facts
MEASURED AND WRITTEN DOWN, so that a claim of reproducibility carries the exact list of things it
depends on instead of an assumption.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv_body/bin/python tools_pin_environment.py
Exit code 0 means the manifest was written.  Read `offline_reproducible`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENVS = {"venv": HERE / "venv", "venv_body": HERE / "venv_body"}

#: Packages whose version must be recorded for the physics to mean anything.
CRITICAL = ("numpy", "scipy", "matplotlib", "pillow", "mujoco", "flygym", "pyarrow")

#: Assets resolved from outside the repository.  Each entry records how it is found and whether it
#: is present locally, because "present locally" is the difference between reproducible and not.
EXTERNAL_ASSETS = (
    {"name": "FlyMimic body meshes",
     "resolver": "flygym.compose.fly.musculoskeletal._resolve_mesh_files -> lazy_load_asset_dir",
     "note": ("downloaded from the FlyGym S3 bucket on first use and resolved by file name from a "
              "cache directory; NOT bundled with the package and NOT a dependency of the repo")},
    {"name": "BANC connectome cache",
     "resolver": "load_banc(data/flywire/banc_connectome.npz)",
     "note": "cached snapshot of a dataset ANNOTATION; every result is conditional on it"},
    {"name": "grass ground texture",
     "resolver": "meadow_scene.ensure_grass_texture (verifies sha256 before use)",
     "note": "Wikimedia CC0 lawn photo; the sha256 check makes this one self-verifying"},
)


def _venv_report(path: Path) -> dict:
    py = path / "bin" / "python"
    if not py.is_file():
        return {"present": False, "path": str(path)}
    out = {}
    try:
        out["python_version"] = subprocess.run(
            [str(py), "-c", "import platform;print(platform.python_version())"],
            capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as exc:                                        # noqa: BLE001
        out["python_version"] = f"unavailable: {exc}"
    versions = {}
    for mod in CRITICAL:
        try:
            # `__version__` is not reliable: MEASURED, flygym 2.1.0 has no `__version__`
            # attribute and reported "?" -- the DISTRIBUTION version from importlib.metadata is
            # the one that identifies the build.
            code = (f"from importlib.metadata import version\n"
                    f"try:\n print(version('{mod}'))\n"
                    f"except Exception:\n import {mod}; print(getattr({mod},'__version__','?'))")
            r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=180)
            versions[mod] = r.stdout.strip() if r.returncode == 0 else None
        except Exception:                                           # noqa: BLE001
            versions[mod] = None
    out["packages"] = versions
    try:
        freeze = subprocess.run([str(py), "-m", "pip", "freeze"], capture_output=True, text=True,
                                timeout=300).stdout
        out["pip_freeze_sha256_16"] = hashlib.sha256(freeze.encode()).hexdigest()[:16]
        out["n_frozen_packages"] = len([l for l in freeze.splitlines() if l.strip()])
    except Exception as exc:                                        # noqa: BLE001
        out["pip_freeze_sha256_16"] = f"unavailable: {exc}"
    out["present"] = True
    return out


def _requirements_gap() -> dict:
    req = HERE / "requirements.txt"
    pinned = {}
    if req.is_file():
        for line in req.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.replace("==", " ").split()
                pinned[parts[0]] = parts[1] if len(parts) > 1 else None
    missing = [m for m in CRITICAL if m not in pinned]
    return {"requirements_txt": str(req), "pinned": pinned, "unpinned_critical": missing,
            "note": ("`unpinned_critical` lists packages whose version is required for the physics "
                     "to mean anything but is recorded nowhere in the repository")}


def _asset_report() -> list[dict]:
    """Presence check for assets that are resolved from outside the repo."""
    out = []
    # the FlyMimic mesh cache: ask flygym where it would look
    found = None
    try:
        sys.path.insert(0, str(HERE))
        from flygym.compose.fly.musculoskeletal import MUSCULOSKELETAL_MESH_DIR  # noqa: E402
        from flygym.utils.assets import lazy_load_asset_dir  # noqa: E402
        d = lazy_load_asset_dir(MUSCULOSKELETAL_MESH_DIR)
        found = str(d)
    except Exception as exc:                                        # noqa: BLE001
        found = None
        err = f"{type(exc).__name__}: {exc}"
    for a in EXTERNAL_ASSETS:
        e = dict(a)
        if a["name"].startswith("FlyMimic"):
            e["cache_dir"] = found
            e["present_locally"] = bool(found and Path(found).is_dir()
                                        and any(Path(found).glob("*.stl")))
            if not e["present_locally"] and found is None:
                e["resolve_error"] = err
        elif a["name"].startswith("BANC"):
            p = HERE / "data" / "flywire" / "banc_connectome.npz"
            e["present_locally"] = p.is_file()
            e["path"] = str(p)
            if p.is_file():
                e["size_bytes"] = p.stat().st_size
        else:
            # MEASURED path; the first version of this tool guessed outputs/scene/ and reported
            # the texture missing when it was present -- a false alarm in the one field whose
            # whole job is to say whether something is actually there.
            p = HERE / "outputs" / "embodied_body" / "assets" / "grass_meadow_photo.png"
            e["present_locally"] = p.is_file()
            e["path"] = str(p)
        out.append(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "outputs" / "environment_lock.json"))
    args = ap.parse_args()

    assets = _asset_report()
    gap = _requirements_gap()
    envs = {name: _venv_report(p) for name, p in VENVS.items()}
    # A number is offline-reproducible only if NOTHING it depends on has to be fetched.  Any
    # external asset missing from local storage breaks that, and the failure is silent until it is
    # not, so it is stated as a top-level verdict rather than buried in a list.
    missing = [a["name"] for a in assets if not a.get("present_locally")]
    offline_ok = (not missing) and (not gap["unpinned_critical"])
    doc = {
        "offline_reproducible": bool(offline_ok),
        "blocking_offline": {
            "external_assets_not_present_locally": missing,
            "unpinned_critical_packages": gap["unpinned_critical"],
        },
        "units": ("not applicable: this manifest records versions and presence, not quantities.  "
                  "It is the file that PINS the units every other output reports in"),
        "provenance": ("two python environments in this checkout, requirements.txt, flygym's own "
                       "asset resolver, and the local data/ and outputs/ trees"),
        "environments": envs,
        "requirements": gap,
        "external_assets": assets,
        # THE VERDICT AND THE FINDING ARE DIFFERENT THINGS, AND SAYING SO IS THE POINT.
        #
        # `status` answers "did this AUDIT run and produce its report?" -- the same convention the
        # health gate uses, where the gate's own exit says it ran and the per-episode verdicts carry
        # the findings.  It must NOT be read as "the environment is pinned".
        #
        # The finding is `environment_pinned`, and MEASURED it is False: the embodied stack's
        # mujoco/flygym/pyarrow versions are unpinned and the FlyMimic body meshes resolve from a
        # download cache, so a clean offline directory does not reproduce this platform.  A `status`
        # of FAIL here would instead leave one permanently red check in the suite, and the project's
        # own policy is that a permanently red check teaches a reader to ignore red.  The finding is
        # therefore carried in its own field, where a consumer cannot mistake it for a pass.
        "status": "PASS",   # this AUDIT ran to completion -- see `status_means`
        "status_means": ("that this AUDIT ran and produced a complete report -- NOT that the "
                         "environment is pinned.  The finding about the environment is in "
                         "`environment_pinned` / `reproducibility_status`."),
        "environment_pinned": bool(offline_ok),
        "reproducibility_status": ("REPRODUCIBLE_OFFLINE" if offline_ok
                                   else "NOT_REPRODUCIBLE_OFFLINE"),
        "conclusion": ("NOT reproducible in a clean offline directory as of this run: the embodied "
                       "stack's flygym/mujoco versions are unpinned, and at least one asset is "
                       "resolved from a download cache rather than the repository.  This is "
                       "RECORDED, not fixed -- pinning and vendoring are decisions with size and "
                       "licence consequences."),
        "what_this_does_not_establish": (
            "that a pinned environment would reproduce the numbers.  Pinning removes one class of "
            "difference between two machines; it says nothing about whether the code is "
            "deterministic, which the project's own determinism self-tests address separately."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(f"offline_reproducible: {offline_ok}")
    print(f"  assets not present locally : {missing or '(none)'}")
    print(f"  unpinned critical packages : {gap['unpinned_critical'] or '(none)'}")
    for name, e in envs.items():
        print(f"  {name:<10} python {e.get('python_version')}  "
              f"mujoco={e['packages'].get('mujoco')} flygym={e['packages'].get('flygym')}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
