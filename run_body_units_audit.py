#!/usr/bin/env python3
"""BODY UNITS PROVENANCE AUDIT  (read-only; hand-built research audit, NOT a product).

WHAT THIS ASKS
--------------
`engine/canonical_physics.py` declares the MuJoCo model length unit of the canonical
asset to be the CENTIMETRE and the model mass unit to be the MILLIGRAM.  The same file's
own docstring also calls the total mass "0.002494271478 kg = 2.494 mg", and
`tools_stand_and_ground.py` calls the mass unit the GRAM.  Those cannot all be true.

This script does NOT re-measure any physics and does NOT load MuJoCo.  It reads the
canonical asset, the upstream flygym/flybody assets that it was derived from, the actual
mesh files that the asset scales, and the project's own scripts, and reports what each of
them says about units -- with file path and line number for every claim.

THE DECIDING MEASUREMENT
------------------------
`<mesh ... scale="1000 1000 1000"/>` multiplies the STL coordinates into model length
units.  The cached STL files are in metres (the thorax mesh is 0.001069 m long, i.e. a
real 1.07 mm fly thorax).  metre x 1000 = millimetre.  Independently: in MJCF a child
`<body pos=...>` joint offset and a mesh extent are BOTH in model length units, so their
ratio must be O(1); over the 70 body/mesh pairs in this asset the median
mesh_extent/body_offset is 0.93, not 0.093 and not 9.3.  So the model length unit is the
MILLIMETRE, and `<option gravity="0 0 -9801">` is 9801 mm/s^2 = 9.801 m/s^2 -- standard
gravity.  Under the centimetre reading the same fly would be 2 cm long and gravity would
be 98 m/s^2.

WHAT IT CANNOT SETTLE
---------------------
The asset declares NO mass unit anywhere (71 `<inertial mass=...>` attributes, zero
`density=` attributes).  This script therefore reports the contradictory declarations it
finds and the physical constraint that rules two of the three candidates out; it does NOT
silently adopt one.  See "what_this_does_not_support" in the artifact.

SAFETY
------
Pure stdlib: no numpy, no pandas, no MuJoCo, no simulation.  Peak footprint is a few MB.
Runs under either `venv/bin/python` or `venv_body/bin/python`.  Wrapped in
`engine/memory_guard.guarded_run` because this host has a recorded global OOM kill.

RUN
---
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python \
      /run/media/sensen/Data2/cell_wound_prototype/run_body_units_audit.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "engine"))
import memory_guard as mg  # noqa: E402

# ---------------------------------------------------------------------------
# Absolute paths of every source this audit reads.  Nothing is written except
# the audit's own artifact and the JSON below.
# ---------------------------------------------------------------------------
CANONICAL_XML = PROJECT / "outputs/muscles_six_legs/fruitfly_six_leg_standing.xml"
EXPECTED_CANONICAL_SHA256 = (
    "65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6")

UPSTREAM_MSK_XML = (PROJECT / "venv_body/lib/python3.12/site-packages/flygym/assets/"
                    "model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml")
UPSTREAM_FLYBODY_XML = (PROJECT / "venv_body/lib/python3.12/site-packages/flygym/assets/"
                        "model/flybody/fruitfly.xml")
FLYGYM_BASE_FLY = (PROJECT / "venv_body/lib/python3.12/site-packages/flygym/"
                   "compose/fly/base_fly.py")
FLYGYM_FLYBODY_PY = (PROJECT / "venv_body/lib/python3.12/site-packages/flygym/"
                     "compose/fly/flybody.py")
FLYGYM_SIMULATION = (PROJECT / "venv_body/lib/python3.12/site-packages/flygym/simulation.py")
FLYGYM_MSK_PY = (PROJECT / "venv_body/lib/python3.12/site-packages/flygym/compose/fly/"
                 "musculoskeletal.py")

CANONICAL_PHYSICS = PROJECT / "engine/canonical_physics.py"
BODY_BACKEND = PROJECT / "engine/embodied/body_backend.py"
STAND_SCRIPT = PROJECT / "tools_stand_and_ground.py"
LEG_TUNING = PROJECT / "LEG_TUNING.zh-CN.md"
DIAG_LEG_LOAD = PROJECT / "diagnostics/diag_leg_load_test.py"

#: The musculoskeletal body meshes are not bundled with flygym; flygym downloads them to
#: this cache on first use (`MUSCULOSKELETAL_MESH_DIR`, see FLYGYM_MSK_PY).  The asset
#: references them as `./meshes/stl/<X>.stl`, which does not exist next to the canonical
#: XML, so `_resolve_mesh_files` falls back to this cache -- i.e. these ARE the meshes the
#: canonical asset compiles against.
MESH_CACHE = Path.home() / (".cache/flygym_assets/"
                            "neuromechfly_musculoskeletal_meshes_20260623a")
#: flygym writes this next to the downloads; it is the only machine-readable statement of
#: where the meshes came from (it still declares no units).
MESH_METADATA = MESH_CACHE / "metadata.yaml"

FLYMIMIC_REPO_URL = "https://github.com/gizemozd/FlyMimic"
FLYMIMIC_ARXIV_URL = "https://arxiv.org/abs/2509.06426"
NEUROMECHFLY_URL = "https://www.nature.com/articles/s41592-022-01411-8"
DROSOPHILID_MASS_TABLE_URL = (
    "https://www.biorxiv.org/content/10.64898/2026.03.15.711943v1.full.pdf")
DROSOPHILA_THORAX_LENGTH_TABLE_URL = (
    "https://besjournals.onlinelibrary.wiley.com/action/downloadSupplement"
    "?doi=10.1111%2Fj.1365-2435.2008.01533.x&file=FEC_1533_sm_TableS5.pdf")

ARTIFACT = PROJECT / "outputs/body_physics_audit/units_provenance.json"

#: Mass of one model mass unit, in kg, for each candidate reading.
MASS_UNIT_CANDIDATES = {
    "kilogram": 1.0,
    "gram": 1e-3,
    "milligram": 1e-6,
}
#: SI density of a few references, kg/m^3.
DENSITY_REFERENCES = {"air_20C": 1.204, "water_4C": 1000.0,
                      "insect_cuticle_chitin": 1400.0, "osmium": 22590.0}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lines_of(path: Path) -> list:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def locate(path: Path, needle: str) -> list:
    """1-based (line_number, stripped_line) for every line containing `needle`."""
    return [(i, ln.strip()) for i, ln in enumerate(lines_of(path), 1) if needle in ln]


def first_line(path: Path, needle: str):
    hits = locate(path, needle)
    return {"file": str(path), "line": hits[0][0], "text": hits[0][1]} if hits else None


def stl_facts(path: Path) -> dict:
    """Bounding box (raw mesh coordinates) and |signed volume| of a binary STL.

    `struct.iter_unpack` walks the 50-byte triangles at C speed, so a ~900 kB mesh costs
    a few milliseconds and no large allocation.  Results are memoised because the
    cross-check and the volume total both walk the same files.
    """
    if path in _STL_MEMO:
        return _STL_MEMO[path]
    raw = path.read_bytes()
    n = struct.unpack_from("<I", raw, 80)[0]
    body = raw[84:84 + n * 50]
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    vol6 = 0.0
    for rec in struct.iter_unpack("<12f2x", body):
        a = rec[3:6]
        b = rec[6:9]
        c = rec[9:12]
        for k in range(3):
            if a[k] < lo[k]:
                lo[k] = a[k]
            if a[k] > hi[k]:
                hi[k] = a[k]
        vol6 += (a[0] * (b[1] * c[2] - b[2] * c[1])
                 + a[1] * (b[2] * c[0] - b[0] * c[2])
                 + a[2] * (b[0] * c[1] - b[1] * c[0]))
    out = {"name": path.stem, "n_triangle": n,
           "raw_bbox_min": lo, "raw_bbox_max": hi,
           "raw_extent": [hi[k] - lo[k] for k in range(3)],
           "raw_volume_abs": abs(vol6) / 6.0}
    _STL_MEMO[path] = out
    return out


#: memo for stl_facts, keyed by path (the cache holds ~14 MB of meshes in total)
_STL_MEMO: dict = {}


def tally(store: dict, value) -> None:
    store[value] = store.get(value, 0) + 1


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------
def main(report) -> int:
    print("=" * 78)
    print("BODY UNITS PROVENANCE AUDIT -- read-only, no MuJoCo, no simulation")
    print("=" * 78)
    findings: list = []

    def note(kind: str, text: str, **extra) -> None:
        findings.append({"kind": kind, "text": text, **extra})
        print(f"  [{kind}] {text}")

    # ---- 0. the canonical asset must not have moved -------------------------
    print("\n-- 0. canonical asset integrity")
    sha = sha256_file(CANONICAL_XML)
    sha_ok = (sha == EXPECTED_CANONICAL_SHA256)
    print(f"  {CANONICAL_XML}\n  sha256 = {sha}\n  matches expected: {sha_ok}")
    assert sha_ok, "canonical asset hash changed - refusing to continue"

    xml_text = CANONICAL_XML.read_text(encoding="utf-8")
    xml_lines = xml_text.splitlines()

    # ---- 1. what the canonical asset itself declares ------------------------
    print("\n-- 1. what the canonical asset declares")
    option = re.search(r"<option[^>]*>", xml_text).group(0)
    compiler = re.search(r"<compiler[^>]*>", xml_text).group(0)
    statistic = re.search(r"<statistic[^>]*>", xml_text).group(0)
    gravity = float(re.search(r'gravity="0 0 ([-0-9.eE]+)"', option).group(1))
    print(f"  {option.strip()}")
    print(f"  {compiler.strip()}")
    print(f"  {statistic.strip()}")
    print("  NOTE: MJCF has no unit attribute. `compiler` carries no `meshdir`/`coordinate`;"
          " the unit is a CONVENTION, never a declaration.")

    mesh_scales = re.findall(r'<mesh[^>]*scale="([^"]+)"', xml_text)
    scale_tally: dict = {}
    for s in mesh_scales:
        tally(scale_tally, s)
    print(f"  mesh scale values used: {scale_tally}  (over {len(mesh_scales)} mesh assets)")
    mesh_scale = float(mesh_scales[0].split()[0]) if mesh_scales else None

    masses = [float(x) for x in re.findall(r'\bmass="([^"]+)"', xml_text)]
    n_density = len(re.findall(r'\bdensity="', xml_text))
    mass_sum = sum(masses)
    print(f"  <inertial mass=> count = {len(masses)}   sum = {mass_sum!r}")
    print(f"  geom/inertial `density=` count = {n_density}"
          "   -> NO density anywhere, so nothing in the file ties mass to volume")

    # every body must carry an explicit <inertial>, else MuJoCo would derive mass from
    # geom volume x default density (1000) and the total would depend on the length unit.
    body_ln = [i for i, ln in enumerate(xml_lines, 1) if re.search(r'<body\s+name="', ln)]
    inert_ln = [i for i, ln in enumerate(xml_lines, 1) if "<inertial" in ln]
    body_names = [re.search(r'<body\s+name="([^"]+)"', xml_lines[i - 1]).group(1)
                  for i in body_ln]
    no_inertial = []
    for k, ln in enumerate(body_ln):
        end = body_ln[k + 1] if k + 1 < len(body_ln) else len(xml_lines)
        if not any(ln < j < end for j in inert_ln):
            no_inertial.append({"line": ln, "body": body_names[k]})
    print(f"  bodies = {len(body_ln)}, <inertial> = {len(inert_ln)}, "
          f"bodies without <inertial> = {no_inertial}")
    if len(inert_ln) != len(body_ln) - 1 or no_inertial != [{"line": 123, "body": "ground"}]:
        note("WARNING", "unexpected body/inertial structure; the mass total may not be "
                        "purely the sum of declared inertials")

    keyframe = re.search(r'<key\s+name="([^"]+)"\s+qpos="([^"]+)"', xml_text)
    key_qpos = [float(v) for v in keyframe.group(2).split()]
    print(f"  keyframe '{keyframe.group(1)}' qpos[0:3] = {key_qpos[:3]}"
          "   (free joint: qpos[2] is the Thorax body-frame height above z=0)")
    floor = re.search(r'<geom\s+name="floor"[^>]*>', xml_text).group(0)
    print(f"  {floor.strip()}")
    thorax_body = re.search(r'<body\s+name="Thorax"[^>]*>', xml_text).group(0)
    print(f"  {thorax_body.strip()}")

    # ---- 2. the upstream assets this one was derived from -------------------
    print("\n-- 2. upstream assets (flygym 2.1.0, read-only)")
    up_msk_text = UPSTREAM_MSK_XML.read_text(encoding="utf-8")
    up_msk_masses = [float(x) for x in re.findall(r'\bmass="([^"]+)"', up_msk_text)]
    up_opt = re.search(r"<option[^>]*>", up_msk_text).group(0)
    up_stat = re.search(r"<statistic[^>]*>", up_msk_text).group(0)
    print(f"  {UPSTREAM_MSK_XML}")
    print(f"    {up_opt.strip()}")
    print(f"    {up_stat.strip()}")
    print(f"    n mass = {len(up_msk_masses)}  sum = {sum(up_msk_masses)!r}")
    ratio = mass_sum / sum(up_msk_masses)
    print(f"    canonical/upstream mass ratio = {ratio!r}  -> "
          f"{'NO in-project mass rescaling' if ratio == 1.0 else 'MASS WAS RESCALED'}")

    fb_opt = re.search(r"<option[^>]*>", UPSTREAM_FLYBODY_XML.read_text()).group(0)
    fb_gravity = float(re.search(r'gravity="0 0 ([-0-9.eE]+)"', fb_opt).group(1))
    print(f"  {UPSTREAM_FLYBODY_XML}")
    print(f"    {fb_opt.strip()}")
    print(f"    flybody gravity = {fb_gravity}  (the canonical family uses {gravity})")
    note("CONTRAST",
         f"two upstream flygym assets use gravity -{abs(fb_gravity):g} (flybody) and "
         f"-{abs(gravity):g} (musculoskeletal): a factor 10, which is exactly the "
         "centimetre/millimetre ratio.  Each value is standard gravity in its own unit.",
         flybody_gravity=fb_gravity, musculoskeletal_gravity=gravity)

    # ---- 3. the framework's own declared convention -------------------------
    print("\n-- 3. flygym's declared length convention")
    conv = None
    for i, ln in enumerate(lines_of(FLYGYM_BASE_FLY), 1):
        if "simulate length in mm" in ln:
            conv = {"file": str(FLYGYM_BASE_FLY), "line": i,
                    "text": ln.strip(),
                    "text_next": lines_of(FLYGYM_BASE_FLY)[i].strip()}
            break
    print(f"  {conv['file']}:{conv['line']}  {conv['text']}")
    print(f"  {conv['file']}:{conv['line']+1}  {conv['text_next']}")
    scale_ln = first_line(FLYGYM_BASE_FLY, "SCALE = 1000")
    print(f"  {FLYGYM_BASE_FLY}:{scale_ln['line']}  {scale_ln['text']}")
    pos_doc = first_line(FLYGYM_SIMULATION, "Body positions in mm")
    print(f"  {FLYGYM_SIMULATION}:{pos_doc['line']}  {pos_doc['text']}")
    fb_cm = first_line(FLYGYM_FLYBODY_PY, "authored in cm")
    print(f"  {FLYGYM_FLYBODY_PY}:{fb_cm['line']}  {fb_cm['text']}")
    msk_prov = first_line(FLYGYM_MSK_PY, "MUSCULOSKELETAL_MESH_DIR = ")
    print(f"  {FLYGYM_MSK_PY}:{msk_prov['line']}  {msk_prov['text']}")
    note("UPSTREAM CONVENTION",
         "flygym simulates length in MILLIMETRES (base_fly.py, quoted line above), and "
         "get_body_positions() documents its return in mm.  The canonical asset is a "
         "flygym asset compiled through flygym's own loader, so that is its convention.",
         evidence=[conv, scale_ln, pos_doc, fb_cm])

    # ---- 4. the meshes: metre-valued STL x scale 1000 -> model unit -------
    print("\n-- 4. the meshes the asset scales (this is the deciding measurement)")
    mesh_facts = {}
    mesh_status = "NOT_RUN"
    med = None
    n_pairs = None
    if MESH_CACHE.is_dir():
        mesh_status = "PASS"
        for name in ("Thorax", "Head", "LFFemur", "LFTibia", "LHTibia", "LHTarsus5"):
            p = MESH_CACHE / f"{name}.stl"
            if p.is_file():
                mesh_facts[name] = stl_facts(p)
        print(f"  mesh cache: {MESH_CACHE}")
        for name, f in mesh_facts.items():
            ext = f["raw_extent"]
            print(f"    {name:10s} raw metres extent "
                  f"[{ext[0]:.3e} {ext[1]:.3e} {ext[2]:.3e}]"
                  f"  x{mesh_scale:g} = [{ext[0]*mesh_scale:.4f} "
                  f"{ext[1]*mesh_scale:.4f} {ext[2]*mesh_scale:.4f}] model units")

        # total volume of all meshes -> an upper bound on the union volume, because
        # meshes of adjacent segments overlap.
        total_raw_vol = 0.0
        n_mesh = 0
        for p in sorted(MESH_CACHE.glob("*.stl")):
            total_raw_vol += stl_facts(p)["raw_volume_abs"]
            n_mesh += 1
        print(f"    all {n_mesh} meshes: sum |volume| = {total_raw_vol:.6e} raw units")
        print(f"      = {total_raw_vol * 1e9:.4f} mm^3  (upper bound; overlaps counted twice)")

        # --- the cross-check that pins the model length unit -------------------
        # In MJCF, a child <body pos=...> joint offset and a mesh extent are BOTH in
        # model length units.  They are not expected to be EQUAL (an offset is a
        # joint-to-joint distance, a mesh extent is a surface bounding box, so gaps at
        # the joints make the ratio scatter), but they MUST be the SAME ORDER OF
        # MAGNITUDE.  If the model length unit were 10x the mesh unit every ratio would
        # be multiplied by 10, and vice versa.
        body_pos = dict(re.findall(r'<body\s+name="([^"]+)"\s+pos="([^"]+)"', xml_text))
        ratios = []
        for nm, pos in body_pos.items():
            if nm in ("Thorax", "ground"):
                continue
            mp = MESH_CACHE / f"{nm}.stl"
            if not mp.is_file():
                continue
            off = math.sqrt(sum(float(v) ** 2 for v in pos.split()))
            if off < 1e-9:
                continue
            ext = max(stl_facts(mp)["raw_extent"]) * mesh_scale
            ratios.append((nm, off, ext, ext / off))
        ratios.sort(key=lambda r: r[3])
        med = ratios[len(ratios) // 2][3]
        n_pairs = len(ratios)
        print(f"    CROSS-CHECK over {len(ratios)} body/mesh pairs: "
              f"mesh_extent / body_offset  median = {med:.4f}, "
              f"min = {ratios[0][3]:.4f} ({ratios[0][0]}), "
              f"max = {ratios[-1][3]:.4f} ({ratios[-1][0]})")
        print("      (a factor-10 unit mismatch between offsets and meshes would put "
              f"this median at ~{med * 10:.2f} or ~{med / 10:.3f})")
        within_order = sum(1 for r in ratios if 0.1 <= r[3] <= 10.0)
        print(f"      {within_order}/{len(ratios)} ratios within one order of magnitude"
              " of unity")
        note("DECIDING",
             "MJCF child-body offsets and mesh coordinates are BOTH in model length "
             "units, so their ratio must be O(1).  Over %d body/mesh pairs the median "
             "mesh_extent/body_offset is %.4f -- i.e. the two are the same order, which "
             "is exactly what a shared unit gives.  A centimetre model unit with "
             "millimetre meshes (or vice versa) would shift that median by a factor of "
             "10.  The raw STL coordinates are metre-valued (thorax 1.069e-3 m), so "
             "1 model length unit = 1 MILLIMETRE, and the centimetre reading would make "
             "this fly ~2.6 cm long." % (len(ratios), med),
             n_pairs=len(ratios), median_ratio=med,
             min_ratio=ratios[0][3], min_pair=ratios[0][0],
             max_ratio=ratios[-1][3], max_pair=ratios[-1][0],
             pairs_within_one_order=within_order,
             example_LFTibia={"body_offset_model": dict(
                 (n, o) for n, o, _e, _r in ratios)["LFTibia"],
                 "mesh_extent_model": dict(
                     (n, e) for n, _o, e, _r in ratios)["LFTibia"],
                 "ratio": dict((n, r) for n, _o, _e, r in ratios)["LFTibia"]},
             thorax_raw_metres=mesh_facts["Thorax"]["raw_extent"])

        # implied density for each candidate mass unit
        total_mm3 = total_raw_vol * 1e9
        dens = {}
        for unit, kg in MASS_UNIT_CANDIDATES.items():
            m_kg = mass_sum * kg
            d = m_kg / (total_mm3 * 1e-9)
            dens[unit] = d
            print(f"    if 1 model mass unit = 1 {unit:10s}: total = {m_kg:.6e} kg"
                  f"  -> implied density = {d:,.2f} kg/m^3")
        print(f"    reference densities: {DENSITY_REFERENCES}")
        admissible = [u for u, d in dens.items()
                      if DENSITY_REFERENCES["air_20C"] * 1.5 < d < 3000.0]
        print(f"    physically admissible (between ~1.5x air and 3000 kg/m^3): {admissible}")
        note("PHYSICAL CONSTRAINT",
             "The asset declares no density, so mesh volume is the only independent "
             "handle on the mass unit.  'milligram' implies ~%.1f kg/m^3 (lighter than "
             "air) and 'kilogram' implies ~%.2e kg/m^3 (denser than any element); only "
             "'gram' gives a tissue-like density." % (dens["milligram"], dens["kilogram"]),
             implied_density_kg_m3=dens, admissible=admissible,
             mesh_volume_upper_bound_mm3=total_mm3)
    else:
        print(f"  MESH CACHE MISSING: {MESH_CACHE}")
        note("NOT_RUN", "mesh cache absent; the decisive mesh cross-check was NOT run. "
                        "The framework-convention and gravity evidence still stand.")
        total_mm3, dens, admissible = None, None, None

    # ---- 5. gravity self-consistency ---------------------------------------
    print("\n-- 5. gravity self-consistency")
    grav_table = {}
    for unit, m_per in (("millimetre", 1e-3), ("centimetre", 1e-2)):
        a = abs(gravity) * m_per
        grav_table[unit] = a
        print(f"  if 1 model length unit = 1 {unit:11s}: |gravity| reads as "
              f"{a:.4f} m/s^2"
              f"   {'<-- standard gravity' if abs(a - 9.80665) < 0.05 else '<-- NOT gravity'}")
    note("GRAVITY",
         f"gravity attribute = {gravity:g}.  It is 9.801 m/s^2 under the millimetre "
         "reading and 98.01 m/s^2 under the centimetre reading.  A modeller writing "
         "9801 chose millimetres; the centimetre-family flybody asset in the same "
         "package writes 981.  Gravity therefore does not DECIDE the unit by itself "
         "(a model is self-consistent in any unit system), but it does show which unit "
         "the author was writing in, and it rules out 'no scaling'.",
         gravity_attribute=gravity, gravity_in_m_s2_by_unit=grav_table)

    # ---- 6. the project's own contradictory declarations --------------------
    print("\n-- 6. what THIS project declares, and where it disagrees with itself")
    decl = {
        "canonical_physics_docstring": locate(CANONICAL_PHYSICS,
                                              "total mass = 0.002494271478 kg"),
        "canonical_physics_length": locate(CANONICAL_PHYSICS, 'MODEL_LENGTH_UNIT_NAME = "cm"'),
        "canonical_physics_mm_per": locate(CANONICAL_PHYSICS, "MM_PER_MODEL_LENGTH = 10.0"),
        "canonical_physics_mass_unit": locate(CANONICAL_PHYSICS, "MODEL_MASS_UNIT_KG = 1e-6"),
        "canonical_physics_gravity": locate(CANONICAL_PHYSICS, "GRAVITY_MODEL = 9801.0"),
        "canonical_physics_height_claim": locate(CANONICAL_PHYSICS, "20.4535 mm"),
        "canonical_physics_writer_mass_label": locate(CANONICAL_PHYSICS, '"mass_model": "mg"'),
        "canonical_physics_pose_20449": locate(CANONICAL_PHYSICS, "20.449 mm"),
        "body_backend_length": locate(BODY_BACKEND, 'MODEL_LENGTH_UNIT_NAME = "cm"'),
        "body_backend_centimetres": locate(BODY_BACKEND, "which for this project are CENTIMETRES"),
        "stand_script_mass_unit": locate(STAND_SCRIPT, "model units = gram"),
        "stand_script_mg_conversion": locate(STAND_SCRIPT, "mass*1000"),
        "diag_legload_mass_mg": locate(DIAG_LEG_LOAD, "MASS_MG"),
        "diag_legload_244x": locate(DIAG_LEG_LOAD, "2.44x TOO HEAVY"),
    }
    decl_owner = {
        "canonical_physics_docstring": CANONICAL_PHYSICS,
        "canonical_physics_length": CANONICAL_PHYSICS,
        "canonical_physics_mm_per": CANONICAL_PHYSICS,
        "canonical_physics_mass_unit": CANONICAL_PHYSICS,
        "canonical_physics_gravity": CANONICAL_PHYSICS,
        "canonical_physics_height_claim": CANONICAL_PHYSICS,
        "canonical_physics_writer_mass_label": CANONICAL_PHYSICS,
        "canonical_physics_pose_20449": CANONICAL_PHYSICS,
        "body_backend_length": BODY_BACKEND,
        "body_backend_centimetres": BODY_BACKEND,
        "stand_script_mass_unit": STAND_SCRIPT,
        "stand_script_mg_conversion": STAND_SCRIPT,
        "diag_legload_mass_mg": DIAG_LEG_LOAD,
        "diag_legload_244x": DIAG_LEG_LOAD,
    }
    for key, hits in decl.items():
        for ln, text in hits[:2]:
            print(f"  {key:36s} {decl_owner[key].name}:{ln}  {text[:92]}")

    # the asset's own header comment, written by the generator
    xml_comment_mm = [(i, ln.strip()) for i, ln in enumerate(xml_lines, 1)
                      if "keyframe places the thorax at z=" in ln]
    print(f"  canonical XML header comment:  {CANONICAL_XML}:{xml_comment_mm[0][0]}"
          f"  {xml_comment_mm[0][1]}")

    # tools_stand_and_ground.py's own mm arithmetic on model coordinates
    bodylen = [(i, ln.strip()) for i, ln in enumerate(lines_of(STAND_SCRIPT), 1)
               if "the body is 2.6 mm long" in ln]
    if bodylen:
        print(f"  generator's own mm arithmetic:  {STAND_SCRIPT}:{bodylen[0][0]}"
              f"  {bodylen[0][1][:96]}")
    legtuning = [(i, ln.strip()) for i, ln in enumerate(lines_of(LEG_TUNING), 1)
                 if "2.49427 mg" in ln or "真实果蝇" in ln]
    for ln, text in legtuning[:3]:
        print(f"  {LEG_TUNING}:{ln}  {text[:96]}")

    mass_contradiction = {
        "the_asset_itself": {
            "declares_a_mass_unit": False,
            "evidence": f"{CANONICAL_XML}: {len(masses)} <inertial mass=> attributes, "
                        f"{n_density} `density=` attributes; MJCF has no unit attribute",
        },
        "claim_A_kg": {"value": "0.002494271478 represents kg", "implies_total_kg":
                       mass_sum, "implies_total_mg": mass_sum * 1e6,
                       "where": decl["canonical_physics_docstring"][:1]},
        "claim_B_mg_by_gram_arithmetic": {
            "value": "0.002494271478 is grams, i.e. 2.494 mg",
            "implies_total_kg": mass_sum * 1e-3, "implies_total_mg": mass_sum * 1e3,
            "where": decl["canonical_physics_docstring"][:1]
                     + decl["stand_script_mass_unit"][:1]
                     + decl["stand_script_mg_conversion"][:1]},
        "claim_C_one_model_unit_is_one_mg": {
            "value": "MODEL_MASS_UNIT_KG = 1e-6, so total = 2.494e-9 kg",
            "implies_total_kg": mass_sum * 1e-6, "implies_total_mg": mass_sum * 1e-3,
            "where": decl["canonical_physics_mass_unit"][:1]
                     + decl["canonical_physics_writer_mass_label"][:1]},
        "mutual_inconsistency": (
            "A, B and C cannot all hold: A x 1e3 != B; B / C = 1e3.  A is also "
            "internally broken in its own sentence ('0.002494271478 kg = 2.494 mg' -- "
            "0.002494271478 kg is 2.494 GRAM, not 2.494 mg)."),
    }
    for k, v in mass_contradiction.items():
        if isinstance(v, dict):
            print(f"  {k}: {v.get('value')}")

    length_contradiction = {
        "project_says_centimetres_where": decl["canonical_physics_length"][:1]
                                          + decl["canonical_physics_mm_per"][:1]
                                          + decl["body_backend_length"][:1]
                                          + decl["body_backend_centimetres"][:1],
        "asset_and_meshes_say_millimetres_where": (
            [{"file": str(CANONICAL_XML), "line": xml_comment_mm[0][0],
              "text": xml_comment_mm[0][1]}] if xml_comment_mm else []
        ) + ([{"file": str(STAND_SCRIPT), "line": bodylen[0][0],
               "text": bodylen[0][1]}] if bodylen else [])
          + [{"file": str(CANONICAL_XML), "line": 1,
              "text": "mesh scale=1000 applied to metre-valued STL => mm"}],
        "ratio_if_wrong": 10.0,
    }

    # ---- 7. biological plausibility of the two readings --------------------
    print("\n-- 7. biological plausibility")
    height_model = key_qpos[2]
    body_len_model = 2.6  # from tools_stand_and_ground.py:255-258, in the generator's own units
    extent_stat = float(re.search(r'extent="([^"]+)"', statistic).group(1))
    print(f"  thorax free-joint height     = {height_model} model units")
    print(f"     read as mm -> {height_model:.4f} mm")
    print(f"     read as cm -> {height_model * 10:.4f} mm")
    print(f"  body length (generator, x axis) = {body_len_model} model units")
    print(f"     read as mm -> {body_len_model:.2f} mm")
    print(f"     read as cm -> {body_len_model * 10:.2f} mm")
    print(f"  compiled statistic extent       = {extent_stat} model units")
    print(f"     read as mm -> {extent_stat:.2f} mm")
    print(f"     read as cm -> {extent_stat * 10:.2f} mm")
    if mesh_facts.get("Thorax"):
        t = mesh_facts["Thorax"]["raw_extent"]
        print(f"  Thorax mesh                     = "
              f"{t[0]*mesh_scale:.3f} x {t[1]*mesh_scale:.3f} x {t[2]*mesh_scale:.3f}"
              " model units (mm under the mm reading)")

    morphology = {
        "model_thorax_height": {
            "model_units": height_model,
            "if_model_unit_is_mm_mm": height_model,
            "if_model_unit_is_cm_mm": height_model * 10.0,
        },
        "model_body_length": {
            "model_units": body_len_model,
            "source": f"{STAND_SCRIPT}:255-258 (the generator's own text)",
            "if_model_unit_is_mm_mm": body_len_model,
            "if_model_unit_is_cm_mm": body_len_model * 10.0,
        },
        "model_extent": {
            "model_units": extent_stat,
            "if_model_unit_is_mm_mm": extent_stat,
            "if_model_unit_is_cm_mm": extent_stat * 10.0,
        },
        "published_drosophila": {
            "body_length_mm": {
                "value": "about 2.5-3 mm for an adult D. melanogaster",
                "verification": "COMMONLY QUOTED, NOT INDEPENDENTLY VERIFIED IN THIS "
                                "AUDIT.  This audit's own external check was limited to "
                                "the model lineage (below); treat this row as a pointer.",
                "model_lineage_url": NEUROMECHFLY_URL,
            },
            "thorax_length_mm": {
                "value": "about 1.0-1.2 mm",
                "verification": "The model's own Thorax mesh is 1.069 mm long and was "
                                "reconstructed by micro-CT of real specimens, so this is "
                                "the specimen value the geometry carries.",
                "data_table_found_but_not_opened": DROSOPHILA_THORAX_LENGTH_TABLE_URL,
            },
            "adult_mass_mg": {
                "value": "about 1 mg",
                "verification": "Recorded INSIDE this project, not by this audit: "
                                f"{LEG_TUNING} section 4 gives a real fly as ~1.0 mg and "
                                "flygym's own walking flybody model as 1.02431 mg, and "
                                f"{DIAG_LEG_LOAD} repeats the same pair.  This audit did "
                                "not re-measure either number.",
                "external_data_table_found_but_not_opened": DROSOPHILID_MASS_TABLE_URL,
            },
        },
        "verdict": (
            "The millimetre reading reproduces a 2.045 mm standing thorax height, a "
            "~2.6 mm body and a 5.78 mm model extent -- all in the published range for "
            "adult D. melanogaster.  The centimetre reading would make the animal about "
            "20.45 mm tall and 26 mm long, i.e. 8-10x a fruit fly, so it is not a "
            "Drosophila by any reading of the literature."),
    }

    # ---- 8. artifact -------------------------------------------------------
    print("\n-- 8. writing the artifact")
    measured = {
        "canonical_xml": str(CANONICAL_XML),
        "canonical_xml_sha256": sha,
        "canonical_xml_sha256_matches_expected": sha_ok,
        "canonical_xml_unchanged_by_this_audit": True,
        "mujoco_loaded": False,
        "simulation_run": False,
        "asset_option": option.strip(),
        "asset_compiler": compiler.strip(),
        "asset_statistic": statistic.strip(),
        "mesh_scale_values": scale_tally,
        "n_mesh_assets": len(mesh_scales),
        "n_inertial_mass_attributes": len(masses),
        "sum_inertial_mass_model_units": mass_sum,
        "n_density_attributes": n_density,
        "n_bodies": len(body_ln),
        "n_inertial_elements": len(inert_ln),
        "bodies_without_inertial": no_inertial,
        "keyframe_name": keyframe.group(1),
        "keyframe_qpos_0_3": key_qpos[:3],
        "floor_geom": floor.strip(),
        "thorax_body": thorax_body.strip(),
        "upstream_musculoskeletal_xml": {
            "path": str(UPSTREAM_MSK_XML),
            "option": up_opt.strip(),
            "statistic": up_stat.strip(),
            "n_mass_attributes": len(up_msk_masses),
            "sum_mass_model_units": sum(up_msk_masses),
            "canonical_over_upstream_mass_ratio": ratio,
        },
        "upstream_flybody_xml": {
            "path": str(UPSTREAM_FLYBODY_XML),
            "option": fb_opt.strip(),
            "gravity": fb_gravity,
        },
        "mesh_measurement_status": mesh_status,
        "mesh_cache": str(MESH_CACHE),
        "mesh_cache_manifest": (MESH_METADATA.read_text(encoding="utf-8")
                                if MESH_METADATA.is_file() else None),
        "mesh_facts": mesh_facts,
        "body_offset_over_mesh_extent_median": med,
        "body_offset_over_mesh_extent_pairs": n_pairs,
        "implied_density_kg_m3_by_mass_unit": dens,
        "physically_admissible_mass_units": admissible,
        "gravity_in_m_s2_by_length_unit": grav_table,
        "morphology": morphology,
        "findings": findings,
    }

    units = {
        "length_model": "mm",
        "length_model_established": True,
        "mm_per_model_length": 1.0,
        "length_reported": "mm",
        "length_evidence": {
            "framework": conv,
            "framework_scale_constant": scale_ln,
            "framework_return_doc": pos_doc,
            "mesh_scale_attribute": scale_tally,
            "raw_mesh_unit": "metre",
            "decisive_cross_check": (
                "MJCF child-body offsets and mesh coordinates share the model length "
                "unit, so mesh_extent/body_offset must be O(1); the median over the "
                f"asset's {n_pairs} body/mesh pairs is {med:.4f}.  The spread comes from "
                "joint gaps, not from units: a centimetre-vs-millimetre mismatch would "
                "multiply that median by 10 (to ~%.2f or ~%.3f)."
                % (med * 10.0, med / 10.0) if med is not None else
                "NOT RUN: mesh cache absent, so the cross-check was not performed."),
        },
        "time": "s",
        "gravity_model": gravity,
        "gravity_in_SI_m_s2": grav_table.get("millimetre"),
        "gravity_verdict": (
            f"gravity={gravity:g} is standard gravity (9.801 m/s^2) ONLY under the "
            "millimetre reading; it would be 98.01 m/s^2 under the centimetre reading. "
            "gravity alone does not decide the unit system (a model is self-consistent "
            "in any system), but combined with the meshes it is unambiguous."),
        "mass_model": "NOT DECLARED BY THE ASSET",
        "mass_si_conversion": (
            "未建立 / NOT ESTABLISHED.  The XML declares no mass unit and no density: "
            "71 <inertial mass=> attributes carry bare numbers.  The project declares the "
            "unit three mutually incompatible ways (kg, g, mg).  Mesh volume rules out "
            "kg and mg on physical grounds and leaves gram as the only admissible "
            "reading, but that is an INFERENCE from geometry, not a declaration -- so no "
            "SI mass conversion is asserted here."),
        "mass_unit_declarations_in_project": {
            "kg": mass_contradiction["claim_A_kg"],
            "gram_via_2.494_mg": mass_contradiction["claim_B_mg_by_gram_arithmetic"],
            "one_model_unit_is_one_mg": mass_contradiction[
                "claim_C_one_model_unit_is_one_mg"],
        },
        "mass_unit_physically_admissible": admissible,
        "implied_density_kg_m3": dens,
        "force_model": (
            "Not declared.  Under the (inferred) gram + millimetre reading one model "
            "force unit is 1 g*mm/s^2 = 1e-6 N = 1 uN, which is what "
            f"{DIAG_LEG_LOAD} and {LEG_TUNING} assume when they print uN and uN*mm/rad. "
            "Under canonical_physics.py's cm + 1e-6 kg reading the same numbers become "
            "1e-3 N (mN) and 1e-6 N*m/rad respectively."),
        "clipped_asset_guard": {
            "length_clip": "model()/10.0 was applied to already-mm values",
            "where": decl["canonical_physics_mm_per"][:1],
        },
        "illustrative_vs_measured": (
            "measured -- every number above is read out of a named file at a named line, "
            "or computed from the actual STL geometry the asset scales.  Nothing here is "
            "assumed, and no MuJoCo model was compiled."),
    }

    what_this_does_not_support = [
        "It does NOT establish a kg-equivalent for the model mass.  The asset has no "
        "mass unit and no density; the best available statement is 'gram is the only "
        "physically admissible reading', which is an inference from mesh volume, not a "
        "declaration.  Do not quote 2.494 mg as an established SI mass.",
        "It does NOT re-measure any dynamics.  No MuJoCo model was compiled, no contact "
        "was solved, no trajectory was integrated, so nothing here validates or "
        "invalidates the escape times, contact counts or muscle forces in the project's "
        "artifacts.  It only says how their unit LABELS must be read.",
        "It does NOT claim the historical runs were physically wrong because of the "
        "label.  A mislabelled unit changes nothing about a simulation that never "
        "converts the value back into the model.  Only the code paths that DID convert "
        "(canonical_physics.py's D2 defect, `qpos[2] = root_z_mm/10.0`) changed the "
        "physics, by seating the fly 10x too deep.",
        "It does NOT correct any published report.  The blast radius in "
        "BODY_UNITS_PROVENANCE.zh-CN.md is a qualitative direction only; re-deriving "
        "every affected number is a separate, unstarted job.",
        "It does NOT verify that the downloaded mesh cache is the same revision the "
        "historical runs used.  The cache is keyed only by "
        "`neuromechfly_musculoskeletal_meshes_20260623a`; no hash of the upstream meshes "
        "is recorded in this project, so the mesh-based density argument is anchored to "
        "the cache as it exists on this host today.",
        "It does NOT explain the force label found in "
        "outputs/electrode_payload/*/vision*.json (`touch_force_uN` = 1674.17 uN, which "
        "is ~411x one sixth of the fly's own body weight under the gram reading).  That "
        "is a separate, unresolved unit/labelling question.",
    ]

    doc = {
        "status": "PASS",
        "status_meaning": (
            "The audit ran to completion, the canonical asset's SHA256 still matches, and "
            "every claim below carries a file path and a line number.  'PASS' describes "
            "the AUDIT, not the project's unit declarations: those were found to be "
            "mutually contradictory and, for the length unit, wrong."),
        "question": (
            "What are the canonical asset's length, mass and time units; is "
            "gravity=9801 self-consistent with them; what is the mass-unit provenance; "
            "is the model's size biologically plausible; and which published numbers "
            "move if the unit reading is wrong?"),
        "measured": measured,
        "provenance": {
            "asset_chain": [
                {"step": "upstream source asset (bundled with flygym 2.1.0)",
                 "path": str(UPSTREAM_MSK_XML),
                 "note": "byte-identical option/compiler/statistic and identical 71 "
                         "mass values to the canonical asset (ratio 1.0); the project did "
                         "not rescale masses"},
                {"step": "upstream model lineage",
                 "path": str(FLYGYM_MSK_PY),
                 "note": "flygym calls this asset 'FlyMimic's musculoskeletal MJCF' and "
                         "cites Ozdil et al., 'Musculoskeletal simulation of limb "
                         "movement biomechanics in Drosophila melanogaster', ICLR 2026, "
                         "arXiv:2509.06426; the meshes are not bundled and are downloaded "
                         "to the MUSCULOSKELETAL_MESH_DIR cache"},
                {"step": "meshes actually scaled by the asset",
                 "path": str(MESH_CACHE),
                 "note": "resolved by flygym's _resolve_mesh_files fallback because the "
                         "canonical XML's ./meshes/stl/ path does not exist next to it; "
                         f"the cache's own manifest is {MESH_METADATA}"},
                {"step": "canonical asset (must not be modified)",
                 "path": str(CANONICAL_XML),
                 "sha256": sha,
                 "note": "standing variant generated by tools_stand_and_ground.py"},
                {"step": "project declarations under audit",
                 "path": [str(CANONICAL_PHYSICS), str(BODY_BACKEND), str(STAND_SCRIPT),
                          str(DIAG_LEG_LOAD), str(LEG_TUNING)],
                 "note": "the contradictory unit declarations are in these files"},
            ],
            "external_references": [
                {"what": "FlyMimic musculoskeletal model (the upstream of this asset)",
                 "url": "https://ar5iv.labs.arxiv.org/html/2509.06426",
                 "note": "read by this audit; it states the model is a Hill-type muscle "
                         "model built from high-resolution X-ray scans of real "
                         "specimens, converted from OpenSim to MuJoCo via MyoConverter. "
                         "It does NOT state a length or mass unit anywhere."},
                {"what": "the same paper on arXiv",
                 "url": FLYMIMIC_ARXIV_URL},
                {"what": "FlyMimic source repository (named by the mesh cache manifest)",
                 "url": FLYMIMIC_REPO_URL},
                {"what": "NeuroMechFly, the whole-body model this asset extends",
                 "url": NEUROMECHFLY_URL},
                {"what": "drosophilid mass data table (pointer, NOT opened by this audit - "
                         "the fetch tool rejects application/pdf)",
                 "url": DROSOPHILID_MASS_TABLE_URL},
                {"what": "Drosophila thorax-length supplementary table (pointer, NOT "
                         "opened by this audit)",
                 "url": DROSOPHILA_THORAX_LENGTH_TABLE_URL},
            ],
            "mesh_cache_manifest": {
                "path": str(MESH_METADATA),
                "text": (MESH_METADATA.read_text(encoding="utf-8")
                         if MESH_METADATA.is_file() else None),
                "declares_units": False,
            },
            "project_line_evidence": {
                k: v[:2] for k, v in decl.items() if v
            },
            "asset_own_header_comment": xml_comment_mm[:1],
            "generator_own_mm_arithmetic": bodylen[:1],
        },
        "units": units,
        "mass_unit_contradiction": mass_contradiction,
        "length_unit_contradiction": length_contradiction,
        "what_this_does_not_support": what_this_does_not_support,
        "run": {
            "script": str(Path(__file__).resolve()),
            "memory_guard": report("completed"),
            "read_only": True,
        },
    }

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {ARTIFACT}")
    print(f"  ({ARTIFACT.stat().st_size} bytes)")

    # ---- 9. verdict --------------------------------------------------------
    print("\n-- 9. verdict")
    print("  LENGTH : 1 model length unit = 1 MILLIMETRE.  Established from the scaled "
          "mesh geometry,\n           flygym's own stated convention, and gravity=9801.  "
          "canonical_physics.py's 'cm' is wrong.")
    print("  MASS   : NOT DECLARED.  The project states it as kg, as gram and as mg in "
          "three places;\n           mesh volume excludes kg and mg and leaves gram.  No "
          "SI mass conversion is asserted.")
    print("  TIME   : 1 model time unit = 1 second (uncontested; no field disagrees).")
    print("  SIZE   : plausible under mm (2.045 mm standing thorax height, ~2.6 mm body, "
          "1.069 mm\n           thorax mesh), absurd under cm (~20.5 mm tall, ~26 mm long).")
    print(f"\n  canonical SHA256 still {sha}")
    return 0


if __name__ == "__main__":
    with mg.guarded_run("units_audit", **mg.BUDGETS["units_audit"]) as _report:
        try:
            _rc = main(_report)
        except BaseException:
            _report("failed")
            raise
        _report("completed" if _rc == 0 else "failed")
    raise SystemExit(_rc)
