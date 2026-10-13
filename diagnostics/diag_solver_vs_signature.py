"""DIAGNOSTIC (read-only on the canonical asset): is the '26-34 ms collapse' a property of the
fly body, or of the contact/solver numerics that were never matched to the upstream calibrated
asset?

Everything below is held fixed except the NUMERICS:
  - the same canonical XML (never modified on disk; every variant is an in-memory string)
  - the same zero-adhesion balance pose from round 37 (root_z + 42 joint offsets)
  - the same constant LP activation vector from round 37
  - the same 0.30 s horizon and the same escape criterion

This is a DIAGNOSTIC CONTROL, not an autonomous result, and not a claim about fly behaviour.

Upstream reference (flygym 2.1.0, assets/model/flybody/fruitfly.xml):
  solref = 0.0002 1   (MuJoCo default 0.02 -> 100x softer)
  solimp = 0.95 0.99 0.01
  margin = 1e-3       (MuJoCo default 0)
The canonical asset has NO solref / solimp / <pair> / margin at all, i.e. all MuJoCo defaults.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import mujoco
import numpy as np

ROOT = "/run/media/sensen/Data2/cell_wound_prototype"
XML = os.path.join(ROOT, "outputs/muscles_six_legs/fruitfly_six_leg_standing.xml")
POSE = os.path.join(ROOT, "outputs/stance_margin_round37/stance_for_reflex.json")
OUT = os.path.join(ROOT, "outputs/diag_solver_signature")
os.makedirs(OUT, exist_ok=True)

HORIZON = 0.30          # s, same order as round 36/37
ESCAPE_MM = 1.8         # round 36 criterion (1.8 mm below the seated height)


def patch_xml(src: str, timestep: float, solref: tuple | None, solimp: tuple | None,
              margin: float | None, integrator: str | None, solver: str | None,
              iterations: int | None, tolerance: float | None) -> str:
    """Return the canonical XML with ONLY numeric settings changed."""
    import re

    opt = re.search(r"<option[^>]*/>", src)
    assert opt, "no <option> tag"
    attrs = {
        "timestep": repr(timestep) if timestep is not None else None,
        "integrator": integrator,
        "solver": solver,
        "iterations": str(iterations) if iterations is not None else None,
        "tolerance": repr(tolerance) if tolerance is not None else None,
    }
    tag = opt.group(0)
    for k, v in attrs.items():
        if v is None:
            continue
        if re.search(rf'{k}="[^"]*"', tag):
            tag = re.sub(rf'{k}="[^"]*"', f'{k}="{v}"', tag)
        else:
            tag = tag[:-2] + f' {k}="{v}"/>'
    src = src[: opt.start()] + tag + src[opt.end():]

    # a global contact default, which is how MuJoCo applies solref/solimp/margin to
    # every contact that has no explicit <pair> (the canonical asset has none)
    if solref is not None or solimp is not None or margin is not None:
        bits = []
        if solref is not None:
            bits.append(f'solref="{solref[0]:g} {solref[1]:g}"')
        if solimp is not None:
            bits.append('solimp="{}"'.format(" ".join(f"{x:g}" for x in solimp)))
        if margin is not None:
            bits.append(f'margin="{margin:g}"')
        # A root-level defaults class is inherited by every element that does not set the
        # attribute itself.  It has to be a plain child of <mujoco>, and it must come
        # BEFORE the existing <default class="main"> so that main keeps its own values.
        import re as _re

        block = ('  <default>\n'
                 f'    <geom {" ".join(bits)}/>\n'
                 '  </default>\n')
        m = _re.search(r"<mujoco[^>]*>", src)
        assert m, "no <mujoco> root"
        src = src[: m.end()] + "\n" + block + src[m.end():]
    return src


def run(tag: str, **kw) -> dict:
    src = patch_xml(open(XML).read(), **kw)
    # the canonical asset resolves meshes as ./meshes/stl/... so the variant has to live
    # next to it; the canonical file itself is never touched
    tmp = os.path.join(os.path.dirname(XML), ".diag_variant_tmp.xml")
    with open(tmp, "w") as fh:
        fh.write(src)
    # flygym's loader resolves the asset's mesh paths (it stages the mesh cache); this is
    # the same entry point the project's own stance search uses, so the variant is loaded
    # exactly the way the canonical asset is.
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    model = _load_mjcf(tmp).compile()
    os.unlink(tmp)

    pose = json.load(open(POSE))
    data = mujoco.MjData(model)
    key = model.key_qpos[0].copy()

    # 1) seat the root and the 42 joint offsets
    names = {}
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        names[nm] = j
    qpos = key.copy()
    qpos[2] = pose["root_z_mm"] / 10.0  # mm -> model length unit (cm)
    for nm, off in pose["joint_offsets"].items():
        j = names[nm]
        qpos[model.jnt_qposadr[j]] = key[model.jnt_qposadr[j]] + off
    data.qpos[:] = qpos

    # 2) constant LP activation from round 37
    acts = json.load(open(os.path.join(ROOT, "outputs/stance_margin_round37/stance_margin.json")))
    act_map = acts["best"]["activation"]
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        data.ctrl[i] = float(act_map.get(nm, 0.0))
    data.ctrl[model.nu - 2:] = 0.0  # adhesion forces off (as in round 36/37)

    # settle the muscle states at the seated pose so the actuator dynamics do not
    # masquerade as a contact effect
    mujoco.mj_forward(model, data)

    h0 = data.xpos[model.body("Thorax").id][2]
    z = []
    first_escape = None
    n = int(round(HORIZON / model.opt.timestep))
    t0 = time.time()
    for k in range(n):
        mujoco.mj_step(model, data)
        h = data.xpos[model.body("Thorax").id][2]
        z.append(h)
        if first_escape is None and h < (h0 - ESCAPE_MM / 10.0):
            first_escape = (k + 1) * model.opt.timestep
    z = np.array(z)

    # contact-force sanity: sum of normal forces vs body weight, at the seated pose
    fs = np.zeros(6)
    for c in range(data.ncon):
        mujoco.mj_contactForce(model, data, c, fs)
        pass
    weight = abs(model.body_mass[1:].sum() * model.opt.gravity[2])

    return {
        "tag": tag,
        "settings": {k: (v if not isinstance(v, tuple) else list(v)) for k, v in kw.items()},
        "seated_height_mm": float(h0 * 10),
        "escape_time_s": first_escape,
        "final_height_mm": float(data.xpos[model.body("Thorax").id][2] * 10),
        "min_height_mm": float(z.min() * 10),
        "body_weight_model_units": float(weight),
        "wall_s": time.time() - t0,
        "n_contacts_final": int(data.ncon),
    }


def main() -> None:
    variants = [
        # tag,                    timestep, solref,        solimp,                    margin,  integrator,       solver, iter, tol
        ("A_canonical_defaults",  1e-4,  None,           None,                      None,    None,             None,   None, None),
        ("B_upstream_contact",    1e-4,  (2e-4, 1.0),    (0.95, 0.99, 0.01),        1e-3,    None,             None,   None, None),
        ("C_upstream_contact_dt1e-5", 1e-5, (2e-4, 1.0), (0.95, 0.99, 0.01),       1e-3,    None,             None,   None, None),
        ("D_upstream_dt1e-6",     1e-6,  (2e-4, 1.0),    (0.95, 0.99, 0.01),        1e-3,    None,             None,   None, None),
        ("E_soft_dt1e-6",         1e-6,  (2e-2, 1.0),    None,                      0.0,     None,             None,   None, None),
        ("F_upstream_tight_solver", 1e-5, (2e-4, 1.0),   (0.95, 0.99, 0.01),        1e-3,    "implicitfast",   "Newton", 200, 1e-12),
        ("G_upstream_implicit",   1e-4,  (2e-4, 1.0),    (0.95, 0.99, 0.01),        1e-3,    "implicit",       "Newton", 200, 1e-12),
        ("H_canonical_tight_dt1e-5", 1e-5, None,         None,                      None,    None,             None,   None, None),
    ]
    results = []
    for tag, dt, sr, si, mg, integ, solv, it, tol in variants:
        try:
            r = run(tag, timestep=dt, solref=sr, solimp=si, margin=mg,
                    integrator=integ, solver=solv, iterations=it, tolerance=tol)
            print(json.dumps(r), flush=True)
        except Exception as e:  # keep the sweep going
            r = {"tag": tag, "error": f"{type(e).__name__}: {e}"}
            print(json.dumps(r), flush=True)
        results.append(r)
    with open(os.path.join(OUT, "diag_solver_signature.json"), "w") as fh:
        json.dump({
            "status": "DIAGNOSTIC_CONTROL_NOT_AUTONOMOUS",
            "question": ("is the 26-34 ms collapse a property of the fly BODY, or of the "
                         "contact/solver numerics that were never matched to the upstream "
                         "calibrated asset?"),
            "canonical_xml": XML,
            "canonical_xml_sha256": __import__("hashlib").sha256(open(XML, "rb").read()).hexdigest(),
            "fixed": {"pose": POSE, "activation": "round37 best LP activation (constant)",
                      "horizon_s": HORIZON, "escape_below_mm": ESCAPE_MM,
                      "adhesion": 0.0},
            "upstream_reference": {"source": "flygym 2.1.0 assets/model/flybody/fruitfly.xml",
                                   "solref": [2e-4, 1.0], "solimp": [0.95, 0.99, 0.01],
                                   "margin": 1e-3},
            "variants": results,
            "limitations": [
                "Diagnostic control, not an autonomous result.",
                "Escape time from one pose is a proxy, not a stability analysis.",
                "The canonical asset is never modified; variants are in-memory strings.",
                "A longer escape time is not standing: it only says the numerics were not the sole cause.",
            ],
        }, fh, indent=2)
    print("wrote", os.path.join(OUT, "diag_solver_signature.json"))


if __name__ == "__main__":
    main()
