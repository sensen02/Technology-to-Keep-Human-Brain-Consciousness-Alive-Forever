"""Dry contact prototype and opt-in full-body MJCF derivative."""

from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from flygym.compose.fly.musculoskeletal import _load_mjcf

PROJECT = Path(__file__).resolve().parents[3]
WALKING_XML = PROJECT / "experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml"
HERE = Path(__file__).resolve().parent
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")


def capsules(with_tips=True):
    """Provide two open hooks in normalized fixture coordinates."""
    for side, y in enumerate((-0.26, 0.26)):
        points = np.array([[-0.45, y, -0.32], [-0.45, y, 0.42],
                           [0.45, y, 0.42], [0.45, y, -0.32]])
        for link in (range(3) if with_tips else (1,)):
            yield f"{side}_{link}", points[link], points[link + 1]


def vector(numbers):
    return " ".join(format(x, ".17g") for x in numbers)


def add_claws(body, name, transform, scale, *, tips=True):
    for label, start, end in capsules(tips):
        ET.SubElement(body, "geom", name=f"{name}_claw_{label}", type="capsule",
                      fromto=vector(np.r_[transform(start), transform(end)]),
                      size=str(0.065 * scale), mass="0", friction="0.01 0 0",
                      contype="2", conaffinity="1", rgba=".22 .12 .06 1")


def fixture(dt, tips, has_edge):
    tree = ET.fromstring(f"""
    <mujoco><option timestep="{dt}" gravity="0 0 0" integrator="implicitfast"/>
    <worldbody>
      <geom name="wall" type="box" pos="0 1 0" size="4 .2 3"
            contype="1" conaffinity="2" friction=".01 0 0"/>
      <body name="foot">
        <inertial pos="0 0 0" mass=".03" diaginertia=".002 .002 .002"/>
        <joint name="pull" type="slide" axis="1 0 0" damping=".16"/>
        <joint name="insert" type="slide" axis="0 0 1" damping=".3"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="insert" kp="160" kv="4" forcerange="-12 12"/>
    </actuator></mujoco>""")
    if has_edge:
        ET.SubElement(tree.find("worldbody"), "geom", name="ridge", type="box",
                      pos="0 .29 0", size=".12 .52 .18", friction=".01 0 0",
                      contype="1", conaffinity="2")
    add_claws(tree.find(".//body"), "test", lambda x: x, 1, tips=tips)
    return mj.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))


def pull_trial(dt, tips, edge):
    model = fixture(dt, tips, edge)
    state = mj.MjData(model)
    state.qpos[1] = 0.9
    forces = np.zeros(6)
    held, clearance = [], []
    for tick in range(round(1.6 / dt)):
        time = tick * dt
        state.ctrl[0] = 0.0 if 0.1 <= time < 1.05 else 0.9
        state.qfrc_applied[0] = min(1, (time - 0.40) / .15) if 0.40 <= time < 0.97 else 0
        mj.mj_step(model, state)
        if 0.8 <= time < 0.95:
            # Capture the resistance from the constraint evaluated by mj_step.
            axial_normal = 0.0
            for idx in range(state.ncon):
                contact = state.contact[idx]
                mj.mj_contactForce(model, state, idx, forces)
                axial_normal += forces[0] * abs(contact.frame[0])
            held.append((float(state.qpos[0]), axial_normal))
        if time >= 1.45:
            clearance.append(state.ncon)
    slide = np.array(held)
    return dict(dt=dt, tips=tips, edge=edge, max_x=float(slide[:, 0].max()),
                load_window_motion=float(np.ptp(slide[:, 0])),
                min_axial_normal=float(slide[:, 1].min()),
                detached=max(clearance) == 0, terminal_lift=float(state.qpos[1]))


def build_whole_body(destination):
    if destination.exists():
        raise FileExistsError(destination)
    base = _load_mjcf(WALKING_XML).compile()
    baseline = mj.MjData(base)
    mj.mj_resetDataKeyframe(base, baseline, 0)
    mj.mj_fwdPosition(base, baseline)
    tree = ET.parse(WALKING_XML)
    geom_scale = 0.008  # Project triad-pad radius provides an engineering scale.
    for leg in LEGS:
        name = f"{leg}Tarsus5"
        foot = tree.find(f".//body[@name='{name}']")
        orientation = baseline.xmat[base.body(name).id].reshape(3, 3)
        up = orientation.T @ np.array([0.0, 0.0, 1.0])
        pads = np.array([np.fromstring(g.attrib["pos"], sep=" ") for g in
                         foot.findall("geom") if g.attrib["name"].startswith(f"{name}_pad")])
        assert pads.shape == (3, 3)
        fore = pads[0] - pads[1:].mean(axis=0)
        fore -= up * (fore @ up)
        fore /= np.linalg.norm(fore)
        cross = np.cross(up, fore)
        center = pads.mean(axis=0)
        # Keep a clearance margin above the triad's lowest support point at this pose.
        center += up * (np.min(pads @ up) - geom_scale - center @ up
                        + geom_scale * (.32 + .065 + .25))
        add_claws(foot, name, lambda v: center + geom_scale * (
            v[0] * fore + v[1] * cross + v[2] * up), geom_scale)
    with destination.open("xb") as stream:
        tree.write(stream, encoding="utf-8")
    model = _load_mjcf(destination).compile()
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat",
                  "jnt_range", "jnt_axis", "actuator_ctrlrange", "actuator_trnid"):
        assert np.array_equal(getattr(model, field), getattr(base, field)), field
    state = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, state, 0)
    mj.mj_fwdPosition(model, state)
    foot_ids = {}
    for leg in LEGS:
        foot_ids[leg] = [model.geom(i).name for i in range(model.ngeom)
                        if model.geom(i).name.startswith(f"{leg}Tarsus5_claw_")]
        assert len(foot_ids[leg]) == 6
        for name in foot_ids[leg]:
            gid = model.geom(name).id
            assert model.geom_contype[gid] == 2 and model.geom_conaffinity[gid] == 1
            extent_z = (abs(state.geom_xmat[gid].reshape(3, 3)[2, 2]) *
                        model.geom_size[gid, 1] + model.geom_size[gid, 0])
            low = state.geom_xpos[gid, 2] - extent_z
            pad_low = min(state.geom_xpos[model.geom(f"{leg}Tarsus5_pad{p}").id, 2]
                          - geom_scale for p in range(3))
            assert low > pad_low + 0.001
    return foot_ids


def main():
    before = hashlib.sha256(WALKING_XML.read_bytes()).hexdigest()
    assert before == "d0839271100561cbd288cc662f0c94639755e6ed773a2b7a7d547f5fec2fe93a"
    trials = [pull_trial(dt, tips, edge) for dt in (0.001, 0.0005)
              for tips, edge in ((True, True), (False, True), (True, False))]
    passed = all(t["detached"] and t["terminal_lift"] > .85 and (
        t["min_axial_normal"] >= .8 and t["load_window_motion"] < .02
        if t["tips"] and t["edge"] else t["max_x"] > .25 and t["min_axial_normal"] < .05
    ) for t in trials)
    result = dict(fixture_pass=passed, trials=trials, whole_body=None)
    if passed:
        result["whole_body"] = build_whole_body(HERE / "BODY_CLAWS_EXPERIMENTAL.xml")
    assert hashlib.sha256(WALKING_XML.read_bytes()).hexdigest() == before
    with (HERE / "HOOK_RESULT.json").open("x") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(dict(fixture_pass=passed, trials=trials)))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
