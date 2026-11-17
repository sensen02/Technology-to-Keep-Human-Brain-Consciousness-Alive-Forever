"""Cross forefoot support and lateral contact in a source-identified engineering bench."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from bind import BINDING, GRAPH, LOAD, TOUCH
from sensory_corpus import Apparatus, FRONT
from touch_runtime import W, TOUCH_W, Fixture, Recording, SourceTransport, TouchLink, sha, turn

DURATION = 680
GROUPS = 16
SCALE = .45


@lru_cache(maxsize=1)
def brain_projection():
    graph = json.loads(GRAPH.read_text())
    roots = TOUCH + LOAD
    targets = sorted({row["post"] for root in roots for row in graph["brain_outputs"][root]})
    matrix = np.zeros((len(targets),4))
    index = {root:i for i,root in enumerate(targets)}
    for column, root in enumerate(roots):
        for item in graph["brain_outputs"][root]:
            matrix[index[item["post"]],column] = item["count"]
    matrix /= matrix.sum(1,keepdims=True)
    pools = [np.flatnonzero((matrix[:,i] > 0.) & (np.count_nonzero(matrix,axis=1)==1)) for i in range(4)]
    assert all(len(rows) for rows in pools)
    return matrix, pools


class SeparatedLink(TouchLink):
    def __init__(self):
        super().__init__()
        original_an = self.an_ids
        self.transport = SourceTransport(BINDING,sha(BINDING))
        self.an_ids = self.transport.ports["AN"]
        assert list(TOUCH+LOAD) == list(self.an_ids)
        assert self.root_ids == self.transport.ports["DN"]
        assert list(original_an) == list(self.an_ids[:2])
        self.matrix, self.pool = brain_projection()
        self.an = np.zeros(4)

    def observe_packet(self):
        values = self.observed()
        if values is None:
            return np.zeros(4),False
        return values,True


def case_for(index):
    controls = np.random.default_rng(224240+index)
    return dict(group=index, side=index%2,
                split="train" if index<8 else "calibration" if index<12 else "held",
                onset=int(controls.integers(185,340)),
                length=int(controls.integers(44,75)),
                depth=float(controls.uniform(.0031,.0065)),
                move=int(controls.integers(420,580)))


def expose(bench, path, case, support, lateral):
    bench.reset()
    path.mkdir()
    body = bench.body
    names = [body.plant.geom(i).name for i in range(body.plant.ngeom)]
    row_ids = np.asarray([i for i,n in enumerate(names) if n.startswith(("LF","RF")[case["side"]]+"Tarsus")
                          and body.plant.geom_contype[i] == 2])
    assert len(row_ids)==7
    probe_id = body.plant.geom("touch_tip").id
    affine = body.plant.geom_conaffinity.copy()
    types = body.plant.geom_contype.copy()
    # Set plane vs selected foot permissions at the start of this trial. Leave
    # the other five supports and the calibrated lateral fixture available.
    body.plant.geom_contype[probe_id]=4
    body.plant.geom_conaffinity[probe_id]=2
    if not support:
        body.plant.geom_conaffinity[row_ids]=4
    mj.mj_forward(body.plant,body.d)
    fixture = Fixture(body,case["side"],case["onset"],case["depth"] if lateral else -.010,case["length"])
    recording = Recording(body,DURATION)
    body.recorder = recording
    link = SeparatedLink()
    measured = body.measure()
    vectors, truths, inputs, motor, poses, angles = [], [], [], [], [], []
    action = np.zeros(42)
    try:
        for ms in range(DURATION):
            arrived,valid = link.observe_packet()
            normal = measured["loads"][FRONT]/W
            sensed = np.r_[recording.activity(), np.clip(normal/SCALE,0.,1.)]
            vectors.append(np.r_[arrived,valid])
            truths.append(np.r_[normal,measured["nonfoot"]/W,body.d.subtree_com[body.th,2]])
            inputs.append(sensed)
            if ms%5==0:
                command = int(ms>=case["move"])
                link.send(command)
                action = bench.motor(turn.features(body,measured),link.action(),0)
                poses.append(np.r_[body.d.qpos,body.d.qvel])
                angles.append(body.d.qpos[bench.angles].copy())
            motor.append(command)
            fixture.move(ms)
            link.sense(sensed.tolist())
            measured = body.tick(action)
            if body.d.qfrc_applied.any() or body.d.xfrc_applied.any():
                raise RuntimeError("Unexpected applied force")
            link.advance()
    finally:
        body.plant.geom_contype[:] = types
        body.plant.geom_conaffinity[:] = affine
        body.recorder = None
    raw = recording.raw
    assert recording.count==DURATION*80
    assert np.isfinite(raw).all()
    truths = np.asarray(truths)
    actual = (raw[:,2:4].reshape(DURATION,80,2)>W*TOUCH_W).sum(1)>=8
    load = truths[:,:2]
    side = case["side"]
    onset = np.flatnonzero(actual[:,side])
    # A genuine intervention is verified by forces at the native contact pair,
    # and by load on the separate floor. Load status must hold through validation windows.
    checks = dict(tactile=bool(len(onset)) == lateral,
                  opposite_tactile_zero=not bool(actual[:,1-side].any()),
                  other_probe_contact_zero=bool(raw[:,4].max()<W*TOUCH_W),
                  other_ground_collision_safe=bool(truths[:,2].max()<.2),
                  stable_com=bool(truths[:,3].min()>=1.20))
    decision_window = (np.arange(DURATION)>=case["onset"]) & (
        np.arange(DURATION)<case["onset"]+case["length"])
    checks["normal_load_bracket"] = bool(np.median(load[decision_window,side])>=.05) if support else bool(
        load[decision_window,side].max()<.02)
    np.savez_compressed(path/"SAMPLES.npz",received=np.asarray(vectors),input_activity=np.asarray(inputs),
                        normal_load=load,truth_tactile=actual,body_truth=truths,
                        native=raw,commands=np.asarray(motor),poses=np.asarray(poses),angles=np.asarray(angles))
    result = dict(case=case,support=support,lateral=lateral,checks=checks,passed=all(checks.values()),
                  median_force_bodyweight=float(np.median(load[decision_window,side])),
                  physical_probe_ms=int(onset[0]) if len(onset) else None,native_steps=recording.count,
                  sha256=sha(path/"SAMPLES.npz"),routing=link.transport.verify_final())
    turn.stand.put(path/"RESULT.json",result)
    print(path, result["median_force_bodyweight"],result["physical_probe_ms"],checks,flush=True)
    return result


def main(root,groups):
    torch.set_num_threads(1)
    target = root/f"worker_{min(groups):02d}"
    target.mkdir()
    bench = Apparatus(target)
    pin_paths = [Path(__file__), BINDING, GRAPH, Path(__file__).with_name("bind.py")]
    original = {str(path):sha(path) for path in pin_paths}
    turn.stand.put(target/"INPUT_IDENTITY.json",original)
    for group in groups:
        case = case_for(group)
        folder = root/f"group_{group:02d}"
        folder.mkdir()
        for loaded in (False,True):
            for touched in (False,True):
                condition = f"L{int(loaded)}T{int(touched)}"
                expose(bench,folder/condition,case,loaded,touched)
    assert original=={str(path):sha(path) for path in pin_paths}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("groups",type=int,nargs="+")
    args=parser.parse_args()
    main(args.root,args.groups)
