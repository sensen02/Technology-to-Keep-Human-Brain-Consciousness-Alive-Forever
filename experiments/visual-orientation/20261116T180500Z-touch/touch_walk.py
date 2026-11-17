"""Run a learned sensory readout in the frozen visual walk-stop loop."""
import argparse
from pathlib import Path

import numpy as np
import torch

from collect_touch import classify, physical_onsets
from touch_runtime import (HERE, VISUAL, BINDING, GRAPH, START, W, TOUCH_W, build,
                           Fixture, Recording, TouchLink, Eyes, visual_encoder, WalkHead,
                           turn, stand_window, load_body_policy, sha)

BEGIN = 780
END = 4050
FINISH = END + 1260
CONTACT = 2400
WEIGHT = VISUAL / "WALK_HEAD.pt"


def trial(root, mode):
    torch.set_num_threads(1)
    out = root/mode
    out.mkdir()
    model_path = HERE/"DATA/TOUCH_READOUT.json"
    readout = turn.read(model_path)
    assert (readout["binding_sha256"],readout["graph_sha256"]) == (sha(BINDING),sha(GRAPH))
    assert turn.read(model_path.with_name("TOUCH_READOUT_ACCEPTANCE.json"))["passed"]
    brain = WalkHead().eval()
    brain.load_state_dict(torch.load(WEIGHT,map_location="cpu",weights_only=True))
    encoder = visual_encoder()
    body = build(out)
    side = int(mode == "right")
    fixture = Fixture(body,side,CONTACT,-.010 if mode=="sham" else .0038,55)
    motor = load_body_policy()
    rec = Recording(body,FINISH)
    body.recorder = rec
    link = TouchLink()
    if mode in {"cut","restore"}:
        link.cut_an(True)
    dependencies = [Path(__file__),HERE/"touch_runtime.py",HERE/"probe_contact.so",
                    START,model_path,GRAPH,BINDING,WEIGHT]
    identity = turn.stand.identity.capture(body,dict(scope="visual hold plus tactile veto",
                                            mode=mode),extra_paths=dependencies)
    turn.stand.put(out/"RUN_IDENTITY.json",identity)
    eyes = Eyes(body)
    measured = body.measure()
    rows = [turn.sample(body,measured)]
    probe, commands, frames = [], [], []
    target = None
    visual_intent = 0
    paused = False
    first_decoded = None
    action = np.zeros(42)
    background = (eyes.pose()[0].copy(), .133, .27)
    try:
        for step in range(FINISH):
            if step==BEGIN:
                target = eyes.place(-1.5,3.52)
            if step==END:
                target = None
            if mode=="restore" and step==CONTACT+8:
                link.cut_an(False)
            if step%20==0:
                frame = eyes.capture(step,target,.161,(1.,.045,.035,1.),background)
                if frame is not None:
                    _, rgb = frame
                    with torch.inference_mode():
                        visual_intent = int(brain(encoder(torch.as_tensor(rgb[None]))[0]).item()>0.)
                    frames.append(rgb.copy())
            if step%5==0:
                decoded = classify(link.observed(),readout)
                if decoded >= 0 and not paused:
                    paused = True
                    first_decoded = (step,decoded)
                intended = int(visual_intent and not paused)
                link.send(intended)
                effective = link.action()
                action = motor(turn.features(body,measured),effective,0)
                pose = -1 if link.an_time is None else link.an_time
                commands.append((step,visual_intent,decoded,intended,effective,pose))
            fixture.move(step)
            probe.append(body.d.mocap_pos[0].copy())
            link.sense(rec.activity())
            measured = body.tick(action)
            if body.d.xfrc_applied.any() or body.d.qfrc_applied.any():
                raise RuntimeError("Unexpected applied wrench")
            rows.append(turn.sample(body,measured))
            link.advance()
    finally:
        eyes.close()
    assert rec.count == FINISH*80
    trace = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    commands = np.asarray(commands)
    onsets = physical_onsets(rec.raw)
    measured_onset = onsets[side]
    moved = np.flatnonzero(commands[:,4])
    first_forward_ms = int(commands[moved[0],0]) if len(moved) else None
    last_active = int(commands[moved[-1],0])+5 if len(moved) else None
    expected_tactile_stop = mode in {"intact","restore","right"}
    truthful = (measured_onset is None) if mode=="sham" else measured_onset is not None
    if expected_tactile_stop and first_decoded is not None:
        detection_delay = first_decoded[0]-measured_onset if measured_onset is not None else -1.
        tactile = first_decoded[1]==side and 0.<=detection_delay<=60. and abs(last_active-first_decoded[0])<=5
    else:
        tactile = first_decoded is None and last_active is not None and last_active>=END
        detection_delay = None
    stood = stand_window(trace,FINISH)
    after_stop = next((step - last_active for step in range(last_active+200,FINISH+1,5)
                       if stand_window(trace,step)["passed"]), None) if last_active is not None else None
    checks = dict(truthful_fixture=truthful and onsets[1-side] is None and rec.raw[:,4].max()/W<TOUCH_W,
                  correct_behavior=bool(tactile), paused_if_target_in_view=bool(not expected_tactile_stop or
                    (first_decoded is not None and commands[commands[:,0]==first_decoded[0],1].item()==1)),
                  starts=first_forward_ms is not None and first_forward_ms<CONTACT,
                  stable_finish=stood["passed"],no_floor_nonfoot=not trace["nonfoot"].any(),
                  settle=after_stop is not None and after_stop<=1200)
    if mode=="restore":
        checks["new_delivery_before_stop"] = bool(first_decoded is not None and
                commands[commands[:,0]==first_decoded[0],5].item() > CONTACT+8)
    np.savez_compressed(out/"RUN.npz",raw=rec.raw,commands=commands,probe=np.asarray(probe),
                        eyes=np.asarray(frames),**trace)
    turn.stand.put(out/"TRANSPORT.json",link.transport.journal())
    turn.stand.identity.verify_runtime(identity,body)
    outcome = dict(mode=mode,passed=all(checks.values()),checks={key:bool(value) for key,value in checks.items()},native_steps=rec.count,
                   onsets_ms=onsets,first_decoded=first_decoded,latency_ms=detection_delay,
                   first_forward_ms=first_forward_ms,last_forward_ms=last_active,
                   settle_ms=after_stop,stand=stood,route=link.transport.verify_final(),
                   hashes={p:sha(out/p) for p in ("RUN.npz","TRANSPORT.json")})
    turn.stand.put(out/"RESULT.json",outcome)
    print(outcome,flush=True)
    return outcome["passed"]


def audit(root):
    modes = ("intact","cut","restore","sham","right")
    result = {name:turn.read(root/name/"RESULT.json") for name in modes}
    exact = {}
    with np.load(root/"intact/RUN.npz",allow_pickle=False) as original:
        change = result["intact"]["last_forward_ms"]
        stop = change if change is not None else CONTACT
        for mode in ("cut","restore"):
            with np.load(root/mode/"RUN.npz",allow_pickle=False) as pair:
                comparisons = {key: bool(np.array_equal(original[key][:stop+1],pair[key][:stop+1]))
                               for key in ("qpos","qvel","com","loads")}
                # Both runs use the same contacts until the first change of motor action.
                comparisons["probe"] = bool(np.array_equal(original["raw"][:stop*80],pair["raw"][:stop*80]))
                exact[mode] = comparisons
    for mode,row in result.items():
        assert all(sha(root/mode/name)==digest for name,digest in row["hashes"].items())
    aligned = all(exact["cut"].values())
    passed = aligned and all(r["passed"] for r in result.values())
    summary = dict(passed=passed,matched_prefix=exact,visual_and_dn_frozen=True,
                   fit_scope="supervised tactile readout; explicit task pause policy",cases=result)
    turn.stand.put(root/"TOUCH_WALK_ACCEPTANCE.json",summary)
    print(passed,exact,flush=True)
    return passed


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("mode",choices=("intact","cut","restore","sham","right","audit"))
    a=p.parse_args()
    raise SystemExit(0 if (audit(a.root) if a.mode=="audit" else trial(a.root,a.mode)) else 1)
