"""Save same-stance native contact trials and supervise a two-sided brain readout."""
import argparse
from pathlib import Path

import numpy as np
import torch

from touch_runtime import (HERE, W, TOUCH_W, build, Fixture, Recording, TouchLink,
                           turn, load_body_policy, START, BINDING, GRAPH, sha)

FINISH = 675


def physical_onsets(raw):
    answers = []
    for side in range(2):
        indices = np.flatnonzero(raw[:,2+side] >= TOUCH_W*W)
        spans = np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
        span = next((part for part in spans if len(part)>=8),None)
        answers.append(float(span[0]/80) if span is not None else None)
    return answers


def plan(group, index):
    held = group=="held"
    if held:
        count = 12
        cls = index//count
        n = index%count
    else:
        count = 5
        cls = index//count
        n = index%count
    side = cls if cls<2 else n%2
    onset = (361 + 7*n if held else 345 + 11*n)
    depth = (.0025+.00034*n if held else .002+.0011*n) if cls<2 else -.008-.0007*n
    return dict(group=group,index=index,side=side,onset=onset,depth=depth,length=39+(n*7)%27,
                walk=(n%3!=0), expected_side=side if cls<2 else None)


def collect(root, group, indices):
    torch.set_num_threads(1)
    motor = load_body_policy()
    for index in indices:
        case = plan(group,index)
        out = root/f"{group}_{index:02d}"
        out.mkdir()
        body = build(out)
        rec = Recording(body,FINISH)
        fixture = Fixture(body,case["side"],case["onset"],case["depth"],case["length"])
        link = TouchLink()
        body.recorder = rec
        observed, states, delivered = [], [], []
        measured = body.measure()
        action = np.zeros(42)
        for step in range(FINISH):
            observed.append(link.observed())
            delivered.append(link.an_time)
            if step%5==0:
                action = motor(turn.features(body,measured),int(case["walk"] and step>=200),0)
            fixture.move(step)
            link.sense(rec.activity())
            measured = body.tick(action)
            if body.d.xfrc_applied.any() or body.d.qfrc_applied.any():
                raise RuntimeError("Unexpected applied wrench")
            states.append(np.r_[body.d.qpos,body.d.subtree_com[body.th]])
            link.advance()
        assert rec.count==FINISH*80
        assert np.allclose(np.diff(rec.raw[:,5]),12.5e-6,rtol=0.,atol=1e-10)
        observed = np.asarray([np.full(2,np.nan) if item is None else item for item in observed])
        np.savez_compressed(out/"SAMPLES.npz",raw=rec.raw,received=observed,
                            arrival_ms=np.array([-1 if n is None else n for n in delivered]),
                            body=np.asarray(states))
        truth = physical_onsets(rec.raw)
        real = [i for i,t in enumerate(truth) if t is not None]
        expected = [] if case["expected_side"] is None else [case["expected_side"]]
        result = dict(case=case,steps=rec.count,onset_ms=truth,physical_case_matched=real==expected,
                      unexpected_probe_force_W=float(rec.raw[:,4].max()/W),
                      peak_probe_force_W=(rec.raw[:,2:4].max(0)/W).tolist(),
                      zero_applied_wrench=True,sample_sha256=sha(out/"SAMPLES.npz"),
                      input_hashes={str(p):sha(p) for p in (START,BINDING,GRAPH,Path(__file__),HERE/"touch_runtime.py",HERE/"probe_contact.so")},
                      route_audit=link.transport.verify_final())
        turn.stand.put(out/"RESULT.json",result)
        print(out.name,truth,result["physical_case_matched"],result["unexpected_probe_force_W"],flush=True)


def load_case(folder):
    result = turn.read(folder/"RESULT.json")
    data = folder/"SAMPLES.npz"
    assert result["sample_sha256"] == sha(data)
    with np.load(data,allow_pickle=False) as archive:
        return result, archive["received"]


def fit(root):
    source = [load_case(path) for path in sorted(root.glob("train_*"))]
    assert len(source)==15 and all(x["physical_case_matched"] for x,_ in source)
    thresholds = []
    for side in range(2):
        non_touch = []
        touch = []
        for result, signals in source:
            onset = result["onset_ms"][side]
            if onset is None:
                non_touch.append(float(np.nanmax(signals[:,side])))
            else:
                start = int(np.ceil(onset))
                touch.append(float(np.nanmax(signals[start:start+50,side])))
        lower, upper = max(non_touch),min(touch)
        assert upper>lower and upper>0.
        thresholds.append(float(lower + .20*(upper-lower)))
    turn.stand.put(root/"TOUCH_READOUT.json",dict(thresholds=thresholds, source_trials=len(source),
                  rule="One trained scalar threshold per arrived projected side; unknown if no valid AN message.",
                  binding_sha256=sha(BINDING),graph_sha256=sha(GRAPH)))
    print("learned thresholds",thresholds,flush=True)


def classify(signal, model):
    if signal is None or not np.isfinite(signal).all():
        return -2
    scaled = signal/np.asarray(model["thresholds"])
    return int(scaled.argmax()) if scaled.max()>=1 else -1


def audit(root):
    model = turn.read(root/"TOUCH_READOUT.json")
    sources = [load_case(path) for path in sorted(root.glob("held_*"))]
    assert len(sources)==36
    records = []
    for truth, signals in sources:
        case = truth["case"]
        observed = [classify(row,model) for row in signals]
        event = case["expected_side"]
        onset = truth["onset_ms"][event] if event is not None else None
        assert truth["physical_case_matched"] and truth["unexpected_probe_force_W"] < TOUCH_W
        positives = [i for i,v in enumerate(observed) if v>=0]
        first = positives[0] if positives else None
        correct = bool(first is not None and observed[first]==event and onset is not None
                       and 0.<=first-onset<=60.)
        records.append(dict(case=case,ground_truth_ms=onset,decoded_ms=first,
                            decoded_side=observed[first] if first is not None else None, correct=correct,
                            latency_ms=first-onset if correct else None,
                            unknown_ticks=observed.count(-2)))
    positives = [r for r in records if r["ground_truth_ms"] is not None]
    blank = [r for r in records if r["ground_truth_ms"] is None]
    recall = [sum(r["correct"] for r in positives if r["case"]["expected_side"]==i)/12 for i in (0,1)]
    false = sum(r["decoded_ms"] is not None for r in blank)/len(blank)
    wrong = sum(r["decoded_ms"] is not None and r["decoded_side"]!=r["case"]["expected_side"] for r in positives)/len(positives)
    latencies = [r["latency_ms"] for r in positives if r["correct"]]
    p95 = float(np.percentile(latencies,95)) if latencies else None
    passed = min(recall)>=.90 and false<=.05 and wrong<=.05 and p95 is not None and p95<=60.
    result = dict(passed=passed,recall_by_side=recall,negative_false_alarm=false,wrong_side=wrong,
                  latency_p95_ms=p95,positive_events=len(positives),negative_cases=len(blank),cases=records)
    turn.stand.put(root/"TOUCH_READOUT_ACCEPTANCE.json",result)
    print({k:v for k,v in result.items() if k!="cases"},flush=True)
    return passed


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("mode",choices=("train","held","fit","audit"))
    p.add_argument("indices",nargs="*",type=int)
    a = p.parse_args()
    if a.mode=="fit":
        fit(a.root)
    elif a.mode=="audit":
        raise SystemExit(0 if audit(a.root) else 1)
    else:
        collect(a.root,a.mode,a.indices)
