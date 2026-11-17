"""Audit event identity and reproduce one held arrival, including independent AN cut."""
import argparse
from pathlib import Path

import numpy as np

from touch_runtime import W, TouchLink, turn
from sensory_corpus import SAMPLE_MS, TOUCH_W
from train import rows


def verify(root, out):
    data = rows(root)
    counts = {}
    examples = []
    for record in data:
        native = Path(record["path"])
        with np.load(native, allow_pickle=False) as z:
            condition = record["mode"]
            commands, state, activity, mixed, valid, contacts = (z[key] for key in
                ("commands","state","activity","mixed","valid","touch"))
            assert np.isfinite(state).all() and (state[:,12:18]>=0).all()
            np.testing.assert_array_equal(commands[:,0], np.arange(len(state))*SAMPLE_MS)
            assert np.all(commands[1:,3] <= commands[1:,0]) and np.all(valid[1:])
            np.testing.assert_array_equal(state[:,19:25]!=0,state[:,12:18]>=.02)
            if condition in ("clear","sham"):
                assert not contacts.any(), record["path"]
            tip_side = record["case"]["side"]
            wrong_side = bool(contacts[:,1-tip_side].any())
            nonfront = record["nonfront_peak"] >= TOUCH_W
            assert not wrong_side
            key = f"{record['case']['split']}_{condition}"
            row = counts.setdefault(key, dict(episodes=0, touched=0, nonfront=0))
            row["episodes"] += 1
            row["touched"] += int(contacts.any())
            row["nonfront"] += int(nonfront)
            early = (record["case"]["start"]+100)//SAMPLE_MS
            if condition=="fixed":
                assert not contacts[:early].any(), record["path"]
            if record["case"]["split"]=="held" and condition=="fixed" and contacts.any():
                examples.append(record)

    with np.load(examples[0]["path"], allow_pickle=False) as z:
        native, arrived, contacts = z["native"], z["activity"], z["touch"]
    first = int(np.flatnonzero(contacts.any(1))[0]*SAMPLE_MS)
    cut_begins, restores = first-15, first+70
    reference, interrupted = TouchLink(),TouchLink()
    base, cut = [], []
    for step in range(len(arrived)*SAMPLE_MS):
        if step == cut_begins:
            interrupted.cut_an(True)
        if step == restores:
            interrupted.cut_an(False)
        if step % SAMPLE_MS == 0:
            base.append(reference.observed())
            cut.append(interrupted.observed())
        if step:
            force = native[(step-1)*80:step*80,:2].mean(0)/W
            activity = np.clip(force/.02,0.,1.).tolist()
        else:
            activity = [0.,0.]
        reference.sense(activity)
        interrupted.sense(activity)
        reference.advance()
        interrupted.advance()
    base_values = np.asarray([np.zeros(2) if item is None else item for item in base])
    np.testing.assert_array_equal(base_values, arrived)
    assert all(cut[i] is None for i in range((cut_begins+4)//5,(restores+4)//5))
    first_restored = next(
        i*SAMPLE_MS for i in range((restores+4)//5,len(cut)) if cut[i] is not None)
    for i in range(first_restored//SAMPLE_MS+1,len(cut)):
        np.testing.assert_array_equal(base[i],cut[i])
    result = dict(episodes=len(data), native_steps=sum(r["source_native_steps"] for r in data),
                  condition_counts=counts, independent_link_replay=dict(
                      path=examples[0]["path"], exact_arrived=True, cut_ms=cut_begins, restored_ms=restores,
                      first_valid_after_restore_ms=first_restored, runtime_cut_returns_missing=True),
                  field_order="6 tibia angles, 6 angular velocities, 6 loads, COM forward velocity, 6 support bits, upright",
                  blind_group_ids=sorted({r["case"]["group"] for r in data if r["case"]["split"]=="held"}))
    turn.stand.put(out, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("out",type=Path)
    args = parser.parse_args()
    verify(args.data,args.out)
