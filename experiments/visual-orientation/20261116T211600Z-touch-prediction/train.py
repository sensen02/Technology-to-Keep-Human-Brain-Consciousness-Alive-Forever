"""Train and independently evaluate causal forecasts from arrived tactile data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from touch_runtime import BINDING, GRAPH, sha, turn
from sensory_corpus import GROUPS, LEGS, SAMPLE_MS

HORIZONS = (20, 50, 100, 200)
MODES = ("sense", "ordered", "shuffled", "mixed")
UPDATES = 260
STATE_HORIZON = 50
# Six angles, six angular speeds, six foot loads, body forward velocity,
# then six support bits, an upright projection, and a measured motion bit.


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = nn.GRU(4, 24, batch_first=True)
        self.events = nn.Linear(24, 8)
        self.band = nn.Linear(24, 16)
        self.body = nn.Linear(24, 27)

    def forward(self, observed):
        state, _ = self.recurrent(observed)
        shape = (*state.shape[:-1], len(HORIZONS), 2)
        lower, higher = self.band(state).sigmoid().reshape(*shape, 2).unbind(-1)
        return self.events(state).reshape(shape), torch.stack(
            (torch.minimum(lower, higher), torch.maximum(lower, higher)), -1), self.body(state)


def rows(root, split=None):
    records = []
    for group in range(GROUPS):
        if split == "train" and group>=12:
            continue
        for condition in ("clear", "fixed", "external", "sham"):
            folder = root / f"group_{group:02d}" / condition
            result = turn.read(folder / "RESULT.json")
            path = folder / "SAMPLES.npz"
            assert sha(path) == result["sha256"]
            with np.load(path, allow_pickle=False) as z:
                data = {key:z[key] for key in ("activity", "mixed", "valid", "commands", "state", "touch")}
            records.append({**result, **data, "path":str(path)})
    return records


def body_targets(state):
    speed = np.pad(state[:,18], (9,0), mode="edge")
    movement = np.median(np.lib.stride_tricks.sliding_window_view(speed,10),axis=1) >= .025
    return np.c_[state, movement]


def pack(records, mode, center, scale):
    length = records[0]["activity"].shape[0] - max(HORIZONS)//SAMPLE_MS
    observations, truth, levels, truth_masks, state, eligible = [], [], [], [], [], []
    for index, row in enumerate(records):
        # Same-sized ablations. Other family only, trajectory reassignment is offline.
        scrambled = next(records[(index+i)%len(records)] for i in range(1,len(records))
                         if records[(index+i)%len(records)]["case"]["group"] != row["case"]["group"])
        method = "mixed" if mode=="mixed" else "activity"
        signal = row[method]
        valid = row["valid"][:,int(mode=="mixed")]
        if mode == "sense":
            used_command = np.zeros(len(signal))
        elif mode == "shuffled":
            used_command = scrambled["commands"][:,1]
        else:
            used_command = row["commands"][:,1]
        observations.append(np.c_[signal[:length], valid[:length], used_command[:length]])
        horizons, signals, masks = [], [], []
        for span_ms in HORIZONS:
            span = span_ms//SAMPLE_MS
            windows = np.stack([row["touch"][offset:offset+length] for offset in range(1,span+1)])
            horizons.append(windows.any(0))
            # Forecast a true arriving tactile signal. Mixed candidate remains an input only.
            signals.append(row["activity"][span:span+length])
            masks.append(np.stack([row["valid"][offset:offset+length,0]
                                   for offset in range(1,span+1)]).all(0))
        truth.append(np.stack(horizons, 1))
        levels.append(np.stack(signals, 1))
        truth_masks.append(np.stack(masks, 1))
        offset = STATE_HORIZON//SAMPLE_MS
        state.append((body_targets(row["state"])[offset:offset+length]-center)/scale)
        eligible.append(~row["touch"][:length])
    return tuple(torch.as_tensor(np.array(values), dtype=torch.float32) for values in
                 (observations, truth, levels, truth_masks, state, eligible))


def fit(train, mode, out):
    physical = np.concatenate([body_targets(record["state"]) for record in train])
    center, scale = physical.mean(0), np.maximum(physical.std(0), .015)
    features, truth, levels, masks, state, _ = pack(train, mode, center, scale)
    torch.manual_seed(20261116)
    net = Predictor()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002)
    progress = []
    for update in range(1, UPDATES + 1):
        batch = torch.randint(len(features), (8,))
        logits, band, body = net(features[batch])
        actual, valid = truth[batch], masks[batch,:,:,None]
        event = nn.functional.binary_cross_entropy_with_logits(logits, actual, reduction="none")
        event = (event*valid).sum() / (valid.sum()*2)
        error = levels[batch].unsqueeze(-1)-band
        quantiles = torch.tensor([.1,.9])
        activity = (torch.maximum(quantiles*error, (quantiles-1)*error)*valid.unsqueeze(-1)).mean()
        body_loss = nn.functional.smooth_l1_loss(body, state[batch])
        loss = event + .50*activity + .12*body_loss
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),1., error_if_nonfinite=True)
        optimizer.step()
        if update in (1, 50, 100, 150, 200, UPDATES):
            progress.append(dict(update=update,total=loss.item(),event=event.item(),
                                 activity=activity.item(),body=body_loss.item()))
    output = out / f"{mode}.pt"
    tmp = output.with_suffix(".tmp")
    torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),
                    center=torch.tensor(center),scale=torch.tensor(scale), updates=UPDATES), tmp)
    tmp.rename(output)
    log = dict(mode=mode, train_curves=progress, weight_sha256=sha(output), rows=len(train),
               corpus=[dict(path=row["path"],sha256=row["sha256"]) for row in train])
    turn.stand.put(out/f"{mode}_TRAIN.json",log)
    print(mode, progress[-1], flush=True)


def average_precision(y, p):
    if not y.any():
        return None
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(ps)!=0), len(ys)-1]
    at = np.cumsum(ys)[last]
    return float(np.sum(np.diff(np.r_[0,at]) * at/(last+1))/at[-1])


def metrics(truth, probability, mask):
    truth, probability = truth[mask], probability[mask]
    return dict(brier=float(np.mean((truth-probability)**2)), average_precision=average_precision(truth,probability),
                positive=int(truth.sum()), points=int(len(truth)))


@torch.inference_mode()
def evaluate(records, model_path, mode):
    saved = torch.load(model_path, map_location="cpu", weights_only=True)
    brain = Predictor().eval()
    brain.load_state_dict(saved["model"])
    center, scale = saved["center"].numpy(), saved["scale"].numpy()
    inputs, truth, levels, valid, state, pre_contact = pack(records, mode, center, scale)
    logits, ranges, prediction = brain(inputs)
    output = logits.sigmoid().numpy()
    shape = truth.shape
    truth, valid = truth.numpy().astype(bool), valid.numpy().astype(bool)
    pre_contact = pre_contact.numpy().astype(bool)
    physical_prediction = prediction.numpy()*scale+center
    target_motion = (state.numpy()[...,26]*scale[26] + center[26]) > .5
    estimated_motion = physical_prediction[...,26] > .5
    error = np.abs(prediction.numpy()-state.numpy())*scale
    # Body-state comparison uses the training mean in the same held episodes.
    mean_error = np.abs(state.numpy())*scale
    views = []
    for sub in ("all", "nominal", "external"):
        selected = np.asarray([sub=="all" or (row["mode"]=="external")== (sub=="external")
                               for row in records])
        span = {}
        for h, ms in enumerate(HORIZONS):
            valid_h = np.broadcast_to(valid[:, :, h, None],shape[:2]+(2,)) & selected[:,None,None]
            span[str(ms)] = dict(all=metrics(truth[:,:,h], output[:,:,h], valid_h),
                     before_contact=metrics(truth[:,:,h],output[:,:,h], valid_h & pre_contact))
        views.append((sub,span))
    signal_band = ((levels.numpy() >= ranges[...,0].numpy()) & (levels.numpy() <= ranges[...,1].numpy()))
    zero_events = {str(ms):metrics(truth[:,:,h],np.zeros_like(truth[:,:,h],dtype=float),
                        np.broadcast_to(valid[:,:,h,None],shape[:2]+(2,)) & pre_contact)
                   for h,ms in enumerate(HORIZONS)}
    return dict(mode=mode, weights_sha256=sha(model_path), groups=sorted({r["case"]["group"] for r in records}),
                event=dict(views), zero_before_contact=zero_events,
                estimated_interval_coverage=float(signal_band.mean()),
                aux_mae=error.mean((0,1)).tolist(), train_mean_aux_mae=mean_error.mean((0,1)).tolist(),
                measured_motion=dict(accuracy=float((target_motion==estimated_motion).mean()),
                   balanced_accuracy=float(.5*((estimated_motion[target_motion]).mean()+
                                               (~estimated_motion[~target_motion]).mean())),
                   frames_moving=int(target_motion.sum()), frames_still=int((~target_motion).sum())))


def audit(records, out):
    groups = {"training": [r for r in records if r["case"]["split"]=="train"],
              "calibration":[r for r in records if r["case"]["split"]=="calibration"],
              "held":[r for r in records if r["case"]["split"]=="held"]}
    assert len(groups["held"]) == 16 and len(groups["calibration"]) == 16
    result = {name:{mode:evaluate(split, out/f"{mode}.pt", mode) for mode in MODES}
              for name, split in groups.items() if name!="training"}
    for name, split in groups.items():
        real = {condition:sum(any(r["touched_bins"]) for r in split if r["mode"]==condition)
                for condition in ("clear","fixed","external","sham")}
        print(name,real,flush=True)
    event = {mode:report["event"]["nominal"]["100"]["before_contact"]
             for mode,report in result["held"].items()}
    primary = event["ordered"]
    ap = primary["average_precision"]
    comparison = [event[mode] for mode in ("sense","shuffled")]
    result["registered_gate"] = dict(
        ap_gain_at_least_002=ap is not None and all(row["average_precision"] is not None and
                        ap >= row["average_precision"]+.02 for row in comparison),
        brier_within_0005=all(primary["brier"]<=row["brier"]+.005 for row in comparison))
    result["held_nominal_100ms_before_contact"] = event
    result["source"] = dict(binding_sha256=sha(BINDING), graph_sha256=sha(GRAPH), leg_order=LEGS,
                            horizon_ms=HORIZONS, body_horizon_ms=STATE_HORIZON, steps=UPDATES,
                            model_input="arriving AN activity, valid flag, copy of currently issued DN request",
                            engineering_mix="same two identified scalar AN ports with .70 lateral touch + .30 normalized front load",
                            no_an_cut_trajectories_in_primary_corpus=True)
    turn.stand.put(out/"EVALUATION.json",result)
    print(json.dumps(dict(held_100ms=event, gates=result["registered_gate"])),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("mode",choices=(*MODES,"audit"))
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.mode=="audit":
        audit(rows(args.data),args.output)
    else:
        selected = [record for record in rows(args.data, "train") if record["mode"]!="external"]
        fit(selected,args.mode,args.output)
