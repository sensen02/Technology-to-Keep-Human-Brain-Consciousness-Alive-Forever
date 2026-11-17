"""Train a paired fixed-replay SAC experiment, changing dense reward only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/"20261117T022700Z-sac-startup-replay"
sys.path.insert(0,str(SOURCE))
from course import sha
from model import SACBrain
from sac_replay import Replay
from train_sac import SAC


def load(path):
    return json.loads(path.read_text())


def pair(warm, report, plan, out):
    assert report["both_demos_qualified"]
    seed, updates, anchor = plan["seed"], plan["updates_per_arm"], plan["anchor"]
    scores = report["comparisons"]
    paths = [Path(row["source"]) for row in scores]
    models = (SAC(warm,seed,anchor), SAC(warm,seed,anchor))
    replays = tuple(Replay(paths, agent.rng) for agent in models)
    assert replays[0].manifest == replays[1].manifest
    for i, (entry, pin) in enumerate(zip(scores,replays[0].manifest,strict=True)):
        assert entry["sha256"]==pin["sha256"]
        score_path = HERE/"SCORES"/f"{pin['case']}.npz"
        assert sha(score_path)==entry["score_sha256"]
        is_demo = pin["policy"]=="detour"
        assert not is_demo or (pin["ending"]=="success" and not pin["truncated"])
        with np.load(score_path,allow_pickle=False) as scored:
            assert scored["episode_sha256"]==pin["sha256"]
            reward = scored["reward"]
        assert reward.shape == replays[1].rows[i]["reward"].shape
        replays[1].rows[i]["reward"] = reward
        replays[1].manifest[i] = pin | dict(reward_file=str(score_path),reward_sha256=entry["score_sha256"])
    out.mkdir()
    base, changed = (out/"original",out/"route")
    for path in (base,changed):
        path.mkdir()
    trace = out/"UPDATES.jsonl"
    last_checkpoints = ()
    with trace.open("x") as diary:
        for step in range(1,updates+1):
            batches = [replay.sample(12,early=step%3==1) for replay in replays]
            assert batches[0].keys()==batches[1].keys()
            for name,tensor in batches[0].items():
                if name!="reward":
                    assert torch.equal(tensor,batches[1][name]), name
            # Each arm draws the same critic and actor noise regardless of
            # score-dependent parameter changes.
            rng = torch.get_rng_state()
            original = models[0].step(batches[0])
            after = torch.get_rng_state()
            torch.set_rng_state(rng)
            modified = models[1].step(batches[1])
            assert torch.equal(torch.get_rng_state(),after)
            row = dict(step=step,original=original,route=modified)
            if step%128==0 or step==updates:
                last_checkpoints=tuple(path/f"{step:04d}.pt" for path in (base,changed))
                for saved, agent, replay in zip(last_checkpoints,models,replays,strict=True):
                    agent.save(saved,replay,warm)
                row["checkpoint"]={str(p):sha(p) for p in last_checkpoints}
            diary.write(json.dumps(row)+"\n")
            if step%128==0 or step==updates:
                diary.flush()
                print(step,original["critic"],modified["critic"],flush=True)
    return last_checkpoints


def teacher_frame_diagnostics(checkpoint, roots, warm):
    saved = torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert saved["source"]["sha256"] == sha(warm)
    count = saved["sac_updates"]
    for name in ("policy_optimizer","critic_optimizer","alpha_optimizer"):
        assert all(int(row["step"].item()) == count for row in saved[name]["state"].values())
    model = SACBrain()
    model.load_state_dict(saved["model"])
    replay = Replay(roots,np.random.default_rng(12))
    report = {}
    for row,root in zip(replay.rows,roots,strict=True):
        with np.load(root/"REFERENCE.npz",allow_pickle=False) as z:
            phase = z["phase"][1:]
        x = torch.from_numpy(row["x"])[None]
        vision = torch.from_numpy(row["vision"])[None]
        prefix = torch.from_numpy(row["prefix"]).reshape(1,-1,7)
        with torch.inference_mode():
            _,memory = model.brain.state(prefix)
            state,_ = model.sequence(x,vision,memory)
            logs,_,_ = model.distribution(state[0,:-1])
        probabilities = logs.exp().numpy()
        prediction = probabilities.argmax(axis=1)
        agreed = prediction == row["mode"]
        metrics = {}
        labels = dict(start_turn=(np.arange(len(phase))<50)&np.isin(row["mode"],[2,3]),
                      rejoin_forward=(phase=="rejoin")&(row["mode"]==1),
                      goal_hold=(phase=="goal_hold")&(row["mode"]==0))
        for key,mask in labels.items():
            assert mask.any(), key
            real = row["mode"][mask]
            p = probabilities[mask,real]
            metrics[key] = dict(ticks=int(mask.sum()),mode_accuracy=float(agreed[mask].mean()),
                                mean_probability=float(p.mean()))
        report[root.name] = metrics
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("warm",type=Path)
    cli.add_argument("out",type=Path)
    args = cli.parse_args()
    torch.set_num_threads(1)
    report = load(HERE/"COMPARISON.json")
    plan = load(HERE/"SCORING_PLAN.json")["offline_sac"]
    original, route = pair(args.warm.resolve(),report,plan,args.out.resolve())
    by_case = {row["case"]:Path(row["source"]) for row in report["comparisons"]}
    roots = [by_case[key] for key in ("cal_left","cal_right")]
    output = dict(updates=plan["updates_per_arm"],same_samples_and_noise_verified=True,
                  original=teacher_frame_diagnostics(original,roots,args.warm),
                  route=teacher_frame_diagnostics(route,roots,args.warm),
                  comparison_sha256=sha(HERE/"COMPARISON.json"),code_sha256=sha(Path(__file__)),
                  checkpoint={"original":str(original),"route":str(route)})
    (args.out/"PAIRED.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2),flush=True)


if __name__ == "__main__":
    main()
