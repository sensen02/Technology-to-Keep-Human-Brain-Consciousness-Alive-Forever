"""Independently verify physical routing, action consumption and saved updates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from course import HERE, Neck, sha
from model import SACBrain
from sac_replay import Replay
from train_sac import SAC


def packet_rows(root):
    result=json.loads((root/"RESULT.json").read_text())
    assert all(sha(root/name)==digest for name,digest in result["files"].items())
    with np.load(root/"EPISODE.npz",allow_pickle=False) as z:
        rows={k:z[k] for k in ("delivered","gated","x","requested","contact_W","terminal","truncated",
                               "qpos","vision","reward","terms","term_names")}
    journal=json.loads((root/"TRANSPORT.json").read_text())
    return result,rows,journal


def verify(root):
    result,z,journal=packet_rows(root)
    n=result["actor_ticks"]
    assert z["delivered"].shape==(n,4,2) and len(z["qpos"])==n+1
    assert result["source"]["actual_steps"]==n*20+500
    assert np.count_nonzero(z["terminal"]|z["truncated"])==1
    assert bool(z["terminal"][-1])==result["terminated"]
    assert bool(z["truncated"][-1])==result["truncated"]
    assert float(z["contact_W"].max())==result["contact_peak_W"]
    if result["reason"]=="obstacle_contact":
        assert z["contact_W"][-1]>1.e-6 and not (z["contact_W"][:-1]>1.e-6).any()
    dense=z["terms"][:,z["term_names"].tolist().index("progress")]
    assert result["source"]["source_files_unchanged"] and not result["source"]["physiology_verified"]
    neck=Neck()
    dn_cut=any(row["event"]=="cut" and row["direction"]=="DN" for row in journal)
    counts={}
    for direction,interval,roots in (("DN",5,neck.ids),("AN",1,neck.an_roots)):
        sent=[r for r in journal if r["direction"]==direction and r["event"]=="enqueue"]
        arrived=[r for r in journal if r["direction"]==direction and r["event"]=="release"]
        stop=result["source"]["actual_steps"]
        expected=[] if dn_cut and direction=="DN" else list(range(0,stop,interval))
        assert [r["capture_step"] for r in sent]==expected
        for request,reply in zip([s for s in sent if s["capture_step"]+2<=stop],arrived,strict=True):
            assert request["root_ids"]==reply["root_ids"]==roots
            assert request["values"]==reply["values"] and request["release_step"]==reply["release_step"]
            assert reply["consume_step"]==reply["capture_step"]+2
            assert reply["source_part"]=="neck"
            assert reply["target_part"]==("brain" if direction=="AN" else "other")
            assert reply["source_bindings_sha256"]==result["source"]["binding_sha256"]
        due=[e["release_step"] for e in arrived]
        ticks=500+np.arange(n*4)*5
        if direction=="DN":
            if dn_cut:
                assert not z["delivered"].any()
            else:
                entries=np.asarray([r["values"] for r in arrived])
                take=np.searchsorted(due,ticks,side="right")-1
                v=entries[take]
                angle={tuple(a):neck.codec.read(list(a))[0] for a in np.unique(v[:,:2],axis=0)}
                decoded=np.c_[.1*v[:,2],[angle[tuple(a)] for a in v[:,:2]]]
                np.testing.assert_allclose(decoded,z["delivered"].reshape(-1,2),rtol=0.,atol=1e-13)
                gate=np.asarray([e["values"] for e in sent])[ticks//5]
                gates=np.c_[.1*gate[:,2],[angle[tuple(a)] if tuple(a) in angle
                    else neck.codec.read(list(a))[0] for a in gate[:,:2]]]
                np.testing.assert_allclose(gates,z["gated"].reshape(-1,2),rtol=0.,atol=1e-13)
        else:
            take=np.searchsorted(due,ticks,side="right")-1
            activity=np.asarray([r["values"] for r in arrived])[take]
            network=activity@neck.matrix.T
            projection=np.column_stack([network[:,i].mean(1) for i in neck.pool])
            np.testing.assert_allclose(projection,z["x"][1:,:,1:5].reshape(-1,4),rtol=0.,atol=7e-8)
            assert np.all(z["x"][1:,:,5]==1.)
        counts[direction]=len(arrived)
    return (result|{"delivered_packets":counts,"dn_cut_observed":dn_cut,
                    "requested_motion_ticks":int(np.count_nonzero(z["requested"].any(1))),
                    "gated_motion_ticks":int(np.count_nonzero(z["gated"].any((1,2)))),
                    "positive_progress_ticks":int(np.count_nonzero(dense>0.)),
                    "total_progress_reward":float(dense.sum()),
                    "physical_forward_ticks":int(np.count_nonzero(z["delivered"][:,0,0]>0)),
                    "physical_turn_ticks":int(np.count_nonzero(z["delivered"][:,0,1])),
                    "moving_without_target_visible":int(np.count_nonzero((z["delivered"][:,0,0]>0)&(z["vision"][:-1,2]==0)))},z)


def weights(path,first,journal):
    original=torch.load(first,map_location="cpu",weights_only=True)["model"]
    saved=torch.load(path,map_location="cpu",weights_only=True)
    model=SACBrain()
    model.load_state_dict(saved["model"])
    updates=[json.loads(line) for line in journal.read_text().splitlines()]
    assert updates[-1]["update"]==saved["sac_updates"]
    assert any(row["terminals"] for row in updates) and any(row["truncations"] for row in updates)
    assert all(row["grad_shared"]>0 for row in updates)
    changes={}
    for prefix in ("actor","brain.state","brain.auxiliary","brain.future_body","intent"):
        base=[(old-saved["model"][key]).square().sum().item() for key,old in original.items()
              if key.startswith(prefix)]
        changes[prefix]=float(sum(base)**.5)
        assert changes[prefix]>0.
    for prefix in ("brain.vision","brain.visual_head","brain.range_vision","brain.action","brain.tactile_adjustment"):
        assert all(torch.equal(old,saved["model"][key]) for key,old in original.items() if key.startswith(prefix))
    for name in ("policy_optimizer","critic_optimizer","alpha_optimizer"):
        state=saved[name]["state"]
        assert all(int(row["step"].item())==saved["sac_updates"] for row in state.values())
    # Independent mathematical admission test. Actual replay contains both
    # terminal rewards and collector truncations with valid successors.
    agent=SAC(path,441,saved["anchor_weight"])
    replay=Replay([Path(row["path"]).parent for row in saved["replay"]],np.random.default_rng(311))
    for _ in range(50):
        batch=replay.sample(18)
        term=batch["terminal"]
        trunc=batch["truncated"]
        if term.any() and trunc.any():
            with torch.no_grad():
                s,_=agent.model.sequence(batch["x"],batch["vision"])
                v=agent.target_value(s[:,51:],agent.alpha_log.detach().exp())
                r=batch["reward"].float()
                target=r+(~term)*.99955*v
                assert torch.equal(r[term],target[term])
                assert (target[trunc]!=r[trunc]).any()
            break
    else:
        raise AssertionError("Could not draw both episode endings")
    return dict(path=str(path),sha256=sha(path),sac_updates=saved["sac_updates"],replay=saved["replay"],
                parameter_delta=changes,ending_targets_verified=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("checkpoint",type=Path)
    cli.add_argument("initial",type=Path)
    cli.add_argument("journal",type=Path)
    cli.add_argument("rollouts",type=Path,nargs="+")
    a=cli.parse_args()
    torch.set_num_threads(1)
    results={}
    for root in a.rollouts:
        r,z=verify(root)
        results[str(root)]=r
        print(root,r["reason"],r["actor_ticks"],r["delivered_packets"],flush=True)
    result=dict(saved=weights(a.checkpoint,a.initial,a.journal),episodes=results)
    (HERE/"INDEPENDENT.json").write_text(json.dumps(result,indent=2)+"\n")
