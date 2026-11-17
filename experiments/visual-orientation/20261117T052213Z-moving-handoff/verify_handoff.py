"""Independently replay choices and scoped DN/AN delivery for captured suffixes."""
import numpy as np
import torch

from handoff_replay import HERE, PAIRED, course, put, read
from handoff_snapshot import read_checkpoint
from model import SACBrain
from handoff_suffix import restored_memory


def verify_delivery(journal, snapshot, z):
    start = snapshot["link"]["step"]
    n = len(z["mode"])
    end = start+20*n
    ticks = start+np.arange(n*4)*5
    transport = course.Neck()
    receipt = dict()
    for direction,interval,roots in (("DN",5,transport.ids),("AN",1,transport.an_roots)):
        sent = [event for event in journal if event["event"]=="enqueue" and event["direction"]==direction]
        arrived = [event for event in journal if event["event"]=="release" and event["direction"]==direction]
        carry = [item for item in snapshot["link"]["_queue"] if item["direction"]==direction]
        assert [row["capture_step"] for row in sent]==list(range(start,end,interval))
        accepted = [row for row in carry+sent if row["release_step"]<=end]
        for request,reply in zip(accepted,arrived,strict=True):
            assert request["root_ids"]==reply["root_ids"]==roots
            assert request["values"]==reply["values"]
            assert request["capture_step"]==reply["capture_step"]
            assert request["release_step"]==reply["consume_step"]==reply["capture_step"]+2
            assert reply["source_bindings_sha256"]==transport.link.digest
            assert reply["source_part"]=="neck"
            assert reply["target_part"]==("other" if direction=="DN" else "brain")
        due = np.array([r["release_step"] for r in arrived])
        take = np.searchsorted(due,ticks,side="right")-1
        entries = np.array([r["values"] for r in arrived])
        if direction=="DN":
            known = np.copy(z["delivered"]).reshape(-1,2)
            decoded = np.empty_like(known)
            decoded[take<0] = snapshot["neck"]["decoded"]
            choices = {tuple(a):transport.codec.read(list(a))[0] for a in np.unique(entries[:,:2],axis=0)}
            new = entries[take[take>=0]]
            decoded[take>=0] = np.c_[.1*new[:,2],[choices[tuple(a)] for a in new[:,:2]]]
            np.testing.assert_allclose(decoded,known,rtol=0.,atol=1e-13)
            encoded = np.asarray([row["values"] for row in sent])
            if tuple(encoded[-1,:2]) not in choices:
                choices[tuple(encoded[-1,:2])] = transport.codec.read(encoded[-1,:2].tolist())[0]
            expected=np.c_[.1*encoded[:,2],[choices[tuple(a)] for a in encoded[:,:2]]]
            np.testing.assert_allclose(expected,z["gated"].reshape(-1,2),rtol=0.,atol=1e-13)
        else:
            projection = entries @ transport.matrix.T
            grouped = np.column_stack([projection[:,idx].mean(1) for idx in transport.pool])
            physical = np.empty((len(take),4))
            physical[take<0] = snapshot["neck"]["an_values"]
            physical[take>=0] = grouped[take[take>=0]]
            np.testing.assert_allclose(physical,z["x"][1:,:,1:5].reshape(-1,4),rtol=0.,atol=7e-8)
            assert np.all(z["x"][1:,:,5]==1.)
        receipt[direction] = dict(enqueued=len(sent),released=len(arrived),initial_pending=len(carry))
    assert len(journal)==sum(v["enqueued"]+v["released"] for v in receipt.values())
    return receipt


@torch.inference_mode()
def verify_actor(pin,history,z):
    agent = SACBrain().eval()
    agent.load_state_dict(torch.load(pin,map_location="cpu",weights_only=True)["model"])
    initial = restored_memory(agent,history)
    np.testing.assert_array_equal(initial.numpy(),z["initial_memory"])
    # The first choice reads the inherited hidden state. Each next one reads
    # the post-transition GRU state and the eye frame that has actually arrived.
    remainder = torch.from_numpy(z["x"][1:]).reshape(1,-1,7)
    encoded,_ = agent.brain.state(remainder,initial)
    at_decision = torch.cat((initial[0],encoded[0,3::4]),axis=0)[:-1]
    actor = torch.cat((at_decision,torch.from_numpy(z["vision"][:-1])),axis=-1)
    logs,mu,_ = agent.distribution(actor)
    prediction = logs.argmax(-1).numpy()
    rate = ((mu.tanh()+1.)*.5).numpy()
    rate[prediction!=1]=.5
    np.testing.assert_array_equal(prediction,z["mode"])
    np.testing.assert_allclose(rate,z["pace"],rtol=0.,atol=2.e-6)
    return len(prediction)


def main():
    torch.set_num_threads(1)
    plan = read(HERE/"PLAN.json")
    parent = read(HERE/"RESULTS.json")
    snapshot = read_checkpoint(HERE/"MOVING_CHECKPOINT.pkl",parent["checkpoint"]["snapshot_sha256"])
    history=np.load(HERE/"BRAIN_HISTORY.npy",allow_pickle=False)
    assert course.sha(HERE/"BRAIN_HISTORY.npy")==parent["checkpoint"]["history_sha256"]
    controls = {}
    all_results = {}
    for r in parent["tests"]:
        name=r["arm"]
        root=HERE/name
        assert course.sha(root/"SUFFIX.npz")==r["episode_sha256"]
        assert course.sha(root/"TRANSPORT.json")==r["transport_sha256"]
        assert r["source"]["actual_steps"]==r["first_abs_ms"]+round(1000*r["duration_s"])
        with np.load(root/"SUFFIX.npz",allow_pickle=False) as f:
            z={k:f[k] for k in f.files}
        np.testing.assert_array_equal(z["qpos"][0],snapshot["native_data"].qpos)
        np.testing.assert_array_equal(z["eyes"][0],snapshot["course"]["rgb"])
        np.testing.assert_array_equal(z["eyes"][1],snapshot["eyes"]["pending"][1])
        np.testing.assert_array_equal(z["x"][0],snapshot["course"]["past"])
        before=z["delivered"][0,0]
        np.testing.assert_array_equal(before,snapshot["neck"]["decoded"])
        releases = verify_delivery(read(root/"TRANSPORT.json"),snapshot,z)
        forward_s=.005*int(np.count_nonzero(z["delivered"].reshape(-1,2)[1:,0]>0.))
        radius=np.linalg.norm(z["com"][:,:2]-z["goal"][:2],axis=1)
        assert forward_s==r["autonomous_dn_forward_s"]
        assert float(radius[0]-radius[-1])==r["com_target_net_progress"]
        assert len(z["mode"])*1600==r["native_steps"]
        decision_checks = None
        if name in plan["models"]:
            pin=PAIRED/plan["models"][name]
            assert course.sha(pin)==r["policy"]["sha256"]
            decision_checks=verify_actor(pin,history,z)
        elif name=="neutral":
            assert not z["mode"].any() and not forward_s
            controls["neutral"]=True
        elif name=="teacher":
            assert r["pass_all"] and z["reason"][-1]=="success"
            controls["teacher"]=True
        all_results[name]=dict(delivery=releases,all_match_same_moving_start=True,
                               decision_checks=decision_checks,final_reason=r["ending"],
                               visual_boundary_exact=True)
        print(name,r["ending"],forward_s,r["com_target_net_progress"],decision_checks,flush=True)
    assert controls==dict(teacher=True,neutral=True)
    result=dict(pass_controls=True,progress_and_actor_verified=True,
                checkpoint_sha256=parent["checkpoint"]["snapshot_sha256"],
                per_arm=all_results)
    put(HERE/"INDEPENDENT.json",result)


if __name__=="__main__":
    main()
