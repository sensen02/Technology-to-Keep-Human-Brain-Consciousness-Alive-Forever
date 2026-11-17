"""Check the stored SAC supervision and score without training or simulation."""
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
VISUAL = HERE.parent
RUN = VISUAL / "20261117T022700Z-sac-startup-replay"
REJOIN = VISUAL / "20261116T203100Z-rejoin"
sys.path.insert(0, str(RUN))
from course import ACTOR_S, FAST, GAMMA, Teacher, action_vector, command_code, sha
from reward import Measurement, Motion, Reward, Task
from sac_replay import read_episode


def load(path):
    return json.loads(path.read_text())


def position_boundary(mode, phases, start, length):
    assert start + length <= len(mode)
    first = mode[start]
    assert first in phases and np.all(mode[start:start+length] == first)
    return start + length


def summary(z, row, receipt, start, end, names):
    terms = z["terms"][start:end]
    rewards = z["reward"][start:end]
    mode = row["mode"][start:end]
    delivered = z["delivered"][start:end]
    start_range = receipt["initial_range"] if start == 0 else z["distance"][start-1]
    com = z["com"]
    return dict(tick_range=[start, end], actor_seconds=[start*ACTOR_S, end*ACTOR_S],
                head_range=[float(start_range), float(z["distance"][end-1])],
                com_shift=float(np.linalg.norm(com[end-1,:2] - com[max(start-1,0),:2])),
                com_goal_range=[float(np.linalg.norm(com[i,:2] - z["goal"][:2]))
                                for i in (max(start-1,0), end-1)],
                labeled_modes=dict(zip(("stop", "forward", "left", "right"),
                                       np.bincount(mode[row["demo"][start:end] > 0], minlength=4).tolist())),
                delivered_turn_ticks=int(np.count_nonzero(np.any(delivered[...,1] != 0.,axis=1))),
                delivered_forward_ticks=int(np.count_nonzero(np.any(delivered[...,0] > 0.,axis=1))),
                red_visible_ticks=int(z["vision"][start:end,2].sum()),
                reward=float(rewards.sum()),
                discounted_from_phase_start=float(rewards @ GAMMA**np.arange(len(rewards))),
                terms=dict(zip(names, terms.sum(axis=0).tolist())),
                positive_reward_ticks=int((rewards > 0).sum()),
                novel_distance_ticks=int((terms[:,names.index("progress")] > 0.).sum()))


def inspect_episode(pin):
    path = Path(pin["path"])
    row, actual_pin = read_episode(path.parent)
    assert actual_pin == pin
    receipt = load(path.parent / "RESULT.json")
    n = len(row["mode"])
    result = dict(input=pin, qualified_terminal_action_demo=bool(
        receipt["reason"] == "success" and row["terminal"][-1] and not row["truncated"][-1]),
        historical_demo_ticks=int(row["demo"].sum()))
    if pin["policy"] != "detour":
        return result

    with np.load(path, allow_pickle=False) as z:
        names = z["term_names"].tolist()
        np.testing.assert_allclose(z["reward"], z["terms"].sum(axis=1), rtol=0., atol=2e-14)
        mode = row["mode"]
        teacher = Teacher("detour", receipt["plan"]).detour
        pivot_first = int(np.flatnonzero(np.isin(mode, [2,3]))[0])
        pivot_end = position_boundary(mode, (2,3), pivot_first, teacher.turn_frames)
        walk_first = position_boundary(mode, (0,), pivot_end, 65)
        walk_end = position_boundary(mode, (1,), walk_first, teacher.walk_frames)
        rejoin_first = position_boundary(mode, (0,), walk_end, 60)
        assert np.all(z["pace"][walk_first:walk_end] == 1.)
        assert np.all(z["requested"][walk_first:walk_end,0] == FAST)
        assert not np.any(z["contact_W"])
        phases = [("initial_turn",pivot_first,pivot_end), ("settle_after_pivot",pivot_end,walk_first),
                  ("oblique_walk",walk_first,walk_end), ("settle_after_walk",walk_end,rejoin_first),
                  ("rejoin_until_cut",rejoin_first,n), ("final_5_s",n-250,n)]
        result.update(actual_diagonal_request=float(FAST),
                      phase_results={name:summary(z,row,receipt,start,end,names)
                                     for name,start,end in phases},
                      closest_head_distance_at_rejoin=float(np.min(z["distance"][:rejoin_first])),
                      closest_head_distance_at_cut=float(np.min(z["distance"])),
                      final_tail_gap_to_far=load(RUN/"FAILURE_DIAG_20261117T030647Z.json")[
                          pin["case"].replace("cal_", "teacher_")]["final_tail_gap_to_far"])
    return result


def toy_reward(distance, before_best, approach_speed, seconds):
    """Score fixed hypothetical progress under the exact recorded formula."""
    score = Reward(Task(5.8,(1.18,1.72),True,require_motion=True))
    score.best_goal_error = before_best-1.72
    start = distance
    values = []
    terms = []
    tags = []
    for _ in range(round(seconds / ACTOR_S)):
        end = distance - approach_speed*ACTOR_S
        shift = start-end
        m = Measurement(end,0.,abs(approach_speed),1.,1.3,1.,6,
                        neutral_command=approach_speed==0.,motion=Motion((shift,0.),0.))
        step = score.step(distance,m)
        values.append(step["reward"])
        terms.append(step["terms"])
        distance = end
        if step["terminated"]:
            tags.append(step["reason"])
            break
    return dict(seconds=len(values)*ACTOR_S,start_range=start,end_range=distance,
                reward=sum(values),mean_per_tick=float(np.mean(values)),
                discounted=float(values @ GAMMA**np.arange(len(values))),ending=tags or None,
                terms={k:sum(v[k] for v in terms) for k in terms[0]})


def main():
    files = [RUN/"TRAIN/REPLAY.json", RUN/"WARM_BOX.pt", RUN/"TRAIN/0320.pt",
             REJOIN/"wider/RESULT.json", REJOIN/"old/RESULT.json", REJOIN/"PAIR_B_SHORT.json",
             Path(inspect.getfile(Teacher)), Path(inspect.getfile(Teacher("detour",{}).detour.__class__)),
             Path(inspect.getfile(Teacher("detour",{}).detour.__class__.__mro__[1])),
             Path(inspect.getfile(Teacher("detour",{}).detour.__class__.__mro__[2])),
             Path(inspect.getfile(Reward)),Path(inspect.getfile(read_episode))]
    manifest = load(files[0])
    warm = torch.load(files[1], map_location="cpu", weights_only=True)
    sac = torch.load(files[2], map_location="cpu", weights_only=True)
    assert sac["replay"] == manifest and sac["source"]["sha256"] == sha(files[1])
    assert {((RUN/p["path"]).resolve(),p["sha256"]) for p in warm["source"] if p["policy"] == "detour"} == {
        (Path(p["path"]),p["sha256"]) for p in manifest if p["policy"] == "detour"}
    assert all(str(REJOIN) not in p["path"] for p in manifest)
    warm_parent = Path(warm["parent"]["path"])
    assert sha(warm_parent) == warm["parent"]["sha256"]
    parent = torch.load(warm_parent, map_location="cpu", weights_only=True)
    assert [Path(p["path"]).parent.name for p in parent["source"]] == ["straight", "left", "right", "pace"]
    assert all(sha(Path(p["path"])) == p["sha256"] for p in parent["source"])
    files.extend([warm_parent,RUN/"FAILURE_DIAG_20261117T030647Z.json"])
    restored = load(files[3])
    stopped = load(files[4])
    assert restored["full_body_beyond_box"] > 0 and restored["checks_rejoin"]["rejoined"]
    assert stopped["forward_ms"] == 0 and load(files[5])["passed"]
    assert action_vector(*command_code((.0955,0.)))[0] == .08318
    results = [inspect_episode(pin) for pin in manifest]
    comparison = {
        "wait_5_s": toy_reward(4.4,4.4,0.,5),
        "wait_until_stall": toy_reward(4.4,4.4,0.,13),
        "advance_fresh_bests_5_s": toy_reward(4.4,4.4,.08,5),
        "advance_recover_old_bests_5_s": toy_reward(4.4,3.9,.08,5),
    }
    assert comparison["wait_until_stall"]["ending"] == ["stalled_idle"]
    for example in results:
        if example["input"]["policy"] == "detour":
            assert example["historical_demo_ticks"] == 2400
            assert not example["qualified_terminal_action_demo"]
            assert example["phase_results"]["rejoin_until_cut"]["delivered_forward_ticks"] == 0
    report = dict(replay_update=sac["sac_updates"],sac_anchor=sac["anchor_weight"],
                  warm_actions=warm["mode_counts"],warm_sources=warm["source"],warm_parent_sources=parent["source"],
                  old_short={k:stopped[k] for k in ("actual_ms","forward_ms","body_displacement")},
                  wider={k:restored[k] for k in ("actual_ms","impact_count_native","full_body_beyond_box",
                                                  "initial_range","final_range","checks_rejoin")},
                  stage_speed_from_historical_teacher=.0955,
                  stage_speed_in_current_replay=FAST,
                  episodes=results, fixed_kinematics_score_checks=comparison, physical_steps=0,
                  optimizer_updates=0, inputs={str(path):sha(path) for path in files})
    (HERE/"AUDIT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(dict(episode_rewards={r["input"]["case"]:{k:round(w["reward"],5)
                 for k,w in r.get("phase_results",{}).items()}
                 for r in results if "phase_results" in r},
                 comparisons={k:round(v["reward"],5) for k,v in comparison.items()}),indent=2))


if __name__ == "__main__":
    main()
