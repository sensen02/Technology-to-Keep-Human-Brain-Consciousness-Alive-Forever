"""Identify observation dropout and exercise the candidate correction on saved images."""
import argparse
from pathlib import Path

import numpy as np
import torch

from approach_runtime import turn, sha
from visual_rejoin import VisualRejoin, marker_bearing


def audit(source):
    torch.set_num_threads(1)
    result = turn.read(source/"RESULT.json")
    with np.load(source/"OBS.npz") as data:
        decisions = data["decisions"]
        stamps, images = data["stamps"], data["eyes"]
    longitudinal = decisions[:,3]
    rotations = decisions[:,4]
    first_yaw = int(decisions[rotations>0,0][0])
    settle_from = first_yaw+20*(result["spec"]["turn_frames"] + 125 + result["spec"]["walk_frames"])-10
    last_cmd = int(decisions[np.any(abs(decisions[:,3:5])>1e-10,axis=1),0][-1])
    stream = []
    for t, image in zip(stamps,images):
        s = int((t+20)//5)
        if not settle_from <= t+20 or len(decisions) <= s:
            continue
        old = decisions[s]
        bearing = marker_bearing(image)
        stream.append((int(t+20), float(old[5]), float(old[6]), float(old[7]), bearing))
    terminal = [row for row in stream if row[0] > last_cmd+1000]
    still_image = images[-2]
    test = VisualRejoin()
    actions = [test.act(still_image)[0] for _ in range(55)]
    blank = images[0]
    if marker_bearing(blank) is not None:
        raise AssertionError("Blank reference frame has a red marker")
    scanning = VisualRejoin()
    lost = [scanning.act(blank)[0] for _ in range(300)]
    assert not any(step[0] for step in lost) and all(step == (0.,0) for step in lost[-30:])
    return dict(trial_sha256=sha(source/"RESULT.json"), analysis_only=True, old_last_applied_ms=last_cmd,
                old_rejoin_window_ms=[settle_from, int(decisions[-1,0])],
                old_rejoin_forward_decisions=int(np.count_nonzero(longitudinal[decisions[:,0]>settle_from])),
                terminal=dict(frames=len(terminal), old_confidence_nonnegative=sum(item[3] >= 0. for item in terminal),
                              auxiliary_red_pixels_present=sum(item[4] is not None for item in terminal),
                              initial=terminal[:3], final=terminal[-3:]),
                repeated_terminal_frame_candidate=dict(frames=55, raw_bearing=marker_bearing(still_image),
                    emitted_turn=sum(a[1]!=0 for a in actions), emitted_forward=sum(a[0]>0 for a in actions),
                    all_unique_commands=sorted(set(actions))),
                invisible_bounds=dict(no_forward=True, turn_frames=sum(t[1]!=0 for t in lost), stopped_by_frame=271))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("trial",type=Path)
    args=parser.parse_args()
    result=audit(args.trial)
    turn.stand.put(args.out,result)
    print(result,flush=True)
