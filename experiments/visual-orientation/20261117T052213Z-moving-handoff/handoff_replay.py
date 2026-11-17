"""Reconstruct the physics and transport at an exact matched moving checkpoint."""
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
VISUAL = HERE.parent
SOURCE = VISUAL / "20261117T022700Z-sac-startup-replay"
PAIRED = VISUAL / "20261117T032829Z-sac-detour-complete"
sys.path.insert(0,str(SOURCE))
import course


def read(path):
    return json.loads(path.read_text())


def put(path,obj):
    path.write_text(json.dumps(obj,indent=2)+"\n")


class MatchedCourse(course.Course):
    """Use physically archived eye frames while proving state-by-state replay."""

    def __init__(self, case, path, brain, prefix):
        self.prefix = prefix
        self.prefix_stop = read(HERE/"PLAN.json")["decision_tick"]
        super().__init__(case,path,brain)
        self.scene = dict(self.scene)

    def capture(self):
        tick = (self.time-500)//20
        if self.prefix is None or self.time<500:
            return super().capture()
        assert 0 <= tick <= self.prefix_stop
        if tick==0:
            super().capture()
            np.testing.assert_array_equal(self.rgb,self.prefix["eyes"][0])
            return
        self.rgb = self.prefix["eyes"][tick].copy()
        self.brain.new_frame(self.rgb)

    def resume_live_eyes(self):
        # The pending image was rendered at the current state and is released
        # on the next observation. The public frame remains one tick old.
        pending = self.eyes.image(self.target,course.WIDTH,course.COLOR,self.background)
        np.testing.assert_array_equal(pending,self.prefix["eyes"][self.n+1])
        self.eyes.pending = (self.time,pending)
        self.eyes.released = (self.time-20,self.rgb.copy())
        self.prefix = None


def reconstruct():
    plan = read(HERE/"PLAN.json")
    source = VISUAL/plan["source"]
    receipt = read(source/"RESULT.json")
    assert read(source/"QUALIFIED.json")["passed"]
    assert receipt["files"]["EPISODE.npz"]==course.sha(source/"EPISODE.npz")
    prefix_file = source/"EPISODE.npz"
    with np.load(prefix_file,allow_pickle=False) as z:
        prefix = {key:z[key] for key in (
            "eyes","x","vision","mode","pace","qpos","com","distance","bearing",
            "gated","delivered","contact_W","reward","requested")}
    course.SCENES = PAIRED/"SCENES.json"
    root = HERE/"prefix"
    cut = plan["decision_tick"]
    # Validate the preregistered condition on the immutable source before the long replay.
    available = prefix["delivered"][cut-25:cut,:,0]
    assert np.count_nonzero(available>0)>=plan["required_forward_samples_last_500ms"]
    assert np.all(available[-1]>0)
    tick = monotonic()
    env = MatchedCourse(plan["scene"],root,course.new_brain(),prefix)
    np.testing.assert_array_equal(env.body.d.qpos,prefix["qpos"][0])
    np.testing.assert_array_equal(env.past,prefix["x"][0])
    for i in range(cut):
        mode,pace = int(prefix["mode"][i]),float(prefix["pace"][i])
        reward,_,com,d,b,contact,delivered,gated = env.step(mode,pace)
        assert env.end is None and not reward["terminated"]
        np.testing.assert_array_equal(env.body.d.qpos,prefix["qpos"][i+1])
        np.testing.assert_array_equal(env.past,prefix["x"][i+1])
        np.testing.assert_array_equal(com,prefix["com"][i])
        np.testing.assert_array_equal(delivered,prefix["delivered"][i])
        np.testing.assert_array_equal(gated,prefix["gated"][i])
        np.testing.assert_allclose([d,b,contact,reward["reward"]],
            [prefix["distance"][i],prefix["bearing"][i],prefix["contact_W"][i],prefix["reward"][i]],
            atol=1.e-12,rtol=0.)
        if (i+1)%500==0:
            print("matched prefix",i+1,round(d,4),round(monotonic()-tick,1),flush=True)
    np.testing.assert_array_equal(env.observe()[1],prefix["vision"][cut])
    env.resume_live_eyes()
    tail = env.footprint.bounds(env.body.d.qpos)[0]-env.box_edge
    assert tail>=plan["minimum_clearance"] and env.previous[0]>0 and env.brain.cue[2]
    forward_samples = int(np.count_nonzero(prefix["delivered"][cut-25:cut,:,0]>0))
    assert forward_samples >= plan["required_forward_samples_last_500ms"]
    assert np.all(prefix["delivered"][cut-1,:,0]>0)
    assert env.forces.count==env.time*80 and not env.forces.counts.any()
    put(HERE/"PREFIX_MATCH.json",dict(case=plan["scene"],decision_tick=cut,step_ms=env.time,
        native_steps_with_warm=env.forces.count,source_sha256=receipt["files"]["EPISODE.npz"],
        qpos_and_com_exact=True,neck_and_commands_exact=True,pending_eye_exact=True,
        final_range=env.distance(),full_tail_margin=tail,previous_forward_samples_5ms=forward_samples,
        wall_s=monotonic()-tick))
    print("reconstructed moving checkpoint",env.n,env.time,env.distance(),tail,flush=True)
    return env,prefix,source


def main():
    torch.set_num_threads(1)
    # Resolve both modules before allocating the long native replay.
    from handoff_snapshot import write_checkpoint
    from handoff_suffix import run_from_checkpoint
    if sys.argv[1:]==["--check-imports"]:
        print("handoff modules resolved",flush=True)
        return
    env,prefix,source = reconstruct()
    try:
        digest = write_checkpoint(env,HERE/"MOVING_CHECKPOINT.pkl")
        put(HERE/"SNAPSHOT_WRITTEN.json",dict(sha256=digest,step_ms=env.time))
        run_from_checkpoint(env,prefix,source,digest)
    finally:
        env.close()


if __name__=="__main__":
    main()
