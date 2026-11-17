"""Physical 20 ms transition provider for the scoped single-box SAC experiment."""
import argparse
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
VISUAL = HERE.parent
UNIFIED = VISUAL/"20261117T010100Z-unified-neck-final"
BENCH = VISUAL/"20261116T235500Z-navigation-benchmark"
REWARD_SOURCE = VISUAL/"20261117T014000Z-distance-reward"
sys.path[:0] = [str(UNIFIED),str(BENCH),str(REWARD_SOURCE)]
from unified import Neck, NavigationBrain, CHECKPOINT, BINDING, Motion as BodyMotion, turn, sha
from evaluate import Eyes, BODY, Forces, Footprint, prepare, FRONT, SCALE, W, COLOR, WIDTH, WiderDetour
sys.path.insert(0,str(REWARD_SOURCE))
import reward as scored_reward
assert Path(scored_reward.__file__).resolve()==REWARD_SOURCE/"reward.py"
from reward import ACTOR_S, GAMMA, Reward, Task, Measurement, Motion as PhysicalMotion
from run_brain import LEGS

SCENES=HERE/"SCENES.json"
SLOW,FAST=.071713,.08318
FRACTION=(FAST+SLOW)/2
VIS_DIM=14
SUBSTEPS=4


def representation(brain,rgb,previous):
    """Actor features come from the two rendered eyes and its own past request."""
    angle,distance,visible=brain.cue
    value=rgb.astype(np.float32)
    red,green,blue=(value[...,i] for i in range(3))
    signal=(green>55.) & (green>1.7*red) & (green>1.7*blue)
    sectors=signal.reshape(2,32,4,8).sum((1,3)).reshape(-1)
    return np.asarray([
        angle/45.,distance/6.,float(visible),brain.original_visual/4.,
        *np.clip(sectors/75.,0.,1.),float(previous[0])/FAST,float(previous[1])],dtype=np.float32)


def action_vector(mode,pace):
    return (SLOW+(FAST-SLOW)*pace,0) if mode==1 else (0.,1 if mode==2 else -1 if mode==3 else 0)


def command_code(command):
    speed,yaw=command
    mode=2 if yaw>0 else 3 if yaw<0 else int(speed>0.)
    return mode,float(np.clip((speed-SLOW)/(FAST-SLOW),0.,1.)) if mode==1 else .5


def new_brain():
    core=NavigationBrain().eval()
    core.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",weights_only=True)["brain"])
    return core


class Course:
    def __init__(self,case,path,brain,cut=False):
        self.scene=json.loads(SCENES.read_text())[case]
        assert self.scene["pit"] is None
        self.body,self.geometry=prepare(path,self.scene)
        self.footprint=Footprint(self.body)
        self.forces=Forces(self.body,self.geometry)
        self.body.recorder=self.forces
        self.eyes=Eyes(self.body)
        self.target=self.eyes.place(self.scene["angle"],self.scene["distance"])
        self.background=(self.eyes.pose()[0].copy(),self.scene["shift"],.263)
        self.brain=brain
        self.neck=Neck()
        if cut:
            self.neck.switch(True)
        self.motor=BodyMotion(self.body)
        self.measured=self.body.measure()
        self.action=np.zeros(42)
        self.afferents=np.zeros(2)
        self.time=0
        self.n=0
        self.rgb=None
        self.past=np.zeros((4,7),dtype=np.float32)
        self.previous=(0.,0)
        self.end=None
        self.box_edge=max((b[0]+b[3] for b in self.geometry["boxes"]),default=None)
        self.angles=[int(self.body.plant.joint(f"joint_{leg}Tibia_pitch").qposadr[0]) for leg in LEGS]
        self.vel=[int(self.body.plant.joint(f"joint_{leg}Tibia_pitch").dofadr[0]) for leg in LEGS]
        self.physical_duration=500
        for _ in range(25):
            self.capture()
            self.integrate((0.,0))
        self.capture()
        self.initial_distance=self.distance()
        self.reward=Reward(Task(self.initial_distance,tuple(self.scene["band"]),self.box_edge is not None,require_motion=True))

    def close(self):
        self.eyes.close()

    def distance(self):
        return float(np.linalg.norm(self.target[:2]-self.eyes.pose()[0][:2]))

    def capture(self):
        sampled=self.eyes.capture(self.time,self.target if self.time>=self.physical_duration else None,
                                  WIDTH,COLOR,self.background)
        if sampled is not None:
            _,self.rgb=sampled
            self.brain.new_frame(self.rgb)

    def observe(self):
        return self.past.copy(),representation(self.brain,self.rgb,self.previous)

    def pose(self):
        c=self.body
        return c.d.subtree_com[c.th].copy(), float(np.arctan2(c.d.xmat[c.th,3],c.d.xmat[c.th,0]))

    @torch.inference_mode()
    def integrate(self,proposal):
        c=self.body
        samples=[]
        trajectory=[self.pose()]
        loads=[]
        upright=[]
        nonfoot=[]
        contact=[]
        effective=[]
        selected=[]
        self.brain.proposal=proposal
        # A policy may keep an intention while the marker is occluded. The
        # previously trained shared GRU still vetoes that intention from AN.
        self.brain.visual=4. if proposal!=(0.,0) else -4.
        for i in range(20):
            if i%5==0:
                signals,valid=self.neck.observe()
                drive=self.brain.visual
                samples.append(np.r_[drive,signals,float(valid),self.brain.previous])
                cmd,_,_=self.brain.decide(signals,valid)
                self.neck.send(*cmd)
                delivered=self.neck.current()
                self.action=self.motor.action(turn.features(c,self.measured),*delivered)
                effective.append(delivered)
                selected.append(cmd)
            self.neck.sense(np.r_[self.afferents,np.clip(self.measured["loads"][FRONT]/(SCALE*W),0.,1.)])
            self.forces.reset_ms()
            self.measured=c.tick(self.action)
            self.neck.advance()
            self.afferents=np.clip(self.forces.sides/(80*W*.02),0.,1.)
            self.forces.peak=np.maximum(self.forces.peak,self.forces.frame)
            contact.append(float(self.forces.frame.max())/W if self.forces.nbox else 0.)
            trajectory.append(self.pose())
            loads.append(self.measured["loads"].copy()/W)
            nonfoot.append(self.measured["nonfoot"]/W)
            upright.append(float(c.d.xmat[c.th,8]))
            assert not (c.d.qfrc_applied.any() or c.d.xfrc_applied.any())
        self.time+=20
        self.past=np.asarray(samples,dtype=np.float32)
        self.previous=proposal
        return trajectory,np.asarray(loads),nonfoot,upright,contact,np.asarray(effective),np.asarray(selected)

    def step(self,mode,pace):
        start=self.distance()
        command=action_vector(mode,pace)
        trajectory,loads,nonfoot,upright,contact,effective,selected=self.integrate(command)
        self.n+=1
        end=self.distance()
        self.capture()
        com=np.asarray([p for p,_ in trajectory])
        yaw=np.unwrap([y for _,y in trajectory])
        differential=self.target[:2]-self.eyes.pose()[0][:2]
        angle=(np.degrees(np.arctan2(differential[1],differential[0])-yaw[-1])+180.)%360.-180.
        # The score alone sees the geometric footprint. Recheck it whenever a
        # terminal success is plausible and retain continuous pose/contact logs.
        tail=None
        if self.box_edge is not None and end<=self.scene["band"][1]+.20:
            tail=self.footprint.bounds(self.body.d.qpos)[0]
        cleared=self.box_edge is None or (tail is not None and tail>=self.box_edge+.03)
        v=np.diff(com,axis=0)*1000.
        measurement=Measurement(end,float(angle),float(np.sqrt(np.sum(v[:,:2]**2,axis=1).mean())),
                    min(upright),float(com[:,2].min()),float(loads.sum(1).mean()),int((loads>=.02).sum(1).min()),
                    yaw_speed_deg_s=float(np.sqrt(np.square(np.degrees(np.diff(yaw))*1000.).mean())),
                    nonfoot_W=max(nonfoot),obstacle_peak_W=max(contact),cleared=cleared,
                    neutral_command=not effective.any(),motion=PhysicalMotion(tuple(com[-1,:2]),float(np.degrees(yaw[-1]))))
        step=self.reward.step(start,measurement,deadline=self.n==self.scene["deadline_ticks"],
                              truncated=self.n==self.scene["ticks"] and self.n<self.scene["deadline_ticks"])
        self.end=step if step["terminated"] or step["truncated"] else None
        state=self.body.d
        velocity=float(self.body.com_velocity()@state.xmat[self.body.th].reshape(3,3)[:,0])
        truth=np.r_[state.qpos[self.angles],state.qvel[self.vel],self.measured["loads"]/W,velocity,state.xmat[self.body.th,8]]
        return step,truth,com[-1],end,angle,max(contact),effective,selected


class Teacher:
    def __init__(self,kind,scene):
        self.kind=kind
        self.mirror=scene.get("mirror",False)
        self.detour=WiderDetour() if kind=="detour" else None
        self.seen_box=0

    def choose(self,env, _obs):
        if self.kind=="detour":
            rgb=env.rgb[[1,0],:,::-1,:].copy() if self.mirror else env.rgb
            if self.detour.elapsed is None:
                v=rgb.astype(np.float32)
                red,green,blue=np.moveaxis(v,-1,0)
                box=(green>75.)&(green>2.*red)&(green>2.*blue)
                self.seen_box=self.seen_box+1 if int(box.sum())>=6 else 0
                if self.seen_box>=2:
                    self.detour.hazard_frames=2
                    self.detour.elapsed=0
            command,_=self.detour.act(rgb)
            if self.mirror:
                command=(command[0],-command[1])
            return command_code(command)
        return command_code(env.brain.proposal)


def rollout(out,case,policy,brain,cfg,pin=None,cut=False):
    tick=monotonic()
    env=Course(case,out,brain,cut=cut)
    subs,visions=[],[]
    modes,paces,rewards,terminal,truncated,reasons,physical,coms,distance,bearing,contacts=[],[],[],[],[],[],[],[],[],[],[]
    requests,selected,executed,terms,poses,eyes=[],[],[],[],[],[]
    obs=env.observe()
    subs.append(obs[0])
    visions.append(obs[1])
    poses.append(env.body.d.qpos.copy())
    eyes.append(env.rgb.copy())
    try:
        while env.end is None:
            mode,pace=policy.choose(env,obs)
            step,truth,com,d,a,contact,actual,gated=env.step(mode,pace)
            modes.append(mode)
            paces.append(pace)
            rewards.append(step["reward"])
            terminal.append(step["terminated"])
            truncated.append(step["truncated"])
            reasons.append(step["reason"] or "")
            physical.append(truth)
            coms.append(com)
            distance.append(d)
            bearing.append(a)
            contacts.append(contact)
            requests.append(action_vector(mode,pace))
            selected.append(gated)
            executed.append(actual)
            terms.append(list(step["terms"].values()))
            poses.append(env.body.d.qpos.copy())
            eyes.append(env.rgb.copy())
            obs=env.observe()
            subs.append(obs[0])
            visions.append(obs[1])
            if env.n%250==0:
                print(case,env.n,"last_distance",round(d,4),"wall",round(monotonic()-tick,1),flush=True)
    finally:
        env.close()
    assert env.forces.count==env.time*80
    source=env.neck.link.verify_final()
    assert source["actual_steps"]==env.time
    np.savez_compressed(out/"EPISODE.npz",x=subs,vision=visions,mode=modes,pace=paces,reward=rewards,
                        terminal=terminal,truncated=truncated,reasons=reasons,physical=physical,com=coms,
                        distance=distance,bearing=bearing,contact_W=contacts,requested=requests,gated=selected,
                        delivered=executed,terms=terms,term_names=list(env.end["terms"]),qpos=poses,eyes=eyes,
                        goal=env.target)
    turn.stand.put(out/"TRANSPORT.json",env.neck.link.journal())
    result=dict(case=case,policy=cfg,plan=env.scene,geometry=env.geometry,actor_ticks=env.n,
                native_steps=env.n*1600,warm_native_steps=40000,initial_range=env.initial_distance,
                final_range=distance[-1],closest_range=min(distance),final_bearing=bearing[-1],
                reason=env.end["reason"],terminated=env.end["terminated"],truncated=env.end["truncated"],
                contact_peak_W=max(contacts),reward=float(sum(rewards)),
                discounted_reward=float(np.asarray(rewards)@GAMMA**np.arange(env.n)),
                source=source,pinned_checkpoint=pin,intent_drive="proposal_active",wall_s=monotonic()-tick,
                files={n:sha(out/n) for n in ("EPISODE.npz","TRANSPORT.json")})
    turn.stand.put(out/"RESULT.json",result)
    print(result,flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("out",type=Path)
    parser.add_argument("policy",choices=("visual","detour"))
    args=parser.parse_args()
    torch.set_num_threads(1)
    brain=new_brain()
    scene=json.loads(SCENES.read_text())[args.case]
    rollout(args.out,args.case,Teacher(args.policy,scene),brain,args.policy)
