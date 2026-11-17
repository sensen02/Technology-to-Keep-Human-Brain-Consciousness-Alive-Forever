"""Combine the four admitted DN and four admitted AN roots in one transport."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from collections import deque

HERE=Path(__file__).resolve().parent
JOINT=HERE.parent/"20261116T232400Z-joint-brain"
SEPARATED=HERE.parent/"20261116T225900Z-separated-afferents"
FAR=HERE.parent/"20261116T191700Z-far-obstacle"
APPROACH=HERE.parent/"20261116T184600Z-approach"
TOUCH=HERE.parent/"20261116T180500Z-touch"
sys.path[:0]=[str(p) for p in (
    JOINT,SEPARATED,HERE.parent/"20261116T211600Z-touch-prediction",TOUCH,APPROACH,FAR)]
from approach_runtime import ApproachLink, BINDING as DN_BINDING, Motion, stand_window, turn, sha
from approach_vision import features, network
from far_optics import WEIGHTS as FAR_WEIGHTS
from visual_rejoin import marker_bearing
from run_brain import SharedBrain, SCALE
from separation import brain_projection, BINDING as AN_BINDING, GRAPH as AN_GRAPH
from touch_runtime import SourceTransport

BINDING=HERE/"UNIFIED_BINDING.json"
PARENT=JOINT/"MODEL/joint.pt"
CHECKPOINT=HERE/"BRAIN.pt"
STOP, MOVE, LEFT, RIGHT = 0,1,2,3
MODES=("stop","walk","left","right")
PARAMS=dict(start_turn_deg=17., align_deg=3.5, turn_rearm_deg=20., settle_frames=35,
            stop_range=2.13, stop_heading_deg=12., slow_range=2.52, fast=.08318, slow=.071713)


def bind():
    dn=json.loads(DN_BINDING.read_text())
    an=json.loads(AN_BINDING.read_text())
    assert dn["schema"]==an["schema"] and dn["transport_config"]==an["transport_config"]
    downward=[p for p in dn["ports"] if p["direction"]=="DN"]
    upward=[p for p in an["ports"] if p["direction"]=="AN"]
    assert tuple(p["source_cell_type"] for p in downward+upward)==(
        "DNa02","DNa02","DNg97","DNg97","AN06B002","AN06B002","AN19B004","AN19B004")
    assert all(p["source_reviewed"] for p in downward+upward)
    paths={}
    for row in dn["source_files"]+an["source_files"]:
        path,digest=row["path"],row["sha256"]
        assert path not in paths or paths[path]==digest
        paths[path]=digest
    for p in (DN_BINDING,AN_BINDING,AN_GRAPH):
        paths[str(p)]=sha(p)
    dn.update(ports=downward+upward,source_files=[dict(path=p,sha256=d) for p,d in paths.items()],
              status="one BANC-source scoped DN and AN queue; engineering transfer")
    turn.stand.put(BINDING,dn)
    SourceTransport(BINDING,sha(BINDING)).verify_final()


class Neck(ApproachLink):
    def __init__(self):
        super().__init__()
        old_dn=self.ids
        self.link=SourceTransport(BINDING,sha(BINDING))
        self.ids=self.link.ports["DN"]
        assert old_dn==self.ids
        self.an_roots=self.link.ports["AN"]
        assert self.an_roots==json.loads(AN_GRAPH.read_text())["root_order"]
        self.matrix,self.pool=brain_projection()
        self.an_cut=False
        self.an_step=None
        self.an_values=np.zeros(4)
        self.last_dn_step=None

    def cut_an(self,value):
        self.an_cut=value
        self.link.set_cut("AN",value)
        self.an_step=None
        self.an_values.fill(0.)

    def sense(self,values):
        if self.an_cut:
            return
        self.link.send(dict(direction="AN",root_ids=self.an_roots,values=list(map(float,values)),
                            capture_step=self.link.step,source_bindings_sha256=self.link.digest,
                            **{"from":"other","to":"neck"}))

    def observe(self):
        valid=self.an_step is not None and self.link.step-self.an_step<10
        return (self.an_values.copy() if valid else np.zeros(4)),valid

    def advance(self):
        for packet in self.link.advance():
            values=packet["values"]
            if packet["direction"]=="AN":
                targets=self.matrix@values
                self.an_values=np.asarray([targets[idx].mean() for idx in self.pool])
                self.an_step=packet["release_step"]
            else:
                speed=self.receiver.velocity({**packet,"root_ids":packet["root_ids"][2:],"values":values[2:]})
                yaw,_=self.codec.read(values[:2])
                self.time=self.last_dn_step=packet["release_step"]
                self.decoded=(speed,yaw)


class NavigationBrain(SharedBrain):
    """One checkpoint contains the original shared GRU and binocular range branch."""
    def __init__(self):
        super().__init__()
        self.range_vision=network()
        self.range_vision.requires_grad_(False)
        self.estimate=deque(maxlen=5)
        self.memory=None
        self.previous=0.
        self.visual=0.
        self.original_visual=-4.
        self.turning=0
        self.settle=0
        self.near=False
        self.parked=False
        self.proposal=(0.,0)
        self.cue=(0.,0.,False)

    @torch.inference_mode()
    def new_frame(self,image):
        marker=marker_bearing(image)
        old=self.visual_feature(torch.from_numpy(image[None]))
        predicted=self.range_vision(features(torch.from_numpy(image[None])))[0]
        angle,distance,confidence=predicted.tolist()
        self.original_visual=float(old.item())
        visible=marker is not None and (self.original_visual>0. or confidence>0.)
        self.visual=4. if visible else -4.
        if not visible:
            self.estimate.clear()
            self.turning=0
            self.settle=0
            self.parked=False
            self.near=False
            self.proposal=(0.,0)
            self.cue=(0.,0.,False)
            return
        angle*=40.
        angle=marker if abs(angle-marker)>9. or confidence<0. else angle
        self.estimate.append((angle,distance+2.5))
        angle,distance=np.median(self.estimate,axis=0)
        self.cue=(float(angle),float(distance),True)
        p=PARAMS
        at_stop_range=distance<p["stop_range"]
        if self.parked or (at_stop_range and abs(angle)<=p["stop_heading_deg"]):
            self.parked=True
            self.proposal=(0.,0)
            return
        threshold=(p["stop_heading_deg"] if at_stop_range else
                   p["turn_rearm_deg"] if self.near else p["start_turn_deg"])
        if self.settle:
            self.settle-=1
            self.proposal=(0.,0)
            return
        if self.turning or abs(angle)>threshold:
            if not self.turning:
                self.turning=1 if angle>0 else -1
            if angle*self.turning>p["align_deg"]:
                self.proposal=(0.,self.turning)
                return
            self.turning=0
            self.settle=p["settle_frames"]
            self.proposal=(0.,0)
            return
        self.near=self.near or distance<p["slow_range"]
        self.proposal=(p["slow"] if self.near else p["fast"],0)

    @torch.inference_mode()
    def decide(self,arrived,valid):
        # The shared GRU learned a new locomotion cue can rearm a stopped
        # episode. A visual pause removes that cue; a sensory veto leaves it
        # in place so tactile stop-and-hold remains learned.
        drive=self.visual if self.proposal!=(0.,0) else -4.
        x=torch.as_tensor(np.r_[drive,arrived,float(valid),self.previous],dtype=torch.float32)[None,None]
        logit,latent,self.memory=self.advance(x,self.memory)
        enabled=float(logit.item())>0.
        speed,yaw=self.proposal if enabled else (0.,0)
        self.previous=float(speed>0. or yaw!=0)
        prediction=self.predict(latent,torch.full((1,1),self.previous))
        return (float(speed),int(yaw)),float(logit.item()), prediction


def initialize():
    prediction=torch.load(PARENT,weights_only=True,map_location="cpu")
    model=NavigationBrain().eval()
    missing,unexpected=model.load_state_dict(prediction["brain"],strict=False)
    assert not unexpected and all(k.startswith("range_vision.") for k in missing)
    model.range_vision.load_state_dict(torch.load(FAR_WEIGHTS,weights_only=True,map_location="cpu"))
    torch.save(dict(brain=model.state_dict(),mean=prediction["mean"],scale=prediction["scale"],
                    parents={str(p):sha(p) for p in (PARENT,FAR_WEIGHTS)},initialization=PARAMS,
                    optimizer_updates=0,drive_semantics="visual locomotion proposal vs stillness"),CHECKPOINT)
    return model


if __name__=="__main__":
    torch.set_num_threads(1)
    bind()
    initialize()
