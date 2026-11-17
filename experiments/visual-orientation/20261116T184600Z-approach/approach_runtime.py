"""Frozen physical body and scoped DNa02/DNg97 delivery for visual approach."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TURN = HERE.parent/"20261116T023521Z"
FORWARD = HERE.parent/"20261116T170000Z"
SPEED = HERE.parent/"20261116T172138Z-dng97"
sys.path[:0]=[str(TURN),str(FORWARD),str(SPEED)]
from bench_runtime import prepare, turn, stand_window
from codec import RuntimeCodec
from optics import Eyes
from forward_link import sha, SourceTransport
from run_speed import VelocityReceiver
from pace_lookup import ObservedPhase
from control_body import load_body_policy
from verify_native import Record

BINDING = HERE/"APPROACH_BINDING.json"
PIVOT = TURN/"DNa02_BINDING.json"
PACE = SPEED/"DNg97_BINDING.json"


def binding():
    yaw,speed = (json.loads(p.read_text()) for p in (PIVOT,PACE))
    yaw_ports = [p for p in yaw["ports"] if p["direction"]=="DN"]
    speed_ports = [p for p in speed["ports"] if p["direction"]=="DN"]
    an = [p for p in yaw["ports"] if p["direction"]=="AN"]
    yaw.update(ports=yaw_ports+speed_ports+an,source_files=yaw["source_files"]+speed["source_files"]+
               [dict(path=str(path),sha256=sha(path)) for path in (
                   TURN/"OFFICIAL_GRAPH.json",SPEED/"OFFICIAL_GRAPH.json",TURN/"CODEC.pt",PIVOT,PACE)],
               status="DNa02 and DNg97 scoped conjunctive engineering binding; AN inactive")
    with BINDING.open("x") as f:
        json.dump(yaw,f,indent=2)


class ApproachLink:
    def __init__(self):
        self.link = SourceTransport(BINDING,sha(BINDING))
        self.codec = RuntimeCodec(TURN/"CODEC.pt","clean",0)
        self.receiver = VelocityReceiver()
        self.ids = self.link.ports["DN"]
        self.time = None
        self.decoded = (0.,0)
        self.cut = False

    def switch(self,cut):
        self.cut=cut
        self.link.set_cut("DN",cut)
        self.time=None
        self.decoded=(0.,0)

    def send(self,speed,yaw):
        if self.cut:
            return
        assert 0.<=speed<=.1 and yaw in (-1,0,1)
        amplitude = float(speed/.1)
        self.link.send(dict(direction="DN",root_ids=self.ids,
             values=self.codec.send(yaw)+[amplitude,amplitude],capture_step=self.link.step,
             source_bindings_sha256=self.link.digest,**{"from":"brain","to":"neck"}))

    def advance(self):
        for packet in self.link.advance():
            speed=self.receiver.velocity({**packet,"root_ids": packet["root_ids"][2:], "values":packet["values"][2:]})
            yaw,_=self.codec.read(packet["values"][:2])
            self.time=packet["release_step"]
            self.decoded=(speed,yaw)

    def current(self):
        return self.decoded if self.time is not None and self.link.step-self.time<65 else (0.,0)


class Motion:
    def __init__(self,body):
        self.stock=load_body_policy()
        self.pace=ObservedPhase(body)

    def action(self,x,speed,yaw):
        if yaw or speed<1.e-7:
            return self.stock(x,0,yaw)
        return self.pace(x,[speed,0.])


if __name__ == "__main__":
    binding()
