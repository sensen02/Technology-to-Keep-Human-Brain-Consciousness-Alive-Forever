"""Keep the proven gap geometry and clear the two artificial tile boundaries."""
import argparse
from pathlib import Path

import anchored_course as actor


class SeamReference(actor.SlotReference):
    margin=.055

    def _foothold(self, endpoint):
        x0,x1=self.box_min[0],self.box_max[0]
        if x0-self.margin < endpoint[0] < x1+self.margin:
            inside_actual_slot=(self.box_min[1]-.020 < endpoint[1] < self.box_max[1]+.020)
            if inside_actual_slot or endpoint[0] >= (x0+x1)/2.:
                endpoint[0]=x1+self.margin
            else:
                endpoint[0]=x0-self.margin
        return 0.

    def _near_box(self, start, end):
        lo,hi=sorted((start[0],end[0]))
        return bool(lo < self.box_max[0]+self.margin and hi > self.box_min[0]-self.margin)


def main(output, duration, native):
    original=actor.q.stan.identity.capture
    actor.SlotReference=SeamReference
    def identity(core, setup, *, extra_paths):
        return original(core,{**setup,"tile_seam_contact_management":"raise swing and then land .055 from x tile boundaries",
                              "geometry_mode":"same frozen X-first terrain"},
                        extra_paths=[*extra_paths,Path(__file__).resolve()])
    actor.q.stan.identity.capture=identity
    try:
        actor.run(output,duration,native)
    finally:
        actor.q.stan.identity.capture=original


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--duration",type=int,default=actor.END)
    p.add_argument("--native",action="store_true")
    a=p.parse_args()
    main(a.output.resolve(),a.duration,a.native)
