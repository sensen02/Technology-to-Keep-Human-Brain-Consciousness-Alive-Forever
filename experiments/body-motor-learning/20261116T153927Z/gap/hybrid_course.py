"""Test the corrected physical fixture with the same body and reflex."""
import argparse
from pathlib import Path

import anchored_course as actor
from hybrid_floor import applied
from seam_course import SeamReference

HERE=Path(__file__).resolve().parent


def run(mode, output, duration, native):
    q=actor.q
    original=q.stan.identity.capture
    q.SOURCE=HERE/"HYBRID_SLOT.json"
    q.FILLED=HERE/"HYBRID_FILLED_SLOT.json"
    actor.SlotReference=SeamReference

    def identity(core, config, *, extra_paths):
        return original(core,{**config,"tile_seam_contact_management":
                        "uninterrupted left plate; existing right patch; right/left anchors .055 from partition faces"},
                        extra_paths=[*extra_paths,Path(__file__).resolve(),HERE/"hybrid_floor.py",HERE/"seam_course.py"])

    q.stan.identity.capture=identity
    try:
        with applied():
            if mode=="course":
                actor.run(output,duration,native)
            else:
                q.run(output,"flat" if mode=="filled" else "off",.036,20.,400,duration,native)
    finally:
        q.stan.identity.capture=original


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("course","off","filled"))
    p.add_argument("output",type=Path)
    p.add_argument("--duration",type=int,default=actor.END)
    p.add_argument("--native",action="store_true")
    a=p.parse_args()
    run(a.mode,a.output.resolve(),a.duration,a.native)
