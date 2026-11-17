"""Vary the local settle interval in the visual turn-to-gait handoff."""
import argparse
from pathlib import Path
import approach
from approach_runtime import turn,sha

Base=approach.Policy

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--wait",type=int,required=True)
    p.add_argument("--angle",type=float,default=20.)
    a=p.parse_args()

    class Settled(Base):
        def act(self,image):
            before=self.phase
            result=super().act(image)
            if before=="align" and self.phase=="settle":
                self.wait=a.wait
            return result

    approach.Policy=Settled
    result=approach.trial(a.out.resolve(),a.angle,2.68,"intact",.12,18800)
    turn.stand.put(a.out/"VARIANT.json",dict(wait_ticks=a.wait,phase_period_ms=20,wrapper_sha256=sha(Path(__file__))))
    raise SystemExit(0 if result["passed"] else 1)
