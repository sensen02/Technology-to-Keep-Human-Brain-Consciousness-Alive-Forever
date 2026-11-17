"""Continue the selected slot maneuver until the last feet pass the far x plane."""
import argparse
from pathlib import Path

import crossing


def main(output):
    wrapper = Path(__file__).resolve()
    original = crossing.stan.identity.capture
    resumed_ms = 45600
    crossing.RESUMED_WALK = resumed_ms
    crossing.END = crossing.RELEASE + resumed_ms + crossing.TERMINAL

    def bind(core, scope, *, extra_paths):
        return original(core,{**scope,"resumed_walk_ms":resumed_ms},extra_paths=[*extra_paths,wrapper])

    crossing.stan.identity.capture = bind
    try:
        crossing.run(output,"far",.036,20.,400,crossing.END,True)
    finally:
        crossing.stan.identity.capture = original


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    main(a.output.resolve())
