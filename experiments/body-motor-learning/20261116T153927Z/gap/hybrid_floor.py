"""Keep RF-side slot tiles and use an uninterrupted left-side support surface."""
from contextlib import contextmanager

import geometry

original=geometry.box_layout


def layout(case):
    if case.get("partition")!="left_continuous":
        yield from original(case)
        return
    for name,position,size in original(case):
        lower=position[1]-size[1]
        upper=min(position[1]+size[1],0.)
        if upper>lower:
            yield (name,(position[0],(upper+lower)/2.,position[2]),
                        (size[0],(upper-lower)/2.,size[2]))
    d=geometry.LIMIT
    yield geometry.TERRAIN+"left_continuous",(0.,d/2.,-.4),(d,d/2.,.4)


@contextmanager
def applied():
    old=geometry.box_layout
    geometry.box_layout=layout
    try:
        yield
    finally:
        geometry.box_layout=old
