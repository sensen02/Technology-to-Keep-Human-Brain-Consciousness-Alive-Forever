"""Save declared command timelines and held-out variations for physical evaluation."""
from course import CENTER_SPEED, UNITS
from physical import HERE, write


def add(rows,name,duration,p,q=None):
    rows.append(dict(name=name,duration_ms=duration,first=list(p),last=list(p if q is None else q)))


def speed_segments(high):
    start=(0.,0.)
    slow=(CENTER_SPEED,0.)
    rapid=(high,0.)
    rows=[]
    add(rows,"stand",500,start)
    add(rows,"low",2400,slow)
    add(rows,"accel",1600,slow,rapid)
    add(rows,"high",3200,rapid)
    add(rows,"decel",1600,rapid,slow)
    add(rows,"slowed",800,slow)
    return rows


def speed_curve(high,sign=None):
    rows=speed_segments(high)
    slow=(CENTER_SPEED,0.)
    if sign is not None:
        arc=(.13,sign*.115)
        straight=(.091,0.)
        add(rows,"turn_in",1200,slow,arc)
        add(rows,"arc",8400,arc)
        add(rows,"turn_out",1200,arc,straight)
        add(rows,"exit",5200,straight)
        add(rows,"brake",500,straight,(0.,0.))
    add(rows,"stop",1200,(0.,0.))
    data=dict(units=list(UNITS),segments=rows,
              speed_pair=dict(low="low",high="high",slowed="slowed",delta_min=.005))
    if sign is not None:
        data["turn"]=dict(direction=sign,arc="arc",exit="exit",minimum_yaw_deg=20.)
    return data


if __name__=="__main__":
    for name,high,sign in [("RAMP_BENCH",1.16*CENTER_SPEED,None),
                           ("COMBINED_LEFT",1.16*CENTER_SPEED,1),
                           ("COMBINED_RIGHT_HELD",1.135*CENTER_SPEED,-1)]:
        write(HERE/f"{name}.json",speed_curve(high,sign))
