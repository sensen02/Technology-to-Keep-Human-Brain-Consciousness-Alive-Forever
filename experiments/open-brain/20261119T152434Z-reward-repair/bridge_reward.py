"""Use continuous approach progress, support margins and real food as training rewards."""
from dataclasses import asdict, dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class Weights:
    foot_margin_cm: float = .112
    approach_cm: float = .24
    gamma: float = .985
    fall_cost: float = 14.
    contact_cost: float = .65
    time_cost: float = .024
    approach_cost: float = 5.
    overhang_cost: float = .22
    crossing_bonus: float = 2.5
    intake_credit: float = 5.
    meal_bonus: float = 7.
    meal_volume: float = .18


CFG=Weights()


@dataclass(frozen=True)
class Snapshot:
    x: float
    y: float
    intake: float
    hit: bool = False
    fall: bool = False
    has_food: bool = True


@dataclass(frozen=True)
class RewardParts:
    route: float
    unsafe_advance: float
    overhang: float
    contact: float
    fall: float
    time: float
    effort: float
    crossing: float
    food: float
    meal: float

    @property
    def value(self):
        return (self.route+self.unsafe_advance+self.overhang+self.contact+self.fall+self.time+
                self.effort+self.crossing+self.food+self.meal)


def clearance(t,y):
    return t.side*y-t.edge if t.side else 1.


def prelip(t,x):
    return min(1.,max(0.,(x-(t.near-CFG.approach_cm))/CFG.approach_cm))


def potential(t,s,crossed):
    """Before the lip, progress counts when it reaches the alignment line."""
    if not s.has_food:
        return 0.
    if not t.side or crossed:
        return -.68*((t.goal_x-s.x)**2+(t.goal_y-s.y)**2)**.5
    stop=t.near-CFG.approach_cm
    target=t.side*(t.edge+CFG.foot_margin_cm)
    far_to_food=((t.goal_x-t.far)**2+(t.goal_y-target)**2)**.5
    deficit=max(0.,CFG.foot_margin_cm-clearance(t,s.y))
    alignment=min(1.,deficit/CFG.foot_margin_cm)
    approach=max(0.,stop-s.x)
    premature=alignment*max(0.,s.x-stop)
    corridor=max(0.,t.far-max(stop,s.x))* (1.-alignment) + (t.far-stop)*alignment
    return -.78*(approach+corridor+far_to_food+1.5*deficit+premature)


def cost(t,old,new,*,crossed_before,crossed_now,success,terminal,truncated,speed,turn):
    if terminal and truncated:
        raise ValueError('An episode cannot terminate and truncate in the same transition')
    prev_phi=potential(t,old,crossed_before)
    next_phi=potential(t,new,crossed_now) if not terminal else (0. if success else min(0.,prev_phi/CFG.gamma))
    cost_advance=0.
    overhang=0.
    if t.side and not crossed_before:
        in_zone=prelip(t,max(old.x,new.x))
        deficit=max(0.,CFG.foot_margin_cm-min(clearance(t,old.y),clearance(t,new.y)))
        unsafe=min(1.,deficit/CFG.foot_margin_cm)*in_zone
        cost_advance=-CFG.approach_cost*unsafe*max(0.,new.x-old.x)
        if t.near-.05<=new.x<=t.far+.025:
            overhang=-CFG.overhang_cost*unsafe
    credit=CFG.intake_credit*(min(new.intake,CFG.meal_volume)-min(old.intake,CFG.meal_volume))/CFG.meal_volume
    return RewardParts(
        route=CFG.gamma*next_phi-prev_phi,
        unsafe_advance=cost_advance,
        overhang=overhang,
        contact=-CFG.contact_cost*new.hit,
        fall=-CFG.fall_cost*new.fall,
        time=-CFG.time_cost,
        effort=-.010*abs(speed)-.014*abs(turn),
        crossing=CFG.crossing_bonus*(new.has_food and crossed_now and not crossed_before and not new.fall),
        food=credit,
        meal=CFG.meal_bonus*success,
    )


def smoke():
    t=SimpleNamespace(side=1,near=.78,far=1.06,edge=.14,goal_x=1.87,goal_y=.06)
    def step(a,b,**values):
        opts=dict(crossed_before=False,crossed_now=False,success=False,terminal=False,
                  truncated=False,speed=.5,turn=.15)
        return cost(t,a,b,**(opts|values))
    start=Snapshot(.68,.04,0.)
    rush=step(start,Snapshot(.776,.055,0.))
    align=step(start,Snapshot(.692,.244,0.))
    fall=step(start,Snapshot(.79,.02,0.,hit=True,fall=True),terminal=True)
    stall=step(start,start)
    backing=step(Snapshot(.9,.05,0.),Snapshot(.84,.05,0.))
    crossed=step(Snapshot(1.04,.26,0.),Snapshot(1.09,.27,0.),crossed_now=True)
    repeated=step(Snapshot(1.09,.27,0.),Snapshot(1.08,.27,0.),crossed_before=True,crossed_now=True)
    partial=step(Snapshot(1.6,.06,.14),Snapshot(1.60,.06,.184),
                 crossed_before=True,crossed_now=True)
    meal=step(Snapshot(1.6,.06,.184),Snapshot(1.6,.06,.23),crossed_before=True,crossed_now=True,
              success=True,terminal=True)
    nofood=step(Snapshot(.1,.25,0.,has_food=False),Snapshot(.2,.25,0.,has_food=False),
                crossed_now=True)
    timeout=step(start,Snapshot(.70,.1,0.),truncated=True)
    timeout_reference=step(start,Snapshot(.70,.1,0.))
    assert align.value>stall.value>rush.value>fall.value
    assert backing.unsafe_advance==0. and rush.unsafe_advance<0.
    assert fall.route<=1e-12 and fall.value<-CFG.fall_cost
    assert crossed.crossing==CFG.crossing_bonus and repeated.crossing==0.
    assert partial.food>0. and partial.meal==0. and meal.food==0. and meal.meal==CFG.meal_bonus
    assert nofood.route==nofood.crossing==nofood.meal==0.
    assert timeout==timeout_reference
    return {k:asdict(v)|dict(total=v.value) for k,v in locals().items() if isinstance(v,RewardParts)}
