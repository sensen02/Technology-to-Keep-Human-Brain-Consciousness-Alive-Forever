"""Navigation reward for a 20 ms actor, computed exclusively by the environment."""
from dataclasses import dataclass
from math import cos, isfinite, radians

from stall import COSTS as STALL_COSTS, Motion, Stall

GAMMA = .99955
ACTOR_S = .020
WEIGHTS = dict(success=100., fall=-120., obstacle_contact=-80., overshoot=-45., deadline=-28.)
WEIGHTS.update({f"stalled_{name}":cost[1] for name,cost in STALL_COSTS.items()})
CONTACT_W = 1.e-6
PROGRESS_PER_UNIT = 4.
PROGRESS_MAX = 16.
UNSAFE_ENDINGS = ("fall", "obstacle_contact", "overshoot")


@dataclass(frozen=True)
class Task:
    start_distance: float
    band: tuple[float,float]
    must_clear_course: bool
    may_step_on_pit_bottom: bool = False
    require_motion: bool = False


@dataclass(frozen=True)
class Measurement:
    distance: float
    bearing_deg: float
    com_xy_speed: float
    upright: float
    com_z: float
    support_W: float
    supporting_feet: int
    yaw_speed_deg_s: float = 0.
    nonfoot_W: float = 0.
    obstacle_peak_W: float = 0.
    pit_bottom_peak_W: float = 0.
    pit_body_peak_W: float = 0.
    cleared: bool = False
    neutral_command: bool = False
    navigation_active: bool = True
    motion: Motion | None = None


class Reward:
    def __init__(self, task: Task):
        self.task = task
        low,high = task.band
        if not (0<low<high<task.start_distance and isfinite(task.start_distance)):
            raise ValueError("The goal band must be valid and outside the starting radius.")
        self.settle_s = 0.
        self.unstable_s = 0.
        self.toe_spent = 0.
        self.best_goal_error = task.start_distance-high
        self.progress_paid = 0.
        self.stall = Stall()
        self.ended = False

    def potential(self, distance):
        low,high=self.task.band
        outside=max(low-distance,distance-high,0.)
        return -6.*min(outside/6.,1.)

    def step(self, d_before, m: Measurement, *, dt=ACTOR_S, deadline=False, truncated=False):
        """True task deadline ends the episode. Rollout cuts retain the bootstrap."""
        if self.ended or not (0.<dt<=ACTOR_S) or (deadline and truncated):
            raise ValueError("Invalid reward transition.")
        if self.task.require_motion and m.navigation_active and m.motion is None:
            raise ValueError("An active navigation step requires physical motion telemetry.")
        raw=(d_before,m.distance,m.bearing_deg,m.com_xy_speed,m.upright,m.com_z,m.support_W,
             m.yaw_speed_deg_s,m.nonfoot_W,m.obstacle_peak_W,m.pit_bottom_peak_W,m.pit_body_peak_W)
        if not all(isfinite(v) for v in raw):
            raise ValueError("Invalid physical measurement.")
        low,high=self.task.band
        standing = (m.navigation_active and low<=m.distance<=high and abs(m.bearing_deg)<=15. and
                    m.com_xy_speed<=.01 and m.yaw_speed_deg_s<=2.5 and m.upright>=cos(radians(20.)) and
                    m.com_z>=1.2 and m.support_W>=.90 and m.supporting_feet>=3 and
                    m.nonfoot_W<.01 and m.neutral_command and
                    (not self.task.must_clear_course or m.cleared))
        self.settle_s=self.settle_s+dt if standing else 0.
        unstable=m.com_z<.98 or m.upright<cos(radians(50.))
        self.unstable_s=self.unstable_s+dt if unstable else 0.
        stall_kind, stall_cost, stuck = self.stall.update(m.motion, dt, active=m.navigation_active and not standing)
        tag=None
        if m.pit_body_peak_W>=.005 or self.unstable_s>=.060-1e-12:
            tag="fall"
        elif m.obstacle_peak_W>CONTACT_W:
            tag="obstacle_contact"
        elif m.distance<low-.15 and m.com_xy_speed>.01:
            tag="overshoot"
        elif self.settle_s>=.600-1e-12:
            tag="success"
        elif stuck:
            tag=f"stalled_{stall_kind}"
        elif deadline:
            tag="deadline"

        event=WEIGHTS[tag] if tag else 0.
        if tag=="deadline":
            advance=(self.task.start_distance-m.distance)/max(self.task.start_distance-high,.1)
            event+=5.*max(0.,min(advance,1.))
        toe_contact=m.pit_bottom_peak_W>CONTACT_W
        toe_cost=0.
        if toe_contact and tag!="fall":
            cap, per_step = (2., .012) if self.task.may_step_on_pit_bottom else (3., .015)
            toe_cost=min(max(cap-self.toe_spent, 0.), per_step*dt/ACTOR_S*min(m.pit_bottom_peak_W/.35,1.))
            self.toe_spent+=toe_cost

        progress=0.
        if m.navigation_active:
            before=max(d_before-high,0.)
            after=max(m.distance-high,0.)
            prior=min(self.best_goal_error,before)
            self.best_goal_error=min(prior,after)
            if tag not in UNSAFE_ENDINGS:
                advance=prior-self.best_goal_error
                progress=min(max(PROGRESS_MAX-self.progress_paid,0.),PROGRESS_PER_UNIT*advance)
                self.progress_paid+=progress

        self.ended=bool(tag or truncated)
        next_phi=0. if tag else self.potential(m.distance)
        parts=dict(event=event,time=-.005*dt/ACTOR_S,
                   potential=GAMMA**(dt/ACTOR_S)*next_phi-self.potential(d_before),
                   toe=-toe_cost,stall=-stall_cost,progress=progress)
        return dict(reward=sum(parts.values()),terms=parts,reason=tag,
                    stall_kind=stall_kind,stall_seconds=self.stall.duration,pit_bottom_contact=toe_contact,
                    terminated=bool(tag),truncated=bool(truncated and not tag))
