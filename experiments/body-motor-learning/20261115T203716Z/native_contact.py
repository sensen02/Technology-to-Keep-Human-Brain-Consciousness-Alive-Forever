"""Compact native-step ground contact registry for the standing bench."""
import numpy as np

from bench import stand

DT = stand.physics.NATIVE_DT
W = stand.physics.W
STEPS_PER_MS = 80


class ContactLog:
    def __init__(self, body, steps):
        self.core = body
        self.step = 0
        self.foot = np.zeros((steps, 6, 3))
        self.nonfoot_normal = np.zeros(steps)
        self.trunk_normal = np.zeros(steps)
        self.upright = np.zeros(steps)
        self.com_z = np.zeros(steps)
        self.qpos = np.zeros((steps // STEPS_PER_MS + 1, body.plant.nq))
        self.qvel = np.zeros((steps // STEPS_PER_MS + 1, body.plant.nv))
        self.qpos[0] = body.d.qpos
        self.qvel[0] = body.d.qvel
        self.nonfoot = []
        self.local = np.zeros(6)
        self.ids = body.plant.geom_bodyid.copy()
        self.body_names = [body.plant.body(i).name for i in range(body.plant.nbody)]
        self.geom_names = [body.plant.geom(i).name for i in range(body.plant.ngeom)]
        self.trunk = np.array([n == "Thorax" or n.startswith("A") and n[1:2].isdigit()
                               or n.startswith("Abdomen") for n in self.body_names])
        if self.body_names[body.th] != "Thorax" or self.trunk.sum() < 2:
            raise ValueError("Confirm named thorax and abdominal body identities")

    def before(self, c):
        self.upright[self.step] = c.d.xmat[c.th, 8]
        self.com_z[self.step] = c.d.subtree_com[c.th, 2]

    def solved(self, c, before):
        d = c.d
        contacts = d.contact
        geom = contacts.geom
        planes = c.planes[geom]
        ids = np.flatnonzero(planes[:, 0] ^ planes[:, 1])
        for i in ids:
            item = contacts[i]
            if item.efc_address < 0:
                continue
            ground_first = planes[i, 0]
            other = int(geom[i, 1 if ground_first else 0])
            stand.physics.mj.mj_contactForce(c.plant, d, int(i), self.local)
            if self.local[0] <= 0.:
                continue
            world = item.frame.reshape(3, 3).T @ self.local[:3]
            if not ground_first:
                world = -world
            leg = c.geom_leg[other]
            if leg >= 0:
                self.foot[self.step, leg] += world
            else:
                body_id = self.ids[other]
                normal = max(0., world[2])
                self.nonfoot_normal[self.step] += normal
                if self.trunk[body_id]:
                    self.trunk_normal[self.step] += normal
                self.nonfoot.append((self.step, body_id, other, *item.pos, *world))

    def after(self, c, before, solved):
        self.step += 1
        if self.step % STEPS_PER_MS == 0:
            tick = self.step // STEPS_PER_MS
            self.qpos[tick] = c.d.qpos
            self.qvel[tick] = c.d.qvel

    def arrays(self):
        if self.step != len(self.foot):
            raise RuntimeError("Native capture length mismatch")
        if self.foot[:, :, 2].min() < -1e-10:
            raise RuntimeError("Unexpected negative normal in dry plane support")
        if not np.isfinite(self.foot).all() or not np.isfinite(self.upright).all():
            raise RuntimeError("Nonfinite contact output")
        return dict(foot=self.foot, nonfoot_normal=self.nonfoot_normal, trunk_normal=self.trunk_normal,
                    upright=self.upright, com_z=self.com_z,
                    nonfoot=np.asarray(self.nonfoot).reshape(-1, 9),
                    qpos_ms=self.qpos, qvel_ms=self.qvel,
                    body_names=np.array(self.body_names), geom_names=np.array(self.geom_names),
                    trunk_body=self.trunk)


def spans(mask):
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(a), int(b)) for a, b in changes.reshape(-1, 2)]


def evaluate(native, ms, release_ms):
    foot = native["foot"][:, :, 2] / W
    span = foot.shape[0] // STEPS_PER_MS
    per_tick = (span, STEPS_PER_MS)
    body_touch = native["trunk_normal"] / W
    upright = native["upright"]
    com = native["com_z"]
    contacts = spans(native["nonfoot_normal"] >= .001 * W)
    fall_pose = (upright <= np.cos(np.deg2rad(50.))) | (com <= .95)
    falls = [(a, b) for a, b in spans(fall_pose & (body_touch >= .015))
             if b - a >= 20 * STEPS_PER_MS]
    speed_squared = ((np.diff(ms["com"][:, :2], axis=0) / .001 / stand.LENGTH) ** 2).sum(1)
    support = foot.sum(1).reshape(per_tick).mean(1)
    triple = ((foot >= .02).sum(1) >= 3).reshape(per_tick).mean(1)
    stance = ((upright >= np.cos(np.deg2rad(20.))) & (com >= 1.2)
              & (body_touch < .01)).reshape(per_tick).all(1)
    if len(speed_squared) != span:
        raise ValueError("Native and millisecond traces are not aligned")

    def stable(start, width):
        ticks = slice(start, start + width)
        return bool(speed_squared[ticks].mean() <= .01**2 and support[ticks].mean() >= .9
                    and triple[ticks].mean() >= .95 and stance[ticks].all())

    recovery = next((i - release_ms for i in range(release_ms, span - 99) if stable(i, 100)), None)
    recovered = recovery is not None and stable(span-200, 200)
    behavior = "unrecovered"
    if recovered:
        behavior = "fall_recovered" if falls else "contact_recovered" if contacts else "steady_recovered"
    body_summary = {}
    events = native["nonfoot"]
    for body_id in np.unique(events[:, 1]).astype(int):
        t, *_ = events[events[:, 1] == body_id].T
        normals = np.bincount(t.astype(int), weights=events[events[:, 1] == body_id, 8], minlength=len(foot))
        body_summary[str(native["body_names"][body_id])] = dict(peak_w=float(normals.max() / W),
            time_ms=float((normals >= .001 * W).sum() * DT * 1000))
    return dict(behavior=behavior, recovered=recovered, recovery_ms=recovery if recovered else None,
                contact_intervals_steps=contacts, fall_intervals_steps=falls,
                nonfoot_peak_w=float(native["nonfoot_normal"].max()/W),
                nonfoot_total_ms=float(sum(b-a for a, b in contacts)*DT*1000),
                trunk_peak_w=float(body_touch.max()),
                minimum_com=float(com.min()),
                min_supporting_feet=int((foot >= .02).sum(1).min()),
                nonfoot_bodies=body_summary)
