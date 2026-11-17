"""Search for whole-body righting trajectories without changing the root state."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import mujoco as mj
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "20261115T203716Z"
sys.path.insert(0, str(PREVIOUS))
from snapshots import capture, load, model, stand
from native_contact import ContactLog, W, spans, evaluate

INITIALS = PREVIOUS / "RUN/RIGHTING_SEEDS_VERIFIED"
STARTS = ("side_fallen", "near_supine")
ZERO = np.zeros(42)


def read(file):
    return json.loads(file.read_text())


def carray(array):
    return ct.c_void_p(array.ctypes.data)


@dataclass
class Point:
    state: np.ndarray
    target: np.ndarray
    qpos: np.ndarray
    goals: tuple
    history: tuple
    score: float
    furthest: float


class Engine:
    def __init__(self, body, dt=None, exact=False):
        self.m = body.plant if dt is None else load_plant(dt)
        self.th = body.th
        self.exact = exact
        a = body.actuator
        self.params = [np.ascontiguousarray(x) for x in
                       (a.qpos_addresses.astype(np.intc), a.dof_addresses.astype(np.intc),
                        a.kp, a.kd, a.torque_limit, a.target_bounds_rad)]
        self.foot_ids = np.ascontiguousarray(body.geom_leg, dtype=np.intc)
        self.foots = [body.pads[leg] for leg in stand.physics.LEG_ORDER]
        self.bounds = a.target_bounds_rad
        self.qa = a.qpos_addresses
        self.va = a.dof_addresses
        self.spec = mj.mjtState.mjSTATE_INTEGRATION
        self.dll = ct.CDLL(str(HERE / "rollout.so"))
        self.drive = self.dll.drive
        self.drive.restype = ct.c_int
        self.drive.argtypes = ([ct.c_void_p] * 3 + [ct.c_int] + [ct.c_void_p] * 7
                               + [ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p])

    def sim(self, point, goals):
        data = mj.MjData(self.m)
        mj.mj_setState(self.m, data, point.state, self.spec)
        mj.mj_forward(self.m, data)
        target = point.target.copy()
        trace = np.empty((len(goals), 10))
        code = self.drive(self.m._address, data._address, carray(goals), 5 * len(goals),
                          carray(target), *(carray(a) for a in self.params),
                          int(not self.exact), self.th, carray(self.foot_ids), carray(trace))
        if code or np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
            raise ValueError(f"Invalid screening rollout ({code})")
        state = np.empty_like(point.state)
        mj.mj_getState(self.m, data, state, self.spec)
        recent = trace[-10:]
        z = recent[:, 0]
        height = recent[:, 1]
        feet = (recent[:, 4:] >= .02 * W).sum(axis=1)
        score = float(3.4*z.mean() + 1.55*height.mean() + .12*feet.mean()
                      - .005*recent[:, 2].mean() - .012*recent[:, 3].mean())
        reached = float((trace[:, 0] + .36*trace[:, 1]).max())
        return Point(state, target, data.qpos.copy(), (*point.goals, goals), (*point.history, trace),
                     score, max(point.furthest, reached))

    def ik_choices(self, point):
        data = mj.MjData(self.m)
        mj.mj_setState(self.m, data, point.state, self.spec)
        mj.mj_forward(self.m, data)
        if data.xmat[self.th, 8] > .88:
            return []
        starts = np.array([data.geom_xpos[geoms].mean(0) for geoms in self.foots])
        jac = np.zeros((3, self.m.nv))
        current = point.target
        center = data.xpos[self.th].copy()
        across = data.xmat[self.th].reshape(3,3)[:, 1].copy()
        across[2] = 0.
        across /= max(np.linalg.norm(across), 1e-6)
        choices = []
        for which in (np.arange(6), np.arange(3), np.arange(3,6)):
            for side in (-.5, 0., .5):
                targets = starts.copy()
                targets[which] += side*across
                targets[which, 2] -= np.clip(starts[which, 2]-.045, .10, 1.6)
                q = current.copy()
                for _ in range(9):
                    data.qpos[self.qa] = q
                    mj.mj_fwdPosition(self.m, data)
                    for leg in which:
                        names = self.foots[leg]
                        col = slice(7*leg,7*(leg+1))
                        site = data.geom_xpos[names].mean(0)
                        body = int(self.m.geom_bodyid[names[0]])
                        mj.mj_jac(self.m, data, jac, None, site, body)
                        j = jac[:, self.va[col]]
                        dq = j.T @ np.linalg.solve(j @ j.T + .07*np.eye(3), targets[leg]-site)
                        q[col] += np.clip(.65*dq, -.20,.20)
                    q = np.clip(q, self.bounds[:, 0], self.bounds[:, 1])
                choices.append(q)
        return choices


def load_plant(dt):
    core = stand.physics.Core(adhesion=False)
    m = core.plant
    m.opt.timestep = dt
    if not np.isclose(.001/dt, round(.001/dt), atol=1e-11, rtol=0.):
        raise ValueError("Screening step must divide the actuator period")
    return m


def seeds(bench):
    bank = read(INITIALS / "BANK.json")["seeds"]
    p = {}
    for key in STARTS:
        load(bench, INITIALS / f"{key}.npz", bank[key]["snapshot_sha256"])
        s = capture(bench)
        if np.any(bench.body.d.xfrc_applied) or np.any(bench.body.d.qfrc_applied):
            raise ValueError("Nonzero disturbance in the righting seed")
        p[key] = Point(s["state"], s["act_target_rad"], s["qpos"], (), (), -np.inf, -np.inf)
    return p


def random_goal(engine, point, home, gen, index):
    current = point.target
    if index == 0:
        return home
    if index == 1:
        return current
    if index <= 7:
        q = current.copy().reshape(6,7)
        selected = (index-2)%6
        q[selected] += gen.uniform(-.85,.85,7)
        return np.clip(q.ravel(),engine.bounds[:,0],engine.bounds[:,1])
    center = current if index % 3 else .6*current + .4*home
    magnitude = gen.uniform(.24,1.4)
    pattern = gen.uniform(-1.,1.,(6,7))
    if index % 5 in (0,1):
        pattern[3:] = pattern[:3] * (-1. if index % 5 else 1.)
    elif index % 5 == 2:
        pattern[:3] = 0.
    elif index % 5 == 3:
        pattern[3:] = 0.
    return np.clip(center + magnitude * pattern.ravel(), engine.bounds[:,0], engine.bounds[:,1])


def constant(goal, ticks):
    return np.broadcast_to(goal, (ticks//5, 42)).copy()


def diversity(points, width):
    best = []
    for point in sorted(points, key=lambda p:p.score, reverse=True):
        if all(np.linalg.norm(chosen.target-point.target) > .25
               or np.linalg.norm(chosen.qpos[3:7]-point.qpos[3:7]) > .08 for chosen in best):
            best.append(point)
        if len(best) == width:
            break
    return best


def metrics(p, dt, window=20):
    trace = p.history[-1][-window:]
    return dict(score=p.score, end_z=float(trace[-1,0]), end_com=float(trace[-1,1]),
                end_foot_w=(trace[-1,4:]/W).tolist(),
                best_tilt_deg=float(np.rad2deg(np.arccos(np.clip(np.concatenate(p.history)[:,0].max(),-1,1)))),
                elapsed_ms=sum(len(row)*5 for row in p.history), native_tick_s=dt)


def search(out, width, count, depths, ticks, dt, workers):
    out.mkdir()
    bench = stand.Bench(out)
    initial = seeds(bench)
    engine = Engine(bench.body, dt=dt)
    home = bench.target
    gen = np.random.default_rng(202611152127)
    beam = {k:[v] for k,v in initial.items()}
    seen = {k:[] for k in initial}
    batches = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for level in range(depths):
            jobs = []
            for name, options in beam.items():
                for j, point in enumerate(options):
                    goals = [random_goal(engine,point,home,gen,i) for i in range(count if j==0 else count//2)]
                    goals.extend(engine.ik_choices(point))
                    jobs.extend((name, point, constant(q, ticks)) for q in goals)
            start = time.monotonic()
            candidates = {k:[] for k in initial}
            for (name, p, q), result in zip(jobs, pool.map(lambda job:engine.sim(job[1],job[2]), jobs),strict=True):
                candidates[name].append(result)
            status = {}
            for name, items in candidates.items():
                beam[name] = diversity(items, width)
                seen[name] = diversity([*seen[name], *beam[name]], width)
                top = beam[name][0]
                status[name] = metrics(top, dt)
                chosen = out/f"{name}_depth{level+1:02d}.npz"
                np.savez_compressed(chosen, goals=np.concatenate(top.goals), trace=np.concatenate(top.history))
                print(name, "depth",level+1,status[name],flush=True)
            batches.append(dict(level=level+1, candidates=len(jobs), status=status,
                                seconds=time.monotonic()-start))
            stand.put(out / f"DEPTH_{level+1:02d}.json",batches[-1])
    for name, chosen in seen.items():
        for i,p in enumerate(chosen):
            np.savez_compressed(out/f"{name}_best{i:02d}.npz", goals=np.concatenate(p.goals), trace=np.concatenate(p.history))
    stand.identity.verify_runtime(bench.manifest, bench.body)
    stand.put(out / "SEARCH.json", dict(batches=batches, rollout_library=stand.sha(HERE/"rollout.so"),
                                     ticks_per_stage=ticks, seeds=read(INITIALS/"BANK.json")["seeds"]))


def preflight(out):
    out.mkdir()
    bench = stand.Bench(out)
    p = seeds(bench)
    p = p["near_supine"]
    baseline = Engine(bench.body, exact=True)
    fast = Engine(bench.body)
    coarse = Engine(bench.body, dt=50.e-6)
    q = constant(np.clip(p.target + np.random.default_rng(1).uniform(-.5,.5,42),
                         baseline.bounds[:,0],baseline.bounds[:,1]),300)
    results = {}
    points = {}
    for name, runner in (("exact_cpp",baseline),("skipped_forward",fast),("coarse",coarse)):
        start = time.monotonic()
        v = runner.sim(p,q)
        results[name] = dict(**metrics(v,runner.m.opt.timestep), wall=time.monotonic()-start)
        points[name]=v
    load(bench, INITIALS/"near_supine.npz",read(INITIALS/"BANK.json")["seeds"]["near_supine"]["snapshot_sha256"])
    c = bench.body
    start = time.monotonic()
    for tick in range(len(q)*5):
        if tick%5==0:
            rate=np.clip((q[tick//5] - c.actuator.target_rad)/.005, -2., 2.)
        c.tick_rate(rate)
    results["exact_python"]=dict(wall=time.monotonic()-start)
    for name,v in points.items():
        results[name]["qpos_maxdiff_vs_python"]=float(np.max(np.abs(v.qpos-c.d.qpos)))
        results[name]["target_maxdiff_vs_python"]=float(np.max(np.abs(v.target-c.actuator.target_rad)))
    stand.put(out/"PREFLIGHT.json",results)
    print(results,flush=True)


def main():
    a=argparse.ArgumentParser()
    a.add_argument("mode", choices=("preflight","search"))
    a.add_argument("output",type=Path)
    a.add_argument("--width",type=int,default=3)
    a.add_argument("--count",type=int,default=32)
    a.add_argument("--depth",type=int,default=5)
    a.add_argument("--ticks",type=int,default=300)
    a.add_argument("--dt",type=float,default=50.e-6)
    a.add_argument("--workers",type=int,default=12)
    args=a.parse_args()
    torch.set_num_threads(1)
    if args.mode=="preflight":
        preflight(args.output.resolve())
    else:
        search(args.output.resolve(),args.width,args.count,args.depth,args.ticks,args.dt,args.workers)


if __name__=="__main__":
    main()
