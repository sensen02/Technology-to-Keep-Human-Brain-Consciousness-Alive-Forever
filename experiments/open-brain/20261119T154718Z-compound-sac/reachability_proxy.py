"""Check if the existing action envelope admits supported compound routes in the proxy."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from train_timed import HERE, HORIZON, new_env
from bridge_reward import CFG


def fork(parent):
    clone = copy.copy(parent)
    clone.position = parent.position.copy()
    clone.contacts = parent.contacts.copy()
    clone.risk_memory = parent.risk_memory.copy()
    clone.rng = copy.copy(parent.rng)
    return clone


def score(env):
    terrain = env.terrain
    gate = env.active_gate()
    target = terrain.goal.copy()
    if gate is not None:
        sign = gate.side
        lane = sign*(gate.edge+CFG.foot_margin_cm+.045)
        if env.position[0] < gate.near-.08:
            target = np.array([gate.near-.08,lane])
        else:
            target = np.array([gate.far+.085,lane])
    delta = target - env.position
    reference = np.array([np.cos(env.heading),np.sin(env.heading)])
    distance = float(np.linalg.norm(delta))
    heading_cost = .14*(1.-(reference@delta)/max(1e-8,distance))
    excess = 0.
    if gate is not None:
        excess = max(0., (gate.edge+CFG.foot_margin_cm+.025)-gate.side*env.position[1])
        excess *= float(np.clip((env.position[0]-gate.near+.24)/.24,0.,1.))
    return 4*env.cross_count + 5*env.ingested - distance - heading_cost - .75*excess - .12*env.hits


def candidate(env, sense, action, previous_path):
    state = fork(env)
    next_observation, _, terminal, truncated, info = state.advance_actor(sense,action)
    # Once an attempt has used less margin, this planner must back out
    # completely to start a fresh crossing. Keep safe candidates in the beam.
    unsafe = state.min_margin is not None and state.min_margin < CFG.foot_margin_cm
    if info['fall'] or info['invalid'] or unsafe or (terminal and not info['success']):
        return None
    point = tuple(state.position.tolist())
    path = previous_path+(point,)
    return score(state), state, next_observation, path, info


def trial(seed, stage, side):
    env = new_env()
    sense = env.reset(seed,stage,side,food=True)
    actions = [np.array((v,w),dtype=np.float32)
               for v in (-.95,.20,.96) for w in (-1.,-.48,0.,.48,1.)]
    states = [(score(env),env,sense,(tuple(env.position.tolist()),),None)]
    best = states[0]
    for _ in range(HORIZON):
        possibilities = (candidate(parent,x,act,path)
                         for _,parent,x,path,_ in states for act in actions)
        ordered = sorted((c for c in possibilities if c is not None),
                         reverse=True,key=lambda row: row[0])
        if not ordered:
            break
        best = ordered[0]
        passed = next((n for n in ordered if n[-1]['success']),None)
        if passed:
            best = passed
            break
        states = []
        cells = set()
        for row in ordered:
            pos = row[1].position
            cell = (round(pos[0]/.029),round(pos[1]/.036),round(row[1].heading/.11),row[1].gate_index)
            if cell not in cells:
                states.append(row)
                cells.add(cell)
            if len(states) == 22:
                break
    return dict(seed=seed,stage=stage,side=side,score=best[0],result=best[-1],trajectory_cm=best[3])


def run(out):
    results = []
    for seed,stage in ((27261,'slope_bridge'),(27263,'staggered'),(27267,'staggered_slope')):
        for side in (-1,1):
            row = trial(seed,stage,side)
            results.append(row)
            print(stage,side,row['result'],flush=True)
    out.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path,default=HERE/'REACHABILITY_PROXY.json',nargs='?')
    run(parser.parse_args().out)
