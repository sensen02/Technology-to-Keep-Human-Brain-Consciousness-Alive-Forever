"""Deploy the learned CPG readout on the same native physics and neck."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import math
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import torch

from phase import HERE, Oscillator, read_policy, study
import body_run as first
from cases import HELD as OLD_HELD


@torch.inference_mode()
def rollout(case, mode, checkpoint, out):
    started = perf_counter()
    torch.set_num_threads(1)
    body = study.Body()
    body.at_start(case.seed)
    old_case = case.name in {c.name for c in OLD_HELD}
    neck = first.Neck() if old_case else study.Neck(case)
    oscillator = Oscillator(body)
    policy = read_policy(checkpoint)
    trace = np.empty((study.TICKS,15))
    emitted = np.empty((study.TICKS,len(neck.roots)),dtype=np.uint8)
    arrived = np.empty_like(emitted)
    observations, labels, controls, positions = [], [], [], []
    for tick in range(study.TICKS):
        contacts = body.contact()
        found, forces = contacts
        front_forces = np.linalg.norm(forces[body.fronts],axis=1)
        selector = case if old_case else case.pulse_every
        emitted[tick],arrived[tick] = neck.step(tick,selector,front_forces)
        if mode == 'correct':
            label = body.clip_action(body.teacher(neck.command))
        else:
            oscillator.step(neck.command)
        if tick % study.HOLD == 0:
            local = body.observation(neck.command, contacts)
            phase = oscillator.features()
            estimate = body.clip_action(body.neutral+policy.act(local,phase,mode!='gait'))
            if mode == 'correct':
                control = .52*estimate+.48*label
                labels.append((label-body.neutral).astype(np.float32))
            else:
                control = estimate
            observations.append(local)
            controls.append(control.copy())
            positions.append(body.d.qpos.copy())
        body.d.xfrc_applied[body.thorax,1] = .12*body.weight*case.push_side if 6500<=tick<6700 else 0
        body.apply(control)
        rot = body.d.xmat[body.thorax]
        trace[tick] = [(tick+1)*study.DT,*body.d.xpos[body.thorax],rot[8],
                       math.atan2(rot[3],rot[0]),*neck.command,body.nonfoot(),*found]
    assert not body.d.warning.number.any() and np.isfinite(trace).all()
    out.mkdir()
    record = out/'TRACE.npz'
    np.savez_compressed(record,trace=trace,emitted=emitted,arrived=arrived,
                        x=np.asarray(observations),y=np.asarray(labels),ctrl=np.asarray(controls),qpos=np.asarray(positions))
    result=first.score(case,trace) if old_case else study.score(case,trace)
    result.update(case=asdict(case),mode=mode,weight_sha256=study.sha(checkpoint),
                  body_snapshot_sha256=study.sha(study.MODEL_FILE),catalog_sha256=study.sha(study.PARENT/'CATALOG.json'),
                  arrival_roots=neck.roots,probes=neck.receiver.manifest(),
                  cut_roots=[] if old_case else [root for root, cut in zip(neck.roots,neck.cut,strict=True) if cut],
                  trace_sha256=study.sha(record),seconds=perf_counter()-started)
    study.write(out/'RESULT.json',result)
    print(case.name, result['pass_'], round(result['active_speed_mm_s'],3), result.get('yaw_degrees'),
          {k:v for k,v in result['gates'].items() if not v},flush=True)


def batch(pool_name,mode,source,out,jobs):
    tasks={'dev':study.DEVELOP,'held':(*study.HELD,study.CUT),'old':OLD_HELD,'correct':study.CORRECT}
    out = out.resolve()
    out.mkdir()
    def run(case):
        with (out/f'{case.name}.log').open('x') as log:
            cmd=['bash',str(study.HERE/'drive.sh'),'PHASE/phase_rollout.py',mode,case.name,str(source.resolve()),str(out/case.name)]
            code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
        (out/f'{case.name}.exit').write_text(f'{code}\n')
        print(case.name,code,flush=True)
        return code
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return int(any(tuple(pool.map(run,tasks[pool_name]))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('gait','evaluate','correct'))
    parser.add_argument('target',type=str)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--jobs',type=int,default=9)
    arg=parser.parse_args()
    if arg.target in ('dev','held','old','correct'):
        raise SystemExit(batch(arg.target,arg.mode,arg.checkpoint,arg.out,arg.jobs))
    cases={**study.CASES,**{c.name:c for c in OLD_HELD}}
    rollout(cases[arg.target],arg.mode,arg.checkpoint,arg.out)
