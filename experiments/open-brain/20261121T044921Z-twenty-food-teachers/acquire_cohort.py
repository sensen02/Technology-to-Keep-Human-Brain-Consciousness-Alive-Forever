"""Collect eighteen independently validated physical teacher meals, serially."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from cohort import BANDS, SEEDS, scenario

HERE = Path(__file__).resolve().parent
FLIGHT = HERE.parents[2]/'experiments/open-brain/20261117T051249Z-bidirectional/flight_env/bin/python'
PROGRESS = HERE/'cohort-progress.json'
TARGET = 18
ASSETS = ('RESULT.json','BODY.npz','NECK.npz','floor.npz','external.npz','electrode.npz')


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def execute(argv, log):
    with log.open('a') as output:
        return subprocess.run(argv, cwd=HERE, stdout=output, stderr=subprocess.STDOUT).returncode


def verify_physical(dest):
    info=read(dest/'RESULT.json')
    assert info['first_fall_s'] is None or not info['meal_pass']
    assert abs(info['stock_loss']-info['intake'])<1e-9
    with np.load(dest/'BODY.npz') as archive:
        ledger=archive['meal']
    increments=ledger[:,4]
    assert (increments>=0).all() and not np.any((increments>0)&(ledger[:,3]==0))
    assert abs(increments.sum()-info['intake'])<1e-9
    return info


def prep(dest, seed):
    log=HERE/'logs'/f'sensors-{seed}.log'
    if not (dest/'external.npz').is_file():
        rc=execute(['nice','-n','13',str(FLIGHT),'-Bu','prepare.py',str(dest)],log)
        if rc:
            raise RuntimeError(f'Camera or force preparation failed for {seed}. See {log}')
    if not (dest/'electrode.npz').is_file():
        rc=execute(['nice','-n','13','bash','env.sh','voltages.py',str(dest)],log)
        if rc:
            raise RuntimeError(f'Voltage preparation failed for {seed}. See {log}')
    with np.load(dest/'external.npz') as obs, np.load(dest/'electrode.npz') as el:
        valid=obs['valid']
        assert len(valid)==len(el['counts']) and len(valid)>=100
        assert valid[:2].all() and valid.mean()>=.98 and np.isfinite(obs['force_uN']).all()
        assert el['counts'].shape[1]==17
        return len(valid),int(valid.sum())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--accepted-limit',type=int,default=TARGET)
    limit=p.parse_args().accepted_limit
    assert 1<=limit<=TARGET
    if PROGRESS.exists():
        manifest=read(PROGRESS)
    else:
        manifest=dict(target=TARGET,original_teacher_seeds=[88207,88209],accepted=[],attempted=[])
    tried={x['seed'] for x in manifest['attempted']}
    band_counts=Counter(row['band'] for row in manifest['accepted'])
    side_counts=Counter((row['band'],row['side']) for row in manifest['accepted'])
    while len(manifest['accepted'])<limit:
        options=[]
        for band in BANDS:
            start,end=SEEDS[band]
            # Each distance band contributes equal weight; within it alternate sides.
            for seed in range(start,end):
                if seed in tried:continue
                sc=scenario(seed)
                side=1 if sc.goal_y>0 else -1
                options.append((band_counts[band],side_counts[(band,side)],seed,band,side,sc))
        if not options:
            raise RuntimeError('All predeclared scenarios exhausted.')
        _,_,seed,band,side,sc=min(options,key=lambda item:item[:3])
        dest=HERE/f'aligned-{seed}'
        print('starting',seed,band,'accepted',len(manifest['accepted']),flush=True)
        if not ((dest/'RESULT.json').is_file() and (dest/'floor.npz').is_file()):
            rc=execute(['bash','physical_one.sh',str(seed)],HERE/'logs'/f'teacher-{seed}.log')
            if rc:
                raise RuntimeError(f'Teacher process {seed} exited with {rc}. Resume after inspecting its log.')
        info=verify_physical(dest)
        result=dict(seed=seed,band=band,side=side,goal_mm=[sc.goal_x*10,sc.goal_y*10],
                    initial_yaw_deg=sc.yaw_deg,intake=info['intake'],
                    duration_s=info['duration_s'],native_steps=info['native_steps'],
                    ended_standing=info['ended_standing'],first_bite_s=info['first_intake_s'])
        if info['intake']>=.18 and info['ended_standing'] and info['first_fall_s'] is None:
            n,valid=prep(dest,seed)
            result.update(status='accepted',observed_frames=n,valid_frames=valid,
                          hashes={name:digest(dest/name) for name in ASSETS})
            manifest['accepted'].append(result)
            band_counts[band]+=1
            side_counts[(band,side)]+=1
        else:
            result['status']='teacher_below_gate'
        manifest['attempted'].append(result)
        tried.add(seed)
        write(PROGRESS,manifest)
        print(seed,result['status'],'intake',round(info['intake'],5),'and standing',info['ended_standing'],
              'accepted',len(manifest['accepted']),flush=True)
    if len(manifest['accepted'])==TARGET:
        write(HERE/'cohort.json',manifest)


if __name__=='__main__':
    main()
