"""Managed fixed motor-stage pipeline. All child processes collected, fail fast."""
from pathlib import Path
import argparse,json,os,subprocess,sys
HERE=Path(__file__).resolve().parent
PYTHON=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')

def command(argv,label):
    print(json.dumps(dict(phase=label,command=[str(x) for x in argv])),flush=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (HERE/(label+'.log')).open('x') as f:
        result=subprocess.run([str(PYTHON),'-B']+[str(x) for x in argv],stdout=f,stderr=subprocess.STDOUT,env=env)
    print(json.dumps(dict(phase=label,exit_code=result.returncode)),flush=True)
    if result.returncode:raise RuntimeError(label+' failed; inspect '+str(HERE/(label+'.log')))


def evaluate(step,checkpoint):
    reports=[]
    for name,turn in [('neutral',0.),('left',1.),('right',-1.)]:
        out=HERE/f'EVAL_MOTOR_{step}_{name}'
        command([HERE/'evaluate_motor.py','--checkpoint',checkpoint,'--turn',turn,'--out',out],f'eval_motor_{step}_{name}')
        r=json.loads((out/'RESULT.json').read_text())
        eligible=r['autonomous']['all_gates'] and r['autonomous_cycles']['all_legs_completed']
        if turn:eligible=eligible and r['yaw_change_rad']*turn>=.01
        reports.append(dict(name=name,path=str(out),eligible=bool(eligible),gates=r['autonomous']['gates'],cycles=r['autonomous_cycles']['complete_per_leg'],yaw_change_rad=r['yaw_change_rad']))
    return dict(step=step,checkpoint=str(checkpoint),all_conditions_eligible=all(r['eligible'] for r in reports),conditions=reports)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['adapter','finetune'],required=True);args=p.parse_args()
    if args.stage=='adapter':
        step=100;checkpoint=HERE/'TRAIN_ADAPTER_100/checkpoint_000100.pt'
    else:
        trainout=HERE/'TRAIN_FINETUNE_500'
        command([HERE/'train_motor.py','--stage','finetune','--resume',HERE/'TRAIN_ADAPTER_100/checkpoint_000100.pt','--out',trainout],'train_finetune_500')
        step=500;checkpoint=trainout/'checkpoint_000500.pt'
    report=evaluate(step,checkpoint)
    (HERE/f'MOTOR_STAGE_{step}_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
