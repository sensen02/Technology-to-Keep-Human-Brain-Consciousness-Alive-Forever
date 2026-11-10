"""Preregistered independent physical perturbation confirmation; no state writes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

SNAPSHOT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
sys.path.insert(0,str(SNAPSHOT))
import student_wave as sw
from identity import atomic_json,capture,verify,verify_runtime,sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--baseline',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,required=True,choices=[20261111,20261112])
    a=p.parse_args()
    prereg=Path(__file__).with_name('PREREG.json')
    baseline=Path(a.baseline).resolve()
    b=json.loads((baseline/'RESULT.json').read_text())
    bi=json.loads((baseline/'IDENTITY.json').read_text())
    checkpoint=Path(a.checkpoint).resolve()
    if not b['autonomous']['all_gates'] or not b['autonomous_cycles']['all_legs_completed']:
        raise ValueError('nominal candidate did not pass engineering gates and all6 cycles')
    if Path(bi['config']['cli']['checkpoint']).resolve()!=checkpoint:
        raise ValueError('nominal checkpoint mismatch')
    verify(bi)
    if b['control']!='student' or b['sampled'] or not b['transition']['all_exact'] or not b['transition']['no_reset']:
        raise ValueError('baseline must be deterministic student with continuous handover')
    for key,name in [('trace','trace_1ms.npz'),('decisions','decisions_5ms.npz')]:
        if sha256(baseline/name)!=b[key]['sha256']:
            raise ValueError('baseline archive hash mismatch: '+key)
    sw.student_preflight(); teacher_evidence=sw.teacher_preflight(); sw.seed_all(a.seed)
    out=sw.exclusive_out(a.out)
    c=sw.Core(adhesion=False); model,payload=sw.load_student(checkpoint,c)
    ident=capture(c,dict(kind='predeclared_perturbation_confirmation',cli=vars(a),seed=a.seed,
        pulse_before_settle=True,settle_s=1.,teacher_s=4.8,autonomous_s=9.6),
        extra_paths=sw.SOURCE_FILES+[SNAPSHOT/'STUDENT_CHECKS.json',Path(__file__).resolve(),prereg,checkpoint,baseline/'RESULT.json',baseline/'IDENTITY.json',baseline/'trace_1ms.npz',baseline/'decisions_5ms.npz']+[Path(x) for x in payload['identity']['files']]+[Path(x) for x in teacher_evidence['files']])
    atomic_json(out/'IDENTITY.json',ident)
    trace=sw.Trace(full=True)
    previous,pulse=sw.perturb(c,trace,1,a.seed)
    sw.settle(c,trace)
    teacher=sw.StaticTeacher(c,sw.teacher_config(3))
    run=sw.continuous_handover(c,teacher,model,trace,4800,9600,control='student',sampled=False)
    decisions=run.pop('decisions')
    path=out/'decisions_5ms.npz'
    np.savez_compressed(path,**{k:np.asarray([r[k] for r in decisions]) for k in decisions[0]})
    report=dict(control='student',sampled=False,seed=a.seed,perturbation=pulse,
        teacher_reference_evidence=teacher_evidence,algorithm_prefix=dict(settle_s=1.,walk_s=4.8,teacher_metadata=teacher.metadata),
        autonomous=trace.summary(c,3),teacher_warmup=trace.summary(c,2),**run,
        trace=trace.save(out/'trace_1ms.npz'),decisions=dict(path=str(path),sha256=sha256(path)),
        interpretation='physical perturbation confirmation only; same 9.6s ANN window and engineering gates; native original acceptance separate')
    verify(ident);verify_runtime(payload['identity'],c)
    atomic_json(out/'RESULT.json',report)
    print(json.dumps(dict(out=str(out),seed=a.seed,gates=report['autonomous']['all_gates'],cycles=report['autonomous_cycles']['complete_per_leg'],
        slip=report['autonomous']['slip_ratio_max'],forward=report['autonomous']['forward_signed_native_per_s'],COM=report['autonomous']['com_height_min'])),flush=True)

if __name__=='__main__':main()
