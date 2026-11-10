"""Direct selected/zero artifact equality; no neural or physical calls."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
EVAL=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning/evaluation')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    results={}
    for name in ['ACTOR.npz','INITIAL.npz','NEURAL_INITIAL.npz','NEURAL_FINAL.npz','POLICY.npz','BODY.npz','NATIVE.npz','COMMANDS.npz','SLIP.npz','SCORE_INPUTS.npz']:
        a,b=EVAL/'selected'/name,EVAL/'zero'/name
        with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
            keys=sorted(x.files);assert keys==sorted(y.files)
            fields={k:bool(np.array_equal(x[k],y[k])) for k in keys}
        results[name]={'all_fields_exact':all(fields.values()),'fields':fields,'selected_sha256':sha(a),'zero_sha256':sha(b),'file_bytes_identical':sha(a)==sha(b)}
        assert results[name]['all_fields_exact']
    # Both independently solved neural replay finals and every-step histories also match.
    for suffix in ['NEURAL_REPLAY_FINAL.npz','EVERY_STEP.npz']:
        a,b=HERE/('selected_'+suffix),HERE/('zero_'+suffix)
        with np.load(a) as x,np.load(b) as y:assert set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
        results['replay_'+suffix]={'all_fields_exact':True,'selected_sha256':sha(a),'zero_sha256':sha(b)}
    out={'status':'PASS_SELECTED_IS_EXACT_ZERO_TRAJECTORY','neural_steps':0,'physical_steps':0,'scratch_forward_calls':0,'optimizer_updates':0,'artifacts':results,'meaning':'selected theta iszero;body/command/neural trajectories identical;no learnednonzero policy or walking claim'}
    (HERE/'CONDITION_EQUALITY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'artifacts_compared':len(results),'neural_steps':0,'physical_steps':0},indent=2))
if __name__=='__main__':main()
