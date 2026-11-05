"""S5 audited release-boundary interventions; no controller or new connections."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MUJOCO_GL','egl')
os.environ['PCR_SYNTH']='1'
import sys,json,hashlib,time,dataclasses
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R),str(R/'tools')]
SOURCE=Path('/home/sensen/Desktop/engine修复验收/s5-route-a-substitute-v4/20261108T200000Z')
def dump(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=Path(sys.argv[1]); condition=sys.argv[2]
    out.mkdir(parents=True,exist_ok=False)
    import train_per_cell_readout as pcr
    import engine.embodied.continuous_handover as h
    pcr.O=out;pcr.SET=out/'CONNECTIVITY_SET'
    # make_set creates read-only input links; no writes into linked RUN are allowed.
    protected=[SOURCE/'BEST_VEC.json',pcr.REBUILD/'REBUILT_We.npz',pcr.REBUILD/'REBUILT_Wi.npz']
    import train_stage_a_pool_modes as st
    protected += [st.B/'RUN/MAPPING.json',st.B/'RUN/MODEL.xml',st.B/'IDENTITIES.npz']
    before={str(p):sha(p) for p in protected};dump(out/'INPUT_HASHES_BEFORE.json',before)
    pcr.make_set()
    (out/'LAYOUT.json').write_bytes((SOURCE/'LAYOUT.json').read_bytes())
    seed=json.loads((SOURCE/'BEST_VEC.json').read_text())
    original=h.run_neural_continuous
    def continuous(model,data,*args,**kw):
        state=kw.get('runtime_state')
        if state is not None:
            net,enc,dec=state['runtime']
            mapped=sorted({int(i) for row in dec.manifest['rows'] for i in row['local_indices']})
            # Target identities were preregistered from baseline, not selected per outcome.
            cfg=json.loads((out.parent/'PROBE_PREREGISTRATION.json').read_text()) if condition!='baseline' else {}
            cells=np.array(cfg.get('target_cells',[]),int)
            old=net.threshold_offset_mV.copy()
            shift={'baseline':0.,'unmask':0.,'unmask_minus2':-2.,'unmask_minus5':-5.}[condition]
            assert all(int(i) in mapped for i in cells)
            new=old.copy();new[cells]+=shift;new.flags.writeable=False;net.threshold_offset_mV=new
            removed=[]
            if condition!='baseline':
                c=dec.config;indices=[];sources=[];weights=[];pos=0
                for ai,src in zip(c.synthetic_index,c.synthetic_sources):
                    w=c.synthetic_weights[pos:pos+len(src)];pos+=len(src)
                    if dec.groups[ai].size:removed.append(int(ai))
                    else:indices.append(ai);sources.append(src);weights.extend(w)
                dec.config=dataclasses.replace(c,synthetic_index=tuple(indices),synthetic_sources=tuple(sources),synthetic_weights=tuple(weights)).validate()
            dump(out/'INTERVENTION.json',dict(condition=condition,absolute_time_s=float(data.time),target_cells=cells,threshold_shift_mV=shift,threshold_old=old[cells],threshold_new=new[cells],synthetic_removed=removed,pending_unchanged=True,first_new_command_delay_s=.001))
            np.savez_compressed(out/'BOUNDARY_STATE.npz',qpos=data.qpos,qvel=data.qvel,activation=data.act,pending=state['pending'],threshold_before=old)
        return original(model,data,*args,**kw)
    h.run_neural_continuous=continuous
    t=time.monotonic()
    try:
        result=pcr.eval_one((0,np.array(seed['vec']),out/'EVAL'))
        dump(out/'RESULT.json',dict(result=result,wall_s=time.monotonic()-t,reference_J=seed['J'],exact_reproduction=result['J']==seed['J'] if condition=='baseline' else None))
        if result.get('failed'):raise RuntimeError('Evaluation failed; see EVAL/SCORE.json')
        if condition=='baseline':assert result['J']==seed['J'],(result['J'],seed['J'])
    finally:
        h.run_neural_continuous=original
        after={str(p):sha(p) for p in protected};dump(out/'INPUT_HASHES_AFTER.json',after)
        assert before==after,'Historical inputs changed'
    print(json.dumps(json.loads((out/'RESULT.json').read_text()),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
