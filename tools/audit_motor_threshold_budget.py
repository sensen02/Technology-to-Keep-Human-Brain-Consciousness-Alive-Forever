"""Algebraic audit of archived motor membrane states; never step a network."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
S=Path('/home/sensen/Desktop/engine修复验收/s4-neural-boundary-replay/20261101T185129Z')
H=Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out
    assert (out/'PREREGISTRATION.json').exists() and not (out/'BUDGET.json').exists()
    paths=[S/n for n in ['IDENTITY.json','IDENTITIES.npz','ACTUAL_We.npz','ACTUAL_Wi.npz','BASELINE_REPLAY.npz']]+[H/'NEURAL_BOUNDARY.npz',H/'MAPPING.json',Path(__file__)]
    hashes={str(p):sha(p) for p in paths}
    ident=json.loads((S/'IDENTITY.json').read_text());p=ident['conductance_params']
    with np.load(S/'IDENTITIES.npz') as z: motor=z['motor_indices'];roots=z['root_ids'][motor]
    with np.load(S/'BASELINE_REPLAY.npz') as z:
        sp=np.unpackbits(z['spikes_packed'],axis=1)[:30,:12000].astype(bool)
        v=z['motor_voltage'];ge=z['motor_ge'];gi=z['motor_gi'];adapt=z['motor_adaptation'];I=z['motor_current'];refr=z['motor_refractory']
    with np.load(H/'NEURAL_BOUNDARY.npz') as z: ring=z['_ring'];offset=z['threshold_offset_mV'][motor]
    We=load_npz(S/'ACTUAL_We.npz')[motor];Wi=load_npz(S/'ACTUAL_Wi.npz')[motor]
    assert ident['inferred_boundary_scalars']['ring_i']==0 and ring.shape==(5,12000)
    arrivals=np.vstack([ring[k] if k<5 else sp[k-5] for k in range(30)])
    gae=np.asarray(We@arrivals.T).T;gai=np.asarray(Wi@arrivals.T).T
    E=ge+gae;J=gi+gai;dt=.5;C=p['capacitance_nF'];L=p['leak_uS'];thr=p['threshold_mV']+offset
    terms=dict(capacitive=C/dt*(thr-v),leak=np.broadcast_to(L*(thr-p['rest_mV']),v.shape),
        excitation=E*(thr-p['exc_reversal_mV']),inhibition=J*(thr-p['inh_reversal_mV']),
        adaptation=adapt,external=-I)
    budget=sum(terms.values())
    trial=(C/dt*v+L*p['rest_mV']+E*p['exc_reversal_mV']+J*p['inh_reversal_mV']+I-adapt)/(C/dt+L+E+J)
    predicted=(~refr)&(trial>=thr)
    assert np.array_equal(predicted,sp[:,motor])
    assert np.array_equal((~refr)&(budget<=0),predicted)
    assert all(np.isfinite(x).all() for x in [trial,budget,*terms.values()])
    # These diagnostic removals freeze all state and conductance history.
    no_inh=(~refr)&((budget-terms['inhibition'])<=0)
    no_adapt=(~refr)&((budget-terms['adaptation'])<=0)
    no_both=(~refr)&((budget-terms['inhibition']-terms['adaptation'])<=0)
    mapping=json.loads((H/'MAPPING.json').read_text())
    rows=[]
    for j,(idx,root) in enumerate(zip(motor,roots)):
        rows.append(dict(local_index=int(idx),root_id=int(root),
            muscles=[r['actuator'] for r in mapping['rows'] if int(idx) in r['local_indices']],
            observed_spikes=int(predicted[:,j].sum()),
            min_budget_nA=float(budget[:,j].min()),median_budget_nA=float(np.median(budget[:,j])),
            median_terms_nA={key:float(np.median(val[:,j])) for key,val in terms.items()},
            frozen_no_inhibition_crossings=int(no_inh[:,j].sum()),
            frozen_no_adaptation_crossings=int(no_adapt[:,j].sum())))
    eligible=~refr
    result=dict(input_sha256=hashes,shape=list(v.shape),predicted_motor_spikes=int(predicted.sum()),
        refractory_cell_steps=int(refr.sum()),
        median_terms_nA={key:float(np.median(val[eligible])) for key,val in terms.items()},
        budget_quantiles_nA=np.quantile(budget[eligible],[0,.25,.5,.75,1]).tolist(),
        frozen_no_inhibition_crossings=int(no_inh.sum()),frozen_no_adaptation_crossings=int(no_adapt.sum()),
        frozen_no_both_crossings=int(no_both.sum()),
        unique_cells_frozen_no_inhibition=int(no_inh.any(0).sum()),
        unique_cells_frozen_no_adaptation=int(no_adapt.any(0).sum()),
        actual_cells=int(predicted.any(0).sum()),
        sign_convention='budget <=0 predicts crossing if not refractory; excitation negative, other resisting terms positive; model nA, not physiologically calibrated',
        limitations='Removal counts freeze the entire recorded history, are cell-step algebraic tests, not propagated intervention spike counts or a recommendation to remove inhibition/adaptation.',
        calls=dict(tier_build=0,network_init=0,neural_step=0,body_step=0),per_motor_cell=rows)
    assert hashes=={str(p):sha(p) for p in paths}
    np.savez_compressed(out/'BUDGET_ARRAYS.npz',root_ids=roots,motor_indices=motor,
        budget=budget,trial_voltage=trial,spikes=predicted,refractory=refr,
        arrival_exc=gae,arrival_inh=gai,**terms)
    (out/'BUDGET.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_motor_cell','input_sha256')},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
