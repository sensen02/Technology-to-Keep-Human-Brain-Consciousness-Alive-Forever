"""Independent S158 offline units/time-index audit, no simulator imports."""
from pathlib import Path
import json,csv,argparse,hashlib
from decimal import Decimal
import numpy as np
ARMS={'S154':Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z'),'S157':Path('/home/sensen/Desktop/engine修复验收/s157-foreleg-gain/20261024T221335645450Z')}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output;hashes=json.loads((p/'INPUT_HASHES.json').read_text());assert hashes==json.loads((p/'INPUT_HASHES_AFTER.json').read_text());assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in hashes.items());events=list(csv.DictReader((p/'EVENT_TIMES.csv').open()));maxerr=0
 for arm,root in ARMS.items():
  with np.load(root/'init80ms/TRAJECTORY.npz') as a:
   assert len(a['time'])==85;assert np.allclose(a['time'],np.arange(85)/40000,rtol=0,atol=1e-14);load=a['foot_force'][:,:,2].sum(1)/(-a['gravity_force_world'][:,2]);assert np.all(load[:-1]>=.8) and load[-1]<.8
   for r in [x for x in events if x['arm']==arm]:
    i=int(r['index']);expected=float(Decimal(i)*Decimal('0.000025'));assert abs(expected-float(r['time_s']))<1e-14
    for field,div in [('activation_tau_units',.001),('deactivation_tau_units',.0004),('sensory_period_units',.005),('neural_tick_units',.0005),('contact_solref_nominal_units',.002)]:assert abs(expected/div-float(r[field]))<1e-12
    maxerr=max(maxerr,abs(load[i]-float(r['load_ratio'])));assert maxerr<1e-12
   assert abs(a['body_mass'][0].sum()*9801+a['gravity_force_world'][0,2])<1e-12
  with np.load(root/'init80ms/NATIVE_MODEL_ARRAYS.npz') as m:assert np.all(m['actuator_dynprm'][:,:2]==[.001,.0004])
 assert 160*Decimal('.0005')==Decimal('.08');assert Decimal('.0005')/Decimal('.000025')==20;assert Decimal('.005')/Decimal('.000025')==200
 (p/'PARENT_SCALE_CLOCK_INDEPENDENT_CHECK.json').write_text(json.dumps(dict(passed=True,event_count=len(events),max_load_error=maxerr,conversions='Decimal checked: PRE .08 s neural only; tick20steps; sensory200steps; failure84steps=2.1 nominal act tau=5.25 nominal deact tau=.42 sensory periods',physical_unit_calibration=False,model_load=0,forward=0,step=0,neural_ticks=0),indent=2));print('Independent time-index and parameter audit passed')
if __name__=='__main__':main()
