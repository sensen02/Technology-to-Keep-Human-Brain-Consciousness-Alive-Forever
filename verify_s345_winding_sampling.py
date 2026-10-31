"""Independent saved-array angular-increment ledger; does not use np.unwrap."""
import argparse,json,hashlib,csv
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def angle_steps(a):
 v=a[1:]*np.conj(a[:-1]);return np.arctan2(v.imag,v.real)
def path(a):return np.r_[0.,np.cumsum(angle_steps(a))]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);a=ap.parse_args();s=a.source
 manifest=json.loads((s/'MANIFEST_SHA256.json').read_text());assert all(sha(s/k)==v for k,v in manifest.items())
 before=json.loads((s/'INPUT_HASHES.json').read_text());assert all(sha(Path(k))==v for k,v in before.items())
 z=np.load(s/'TRACES.npz');res=json.loads((s/'RESULTS.json').read_text());events=[];checks=[];summ=[]
 for r in res['rows']:
  key=r['pipeline'];f=r['factor'];p=key+f'_x{f}_';za=z[p+'analytic_A'];zb=z[p+'analytic_B'];t=z[p+'t'];prod=za*np.conj(zb)
  u=path(prod);ua=path(za);ub=path(zb);sep=ua-ub;edge=int(.2*r['fs'])
  advance=lambda v:float(np.degrees(np.median(v[-edge:])-np.median(v[:edge])))
  adv=advance(u);sd=advance(sep)
  err=float(np.max(np.abs((z[p+'u']-z[p+'u'][0])-u)))
  checks.append(bool(err<1e-10 and abs(adv-r['advance_deg'])<1e-8 and abs(sd-r['separate_advance_deg'])<1e-8 and int(np.floor(abs(adv)/180))==r['halfturn_count']))
  # Every interval where separate-channel phase transport differs from product transport by a whole turn.
  defect=(angle_steps(za)-angle_steps(zb)-angle_steps(prod))/(2*np.pi)
  for j in np.flatnonzero(np.abs(defect)>.5):
   events.append(dict(pipeline=key,factor=f,time_start_s=float(t[j]),time_end_s=float(t[j+1]),channel_A_increment_deg=float(np.degrees(angle_steps(za)[j])),channel_B_increment_deg=float(np.degrees(angle_steps(zb)[j])),product_increment_deg=float(np.degrees(angle_steps(prod)[j])),separate_minus_product_turns=float(defect[j])))
  if f>1:
   native=z[key+'_x1_u'];d=u[::f]-(native-native[0]);branch=np.rint(d/(2*np.pi)).astype(int)
   for j in np.flatnonzero(np.diff(branch)!=0):
    events.append(dict(pipeline=key,factor=f,time_start_s=float(t[j*f]),time_end_s=float(t[(j+1)*f]),native_vs_interpolant_branch_change_turns=int(branch[j+1]-branch[j])))
  summ.append(dict(pipeline=key,factor=f,angular_increment_path_max_error_rad=err,advance_deg=adv,separate_advance_deg=sd,transport_defect_intervals=int(np.count_nonzero(np.abs(defect)>.5))))
 assert all(checks)
 fields=list(dict.fromkeys(k for r in events for k in r))
 with (s/'BRANCH_EVENT_LEDGER.csv').open('w',newline='') as out:
  w=csv.DictWriter(out,fieldnames=fields);w.writeheader();w.writerows(events)
 dump(s/'INDEPENDENT_VERIFICATION.json',dict(passed=True,cases=len(checks),checks_passed=sum(checks),method='atan2 cross/dot increments and cumulative sum; no unwrap',source_manifest_passed=True,input_hashes_unchanged=True,rows=summ,events=events,limitation='Algebraic verification of saved discrete paths, not continuous phase truth. Endpoint median halfturn count is not a topological winding number.'))
 print(json.dumps(dict(passed=True,cases=len(checks),events=events)))
if __name__=='__main__':main()
