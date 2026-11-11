"""Read-only exact front-leg completion-predicate diagnosis from saved1ms arrays."""
from pathlib import Path
import json
import numpy as np
from api import HERE
import physics

def longest(mask):
    best=run=0
    for v in mask:
        run=run+1 if v else 0;best=max(best,run)
    return best

def stats(a,i,mask):
    load=a['loads'][mask,i]/physics.W;z=a['padz'][mask,i];foot=a['relativefoot'][mask,i,0]
    carrying=load>=.1;ground=z<=.004;loaded=carrying & ground
    return dict(samples=len(load),load_min_W=float(load.min()),load_max_W=float(load.max()),
        padz_min=float(z.min()),padz_max=float(z.max()),relativefoot_x_min=float(foot.min()),relativefoot_x_max=float(foot.max()),
        relativefoot_x_range=float(np.ptp(foot)),carrying_samples=int(carrying.sum()),ground_samples=int(ground.sum()),
        loaded_grounded_samples=int(loaded.sum()),longest_carrying_ms=longest(carrying),longest_grounded_ms=longest(ground),
        longest_loaded_grounded_ms=longest(loaded),
        min_padz_while_carrying=None if not carrying.any() else float(z[carrying].min()),
        max_load_W_while_grounded=None if not ground.any() else float(load[ground].max()),
        required_loaded_grounded_dwell_ms=10)

def main():
    cases={}
    for case in ('halfspeed_straight','halfspeed_smallleft'):
        folder=HERE/case;r=json.loads((folder/'RESULT.json').read_text())
        with np.load(folder/'TRACE_1MS.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        post=a['control_source']==5;per={}
        for leg,i in (('LF',0),('RF',3)):
            ev=[e for e in r['cycles']['events'] if e['leg']==leg];events=[]
            for e in ev:
                mask=post & (a['time']>=e['start_time']-1e-9) & (a['time']<=e['time']+1e-9)
                st=stats(a,i,mask)
                st.update(event=e,lift_qualified=bool(e.get('lift',0)>.015),advance_qualified=bool(e.get('advance',0)+1e-12>=.03))
                # Qualification is monotonic within each original active event.
                ids=np.flatnonzero(mask);x0=float(a['relativefoot'][ids[0],i,0]);qualified=False;reload_run=best=0;qualified_rows=[]
                low=a['loads'][:,i]/physics.W<=.05
                peak_lift=peak_advance=0.
                for j in ids:
                    if low[j]:
                        peak_lift=max(peak_lift,float(a['padz'][j,i]))
                        if a['padz'][j,i]>.015:
                            peak_advance=max(peak_advance,float(a['relativefoot'][j,i,0])-x0)
                    # Event metadata supplies frozen qualification; online reconstruction
                    # below is diagnostic and one boundary-ms difference is not gate replay.
                    qualified=peak_lift>.015 and peak_advance+1e-12>=.03
                    loaded=a['loads'][j,i]/physics.W>=.1 and a['padz'][j,i]<=.004
                    reload_run=reload_run+1 if qualified and loaded else 0;best=max(best,reload_run)
                    if qualified:qualified_rows.append(j)
                st['diagnostic_longest_loaded_grounded_after_qualification_ms']=best
                # Locate actual later reload under the same frozen numeric predicate.
                future=np.flatnonzero(post & (a['time']>e['time']+1e-9))
                dwell=0;later=None
                for j in future:
                    loaded=a['loads'][j,i]/physics.W>=.1 and a['padz'][j,i]<=.004
                    dwell=dwell+1 if loaded else 0
                    if dwell>=10:
                        later=dict(completed_dwell_time_s=float(a['time'][j]),delay_after_event_end_s=float(a['time'][j]-e['time']),age_since_event_start_s=float(a['time'][j]-e['start_time']))
                        break
                st['next_actual10ms_loaded_grounded_dwell_after_event']=later
                if qualified_rows:
                    qm=np.zeros(len(a['time']),bool);qm[qualified_rows]=True
                    st['after_diagnostic_qualification']=stats(a,i,qm)
                st['exact_failed_predicates_from_event']=dict(
                    clearance=not st['lift_qualified'],advance=not st['advance_qualified'],
                    completion_not_recorded=e['type']!='complete',reason=e.get('reason'),
                    interpretation='At timeout, qualified swing still needs grounded loaded dwell within4s. Thresholds unchanged; no infeasibility inferred.')
                events.append(st)
            per[leg]=dict(whole_modified_window=stats(a,i,post),complete_count=r['cycles']['complete_per_leg'][leg],
                abort_reason_counts={reason:sum(e['type']=='abort' and e.get('reason')==reason for e in ev) for reason in ('timeout','recontact','terminal')},events=events)
        cases[case]=per
    result=dict(scope='Saved1ms endpoint descriptive diagnosis only; authoritative cycle decisions are independently replayed unchanged CycleTracker in VERIFICATION.json',
        predicates=dict(loaded='load/W>=.1 AND padz<=.004 for10ms',unloaded='load/W<=.05 for5ms',lift='peak padz>.015',advance='unloaded clear relativefoot X advance>=.03',timeout_s=4),cases=cases)
    (HERE/'FRONT_PREDICATE_DIAGNOSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({c:{l:{k:v for k,v in d.items() if k!='events'} for l,d in p.items()} for c,p in cases.items()},indent=2))
if __name__=='__main__':main()
