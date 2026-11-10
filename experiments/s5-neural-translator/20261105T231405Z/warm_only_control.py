"""Post-hoc registered control for the simultaneous teacher-label timing change."""
import time
from pathlib import Path
import continuation as c
O=c.O;dump=c.dump;js=c.js;sha=c.sha

def main():
    stamp=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+str(time.time_ns()%1000000000).zfill(9)+'Z';register=O/'CONFIGS'/(stamp+'_warm_only_control');register.mkdir()
    dump(register/'PREREGISTRATION.json',dict(reason='Main experiment changed both data coverage and target timing; isolate timing-only warm-data retraining. Diagnostic supplement registered after main results, not a new untouched experiment.',protocol='Two successive fits at closed_weight0, same start-point labels/architecture/initialization/optimizer/epoch ceiling; dev-select between two; frozen best on same four final times. Main candidate/deployment choice remains frozen.',source_file=str(Path(__file__)),source_sha256=sha(__file__),old_main_summary_sha256=sha(O/'SUMMARY.json'),importance='核心必需'))
    t=c.runtime(fresh=True);closed=c.load('closed_train');initial=c.OLD/'MODELS/offline_mlp.npz';candidates=[]
    for number in [0,1]:
        name='warm_only_control_'+str(number);path=c.fit(name,0.,closed,initial);rs=[t.episode(name+'_'+str(phase),phase,c.actor(path),path)[0] for phase in c.rt.DEV];candidates.append(dict(name=name,model=str(path),results=rs,score=c.score(rs)));initial=path
    best=max(candidates,key=lambda x:x['score']);dump(O/'WARM_ONLY_FROZEN_SELECTION.json',dict(best=best,main_selection_unchanged=True,registered_supplement=str(register)))
    final=[t.episode('WARM_ONLY_FINAL_'+str(phase),phase,c.actor(best['model']),best['model'])[0] for phase in c.FRESH]
    dump(O/'WARM_ONLY_CONTROL.json',dict(candidates=candidates,best=best,final=final,main_selection_unchanged=True,posthoc_diagnostic=True))
    print('WARM-ONLY CONTROL COMPLETE',best['name'],flush=True)
if __name__=='__main__':main()
