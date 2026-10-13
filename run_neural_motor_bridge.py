"""Independent BANC DNa01 spike -> engineered intent -> shipped walking controller.

Virtual heading sensor and direct DN current injection are ENGINEERED interfaces,
not reconstructed visual circuitry. Forward permission is external, not decoded
from DNa01. Actual conductance-model spikes, not fabricated spike waveforms, drive
steering. No canonical muscle model is touched. Simulator-only prototype.
"""
from pathlib import Path
from dataclasses import asdict
import argparse,json,sys,math,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from engine.embodied.adapters import TierConfig,build_neural_tier,NeuralTier,OwnershipLedger
from engine.embodied.neural_motor_bridge import SpikeIntentBridge,BridgeConfig
from engine.embodied.bridge_body import BridgeBody,BridgeBodyConfig

ROOT_IDS={'left':'720575941535862506','right':'720575941432123640'}

def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=2.)
    ap.add_argument('--conditions',nargs='+',default=['intact','sham','swap','silence','cut','feedback_cut'])
    ap.add_argument('--target-deg',type=float,default=45.)
    ap.add_argument('--render',action='store_true')
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/neural_motor_bridge/closed_loop.json')
    args=ap.parse_args()
    chunk=.005
    if args.seconds<=0 or not np.isclose(args.seconds/chunk,round(args.seconds/chunk)):
        ap.error('seconds must be positive multiple of .005')
    cfg=TierConfig(background_sd_nA=0.,weight_scale_uS_per_synapse=1e-6)
    built=build_neural_tier(cfg)
    roots=np.asarray(built['data']['root_ids']).astype(str)
    indices={}
    for side,root in ROOT_IDS.items():
        glob=np.flatnonzero(roots==root)
        if glob.size!=1:raise ValueError('Root identity absent/ambiguous: '+root)
        local=int(built['local_of_global'][int(glob[0])])
        if local<0:raise ValueError('Identified root absent from selected tier: '+root)
        indices[side]=local
    audit_path=ROOT/'outputs/neural_motor_bridge/identity_audit.json'
    audit=json.loads(audit_path.read_text())
    for side,root in ROOT_IDS.items():
        matches=[r for r in audit['banc_named_neurons'] if r['root_id']==root and r['primary_type']=='DNa01' and r['soma_side']==side]
        if len(matches)!=1:raise ValueError('Identity audit does not confirm requested pair')
    provenance_files=[Path(__file__),ROOT/'engine/embodied/neural_motor_bridge.py',ROOT/'engine/embodied/adapters.py',audit_path]
    provenance_files += [Path(r['path']) for r in audit['source_provenance'] if 'banc_' in r['path']]
    for source in audit['source_provenance']:
        if 'banc_' in source['path'] and hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Identity source changed since audit: '+source['path'])
    source_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance_files}
    identity_note='BANC primary DNa01, soma-side CSV audit; root IDs not FAFB; inspect identity_audit.json'
    print('NONCOMPLIANT_WITH_AUTONOMOUS_GOAL: historical controlled demo only; external walking permission, heading-error injection and gait controller remain.',flush=True)
    result={'status':'MEASURED_NOT_BIOLOGICAL_VALIDATION',
            'autonomous_goal_status':'NONCOMPLIANT_WITH_AUTONOMOUS_GOAL',
            'not_completion_evidence':True,
            'description':__doc__,'tier_config':asdict(cfg),'tier_fingerprint':built['fingerprint'],
            'source_sha256':source_hashes,'body_settle_before_neural_s':.2,
            'intervention_scope':'independent closed-loop branches, not matched spike replay; sham is exact matched no-op; silence suppresses readout only; cut gates entire engineered bridge including external forward; feedback_cut removes virtual yaw injection only, not body reflexes',
            'declared_engineering_thresholds':{'intact_heading_error_deg':10,'all_peak_tilt_deg':20,'all_min_height_mm':.5}, 
            'root_ids':ROOT_IDS,'local_indices':indices,'identity_audit_path':str(audit_path),
            'engineering_assumptions':{'feedback':'simulator yaw error -> direct DNa01 current; NOT visual neuroanatomy',
              'current_law':'max .05 nA per side, proportional to wrapped heading error / .3 rad',
              'locomotion_permission':.8,'forward_is_external':True,'decoder':asdict(BridgeConfig())},
            'conditions':{}}
    frame_paths=[]
    for condition in args.conditions:
        if condition not in ('intact','sham','swap','silence','cut','feedback_cut'):raise ValueError(condition)
        tier=NeuralTier(built,cfg,OwnershipLedger())
        dec=SpikeIntentBridge(built['n'],[indices['left']],[indices['right']],provenance=identity_note)
        rows=[]
        with BridgeBody(BridgeBodyConfig(render=args.render and condition=='intact')) as body:
            # Same explicit 0.2s controller settling for every condition; tier is not yet advanced.
            body.step(command='stand',duration_s=.2)
            start=body.snapshot();heading0=float(start['yaw_rad'])
            target=heading0+math.radians(args.target_deg)
            frames=[]
            for k in range(round(args.seconds/chunk)):
                obs=body.snapshot()
                err=float(math.atan2(math.sin(target-obs['yaw_rad']),math.cos(target-obs['yaw_rad'])))
                current=np.zeros(built['n'])
                if condition!='feedback_cut':
                    current[indices['left' if err>=0 else 'right']]=.05*min(abs(err)/.3,1.)
                spikes=tier.advance(round(chunk/(cfg.dt_ms/1000)),current)
                actual_counts={s:int(spikes[:,i].sum()) for s,i in indices.items()}
                transmitted=spikes.copy()
                if condition=='swap':
                    transmitted[:,indices['left']]=spikes[:,indices['right']]
                    transmitted[:,indices['right']]=spikes[:,indices['left']]
                if condition=='silence':transmitted[:,list(indices.values())]=False
                intent=dec.advance(transmitted,spike_dt_s=cfg.dt_ms/1000,
                                   locomotion_permission=.8,connected=condition!='cut')
                new=body.step(forward=intent['forward'],turn=intent['turn'],duration_s=chunk)
                rows.append({'time_s':(k+1)*chunk,'input_heading_error_rad':err,
                             'current_left_nA':float(current[indices['left']]),'current_right_nA':float(current[indices['right']]),
                             'spike_counts':actual_counts,'intent':intent,
                             'position_mm':new['position_mm'],'yaw_rad':new['yaw_rad'],'tilt_rad':new['tilt_rad']})
                if args.render and condition=='intact' and k%10==0:frames.append(body.render())
            yaw=np.unwrap([heading0]+[r['yaw_rad'] for r in rows])
            final_error=float(math.atan2(math.sin(target-yaw[-1]),math.cos(target-yaw[-1])))
            result['conditions'][condition]={'trace':rows,'target_absolute_yaw_rad':target,
                'summary':{'yaw_change_deg':float(np.degrees(yaw[-1]-heading0)),
                           'final_heading_error_deg':float(np.degrees(final_error)),
                           'displacement_mm':float(np.linalg.norm(rows[-1]['position_mm'][:2]-start['position_mm'][:2])),
                           'peak_tilt_deg':float(np.degrees(max(r['tilt_rad'] for r in rows))),
                           'min_height_mm':float(min(r['position_mm'][2] for r in rows)),
                           'dn_spikes':{s:sum(r['spike_counts'][s] for r in rows) for s in indices}},
                'body_provenance':body.provenance()}
            if frames:
                from PIL import Image
                args.out.parent.mkdir(parents=True,exist_ok=True)
                imgs=[Image.fromarray(f) for f in frames]
                imgs[0].save(args.out.parent/'closed_loop_intact.gif',save_all=True,append_images=imgs[1:],duration=50,loop=0)
                imgs[len(imgs)//2].save(args.out.parent/'closed_loop_intact.png')
        print(condition,result['conditions'][condition]['summary'],flush=True)
    c=result['conditions']
    checks={}
    if 'intact' in c and 'sham' in c:
        checks['matched_sham_exact']=json.dumps(c['intact']['trace'],default=plain)==json.dumps(c['sham']['trace'],default=plain)
    if 'intact' in c and 'swap' in c:
        checks['swapped_yaw_sign']=c['intact']['summary']['yaw_change_deg']*c['swap']['summary']['yaw_change_deg']<0
    if 'cut' in c:checks['cut_zero_intent']=all(r['intent']['forward']==r['intent']['turn']==0 for r in c['cut']['trace'])
    if 'silence' in c:checks['silence_zero_turn']=all(r['intent']['turn']==0 for r in c['silence']['trace'])
    if 'intact' in c and 'feedback_cut' in c:
        checks['feedback_reduces_heading_error']=abs(c['intact']['summary']['final_heading_error_deg'])<abs(c['feedback_cut']['summary']['final_heading_error_deg'])
    if 'intact' in c:
        checks['intact_target_within_10deg']=abs(c['intact']['summary']['final_heading_error_deg'])<=10
    checks['all_postures_within_declared_bounds']=all(v['summary']['peak_tilt_deg']<=20 and v['summary']['min_height_mm']>=.5 for v in c.values())
    checks['finite_trace']=all(np.isfinite([r['yaw_rad'],r['tilt_rad'],*r['position_mm']]).all() for v in c.values() for r in v['trace'])
    result['engineering_checks']=checks
    result['engineering_check_status']='PASS' if checks and all(checks.values()) else 'FAIL_OR_INCOMPLETE'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,default=plain,indent=2))
    print('checks',checks,'wrote',args.out)
    return 0 if result['engineering_check_status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
