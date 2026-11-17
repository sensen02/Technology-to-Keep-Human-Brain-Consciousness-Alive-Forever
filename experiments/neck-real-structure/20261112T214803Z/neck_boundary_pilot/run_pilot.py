"""Parent-only source-qualified engineering pulse/cut/restore check, not neural physiology.

No real brain, whole-body integration, training, membrane model or electrode operation.
"""
from pathlib import Path
import argparse
import json
import sys
from boundary import BoundaryTransport, BoundaryError, ROOTS, STATUS, require, sha256
HERE=Path(__file__).resolve().parent

def fresh(path,obj):
    with Path(path).open('x') as handle:json.dump(obj,handle,indent=2,allow_nan=False);handle.write('\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute-parent-pilot',action='store_true');parser.add_argument('--source-bindings-sha256');args=parser.parse_args()
    if not args.execute_parent_pilot:print('CODE_ONLY no real source transport executed');return
    require(not any((HERE/name).exists() for name in ('INPUTS_BEFORE.json','PILOT_RESULT.json','TRANSPORT_JOURNAL.json')),'inspect prior output before any rerun')
    path=HERE/'SOURCE_BINDINGS.json';engine=BoundaryTransport(path,args.source_bindings_sha256)
    require(engine.qualification['source_qualified_two_fibres'] and not engine.qualification['synthetic_fixture'],'real source qualification required')
    bindings=json.loads(path.read_text());hashes={item['path']:item['sha256'] for item in bindings['files']}
    for p in (path,HERE/'run_pilot.py',HERE/'test_boundary.py'):hashes[str(p)]=sha256(p)
    fresh(HERE/'INPUTS_BEFORE.json',hashes);checks={}
    def check(name,value):
        checks[name]=bool(value)
        require(checks[name],'pilot check failed: '+name)
    def packet(direction,value):
        return {'from':'brain' if direction=='DN' else 'other','to':'neck','direction':direction,'root_ids':[ROOTS[direction]],'values':[value],'capture_step':engine.step,'source_bindings_sha256':args.source_bindings_sha256}
    engine.send(packet('DN',.5));engine.send(packet('AN',-.25));check('no_future_at_step1',engine.advance()==[])
    first=engine.advance();check('two_source_IDs_arrive_step2',[x['root_id'] for x in first]==[ROOTS['DN'],ROOTS['AN']]);check('gain1_signs_and_values_preserved',[x['value'] for x in first]==[.5,-.25]);check('DN_and_AN_reach_opposite_sides',[x['target_part'] for x in first]==['other','brain']);check('fixed_engineering_delay2',all(x['release_step']==x['consume_step']==x['capture_step']+2 for x in first))
    engine.send(packet('DN',.4));engine.send(packet('AN',.3));engine.set_cut('DN',True)
    try:engine.send(packet('DN',.7))
    except BoundaryError:cut_rejected=True
    else:cut_rejected=False
    check('inputs_while_cut_rejected',cut_rejected);check('no_early_release_after_cut',engine.advance()==[])
    cut_output=engine.advance();check('AN_continues_DN_blocked',[x['direction'] for x in cut_output]==['AN'] and cut_output[0]['value']==.3)
    before_restore=engine.step;engine.set_cut('DN',False);check('restore_does_not_reset_clock',engine.step==before_restore==4)
    engine.send(packet('DN',.75));check('restore_no_early_signal',engine.advance()==[]);restored=engine.advance();check('restored_future_packet_only',len(restored)==1 and restored[0]['direction']=='DN' and restored[0]['value']==.75 and restored[0]['capture_step']==4 and restored[0]['consume_step']==6)
    journal=engine.journal();drops=[x for x in journal if x['event']=='drop_inflight_on_cut'];check('cut_discards_old_inflight_with_record',len(drops)==1 and drops[0]['value']==.4);check('no_reset_API',not hasattr(engine,'reset'))
    for source,digest in hashes.items():check('source_unchanged:'+source,sha256(source)==digest)
    check('no_heavy_runtime_imports',not any(name in sys.modules for name in ('numpy','torch','mujoco','physics','model')))
    fresh(HERE/'TRANSPORT_JOURNAL.json',journal)
    result={'status':STATUS,'all_scoped_engineering_checks_pass':all(checks.values()),'checks':checks,'source_bindings':str(path),'source_bindings_sha256':args.source_bindings_sha256,'source_hashes':hashes,'journal_sha256':sha256(HERE/'TRANSPORT_JOURNAL.json'),'selected_fibres':ROOTS,'actual_steps':engine.step,'engineering_dt_s':.001,'engineering_delay_s':.002,'elapsed_engineering_time_s':engine.step*.001,'physiological_time_validated':False,'queue_samples_not_biological_spikes':True,'full_neck_training_allowed':False,'full_neck_structure_coverage_verified':False,'new_architecture_walking_verified':False,'neural_membrane_forward_calls':0,'ANN_forward_calls':0,'training_updates':0,'Core_constructions':0,'physical_steps':0,'electrode_coupling_verified':False,'biological_consciousness_claim':False,'limitations':'This is a causal identity-preserving transport approximation over two source-qualified fibres, not reconstructed neural conduction, full-neck function or surgery/recording/stimulation validation.'}
    fresh(HERE/'PILOT_RESULT.json',result)
    print(json.dumps({'status':STATUS,'all_scoped_engineering_checks_pass':True,'checks':len(checks),'engineering_steps':engine.step,'full_neck_training_allowed':False,'physical_steps':0}))
if __name__=='__main__':main()
