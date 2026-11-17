"""Stdlib-only fail-closed three-part interface preparation; not a neural simulator.

A valid pending contract is NOT admission. Fixture mode only tests schema mechanics.
Source hashes establish byte identity, not anatomical truth; admission also requires
independent source/crossing review and separately preregistered dynamics/runtime.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

SCHEMA = 'THREE_PART_NECK_INTERFACE_V1'
PENDING = 'NOT_ADMITTED_NECK_SOURCE_PENDING'
ALLOWED_ROUTES = {('brain','neck','DN'), ('neck','other','DN'),
                  ('other','neck','AN'), ('neck','brain','AN')}

class ContractError(ValueError):
    pass

def require(value, message):
    if not value:
        raise ContractError(message)

def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def validate(contract, *, fixture=False, verify_files=True):
    require(contract.get('schema') == SCHEMA, 'unsupported schema')
    parts = contract.get('parts', {})
    require(set(parts) == {'brain','neck','other'}, 'exactly three parts required')
    require(parts['brain'].get('kind') == 'ARTIFICIAL_TRAINABLE_BRAIN_ANN', 'brain must be declared artificial ANN')
    require(parts['brain'].get('motor_output') == 'SOURCE_IDENTIFIED_DN_PORT_ACTIVITY_ONLY', 'brain action bypass forbidden')
    require(parts['brain'].get('observation') == 'SOURCE_IDENTIFIED_AN_FEEDBACK_AND_DECLARED_BRAIN_SIDE_SENSORY_ONLY', 'brain observation bypass forbidden')
    require(parts['neck'].get('kind') == 'SOURCE_GROUNDED_FIBRE_BOUNDARY', 'dense ANN cannot stand for real neck')
    require(parts['neck'].get('topology_trainable') is False and parts['neck'].get('invented_internal_synapses_allowed') is False, 'neck topology must be immutable')
    require(parts['other'].get('direct_task_target_access') is False, 'body task-target bypass forbidden')
    timing = contract.get('timing', {})
    require(timing.get('future_feedback_allowed') is False and timing.get('reset_at_handover_allowed') is False, 'causal continuous timing required')
    ports = contract.get('ports', [])
    links = contract.get('neck_links', [])
    routes = contract.get('boundary_routes', [])
    sensory = contract.get('brain_side_sensory', [])
    require(all(isinstance(x,list) for x in (ports,links,routes,sensory)), 'list fields required')
    bundle = contract.get('source_bundle')
    if bundle is None:
        require(contract.get('status') == PENDING, 'missing source cannot be admitted')
        require(not ports and not links and not routes and not sensory, 'unbound source IDs cannot be declared real')
        require(contract.get('declared_dimensions') == {'DN':None,'AN':None}, 'pending dimensions must remain unknown')
        require(contract.get('admission', {}).get('training_allowed') is False, 'pending source cannot train')
        return {'valid_contract':True, 'admitted':False, 'status':PENDING,
                'reason':'No source/crossing evidence bound; no invented port dimensions', 'fixture':fixture}
    require(isinstance(bundle,dict), 'source bundle object required')
    is_toy = bundle.get('synthetic_fixture') is True
    require(fixture or not is_toy, 'synthetic fixture cannot be admitted as biology')
    sources = bundle.get('files', [])
    require(isinstance(sources,list) and bool(sources), 'source files required')
    source_ids = set()
    for source in sources:
        sid = source.get('id')
        require(isinstance(sid,str) and sid and sid not in source_ids, 'unique source IDs required')
        source_ids.add(sid)
        path = source.get('path')
        require(isinstance(path,str) and Path(path).is_absolute() and digest(source.get('sha256')), 'absolute source path and SHA256 required')
        require(all(source.get(k) for k in ('dataset','version','sample','species','role')), 'source identity metadata required')
        if verify_files:
            require(Path(path).is_file() and sha256(path)==source['sha256'], 'source byte identity mismatch')
    def evidence(ref):
        require(isinstance(ref,dict) and ref.get('source_id') in source_ids and isinstance(ref.get('record'),str) and bool(ref['record']), 'evidence needs known source and record locator')
    evidence(bundle.get('crossing_definition_evidence'))
    topology_hash = bundle.get('topology_sha256')
    require(digest(topology_hash), 'immutable topology digest required')
    topology_source = next((s for s in sources if s['id']==bundle.get('topology_source_id')), None)
    require(topology_source is not None and topology_source['sha256']==topology_hash, 'topology digest must bind an actual registered source file')
    ids = set()
    for port in ports:
        pid = port.get('id')
        require(isinstance(pid,str) and pid and pid not in ids, 'duplicate or absent port ID')
        ids.add(pid)
        require(port.get('direction') in ('DN','AN'), 'port direction must be DN or AN')
        require(isinstance(port.get('source_neuron_id'),str) and bool(port['source_neuron_id']), 'original neuron/fibre identity required')
        evidence(port.get('identity_evidence')); evidence(port.get('crossing_evidence'))
        require(port.get('signal_units') and finite(port.get('minimum')) and finite(port.get('maximum')) and port['minimum'] < port['maximum'], 'explicit finite port activity bounds/units required')
    require(bool(ports), 'bound contract needs actual ports')
    by_id = {p['id']:p for p in ports}
    dimensions = {d:sum(p['direction']==d for p in ports) for d in ('DN','AN')}
    require(all(dimensions.values()) and contract.get('declared_dimensions') == dimensions, 'DN/AN dimensions must derive from port identities')
    linked = set()
    for link in links:
        pid=link.get('port_id')
        require(pid in by_id and pid not in linked, 'one declared neck transmission per port required')
        linked.add(pid)
        require(link.get('kind') in ('same_fibre_transmission','source_synaptic_mapping'), 'invented dense neck operator forbidden')
        evidence(link.get('evidence'))
        require(link.get('topology_trainable') is False, 'link topology cannot train')
        if link['kind']=='same_fibre_transmission':
            require(link.get('source_neuron_id')==by_id[pid]['source_neuron_id'], 'fibre transmission must preserve original identity')
    require(linked==ids, 'every port must have evidenced neck transmission')
    expected={(a,b,d,p['id']) for p in ports for a,b,d in ALLOWED_ROUTES if d==p['direction']}
    actual=set()
    for route in routes:
        key=(route.get('from'),route.get('to'),route.get('direction'),route.get('port_id'))
        require(key in expected and key not in actual, 'bypass, wrong-direction, unknown or duplicate route')
        actual.add(key)
        require(route.get('signal') == 'PORT_ACTIVITY', 'joint actions/targets cannot pass as neural port activity')
    require(actual==expected, 'complete DN/AN boundary routes required')
    sensory_ids=set()
    for item in sensory:
        sid=item.get('id')
        require(isinstance(sid,str) and sid and sid not in sensory_ids and sid not in ids, 'unique direct sensory ID required')
        sensory_ids.add(sid)
        require(item.get('target_part')=='brain' and item.get('anatomy_crosses_neck') is False, 'direct sensation requires non-neck anatomy declaration')
        require(item.get('kind')=='BRAIN_SIDE_SENSORY_NOT_BODY_MOTOR_OR_FEEDBACK', 'no body-state leak disguised as direct visual input')
        evidence(item.get('anatomy_evidence'))
    if is_toy:
        return {'valid_contract':True,'admitted':False,'status':'SYNTHETIC_FIXTURE_ONLY_NOT_BIOLOGY','fixture':True,'dimensions':dimensions}
    ready=contract.get('admission',{})
    required_reviews=('structural_source_review','neck_crossing_or_cut_definition_review','dynamics_preregistration','runtime_binding')
    for key in required_reviews:
        value=ready.get(key)
        require(isinstance(value,dict) and value.get('status')=='REVIEWED', 'independent admission review missing: '+key)
        evidence(value.get('evidence'))
    for key in ('neural_dt_s','neck_delay_s','brain_decision_s','body_dt_s'):
        require(finite(timing.get(key)) and timing[key]>0, 'timing must be preregistered: '+key)
    require(contract.get('dynamics',{}).get('model') and contract['dynamics'].get('evidence_status') in ('ENGINEERING_ASSUMPTION','MEASURED_AND_CITED'), 'explicit dynamics model and evidence class required')
    require(ready.get('training_allowed') is True and contract.get('status')=='ADMITTED_INTERFACE_ONLY', 'explicit admission status required')
    return {'valid_contract':True,'admitted':True,'status':'ADMITTED_INTERFACE_ONLY_NOT_BEHAVIOR_OR_CONSCIOUSNESS','dimensions':dimensions,'topology_sha256':topology_hash}

def validate_packet(contract, packet, *, fixture=False):
    result=validate(contract,fixture=fixture)
    require(result.get('admitted') or (fixture and result['status']=='SYNTHETIC_FIXTURE_ONLY_NOT_BIOLOGY'), 'source-pending interface refuses runtime packets')
    require(set(packet)=={'from','to','direction','port_ids','values','capture_time_s','release_time_s','consume_time_s','topology_sha256'}, 'packet contains undeclared fields/hidden inputs')
    require((packet['from'],packet['to'],packet['direction']) in ALLOWED_ROUTES, 'runtime bypass forbidden')
    require(packet['topology_sha256']==contract['source_bundle']['topology_sha256'], 'runtime topology drift')
    ports=[p for p in contract['ports'] if p['direction']==packet['direction']]
    require(packet['port_ids']==[p['id'] for p in ports] and len(packet['values'])==len(ports), 'packet port ordering/dimension mismatch')
    for p,value in zip(ports,packet['values']):require(finite(value) and p['minimum']<=value<=p['maximum'], 'nonfinite or unbounded neural activity')
    times=[packet[k] for k in ('capture_time_s','release_time_s','consume_time_s')]
    require(all(finite(t) and t>=0 for t in times) and times==sorted(times), 'future or invalid feedback time')
    return True

def main():
    parser=argparse.ArgumentParser();parser.add_argument('contract',type=Path);args=parser.parse_args()
    try:
        result=validate(json.loads(args.contract.read_text()))
        print(json.dumps(result,indent=2))
        return 0 if result['admitted'] else 2
    except (ContractError,KeyError,TypeError,ValueError,OSError) as exc:
        print(json.dumps({'valid_contract':False,'admitted':False,'error':str(exc)}));return 1
if __name__=='__main__':raise SystemExit(main())
