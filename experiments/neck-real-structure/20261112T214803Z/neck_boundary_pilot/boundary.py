"""Stdlib identity-preserving two-fibre boundary transport, NOT membrane physiology.

Geometry selects real same-axon boundary identity, never an invented synapse/RNN.
Queue delay/gain/activity are declared engineering approximations, not measurements.
"""
from pathlib import Path
import hashlib
import json
import math

STATUS='SCOPED_TWO_FIBRE_TRANSPORT_NOT_FULL_NECK_ADMISSION'
ROOTS={'DN':'720575941352057904','AN':'720575941351919408'}

class BoundaryError(ValueError):
    pass

def require(ok, message):
    if not ok:raise BoundaryError(message)

def sha256(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def finite(value):
    return type(value) in (int,float) and math.isfinite(value)

def checked_sources(bindings, *, allow_fixture=False):
    require(bindings.get('schema')=='SOURCE_BOUND_TWO_FIBRE_TRANSPORT_V1','unsupported binding schema')
    require(bindings.get('status')==STATUS and bindings.get('full_neck_training_allowed') is False,'subset cannot admit full-neck training')
    require(bindings.get('electrophysiology_verified') is False and bindings.get('electrode_coupling_verified') is False,'transport cannot claim physiology/electrode validation')
    toy=bindings.get('synthetic_fixture') is True
    require(not toy or allow_fixture,'toy source cannot be used as real source-qualified binding')
    files=bindings.get('files',[]);require(isinstance(files,list) and bool(files),'bound files required')
    filemap={}
    for item in files:
        path=item.get('path');digest=item.get('sha256')
        require(isinstance(path,str) and Path(path).is_absolute() and valid_sha(digest),'absolute source path/SHA required')
        require(path not in filemap,'duplicate source path')
        require(Path(path).is_file() and sha256(path)==digest,'source identity drift: '+str(path))
        filemap[path]=digest
    fibres=bindings.get('fibres');require(isinstance(fibres,list) and len(fibres)==2,'exactly selected two fibres required')
    require([f.get('direction') for f in fibres]==['DN','AN'],'stable DN/AN ordering required')
    require(len({f.get('root_id') for f in fibres})==2,'duplicate fibre identity')
    expected_config={'dt_s':.001,'delay_s':.002,'delay_steps':2,'activity_units':'DIMENSIONLESS_ENGINEERING_ACTIVITY','activity_min':-1.,'activity_max':1.,'gain':1.}
    require(bindings.get('engineering_config')==expected_config,'engineering timing/activity config drift')
    for f in fibres:
        direction=f['direction'];require(f.get('transport_kind')=='SAME_AXON_IDENTITY_NO_SYNAPSE','invented neck synapse forbidden')
        if toy:
            require(f['root_id'].startswith('TOY_'),'fixture must visibly use fictional IDs')
            continue
        require(f['root_id']==ROOTS[direction],'unexpected selected original root')
        for key in ('metadata_path','independent_check_path','swc_path','receipt_path'):
            require(f.get(key) in filemap,'source file not in immutable bindings: '+key)
        metadata=json.loads(Path(f['metadata_path']).read_text());check=json.loads(Path(f['independent_check_path']).read_text());receipts=json.loads(Path(f['receipt_path']).read_text())
        require(metadata.get('dataset')=='BANC' and metadata.get('materialization')==888 and metadata.get('root_id')==f['root_id'],'BANC version/root mismatch')
        identity=metadata.get('version_identity_evidence',{});row=identity.get('source_row',{})
        require(identity.get('verified') is True and identity.get('root_888')==row.get('root_888')==row.get('root_id')==f['root_id'],'root_888 identity not established')
        require(row.get('super_class')==('descending' if direction=='DN' else 'ascending') and row.get('tract')=='cervical_connective','direction must come from source metadata, not SWC label')
        require(metadata.get('coordinate_space')=='BANC_NATIVE' and metadata.get('coordinate_units')=='nm','coordinate identity mismatch')
        require(metadata.get('planes',{}).get('nm_y')==[370000.,484000.] and metadata['planes'].get('definition')=='CURATED_PROFILE_SEED_PLANES_NOT_SYNAPSES','source planes identity mismatch')
        require(check.get('status')=='SOURCE_PINNED_GEOMETRY_CHECK_ONLY_NOT_MODEL_ADMISSION' and check.get('root_id')==f['root_id'] and check.get('each_plane_intersected') is True and check.get('is_single_tree') is True,'independent geometry not passed')
        require(check.get('metadata')==metadata and check.get('plane_y_nm')==[370000.,484000.],'independent check metadata drift')
        require(all(check.get('plane_intersections',{}).get(str(float(y))) for y in (370000,484000)),'missing both actual plane crossings')
        require(check.get('training_admitted') is False and check.get('physiology_verified') is False and check.get('electrode_mapping_verified') is False,'geometry must not silently claim train/physiology')
        require(metadata.get('swc_sha256')==check.get('source_swc_sha256')==filemap[f['swc_path']],'SWC hash not matched')
        require(check.get('projection_direction_not_inferred_from_label2') is True and check.get('neck_local_synapses_inferred') is False,'compartment labels cannot create direction/synapses')
        for evidence in metadata.get('source_evidence',[]):
            require(evidence.get('path') in filemap and filemap[evidence['path']]==evidence.get('sha256'),'metadata source evidence not pinned')
            if Path(evidence['path']).name=='NECK_POINT_AND_SAMPLE_GEOMETRY_AUDIT.json':
                underlying=json.loads(Path(evidence['path']).read_text()).get('source_hashes',{})
                require(bool(underlying),'underlying geometry sources missing')
                for source,digest in underlying.items():
                    require(source in filemap and filemap[source]==digest,'underlying v888 metadata/point/SWC sources not pinned')
        matched=[obj for obj in receipts.get('objects',[]) if obj.get('path')==f['swc_path']]
        require(len(matched)==1 and matched[0].get('sha256')==metadata['swc_sha256'] and matched[0].get('md5_verified') is True and matched[0].get('generation')==matched[0].get('response_generation'),'public pinned download receipt mismatch')
    return {'status':STATUS,'source_qualified_two_fibres':not toy,'synthetic_fixture':toy,'full_neck_training_allowed':False}

class BoundaryTransport:
    """Tick-causal zero-inference gain1 transport; no neural/membrane state or reset API."""
    def __init__(self, bindings_path, expected_sha256, *, _allow_synthetic_fixture=False):
        self._path=Path(bindings_path);self._expected=expected_sha256
        require(valid_sha(expected_sha256) and self._path.is_file() and sha256(self._path)==expected_sha256,'binding bytes/SHA mismatch')
        self._original=self._path.read_bytes();self._bindings=json.loads(self._original)
        self._fixture=_allow_synthetic_fixture;self.qualification=checked_sources(self._bindings,allow_fixture=self._fixture)
        self._fibres={f['direction']:f['root_id'] for f in self._bindings['fibres']}
        self._step=0;self._queue=[];self._cut=set();self._seen=set();self._journal=[]
    @property
    def step(self):return self._step
    def _verify(self):
        require(self._path.read_bytes()==self._original and sha256(self._path)==self._expected,'binding identity changed')
        checked_sources(self._bindings,allow_fixture=self._fixture)
    def send(self, packet):
        self._verify()
        require(isinstance(packet,dict) and set(packet)=={'from','to','direction','root_ids','values','capture_step','source_bindings_sha256'},'packet extra/missing fields; joint actions or targets forbidden')
        direction=packet['direction'];require(direction in self._fibres,'unknown direction')
        require((packet['from'],packet['to'])==(('brain','neck') if direction=='DN' else ('other','neck')),'source-side bypass forbidden')
        require(packet['root_ids']==[self._fibres[direction]],'original fibre identity/order mismatch')
        require(type(packet['capture_step']) is int and packet['capture_step']==self._step,'past/future packet timestamp forbidden')
        require(packet['source_bindings_sha256']==self._expected,'packet source identity drift')
        values=packet['values'];require(isinstance(values,list) and len(values)==1 and finite(values[0]) and -1<=values[0]<=1,'finite bounded dimensionless one-fibre activity required')
        require(direction not in self._cut,'fibre is cut')
        require((self._step,direction) not in self._seen,'duplicate fibre sample at same tick')
        item={'direction':direction,'root_id':self._fibres[direction],'value':float(values[0]),'capture_step':self._step,'release_step':self._step+2}
        self._queue.append(item);self._seen.add((self._step,direction));self._journal.append(dict(event='enqueue',**item))
    def advance(self):
        self._verify();self._step+=1;released=[];pending=[]
        for item in self._queue:
            if item['release_step']>self._step:pending.append(item);continue
            require(item['release_step']==self._step,'overdue transport packet')
            if item['direction'] in self._cut:
                self._journal.append(dict(event='drop_cut',consume_step=self._step,**item));continue
            output=dict(item,source_part='neck',target_part='other' if item['direction']=='DN' else 'brain',consume_step=self._step,source_bindings_sha256=self._expected)
            released.append(output);self._journal.append(dict(event='release',**output))
        self._queue=pending;self._seen={p for p in self._seen if p[0]>=self._step};return released
    def set_cut(self,direction,cut):
        self._verify();require(direction in self._fibres and type(cut) is bool,'explicit fibre/bool cut command required')
        if cut:
            self._cut.add(direction);pending=[]
            for item in self._queue:
                if item['direction']==direction:self._journal.append(dict(event='drop_inflight_on_cut',cut_step=self._step,**item))
                else:pending.append(item)
            self._queue=pending
        else:self._cut.discard(direction)
        self._journal.append({'event':'cut' if cut else 'restore','direction':direction,'step':self._step,'clock_reset':False,'physiological_cut_claim':False})
    def journal(self):
        self._verify();return json.loads(json.dumps(self._journal))
