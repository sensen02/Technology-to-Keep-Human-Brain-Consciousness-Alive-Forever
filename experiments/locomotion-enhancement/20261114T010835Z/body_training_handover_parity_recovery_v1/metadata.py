"""Stdlib schema recovery only; original array-analysis mathematics unchanged."""
from pathlib import Path
import ast
import hashlib
import json

HERE=Path(__file__).resolve().parent
FAILED=HERE.parent/'body_training_handover_parity'
ORIGINAL_METADATA_SHA='031912094e6bd8a225a7aacd2604da5604d163014b0f403b0fd5b9e5efd577ce'
FAILED_LOG_SHA='6bb1e21f4b21a112042d7f61ad8ee65366ad896675a59814290a3e4effeb162a'


def _digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _source_window_length(result,entry,paths):
    """Real source-kind counts; missing actual_ms is never fabricated in RESULT."""
    native=result['native'];trace=result['trace'];teacher=entry['teacher'] is True
    if trace['samples']!=15400 or native['actual_native_steps']!=1232000 or native['post_native_steps']!=768000:raise ValueError('Actual trace artifact and native full-window source proofs required')
    if header(paths['trace'],'time')['shape']!=[15400]:raise ValueError('Actual saved trace header does not cover15400')
    if teacher:
        expected='STOP_FIXED_SHADOW_MATCHED_CONTROL_V1' if entry['id']=='stop' else 'FIXED_PRIVATE_SHADOW_PIVOT_TEACHER_V2'
        if result['kind']!=expected or result['condition']!=entry['id'] or result['actual_ms']!=15400 or result['actual_native_steps']!=1232000 or result['teacher_decisions']!=2880 or result['teacher_samples']['rows']!=2880 or result['full_duration'] is not True:raise ValueError('Actual teacher-kind count and sample contract mismatch')
    else:
        if result['kind']!='SOURCE_NECK_ROUGH_TERRAIN_BASELINE_V1' or result['terrain_case']!='plane_control' or result['condition']!='V_empty' or result['physical_steps']!=1232000 or result['neural_records']!=3080 or result['neural_decisions']['rows']!=3080:raise ValueError('Actual old fullNN plane-source count and sample contract mismatch')
        if 'actual_ms' in result or 'actual_native_steps' in result:raise ValueError('Archived oldNN schema unexpectedly changed; no guessed aliases')
    return trace['samples']


path=FAILED/'metadata.py'
if _digest(path)!=ORIGINAL_METADATA_SHA:raise ValueError('Frozen failed metadata source drift')
_tree=ast.parse(path.read_text());_changes=[]
class SourceSchema(ast.NodeTransformer):
    def visit_Subscript(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=='result' and isinstance(node.slice,ast.Constant) and node.slice.value=='actual_ms':
            _changes.append(node.lineno)
            return ast.copy_location(ast.Call(func=ast.Name(id='_source_window_length',ctx=ast.Load()),args=[ast.Name(id='result',ctx=ast.Load()),ast.Name(id='entry',ctx=ast.Load()),ast.Name(id='paths',ctx=ast.Load())],keywords=[]),node)
        return self.generic_visit(node)
_tree=SourceSchema().visit(_tree);ast.fix_missing_locations(_tree)
if _changes!=[103]:raise ValueError('Exactly the known source-schema field read must change')
_ns={'__name__':'_handover_recovery_single_original_metadata','__file__':str(HERE/'metadata.py'),'_source_window_length':_source_window_length}
exec(compile(_tree,str(path)+'::only_actual_source_window_schema','exec'),_ns)
globals().update({key:value for key,value in _ns.items() if not key.startswith('__')})
_original_source=_ns['_source'];_original_qualify=_ns['qualify']


def _source(entry,index):
    row=_original_source(entry,index);r=row['source_result'];p=row['sourcepaths'];teacher=entry['teacher']
    initial=read(p['initial_state']);identity=read(p['identity'])
    row['source_contract']=dict(actual_RESULT_keys=list(r),actual_VERIFY_keys=list(row['source_verify']),kind=r['kind'],actual_ms=dict(status='RECORDED',value=r['actual_ms']) if 'actual_ms' in r else dict(status='NOT_RECORDED',value=None),actual_native_steps=dict(status='RECORDED',value=r['actual_native_steps']) if 'actual_native_steps' in r else dict(status='NOT_RECORDED',value=None),native=r['native'],physical_steps=r.get('physical_steps'),teacher_decisions=r.get('teacher_decisions'),neural_records=r.get('neural_records'),trace_artifact=r['trace'],sample_artifact=r['teacher_samples' if teacher else 'neural_decisions'],full_window_evidence='trace.samples15400 + NPYtime.shape15400 + native.actual1232000/post768000 + actual kind-specific count fields; no sourceJSON mutation',initial_state_keys=list(initial),initial_state_sha256=sha(p['initial_state']),identity_keys=list(identity),identity_sha256=sha(p['identity']),normalizer_identity_note='Original binding normalizer_mapping and compiled source contract retained; absent top-level actuator/physics remain NOT_RECORDED.')
    return row
_ns['_source']=_source


def qualify():
    """Parent stdlib admission must fully pass before saved-array analysis."""
    failed_code=read(FAILED/'CODECHECK.json');verify(failed_code['source_hashes'])
    if sha(FAILED/'EXECUTION.log')!=FAILED_LOG_SHA:raise ValueError('Original failed execution log identity drift')
    if any((FAILED/name).exists() for name in ('RESULT.json','REPORT.zh-CN.md','VERIFY.json')):raise ValueError('Original failed scope unexpectedly has final artifacts; inspect separately')
    evidence={**failed_code['source_hashes'],str(FAILED/'CODECHECK.json'):sha(FAILED/'CODECHECK.json'),str(FAILED/'EXECUTION.log'):FAILED_LOG_SHA}
    verify(evidence);meta=_original_qualify()
    meta['proof_hashes']=_merge(meta['proof_hashes'],evidence)
    meta['recovery']=dict(kind='OLD_WALK_RESULT_SCHEMA_RECOVERY_ONLY',original_scope=str(FAILED),original_exit_code=1,original_error="KeyError('actual_ms') before NumPy import",original_log_sha256=FAILED_LOG_SHA,array_analysis_numeric_AST_unchanged=True)
    verify(evidence)
    for key in ('source_hashes','data_hashes','proof_hashes'):verify(meta[key])
    return meta
