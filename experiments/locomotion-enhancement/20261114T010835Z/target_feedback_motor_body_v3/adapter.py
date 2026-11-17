"""Minimal identity relocation of frozen body adapter; body AST still original161918."""
from pathlib import Path
import ast
import hashlib
import types

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'target_feedback_motor_body_component'
RELOCATIONS={'adapter.py':'ddf20f696338faee8d90525f396e39b765fa9948dff4de4e6d1b1a43d842fb2a','prepare.py':'0ba20c4a2eddb63aa6d63adb31a91b3b45288a17a862514850715f8a0717dba4'}


def relocated(name):
    path=PREVIOUS/name
    with path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=RELOCATIONS[name]:raise ValueError('Frozen previous body identity drift: '+str(path))
    tree=ast.parse(path.read_text())
    class Identity(ast.NodeTransformer):
        def visit_Assign(self,node):
            if name=='prepare.py' and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='lineage' and isinstance(node.value,ast.Subscript) and isinstance(node.value.value,ast.Name) and node.value.value.id=='b' and isinstance(node.value.slice,ast.Constant) and node.value.slice.value=='resume_lineage':
                node.value=ast.Call(func=ast.Name(id='_v4_lineage',ctx=ast.Load()),args=[ast.Name(id='b',ctx=ast.Load())],keywords=[])
                return node
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if node.value=='target_feedback_order_repair_v2':return ast.copy_location(ast.Constant('target_feedback_order_repair_v4'),node)
            return node
    return ast.fix_missing_locations(Identity().visit(tree))


_namespace={'__name__':'_targetfeedback_v3_frozen_adapter','__file__':str(HERE/'adapter.py')}
exec(compile(relocated('adapter.py'),str(PREVIOUS/'adapter.py')+'::v3_CAL_identity','exec'),_namespace)
globals().update({key:value for key,value in _namespace.items() if not key.startswith('__')})


def _v4_lineage(binding):
    # Actual v4 stores the reference-only resource amendment. Namespace and
    # resume declarations are read only from its SHA-pinned ancestor metadata.
    import json
    amendment={'kind':'PREFLIGHT_REFERENCE_ONLY_RSS3072_TRAINING1800','reference_preflight_RSS_MiB':3072,'training_RSS_MiB':1800,'wall_s':5400,'prior_v3_serialized_PREFLIGHT_present':False}
    if binding['resource_amendment']!=amendment:raise ValueError('Actual v4 reference3072/training1800 amendment required')
    pre_path=CAL/'PREFLIGHT.json';pre_digest=sha(pre_path);pre=json.loads(pre_path.read_text())
    if pre['all_passed'] is not True or len(pre['checks'])!=92 or not all(value is True for value in pre['checks'].values()) or pre['resource_scope']!='fullB12reference_preflight_only' or pre['preflight_RSS_limit_MiB']!=3072 or pre['training_RSS_limit_MiB']!=1800 or pre['prior_v3_exit1_preserved'] is not True:raise ValueError('Actual v4 92check reference-only preflight required')
    namespace_path=HERE.parent/'target_feedback_order_repair_v3/INPUT_BINDINGS.json'
    if binding['data_hashes'].get(str(namespace_path))!=sha(namespace_path):raise ValueError('SHA-pinned v3 namespace declaration required')
    namespace=json.loads(namespace_path.read_text())
    expected={'kind':'V2_CONTEXTLIB_GLOBALS_ENTRY_FAILURE_RECOVERY','scientific_changes':False,'original_completed_updates':300,'new_updates':300,'total600':True,'decorated_helpers_preserve_original_globals':True}
    if namespace['namespace_recovery']!=expected:raise ValueError('Actual ancestor namespace recovery identity required')
    source=HERE.parent/'target_feedback_order_repair_v2/INPUT_BINDINGS.json'
    if binding['data_hashes'].get(str(source))!=sha(source):raise ValueError('SHA-pinned original resume lineage declaration required')
    old=json.loads(source.read_text())
    if any(binding[key]!=old[key] or binding[key]!=namespace[key] for key in ('resume_checkpoint','resume_checkpoint_sha256')):raise ValueError('Actual v4 resume source differs from declared ancestors')
    if sha(pre_path)!=pre_digest or any(binding['data_hashes'].get(str(path))!=sha(path) for path in (namespace_path,source)):raise ValueError('Metadata proof drift during projection')
    return old['resume_lineage']


def prepare_namespace():
    namespace={'__name__':'_targetfeedback_v3_frozen_prepare','__file__':str(HERE/'prepare.py'),'_v4_lineage':_v4_lineage}
    exec(compile(relocated('prepare.py'),str(PREVIOUS/'prepare.py')+'::v3_CAL_identity','exec'),namespace)
    if namespace['ROOT']!=HERE or namespace['CAL']!=HERE.parent/'target_feedback_order_repair_v4':raise ValueError('Wrong v4 prepare identity')
    for value in namespace.values():
        if isinstance(value,types.FunctionType) and 'ROOT' in value.__globals__ and value.__globals__['ROOT']!=HERE:raise ValueError('Stale prepare ROOT')
    return namespace
