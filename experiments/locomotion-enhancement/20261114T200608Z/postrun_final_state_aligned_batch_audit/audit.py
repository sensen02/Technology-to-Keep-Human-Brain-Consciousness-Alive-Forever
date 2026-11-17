"""Exact frozen final900 audit AST; only qualification wording/resource text relocated."""
from pathlib import Path
import ast,hashlib
BASE=Path(__file__).resolve().parent.parent/'final_state_aligned_batch_audit'
EXPECTED='ceb7f5838efe6adb0439db9ce4ca051f4d7af3d7415752595088d67270779b2e'
REPLACEMENTS={
 'Predeclared final900 audit600s budget reached; no automatic extension':'Predeclared postrun final900 audit1800s budget reached; no automatic extension',
 'FINAL900_ALL3_STAGES_STORAGE600_750_900_EVAL_AND_REUSED_ACTUAL96_WALL9000':'ORIGINAL_PIPELINE_FAILED124_SAVED900_POSTRUN_QUALIFIED',
 'ACTUAL_FINAL900_COMPLETE_QUALIFICATION_ACCEPTED':'ACTUAL_POSTRUN_QUALIFIED900_ORIGINAL_FAILED124_ACCEPTED',
 '新900审核首次READY前预登记600秒/1800MiB/单CPU；不是已测速度，也不修改旧600审核或FD300秒。完整输出后的wall/RSS见RESOURCE_COMPLETION.json。':'新postrun900审核首次READY前预登记1800秒/1800MiB/单CPU；未实测新速度，不修改旧审核600/300秒及原pipeline9000秒。完整输出后的wall/RSS见RESOURCE_COMPLETION.json。'
}
def audit_tree():
    source=BASE/'audit.py'
    with source.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=EXPECTED:raise ValueError('Frozen base audit scientific source drift')
    tree=ast.parse(source.read_text());counts={key:0 for key in REPLACEMENTS}
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value in REPLACEMENTS:
            value=node.value;node.value=REPLACEMENTS[value];counts[value]+=1
    if any(value!=1 for value in counts.values()):raise ValueError('Exactly registered qualification/resource text changes required')
    class MetadataView(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Subscript) and isinstance(node.targets[0].value,ast.Name) and node.targets[0].value.id=='result' and isinstance(node.targets[0].slice,ast.Constant) and node.targets[0].slice.value=='qualification_basis':
                extra=ast.parse("result['derived_qualification_view']=tr['derived_view_fromproof']\nledgers['qualification_source_domains']={'actual_model_source_hashes':tr['source_hashes'],'actual_model_data_hashes':tr['data_hashes'],'qualification_source_hashes':tr['qualification_source_hashes'],'qualification_data_hashes':tr['qualification_data_hashes'],'qualification_mode':tr['qualification_mode'],'derived_view_fromproof':tr['derived_view_fromproof']}").body
                return [node,*extra]
            return node
    tree=MetadataView().visit(tree)
    return ast.fix_missing_locations(tree)
exec(compile(audit_tree(),str(BASE/'audit.py')+'::POSTRUN_METADATA_RESOURCE_OUTPUT_SCOPE_ONLY','exec'),globals())
