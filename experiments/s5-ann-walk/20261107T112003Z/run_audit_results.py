"""Encoder-only adapter for frozen audit_results.py; no score/check changes.
The delegated audit agent owns execution. Parent reads output and does not rerun.
If an equivalent externally encoded audit already exists, preserve its data and
attach this persisted adapter's provenance without duplicating the calculation.
"""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
SOURCE=Path(__file__).with_name('audit_results.py')
EXPECTED='eed1e377f7f09fcf19fb3a19247f60c4227902c476ff4de71de4be2a9177f434'
OUTPUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T112003Z/AUDIT_SEARCH_RESULTS.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def scalar(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError('Unsupported JSON value '+type(value).__name__)


def main():
    assert sha(SOURCE)==EXPECTED,'Frozen auditor identity changed'
    provenance=dict(reason='Frozen auditor produces np.bool_ values; encoder converts numpy scalars via.item and arrays via.tolist only.',
        frozen_source=str(SOURCE),frozen_source_sha256=EXPECTED,source_modified=False,
        adapter=str(Path(__file__).resolve()),adapter_sha256=sha(__file__),score_or_check_changes=False,
        physics_steps=0,execution_owner='delegated audit agent; parent reads resulting artifact')
    if OUTPUT.exists():
        report=json.loads(OUTPUT.read_text())
        assert report.get('status')=='SEARCH_ACCOUNTING_VERIFIED_NOT_ACCEPTANCE'
        assert all(report['checks'].values())
        assert report.get('execution_adapter',{}).get('frozen_source_sha256')==EXPECTED
        for path,digest in report['input_hashes'].items():assert sha(path)==digest,path
        report['execution_adapter']=dict(provenance,reused_equivalent_completed_external_encoder_run=True)
        OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False,default=scalar)+'\n')
        print(json.dumps(dict(status=report['status'],preserved_completed_checks_and_scores=True,adapter=provenance)))
        return 0
    original=json.dumps
    def dumps(value,*args,**kwargs):
        kwargs.setdefault('default',scalar)
        if isinstance(value,dict) and 'trials' in value and 'checks' in value:
            value=dict(value,execution_adapter=provenance)
        return original(value,*args,**kwargs)
    spec=importlib.util.spec_from_file_location('frozen_search_accounting',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    json.dumps=dumps
    try:return module.main()
    finally:json.dumps=original

if __name__=='__main__':raise SystemExit(main())
