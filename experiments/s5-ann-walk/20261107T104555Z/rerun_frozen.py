"""Recheck unchanged conditional-physics inputs against the final frozen verifier.
Artifact paths change only; archived V1 evidence and original method files remain.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T103309Z/verify_conditional.py')
EXPECTED='f10bb52e51befffe53748415c38ce284832c39ea52834733a25b03d755e1474a'
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T104555Z')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

if __name__=='__main__':
    assert sha(SOURCE)==EXPECTED
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'METHOD_REDIRECT.json'
    if not path.exists():
        path.write_text(json.dumps(dict(stage='conditional verifier freeze/provenance recovery',
            reason='AchildaddedvalidationafteritsREADYhandoffwhilefirstverificationwasrunning. Coreeffectivepolicy/replaycodeidentical; finalsourcevalidationmustberechecked andregistered separately.',
            old_method=str(OUT.parent/'20261107T103309Z/VERIFIER_PROVENANCE.json'),
            old_reports='teacher_conditional_4s/*_v1_ready_source.json/md retained',
            final_verifier=str(SOURCE),final_verifier_sha256=EXPECTED,
            redirect_script=str(Path(__file__)),redirect_sha256=sha(__file__),
            changes='OUT/EFFECTIVE/PROVENANCE artifactpaths only; HERE/controllerpathsandactualphysicalinputsunchanged',
            effective_code_sha256='9d0feb2313883302ddb26a1bb9c5a7ba391fed261d4770f29e68eca84d30313c',
            verification='Repeatindependentphysicsreplay ofsameactualcontrolrecords, no originalsimulationorANNrerun, no thresholdchange'),indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('final_frozen_conditional_audit',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=OUT;module.EFFECTIVE=OUT/'EFFECTIVE_VERIFIER.py';module.PROVENANCE=OUT/'VERIFIER_PROVENANCE.json'
    raise SystemExit(module.main(sys.argv[1:]))
