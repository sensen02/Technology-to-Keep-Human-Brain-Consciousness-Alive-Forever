"""Strict evaluator for the measured worst-foot refinement winner.
Reuse unchanged singlecandidate runtime; redirect only frozenselection directory.
"""
import importlib.util
from pathlib import Path
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T112003Z/run_selected.py')
spec=importlib.util.spec_from_file_location('strict_single_candidate_runtime',SOURCE)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.SEARCH=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T113529Z')

if __name__=='__main__':
    r.SEARCH.mkdir(parents=True,exist_ok=True)
    p=r.SEARCH/'STRICT_ENTRYPOINT.json'
    if not p.exists():
        r.n.dump(p,dict(stage='S5 strict evaluation of measured second-stage winner',
            source=str(Path(__file__)),source_sha256=r.n.sha(__file__),
            original_single_candidate_runtime=str(SOURCE),runtime_sha256=r.n.sha(SOURCE),
            only_redirect='SEARCH directory; parameterreferencewarm/gait/nativebody/actuator unchanged',
            original_caps='same nativeforce/ratecaps andphysicalmodel, no rootforces/statewrites',
            warning='Searchmaxratioobjective isdiagnostic; fullunchangedD/realsteps/heading/matpointslip<=.5 stillrequired; teacherdoesnotcountasneuralgoal'))
    r.main()
