"""Matched physical teacher for measured-load-based leg rephasing.
All perleg geometry/gain/native force limits unchanged; continuous teacher only.
"""
import importlib.util
from pathlib import Path
import sys
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T101753Z/run_gain128.py')
spec=importlib.util.spec_from_file_location('broadlift_gain128_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T110150Z')
n.GAIT=ROOT/'GAIT.npz';n.OUT=ROOT/'neural';n.FREQ=1.


def prereg():
    n.OUT.mkdir(parents=True,exist_ok=True)
    p=n.OUT/'PREREGISTRATION.json'
    if p.exists():return
    n.dump(p,dict(importance='核心必需',stage='S5 one measured-load phase-coordination physical diagnostic',
        reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),frequency_hz=1.,
        hypothesis='MoveLHunloadingtoearlierRHstance instead oforiginalRHmidstancewhereoriginalτcaps werefrequent. Sameperlegacceptedbroadliftgeometry andnominal5stance count; expectedloadtransfer notguaranteed.',
        control='continuousalgorithmteacher, ornewreplacementANN onlyif separatelytrainedandtested; teachernotneuralgoal',
        unchanged='full71mesh49/48/42neq0/freeroot/originalgravity/contact/friction/nativeτcaps,30rad/s,finitePD128 andoriginaltimesteps',
        assessment='originalwhole3sreleaseD/sixreallegs/heading/fullreplaymaterialpointslip<=.5body perleg,0.5ms material sampling. Preserveallfailures.',
        sources={str(q):n.sha(q) for q in [Path(__file__),SOURCE,Path(n.__file__),n.GAIT,Path(__file__).with_name('rephase_load.py')]},
        units='native; SIforce/originalbrainnecessity/biologicalvalidity NOT_ESTABLISHED'))

n.prereg=prereg

if __name__=='__main__':
    prereg()
    if len(sys.argv)>1 and sys.argv[1]=='teacher':
        seconds=float(sys.argv[2]) if len(sys.argv)>2 else 4.
        tag=sys.argv[3] if len(sys.argv)>3 else 'teacher_paired_4s'
        inert=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z/MODEL.npz')
        n.simulate(inert,tag,seconds,'teacher',False)
    else:n.main()
