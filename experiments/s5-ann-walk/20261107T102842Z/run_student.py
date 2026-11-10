"""Exploratory neural imitation screen of the improved but slip-incomplete teacher.
No teacher-pass claim. Final student is judged independently by original physical gates.
"""
import importlib.util
from pathlib import Path
import sys
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T101753Z/run_gain128.py')
spec=importlib.util.spec_from_file_location('improved_lift_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
n.OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T102842Z')


def prereg():
    n.OUT.mkdir(parents=True,exist_ok=True)
    path=n.OUT/'PREREGISTRATION.json'
    if path.exists():return
    n.dump(path,dict(importance='核心必需',stage='S5 exploratory replacement-ANN imitation, teacher remains slip-incomplete',
        context='Newreference+gain128 passesD/sixlegsteps/heading; slipratios [.2491,.3544,.7531,.3039,.5216,.7440], so notacceptedlow-slipteacher. Explicitlyparallel student screen, not declarationteacherpassed.',
        reason='Teacherisphysicallysupported andmovesconsistently. Evaluatelearnedcontinuouscausaldynamics independently while remainingcontacttrackingisdiagnosed; learningmayhelpordeteriorate andbothoutcomesretained.',
        model='learnedMLP294->128tanh->128tanh->42 nexttarget, 40framecausalownjointtargethistory7lags, normalizedperjoint',
        training='3000fixedoptimizationsteps, referenceofflinemovementtargets+perturbedhistory+multistepselfrollouts; no biologicalbrainclaim',
        deployment='weights+causaltargethistory only; genuinewarmhistoryfilledfromt0; samephysicalteacherfirst1s thenANNwithouttime/phase/lookup/teacher. No root/body assist or proprioceptivebranchhiddeninactuator.',
        original_brain='NOTRUN; replacement learned motor-network arm, originalbrainnecessity notclaimed',
        reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),frequency_teacher_hz=1.,
        physics='original49/48/42/71/free root/neq0, gravity[0,0,-9801], unchangedcontact/friction/nativeforcecaps and30rad/s targetrate; finitePDgain128',
        assessment='Offline20sfree-neuraldynamicscheck then4sphysics screen; independentallnativeD/realsteps/heading/matpointslip at0.5ms. Ifcandidate passesextend>=10sANNrelease+zero-output/neuralcut hold/motorcut anddeclaredperturbation.',
        thresholds='allunchanged; no inferredstudentphaseclock injected, no droppingfailedframes',
        sources={str(q):n.sha(q) for q in [Path(__file__),SOURCE,Path(n.__file__),n.GAIT]},
        units='native; absoluteSIforce/biologicalvalidity NOT_ESTABLISHED'))

n.prereg=prereg

if __name__=='__main__':n.main()
