"""Duration/causal diagnostics of current ANN, explicitly still slip-incomplete.
No weight/physics/gain changes and no acceptance-threshold relaxation.
"""
import importlib.util
from pathlib import Path
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T102842Z/run_student.py')
spec=importlib.util.spec_from_file_location('current_neural_motor_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
MODEL=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T102842Z/MODEL.npz')
n.OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T103607Z')

if __name__=='__main__':
    n.OUT.mkdir(parents=True,exist_ok=True)
    p=n.OUT/'PREREGISTRATION.json'
    if p.exists():raise FileExistsError('preserve existing duration evidence')
    n.dump(p,dict(importance='核心必需',stage='S5 ANNdurationandcausal-cut diagnostics, NOTfinalwalkingacceptance',
        previous='102842ANN4s screen passesallnativeD/sixlegsteps/heading butmaterialslipfails; teacheralsostillfails. No improvementassumedfromlongerwindow.',
        amendment='ExtenddurationnowonlytomeasurewhethernewANNsupport/straightsteppingisstable beyondshortscreen, andtoquantifyneuraloutputnecessity. Thischangesgatingorderoftheearlierexploratoryprotocol, notthefailurethresholds; materialslipfailurestillprecludesgoalcompletion.',
        model=str(MODEL),model_sha256=n.sha(MODEL),weights='identical to4s ANNscreen, no retraining',
        arms=[dict(tag='ann_12s',mode='ann',seconds=12.,teacher_seconds=1.,release_seconds=11.),
              dict(tag='neural_cut_hold_4s',mode='hold',seconds=4.,teacher_seconds=1.,release_seconds=3.),
              dict(tag='motor_cut_4s',mode='motor_cut',seconds=4.,teacher_seconds=1.,release_seconds=3.)],
        neural_cut='After1s, replaceall42ANNoutputs byzero rates, keep localfinitePDtargetmemory. Thisisreplacement-networkcut/hold, notoriginalconnectomeablation.',
        original_brain='Notrun inreplacementarm; originalconnectome necessity remainsNOT_ESTABLISHED',
        physical='same full71mesh/free-root/49/48/42neq0/gravity/contact/friction/nativecaps/30rad/s/finitePD128, no externalforces/rootwrites',
        deployment='Onlyweightsandcausalown-targethistory after1s, no gaitclock/reference/teacher; warmstatepreserved',
        assessment='Fullwhole11sreleaseoriginalDgates/actualsixlegsteps/heading/materialpointslip<=.5body at0.5ms/independentreplay. No droppingfailedframes, no claimingpassfromsupportalone.',
        sources={str(q):n.sha(q) for q in [Path(__file__),SOURCE,Path(n.__file__),n.GAIT,MODEL]},
        units='native; SIforce/biologicalvalidity NOT_ESTABLISHED'))
    for tag,seconds,mode in [('ann_12s',12.,'ann'),('neural_cut_hold_4s',4.,'hold'),('motor_cut_4s',4.,'motor_cut')]:
        n.simulate(MODEL,tag,seconds,mode,False)
