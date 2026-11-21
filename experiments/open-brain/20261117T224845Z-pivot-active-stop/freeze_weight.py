from scope import *
r=json.loads((HERE/'DEV/RESULT.json').read_text())
write(HERE/'WEIGHT_SELECTION.json',dict(checkpoint=str(WEIGHT),sha256=sha(WEIGHT),actual_updates=0,
 reason='All 10 initial independent pairs satisfy teacher-relative gates; three brake groups also absolute pass6/6; pivot shared high-stimulation failure handled by stimulus calibration only, not by changing neural or physical parameters.',
 source='frozen before held8861; stopping and pivot both use this unchanged body'))
print('frozen body',sha(WEIGHT))
