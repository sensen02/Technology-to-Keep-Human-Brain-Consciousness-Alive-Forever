exec(open(__file__.replace('inspect_phase.py','diagnose_saved.py')).read().split('results=[]')[0])
from flygym_demo.complex_terrain.hybrid_controller import _step_phase_gain
b=Body();b.restore(1723)
with np.load(PARENT/'HELD/block015_1723/TRACE.npz',allow_pickle=False) as z:
 for t in (1.32,1.33,1.34,1.35,1.36,1.37):
  p=z['phase'][round(t*1000)]
  gains=[_step_phase_gain(p[i],b.motor.preprogrammed_steps.swing_period[l],b.motor.swing_extension) for i,l in enumerate(b.legs)]
  q,pad,raw=actor.act(z['x'][round(t*1000)],p)
  print(t,'gain',np.round(gains,3),'pads',pad,'raw',np.round(raw,3))
