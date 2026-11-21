"""Neural correction drives with explicit signed phase-feasible motor readout."""
from common import *
from policy_plus import Policy as Legacy

class Policy(Legacy):
    def __init__(self):
        super().__init__()
        self.register_buffer('phase_points',torch.zeros(6,5,dtype=torch.float64))
        self.register_buffer('phase_values',torch.tensor([0.,.8,0.,-.1,0.],dtype=torch.float64))

    def phase_gain(self,x):
        phase=torch.remainder(torch.atan2(x[...,212:218].double(),x[...,218:224].double()),2*np.pi)
        points=self.phase_points
        gain=torch.zeros_like(phase)
        for i in range(4):
            value=self.phase_values[i]+(self.phase_values[i+1]-self.phase_values[i])*(phase-points[:,i])/(points[:,i+1]-points[:,i])
            gain=torch.where((phase>=points[:,i])&(phase<=points[:,i+1]),value,gain)
        return gain

    def forward(self,x,base):
        raw=self.reflex(self.inputs(x,base))
        gate=(x[...,224:230].abs().amax(-1,keepdim=True)/.05).clamp(0,1)
        requested=(raw*self.lambda_scale*gate).double()
        endpoint=80*self.phase_gain(x)
        lo=torch.minimum(endpoint,torch.zeros_like(endpoint));hi=torch.maximum(endpoint,torch.zeros_like(endpoint))
        drive=torch.maximum(torch.minimum(requested,hi),lo)
        target=base+drive@self.synergy.T
        return torch.maximum(torch.minimum(target,self.maximum),self.minimum),raw,gate

def migrate(parent):
    m=Policy();state=m.state_dict()
    state.update(parent);m.load_state_dict(state)
    b=Body();b.restore(119)
    for i,leg in enumerate(b.legs):
        start,end=b.motor.preprogrammed_steps.swing_period[leg]
        m.phase_points[i]=torch.tensor([start,(start+end)/2,end+b.motor.swing_extension,(end+2*np.pi)/2,2*np.pi])
    return m

def read(path):
    saved=torch.load(path,weights_only=True,map_location='cpu');m=Policy();m.load_state_dict(saved['state']);return m.eval()

if __name__=='__main__':
    torch.set_num_threads(1);torch.manual_seed(123)
    parent=torch.load(CHECKPOINT,weights_only=True,map_location='cpu');actor=migrate(parent['state'])
    from flygym_demo.complex_terrain.hybrid_controller import _step_phase_gain
    b=Body();b.restore(119);grid=np.linspace(.0001,2*np.pi-.0001,10001)
    x=torch.zeros((len(grid),254));x[:,212:218]=torch.from_numpy(np.sin(grid[:,None])*np.ones((1,6)))
    x[:,218:224]=torch.from_numpy(np.cos(grid[:,None])*np.ones((1,6)))
    actual=actor.phase_gain(x).numpy();expected=np.array([[_step_phase_gain(t,b.motor.preprogrammed_steps.swing_period[l],b.motor.swing_extension) for l in b.legs] for t in grid])
    error=float(abs(actual-expected).max());assert error<1e-6
    out=HERE/'ROUTING_FIXED.pt';torch.save(dict(**{k:v for k,v in parent.items() if k!='state'},state=actor.state_dict()),out)
    write(HERE/'ROUTING_IDENTIFICATION.json',dict(phase_gain_max_error=error,source='Frozen teacher phase gain and max_correction=80; explicit engineering readout constraint',
        parent_sha256=sha(CHECKPOINT),checkpoint_sha256=sha(out),optimizer_updates=0,physical_steps=0))
    print('phase routing checked',error)
