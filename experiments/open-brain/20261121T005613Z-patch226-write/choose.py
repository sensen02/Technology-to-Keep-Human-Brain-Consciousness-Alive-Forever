from common import *
from scipy.spatial.distance import pdist

def main():
    probes=[AxonProbe(p.root,p.points_m,p.speed_m_s) for p in paths]
    model=ElectrodeObservation(tips,probes,contact_radius_m=.15e-6)
    model.attach(None,None)
    q=model.activation_spacing_m
    pairs=np.triu_indices(len(tips),1)
    peaks=[]
    for _,points,arc in model._links.values():
        grid=np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]/q-1e-12)))+1)
        v=model._lead(model._at(points,arc,grid),tips)
        curvature=np.diff(v,n=2,axis=0)*(q/(grid[1]-grid[0]))**2
        peaks.append(pdist(curvature.T,metric='chebyshev'))
    peaks=np.stack(peaks)
    sides=[]
    for k in target:
        other=peaks[np.arange(len(paths))!=k].max(axis=0)
        ratio=peaks[k]/other
        j=int(np.argmax(ratio))
        inds=[int(x[j]) for x in pairs]
        threshold=probes[k].activation_V/peaks[k,j]
        sides.append(dict(root=roots[k],live_indices=inds,contact_numbers_1based=(live[inds]+1).tolist(),
                          working_A=1.2*threshold,nominal_threshold_A=threshold,
                          minimum_other_threshold_A=probes[k].activation_V/other[j],selectivity_ratio=float(ratio[j])))
    write('selection.json',dict(live_count=len(tips),candidate_bipoles=len(pairs[0]),sides=sides))
    print(sides)
if __name__=='__main__':main()
