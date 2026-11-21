"""Independent off-target screen from BANC segmented local axon trajectories."""
from common import *
from omnisim.modules.flex_depth import GROUPS
def main():
    pat=patterns()
    # Generate a longitudinal profile for each visible root independently of all current patterns.
    with np.load(FULL/'anatomy.npz') as z:
        labels=z['labels']; lower=z['lower']; ids=z['roots'].astype(str)
    h=float(geom['cell_m'])
    classes=geom['material_classes']
    assert np.array_equal(ids,geom['material_roots'].astype(str))
    neural=np.isin(classes,GROUPS)
    obs=ElectrodeObservation(tips,[],contact_radius_m=.15e-6)
    active=np.flatnonzero(np.any(pat,axis=0))
    out={}
    for step_um in (1,2,4):
        step=int(round(step_um*1e-6/h))
        rows=[]
        for i in range(0,labels.shape[1]-step+1,step):
            piece=labels[:,i:i+step,:]
            coord=np.nonzero(piece)
            indexes=piece[coord]; n=np.bincount(indexes,minlength=len(ids)+1)
            means=np.stack([np.divide(np.bincount(indexes,weights=v,minlength=len(n)),n,
                out=np.full(len(n),np.nan),where=n>0) for v in coord],axis=1)
            rows.append((means[1:]+lower+[.5,i+.5,.5])*h)
        centers=np.stack(rows,axis=1)
        maxima=np.zeros((len(ids),4))
        covered=np.zeros(len(ids),bool)
        search=(step_um==2)
        if search:
            counters=np.full((2,len(tips)),-np.inf)
            witness=np.zeros((2,len(tips),len(tips)))
        for j in np.flatnonzero(neural):
            q=centers[j]
            valid=np.flatnonzero(np.isfinite(q[:,0]))
            for run in np.split(valid,np.where(np.diff(valid)>1)[0]+1):
                if len(run)<3:continue
                points=q[run]
                arc=np.r_[0,np.linalg.norm(np.diff(points,axis=0),axis=1).cumsum()]
                if arc[-1]<4e-6:continue
                grid=np.linspace(0,arc[-1],int(np.ceil(arc[-1]/2e-6))+1)
                local=obs._at(points,arc,grid)
                if search and ids[j] not in roots:
                    all_lead=obs._lead(local,tips)
                    kernel=np.diff(all_lead,n=2,axis=0)*(2e-6/grid[1])**2
                    for sign in (0,1):
                        kk=kernel if sign else -kernel
                        indices=np.argmax(kk,axis=0)
                        best=kk[indices,np.arange(len(tips))]
                        winner=best>counters[sign]
                        witness[sign,winner]=kernel[indices[winner]]
                        counters[sign,winner]=best[winner]
                    volts=all_lead[:,active]@pat[:,active].T
                else:
                    volts=obs._lead(local,tips[active])@pat[:,active].T
                maxima[j]=np.maximum(maxima[j],np.abs(np.diff(volts,n=2,axis=0)).max(0)*(2e-6/grid[1])**2)
                covered[j]=True
        if search:
            from scipy.spatial.distance import pdist
            # Each row is an actual curvature sample on a non-target. Maximizing over this
            # subset lower-bounds its worst excitation, upper-bounding bipolar selectivity.
            bg_floor=pdist(witness.reshape(-1,len(tips)).T,metric='chebyshev')
            bounds=[]
            for k in target:
                pts=paths[k].points_m
                arc=np.r_[0.,np.linalg.norm(np.diff(pts,axis=0),axis=1).cumsum()]
                grid=np.linspace(0,arc[-1],int(np.ceil(arc[-1]/2e-6-1e-12))+1)
                htar=np.diff(obs._lead(obs._at(pts,arc,grid),tips),n=2,axis=0)*(2e-6/grid[1])**2
                ratio=pdist(htar.T,metric='chebyshev')/bg_floor
                bounds.append(dict(target=roots[k],max_ratio_upper=float(ratio.max()),possible=int((ratio>=1.2).sum()),
                              at_least_unity=int((ratio>=1.).sum()),pairs=len(ratio)))
            write('bipole_bound.json',dict(y_bin_um=step_um,results=bounds))
        background=covered&~np.isin(ids,roots)
        out[step_um]={}
        for a,name in enumerate(arms):
            hits=background&(maxima[:,a]>=.004)
            out[step_um][name]=dict(evaluable=int(background.sum()),extra_local_crossings=int(hits.sum()),
                by_class={s:int((hits&(classes==s)).sum()) for s in np.unique(classes[hits])},
                roots=ids[hits].tolist(),target_proxy_V=[float(maxima[np.where(ids==r)[0][0],a]) for r in AN])
        print(step_um,[v['extra_local_crossings'] for v in out[step_um].values()],flush=True)
    write('background.json',out)
if __name__=='__main__':main()
