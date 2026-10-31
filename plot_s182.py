from pathlib import Path
import argparse, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);r=ap.parse_args().run
s=json.loads((r/'SUMMARY.json').read_text());c=json.loads((r/'CANDIDATES.json').read_text())
fig,axs=plt.subplots(1,3,figsize=(19,6))
ax=axs[0]
n6=np.array([x['n_target_rows_reachable'] for x in c]);root=np.array([max(x['root_norm_ungated'],1e-13) for x in c]);gok=np.array([x['geom_ok'] for x in c])
ax.scatter(n6[~gok]+np.random.RandomState(0).uniform(-.15,.15,(~gok).sum()),root[~gok],s=12,c='tab:gray',label='geometry FAIL (all but one)')
ax.scatter(n6[gok],root[gok],s=140,marker='*',c='tab:green',label='geometry PASS')
ax.set_yscale('log');ax.set_xlabel('number of the six target rows reachable');ax.set_ylabel('ungated root |r6|inf')
ax.legend(fontsize=8);ax.set_title('reaching all six rows is easy with degenerate contact\n(168 poses hit 6/6, all with geometry violations)')
ax=axs[1]
ks=list(range(7));v=[s['crosstab_n6_by_geom']['False'][str(k)] if isinstance(list(s['crosstab_n6_by_geom']['False'].keys())[0],str) else s['crosstab_n6_by_geom']['False'][k] for k in ks]
ax.bar(ks,v,color='tab:gray',label='geometry FAIL')
ax.bar(ks,[s['crosstab_n6_by_geom']['True'][str(k)] if isinstance(list(s['crosstab_n6_by_geom']['True'].keys())[0],str) else s['crosstab_n6_by_geom']['True'][k] for k in ks],color='tab:green',label='geometry PASS')
ax.set_xlabel('# of six target rows reachable');ax.set_ylabel('# poses');ax.legend();ax.set_title('only 1 of 481 poses passes the geometry gates')
ax=axs[2]
fd=np.array([x['geometry']['foot_depth'] for x in c]);bl=np.array([x['geometry']['bearing_legs'] for x in c])
sc=ax.scatter(fd,np.maximum(root,1e-13),c=n6,cmap='viridis',s=22)
ax.axvline(0.03442910871489635,color='r',ls='--',label='foot depth cap')
ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('max foot penetration');ax.set_ylabel('ungated root |r6|inf')
plt.colorbar(sc,ax=ax,label='# target rows reachable');ax.legend(fontsize=8)
ax.set_title('the six-row wins live at deep penetration / lost contact')
fig.suptitle('S182 geometric screen: 481 poses, 962 forwards; the interval test is not discriminative')
fig.tight_layout();fig.savefig(r/'S182_SCREEN.png',dpi=150)
print('saved S182_SCREEN.png')
