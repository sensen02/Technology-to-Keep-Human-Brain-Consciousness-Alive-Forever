"""Check the stored arrays and draw the same registered cross-section in both layouts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from placement import DATA, SIDES, HOME_DIR

variants={'old':'cmos-17um-wide','extended':'cmos196-extended','dense':'cmos196-dense'}
placement=json.loads((HOME_DIR/'placement.json').read_text())
rows={}
for k,folder in variants.items():
    parent=DATA/folder
    report=json.loads((parent/'cmos-audit.json').read_text())
    mesh=report['mesh']['section']
    assert mesh['native_neural_roots_facing_pad']==placement[k]['root_hits'] and mesh['native_pixels_outside_chip']==0
    contacts=np.array(report['contacts_um'])
    assert len(np.unique(contacts,axis=0))==placement[k]['side']**2
    with np.load(parent/'cmos-frontend.npz') as raw,np.load(parent/'cmos-local-probes.npz') as local:
        assert raw['voltage_V'].shape[-1]==len(contacts)
        np.testing.assert_allclose(raw['contacts_m']*1e6,contacts)
        ix=[local['roots'].tolist().index(r) for r in SIDES]
        snr=local['peak_over_rms'][ix].tolist()
    front=report['frontend']
    rows[k]=dict(**placement[k],directory=str(parent),target_peak_rms=snr,
        blank_calls=front['heldout_blank_calls'],blank_windows=front['heldout_blanks'],
        detections=[v['detected'] for v in front['coverage']],trials=front['coverage'][0]['trials'],
        local_over_5rms=front['section_signal_coverage']['peak_above_5rms'])
# Replay with the new default CLI and independent output directory.
rerun=DATA/'cmos196-default-replay'
for name in ['cmos-replay','cmos-local-probes','cmos-frontend','cmos-cut-response']:
    with np.load(rerun/(name+'.npz')) as a,np.load(DATA/variants['dense']/(name+'.npz')) as b:
        assert set(a.files)==set(b.files)
        for field in a:
            np.testing.assert_array_equal(a[field],b[field])
ra=json.loads((rerun/'cmos-audit.json').read_text())
assert rows['dense']['detections']==[v['detected'] for v in ra['frontend']['coverage']]
assert rows['dense']['blank_calls']==ra['frontend']['heldout_blank_calls']
(HOME_DIR/'RESULT.json').write_text(json.dumps(dict(rows=rows,default_replay_equal=True),indent=2))
with np.load(DATA/variants['old']/'native-face.npz') as f:
    lab=f['labels']; roots=f['roots'].astype(str).tolist()
    axes=[(np.arange(lab.shape[i])+f['lower_xz'][i]+.5)*f['pitch_nm'][[0,2]][i]/1000 for i in range(2)]
    target_positions=[np.array([a[t].mean() for a,t in zip(axes,np.where(lab==1+roots.index(r)))]) for r in SIDES]
    img=(lab[::4,::3].T>0)
fig,axs=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
for ax,(k,title) in zip(axs,[('old','Original | 64 contacts'),('dense','Dense | 196 contacts')]):
    a=rows[k]; coords=np.array(json.loads((Path(a['directory'])/'cmos-audit.json').read_text())['contacts_um'])[:,[0,2]]
    ax.imshow(img,origin='lower',extent=[axes[0][0],axes[0][-1],axes[1][0],axes[1][-1]],cmap='Greys',alpha=.42,aspect='equal',interpolation='nearest')
    w=a['pad_um']; ax.add_collection(PatchCollection([Rectangle(v-w/2,w,w) for v in coords],facecolor='#168ba1',alpha=.22,edgecolor='#006072',linewidth=.6))
    for v,label,color in zip(target_positions,['L','R'],['#a21b14','#353087']):
        ax.scatter(*v,s=46,marker='+',color=color,zorder=4)
        ax.annotate(label,v,xytext=(3,5),textcoords='offset points',color=color,fontweight='bold')
    ax.set(xlim=(420,550),ylim=(94,166),xlabel='BANC x (µm)',ylabel='BANC z (µm)',title=title)
fig.suptitle('BANC cut face at y = 432 µm | registered tissue and metal pads')
fig.savefig(HOME_DIR/'CMOS196.png',dpi=155)
