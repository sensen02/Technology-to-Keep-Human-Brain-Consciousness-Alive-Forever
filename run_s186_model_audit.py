"""S186 model-side READ-ONLY audit: why the six rows lack usable positive authority.

Reads the frozen baseline B (S177) + the mapping (S166) + the XML (no model edit, 0 forward)
and reports, per positive/negative contributor: actuator name, its tendon sites' bodies,
which joint its path crosses, its arm, and whether that slot has a pool (empty = the 16
unmapped outputs from S168).
"""
from pathlib import Path
import argparse, csv, json, re
import numpy as np
B_=Path('/home/sensen/Desktop/engine修复验收'); P=Path('/run/media/sensen/Data2/cell_wound_prototype')
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
SIX=[19,22,26,40,42,47]
def main(out):
    assert not out.exists(); out.mkdir(parents=True)
    st=np.load(B_/'s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
    mp=np.load(B_/'s166-mapped-support/20261024T231418078692647Z/MAPPING.npz')
    Bm=st['B']; T=st['T']; pool=mp['actuator_pool']; A=Bm@T
    x=XML.read_text()
    gens=[(re.findall(r'name="([^"]+)"',m.group(0))[0], re.findall(r'tendon="([^"]+)"',m.group(0))[0])
          for m in re.finditer(r'<general name="([^"]+)"[^>]*tendon="([^"]+)"[^>]*>',x)]
    assert len(gens)==90
    sites={}
    for m in re.finditer(r'<body name="([^"]+)"[^>]*>(.*?)(?=<body name=|\Z)',x,re.S):
        for sm in re.finditer(r'<site name="([^"]+)"',m.group(2)): sites[sm.group(1)]=m.group(1)
    tendons={}
    for m in re.finditer(r'<spatial name="([^"]+)"[^>]*>(.*?)</spatial>',x,re.S):
        tendons[m.group(1)]=re.findall(r'site="([^"]+)"',m.group(2))
    jb={}
    for m in re.finditer(r'<body name="([^"]+)"[^>]*>(.*?)(?=<body name=|\Z)',x,re.S):
        for jm in re.finditer(r'<joint name="([^"]+)"',m.group(2)): jb[jm.group(1)]=m.group(1)
    jname=[j.get('name') for j in __import__('xml.etree.ElementTree',fromlist=['x']).parse(XML).getroot().find('worldbody').iter('joint')][1:]
    rows=[]
    for dof in SIX:
        i=dof-6; jn=jname[i]; jbody=jb.get(jn)
        for k in range(90):
            arm=float(Bm[dof,k])
            if abs(arm)<=1e-12: continue
            name,ten=gens[k]; bs=[sites.get(s,'?') for s in tendons.get(ten,[])]
            crosses=bool(jbody in bs)
            rows.append(dict(dof=dof, joint_row=i, joint_name=jn, joint_body=jbody, slot=k, actuator=name,
                             tendon=ten, site_bodies='|'.join(dict.fromkeys(bs)), crosses_own_joint=crosses,
                             arm=arm, mapped=bool(pool[k]>=0), pool=int(pool[k]),
                             contributes_to_box=bool(pool[k]>=0), arm_after_map=(arm*0.5 if pool[k]>=0 else 0.0)))
    with (out/'MUSCLE_ATTACHMENT_AUDIT.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    per=[]
    for dof in SIX:
        sub=[r for r in rows if r['dof']==dof]
        pos=[r for r in sub if r['arm']>0]; neg=[r for r in sub if r['arm']<0]
        pos_map=sum(r['arm_after_map'] for r in pos); neg_map=sum(r['arm_after_map'] for r in neg)
        drop=[r['actuator'] for r in pos+neg if not r['mapped']]
        per.append(dict(dof=dof, joint_name=jname[dof-6], joint_body=jb.get(jname[dof-6]),
            pos_authority_mapped=float(np.maximum(A[dof],0).sum()), neg_authority_mapped=float(np.minimum(A[dof],0).sum()),
            pos_authority_mechanical=float(sum(r['arm'] for r in pos))*(-1 if False else 1),
            neg_authority_mechanical=float(sum(r['arm'] for r in neg)),
            unmapped_contributors=drop,
            unmapped_share_of_positive=(float(sum(r['arm'] for r in pos if not r['mapped'])/sum(r['arm'] for r in pos)) if pos else None),
            dominant_positive=[r['actuator'] for r in sorted(pos,key=lambda r:-r['arm'])[:2]],
            dominant_negative=[r['actuator'] for r in sorted(neg,key=lambda r:r['arm'])[:2]],
            extensor_flexor_ratio=(float(max([abs(r['arm']) for r in neg],default=0)/max([abs(r['arm']) for r in pos],default=1e-12)) if pos else None)))
    with (out/'AUTHORITY_BREAKDOWN.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per[0])); w.writeheader(); w.writerows(per)
    verdict=dict(
        dof42=dict(joint='joint_RMCoxa_pitch', finding='positive authority exists mechanically (+6.588 from RMC_pleural_remotor_and_abductor spanning Thorax->RMCoxa) but that slot (47) is one of the 16 UNMAPPED outputs, so the usable positive authority is 0',
                   ties_to='S168 16 missing outputs', classification='CHANNEL_BLOCKED_BY_MISSING_MAPPING'),
        dof47=dict(joint='joint_RMTibia_pitch', finding='positive authority is mapped (RMTibia_flex_93434, pool 9) but only +0.113 after the 0.5 gain while the requirement is +2.539 (m=22.5); the negating extensor RMTibia_extensor_93932 has arm -38.997',
                   classification='AMPLITUDE_HARD_EXTENSOR_DOMINATED'),
        other_rows='DOF 19/26/40 are the same pattern as DOF 47 (mapped but ~18-21x too weak, extensor dominated); DOF 22 needs m=1.32 on the negative side, the closest to feasible',
        model_side='the six failures are explained by (a) one unmapped positive actuator on RMCoxa_pitch and (b) a 100x-170x extensor/flexor lever-arm asymmetry on the tibia joints; neither is a solver or search artifact')
    json.dump(verdict,(out/'MODEL_AUDIT_VERDICT.json').open('w'),indent=2)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(16,6.5))
    ax=axs[0]; xs=np.arange(len(SIX)); wdt=.38
    ax.bar(xs-wdt/2,[float(p['pos_authority_mapped']) for p in per],wdt,label='mapped positive authority')
    ax.bar(xs+wdt/2,[float(p['neg_authority_mapped']) for p in per],wdt,label='mapped negative authority')
    ax.scatter(xs,[float(-(st['base']+st['J'].T@st['f0'])[dof]) for dof in SIX],c='k',zorder=5,label='required')
    ax.set_xticks(xs,[f"{p['dof']}\n{p['joint_name'].replace('joint_','')[:10]}" for p in per],fontsize=7)
    ax.axhline(0,color='k',lw=.5); ax.legend(fontsize=8); ax.set_title('mapped authority vs requirement (baseline)')
    ax=axs[1]
    labels=[]; vals=[]; cols=[]
    for p in per:
        for r in [r for r in rows if r['dof']==p['dof'] and r['arm']>0]:
            labels.append(r['actuator'][:22]); vals.append(r['arm']); cols.append('tab:red' if not r['mapped'] else 'tab:green')
    ax.barh(range(len(vals)),vals,color=cols)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels,fontsize=6)
    ax.set_xlabel('mechanical positive arm (before mapping)')
    ax.set_title('positive contributors (red = the slot has NO pool, i.e. one of the 16 missing outputs)')
    fig.suptitle('S186 read-only model audit: positive authority exists mechanically for all six rows; one is unmapped, the rest are extensor-dominated')
    fig.tight_layout(); fig.savefig(out/'S186_MODEL_AUDIT.png',dpi=150)
    json.dump(dict(verdict=verdict,per_row=per,n_rows=len(rows)),(out/'SUMMARY.json').open('w'),indent=2,default=float)
    print(json.dumps({'rows':len(rows),'per_row':[(p['dof'],round(p['pos_authority_mapped'],3),round(p['neg_authority_mapped'],3),p['unmapped_contributors']) for p in per]}))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); main(ap.parse_args().output)
