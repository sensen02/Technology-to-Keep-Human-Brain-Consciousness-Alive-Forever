"""Parent-only actual source binding builder, no activity transport/arrays/models.

Reads real pinned BANC-v888 metadata, independent geometry checks, download receipts
and hashes referenced files. Never interprets SWC label2 as a projection direction.
"""
from pathlib import Path
import argparse
import json
from boundary import ROOTS, STATUS, checked_sources, require, sha256
HERE=Path(__file__).resolve().parent
GEOMETRY=HERE.parent/'geometry_downloads'

def build():
    paths={str(HERE/'PREREG.json'),str(HERE/'boundary.py'),str(HERE/'build_source_bindings.py')}
    fibres=[]
    for direction,root in ROOTS.items():
        metadata_path=GEOMETRY/(root+'_GEOMETRY_METADATA.json');check_path=GEOMETRY/(root+'_INDEPENDENT_GEOMETRY_CHECK.json');receipt_path=GEOMETRY/'DOWNLOAD_RECEIPTS.json'
        metadata=json.loads(metadata_path.read_text());check=json.loads(check_path.read_text());receipts=json.loads(receipt_path.read_text())
        matching=[obj for obj in receipts['objects'] if obj.get('path','').endswith('/'+root+'_skeleton.swc')]
        require(len(matching)==1,'unique pinned SWC download required');swc_path=matching[0]['path']
        paths.update((str(metadata_path),str(check_path),str(receipt_path),swc_path))
        for item in metadata['source_evidence']:
            paths.add(item['path'])
            require(sha256(item['path'])==item['sha256'],'metadata evidence drift before binding')
            if Path(item['path']).name=='NECK_POINT_AND_SAMPLE_GEOMETRY_AUDIT.json':
                underlying=json.loads(Path(item['path']).read_text()).get('source_hashes',{})
                require(bool(underlying),'geometry audit must pin underlying v888 metadata and point sources')
                for source,digest in underlying.items():
                    require(Path(source).is_absolute() and sha256(source)==digest,'underlying geometry/metadata source drift')
                    paths.add(source)
        for item in receipts['objects']:paths.add(item['path'])
        fibres.append(dict(direction=direction,root_id=root,transport_kind='SAME_AXON_IDENTITY_NO_SYNAPSE',metadata_path=str(metadata_path),independent_check_path=str(check_path),receipt_path=str(receipt_path),swc_path=swc_path,identity_basis='v888 metadata.root_888 and superclass, not SWC compartment labels',plane_y_nm=[370000.,484000.]))
    bindings=dict(schema='SOURCE_BOUND_TWO_FIBRE_TRANSPORT_V1',status=STATUS,synthetic_fixture=False,scope='Selected oneDN+oneAN boundary transmission approximation,not whole3699annotation neck,not reconstructed electrophysiology or electrode coupling',full_neck_training_allowed=False,electrophysiology_verified=False,electrode_coupling_verified=False,engineering_config=dict(dt_s=.001,delay_s=.002,delay_steps=2,activity_units='DIMENSIONLESS_ENGINEERING_ACTIVITY',activity_min=-1.,activity_max=1.,gain=1.),fibres=fibres,files=[dict(path=p,sha256=sha256(p)) for p in sorted(paths)])
    checked_sources(bindings)
    return bindings

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute-parent-source-binding',action='store_true');parser.add_argument('--output',type=Path,default=HERE/'SOURCE_BINDINGS.json');args=parser.parse_args()
    if not args.execute_parent_source_binding:print('CODE_ONLY no real source binding generated');return
    require(args.output.resolve()==HERE/'SOURCE_BINDINGS.json','only preregistered source binding output permitted')
    require(not args.output.exists(),'no blind overwrite/rebinding')
    bindings=build()
    with args.output.open('x') as handle:json.dump(bindings,handle,indent=2);handle.write('\n')
    print(json.dumps({'status':STATUS,'path':str(args.output),'sha256':sha256(args.output),'fibres':2,'training_allowed':False,'activity_transports':0,'physics_steps':0}))
if __name__=='__main__':main()
