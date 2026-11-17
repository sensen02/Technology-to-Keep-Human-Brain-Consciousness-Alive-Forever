"""STD-only resume300 source binding. No Torch/arrays/checkpointdeserialization."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'target_feedback_calibration_env_recovery_v1'
PYTHON=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-resume300-order-repair',action='store_true');args=parser.parse_args()
 if not args.prepare_resume300_order_repair:parser.error('Explicitresume300metadatabindingrequired')
 if Path(sys.executable).absolute()!=PYTHON.absolute() or importlib.util.find_spec('torch') is None:raise ValueError('Correctactual125712Torchenvironmentrequired; noTorchimporthere')
 import common
 target=ROOT/'INPUT_BINDINGS.json'
 if target.exists():raise FileExistsError(target)
 binding=common._read(PARENT/'INPUT_BINDINGS.json');proof=common.resume_qualification_hashes();common.verify_hashes(binding['source_hashes']);common.verify_hashes(binding['data_hashes'])
 # Parentowns othernewcorefiles. Use its explicitFILES whenavailable; never
 # regenerateoldbindings() which would askfailed300VERIFY tobePASS/final600.
 spec=importlib.util.spec_from_file_location('_orderrepair_newsource_manifest',ROOT/'sources.py');source=importlib.util.module_from_spec(spec);sys.modules[spec.name]=source;spec.loader.exec_module(source)
 files=tuple(source.FILES)
 if not all((ROOT/name).is_file() for name in files):raise ValueError('CompletenewcoreFILES required')
 hashes={**binding['source_hashes'],**proof}
 hashes.update({str(ROOT/name):common.sha(ROOT/name) for name in files})
 hashes[str(ROOT/'prepare.py')]=common.sha(ROOT/'prepare.py')
 data={**binding['data_hashes'],**proof,str(common.PARENT_CHECKPOINT):common.PARENT_SHA}
 new=dict(binding,resume_checkpoint=str(common.PARENT_CHECKPOINT),resume_checkpoint_sha256=common.PARENT_SHA,source_hashes=hashes,data_hashes=data,resume_lineage=dict(kind='RESUME300_ORDER_REPAIR_TO600',checkpoint=str(common.PARENT_CHECKPOINT),checkpoint_sha256=common.PARENT_SHA,start_update=300,total_final_update=600,new_updates=300,save_steps=[300,450,600],optimizer='RESTORE_ORIGINAL8ADAM_COUNTERS300_NO_FRESH_ADAM',initial_anchor='ORIGINAL_PARENT_PAYLOAD_CONFIG_INITIAL_ANCHOR_NO_REANCHOR',RNG='EXPLICIT_RESTORE_PAYLOAD_RNG_AFTER_ALL_SETUP_AND_OPTIMIZERLOAD_NO_RESEED',failed_old_VERIFY_preserved=True,old0qualification_passed=True,qualified_python=str(PYTHON)))
 with target.open('x') as stream:json.dump(new,stream,indent=2,allow_nan=False);stream.write('\n')
 print('RESUME300_ORDER_REPAIR_SOURCE_BINDING_PREPARED_NO_NUMERICAL_LOAD',flush=True)


if __name__=='__main__':main()
