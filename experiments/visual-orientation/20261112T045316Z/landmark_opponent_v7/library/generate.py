"""Exactfrozenoptics reuse:old672actualprefix+56newV12/V13actualRGB=728."""
from pathlib import Path
import importlib.util
from layouts import new_configs
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];BASE_ADAPTER=ROOT/'landmark_neutral_v6/library/generate.py'
def build_source():
 spec=importlib.util.spec_from_file_location('_opponent_frozen_neutral_optics',BASE_ADAPTER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);src=m.build_source()
 replacements=[("OLD=ROOT/'landmark_coverage_v5/library'","OLD=ROOT/'landmark_neutral_v6/library'"),("old['kind']!='VISION_COVERAGE_HEAD_LIBRARY_V5'","old['kind']!='VISION_NEUTRAL_HEAD_LIBRARY_V6'"),("prefix.shape!=(616,2,32,32,3)","prefix.shape!=(672,2,32,32,3)"),("fid=616+len(new)","fid=672+len(new)"),("rgb.shape!=(672,2,32,32,3)","rgb.shape!=(728,2,32,32,3)"),("rgb[:616],prefix","rgb[:672],prefix"),("unique_binocular_frames=672,reused_actual_pairs=616","unique_binocular_frames=728,reused_actual_pairs=672"),("old616pixelprefixexact=True","old672pixelprefixexact=True"),("total_actual_pairs=672,old616exact=True","total_actual_pairs=728,old672exact=True"),("HERE/'layouts.py',ORIGINAL,ROOT/'camera.py'","HERE/'layouts.py',BASE_ADAPTER,ORIGINAL,ROOT/'camera.py'")]
 for old,new in replacements:
  if src.count(old)!=1:raise ValueError('opponentactual56optics mismatch '+old)
  src=src.replace(old,new)
 return src
if __name__=='__main__':
 ns={'__name__':'_opponent_actual_optics','__file__':str(HERE/'generate.py'),'new_configs':new_configs,'BASE_ADAPTER':BASE_ADAPTER,'ORIGINAL':ROOT/'landmark_mirror_v4/library/generate.py'};exec(compile(build_source(),str(HERE/'generate.py')+'::old672new56sameoptics','exec'),ns);ns['main']()
