"""ReusefrozenactualRGBprefixappendopticsalgorithm;old616exact+56new672total."""
from pathlib import Path
from layouts import new_configs
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ORIGINAL=ROOT/'landmark_mirror_v4/library/generate.py';OLD=ROOT/'landmark_coverage_v5/library'
def build_source():
 src=ORIGINAL.read_text()
 replacements=[("OLD=ROOT/'landmark_far_v3/library'","OLD=ROOT/'landmark_coverage_v5/library'"),("old['kind']!='VISION_FAR_HEAD_LIBRARY_V3'","old['kind']!='VISION_COVERAGE_HEAD_LIBRARY_V5'"),("prefix.shape!=(42,2,32,32,3)","prefix.shape!=(616,2,32,32,3)"),("configs=pr['library']['new_configs']","configs=new_configs(pr)"),("fid=42+len(new)","fid=616+len(new)"),("rgb.shape!=(56,2,32,32,3)","rgb.shape!=(672,2,32,32,3)"),("rgb[:42],prefix","rgb[:616],prefix"),("unique_binocular_frames=56,reused_actual_pairs=42,new_actual_pairs=14,new_RGB_renders=28","unique_binocular_frames=672,reused_actual_pairs=616,new_actual_pairs=56,new_RGB_renders=112"),("old42pixelprefixexact=True","old616pixelprefixexact=True"),("Image.new('RGB',(896,240),'white')","Image.new('RGB',(896,120*len(configs)),'white')"),("HERE/'generate.py',HERE/'audit.py',ROOT/'camera.py'","HERE/'generate.py',HERE/'audit.py',HERE/'layouts.py',ORIGINAL,ROOT/'camera.py'"),("total_actual_pairs=56,old42exact=True,new14=True","total_actual_pairs=672,old616exact=True,new56=True"),("newRGBpairs14=True","newRGBpairs56=True"),("NEW14_PANEL.png","NEW56_PANEL.png")]
 for old,new in replacements:
  expected=2 if old=='NEW14_PANEL.png' else 1
  if src.count(old)!=expected:raise ValueError('neutral new56optics source shape '+old)
  src=src.replace(old,new)
 # Existinggenerator previews allnewrows;filename retained aslegacy name and explicitnewname addedbyparentaudit.
 return src
if __name__=='__main__':
 ns={'__name__':'_neutral_new56_optics','__file__':str(HERE/'generate.py'),'new_configs':new_configs,'ORIGINAL':ORIGINAL};exec(compile(build_source(),str(HERE/'generate.py')+'::old616plusnew56','exec'),ns);ns['main']()
