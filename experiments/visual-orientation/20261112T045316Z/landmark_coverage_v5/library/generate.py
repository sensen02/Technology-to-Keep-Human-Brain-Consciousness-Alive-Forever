"""Exact frozen farRGB optical algorithm, expanded to616actualpairedimages,0physics."""
from pathlib import Path
from layouts import library_configs
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ORIGINAL=ROOT/'landmark_far_v3/library/generate.py'
def build_source():
 src=ORIGINAL.read_text()
 replacements=[("cfgs=pr['library']['configs']","cfgs=library_configs(pr)"),("HERE/'generate.py',HERE/'audit.py',ROOT/'camera.py'","HERE/'generate.py',HERE/'audit.py',HERE/'layouts.py',ORIGINAL,ROOT/'camera.py'"),("rgb.shape!=(42,2,32,32,3)","rgb.shape!=(616,2,32,32,3)"),("raise ValueError('42 exact real RGB pairs')","raise ValueError('616 exact real RGB pairs')"),("unique_binocular_frames=42,raw_RGB_renders=84","unique_binocular_frames=616,raw_RGB_renders=1232"),("Image.new('RGB',(896,720),'white')","Image.new('RGB',(896,120*len(cfgs)),'white')"),("dict(pair_count=42,physics_steps=0","dict(pair_count=616,physics_steps=0")]
 for old,new in replacements:
  if src.count(old)!=1:raise ValueError('coverageoptics exactsource mismatch '+old)
  src=src.replace(old,new)
 return src
if __name__=='__main__':
 ns={'__name__':'_coverage_actual_optics','__file__':str(HERE/'generate.py'),'library_configs':library_configs,'ORIGINAL':ORIGINAL};exec(compile(build_source(),str(HERE/'generate.py')+'::sameoptics_88configs','exec'),ns);ns['main']()
