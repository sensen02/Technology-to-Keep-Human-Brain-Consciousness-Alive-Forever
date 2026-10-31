"""S289 full-body visualization from saved trajectories only."""
from pathlib import Path
import importlib.util
p=Path('/run/media/sensen/Data2/cell_wound_prototype/render_s284_body.py')
spec=importlib.util.spec_from_file_location('s289_visual_base',p)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ARMS=('decoder_original','decoder_range_x2')
if __name__=='__main__':raise SystemExit(module.main())
