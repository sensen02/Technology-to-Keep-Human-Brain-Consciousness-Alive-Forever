"""Check that missing external viewpoints do not fall back to scene state."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from solve import main
root=Path(sys.argv[1])
with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp)
    shutil.copyfile(root/'rig.json',out/'rig.json')
    cal=json.loads((root/'observations.json').read_text());cal['frames']=[211]
    (out/'observations.json').write_text(json.dumps(cal))
    images=out/'external';images.mkdir()
    for cam in ['H02_NW','H03_SW','H01_NE','H04_SE']:
        image=root/'external'/f'211_{cam}.png'
        shutil.copyfile(image,images/image.name)
    main(out)
    four=json.loads((out/'estimated.json').read_text())['211']
    assert four is not None
    for p in images.glob('*.png'):
        if p.name!='211_H02_NW.png':p.unlink()
    main(out)
    assert json.loads((out/'estimated.json').read_text())['211'] is None
    print('Four overhead cameras resolved; one camera returned unknown.')
