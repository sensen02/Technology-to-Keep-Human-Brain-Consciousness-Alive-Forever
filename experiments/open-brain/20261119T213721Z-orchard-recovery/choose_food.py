import json
from pathlib import Path
d=Path(__file__).resolve().parent
groups=[]
for file in ('FLANK_SEARCH.json','CROUCH_SEARCH.json'):
 if (d/file).exists():
  for row in json.loads((d/file).read_text()):
   row.setdefault('lift',0.)
   if row['continuous_contact_s']>.085:groups.append(row)
if not groups:raise SystemExit(2)
chosen=max(groups,key=lambda r:(r['continuous_contact_s'],r['standing']))
(d/'FOOD_CONFIG.json').write_text(json.dumps(chosen,indent=2)+'\n')
