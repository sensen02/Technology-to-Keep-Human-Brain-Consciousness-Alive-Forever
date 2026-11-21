"""Rescore saved collision-window calls for direction, without matching neuron identity."""
import json
from pathlib import Path
from collections import Counter,defaultdict

source=Path('/run/media/sensen/Data1/omnisim-results/implant-site/banc-neck/single-array-sorting/audit.json')
data=json.loads(source.read_text())
direction={r['root']: 'D' if r['direction']=='DN' else 'U' for r in data['routes']}
groups=defaultdict(Counter)
for trial in data['single_array_sorting']['known_units']['trials']:
    true=[direction[r] for r in trial['truth']]
    called=[direction[r] for r in trial['calls']]
    counts=groups[''.join(sorted(true))]
    counts['windows']+=1
    counts['exact_presence']+=set(true)==set(called)
    counts['exact_counts']+=Counter(true)==Counter(called)
    counts['any_correct_direction']+=bool(set(true)&set(called))
    counts['no_calls']+=not called
    counts['foreign_direction']+=bool(set(called)-set(true))
print(json.dumps(dict(by_truth=groups, total=sum(groups.values(),Counter()),
                      isolated=data['single_array_sorting']['bidirectional_templates']['natural']),indent=2))
