"""Reuse the frozen neural worker and archive only the integrated ticks on early stop."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CHOICE = HERE.parent / '20261119T001532Z-locomotion-choice'
sys.path.insert(0,str(CHOICE))
from live_brain import Brain, BLOCK


def main():
    config = json.loads(sys.stdin.readline())
    brain = Brain(config)
    print(json.dumps({'ready': True}),flush=True)
    for line in sys.stdin:
        observation = json.loads(line)
        if observation.get('finish'):
            brain.ticks = brain.tick
            for name in ('emitted','arrivals','levels','ground','pre','taste','feed'):
                setattr(brain,name,getattr(brain,name)[:brain.ticks])
            brain.inputs_record = brain.inputs_record[:brain.ticks//BLOCK]
            break
        result = brain.advance(observation)
        print(json.dumps(result,separators=(',',':')),flush=True)
        if brain.tick == brain.ticks:
            break
    brain.save(config['stream'])


if __name__ == '__main__':
    main()
