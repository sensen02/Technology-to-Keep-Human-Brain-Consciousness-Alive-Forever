"""Evaluate the reward-selected body adapter in the live brain/neck loop."""
import json
from body_search import HERE,make,tr
from rollout import main

if __name__=='__main__':
    weights=json.loads((HERE/'BODY_SEARCH/SELECTED.json').read_text())['weights']
    tr.docking.ReversedYawWalk=make(weights)
    main()
