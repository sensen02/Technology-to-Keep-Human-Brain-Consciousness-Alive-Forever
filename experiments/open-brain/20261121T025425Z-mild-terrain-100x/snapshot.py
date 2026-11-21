"""Score an existing progress checkpoint without influencing further training."""
from practice import HERE, PARENT, CHECKPOINT, assess, torch, write


def main():
    state = torch.load(CHECKPOINT, weights_only=True, map_location='cpu')
    target = HERE / 'milestones' / f"update-{state['step']}"
    target.mkdir(parents=True, exist_ok=True)
    torch.save(state['model'], target / 'weights.pt')
    (target / 'skeleton.json').write_bytes((PARENT/'skeleton.json').read_bytes())
    write(target / 'snapshot.json', dict(step=state['step'], sample=state['reports'][-1]))
    assess(target)
    print(target)


if __name__=='__main__':
    main()
