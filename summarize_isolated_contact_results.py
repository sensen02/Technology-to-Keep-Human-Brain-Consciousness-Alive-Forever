"""Strictly read-only postprocessing; never imports reconstruction or physics."""
from pathlib import Path
import argparse
import hashlib
import json


def summarize(directory):
    root = Path(directory)
    ledger = json.loads((root/'LEDGER.json').read_text())
    records = []
    for shard in ledger.get('shards', []):
        p = Path(shard['path'])
        if p.resolve().parent != root.resolve():
            raise ValueError('shard outside run directory')
        if hashlib.sha256(p.read_bytes()).hexdigest() != shard['sha256']:
            raise ValueError('shard hash mismatch')
        records.append(json.loads(p.read_text()))
    return dict(status=ledger['status'], saved_endpoints=len(records),
                identities=ledger['identities'], attempted=ledger['attempted'],
                completed=ledger['completed'], failed=ledger['failed'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory')
    print(json.dumps(summarize(p.parse_args().directory), indent=2))
