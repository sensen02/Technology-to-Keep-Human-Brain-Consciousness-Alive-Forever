"""Download official BANC v888 L2/full SWCs and verify object digests."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
BUCKET = ('https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/'
          'compiled_data/banc_888/banc_banc_space_swc/')


def fetch(root):
    """Retain both official resolutions when available for ambiguous morphology."""
    assets = []
    failures = []
    for suffix in ('l2', 'skeleton'):
        filename = f'{root}_{suffix}.swc'
        url = BUCKET + filename
        for attempt in range(3):
            try:
                with urlopen(url, timeout=18) as result:
                    data, headers = result.read(), result.headers
                expected, = [field.strip()[4:] for header in headers.get_all('x-goog-hash', [])
                             for field in header.split(',') if field.strip().startswith('md5=')]
                assert base64.b64encode(hashlib.md5(data).digest()).decode() == expected
                generation = headers['x-goog-generation']
                assert generation.isdecimal()
                (HERE / 'geometry' / filename).write_bytes(data)
                assets.append({'filename': filename, 'generation': generation, 'md5': expected,
                               'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data),
                               'url': f'{url}?generation={generation}'})
                break
            except (HTTPError, URLError, OSError) as err:
                failures.append(f'{suffix}: {err}')
                if isinstance(err, HTTPError) and err.code == 404:
                    break
    return root, {'assets': assets, 'failed_requests': failures}


if __name__ == '__main__':
    ports = json.loads((HERE / 'PORTS.json').read_text())
    ids = sorted({root for _, _, *roots in ports.values() for root in roots})
    with ThreadPoolExecutor(max_workers=10) as pool:
        records = dict(pool.map(fetch, ids))
    with (HERE / 'GEOMETRY_RECEIPTS.json').open('x') as stream:
        json.dump(records, stream, indent=2)
        stream.write('\n')
    print(f'{len(ids)} attempted, {sum(bool(item["assets"]) for item in records.values())} with verified SWC')
