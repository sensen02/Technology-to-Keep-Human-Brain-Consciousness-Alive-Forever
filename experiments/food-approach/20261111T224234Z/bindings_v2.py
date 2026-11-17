"""Additive immutable input binding; inherited v1 code and evidence remain unchanged."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify_all():
 from runtime import verify_frozen
 first=verify_frozen();extra=json.loads((ROOT/'FROZEN_SUPPLEMENT.json').read_text())
 for p,h in extra['source_hashes'].items():
  if sha(p)!=h:raise ValueError('v2 frozen supplement drift '+p)
 merged=dict(first);merged['source_hashes']={**first['source_hashes'],**extra['source_hashes'],str(ROOT/'FROZEN_SUPPLEMENT.json'):sha(ROOT/'FROZEN_SUPPLEMENT.json')}
 return merged
