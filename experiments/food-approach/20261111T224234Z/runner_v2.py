"""Exact inherited physical runner; only additive input proof before any Core."""
from bindings_v2 import verify_all
import runner
runner.verify_frozen=verify_all
if __name__=='__main__':runner.main()
