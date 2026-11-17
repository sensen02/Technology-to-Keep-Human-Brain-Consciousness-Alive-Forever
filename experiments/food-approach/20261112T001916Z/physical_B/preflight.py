"""Reuse bounded candidate2 loader/namespace checks in new candidate3 scope."""
from bindings import ROOT,OLD,module,clone
if __name__=='__main__':
 old=module('_paired_B_inherited_preflight',OLD/'preflight.py');g=dict(old.__dict__);g['ROOT']=ROOT;clone(old.main,g)()
