"""Reuse six NEW sequential B trials; no A execution or resident numerical model."""
from bindings import ROOT,OLD,module,clone
if __name__=='__main__':
 old=module('_paired_B_inherited_orchestrate',OLD/'orchestrate.py');g=dict(old.__dict__);g['ROOT']=ROOT;clone(old.main,g)()
