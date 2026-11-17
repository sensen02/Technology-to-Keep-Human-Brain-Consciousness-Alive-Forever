from fixed_neck import FixedCircuit,full_catalog
c=FixedCircuit(full_catalog())
root='720575941685691375'
for pre,count in c.ports[root]['incoming_receptors']:
 print(pre,count,c.receptor_groups[pre])
