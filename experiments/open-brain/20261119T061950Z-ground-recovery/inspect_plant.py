from fixture import *
p=Plant()
m,d=p.model,p.data
print(m.nq,m.nv,m.na,m.nu,float(m.opt.timestep), m.opt.gravity)
for i,n in enumerate(p.names):
 print(i,n,m.actuator_trnid[i,:],m.actuator_gaintype[i],m.actuator_dyntype[i],m.actuator_biastype[i],
  m.actuator_gainprm[i,:3].tolist(),m.actuator_biasprm[i,:3].tolist(), m.actuator_ctrlrange[i].tolist(),m.actuator_gear[i,:3].tolist(),m.actuator_forcerange[i].tolist())
print('joint addresses',p.qa,'foot bodies',p.foot_body)
for r in (90,-90,135,-135,180,0):
 p.init(r,0,0,(1,0))
 print('seed',r,d.qpos[:7].tolist(),land.angles(d.qpos[3:7]),d.xmat[p.th,8],d.xpos[p.foot_body,2].tolist(),p.scanner.sample()[:4])
