from context import tr
crop=tr.Crop('CLEAR')
env,bank=tr.build(crop)
env.reset()
phy=tr.bind_physics(env,bank)
q=tr.fit_spawn(crop,env,tr.docking.start_pose)
print("controls",env.action_spec().name)
for key in ('rostrum','haustellum','head','head_abduct','head_twist'):
 try:
  joint=phy.model.ptr.joint('walker/'+key)
  print(key, joint.range, q[joint.qposadr])
 except KeyError: pass
print('mesh',crop.skin.shape, crop.skin.reshape(-1,3).min(0),crop.skin.reshape(-1,3).max(0))
env.close()
