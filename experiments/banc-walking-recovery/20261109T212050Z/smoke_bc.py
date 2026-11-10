import walk_ppo_bc as B, walk_ppo as P, numpy as np, json
X,Y=B.collect(episodes=3)
print('data',X.shape,Y.shape,flush=True)
rng=np.random.default_rng(0); net=P.Net(rng)
c=B.bc_fit(net,X,Y,steps=800,rng=rng)
ev=P.evaluate(net)
print('bc mse',round(c[-1]['mse'],5),flush=True)
print(json.dumps({k:round(ev[k],4) for k in ('forward_signed_native_per_s','foot_support_min','nonfoot_force_max_W','com_height_min','slip_ratio_max','all_gates')}))
