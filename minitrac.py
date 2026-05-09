import minitraclib as mini
from functools import partial
from tqdm import tqdm
import numpy as np
import os

# Configure...
DIR_OUT = "./out"
NENS = 500

# Loop over ensemble...
for s_idx in range(NENS):
	print(f"[INFO] Scenario {s_idx} of {NENS}")

	# Define configurations...
	cfg = mini.Config()
	cfg.u0 = np.random.uniform(0,1)
	cfg.show()

	# Initialize particles...
	atm = mini.Atm()
	atm.init(cfg)
	atm.mod_mass( cfg, method="half", axis=np.random.randint(0,1), noise=True)

	# Fix functions...
	gyre_ = partial(mini.gyre, lx=cfg.lx, u0=cfg.u0)
	kernel_ = partial(mini.kernel, beta=cfg.beta, dim=cfg.dim, d=cfg.dmix, dt=cfg.dt)

	for t_idx, t in tqdm(enumerate(np.arange( 0, cfg.tmax, cfg.dt))):

		# Advection...
		vel = gyre_(atm.x)
		atm.x += vel*cfg.dt

		# Diffusion...
		# 1th order...
		atm.x += 2*np.sqrt(cfg.ddiff0*cfg.dt)*np.random.normal(0,1,size=atm.x.shape)
		# 2d order...

		# Enforce boundaries...
		atm.x = np.clip(atm.x, a_min=0, a_max = cfg.lx)

		# Buffer for saving...
		m_buffer = atm.m.copy()

		# Mixing...
		if (cfg.dmix > 0):
			p = kernel_(atm.x)
			p /= (np.sum(p, axis=1, keepdims=True) + np.sum(p, axis=1, keepdims=True))*0.5
			dm =  atm.m[ None, :] - atm.m[:, None]
			atm.m += cfg.beta*np.sum(dm*p, axis=1)

		# Write...
		os.makedirs(f"{DIR_OUT}/{s_idx}/", exist_ok=True)

		np.savez(f"{DIR_OUT}/{s_idx}/data_{t_idx:03d}.npz", x=atm.x, m=m_buffer, m_=atm.m)
        
    
