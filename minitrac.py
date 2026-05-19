"""
MINITRAC is a minimal example to study particle dispersion and mixing.
    Copyright (C) 2026  Jan Clemens

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import minitraclib as mini
from functools import partial
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pylab as plt
from cmcrameri import cm

# Configure...
DIR_OUT = "./out"
PLOT_OUT = "./plot"
NENS = 300

# Loop over ensemble...
for s_idx in range(NENS):
	print(f"[INFO] Scenario {s_idx} of {NENS}")

	# Define configurations...
	cfg = mini.Config()
	cfg.u0 = np.random.uniform(5,20)
	cfg.show()

	# Initialize particles...
	atm = mini.Atm()
	atm.init(cfg)
	atm.mod_mass( cfg, method="half", axis=np.random.randint(0,1), noise=True)

	# Fix functions...
	gyre_ = partial(mini.gyre, lx=cfg.lx, u0=cfg.u0)
	kernel_ = partial(mini.kernel, beta=cfg.beta, dim=cfg.dim, d=cfg.dmix, dt=cfg.dt)

	print("[INFO] Start time loop.")
	for t_idx, t in tqdm(enumerate(np.arange( 0, cfg.tmax + cfg.dt, cfg.dt))):

		# Plot...
		if ( t%cfg.dt_plot == 0 ):
			os.makedirs(f"{PLOT_OUT}/{s_idx}/", exist_ok=True)
			plt.figure()
			sct = plt.scatter( atm.x[:,0]/1000, atm.x[:,1]/1000, c=atm.m,vmin=0, vmax=1, cmap=cm.imola, s=0.1)
			cbar = plt.colorbar(sct)
			plt.savefig(f"{PLOT_OUT}/{s_idx}/mass_{t_idx:03d}.png", dpi=300)

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
    
