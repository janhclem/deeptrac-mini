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
import deeptraclib as deep
from functools import partial
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pylab as plt
from cmcrameri import cm


EMULATION = False

if EMULATION:
	DIR_OUT = "./out_emulation"
	PLOT_OUT = "./plot_emulation"
else:
	DIR_OUT = "./out"
	PLOT_OUT = "./plot"

NENS = 999
PREC_WARN = 1e-14

for s_idx in range(NENS):
    print(f"[INFO] Starting scenario {s_idx} of {NENS}")

    cfg = mini.Config()
    cfg.u0 = np.random.uniform(5, 25)
    cfg.u0 = 62.66
    cfg.lx = 6371000*np.pi
    cfg.dt = 1800
    cfg.tmax = 1800*48*30
    cfg.dt_plot = cfg.dt
    cfg.show()

    atm = mini.Atm()
    atm.init(cfg)
    
    if s_idx%2:
    	atm.init_mass_gauss(cfg)
    else:
    	atm.mod_mass(cfg)

    gyre_ = partial(mini.gyre, lx=cfg.lx, u0=cfg.u0)
    kernel_ = partial(mini.kernel, beta=cfg.beta, dim=cfg.dim, d=cfg.dmix, dt=cfg.dt)

    print("[INFO] Starting time loop.")
    for t_idx, t in tqdm(enumerate(np.arange(0, cfg.tmax + cfg.dt, cfg.dt))):

	# Plotting...
        if t % cfg.dt_plot == 0:
            os.makedirs(f"{PLOT_OUT}/{s_idx}/", exist_ok=True)
            plt.figure()
            sct = plt.scatter(atm.x[:, 0] / 1000, atm.x[:, 1] / 1000,
                              c=atm.m, vmin=0, vmax=1, cmap=cm.imola, s=0.1)
            plt.colorbar(sct)
            plt.xlabel("x [km]")
            plt.ylabel("y [km]")
            plt.savefig(f"{PLOT_OUT}/{s_idx}/mass_{t_idx:03d}.png", dpi=300)
            plt.close()

        # Advection
        # vel = gyre_(atm.x)
        vel = mini.jet(atm.x,lx=cfg.lx, U0=cfg.u0, t=t)
        atm.x += vel * cfg.dt


        # First-order dispersion (random walk)
        atm.x += 2 * np.sqrt(cfg.ddiff0 * cfg.dt) * np.random.normal(0, 1, size=atm.x.shape)

        # Enforce domain boundaries
        #atm.x = np.clip(atm.x, a_min=0, a_max=cfg.lx)
        atm.x[:,0][atm.x[:,0] > cfg.lx] -=  cfg.lx
        atm.x[:,0][atm.x[:,0] < 0] += cfg.lx

        m_buffer = atm.m.copy()

        # Mixing
        if cfg.dmix > 0:
        	if EMULATION:
        		atm.m = deep.mix(atm.x, atm.m, r=cfg.lmix, m0=cfg.m0)
        	else:
            		atm.m = mini.mix(atm.x, atm.m, cfg.beta, cfg.dmix, cfg.dt, cfg.dim,
                             r_cutoff=3 * cfg.lmix)

        mass_balance = np.abs(np.sum(m_buffer - atm.m))
        if mass_balance > PREC_WARN:
            print(f"[WARNING] Mass not conserved: residual = {mass_balance:.2e}")

        os.makedirs(f"{DIR_OUT}/{s_idx}/", exist_ok=True)
        np.savez(f"{DIR_OUT}/{s_idx}/data_{t_idx:03d}.npz",
                 x=atm.x, m=m_buffer, m_=atm.m)
