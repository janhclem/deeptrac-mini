"""
MINITRAC - Main simulation driver for particle dispersion and mixing.

This script runs ensemble simulations of particle dispersion and mixing
using either the physics-based mixing scheme or the deep learning emulator.

Copyright (C) 2026 Jan Clemens

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

Usage
-----
Run with default gyre configuration:
    python minitrac.py

Run with jet configuration:
    python minitrac.py --config jet_large

Run with emulation (set emulation=true in the INI file):
    python minitrac.py --config jet_large_emulation.ini

Run with custom number of ensemble members:
    python minitrac.py --config gyre_100km --nens 10
"""
import minitraclib as mini
import deeptraclib as deep
from functools import partial
from tqdm import tqdm
import numpy as np
import os
from cmcrameri import cm


# Parse command line arguments
command = mini.Command()
args = command.args

# Load configuration from INI file
print(f"[INFO] Loading configuration: {args.config}")
cfg = mini.Config.read_ini(args.config)
cfg.show()

# Get simulation settings from config
NENS = args.nens

for s_idx in range(NENS):
    print(f"[INFO] Starting scenario {s_idx} of {NENS - 1}")

    atm = mini.Atm()
    atm.init(cfg)
    dists = None
    rates = None

    atm.init_mass_gradient(cfg)

    gyre_ = partial(mini.gyre, lx=cfg.lx, u0=cfg.u0)
    kernel_ = partial(mini.kernel, beta=cfg.beta, dim=cfg.dim, d=cfg.dmix, dt=cfg.dt)

    print("[INFO] Starting time loop.")
    for t_idx, t in tqdm(enumerate(np.arange(0, cfg.tmax + cfg.dt, cfg.dt))):

        # Plotting...
        if t % cfg.dt_plot == 0:
            atm.plot(s_idx=s_idx, z=atm.m, cmap=cm.glasgow, levels=500, vmin=0, vmax=1,
                     save_path=f"{cfg.dir_plot}/{s_idx}/mass_{t_idx:03d}.png", dpi=300)

        # Advection
        vel = mini.jet(atm.x, lx=cfg.lx, U0=cfg.u0, t=t)
        atm.x += vel * cfg.dt

        # First-order dispersion (random walk)
        atm.x += 2 * np.sqrt(cfg.ddiff0 * cfg.dt) * np.random.normal(0, 1, size=atm.x.shape)

        # Enforce domain boundaries
        atm.check_boundaries(cfg, method=cfg.boundary_method)

        m_buffer = atm.m.copy()

        # Mixing...
        if (cfg.dmix > 0) and (t % cfg.dt_mix == 0):
            if cfg.mixing_type == 'emulation':
                atm.m = deep.mix(atm.x, atm.m, r=cfg.lmix, m0=cfg.m0)
            elif cfg.mixing_type == 'steering':
                atm.m, dists, rates = mini.mix_steering(atm.x, atm.m, cfg.beta, cfg.dmix, cfg.dt_mix,
                                                      cfg.dim, r_cutoff=3 * cfg.lmix, dists_tm1=dists,
                                                      lbd_c=cfg.lbd_c, w=cfg.w)
            else:  # default
                atm.m = mini.mix(atm.x, atm.m, cfg.beta, cfg.dmix, cfg.dt, cfg.dim, r_cutoff=3 * cfg.lmix)

        mass_balance = np.abs(np.sum(m_buffer - atm.m))
        if mass_balance > cfg.prec_warn:
            print(f"[WARNING] Mass not conserved: residual = {mass_balance:.2e}")

        os.makedirs(f"{cfg.dir_out}/{s_idx}/", exist_ok=True)
        np.savez(f"{cfg.dir_out}/{s_idx}/data_{t_idx:03d}.npz",
                 x=atm.x, m=m_buffer, m_=atm.m)
