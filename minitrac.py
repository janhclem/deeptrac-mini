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

Run with all configs from a folder (NENS = number of config files):
    python minitrac.py --config-folder configs/gyre
"""
import minitraclib as mini
import deeptraclib as deep

from tqdm import tqdm
import numpy as np
import os
from cmcrameri import cm

# Parse command line arguments...
command = mini.Command()
args = command.args
restart_s_idx = 992
stop_s_idx = 992

# Check if we should loop over a folder of configs
if args.config_folder:
	import glob
	config_files = sorted(glob.glob(f"{args.config_folder}/*.ini"))
	if not config_files:
		print(f"[ERROR] No .ini files found in folder: {args.config_folder}")
		exit(1)
	print(f"[INFO] Found {len(config_files)} configuration files in {args.config_folder}")
else:
	# Single config file mode
	config_files = [args.config]

for s_idx, config_file in enumerate(config_files):

	# Handle both folder mode and single file mode
	if args.config_folder:
		config_name = os.path.basename(config_file)[:-4]
		print(f"\n[INFO] Loading configuration {s_idx}: {config_name}")
		cfg = mini.Config.read_ini(config_file)
	else:
		print(f"[INFO] Loading configuration {s_idx}: {args.config}")
		cfg = mini.Config.read_ini(args.config)
	
	if s_idx < restart_s_idx:
		continue
	if s_idx > stop_s_idx:
		break
		
	cfg.show()
	
	# Initialize atmosphere...
	atm = mini.Atm()
	atm.init(cfg)
	dists = None
	rates = None
	
	# Select mass initialization...
	if cfg.init_type == "gradient":
		atm.init_mass_gradient(cfg)
	elif cfg.init_type == "sharp":
		atm.mod_mass(cfg)
	elif cfg.init_type == "gauss" or cfg.init_type == "gaussian":
		atm.init_mass_gauss(cfg)
	elif cfg.init_type == "three_gauss" or cfg.init_type == "three_gaussians":
		atm.init_mass_three_gauss(cfg)
	elif cfg.init_type == "smiley":
		atm.init_mass_smiley(cfg)
	elif cfg.init_type == "checkerboard" or cfg.init_type == "checkerboard_8":
		atm.init_mass_checkerboard(cfg, n_squares=8)
	elif cfg.init_type == "checkerboard_3":
		atm.init_mass_checkerboard(cfg, n_squares=3)
	elif cfg.init_type == "checkerboard_4":
		atm.init_mass_checkerboard(cfg, n_squares=4)
	elif cfg.init_type == "stripes" or cfg.init_type == "stripes_10":
		atm.init_mass_stripes(cfg, n_stripes=10)
	elif cfg.init_type == "stripes_3":
		atm.init_mass_stripes(cfg, n_stripes=3)
	elif cfg.init_type == "stripes_4":
		atm.init_mass_stripes(cfg, n_stripes=4)
	elif cfg.init_type == "random":
		atm.init_mass_random(cfg)
	else:
		print(f"[WARNING] Unknown init_type: {cfg.init_type}. Using gradient.")
		atm.init_mass_gradient(cfg)
	
	print("[INFO] Starting time loop.")
	for t_idx, t in tqdm(enumerate(np.arange(0, cfg.tmax + cfg.dt, cfg.dt))):

		# Plotting...
		if t % cfg.dt_plot == 0:
			os.makedirs(f"{cfg.dir_plot}/{s_idx}", exist_ok=True)
			atm.plot(s_idx=s_idx, z=atm.m,
			cmap=cm.oslo, levels=500, vmin=0, vmax=cfg.m0, lx=cfg.lx/1000.0,
			save_path=f"{cfg.dir_plot}/{s_idx}/mass_{t_idx:03d}.png", dpi=300)

		# Advection...
		if (cfg.flow_type == "jet"):
			vel = mini.jet(atm.x, lx=cfg.lx, U0=cfg.u0, t=t)
		elif (cfg.flow_type == "gyre"):
			vel = mini.gyre(atm.x, lx=cfg.lx, u0=cfg.u0)
		else:
			print(f"[WARNING] Flow type not available! Use 'jet' or 'gyre'.")
		atm.x += vel * cfg.dt

		# First-order dispersion (random walk)...
		atm.x += mini.dispersion(atm.x, cfg.ddiff0, cfg.dt)

		# Enforce domain boundaries...
		atm.check_boundaries(cfg, method=cfg.boundary_method)

		# Buffer to check the mass later...
		m_buffer = atm.m.copy()

		# Mixing...
		if (cfg.dmix > 0) and (t % cfg.dt_mix == 0):
			if cfg.mixing_type == 'emulation':
				atm.m = deep.mix(atm.x, atm.m, r=3*cfg.lmix, m0=cfg.m0)
			elif cfg.mixing_type == 'steering':
				atm.m, dists, rates = mini.mix_steering(atm.x, atm.m, 
				cfg.beta, cfg.dmix, cfg.dt_mix,
				cfg.dim, r_cutoff=2*cfg.lmix,
				dists_tm1=dists, lbd_c=cfg.lbd_c, w=cfg.w)
			else: 
				atm.m = mini.mix(atm.x, atm.m, cfg.beta, cfg.dmix, 
				    cfg.dt, cfg.dim, r_cutoff=2*cfg.lmix)
			
		# Check mass balance...
		mass_balance = np.abs(np.sum(m_buffer - atm.m))
		if mass_balance > cfg.prec_warn:
			print(f"[WARNING] Mass not conserved: residual = {mass_balance:.2e}")

		# Save data...
		os.makedirs(f"{cfg.dir_out}/{s_idx}/", exist_ok=True)
		np.savez(f"{cfg.dir_out}/{s_idx}/data_{t_idx:03d}.npz",
		         x=atm.x, m=m_buffer, m_=atm.m)
