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

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils import k_hop_subgraph
from torch_scatter import scatter_sum
from torch_geometric.nn import MLP
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np

class Config():
	def __init__(self,
	  lx=100_000,
	  nump=10_000,
	  tmax=2000,
	  beta=1.0,
	  dim=2,
	  dt=20,
	  dmix=12500,
	  ddiff0=12500,
	  ddiff1=0.0,
	  u0=None,
	  m0=1.0,
	  dt_plot=2000):
		self.lx = lx  # m
		self.np = nump  # -
		self.tmax = tmax  # s
		self.beta = beta  # -
		self.dim = dim  # -
		self.dt = dt  # s
		self.dmix = dmix  # m**2/s
		self.ddiff0 = ddiff0  # m**2/s
		self.ddiff1 = ddiff1  # m**2/s
		self.u0 = lx / (nump ** 0.5) / dt if u0 is None else u0  # m/s
		self.m0 = m0  # kg
		self.lmix = 2*np.sqrt(self.dmix*self.dt/self.beta) # m
		self.dt_plot = dt_plot # s
		if (dt_plot%dt > 0):
			print("[WARNING] Plotting frequency must be a multiple of integration timestep.")

	def show(self):
		print("[INFO] Parameter:")
		for key, value in self.__dict__.items():
			print(f"{key}: {value}")
		print("[INFO] Characteristic lengths:")
		print("Mixing length", 2*np.sqrt(self.dmix*self.dt/self.beta))
		print("Advection length", self.u0*self.dt)
		print("Dispersion length (1st)", 2*np.sqrt(self.ddiff0*self.dt))
		print("Dispersion length (2nd)", 2*np.sqrt(self.ddiff1*self.dt))

class Atm():
	def __init__(self):
		self.x = None
		self.m = None
		
	def init(self, config):
		self.x = np.random.uniform(0,config.lx, size=(config.np,config.dim))
		self.m = np.ones(shape=config.np)*config.m0
		
	def mod_mass(self, config, noise=False, method="half", **method_config):
		if method == "half":
			if "axis" not in method_config.keys():
				method_config["axis"] = 0
				
			if method_config["axis"] == 0:
				self.m[self.x[:,1]<config.lx/2.0] = 0
			else:
				self.m[self.x[1, :]<config.lx/2.0] = 0
		else:
			print("Method not available")
			
		if noise:
			if "sigma" not in method_config.keys():
				method_config["sigma"] = 0.1
			self.m += np.random.normal( 0, 
						method_config["sigma"], 
						size=self.m.shape)
					
def kernel(x, beta, dim, d, dt):

    factor = 4 * np.pi * d * dt / beta
    norm_const = (1 / factor) ** (dim / 2)
    exp_factor = -beta / (4 * d * dt)

    sq_norms = np.sum(x**2, axis=1)
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2 * np.dot(x, x.T)

    sq_dist = np.maximum(sq_dist, 0)

    return norm_const * np.exp(exp_factor * sq_dist)

def gyre(x, lx, u0):

    arg_x = np.pi * x[:, 0] / lx * 2
    arg_y = np.pi * x[:, 1] / lx

    sin_x = np.sin(arg_x)
    cos_x = np.cos(arg_x)
    sin_y = np.sin(arg_y)
    cos_y = np.cos(arg_y)

    u = -np.pi * u0 * sin_x * cos_y
    v = np.pi * u0 * cos_x * sin_y

    return np.stack((u, v), axis=-1)








			
