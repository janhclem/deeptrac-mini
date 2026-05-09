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
	  np=10_000,
	  tmax=2000,
	  beta=1.0,
	  dim=2,
	  dt=20,
	  dmix=12500,
	  ddiff0=12500,
	  ddiff1=0.0,
	  u0=None,
	  m0=1.0):
		self.lx = lx  # m
		self.np = np  # -
		self.tmax = tmax  # s
		self.beta = beta  # -
		self.dim = dim  # -
		self.dt = dt  # s
		self.dmix = dmix  # m**2/s
		self.ddiff0 = ddiff0  # m**2/s
		self.ddiff1 = ddiff1  # m**2/s
		self.u0 = lx / (np ** 0.5) / dt if u0 is None else u0  # m/s
		self.m0 = m0  # kg

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


###############################################################################


class DeepMix(torch.nn.Module):
	def __init__(self, data):
		super().__init__()
		
		# Set input and output channels to length of edge features...
		input_channels = data.edge_attr.shape[1]
		out_channels = input_channels

		# Encoder Layers...
		self.encoder = MLP([input_channels, 32, 64], act = 'relu')

		# Edge functions...
		self.node_linear_1 = nn.Linear(64,64)
		self.edge_linear = nn.Linear(64,64)
		self.edge_func = MLP([64, 32, 64], act = 'relu')

		# Node functions...
		self.node_linear_2 = nn.Linear(64,64)
		self.msg_linear = nn.Linear(64,64)
		self.node_func = MLP([64, 32, 64], act = 'relu')

		# Decoder Layers...
		self.decoder = MLP([64, 32, out_channels], act = 'relu')

	def forward(self, data):

		# Unpack and prepare data...
		x = data.x  # Node features
		src, dest = data.edge_index
		edge_attr = data.edge_attr # Edge features
		
		# Initial edge encoding...
		e = self.encoder(edge_attr) 
		
		# Initial node embedding (sum incoming edge features)
		h = scatter_sum(
		    e,
		    dest,  # Target nodes for aggregation
		    dim=0, # index along particles/nodes
		    dim_size=x.shape[0]  # Number of nodes
		)
		h_src = h[src]
		
		# Recursive message passing (1 iteration for now)
		for _ in range(1):
		
		    # Messages: Combine edge and node features
		    msg = self.edge_func(self.edge_linear(e)+self.node_linear_1(h_src))
		    
		    # Message aggregation: Sum messages for each node
		    msg_agg = scatter_sum(
			msg,
			dest,  # Target nodes
			dim=0,
			dim_size=x.shape[0]
		    )
		    
		    # Update node and edge embeddings
		    e = msg  # Update edge embeddings (or use a separate update)
		    h = self.node_func(self.node_linear_2(h) + self.msg_linear(msg_agg))

		# Predict influences (mass exchange rate)
		u = self.decoder(e)  # Edge-level predictions
		u_agg = scatter_sum(
		    u,
		    dest,  # Aggregate to target nodes
		    dim=0,
		    dim_size=x.size(0)
		)

		return u, u_agg  # Return edge-level and node-level predictions
	
def plot_graph(data_graph):

	# Convert PyG Data to NetworkX graph
	G = to_networkx(data_graph)

	# Draw the graph
	plt.figure(figsize=(10, 8))
	nx.draw(
	    G,
	    pos={i: data_graph.pos[i].numpy() for i in range(data_graph.num_nodes)},
	    node_size=600,
	    node_color="blue",
	    edge_color="gray",
	    with_labels=True,
	    width=0.5,
	)
	plt.title("Graph Visualization")
	plt.show()	
	
			
