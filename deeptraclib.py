"""
DEEPTRAC is a minimal example to study particle dispersion and mixing.
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

class DeepMix(torch.nn.Module):
	def __init__(self, data):
		super().__init__()
		
		# Set input and output channels to length of edge features...
		input_channels = data.edge_attr.shape[1]
		out_channels = 1

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
		self.decoder = MLP([64, 32, out_channels])

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

		# Recursive message passing (1 iteration for now)
		for _ in range(1):
		    
		    # Set source node embeddings...
		    h_src = h[src]

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
		).squeeze()

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











