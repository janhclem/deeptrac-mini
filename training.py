import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils import k_hop_subgraph
from torch_scatter import scatter_sum

import mixlib
from minitrac import DeepMix

# Settings
LX = 100_000  # m
NP = 10_000   # Number of particles
DIM = 2       # Spatial dimensions
M0 = 1        # kg (default mass)
LMIX = 1400   # m (radius for graph construction)

# Initialize dummy particles
x = np.random.uniform(0, LX, size=(NP, DIM))  # Spatial coordinates
m = np.zeros(NP) * M0
m[x[:, 1] < LX / 2.0] = 1  # Assign mass=1 to nodes below LX/2

# Construct global graph
f = np.concatenate([x, m[:, None]], axis=1)  # Node features: [x, y, mass]
f_graph = torch.from_numpy(f).float()
pos_graph = torch.from_numpy(x).float()

# Build global graph only connecting nearby particles...
edge_index = radius_graph(pos_graph, r=LMIX, batch=None, loop=False)

# Create global Data object
data_graph = Data(
    x=f_graph,          # Node features (x, y, mass)
    edge_index=edge_index, # 
    pos=pos_graph       # Spatial positions
)

# Create the edge attributes...
i, j = data_graph.edge_index
edge_attr = data_graph.x[j] - data_graph.x[i]  # Edge features: [dx, dy, dm]
data_graph.edge_attr = edge_attr				   
data_graph_local = data_graph

# Print info
#mixlib.plot_graph(data_graph_local)
print(f"Global graph: {data_graph.num_nodes} nodes, {data_graph.num_edges} edges")
print(f"Local subgraph: {data_graph_local.num_nodes} nodes, {data_graph_local.num_edges} edges")
print(f"Node features shape: {data_graph_local.x.shape}")
print(f"Edge features shape: {data_graph_local.edge_attr.shape}")
print(f"Position shape: {data_graph_local.pos.shape}")

mix_model = NeuraMix(data_graph_local)
print(mix_model)

# Forward pass
edge_predictions, node_predictions = mix_model(data_graph_local)

print(edge_predictions)
print(node_predictions)

			
################################################################################

# Training loop:

# Optimizer = Adam
# Loss Function = Least mean square

# Randomly select ....
# Loop over different scenarios... # 5
#   Loop over different time steps... # 200
#      Loop over different particles... # 100_000
# ---> a lot learning material...
	
# Validation loop ...
# Reference simulation ...	
		
################################################################################
# Preparation? 
# Calculate the mass and position distances dimension wise... they become the initial edge features.


# Then the neural network makes an embedding of this distances. 
# Then the embedding is summated to aggregate it and create a node embedding.
# Then we can update the messages, based on a learnable function. 




		
		
