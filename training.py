import torch
import numpy as np
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils import k_hop_subgraph
from torch_scatter import scatter_sum
from glob import glob

import deeptraclib as deeptrac
import minitraclib as minitrac

# Get default config...
cfg = minitrac.Config()

# Load particle data...
atm0 = minitrac.Atm()
atm1 = minitrac.Atm()

FILES_OUT = "./out/*/*"
files = glob(FILES_OUT)

data = np.load(files[0])
atm0.x = data["x"]
atm1.x = data["x"]
atm0.m = data["m"]
atm1.m = data["m_"]

x = atm0.x
m = atm0.m

# Construct global graph
f = np.concatenate([x, m[:, None]], axis=1)  # Node features: [x, y, mass]
f_graph = torch.from_numpy(f).float()
pos_graph = torch.from_numpy(x).float()

# Build global graph only connecting nearby particles...
edge_index = radius_graph(pos_graph, r=cfg.lx, batch=None, loop=False)

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

deepmix = deeptrac.DeepMix(data_graph_local)
print(deepmix)

# Select optimizer and loss function...
optimizer = torch.optim.Adam(deepmix.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()

optimizer.zero_grad() # Reset optimizer...
_, m1 = deepmix(data_graph_local) # Forward propagation...
loss = loss_fn(m1, torch.from_numpy(atm1.m).float()) # Calculate loss...
loss.backward() # Backward propagation...
optimizer.step() # Make the optimization step...

print(loss)
		
