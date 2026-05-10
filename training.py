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
import numpy as np
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils import k_hop_subgraph
from torch_scatter import scatter_sum
from glob import glob
from tqdm import tqdm
import csv
from datetime import datetime

import deeptraclib as deeptrac
import minitraclib as minitrac

# Training config...
LR = 0.000001

# Log files...
LOG_FILE = "./training.log"
with open(LOG_FILE, 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(['time', 'mse_loss'])

# Data folder...
FILES_DATA = "./out/*/*"
files = list(glob(FILES_DATA))*10
np.random.shuffle(files) # Randomize training data...

# Get default config...
cfg = minitrac.Config()

# Define particle data...
atm0 = minitrac.Atm()
atm1 = minitrac.Atm()

# Run trainings loop...
for f in tqdm(files):

    # Read the data...
    data = np.load(f)
    atm0.x = data["x"]
    atm1.x = data["x"]
    atm0.m = data["m"]
    atm1.m = data["m_"]

    # Construct global graph properties...
    f = np.concatenate([atm0.x, atm0.m[:, None]], axis=1)  # Node features: [x, y, mass]
    f_graph = torch.from_numpy(f).float()
    pos_graph = torch.from_numpy(atm0.x).float()

    # Build global graph only connecting nearby particles...
    edge_index = radius_graph(pos_graph, r=cfg.lmix, batch=None, loop=False)

    # Create global data object
    data_graph = Data(
        x=f_graph,          # Node features (x, y, mass)
        edge_index=edge_index, #
        pos=pos_graph       # Spatial positions
    )

    # Create the edge attributes and normalize...
    i, j = data_graph.edge_index
    edge_attr = data_graph.x[j] - data_graph.x[i]  # Edge features: [dx, dy, dm]
    norm = torch.tensor([cfg.lmix, cfg.lmix, cfg.m0])
    edge_attr_normalized = (edge_attr - norm/2.0)/norm
    data_graph.edge_attr = edge_attr_normalized.float()
    data_graph_local = data_graph

    deepmix = deeptrac.DeepMix(data_graph_local)

    # Select optimizer and loss function...
    optimizer = torch.optim.Adam(deepmix.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    optimizer.zero_grad() # Reset optimizer...
    _, m1 = deepmix(data_graph_local) # Forward propagation...
    loss = loss_fn(m1, torch.from_numpy(atm1.m).float()) # Calculate loss...
    loss.backward() # Backward propagation...
    optimizer.step() # Make the optimization step...

    # Log information for monitoring...
    #print("[INFO] loss: ", loss)
    with open(LOG_FILE, 'a', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow([datetime.now().isoformat(), f"{loss:.6f}"])
















