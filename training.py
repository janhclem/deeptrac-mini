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
WEIGHTS_FILE = "./deepmix.weights"
LR = 0.001
BATCH_SIZE = 8
NUM_ITERATIONS = 100000
USE_RESTART_FILE = True

# Log files...
LOG_FILE = "./training.log"
with open(LOG_FILE, 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(['time', 'mse_loss'])

# Data folder...
FILES_DATA = "./out/*/*/*/*"
files = list(glob(FILES_DATA))
np.random.shuffle(files)

# Get default config...
cfg = minitrac.Config()

# Initialize model ones...
# Get a sample to determine input dimensions...
sample_data = np.load(files[0])
atm0_sample = minitrac.Atm()
atm1_sample = minitrac.Atm()
atm0_sample.x = sample_data["x"]
atm1_sample.x = sample_data["x"]
atm0_sample.m = sample_data["m"]
atm1_sample.m = sample_data["m_"]

f_sample = np.concatenate([atm0_sample.x, atm0_sample.m[:, None]], axis=1)
f_graph_sample = torch.from_numpy(f_sample).float()
pos_graph_sample = torch.from_numpy(atm0_sample.x).float()
edge_index_sample = radius_graph(pos_graph_sample, r=cfg.lmix, batch=None, loop=False)
data_graph_sample = Data(
    x=f_graph_sample,
    edge_index=edge_index_sample,
    pos=pos_graph_sample
)
i, j = data_graph_sample.edge_index
edge_attr_sample = data_graph_sample.x[j] - data_graph_sample.x[i]
norm = torch.tensor([cfg.lmix, cfg.lmix, cfg.m0])
edge_attr_normalized_sample = edge_attr_sample/norm
data_graph_sample.edge_attr = edge_attr_normalized_sample.float()

deepmix = deeptrac.DeepMix(data_graph_sample)
if USE_RESTART_FILE:
	deepmix.load_state_dict(torch.load(WEIGHTS_FILE, weights_only=True))

# Select optimizer and loss function...
optimizer = torch.optim.Adam(deepmix.parameters(), lr=LR, weight_decay=1e-5)
loss_fn = nn.MSELoss()

# LR scheduler: decay from LR to 0.0000625 over total training steps...
from torch.optim.lr_scheduler import ExponentialLR
LR_FINAL = 0.0000625
total_steps = (NUM_ITERATIONS // len(files) + 1) * (len(files) // BATCH_SIZE)
gamma = (LR_FINAL / LR) ** (1.0 / total_steps)
scheduler = ExponentialLR(optimizer, gamma=gamma)

# Define particle data (reusable)...
atm0 = minitrac.Atm()
atm1 = minitrac.Atm()

# Run training loop...
min_loss = 1.0
epochs = NUM_ITERATIONS // len(files) + 1
print("Epochs:", epochs)
for epoch in range(epochs):
    np.random.shuffle(files)
    for ind in range(0, len(files), BATCH_SIZE):
        batch_files = files[ind:ind+BATCH_SIZE]
        
        optimizer.zero_grad()
        total_loss = 0.0
        
        for f in batch_files:
            # Read the data...
            data = np.load(f)
            atm0.x = data["x"]
            atm1.x = data["x"]
            atm0.m = data["m"]
            atm1.m = data["m_"]

            # Construct global graph properties...
            f_graph = np.concatenate([atm0.x, atm0.m[:, None]], axis=1)
            f_graph = torch.from_numpy(f_graph).float()
            pos_graph = torch.from_numpy(atm0.x).float()

            # Build global graph only connecting nearby particles...
            edge_index = radius_graph(pos_graph, r=cfg.lmix, batch=None, loop=False)

            # Create global data object
            data_graph = Data(
                x=f_graph,
                edge_index=edge_index,
                pos=pos_graph
            )

            # Create the edge attributes and normalize...
            i, j = data_graph.edge_index
            edge_attr = data_graph.x[j] - data_graph.x[i]
            norm = torch.tensor([cfg.lmix, cfg.lmix, cfg.m0])
            edge_attr_normalized = edge_attr/norm
            data_graph.edge_attr = edge_attr_normalized.float()

            # Forward propagation...
            _, dm = deepmix(data_graph)
            dm_minitrac = torch.from_numpy(atm1.m).float()-torch.from_numpy(atm0.m).float()
            dm_std = max(dm_minitrac.std().item(), 0.01)
            loss = loss_fn(dm / dm_std, dm_minitrac / dm_std)
            total_loss += loss

        (total_loss / len(batch_files)).backward()
        torch.nn.utils.clip_grad_norm_(deepmix.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Calculate and log average loss...
        avg_loss = total_loss.item() / len(batch_files)
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(deepmix.state_dict(), WEIGHTS_FILE)
        print("Epoch:", epoch, "File:", ind,"Loss:", np.sqrt(avg_loss))
        #print(dm_minitrac.abs().mean().item(), dm_minitrac.std().item())

        with open(LOG_FILE, 'a', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([datetime.now().isoformat(), f"{avg_loss:.6f}"])
 
torch.save(deepmix.state_dict(), WEIGHTS_FILE)       

