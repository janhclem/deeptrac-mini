"""
Training script for the DeepMix GNN emulator.

This script trains the DeepMix Graph Neural Network to emulate the
particle mass-transfer (mixing) scheme from MINITRAC. It loads training
data from MINITRAC ensemble runs, constructs graphs, and trains the model
using supervised learning.

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
Run training with default parameters:
    python training.py

Run with custom parameters:
    python training.py --batch-size 16 --iterations 100000

Notes
-----
Training data should be generated first by running minitrac.py to create
NPZ files in the ./out/ directory. The script expects files matching the
pattern ./out/*/*.npz.
"""

import csv
from datetime import datetime
from glob import glob

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

import deeptraclib as deeptrac
import minitraclib as minitrac

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
WEIGHTS_FILE    = "./deepmix.weights"
LR              = 0.00003 #0.0003
LR_FINAL        = 0.0000625
BATCH_SIZE      = 8
NUM_ITERATIONS  = 500_000
USE_RESTART     = True
LAMBDA          = 0.01   # mass-conservation penalty weight

LOG_FILE        = "./log/training.log"
FILES_DATA      = "./out/*/*"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
with open(LOG_FILE, 'w', newline='') as log_file:
    writer = csv.writer(log_file)
    writer.writerow(['time', 'mse_loss'])

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
files = list(glob(FILES_DATA))
np.random.shuffle(files)

cfg = minitrac.Config()

# ---------------------------------------------------------------------------
# Build a sample graph to infer model input dimensions
# ---------------------------------------------------------------------------
sample_data = np.load(files[0])
atm0_sample = minitrac.Atm()
atm0_sample.x = sample_data["x"]
atm0_sample.m = sample_data["m"]

f_sample      = np.concatenate([atm0_sample.x, atm0_sample.m[:, None]], axis=1)
f_graph       = torch.from_numpy(f_sample).float()
pos_graph     = torch.from_numpy(atm0_sample.x).float()
edge_index    = radius_graph(pos_graph, r=cfg.lmix, batch=None, loop=False)
data_graph    = Data(x=f_graph, edge_index=edge_index, pos=pos_graph)
i, j          = data_graph.edge_index
norm          = torch.tensor([cfg.lmix, cfg.lmix, cfg.m0])
data_graph.edge_attr = ((data_graph.x[j] - data_graph.x[i]) / norm).float()

# ---------------------------------------------------------------------------
# Model, optimizer, scheduler
# ---------------------------------------------------------------------------
deepmix = deeptrac.DeepMix(data_graph)
if USE_RESTART:
    deepmix.load_state_dict(torch.load(WEIGHTS_FILE, weights_only=True))

optimizer = torch.optim.Adam(deepmix.parameters(), lr=LR, weight_decay=1e-5)
loss_fn   = nn.HuberLoss()

epochs      = NUM_ITERATIONS // len(files) + 1
total_steps = epochs * (len(files) // BATCH_SIZE)
gamma       = (LR_FINAL / LR) ** (1.0 / total_steps)
scheduler   = ExponentialLR(optimizer, gamma=gamma)

# ---------------------------------------------------------------------------
# Global dm normalization constant (estimated from a data sample)
# ---------------------------------------------------------------------------
dm_samples = [np.load(f)["m_"] - np.load(f)["m"] for f in files[:200]]
DM_STD = float(np.concatenate(dm_samples).std())
print(f"[INFO] Global dm_std: {DM_STD:.6f}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
atm0 = minitrac.Atm()
atm1 = minitrac.Atm()
min_loss = 1.0

print(f"[INFO] Training for {epochs} epochs over {len(files)} files (batch size {BATCH_SIZE}).")

for epoch in range(epochs):
    np.random.shuffle(files)

    for ind in range(0, len(files), BATCH_SIZE):
        batch_files = files[ind:ind + BATCH_SIZE]

        optimizer.zero_grad()
        total_loss = 0.0
        total_mse  = 0.0

        for f in batch_files:
            data = np.load(f)
            atm0.x = data["x"]
            atm1.x = data["x"]
            atm0.m = data["m"]
            atm1.m = data["m_"]

            f_graph    = torch.from_numpy(np.concatenate([atm0.x, atm0.m[:, None]], axis=1)).float()
            pos_graph  = torch.from_numpy(atm0.x).float()
            edge_index = radius_graph(pos_graph, r=cfg.lmix, batch=None, loop=False)
            data_graph = Data(x=f_graph, edge_index=edge_index, pos=pos_graph)

            i, j = data_graph.edge_index
            norm = torch.tensor([cfg.lmix, cfg.lmix, cfg.m0])
            data_graph.edge_attr = ((data_graph.x[j] - data_graph.x[i]) / norm).float()

            _, dm = deepmix(data_graph)
            dm_target = torch.from_numpy(atm1.m).float() - torch.from_numpy(atm0.m).float()

            loss = loss_fn(dm / DM_STD, dm_target / DM_STD)
            #mass_penalty = (dm.sum() / DM_STD) ** 2
            mass_penalty = (dm.sum() / (DM_STD * torch.sqrt(torch.tensor(data_graph.num_nodes, dtype=torch.float)))) ** 2
            total_loss += loss + LAMBDA * mass_penalty
            total_mse  += loss.item()
            print(f"loss={loss.item():.5f}  raw_penalty={mass_penalty.item():.5f}  weighted={LAMBDA*mass_penalty.item():.5f}  N={data_graph.num_nodes}")

        (total_loss / len(batch_files)).backward()
        torch.nn.utils.clip_grad_norm_(deepmix.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        avg_loss = total_mse / len(batch_files)
        nrmse    = np.sqrt(avg_loss)
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(deepmix.state_dict(), WEIGHTS_FILE)

        print(f"Epoch {epoch:04d}  file {ind:06d}  NRMSE {nrmse:.4f}")

        with open(LOG_FILE, 'a', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([datetime.now().isoformat(), f"{avg_loss:.6f}"])

torch.save(deepmix.state_dict(), WEIGHTS_FILE)
print(f"[INFO] Training complete. Best NRMSE: {np.sqrt(min_loss):.4f}")
