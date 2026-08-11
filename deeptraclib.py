"""
DEEPTRACLIB - Deep learning emulator for particle mixing.

This library provides a Graph Neural Network (GNN) emulator for the particle
mass-transfer (mixing) scheme in MINITRAC. It replaces the deterministic
physics-based mixing with a learned statistical surrogate.

The DeepMix model follows an Encoder-Processor-Decoder paradigm with message
passing in latent space. Particles are structured as a graph where edges
connect particles within a radius aligned to the mixing length.

Copyright (c) 2026 Forschungszentrum Juelich GmbH

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

Examples
--------
>>> import deeptraclib as deeptrac
>>> import minitraclib as mini
>>> cfg = mini.Config()
>>> atm = mini.Atm()
>>> atm.init(cfg)
>>> # Create graph data
>>> data_graph = create_graph(atm.x, atm.m, cfg.lmix, cfg.m0)
>>> # Initialize model
>>> deepmix = deeptrac.DeepMix(data_graph)
>>> # Run inference
>>> u, dm = deepmix(data_graph)
"""

import torch
from torch import nn
from torch_scatter import scatter_sum
from torch_geometric.nn import MLP
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
import numpy as np


class DeepMix(torch.nn.Module):
    """
    Graph Neural Network emulator for the particle mass-transfer (mixing) scheme.

    Architecture follows an Encoder–Processor–Decoder paradigm:
      - Encoder:   embeds edge features into a 64-dimensional latent space.
      - Processor: one round of message passing updating edge and node embeddings.
      - Decoder:   projects edge embeddings back to scalar mass-exchange predictions,
                   which are summed per node to give the net mass change dm.

    Input edge features are (dx, dy, dm_mass), normalized by (lmix, lmix, m0).
    """

    def __init__(self, data):
        """
        Parameters
        ----------
        data : torch_geometric.data.Data
            A representative graph used to infer input dimensionality.
            Requires data.edge_attr to be set.
        """
        super().__init__()

        input_channels = data.edge_attr.shape[1]
        out_channels = 1

        self.encoder = MLP([input_channels, 32, 64], act='relu', norm='layer_norm')

        self.node_linear_1 = nn.Linear(64, 64)
        self.edge_linear   = nn.Linear(64, 64)
        self.edge_func     = MLP([64, 32, 64], act='relu', norm='layer_norm')

        self.node_linear_2 = nn.Linear(64, 64)
        self.msg_linear    = nn.Linear(64, 64)
        self.node_func     = MLP([64, 32, 64], act='relu', norm='layer_norm')

        self.decoder = MLP([64, 32, out_channels], norm='layer_norm')

    def forward(self, data):
        """
        Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph with node features (x), edge index, and normalized edge attributes.

        Returns
        -------
        u : torch.Tensor, shape (n_edges, 1)
            Per-edge mass-exchange predictions.
        u_agg : torch.Tensor, shape (n_nodes,)
            Net mass change per node (sum of incoming edge contributions).
        """
        x = data.x
        src, dest = data.edge_index
        edge_attr = data.edge_attr

        # Encode edge features into latent space
        e = self.encoder(edge_attr)

        # Initial node embedding: aggregate encoded edge features
        h = scatter_sum(e, dest, dim=0, dim_size=x.shape[0])

        # Message passing (single iteration)
        h_src = h[src]
        msg = self.edge_func(self.edge_linear(e) + self.node_linear_1(h_src))
        msg_agg = scatter_sum(msg, dest, dim=0, dim_size=x.shape[0])
        e = msg
        h = self.node_func(self.node_linear_2(h) + self.msg_linear(msg_agg))

        # Decode to per-edge mass exchange, aggregate to per-node dm
        u = self.decoder(e)
        u_agg = scatter_sum(u, dest, dim=0, dim_size=x.size(0)).squeeze()

        return u, u_agg


def mix(x, m, r, m0=1.0, weights_file="./deepmix.weights.0"):
    """
    Perform mixing using the deep learning emulator.

    This function constructs a graph from particle positions and masses,
    then uses the trained DeepMix GNN to predict mass changes.

    Parameters
    ----------
    x : np.ndarray, shape (n, dim)
        Particle positions (m).
    m : np.ndarray, shape (n,)
        Particle masses (kg).
    r : float
        Graph edge radius (m). Particles within this radius are connected.
    m0 : float, optional
        Normalization mass (kg). Default is 1.0.
    weights_file : str, optional
        Path to trained model weights. Default is "./deepmix.weights.0".

    Returns
    -------
    np.ndarray, shape (n,)
        Updated particle masses after mixing.

    Notes
    -----
    This function loads the model weights each time it's called. For better
    performance in production, consider loading the model once and reusing it.

    Examples
    --------
    >>> import deeptraclib as deeptrac
    >>> import numpy as np
    >>> x = np.random.uniform(0, 100000, size=(1000, 2))
    >>> m = np.ones(1000)
    >>> m_new = deeptrac.mix(x, m, r=10000, m0=1.0)
    """
    # Construct the graph from x, m and r
    f_graph = torch.from_numpy(np.concatenate([x, m[:, None]], axis=1)).float()
    pos_graph = torch.from_numpy(x).float()
    edge_index = radius_graph(pos_graph, r=r, batch=None, loop=False)
    data_graph = Data(x=f_graph, edge_index=edge_index, pos=pos_graph)
    i, j = data_graph.edge_index
    norm = torch.tensor([r, r, m0])
    data_graph.edge_attr = ((data_graph.x[j] - data_graph.x[i]) / norm).float()

    deepmix = DeepMix(data_graph)
    deepmix.load_state_dict(torch.load(weights_file, weights_only=True))
    deepmix.eval()

    return m + deepmix(data_graph)[1].detach().numpy()


def plot_graph(data_graph):
    """Visualize a PyG graph overlaid on particle positions."""
    G = to_networkx(data_graph)
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
