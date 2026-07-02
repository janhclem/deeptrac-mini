# Lagrangian Transport Modeling: Prototyping New Mixing Methods and Deep Emulators

This repository contains **MINITRACLIB** and **DEEPTRACLIB** — a suite of tools for exploring new methods in particle dispersion modeling and for investigating deep learning within Lagrangian frameworks. 

| Component | Description |
|---|---|
| **MINITRACLIB** | Lightweight prototype simulating advection, dispersion, and mixing of particles in a 2D domain. |
| **DEEPTRACLIB** | Neural emulators designed to replace MINITRAC's deterministic mixing scheme with a learned surrogate. |

---

## Contents

1. [Wind Fields and Advection](#0-wind-fields-and-advection)
2. [Numerical Mixing: Mass-Transfer Scheme](#1-numerical-mixing-mass-transfer-scheme)
3. [Numerical Dispersion: Random Walks](#2-numerical-dispersion-random-walks)
4. [Emulation: Deep Graph Neural Networks](#3-emulation-deep-graph-neural-networks)
5. [References](#references)

---

## 1. Wind Fields and Advection

A simple analytical double-gyre wind field drives filamentation on a 100 km × 100 km domain. Particles are advected using the gyre's analytical velocity functions with a first-order Euler integration scheme. The examples below show different simulation results for four different setups: First, only with advection, second with advection but brownian motion added (dispersion), second with the mass-transfer mixing scheme, and the last one with mixing and dispersion.

| Setup | Figure |
|---|---|
| Advection only | ![Advection](https://github.com/janhclem/deeptrac-mini/blob/master/doc/figures/advection_only.png) |
| Advection + Dispersion | ![Advection + Dispersion](https://github.com/janhclem/deeptrac-mini/blob/master/doc/figures/advection_dispersion.png) |
| Advection + Mixing | ![Advection + Mixing](https://github.com/janhclem/deeptrac-mini/blob/master/doc/figures/advection_mixing.png) |
| Advection + Dispersion + Mixing | ![Advection + Dispersion + Mixing](https://github.com/janhclem/deeptrac-mini/blob/master/doc/figures/advection_mixing_dispersion.png) |

---

## 2. Numerical Mixing: Mass-Transfer Scheme

Particle-mass-transfer algorithms calculate inter-particle mixing within a Lagrangian transport framework. They approximate the Eulerian advection-diffusion-reaction equation without introducing artificial numerical diffusion and, when operated in a numerically stable regime, are strictly mass-conserving. The mixing kernels used between particles are identical to the kernel representations found in **Smoothed Particle Hydrodynamics (SPH)**. 

These algorithms have been successfully parallelized and domain-decomposed (Benson et al.). The kernel used here is a **Gaussian**, motivated by the **Green's function** of the diffusion equation: an injected particle begins as a delta-function pulse and disperses into a Gaussian distribution over a sufficiently small, yet non-negligible, time step.

This implementation assumes **isotropy and homogeneity** of the diffusion coefficient. In principle the kernel can be formulated anisotropically to encode spatially varying mixing strengths.

### Algorithm

Each particle is an idealized representative of a finite atmospheric area. Uncertainty arises from two sources:
1. Unresolved sub-grid scale winds.
2. Representing a finite area with a single particle.

In the thermodynamic limit, a parcel large enough to be thermodynamically meaningful is also susceptible to disintegration — this process must be modeled. The Green's function of diffusion over a small time step is a Gaussian; the parcel "disintegrates" and its mass mixes into neighboring particles. When multiple particles are present, their respective Gaussian distributions overlap, quantifying mixing in a rigorous manner.

1. First, study the dispersion of a delta pulse and derive the Gaussian.
2. Formulate the kernel.
3. Create the double stochastic exchange matrix.

> **Further reading:** Benson et al. (2024) describe the algorithm and its parallelization in detail.

---

## 3. Numerical Dispersion: Random Walks

While the mixing scheme parameterizes parcel disintegration, the **random walk** component parameterizes sub-grid scale wind effects. On limited-resolution grids, exact details of the velocity field are unknown at sub-grid scales; we therefore calculate statistics over the possible trajectories parcels may follow.

Sub-grid scale diffusion acts as an additional correction to the center of mass of each particle's Gaussian distribution. Each random walk step effectively shifts the mean of the Gaussian associated with an individual particle.

> **Future work:** Statistical representations of sub-grid scale winds or super-resolution techniques (e.g., AI-driven models such as **AtmoRep**) could improve these schemes.


## 4. Emulation: Deep Graph Neural Networks

**DEEPTRAC** uses a Graph Neural Network (GNN) to replace the deterministic mixing scheme with a learned statistical surrogate, introducing GNNs into a Lagrangian modeling framework.

### Model Architecture

**DeepMix** follows an **Encoder–Processor–Decoder** paradigm with recursive message passing in latent space. Hence it is a neural network that build up a latent space with extended coordinates, and a processor that learns the dynamics of the system. The enlargement of the dimensionality in the encoder helps the processor to learn a proper representation of the dynamics. Intuitively this becomes necessary, as we need to learn a complex gaussian kernel (including distances between particles, the kernel smoothness and the diffusity parameter), which requieres more dimensions for reasoning.

Particles are structured as a graph:
- **Connectivity:** Edges connect particles within a radius $r$ aligned to the mixing length $l_\mathrm{mix}$, balancing local information gathering against computational cost.
- **Edge features:** Normalized coordinate differences $(dx, dy)$ and mass difference $dm$, scaled by $(l_\mathrm{mix},\, l_\mathrm{mix},\, m_0)$.
- **Objective:** Predict the net mass change $\Delta m$ per particle.

**Processing steps:**

| Step | Description |
|---|---|
| Embedding | Edge features (3D) are encoded into a 64D latent space via an MLP with LayerNorm. |
| Message passing | A single round of edge–node message passing updates latent edge and node embeddings. |
| Decoding | Edge embeddings are decoded to scalar mass-exchange values and aggregated per node. |

### Training Setup

Training data are generated by MINITRAC ensemble runs. 999 model runs with 100 time steps each have been performed. Each model run has a different initial positioning of the particles, advection speed and dispersion properties. However the initialisation of mass alternates between a randomly placed gaussian "bloop" and a split along the y coordinate. The differences in advection speed, dispersion and set-ups create larger variation in the data set. The parameters of the mixing scheme are kept constant in all set-ups. Targets are normalized by the global standard deviation of $\Delta m$ (estimated from the first 200 training files) before computing the loss, following the normalization approach of Li et al.

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (weight decay 10⁻⁵) |
| Initial learning rate | 3 × 10⁻⁴ |
| Final learning rate | 6.25 × 10⁻⁵ |
| LR schedule | Exponential decay |
| Batch size | 8 |
| Iterations | ~300,000 |
| Loss function | Huber loss |
| Gradient clipping | max norm = 1.0 |
| Normalization | LayerNorm after each MLP layer; inputs/outputs normalized to zero mean and unit std |

### Results

DeepMix achieves a **normalized RMSE of 0.2** after approximately 300,000 training iterations on MINITRAC-generated ensemble data. Intercomparison with the physics-based scheme show good agreement. However, the conservation of mass is violated with DeepMix. This can be improved with post-training and adding mass-conservation as further constraint. For strict conservation of mass the architecture might be redesigned to take symmetry of mixing into account.

### Application

DeepMix is designed to replace the numerical mixing scheme in simulations equivalent to MINITRAC. This implementation is **edge-focused**. Future work could explore **node-focused GNNs** to replace the advection scheme as well — an approach that could additionally enforce mass conservation as a bias correction to interpolation errors inherent in Lagrangian advection.

---

## References

1. **Li, Z., & Barati Farimani, A.** (2022). Graph neural network-accelerated Lagrangian fluid simulation. *Computers & Graphics*, 103, 201–211.  
   [DOI: 10.1016/j.cag.2022.02.004](https://doi.org/10.1016/j.cag.2022.02.004)

2. **Benson, D. A., Pribec, I., Engdahl, N. B., Pankavich, S., & Schauer, L.** (2024). Parallelization of particle-mass-transfer algorithms on shared-memory, multi-core CPUs. *Advances in Water Resources*, 193, 104818.  
   [DOI: 10.1016/j.advwatres.2024.104818](https://doi.org/10.1016/j.advwatres.2024.104818)
