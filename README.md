# Lagrangian Transport Modeling: Prototyping New Mixing Methods and Deep Emulators

This repository contains **MINITRAC** and **DEEPTRAC**, a suite of tools designed to explore new methods for dispersion modeling and to investigate the application of deep learning within Lagrangian frameworks. It is a playground to speculate, make errors and learn.

- **MINITRAC** (MINImal TRAjectory Calculations): A lightweight prototype model simulating the advection, dispersion, and mixing of particles in a 2D domain.
- **DEEPTRAC** (DEEP learning TRAjectory Calculations): A collection of neural emulators designed to replace the deterministic mixing scheme in MINITRAC with a statistical surrogate.

## Overview

The primary objectives of this project are to:
1.  Explore novel methods for dispersion modeling.
2.  Enhance intuition regarding the distinction between **mixing** (inter-particle mass transfer) and **dispersion** (sub-grid scale transport).
3.  Investigate the integration of Graph Neural Networks (GNNs) into Lagrangian transport models.

---

## 1. Numerical Mixing: Mass-Transfer Scheme

This section describes the deterministic numerical scheme used to calculate mixing within a Lagrangian transport framework. Readers interested primarily in the deep learning components may skip to **Section 3**.

Particle-mass-transfer algorithms are widely used in hydrology to calculate inter-particle mixing. These algorithms approximate the Eulerian advection-diffusion-reaction equation without introducing artificial numerical diffusion. When operated in a numerically stable regime, they are strictly mass-conserving. Furthermore, the mixing kernels employed between particles are identical to the kernel representations of fields found in **Smoothed Particle Hydrodynamics (SPH)**, a method that models fluid dynamics entirely within a Lagrangian framework.

These algorithms have been successfully parallelized and domain-decomposed (e.g., Benson et al.). The kernel used in this implementation is a **Gaussian function**, strongly motivated by analyzing the **Green's function** of an injected particle. Such a particle begins as a delta-function-shaped pulse and subsequently disperses into a Gaussian distribution over a sufficiently small, yet non-negligible, time step.

While the kernel can theoretically be formulated anisotropically to encode varying mixing strengths across different directions and locations, this implementation assumes **isotropy and homogeneity** of the diffusion coefficient.

### Algorithm

Consider first the problem of a single probe particle. This particle possesses no weight or drag; rather, it serves as an idealized representative of a specific area within the atmosphere. In a Lagrangian numerical model, uncertainty arises from two primary sources:
1.  Unresolved sub-grid scale winds.
2.  The representation of a finite area by a single particle.

In a perfect calculation, each particle represents a point in a continuum, transported intact around the atmosphere. If we construct a probability function representing the particle's position, it would be a delta function moving through the domain. However, such perfectly traceable entities exist only if we were tracking inert molecules throughout the atmosphere—a physical impossibility for our purposes. Consequently, we must treat particles as parcels with a volume large enough to satisfy the **thermodynamic limit**. Interestingly, a parcel large enough for the thermodynamic limit is also susceptible to disintegration. This disintegration must be modeled.

To achieve this, we examine the dispersion of a delta pulse under diffusion over a small time step. The corresponding Green's function is a Gaussian. Intuitively, the parcel "disintegrates" over a time step, and its mass mixes across different regions. When incorporating multiple particles, their respective Gaussian functions may overlap, mathematically quantifying the mixing between them in a rigorous manner.

> **Note:** Further details on the particle mass-transfer algorithm can be found in the publication by **Benson et al.**

---

## 2. Numerical Dispersion: Random Walks

While the mixing scheme parameterizes the disintegration process of particles, the **random walk** component parameterizes sub-grid scale wind effects. On limited-resolution grids, even for idealized parcels that would not physically "disintegrate," we must account for uncertainty in the flow field. Since the exact details of the velocity field are unknown at sub-grid scales, we calculate statistics over the possible trajectories that parcels may follow.

This same principle applies to disintegrated parcels. The sub-grid scale diffusion can be understood as an additional correction to the center of mass of the Gaussian distribution. In other words, each random walk step effectively shifts the mean value of the Gaussian curves associated with individual particles.

> **Future Work:** These schemes could be enhanced by using statistical representations of sub-grid scale winds or super-resolution techniques. Both features are potentially provided by AI-driven dynamic models, such as **AtmoRep**.

---

## 3. Emulation: Deep Graph Neural Networks

We employ a Graph Neural Network (GNN) to replace the deterministic mixing scheme with a statistical surrogate. This approach introduces the application of GNNs within a Lagrangian modeling framework.

### Model Architecture

**DEEPTRAC** includes **DeepMix**, a graph neural network designed to emulate the particle mass-transfer algorithm. Its architecture follows an **Encoder – Recursive Message Passing in Latent Space – Decoder** paradigm.

For DeepMix, particles are structured into a graph:
-   **Connectivity:** The graph is not fully connected. Connections between particles are restricted to a selected radius $r$, chosen to align with the mixing length. This ensures sufficient information is gathered from nearby particles without excessively increasing computational demands.
-   **Edge Features:** Mass differences and coordinate differences between neighboring particles.
-   **Objective:** Calculate the updated mass for every particle in the graph.

**Processing Steps:**
1.  **Embedding:** Graph edge features (3D) are embedded into a higher-dimensional latent space (64D). This space enables DeepMix to learn the complexities of the mixing kernel.
2.  **Message Passing:** Using the latent space, DeepMix recursively updates edge and node features via the aggregation of messages passed between nodes. In this implementation, we utilize a single recursion step.
3.  **Decoding:** The model uses a decoder to project features from the latent space back onto the graph edges and nodes.

Training is performed using model simulations generated by MINITRAC. Notably, DeepMix could also incorporate observational data into the mixing process, allowing it to learn more complex mixing kernels. We utilize the **Adam optimizer** and **Mean Squared Error (MSE)** as the loss function, adhering to the setup described by **Li et al.**

### Model Training Status

> ⚠️ **Current Status:** Model training is not yet functioning optimally. Batch sizes, learning rates, and potentially the architecture itself may require further adjustment.

### Application

DeepMix aims to replace the numerical mixing scheme in small-scale simulations equivalent to MINITRAC.

It is important to note that this implementation utilizes an **edge-focused GNN**. Future iterations could explore **node-focused GNNs** to potentially replace the advection scheme as well. Such an approach could enforce mass conservation (via an additional loss function) as a bias correction to interpolation errors inherent in common Lagrangian advection schemes.

---

## References

1.  **Li, Z., & Barati Farimani, A.** (2022). Graph neural network-accelerated Lagrangian fluid simulation. *Computers & Graphics*, 103, 201–211. ISSN 0097-8493.  
    [DOI: 10.1016/j.cag.2022.02.004](https://doi.org/10.1016/j.cag.2022.02.004)

2.  **Benson, D. A., Pribec, I., Engdahl, N. B., Pankavich, S., & Schauer, L.** (2024). Parallelization of particle-mass-transfer algorithms on shared-memory, multi-core CPUs. *Advances in Water Resources*, 193, 104818. ISSN 0309-1708.  
    [DOI: 10.1016/j.advwatres.2024.104818](https://doi.org/10.1016/j.advwatres.2024.104818)


