# Lagrangian Transport Modeling: Prototyping New Mixing Methods and Deep Emulators

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-pilot%20study-orange.svg)
![Python](https://img.shields.io/badge/python-3.x-blue.svg)

This repository contains **MINITRACLIB** and **DEEPTRACLIB** — a suite of tools for exploring new methods in particle dispersion modeling and for investigating deep learning within Lagrangian transport frameworks. The implementations here are mostly inspired by Benson et al. (2024) and Li & Barati Farimani (2022).

| Component | Description |
|---|---|
| **MINITRACLIB** | Lightweight prototype simulating advection, dispersion, and mixing of particles in a 2D domain. |
| **DEEPTRACLIB** | Neural emulators (**DeepMix**) designed to replace MINITRAC's deterministic mixing scheme with a learned Graph Neural Network surrogate. |

> **📄 Accompanying preprint** — *"Emulating mixing in Lagrangian transport models using Graph Neural Networks and Recursive Message Passing – A pilot study"*, J. Clemens, Institute of Climate and Energy Systems 4 – Stratosphere, Forschungszentrum Jülich. Source manuscript: [`doc/preprint/preprint-template.tex`](doc/preprint/preprint-template.tex). See [§6 Preprint & Case Study](#6-preprint--case-study) below for a summary of its findings.

---

## Contents

1. [Wind Fields and Advection](#1-wind-fields-and-advection)
2. [Numerical Mixing: Mass-Transfer Scheme](#2-numerical-mixing-mass-transfer-scheme)
3. [Numerical Dispersion: Random Walks](#3-numerical-dispersion-random-walks)
4. [Emulation: Deep Graph Neural Networks](#4-emulation-deep-graph-neural-networks)
5. [Software, Hardware & Development Tooling](#5-software-hardware--development-tooling)
6. [Preprint & Case Study](#6-preprint--case-study)
7. [Limitations & Outlook](#7-limitations--outlook)
8. [Reproducibility](#8-reproducibility)
9. [References](#references)
10. [License](#license)

---

## 1. Wind Fields and Advection

A simple analytical double-gyre wind field drives filamentation on a 100 km × 100 km domain. Particles are advected using the gyre's analytical velocity functions with a first-order Euler integration scheme. The examples below show different simulation results for four different setups: first, only with advection; second, with advection but Brownian motion added (dispersion); third, with the mass-transfer mixing scheme; and last, with mixing and dispersion combined.

| Setup | Figure |
|---|---|
| Advection only | ![Advection](https://github.com/janhclem/deeptrac-mini/blob/master/doc/figures/advection_only.png) |
| Advection + Dispersion | ![Advection + Dispersion](https://github.com/janhclem/deeptrac-mini/blob/master/doc/figures/advection_dispersion.png) |
| Advection + Mixing | ![Advection + Mixing](https://github.com/janhclem/deeptrac-mini/blob/master/doc/figures/advection_mixing.png) |
| Advection + Dispersion + Mixing | ![Advection + Dispersion + Mixing](https://github.com/janhclem/deeptrac-mini/blob/master/doc/figures/advection_mixing_dispersion.png) |

---

## 2. Numerical Mixing: Mass-Transfer Scheme

Particle-mass-transfer algorithms calculate inter-particle mixing within a Lagrangian transport framework — the **mass-transfer particle tracking (MTPT)** scheme (Benson et al., 2008, 2016, 2019). They approximate the Eulerian advection-diffusion-reaction equation without introducing artificial numerical diffusion and, when operated in a numerically stable regime, are strictly mass-conserving. The mixing kernels used between particles are identical to the kernel representations found in **Smoothed Particle Hydrodynamics (SPH)**, and can be understood as a smoothed-particle representation of the advection-diffusion equation (Pankavich et al., 2025).

These algorithms have been successfully parallelized and domain-decomposed (Benson et al., 2024), but remain computationally expensive due to the construction of a transformation matrix, inter-particle communication, and the creation of k-trees (or similar structures) for efficient neighbour search. The kernel used here is a **Gaussian**, motivated by the **Green's function** of the diffusion equation: an injected particle begins as a delta-function pulse and disperses into a Gaussian distribution over a sufficiently small, yet non-negligible, time step.

This implementation assumes **isotropy and homogeneity** of the diffusion coefficient. In principle the kernel can be formulated anisotropically to encode spatially varying mixing strengths.

### Algorithm

Each particle is an idealized representative of a finite atmospheric area. Uncertainty arises from two sources:
1. Unresolved sub-grid scale winds.
2. Representing a finite area with a single particle.

In the thermodynamic limit, a parcel large enough to be thermodynamically meaningful is also susceptible to disintegration — this process must be modeled. The Green's function of diffusion over a small time step is a Gaussian; the parcel "disintegrates" and its mass mixes into neighboring particles. When multiple particles are present, their respective Gaussian distributions overlap, quantifying mixing in a rigorous manner.

1. First, study the dispersion of a delta pulse and derive the Gaussian.
2. Formulate the kernel.
3. Create the doubly stochastic exchange matrix.

> **Further reading:** Benson et al. (2024) describe the algorithm and its parallelization in detail; Pankavich et al. (2025) establish convergence properties of the MTPT scheme.

---

## 3. Numerical Dispersion: Random Walks

While the mixing scheme parameterizes parcel disintegration, the **random walk** component parameterizes sub-grid scale wind effects. On limited-resolution grids, exact details of the velocity field are unknown at sub-grid scales; we therefore calculate statistics over the possible trajectories parcels may follow.

Sub-grid scale diffusion acts as an additional correction to the center of mass of each particle's Gaussian distribution. Each random walk step effectively shifts the mean of the Gaussian associated with an individual particle.

> **Future work:** Statistical representations of sub-grid scale winds or super-resolution techniques (e.g., AI-driven models such as **AtmoRep**) could improve these schemes.

---

## 4. Emulation: Deep Graph Neural Networks

**DEEPTRAC** uses a Graph Neural Network (GNN), named **DeepMix**, to replace the deterministic MTPT mixing scheme with a learned statistical surrogate, introducing GNNs into a Lagrangian transport modeling framework. The design is adapted from Li & Barati Farimani (2022), with small modifications to generalize their advection/collision emulator to the mixing problem.

### 4.1 Architecture

DeepMix follows an **Encoder–Processor–Decoder** paradigm with recursive message passing in latent space: the encoder lifts pairwise particle features into a latent space, the processor learns how these latent quantities change locally, and the decoder collapses the result back into a physical prediction of the mass difference.

Particles are structured as a graph:
- **Connectivity:** Edges connect particles within a cutoff radius $r_c$ aligned to the mixing length $l_\mathrm{mix}$, balancing local information gathering against computational cost.
- **Edge features:** Normalized coordinate differences $(dx, dy)$ and mass difference $dm$, scaled by $(l_\mathrm{mix},\, l_\mathrm{mix},\, m_0)$ — a 3-dimensional feature vector per edge.
- **Objective:** Predict the net mass change $\Delta m$ per particle.

| Stage | Figure |
|---|---|
| End-to-end pipeline | ![DeepMix overview](https://github.com/janhclem/deeptrac-mini/blob/master/doc/preprint/figures/deepmix_overview_branded.png) |
| Local neighbourhood / message passing | ![Message passing](https://github.com/janhclem/deeptrac-mini/blob/master/doc/preprint/figures/deepmix_message_passing_branded.png) |
| Encoder / Decoder detail | ![Encoder-Decoder](https://github.com/janhclem/deeptrac-mini/blob/master/doc/preprint/figures/deepmix_encoder_decoder_branded.png) |
| Processor detail | ![Processor](https://github.com/janhclem/deeptrac-mini/blob/master/doc/preprint/figures/deepmix_processor_branded.png) |

The **encoder** is a two-layer MLP that expands the 3-dimensional edge features to 32, then 64 dimensions, giving the processor a richer representation in which to reason about the underlying (Gaussian) mixing kernel. The **decoder** mirrors the encoder, mapping the 64-dimensional latent message back down through 32 dimensions to a single scalar per-edge mass-exchange value, which is aggregated per particle in the final step.

The **processor** first embeds encoded edge features onto nodes via a scatter-sum. The initial edge features and this node embedding are then linearly combined and passed through a two-layer edge MLP (64–32–64) that predicts a latent *message* per edge — effectively a minimal autoencoder learning part of the particle-particle interaction. Messages are aggregated per node by another scatter-sum. A second, symmetric combine-and-MLP step updates the node embedding (not used further downstream), while the un-aggregated per-edge messages directly become the updated edge features. One round of message passing proved sufficient in practice, consistent with Li & Barati Farimani (2022). ReLU activations and LayerNorm are used throughout to stabilize training and avoid vanishing/exploding gradients.

### 4.2 Training Data & Setup

Training data are generated with MINITRAC's own MTPT implementation on the same 100 km × 100 km double-gyre domain (10,000 particles per run, 100 time steps of 20 s each). The ensemble spans:

- **10** initial mass distributions (gradient, sharp step, three Gaussians, "smiley"/emoji, checkerboard at 3 scales, stripes at 2 scales, and random),
- **2** dispersion settings (no random-walk dispersion vs. a first-order random-walk term), and
- **50** gyre velocities $u_0$, linearly spaced between 0 and 25 m/s,

for a total of **1,000 ensemble members** (≈1,000,000 time-step samples). Early time steps in a run tend to show steep, localized gradients, while later steps are much smoother — training across this range of conditions is deliberate, but also means the variance across early and late samples differs substantially. Targets $\Delta m$ are normalized by their global standard deviation (estimated from a subsample of the training files) before computing the loss, following the normalization approach of Li & Barati Farimani (2022).

| Hyperparameter | Pilot phase | Fine-tuning phase |
|---|---|---|
| Optimizer | Adam (weight decay 10⁻⁵) | Adam (weight decay 10⁻⁵) |
| Initial learning rate | 3 × 10⁻⁵ | 3 × 10⁻⁶ |
| Final learning rate | 3 × 10⁻⁶ | 3 × 10⁻⁷ |
| LR schedule | Exponential decay | Exponential decay |
| Batch size | 8 | 8 |
| Loss function | Huber loss | Huber loss + mass-conservation penalty ($\lambda=0.001$) |
| Gradient clipping | max norm = 1.0 | max norm = 1.0 |
| Normalization | LayerNorm after each MLP layer; targets normalized to unit std | (same) |

Training proceeds in two phases: an initial pilot run, followed by a fine-tuning phase (warm-started from the pilot checkpoint) that adds an explicit mass-conservation penalty term to the loss.

### 4.3 Results: Convergence & Validation

DeepMix converges to a final **median RMSE of about 0.22**, corresponding to an estimated **Pearson correlation coefficient $R \approx 0.95$** between predicted and reference mass changes; 95% of samples reach $R > 0.90$. Convergence is steady for the majority of samples throughout training, although a minority of cases show markedly lower correlation. Preliminary analysis suggests these outliers coincide with particular edge cases in the training data (e.g. very sparse local neighbourhoods), though this remains a hypothesis that warrants further investigation. Training is also visibly disrupted by intermittent batches of unusually hard samples, seen as transient loss spikes.

| Metric | Figure |
|---|---|
| RMSE vs. training iteration | ![Training loss](https://github.com/janhclem/deeptrac-mini/blob/master/doc/preprint/figures/training_loss.png) |
| Pearson correlation ($R$) vs. training iteration | ![Training R](https://github.com/janhclem/deeptrac-mini/blob/master/doc/preprint/figures/training_r.png) |
| Mean absolute mass balance per particle | ![Mass balance](https://github.com/janhclem/deeptrac-mini/blob/master/doc/preprint/figures/mass_balance.png) |

Notably, **even without an explicit mass-conservation constraint**, DeepMix steadily improves mass conservation over the course of training, reaching a final conservation error on the order of **0.1% per particle**. This suggests DeepMix is not simply overfitting the training data, but is learning a physically meaningful representation of the underlying mixing operator — though the residual error remains many orders of magnitude larger than machine precision, so mass conservation is *not* exact. The fine-tuning phase (§4.2), which adds an explicit mass-conservation penalty, is intended to close this gap further.

### 4.4 Application

DeepMix is designed to replace the numerical mixing scheme in simulations equivalent to MINITRAC. This implementation is **edge-focused** (it predicts mass exchange along edges). Future work could explore **node-focused GNNs** to replace the advection scheme as well — an approach that could additionally enforce mass conservation as a bias correction to interpolation errors inherent in Lagrangian advection.

---

## 5. Software, Hardware & Development Tooling

DeepMix is implemented using **PyTorch Geometric**, whose graph abstractions keep the entire model implementation compact (fewer than 100 lines of code) compared to an equivalent plain-PyTorch implementation.

All training and simulation in this pilot study was run **CPU-only** on a single Tuxedo laptop (11th Gen Intel® Core™ i5-11300H @ 3.10 GHz, ~15.7 GB RAM), using a single core. Full training took approximately **48 hours**. This demonstrates that a relatively inexpensive setup can already yield useful initial training outcomes, without requiring GPU infrastructure.

Development of the code base made extensive use of AI coding assistants (Vibe/Mistral, Big Pickle/OpenCode, Claude Code/Anthropic, and Lumo/Proton), which accelerated architectural iteration — particularly useful given that GNN training and architecture design are often guided by heuristic strategies rather than first principles.

---

## 6. Preprint & Case Study

The methods and results summarized above are described in full in the accompanying preprint (source: [`doc/preprint/preprint-template.tex`](doc/preprint/preprint-template.tex), built on the [arxiv-style](https://github.com/kourgeorge/arxiv-style) LaTeX template for EarthArXiv-style submissions). The preprint additionally reports a small **qualitative case study**: side-by-side simulation movies of the MTPT baseline and the DeepMix emulator, deliberately left unlabeled so a reader can attempt to identify which is which — an informal Turing test in the spirit of Palmer (2016), who argued that a climate model can be considered adequate once its output becomes visually indistinguishable from observations. The double-gyre case study shows only minor deviations between emulator and baseline, with the MTPT scheme appearing slightly smoother; differences become apparent mainly in fine detail or over longer integration times.

---

## 7. Limitations & Outlook

- **Mass conservation is not strictly enforced.** DeepMix improves mass balance during training even without an explicit constraint, but the residual per-particle error (~0.1%) is far from machine precision. A mass-conserving architecture (e.g. one that enforces anti-symmetric exchange between particle pairs by construction) would be needed for applications with strict conservation requirements.
- **Validation so far is limited to an idealized double-gyre setup.** Generalization to geophysical flows (e.g. atmospheric or ocean mixing) has not yet been tested.
- **Scalability of graph construction** (neighbour search, edge-feature computation) at larger particle counts and over longer integration windows remains to be characterized against the baseline MTPT implementation.
- **A minority of low-correlation outlier samples** are not yet fully explained; the current hypothesis — that they correspond to sparse or otherwise atypical local particle configurations — has not been confirmed.

Despite these limitations, this pilot study shows that convergence toward a working mixing-scheme emulator is achievable with minor adaptation of existing GNN methods and on low-cost, CPU-only hardware. This motivates future architectures that account for further constraints of the mixing problem — such as strict mass conservation and non-isotropic mixing kernels in shear flows — and opens a path toward tuning mixing schemes against observational data, not just simulation output.

---

## 8. Reproducibility

Training data and the training run itself can be reproduced on a common desktop computer using the scripts in this repository:

1. Generate the training ensemble configuration files: `python generate_configs.py`
2. Run MINITRAC over the ensemble to produce training data: `python minitrac.py` (see `start_training.sh` for batch execution)
3. Train the DeepMix emulator: `python training.py`
4. Plot training diagnostics: `python plot_training.py`

---

## References

1. **Li, Z., & Barati Farimani, A.** (2022). Graph neural network-accelerated Lagrangian fluid simulation. *Computers & Graphics*, 103, 201–211.
   [DOI: 10.1016/j.cag.2022.02.004](https://doi.org/10.1016/j.cag.2022.02.004)

2. **Benson, D. A., Pribec, I., Engdahl, N. B., Pankavich, S., & Schauer, L.** (2024). Parallelization of particle-mass-transfer algorithms on shared-memory, multi-core CPUs. *Advances in Water Resources*, 193, 104818.
   [DOI: 10.1016/j.advwatres.2024.104818](https://doi.org/10.1016/j.advwatres.2024.104818)

3. **Pankavich, S., Schauer, L., Schmidt, M. J., Engdahl, N. B., Bolster, D., & Benson, D. A.** (2025). Convergence of mass transfer particle tracking schemes for the simulation of advection-diffusion-reaction equations. *Applied Mathematics and Computation*, 496, 129358.
   [DOI: 10.1016/j.amc.2025.129358](https://doi.org/10.1016/j.amc.2025.129358)

4. **Benson, D. A., Pankavich, S., & Bolster, D.** (2019). On the separate treatment of mixing and spreading by the reactive-particle-tracking algorithm: An example of accurate upscaling of reactive Poiseuille flow. *Advances in Water Resources*, 123, 40–53.
   [DOI: 10.1016/j.advwatres.2018.11.001](https://doi.org/10.1016/j.advwatres.2018.11.001)

5. **Benson, D. A., & Bolster, D.** (2016). Arbitrarily complex chemical reactions on particles. *Water Resources Research*, 52(11), 9190–9200.
   [DOI: 10.1002/2016WR019368](https://doi.org/10.1002/2016WR019368)

6. **Benson, D. A., & Meerschaert, M. M.** (2008). Simulation of chemical reaction via particle tracking: Diffusion-limited versus thermodynamic rate-limited regimes. *Water Resources Research*, 44, W12201.
   [DOI: 10.1029/2008WR007111](https://doi.org/10.1029/2008WR007111)

7. **Palmer, T. N.** (2016). A personal perspective on modelling the climate system. *Proceedings of the Royal Society A*, 472(2188), 20150772.
   [DOI: 10.1098/rspa.2015.0772](https://doi.org/10.1098/rspa.2015.0772)

---

## License

This project is licensed under the **MIT License** — see [`COPYING`](COPYING) for the full text. Copyright © 2026 Forschungszentrum Jülich GmbH.
