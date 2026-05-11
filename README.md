# Lagrangian Transport Modeling: Prototyping new mixing methods and deep emulators

This repository contains MINITRAC and DEEPTRAC. MINITRAC (MINIimal TRAjectory Calculations) is a tiny toy model and prototype, simulating the advection, dispersion and mixing of particles in a 2d domain. The mixing scheme is based on a Particle Mass-Transfer Algorithm (e.g. Benson et al). Advection is implemented easily with a Euler solver of the kinematic equation, which is an ordinary differential equation. Sub-grid scale dispersion is simulated with a random walk. DEEPTRAC (DEEP learning TRAjectory Calculations) is a collection of neural emulators to replace the mixing scheme in MINITRAC. The deep neural network for emulation of mixing is inspired by the work of Li et al. The purpose of MINI- and DEEPTRAC is to explore new methods for dispersion models, increase the intuition about mixing vs. dispersion, and learn about deep learning in context of Lagrangian models.

## 1) Numerical Mixing: Mass-transfer Scheme

This chapter describes a numerical scheme to calculate mixing in a Lagrangian transport framework. To learn about the deep learning part, skip it and move on to the chapter 3.

Particle-mass-transfer algorithms are used in hydrology to calculate the inter particle mixing. It can be show, that they approximate the Eulerian advection-diffusion-reaction equation, however without artificial numerical diffusion. If used in a numerical stable regime they are strictly mass conserving. Furthermore, they use mixing kernels between particles, that are identical to kernel representations of fields as found in smooth particle hydrodynamics. Smooth particle hydrodynamics are fluid dynamics modelled entirely in a Lagrangian framework. Particle-mass-transfer algorithm, have been parallelized and domain decomposed (e.g. Benson et al.). The kernel, here it is a gaussian function, can be strongly motivated by studying the greenfunction of a inserted particle, which starts as a delta-function shaped pulse and then disperses into a gauss function (within a small enough, but not too small time-step). The kernel in principle can be formulated anisotropic, encoding different mixing strenght in different directions and at different locations. However, here we assume isotropy and homogenity of the diffusion coefficient.

### Algorithm

Let's first study the problem of a single probe particle. The particle may have no weight and drag, it is rather and idealized representative of a certain area in the atmosphere. In a Lagrangian numerical model, there are at least two sources of uncertainty, first from unresolved sub-grid scale winds, and second from representing certain area with a signle particle. In a perfect calculation each particle represents a point in a continuum, and the point is transported without disintegration around in the atmosphere. If we construct a probability function that represents the position of the particle, it would be a delta function, that moves around in the atmosphere. However, such perfectly traceable entities might only exist when we would track inert molecules throughout the atmosphere. Since this is impossible, and we need to look into particles as parcels with a volume large enough to fulfill the thermodynamic limit. Interestingly, a parcel that is large enough for the thermodynamic limit is also big enough to disintegrate. This disentgration needs to be modelled. To do so, we study the dispersion of the delta pulse, under diffusion in a small time step. The corresponding green function is the gauss function. Speaking intuitively, the parcel disentriagest in a time step and its masses mix acrose different regions. If we incooperate multiple particles, their gaussian functions might overlap, quantifiying the mixing between those particles in a mathematical sound way. Further details on the particle mass-transfer algorithm can be found in the publication by Benson et al.

## 2) Numerical Dispersion: Random Walks

While the mixing scheme parameterizes the disintegration process of particles, the random walk component parameterizes sub-grid scale wind effects. On limited-resolution grids, even for idealized parcels that would not physically "disintegrate," we must account for uncertainty in the flow field. Since the exact details of the velocity field are unknown at sub-grid scales, we calculate statistics over the possible trajectories that parcels may follow.

This same principle applies to disintegrated parcels. The sub-grid scale diffusion can be understood as an additional correction to the center of mass of the Gaussian distribution. In other words, each random walk step effectively shifts the mean value of the Gaussian curves associated with individual particles.

Interestingly, this schemes could be enhanced with using statistical representations of sub-grid scale winds or super-resolution. Both features are actually provided by AI driven dynamic models, such as AtmoRep.


## Emulation: Deep Graph Neural Networks

We are using a graph neural network to replace the deterministic mixing scheme, with a statistical surrogate. This introduces us into the usage of graph neural networks in a Lagrangian modeling framework. 

### Model Architecture

DEEPTRAC includes DeepMix, which is a graph neural network aiming at emulating the particle mass transfer algorithm. Its architecture can be summarized as encoder - recursive message passing in latent space - decoder. For DeepMix, the particles are structured into a graph. The graph is not fully connected, but instead connections between particles are limited to a selected radius $r$. The radius is selected to agree with the mixing length to obtain sufficient information of nearby particles, without increasing computational demands to much. The graph edge features are mass differences and coordinate differences between nearby particles. The target of DeepMix is to calculate the new mass at every particle in the graph. 

First, the graph edge features (3d) are embedded in a higher dimensional latent space (64d). The space enables DeepMix to learn the complexities of the mixing kernel. Using the latent space, DeepMix recursively updates edge and node features of the graph, via aggregation of passed messages from node to node. Here we select only one recursion. Afterwards, the model uses a decoder to project the features from the latent space onto the graph edges and nodes. 

The training is done based on model simulations given by MINITRAC. Interestingly, DeepMix also could incooperate observational data into the mixing, learning more complex mixing kernels. We are using Adam optimizer and mean squared error (MSE) as the loss function, following the set-up given by Li et al.

### Model Training

Here we demonstrate the model training:

So far model training is not working properly. Batch sizes, learning rates and eventually architectures might require further adjustment.


### Application

DeepMix, should be able to replace the numerical mixing scheme in a small simulation equivalent to MINITRAC. Note that this is an edge focused GNN, and that node focused GNNs eventually could replace the advection scheme as well, while enforcing mass-conservation (via additional loss function) as a bias correction to interpolation errors in common Lagrangian advection schemes.


## Sources

Zijie Li, Amir Barati Farimani, Graph neural network-accelerated Lagrangian fluid simulation, Computers & Graphics, Volume 103, 2022, Pages 201-211, ISSN 0097-8493, https://doi.org/10.1016/j.cag.2022.02.004. (https://www.sciencedirect.com science/article/pii/S0097849322000206)

David A. Benson, Ivan Pribec, Nicholas B. Engdahl, Stephen Pankavich, Lucas Schauer, Parallelization of particle-mass-transfer algorithms on shared-memory, multi-core CPUs, Advances in Water Resources, Volume 193, 2024, 104818, ISSN 0309-1708,
https://doi.org/10.1016/j.advwatres.2024.104818. (https://www.sciencedirect.com/science/article/pii/S0309170824002057)


