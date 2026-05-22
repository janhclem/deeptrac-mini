"""
MINITRAC is a minimal example to study particle dispersion and mixing.
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

import numpy as np
from scipy.spatial import cKDTree


class Config():
    """Simulation configuration and derived physical parameters."""

    def __init__(self,
                 lx=100_000,
                 nump=10_000,
                 tmax=2000,
                 beta=1.0,
                 dim=2,
                 dt=20,
                 dmix=12500,
                 ddiff0=12500,
                 ddiff1=0.0,
                 u0=None,
                 m0=1.0,
                 dt_plot=2000):
        """
        Parameters
        ----------
        lx : float
            Domain side length (m).
        nump : int
            Number of particles.
        tmax : float
            Total simulation time (s).
        beta : float
            Mixing kernel shape parameter (-).
        dim : int
            Spatial dimensions.
        dt : float
            Integration time step (s).
        dmix : float
            Mixing diffusivity (m²/s).
        ddiff0 : float
            First-order dispersion diffusivity (m²/s).
        ddiff1 : float
            Second-order dispersion diffusivity (m²/s).
        u0 : float or None
            Characteristic wind speed (m/s). Defaults to lx / sqrt(nump) / dt.
        m0 : float
            Initial particle mass (kg).
        dt_plot : float
            Plotting interval (s). Must be a multiple of dt.
        """
        self.lx = lx
        self.np = nump
        self.tmax = tmax
        self.beta = beta
        self.dim = dim
        self.dt = dt
        self.dmix = dmix
        self.ddiff0 = ddiff0
        self.ddiff1 = ddiff1
        self.u0 = lx / (nump ** 0.5) / dt if u0 is None else u0
        self.m0 = m0
        self.lmix = 2 * np.sqrt(self.dmix * self.dt / self.beta)
        self.dt_plot = dt_plot
        if dt_plot % dt > 0:
            print("[WARNING] Plotting frequency must be a multiple of integration timestep.")

    def show(self):
        """Print all configuration parameters and characteristic length scales."""
        print("[INFO] Parameters:")
        for key, value in self.__dict__.items():
            print(f"  {key}: {value}")
        print("[INFO] Characteristic lengths:")
        print(f"  Mixing length:             {2 * np.sqrt(self.dmix * self.dt / self.beta):.1f} m")
        print(f"  Advection length:          {self.u0 * self.dt:.1f} m")
        print(f"  Dispersion length (1st):   {2 * np.sqrt(self.ddiff0 * self.dt):.1f} m")
        print(f"  Dispersion length (2nd):   {2 * np.sqrt(self.ddiff1 * self.dt):.1f} m")


class Atm():
    """Particle ensemble representing an atmospheric state."""

    def __init__(self):
        self.x = None  # positions (np, dim)
        self.m = None  # masses    (np,)

    def init(self, config):
        """Initialize particles on a uniform random distribution with constant mass."""
        self.x = np.random.uniform(0, config.lx, size=(config.np, config.dim))
        self.m = np.ones(shape=config.np) * config.m0

    def mod_mass(self, config, noise=False, method="half", **method_config):
        """
        Modify particle masses according to a prescribed pattern.

        Parameters
        ----------
        config : Config
        noise : bool
            Add Gaussian noise to the mass field.
        method : str
            "half" sets one half-plane to zero mass.
        **method_config
            axis (int): split axis for "half" method (0 or 1).
            a, b (float): noise mean and std-dev bounds.
        """
        if method == "half":
            axis = method_config.get("axis", 0)
            if axis == 0:
                self.m[self.x[:, 1] < config.lx / 2.0] = 0
            else:
                self.m[self.x[:, 0] < config.lx / 2.0] = 0
        else:
            print(f"[WARNING] Mass modification method '{method}' not available.")

        if noise:
            a = method_config.get("a", -0.25)
            b = method_config.get("b",  0.25)
            self.m += np.random.normal(a, b, size=self.m.shape)
            self.m = np.clip(self.m, a_min=0, a_max=1)

    def init_mass_gradient(self, config, order=1):
        """Initialize masses as a random linear gradient across the domain."""
        a = np.random.uniform(0, 1)
        b = np.random.uniform(0, 1)
        self.m = ((self.x[:, 0] / config.lx * a + self.x[:, 1] / config.lx * b) * 0.5) ** order

    def init_mass_gauss(self, config):
        """Initialize masses as a Gaussian blob at a random location."""
        x0 = np.random.uniform(0, config.lx)
        y0 = np.random.uniform(0, config.lx)
        r2 = (self.x[:, 0] - x0) ** 2 + (self.x[:, 1] - y0) ** 2
        self.m = config.m0 * np.exp(-r2 / config.lx)


def kernel(x, beta, dim, d, dt):
    """
    Evaluate the Gaussian mixing kernel for all particle pairs.

    Computes the full (dense) kernel matrix K[i,j] = N * exp(-beta/(4*d*dt) * |x_i - x_j|^2),
    where N is the normalization constant derived from the Green's function of the diffusion equation.

    Parameters
    ----------
    x : np.ndarray, shape (n, dim)
        Particle positions.
    beta : float
        Kernel shape parameter.
    dim : int
        Spatial dimensions.
    d : float
        Diffusivity (m²/s).
    dt : float
        Time step (s).

    Returns
    -------
    np.ndarray, shape (n, n)
        Kernel matrix.
    """
    factor = 4 * np.pi * d * dt / beta
    norm_const = (1 / factor) ** (dim / 2)
    exp_factor = -beta / (4 * d * dt)

    sq_norms = np.sum(x ** 2, axis=1)
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2 * np.dot(x, x.T)
    sq_dist = np.maximum(sq_dist, 0)

    return norm_const * np.exp(exp_factor * sq_dist)


def mix(x, m, beta, dmix, dt, dim, r_cutoff=None):
    """
    Perform one mixing step via a sparse Gaussian kernel mass-transfer scheme.

    Particle masses are updated by exchanging mass proportional to a Gaussian
    kernel evaluated between neighbouring particles. The kernel matrix is
    symmetrized and row/column normalized to enforce double-stochasticity,
    guaranteeing mass conservation.

    Parameters
    ----------
    x : np.ndarray, shape (n, dim)
        Particle positions (m).
    m : np.ndarray, shape (n,)
        Particle masses (kg).
    beta : float
        Kernel shape parameter.
    dmix : float
        Mixing diffusivity (m²/s).
    dt : float
        Time step (s).
    dim : int
        Spatial dimensions.
    r_cutoff : float or None
        Neighbour search radius (m). Defaults to 3 × lmix.

    Returns
    -------
    np.ndarray, shape (n,)
        Updated particle masses.
    """
    n = len(x)
    if r_cutoff is None:
        r_cutoff = 3 * 2 * np.sqrt(dmix * dt / beta)

    tree = cKDTree(x)
    pairs = tree.query_pairs(r_cutoff)

    factor = 4 * np.pi * dmix * dt / beta
    norm_const = (1 / factor) ** (dim / 2)
    exp_factor = -beta / (4 * dmix * dt)

    p = np.zeros((n, n))
    np.fill_diagonal(p, norm_const)

    if pairs:
        i_arr, j_arr = np.array(list(pairs)).T
        diffs = x[i_arr] - x[j_arr]
        sq_dist = np.sum(diffs ** 2, axis=1)
        p_vals = norm_const * np.exp(exp_factor * sq_dist)
        p[i_arr, j_arr] = p_vals
        p[j_arr, i_arr] = p_vals

    # Double-stochastic normalization: average of row and column sums
    p /= (np.sum(p, axis=1, keepdims=True) + np.sum(p, axis=0, keepdims=True)) * 0.5

    dm = m[None, :] - m[:, None]
    return m + beta * np.sum(dm * p, axis=1)


def gyre(x, lx, u0):
    """
    Evaluate the analytical double-gyre velocity field.

    The stream function is psi = u0 * sin(2*pi*x/lx) * sin(pi*y/lx),
    yielding two counter-rotating gyres on the domain [0, lx]².

    Parameters
    ----------
    x : np.ndarray, shape (n, 2)
        Particle positions (m).
    lx : float
        Domain side length (m).
    u0 : float
        Characteristic wind speed (m/s).

    Returns
    -------
    np.ndarray, shape (n, 2)
        Velocity vectors (m/s).
    """
    arg_x = np.pi * x[:, 0] / lx * 2
    arg_y = np.pi * x[:, 1] / lx

    u = -np.pi * u0 * np.sin(arg_x) * np.cos(arg_y)
    v =  np.pi * u0 * np.cos(arg_x) * np.sin(arg_y)

    return np.stack((u, v), axis=-1)
