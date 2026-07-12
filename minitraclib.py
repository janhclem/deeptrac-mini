"""
MINITRACLIB - Minimal particle dispersion and mixing library.

This library provides core functionality for simulating particle dispersion
and mixing in 2D Lagrangian transport models. It includes:

- Config: Simulation configuration management
- Atm: Particle ensemble class for atmospheric state representation
- Kernel functions: Gaussian mixing kernels for particle mass exchange
- Mixing schemes: Sparse and dense mass-transfer algorithms
- Wind fields: Analytical velocity fields (double-gyre, Bickley jet)

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

Examples
--------
>>> import minitraclib as mini
>>> cfg = mini.Config()
>>> atm = mini.Atm()
>>> atm.init(cfg)
>>> vel = mini.gyre(atm.x, lx=cfg.lx, u0=cfg.u0)
"""

import configparser
import argparse
import os
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree



class Command():

	def __init__(self):
	
		# Parse command line arguments
		self.parser = argparse.ArgumentParser(
		    description="Run particle dispersion and mixing simulations"
		)
		self.parser.add_argument(
		    "--config", "-c",
		    type=str,
		    default="gyre_100km",
		    help="Configuration file name (with or without .ini extension)"
		)
		self.parser.add_argument(
		    "--nens", "-n",
		    type=int,
		    default=999,
		    help="Number of ensemble members (default: 999)"
		)

		self.args = self.parser.parse_args()

class Config():
    """
    Simulation configuration and derived physical parameters.

    This class holds all the parameters needed to configure a simulation,
    including domain properties, numerical settings, and physical parameters.

    For new code, consider using the configs module instead, which provides
    predefined configurations for different setups (gyre, jet).

    Parameters
    ----------
    lx : float, optional
        Domain side length (m). Default is 100,000 m (100 km).
    ly : float or None, optional
        Domain side length in y-direction (m). Defaults to lx.
    nump : int, optional
        Number of particles. Default is 20,000.
    tmax : float, optional
        Total simulation time (s). Default is 2000 s.
    beta : float, optional
        Mixing kernel shape parameter. Default is 1.0.
    dim : int, optional
        Spatial dimensions. Default is 2.
    dt : float, optional
        Integration time step (s). Default is 20 s.
    dt_mix : float, optional
        Mixing time step (s). Default is 2000 s.
    dmix : float, optional
        Mixing diffusivity (m^2/s). Default is 12500 * 100^2 / 2.0.
    ddiff0 : float, optional
        First-order dispersion diffusivity (m^2/s). Default is 0.0.
    ddiff1 : float, optional
        Second-order dispersion diffusivity (m^2/s). Default is 0.0.
    u0 : float or None, optional
        Characteristic wind speed (m/s). If None, computed as
        lx / sqrt(nump) / dt. Default is None.
    m0 : float, optional
        Initial particle mass (kg). Default is 1.0.
    dt_plot : float, optional
        Plotting interval (s). Default is 2000 s.

    Attributes
    ----------
    lmix : float
        Mixing length scale: 2 * sqrt(dmix * dt_mix / beta) (m).
    np : int
        Alias for nump for backwards compatibility.

    See Also
    --------
    configs.BaseConfig : Base configuration class with predefined setups
    configs.load_config : Load predefined configurations (gyre, jet, etc.)

    Examples
    --------
    >>> cfg = Config()
    >>> cfg.show()
    """

    def __init__(self,
                 lx=100_000,
                 ly=None,
                 nump=20_000,
                 tmax=2000,
                 beta=1.0,
                 dim=2,
                 dt=20,
                 dt_mix=2000,
                 dmix=12500*100**2/2.0,
                 ddiff0=0.0,
                 ddiff1=0.0,
                 u0=None,
                 m0=1.0,
                 dt_plot=2000):
        """
        Initialize simulation configuration.

        Parameters
        ----------
        lx : float, optional
            Domain side length in x-direction (m). Default is 100,000 m (100 km).
        ly : float or None, optional
            Domain side length in y-direction (m). Defaults to lx.
        nump : int, optional
            Number of particles. Default is 20,000.
        tmax : float, optional
            Total simulation time (s). Default is 2000 s.
        beta : float, optional
            Mixing kernel shape parameter. Default is 1.0.
        dim : int, optional
            Spatial dimensions. Default is 2.
        dt : float, optional
            Integration time step (s). Default is 20 s.
        dt_mix : float, optional
            Mixing time step (s). Default is 2000 s.
        dmix : float, optional
            Mixing diffusivity (m^2/s). Default is 12500 * 100^2 / 2.0.
        ddiff0 : float, optional
            First-order dispersion diffusivity (m^2/s). Default is 0.0.
        ddiff1 : float, optional
            Second-order dispersion diffusivity (m^2/s). Default is 0.0.
        u0 : float or None, optional
            Characteristic wind speed (m/s). If None, computed as
            lx / sqrt(nump) / dt. Default is None.
        m0 : float, optional
            Initial particle mass (kg). Default is 1.0.
        dt_plot : float, optional
            Plotting interval (s). Default is 2000 s.
        """
        self.lx = lx
        self.ly = ly if ly is not None else lx  # Square domain by default
        self.nump = nump
        self.np = nump  # Alias for backwards compatibility
        self.tmax = tmax
        self.beta = beta
        self.dim = dim
        self.dt = dt
        self.dt_mix = dt_mix
        self.dmix = dmix
        self.ddiff0 = ddiff0
        self.ddiff1 = ddiff1
        self.u0 = lx / (nump ** 0.5) / dt if u0 is None else u0
        self.m0 = m0
        self.lmix = 2 * np.sqrt(self.dmix * self.dt_mix / self.beta)
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

    @classmethod
    def read_ini(cls, filename=None):
        """
        Read configuration from an INI file.

        This method loads a configuration from a Microsoft INI format file
        and returns a Config object with all parameters set.

        Parameters
        ----------
        filename : str, Path, or None, optional
            Path to the INI configuration file. If None, looks for
            'gyre_100km.ini' in the configs directory. Can be a name
            without .ini extension (e.g., 'gyre_100km' or 'jet_large').

        Returns
        -------
        Config
            Configuration object with parameters loaded from the INI file.

        Raises
        ------
        FileNotFoundError
            If the specified configuration file cannot be found.

        Notes
        -----
        The INI file should have sections like DOMAIN, NUMERICS, TIME,
        PHYSICS, MIXING, ADVECTION with appropriate keys.

        If u0 is not specified in the file, it will be computed as:
            u0 = lx / sqrt(nump) / dt

        Examples
        --------
        >>> cfg = Config.read_ini('gyre_100km')
        >>> cfg = Config.read_ini('jet_large.ini')
        >>> cfg = Config.read_ini('/path/to/custom.ini')
        """
        # Determine the config file path
        config_path = None
        
        if filename is None:
            # Default to gyre_100km.ini
            filename = 'gyre_100km'
        
        # If filename looks like a full path
        if isinstance(filename, Path) or (isinstance(filename, str) and 
            (os.path.isabs(filename) or '/' in filename or '\\' in filename)):
            config_path = Path(filename)
        else:
            # Look in the configs directory
            configs_dir = Path(__file__).parent / 'configs'
            
            # Try with and without .ini extension
            possible_paths = [
                configs_dir / f"{filename}",
                configs_dir / f"{filename}.ini",
                Path(filename),
                Path(filename).with_suffix('.ini'),
            ]
            
            for p in possible_paths:
                if p.exists():
                    config_path = p
                    break
        
        if config_path is None or not config_path.exists():
            # Try to find available configs
            configs_dir = Path(__file__).parent / 'configs'
            if configs_dir.exists():
                available = [f.stem for f in configs_dir.glob("*.ini")]
            else:
                available = []
            raise FileNotFoundError(
                f"Configuration file '{filename}' not found. "
                f"Available configurations in {configs_dir}: {available}"
            )
        
        # Load the INI file
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Build a case-insensitive helper for sections
        sections = {s.lower(): s for s in config.sections()}
        
        def get_value(section, key, type_func=None, default=None):
            """Get a value from a section with case-insensitive lookup."""
            actual_section = sections.get(section.lower())
            if actual_section is None:
                return default
            
            # Build case-insensitive key mapping
            try:
                options = dict(config.items(actual_section))
                option_map = {k.lower(): k for k in options.keys()}
                actual_key = option_map.get(key.lower())
                
                if actual_key is None:
                    return default
                
                value = options[actual_key]
                if type_func is not None:
                    return type_func(value)
                return value
            except Exception:
                return default
        
        # Extract all parameters from INI sections
        # DOMAIN section
        lx = get_value('DOMAIN', 'lx', float, 100_000)
        ly = get_value('DOMAIN', 'ly', float, lx)
        
        # NUMERICS section
        nump = get_value('NUMERICS', 'nump', int, 20_000)
        
        # TIME section
        tmax = get_value('TIME', 'tmax', float, 2000)
        dt = get_value('TIME', 'dt', float, 20)
        dt_mix = get_value('TIME', 'dt_mix', float, 2000)
        dt_plot = get_value('TIME', 'dt_plot', float, 2000)
        
        # PHYSICS section
        beta = get_value('PHYSICS', 'beta', float, 1.0)
        dim = get_value('PHYSICS', 'dim', int, 2)
        
        # MIXING section
        dmix = get_value('MIXING', 'dmix', float, 12500 * 100**2 / 2.0)
        ddiff0 = get_value('MIXING', 'ddiff0', float, 0.0)
        ddiff1 = get_value('MIXING', 'ddiff1', float, 0.0)
        lbd_c = get_value('MIXING', 'lbd_c', float, 1e-08)
        w = get_value('MIXING', 'w', float, 0.0000001)
        
        # ADVECTION section
        u0_str = get_value('ADVECTION', 'u0', str, None)
        m0 = get_value('ADVECTION', 'm0', float, 1.0)
        
        # Handle u0: if empty or not specified, compute it
        u0 = None
        if u0_str is not None and str(u0_str).strip() != '':
            try:
                u0 = float(u0_str)
            except ValueError:
                u0 = None
        
        # SIMULATION section (for run-time settings)
        mixing_type = get_value('SIMULATION', 'mixing_type', str, 'default')
        dir_out = get_value('SIMULATION', 'dir_out', str, './out')
        dir_plot = get_value('SIMULATION', 'dir_plot', str, './plot')
        prec_warn = get_value('SIMULATION', 'prec_warn', float, 1e-14)
        boundary_method = get_value('SIMULATION', 'boundary_method', str, 'periodic')
        
        # Create the Config object
        cfg = cls(
            lx=lx,
            ly=ly,
            nump=nump,
            tmax=tmax,
            beta=beta,
            dim=dim,
            dt=dt,
            dt_mix=dt_mix,
            dmix=dmix,
            ddiff0=ddiff0,
            ddiff1=ddiff1,
            u0=u0,
            m0=m0,
            dt_plot=dt_plot
        )
        
        # Add simulation attributes to the config object
        cfg.mixing_type = mixing_type
        cfg.dir_out = dir_out
        cfg.dir_plot = dir_plot
        cfg.prec_warn = prec_warn
        cfg.boundary_method = boundary_method
        cfg.lbd_c = lbd_c
        cfg.w = w
        
        return cfg


class Atm():
    """Particle ensemble representing an atmospheric state."""

    def __init__(self):
        self.x = None  # positions (np, dim)
        self.m = None  # masses    (np,)

    def init(self, config):
        """Initialize particles on a uniform random distribution with constant mass."""
        self.x = np.random.uniform([0,config.lx/4.0], [config.lx, config.lx*3/4], size=(config.np, config.dim))
        self.m = np.ones(shape=config.np) * config.m0

    def mod_mass(self, config, noise=False, method="half", **method_config):
        """
        Modify particle masses according to a prescribed pattern.

        Parameters
        ----------
        config : Config
            Configuration object.
        noise : bool, optional
            Add Gaussian noise to the mass field. Default is False.
        method : str, optional
            Method for mass modification. "half" sets one half-plane to zero mass.
            Default is "half".
        **method_config
            Additional keyword arguments:
            - axis (int): split axis for "half" method (0 or 1).
            - a, b (float): noise mean and std-dev bounds.
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
            b = method_config.get("b", 0.25)
            self.m += np.random.normal(a, b, size=self.m.shape)
            self.m = np.clip(self.m, a_min=0, a_max=1)

    def init_mass_gradient(self, config):
        """
        Initialize masses as a linear gradient across the domain.

        Parameters
        ----------
        config : Config
            Configuration object.
        """
        self.m = (self.x[:, 1] / config.lx * 2 - 1/2) * config.m0

    def init_mass_gauss(self, config):
        """
        Initialize masses as a Gaussian blob at a random location.

        Parameters
        ----------
        config : Config
            Configuration object.
        """
        x0 = np.random.uniform(0, config.lx)
        y0 = np.random.uniform(0, config.lx)
        r2 = (self.x[:, 0] - x0) ** 2 + (self.x[:, 1] - y0) ** 2
        sigma = np.random.uniform(0.1, 0.2) * config.lx**2
        self.m = config.m0 * np.exp(-r2 / sigma)

    def plot(self, s_idx=0, z=None, cmap=None, levels=500, vmin=0, vmax=1,
             dpi=300, save_path=None, show=False):
        """
        Plot the particle mass distribution using tricontourf.

        Parameters
        ----------
        s_idx : int, optional
            Scenario index for file naming. Default is 0.
        z : np.ndarray or None, optional
            Array to plot as the z-values for tricontourf. If None, uses self.m.
            Default is None.
        cmap : str or Colormap or None, optional
            Colormap to use. Default is None (uses matplotlib default).
        levels : int, optional
            Number of contour levels. Default is 500.
        vmin : float, optional
            Minimum value for color scaling. Default is 0.
        vmax : float, optional
            Maximum value for color scaling. Default is 1.
        dpi : int, optional
            DPI for saved figures. Default is 300.
        save_path : str or None, optional
            Path to save the figure. If None, uses '{dir_plot}/{s_idx}/mass_{t_idx}.png'.
            Default is None.
        show : bool, optional
            Whether to display the plot. Default is False.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.

        Notes
        -----
        Particle positions are divided by 1000 to convert from meters to km
        (since domain is typically in km scale).

        Examples
        --------
        >>> atm = Atm()
        >>> atm.init(cfg)
        >>> fig = atm.plot(s_idx=0)  # Plot with default self.m
        >>> fig = atm.plot(s_idx=0, z=rates, vmin=-2e-8, vmax=2e-8)  # Plot rates
        """
        import matplotlib.pylab as plt
        from cmcrameri import cm

        if z is None:
            z = self.m

        # Use cmcrameri colormap if not specified
        if cmap is None:
            cmap = cm.glasgow

        # Positions in km (divide by 1000)
        x_km = self.x[:, 0] / 1000
        y_km = self.x[:, 1] / 1000

        fig = plt.figure()
        sct = plt.tricontourf(x_km, y_km, z, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)

        plt.colorbar(sct)
        plt.xlabel("x [km]")
        plt.ylabel("y [km]")

        if save_path:
            plt.savefig(save_path, dpi=dpi)
        elif hasattr(self, '_dir_plot'):
            # Use instance dir_plot if set
            os.makedirs(f"{self._dir_plot}/{s_idx}/", exist_ok=True)
            plt.savefig(f"{self._dir_plot}/{s_idx}/mass.png", dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def check_boundaries(self, config, method='periodic'):
        """
        Enforce domain boundaries on particle positions.

        Parameters
        ----------
        config : Config
            Configuration object containing domain dimensions (lx, ly).
        method : str, optional
            Boundary condition method. Options:
            - 'periodic': Particles that exit one boundary re-enter from the opposite side (x-direction only).
            - 'clip': Particles are clipped to stay within [0, config.lx] in both dimensions.
            Default is 'periodic'.

        Notes
        -----
        For 'periodic' method (default):
            - In x-direction: particles wrap around using modulo
            - This matches the original minitrac.py behavior
        For 'clip' method:
            - Particles outside [0, config.lx] are clipped to the boundary
            - Uses np.clip for both x and y dimensions

        Examples
        --------
        >>> atm.check_boundaries(cfg, method='periodic')  # Wrap around (x only)
        >>> atm.check_boundaries(cfg, method='clip')     # Clip to domain
        """
        if method == 'periodic':
            # Periodic boundary conditions (wrap around) - x direction only
            # This matches the original minitrac.py behavior
            self.x[:, 0][self.x[:, 0] > config.lx] -= config.lx
            self.x[:, 0][self.x[:, 0] < 0] += config.lx
        elif method == 'clip':
            # Clip boundary conditions - both dimensions
            # Clip x to [0, config.lx] and y to [0, config.ly]
            self.x[:, 0] = np.clip(self.x[:, 0], a_min=0, a_max=config.lx)
            if config.dim >= 2:
                self.x[:, 1] = np.clip(self.x[:, 1], a_min=0, a_max=config.ly)
        else:
            raise ValueError(f"Unknown boundary method: '{method}'. Use 'periodic' or 'clip'.")


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

def mix_steering(x, m, beta, dmix, dt_mix, dim, r_cutoff=None, dists_tm1=None,
        lbd_c=1.2, w=1.0):
    """
    Perform one mixing step via a sparse Gaussian kernel mass-transfer scheme,
    with an optional stretching trigger that enhances local diffusivity where
    the finite-time stretching rate between neighbor pairs exceeds `lbd_c`.

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
        Baseline mixing diffusivity (m^2/s).
    dt_mix : float
        Time step of mixing scheme (s).
    dim : int
        Spatial dimensions.
    r_cutoff : float or None
        Neighbour search radius (m). Defaults to 6 * sqrt(dmix * dt / beta).
    dists_tm1 : dict {(i, j): sq_dist} or None
        Sparse pairwise squared distances from the previous call. Pass the
        `dists` returned by the previous call. None on the first call.
    lbd_c : float
        Critical stretching rate (1/s) above which mixing is enhanced.
        Default 1.2/day, expressed in 1/s.
    w : float
        Smoothness of the tanh trigger around lbd_c. Larger = smoother/
        wider transition; smaller = sharper, closer to a hard threshold.

    Returns
    -------
    m_new : np.ndarray, shape (n,)
        Updated particle masses.
    dists : dict {(i, j): sq_dist}
        Sparse pairwise squared distances at this step. Pass this back in
        as `dists_tm1` on the next call.
    """
    n = len(x)
    if r_cutoff is None:
        r_cutoff = 6 * np.sqrt(dmix * dt_mix / beta)

    tree = cKDTree(x)
    pairs = list(tree.query_pairs(r_cutoff))

    dists_tm1 = dists_tm1 if dists_tm1 is not None else {}

    dists = {}
    i_arr = j_arr = np.zeros(0, dtype=int)
    p_vals = np.zeros(0)

    if pairs:
        i_arr, j_arr = np.array(pairs).T
        diffs = x[i_arr] - x[j_arr]
        sq_dist = np.sum(diffs ** 2, axis=1)
        dists = {(int(i), int(j)): float(d) for i, j, d in zip(i_arr, j_arr, sq_dist)}

        rate = np.full(len(pairs), lbd_c)
        for k in range(len(pairs)):
            i, j = int(i_arr[k]), int(j_arr[k])
            prev = dists_tm1.get((i, j), dists_tm1.get((j, i)))
            if prev is not None and prev > 0 and sq_dist[k] > 0:
                rate[k] = np.log(np.sqrt(sq_dist[k] / prev)) / dt_mix

        dmix_exp = dmix * (1 + np.tanh((rate - lbd_c) /w))
        mask_infty = dmix_exp < 0.0001 # Guard from numerical instability.
        factor = 4 * np.pi * dmix_exp * dt_mix / beta
        norm_const = factor ** (-dim / 2)
        exp_factor = -beta / (4 * dmix_exp * dt_mix)
        p_vals = norm_const * np.exp(exp_factor * sq_dist)
        p_vals[mask_infty] = 0

    diag_norm_const = (4 * np.pi * dmix * dt_mix / beta) ** (-dim / 2)

    p = np.zeros((n, n))
    r = np.zeros((n, n))
    np.fill_diagonal(p, diag_norm_const)
    if pairs:
        p[i_arr, j_arr] = p_vals
        p[j_arr, i_arr] = p_vals
        r[i_arr, j_arr] = rate
        r[j_arr, i_arr] = rate

    rates = np.mean(r,axis=0)
    print(np.max(np.abs(rates)))

    p /= (np.sum(p, axis=1, keepdims=True) + np.sum(p, axis=0, keepdims=True)) * 0.5

    dm = m[None, :] - m[:, None]

    return m + beta * np.sum(dm * p, axis=1), dists, rates

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

def jet(x, t, U0, lx, L=1770000, re=6371000,
                 A=(0.0075, 0.15, 0.3),
                 c_frac=(0.1446, 0.205, 0.461)):
    """
    Evaluate the analytical Bickley jet velocity field.

    The stream function is
        psi = -U0*L*tanh(y/L)
              + sum_i A_i*U0*L*sech(y/L)**2 * cos(k_i*x - sigma_i*t),
    yielding a meandering zonal jet with three regular vortices on
    each side, on a periodic domain in x (period 2*pi*re) and y in
    roughly [-3, 3] (in units of L, unbounded in principle).

    Parameters
    ----------
    x : np.ndarray, shape (n, 2)
        Particle positions (m), with x[:,0] the zonal coordinate
        and x[:,1] the meridional coordinate.
    t : float
        Time (s).
    U0 : float
        Characteristic jet speed (m/s).
    L : float
        Jet width scale (m).
    re : float, optional
        Planetary radius scale used to set the wavenumbers
        k_i = i / re (default matches Rypina et al., 2007).
    A : tuple of float, optional
        Amplitudes of the three meandering modes.
    c_frac : tuple of float, optional
        Phase speeds c_i / U0 for the three modes.

    Returns
    -------
    np.ndarray, shape (n, 2)
        Velocity vectors (m/s).
    """
    xx = x[:, 0]
    yy = x[:, 1]-lx/2

    k = np.array([2.0, 4.0, 6.0]) / re
    c = np.array(c_frac) * U0
    sigma = c * k

    s = 1.0 / np.cosh(yy / L)      # sech(y/L)
    th = np.tanh(yy / L)

    u = U0 * s**2
    v = np.zeros_like(xx)

    for Ai, ki, sigi in zip(A, k, sigma):
        phase = ki * xx - sigi * t
        u += 2.0 * U0 * Ai * s**2 * th * np.cos(phase)
        v += -U0 * L * Ai * ki * s**2 * np.sin(phase)

    return np.stack((u, v), axis=-1)










