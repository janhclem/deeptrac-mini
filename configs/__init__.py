"""
Configuration files for DeepTrac Mini.

This directory contains INI configuration files for different simulation setups.
Use minitraclib.Config.read_ini() to load them.

Available configurations:
    - gyre_100km.ini: Double-gyre flow on 100km x 100km domain
    - jet_large.ini: Bickley jet on 20000km x 10000km stratospheric domain

Usage:
    import minitraclib as mini
    cfg = mini.Config.read_ini('bickley_jet')
    # or with full path
    cfg = mini.Config.read_ini('/path/to/custom.ini')
"""
