#!/usr/bin/env python
"""
Generate 1000 training ensemble config files based on gyre0.ini.

Structure:
- 10 mass initialization types
- 2 ddiff0 values (0.0, 10.0)
- 50 u0 values between 0 and 25
- Total: 10 * 2 * 50 = 1000 configs

Config naming: {init_type}_{ddiff0_val}_{u0_val}.ini
Example: gradient_0_0.ini, gradient_0_1.ini, ..., stripes_10_10_49.ini
"""

import configparser
import os
import shutil
from pathlib import Path
import numpy as np

# Mass initialization types
INIT_TYPES = [
    'gradient',
    'sharp', 
    'three_gauss',
    'smiley',
    'checkerboard_3',
    'checkerboard_4',
    'checkerboard_8',
    'stripes_4',
    'stripes_10',
    'random'
]

# ddiff0 values
DDIFF0_VALUES = [0.0, 10.0]

# u0 values (0 to 49, step 1)
U0_VALUES = list(np.linspace(0, 25,50))

# Base config file
BASE_CONFIG = 'configs/gyre/gyre0.ini'

# Output directory
OUTPUT_DIR = 'configs/training_ensemble'


def load_base_config():
    """Load the base gyre0.ini configuration."""
    config = configparser.ConfigParser()
    config.read(BASE_CONFIG)
    return config


def generate_configs():
    """Generate all 1000 config files."""
    base_config = load_base_config()
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Creating output directory: {OUTPUT_DIR}")
    
    count = 0
    for init_type in INIT_TYPES:
        for ddiff0 in DDIFF0_VALUES:
            for u0 in U0_VALUES:
                # Create a copy of the base config
                new_config = configparser.ConfigParser()
                new_config.read_dict(base_config)
                
                # Modify SIMULATION section
                new_config['SIMULATION']['init_type'] = init_type
                
                # Modify MIXING section
                new_config['MIXING']['ddiff0'] = str(ddiff0)
                
                # Modify ADVECTION section
                new_config['ADVECTION']['u0'] = str(u0)
                
                # Also update flow_type to gyre for consistency
                new_config['SIMULATION']['flow_type'] = 'gyre'
                
                # Set the plotting time step...
                if (u0 == 20):
                	new_config["TIME"]["dt_plot"] = "20.0"
                else:
                	new_config["TIME"]["dt_plot"] = "1e16"
                
                # Generate filename
                # Format ddiff0: 0.0 -> "0", 10.0 -> "10"
                ddiff0_str = f"{int(ddiff0)}" if ddiff0 == int(ddiff0) else str(ddiff0).replace('.', '_')
                filename = f"{init_type}_{ddiff0_str}_{u0}.ini"
                filepath = Path(OUTPUT_DIR) / filename
                
                # Write config file
                with open(filepath, 'w') as f:
                    new_config.write(f)
                
                count += 1
                
                if count % 100 == 0:
                    print(f"Generated {count} configs...")
    
    print(f"\nDone! Generated {count} config files in {OUTPUT_DIR}/")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Init types: {len(INIT_TYPES)}")
    print(f"  ddiff0 values: {DDIFF0_VALUES}")
    print(f"  u0 values: {len(U0_VALUES)} ({U0_VALUES[0]} to {U0_VALUES[-1]})")
    print(f"  Total configs: {len(INIT_TYPES)} * {len(DDIFF0_VALUES)} * {len(U0_VALUES)} = {count}")


if __name__ == '__main__':
    generate_configs()
