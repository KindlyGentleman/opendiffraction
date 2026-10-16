"""Visualization suite and animation generators."""

from .animation import generate_photon_sweep_gif, generate_rotation_sweep_gif
from .figures import generate_all_visualizations
from .figures_16 import generate_all_16_figures

# Alias for backwards compatibility
generate_all_figures = generate_all_visualizations

__all__ = [
    "generate_all_visualizations",
    "generate_all_figures",
    "generate_all_16_figures",
    "generate_rotation_sweep_gif",
    "generate_photon_sweep_gif",
]
