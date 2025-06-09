"""Plot styles for matplotlib.
This module provides a class PlotStyles that contains methods to initialize plot styles and get color palettes for matplotlib plots.
It also provides a method to format labels for plots.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle


class PlotUtils:

    __light = False

    @staticmethod
    def init_plot_style(light: bool = False, large: bool = False) -> None:
        __light = light
        if not light:
            mplstyle.use('dark_background')
        plt.rcParams.update({
            'font.size': 18 if not large else 26,
            'axes.titlesize': 22 if large else 28,
            'axes.labelsize': 20 if not large else 26,
            'xtick.labelsize': 16 if not large else 20,
            'ytick.labelsize': 16 if not large else 20,
            'legend.fontsize': 16 if not large else 26,
            'figure.titlesize': 24 if large else 32,
        })


    @staticmethod
    def get_palette() -> str:
        palettes = [ "Oranges", "Dark2"]
        return palettes[0]

    @staticmethod
    def title_label(label: str) -> str:
        """Return the label with underscores replaced by spaces and each word capitalized. 3 letters are all caps."""
        if len(label) == 3: return label.upper()
        return label.replace('_', ' ').title()


    @staticmethod
    def _get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Plot sweep results from local pickle file")
        parser.add_argument("--sweep-id",       type=str,  required=False, default=None,  help="W&B sweep ID (e.g., 'entity/project/sweep_id')")
        parser.add_argument("--wandb-project",  type=str,  required=False, default=None,  help="Weights & Biases project name")    
        parser.add_argument("--llm-model",      type=str,  required=False, default=None,  help="LLM model")            
        parser.add_argument("--temperature",    type=float, required=False, default=0.7,  help="Temperature for LLM, e.g. '0.1' or '0.7'")
        parser.add_argument("--folders",        type=Path, required=False, default=None,  nargs='+', help="Directory containing sweep results and the pickle file")
        parser.add_argument("--files",          type=str,  required=False, default=None,  nargs='+', help="File to process")      
        parser.add_argument("--conditions",     type=str,  required=False, default=None,  nargs='+', help="Conditions to filter, e.g. 'chunk_size==768' or 'embedding_model==sentence-transformers/all-MiniLM-L6-v2'")
        parser.add_argument("--light",          action="store_true", required=False, default=False, help="Use light theme")
        parser.add_argument("--debug",          action="store_true", required=False, default=False, help="Debug mode")
        return parser

    @staticmethod
    def print_help():
        parser = PlotUtils._get_parser()
        parser.print_help()

    @staticmethod
    def parse_args():
        parser = PlotUtils._get_parser()
        return parser.parse_args()