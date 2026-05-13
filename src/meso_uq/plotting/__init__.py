"""Lightweight plotting contracts.

Matplotlib and other rendering libraries are imported only by rendering
adapters, never by this package's import path.
"""

from .contracts import FigureManifest, PlotSeries, XYPlotRequest, configure_headless_environment

__all__ = [
    "FigureManifest",
    "PlotSeries",
    "XYPlotRequest",
    "configure_headless_environment",
]
