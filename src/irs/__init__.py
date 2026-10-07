"""IRS channel estimation and phase optimization.

Ray-tracing dependencies are loaded only by ``irs.generation``.
"""

from .channels import Channels, channel_rate
from .estimator import ChannelEstimatorConfig, NeuralChannelEstimator, load_estimator
from .solver import SolverConfig, optimize_phases

__all__ = [
    "Channels",
    "ChannelEstimatorConfig",
    "NeuralChannelEstimator",
    "SolverConfig",
    "channel_rate",
    "load_estimator",
    "optimize_phases",
]
