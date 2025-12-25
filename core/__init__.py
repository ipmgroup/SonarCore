"""
CORE модуль - вычислительное ядро системы симуляции гидроакустического сонара.
"""

from .water_model import WaterModel
from .transducer_model import TransducerModel
from .signal_model import SignalModel
from .channel_model import ChannelModel
from .receiver_model import ReceiverModel
from .dsp_model import DSPModel
from .range_estimator import RangeEstimator
from .optimizer import Optimizer
from .simulator import Simulator
from .signal_calculator import SignalCalculator
from .signal_path import SignalPathCalculator

__all__ = [
    'WaterModel',
    'TransducerModel',
    'SignalModel',
    'ChannelModel',
    'ReceiverModel',
    'DSPModel',
    'RangeEstimator',
    'Optimizer',
    'Simulator',
    'SignalCalculator',
    'SignalPathCalculator',
]

