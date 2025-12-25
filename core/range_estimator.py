"""
RangeEstimator - calculation of measured range and error.
"""

import numpy as np
from typing import Tuple, Optional
from .dsp_model import DSPModel
from .water_model import WaterModel


class RangeEstimator:
    """
    Range estimator.
    
    Calculates:
    - Measured range D_measured
    - Standard deviation σ_D
    """
    
    def __init__(self, dsp_model: DSPModel, water_model: WaterModel):
        """
        Initialize range estimator.
        
        Args:
            dsp_model: DSP model
            water_model: Water model
        """
        self.dsp_model = dsp_model
        self.water_model = water_model
    
    def estimate_range(self, received_signal: np.ndarray, reference_signal: np.ndarray,
                      T: float, S: float, z: float, snr_db: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimates range and its uncertainty.
        
        Args:
            received_signal: Received signal
            reference_signal: Reference signal
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
            snr_db: SNR, dB
        
        Returns:
            Tuple (D_measured, sigma_D)
        """
        # Calculate sound speed
        P = self.water_model.calculate_pressure(z)
        c = self.water_model.calculate_sound_speed(T, S, P)
        
        # Calculate TOF and range
        tof, D_measured = self.dsp_model.calculate_tof(received_signal, reference_signal, c)
        
        if D_measured is None:
            return None, None
        
        # Calculate uncertainty
        correlation = self.dsp_model.matched_filter(received_signal, reference_signal)
        sigma_D = self.dsp_model.estimate_range_uncertainty(correlation, c, snr_db)
        
        return D_measured, sigma_D

