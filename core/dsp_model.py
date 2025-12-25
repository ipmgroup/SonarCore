"""
DSPModel - signal processing and TOF (Time of Flight) calculation.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


class DSPModel:
    """
    Digital signal processing model.
    
    Performs:
    - Matched filtering
    - TOF determination
    - Range calculation
    """
    
    def __init__(self, fs: float):
        """
        Initialize DSP model.
        
        Args:
            fs: Sampling frequency, Hz
        """
        self.fs = fs
    
    def matched_filter(self, received: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Performs matched filtering.
        
        Args:
            received: Received signal
            reference: Reference signal (CHIRP)
        
        Returns:
            Matched filter response
        """
        # Matched filtering via correlation
        correlation = np.correlate(received, reference, mode='full')
        # Normalization
        correlation = correlation / np.max(np.abs(correlation))
        return correlation
    
    def find_peak(self, signal: np.ndarray, threshold: float = 0.5, use_interpolation: bool = True) -> Optional[float]:
        """
        Finds peak in signal with optional interpolation for sub-sample accuracy.
        
        Args:
            signal: Input signal
            threshold: Peak search threshold
            use_interpolation: If True, use parabolic interpolation for sub-sample accuracy
        
        Returns:
            Peak index (float if interpolation used, int otherwise) or None
        """
        # Find maximum
        max_idx = np.argmax(np.abs(signal))
        max_val = np.abs(signal[max_idx])
        
        if max_val < threshold:
            return None
        
        if not use_interpolation:
            return float(max_idx)
        
        # Parabolic interpolation for sub-sample accuracy
        # Fit parabola through peak and two neighbors: y = ax^2 + bx + c
        if max_idx > 0 and max_idx < len(signal) - 1:
            y0 = np.abs(signal[max_idx - 1])
            y1 = np.abs(signal[max_idx])
            y2 = np.abs(signal[max_idx + 1])
            
            # Parabolic interpolation: find x where derivative = 0
            # y = a*x^2 + b*x + c, where x is relative to max_idx
            # At x=-1: y0, at x=0: y1, at x=1: y2
            # Solving: a = (y2 - 2*y1 + y0) / 2, b = (y2 - y0) / 2
            # Peak at x = -b / (2*a) = -(y2 - y0) / (2*(y2 - 2*y1 + y0))
            denominator = 2 * (y2 - 2 * y1 + y0)
            if abs(denominator) > 1e-10:  # Avoid division by zero
                offset = -(y2 - y0) / denominator
                # Limit offset to reasonable range (should be between -0.5 and 0.5)
                offset = np.clip(offset, -0.5, 0.5)
                peak_pos = max_idx + offset
            else:
                peak_pos = float(max_idx)
        else:
            peak_pos = float(max_idx)
        
        return peak_pos
    
    def calculate_tof(self, received: np.ndarray, reference: np.ndarray, 
                     c: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates TOF (Time of Flight) and range.
        
        Args:
            received: Received signal
            reference: Reference signal
            c: Sound speed, m/s
        
        Returns:
            Tuple (TOF in seconds, range in meters)
        """
        # Matched filtering
        correlation = self.matched_filter(received, reference)
        
        # Peak search with interpolation for sub-sample accuracy
        peak_idx = self.find_peak(correlation, use_interpolation=True)
        
        if peak_idx is None:
            return None, None
        
        # For np.correlate with mode='full':
        # - Output length = len(received) + len(reference) - 1
        # - Index 0 corresponds to reference completely before received
        # - Index len(reference)-1 corresponds to reference[0] aligned with received[0]
        # - Peak at index peak_idx means delay = peak_idx - (len(reference) - 1)
        # 
        # Correct calculation: delay in samples from start of received signal
        # When mode='full', correlation[len(reference)-1] means reference[0] aligns with received[0]
        # So actual delay = peak_idx - (len(reference) - 1)
        delay_samples = peak_idx - (len(reference) - 1)
        
        # Log for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"DSPModel.calculate_tof: received length={len(received)}, reference length={len(reference)}, "
                   f"correlation length={len(correlation)}, peak_idx={peak_idx:.2f}, "
                   f"len(reference)-1={len(reference)-1}, delay_samples={delay_samples:.2f}")
        
        # TOF = delay in seconds (delay_samples can be float due to interpolation)
        tof = delay_samples / self.fs
        
        # Range = TOF * c / 2 (round trip)
        # TOF is for round trip (TX -> target -> RX), so D = TOF * c / 2 gives one-way distance
        D = tof * c / 2
        
        # Log for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"DSPModel.calculate_tof: delay_samples={delay_samples:.2f}, fs={self.fs:.0f}, "
                   f"tof={tof*1e3:.3f}ms, c={c:.2f}m/s, D_measured={D:.2f}m, "
                   f"tof*c={tof*c:.2f}m (round-trip distance), D_measured should be one-way")
        
        return tof, D
    
    def estimate_range_uncertainty(self, correlation: np.ndarray, 
                                   c: float, snr_db: float) -> float:
        """
        Estimates range measurement uncertainty.
        
        Args:
            correlation: Matched filter response
            c: Sound speed, m/s
            snr_db: SNR, dB
        
        Returns:
            Range standard deviation, m
        """
        # Cramér-Rao lower bound for TOF
        # sigma_TOF ≈ 1 / (BW * sqrt(SNR))
        
        # Approximate bandwidth estimate from peak width
        peak_idx = np.argmax(np.abs(correlation))
        half_max = np.abs(correlation[peak_idx]) / 2
        
        # Find peak width at half maximum
        left_idx = peak_idx
        right_idx = peak_idx
        
        while left_idx > 0 and np.abs(correlation[left_idx]) > half_max:
            left_idx -= 1
        
        while right_idx < len(correlation) - 1 and np.abs(correlation[right_idx]) > half_max:
            right_idx += 1
        
        pulse_width_samples = right_idx - left_idx
        pulse_width_sec = pulse_width_samples / self.fs
        
        # Effective bandwidth ≈ 1 / pulse duration
        BW_eff = 1.0 / pulse_width_sec if pulse_width_sec > 0 else self.fs / 2
        
        # SNR in linear scale
        snr_linear = 10 ** (snr_db / 10) if snr_db > -np.inf else 0
        
        # Cramér-Rao bound
        if snr_linear > 0 and BW_eff > 0:
            sigma_tof = 1.0 / (BW_eff * np.sqrt(snr_linear))
        else:
            sigma_tof = 1.0 / self.fs  # Minimum uncertainty
        
        # Convert to range uncertainty
        sigma_D = sigma_tof * c / 2
        
        return sigma_D

