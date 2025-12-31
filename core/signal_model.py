"""
SignalModel - CHIRP signal generation.
"""

import numpy as np
from typing import Tuple
from scipy import signal as scipy_signal


class SignalModel:
    """
    CHIRP signal model.
    
    Generates linear frequency-modulated signal.
    """
    
    @staticmethod
    def generate_chirp(f_start: float, f_end: float, Tp: float, 
                      fs: float, window: str = 'Hann', limit_tp_for_fast_calc: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates CHIRP signal (reference signal for matched filtering).
        
        This generates the reference signal s_ref(t) for matched filter processing.
        
        The reference signal MUST have the same parameters as the transmitted signal:
        - Same pulse duration T (Tp)
        - Same frequency sweep (f_start, f_end)
        - Same window function (Tukey or Hanning - mandatory for matched filtering!)
        - Same sampling rate (fs)
        
        Signal formula:
        s_ref(t) = w(t) * cos(2π * (f_start*t + (f_end-f_start)/(2*Tp)*t²))
        
        where:
        - w(t) is the window function (Tukey or Hanning recommended)
        - f_start: Start frequency, Hz
        - f_end: End frequency, Hz
        - Tp: Pulse duration, µs
        - fs: Sampling frequency, Hz
        
        For matched filtering, Tukey (α≈0.25) or Hanning window is REQUIRED
        to suppress side lobes in the correlation response.
        
        Args:
            f_start: Start frequency, Hz
            f_end: End frequency, Hz
            Tp: Pulse duration, µs
            fs: Sampling frequency, Hz
            window: Window function ('Rect', 'Hann', 'Tukey')
                   Note: 'Tukey' or 'Hann' is REQUIRED for matched filtering
            limit_tp_for_fast_calc: If True and Tp > 1s, limit to 1s for fast calculation
        
        Returns:
            Tuple (time axis, signal) where signal is s_ref(t)
        """
        Tp_sec = Tp * 1e-6  # Convert from µs to seconds
        
        # Apply limit if requested and Tp > 1 second
        MAX_TP_FOR_FAST_CALC_SEC = 1.0  # 1 second
        original_tp_sec = Tp_sec
        if limit_tp_for_fast_calc and Tp_sec > MAX_TP_FOR_FAST_CALC_SEC:
            Tp_sec = MAX_TP_FOR_FAST_CALC_SEC
            import warnings
            import logging
            logger = logging.getLogger(__name__)
            warning_msg = (f"Tp limited to {MAX_TP_FOR_FAST_CALC_SEC} s for fast calculation "
                          f"(original: {original_tp_sec:.2f} s = {Tp * 1e-6:.2f} s). "
                          f"Uncheck 'Limit Tp to 1s' for full simulation.")
            warnings.warn(warning_msg, UserWarning)
            logger.info(warning_msg)  # Also log to logger for visibility
        
        N = int(fs * Tp_sec)
        t = np.arange(N) / fs
        
        # Linear CHIRP
        # Phase: phi(t) = 2*pi * (f_start*t + (f_end-f_start)/(2*Tp)*t^2)
        phase = 2 * np.pi * (f_start * t + (f_end - f_start) / (2 * Tp_sec) * t**2)
        signal = np.cos(phase)
        
        # Apply window function
        if window == 'Rect':
            window_func = np.ones(N)
        elif window == 'Hann':
            window_func = np.hanning(N)
        elif window == 'Tukey':
            # For SBP, use alpha=0.25 as recommended in SBP.md section 23.9
            # This provides better side lobe suppression (≈-32 dB vs ≈-13 dB for Rect)
            window_func = scipy_signal.windows.tukey(N, alpha=0.25)
        else:
            window_func = np.ones(N)
        
        signal = signal * window_func
        
        return t, signal
    
    @staticmethod
    def get_bandwidth(f_start: float, f_end: float) -> float:
        """
        Calculates CHIRP signal bandwidth.
        
        Args:
            f_start: Start frequency, Hz
            f_end: End frequency, Hz
        
        Returns:
            Bandwidth, Hz
        """
        return abs(f_end - f_start)
    
    @staticmethod
    def get_instantaneous_frequency(t: np.ndarray, f_start: float, 
                                   f_end: float, Tp: float) -> np.ndarray:
        """
        Calculates instantaneous frequency of CHIRP signal.
        
        Args:
            t: Time axis, s
            f_start: Start frequency, Hz
            f_end: End frequency, Hz
            Tp: Pulse duration, µs
        
        Returns:
            Instantaneous frequency, Hz
        """
        Tp_sec = Tp * 1e-6
        return f_start + (f_end - f_start) / Tp_sec * t

