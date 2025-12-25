"""
SignalModel - CHIRP signal generation.
"""

import numpy as np
from typing import Tuple


class SignalModel:
    """
    CHIRP signal model.
    
    Generates linear frequency-modulated signal.
    """
    
    @staticmethod
    def generate_chirp(f_start: float, f_end: float, Tp: float, 
                      fs: float, window: str = 'Hann') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates CHIRP signal.
        
        Args:
            f_start: Start frequency, Hz
            f_end: End frequency, Hz
            Tp: Pulse duration, µs
            fs: Sampling frequency, Hz
            window: Window function ('Rect', 'Hann', 'Tukey')
        
        Returns:
            Tuple (time axis, signal)
        """
        Tp_sec = Tp * 1e-6  # Convert from µs to seconds
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
            window_func = np.tukey(N, alpha=0.5)
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

