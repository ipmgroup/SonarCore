"""
TransducerModel - transducer model.
"""

import numpy as np
from typing import Dict, Optional


class TransducerModel:
    """
    Hydroacoustic transducer model.
    
    Characteristics:
    - Frequency response
    - TX/RX sensitivity
    - Directivity pattern
    - Ring-down time
    """
    
    def __init__(self, params: Dict):
        """
        Initialize transducer model.
        
        Args:
            params: Dictionary with transducer parameters
        """
        self.model = params.get('model', 'Unknown')
        self.f_min = params.get('f_min', 0.0)
        self.f_max = params.get('f_max', 0.0)
        self.f_0 = params.get('f_0', (self.f_min + self.f_max) / 2)
        self.B_tr = params.get('B_tr', self.f_max - self.f_min)  # -3 dB bandwidth
        self.S_TX = params.get('S_TX', 0.0)  # TX sensitivity, dB re 1 µPa/V
        self.S_RX = params.get('S_RX', 0.0)  # RX sensitivity, dB re 1 V/µPa
        self.Theta_BW = params.get('Theta_BW', 10.0)  # Beam width, degrees
        self.Q = params.get('Q', 10.0)  # Q-factor
        self.T_rd = params.get('T_rd', 0.0)  # Ring-down time, µs
        self.Z = params.get('Z', 50.0)  # Impedance, Ohms
        self.source = params.get('source', '')
        self.version = params.get('version', '1.0')
    
    def get_tx_sensitivity(self, f: float) -> float:
        """
        Returns TX sensitivity at frequency f.
        
        Args:
            f: Frequency, Hz
        
        Returns:
            Sensitivity, dB re 1 µPa/V
        """
        # Simple model: sensitivity drops outside bandwidth
        if f < self.f_min or f > self.f_max:
            # Attenuation outside bandwidth
            if f < self.f_min:
                df = (self.f_min - f) / self.B_tr
            else:
                df = (f - self.f_max) / self.B_tr
            attenuation = -40 * df  # -40 dB/octave
            return self.S_TX + attenuation
        return self.S_TX
    
    def get_rx_sensitivity(self, f: float) -> float:
        """
        Returns RX sensitivity at frequency f.
        
        Args:
            f: Frequency, Hz
        
        Returns:
            Sensitivity, dB re 1 V/µPa
        """
        # Similar to TX
        if f < self.f_min or f > self.f_max:
            if f < self.f_min:
                df = (self.f_min - f) / self.B_tr
            else:
                df = (f - self.f_max) / self.B_tr
            attenuation = -40 * df
            return self.S_RX + attenuation
        return self.S_RX
    
    def get_directivity(self, theta: float) -> float:
        """
        Returns directivity coefficient.
        
        Args:
            theta: Angle from axis, degrees
        
        Returns:
            Directivity coefficient (linear)
        """
        # Simplified model: cosine directivity pattern
        theta_rad = np.deg2rad(theta)
        theta_bw_rad = np.deg2rad(self.Theta_BW / 2)
        
        if abs(theta) <= self.Theta_BW / 2:
            return np.cos(theta_rad / theta_bw_rad * np.pi / 2)
        else:
            # Side lobes
            return 0.1 * np.cos(theta_rad / theta_bw_rad * np.pi / 2)
    
    def get_ringdown_time(self) -> float:
        """
        Returns ring-down time.
        
        Returns:
            Ring-down time, µs
        """
        return self.T_rd
    
    def validate_frequency(self, f: float) -> bool:
        """
        Checks if frequency is in operating range.
        
        Args:
            f: Frequency, Hz
        
        Returns:
            True if frequency is in range
        """
        return self.f_min <= f <= self.f_max

