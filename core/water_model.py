"""
WaterModel - calculation of sound speed and attenuation in water.
"""

import numpy as np
from typing import Tuple


class WaterModel:
    """
    Water model for calculating sound speed and attenuation.
    
    Uses empirical formulas for seawater.
    """
    
    @staticmethod
    def calculate_sound_speed(T: float, S: float, P: float) -> float:
        """
        Calculates sound speed in water using Mackenzie (1981) formula.
        
        Args:
            T: Temperature, °C (0-30)
            S: Salinity, PSU (0-35)
            P: Pressure, dBar (1 dBar ≈ 1 m depth)
        
        Returns:
            Sound speed, m/s
        """
        # Mackenzie (1981) formula
        c = 1448.96 + 4.591*T - 5.304e-2*T**2 + 2.374e-4*T**3
        c += 1.340*(S - 35) + 1.630e-2*P + 1.675e-7*P**2
        c -= 1.025e-2*T*(S - 35) - 7.139e-13*T*P**3
        
        return c
    
    @staticmethod
    def calculate_pressure(z: float) -> float:
        """
        Calculates pressure at depth.
        
        Args:
            z: Depth, m
        
        Returns:
            Pressure, dBar
        """
        # Approximation: 1 dBar ≈ 1 m depth
        # More accurate formula: P = 1.0 + 0.1*z (dBar)
        return 1.0 + 0.1 * z
    
    @staticmethod
    def calculate_attenuation(f: float, T: float, S: float, P: float) -> float:
        """
        Calculates sound attenuation coefficient in water.
        
        Uses Francois-Garrison formula for attenuation in seawater.
        
        Args:
            f: Frequency, Hz
            T: Temperature, °C
            S: Salinity, PSU
            P: Pressure, dBar
        
        Returns:
            Attenuation coefficient, dB/m
        """
        f_kHz = f / 1000.0  # Frequency in kHz
        
        # Viscous attenuation
        A1 = 1.03e-8 + 2.36e-10*T - 5.22e-12*T**2
        f1 = 1.32e3 * (T + 273.1) * np.exp(-1700 / (T + 273.1))
        P1 = 1.0
        
        # Relaxation attenuation (boric acid B(OH)3)
        # A2 depends on salinity
        A2 = 5.62e-8 * (1.0 + 0.00654*(S - 35))
        f2 = 1.55e7 * (T + 273.1) * np.exp(-3052 / (T + 273.1))
        P2 = 1.0 - 1.23e-4*P
        
        # Relaxation attenuation (magnesium sulfate MgSO4)
        # Correct Francois-Garrison formula for MgSO4
        # A3 depends on salinity
        A3 = 3.38e-6 * np.exp(-2000 / (T + 273.1)) * (1.0 + 0.00654*(S - 35))
        # f3 formula: different from f2, includes salinity dependence
        f3_ref = 1.55e7 * (T + 273.1) * np.exp(-3052 / (T + 273.1))
        f3 = f3_ref / (1.0 + 0.00654*(S - 35))
        P3 = 1.0 - 3.83e-5*P + 4.9e-10*P**2
        
        # Attenuation calculation
        alpha = (A1 * P1 * f1 * f_kHz**2) / (f1**2 + f_kHz**2)
        alpha += (A2 * P2 * f2 * f_kHz**2) / (f2**2 + f_kHz**2)
        alpha += (A3 * P3 * f3 * f_kHz**2) / (f3**2 + f_kHz**2)
        
        # Convert to dB/m
        alpha_db = alpha * 8.686  # Np to dB
        
        return alpha_db
    
    @staticmethod
    def calculate_spreading_loss(D: float) -> float:
        """
        Calculates losses due to geometric spreading (Spreading Loss).
        
        For spherical propagation: TL = 20 * log10(D)
        
        Args:
            D: Range, m
        
        Returns:
            Spreading loss, dB
        """
        if D <= 0:
            return 0.0
        return 20 * np.log10(D)
    
    @staticmethod
    def calculate_absorption_loss(D: float, f: float, T: float, S: float, z: float) -> float:
        """
        Calculates losses due to sound absorption in water (Absorption Loss).
        
        Args:
            D: Range, m
            f: Frequency, Hz
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
        
        Returns:
            Absorption loss, dB
        """
        P = WaterModel.calculate_pressure(z)
        alpha = WaterModel.calculate_attenuation(f, T, S, P)
        return alpha * D
    
    @staticmethod
    def calculate_transmission_loss(D: float, f: float, T: float, S: float, z: float) -> float:
        """
        Calculates transmission loss.
        
        TL = Spreading Loss + Absorption Loss
        
        Args:
            D: Range, m
            f: Frequency, Hz
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
        
        Returns:
            Transmission loss, dB
        """
        spreading = WaterModel.calculate_spreading_loss(D)
        absorption = WaterModel.calculate_absorption_loss(D, f, T, S, z)
        return spreading + absorption

