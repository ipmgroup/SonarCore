"""
ReceiverModel - receiver chain modeling (LNA, VGA, ADC).
"""

import numpy as np
from typing import Dict, Tuple


class ReceiverModel:
    """
    Receiver chain model.
    
    Includes:
    - LNA (Low Noise Amplifier)
    - VGA (Variable Gain Amplifier)
    - ADC (Analog-to-Digital Converter)
    """
    
    def __init__(self, lna_params: Dict, vga_params: Dict, adc_params: Dict):
        """
        Initialize receiver model.
        
        Args:
            lna_params: LNA parameters
            vga_params: VGA parameters
            adc_params: ADC parameters
        """
        # LNA parameters
        self.G_LNA = lna_params.get('G_LNA', 20.0)  # Gain, dB
        self.NF_LNA = lna_params.get('NF_LNA', 2.0)  # Noise figure, dB
        self.BW_LNA = lna_params.get('BW_LNA', 1e6)  # Bandwidth, Hz
        self.Z_in_LNA = lna_params.get('Z_in', 50.0)  # Input impedance, Ohms
        self.V_max_LNA = lna_params.get('V_max', 1.0)  # Maximum voltage, V
        
        # VGA parameters
        self.G_VGA_min = vga_params.get('G_min', 0.0)  # Minimum gain, dB
        self.G_VGA_max = vga_params.get('G_max', 40.0)  # Maximum gain, dB
        self.delta_G = vga_params.get('delta_G', 1.0)  # Gain step, dB
        self.BW_VGA = vga_params.get('BW_VGA', 1e6)  # Bandwidth, Hz
        self.T_set = vga_params.get('T_set', 1e-6)  # Settling time, s
        
        # ADC parameters
        self.N_bits = adc_params.get('N', 12)  # Resolution
        self.fs = adc_params.get('f_s', 1e6)  # Sampling frequency, Hz
        self.V_FS = adc_params.get('V_FS', 2.0)  # Full scale, V
        self.ENOB = adc_params.get('ENOB', self.N_bits - 1)  # Effective number of bits
        self.SNR_ADC = adc_params.get('SNR', 70.0)  # ADC SNR, dB
        
        # Internal variables
        self.current_G_VGA = self.G_VGA_min
    
    def set_vga_gain(self, G_VGA: float):
        """
        Sets VGA gain.
        
        Args:
            G_VGA: VGA gain, dB
        """
        # Limit to range
        self.current_G_VGA = np.clip(G_VGA, self.G_VGA_min, self.G_VGA_max)
        # Round to step
        self.current_G_VGA = np.round(self.current_G_VGA / self.delta_G) * self.delta_G
    
    def process_signal(self, signal: np.ndarray, add_noise: bool = True) -> Tuple[np.ndarray, float, bool, np.ndarray, np.ndarray]:
        """
        Processes signal through LNA, VGA and ADC.
        
        Args:
            signal: Input signal, V
            add_noise: Whether to add noise
        
        Returns:
            Tuple (digital signal, SNR_ADC, clipping flag, signal_after_lna, signal_after_vga)
        """
        # LNA
        G_LNA_linear = 10 ** (self.G_LNA / 20)
        signal_lna = signal * G_LNA_linear
        
        # Add LNA noise
        if add_noise:
            # Thermal noise
            k_B = 1.38e-23  # Boltzmann constant
            T = 290  # Temperature, K
            noise_power = k_B * T * self.BW_LNA * (10 ** (self.NF_LNA / 10) - 1)
            noise_rms = np.sqrt(noise_power * 50)  # Assume 50 Ohms
            noise = np.random.normal(0, noise_rms, len(signal_lna))
            signal_lna = signal_lna + noise
        
        # VGA
        G_VGA_linear = 10 ** (self.current_G_VGA / 20)
        signal_vga = signal_lna * G_VGA_linear
        
        # Check for clipping before ADC
        clipping_before_adc = np.any(np.abs(signal_vga) > self.V_max_LNA)
        if clipping_before_adc:
            signal_vga = np.clip(signal_vga, -self.V_max_LNA, self.V_max_LNA)
        
        # ADC
        # Quantization
        signal_adc = np.clip(signal_vga, -self.V_FS/2, self.V_FS/2)
        L = 2 ** self.N_bits
        signal_quantized = np.round(signal_adc / self.V_FS * L) * self.V_FS / L
        
        # Check for ADC clipping
        clipping_adc = np.any(np.abs(signal_vga) > self.V_FS/2)
        
        # Calculate SNR at ADC output
        signal_power = np.mean(signal_quantized**2)
        noise_power_adc = (self.V_FS / L)**2 / 12  # Quantization noise
        if add_noise:
            noise_power_adc += noise_power * (G_LNA_linear * G_VGA_linear)**2
        
        if noise_power_adc > 0:
            snr_linear = signal_power / noise_power_adc
            snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf
        else:
            snr_db = np.inf
        
        return signal_quantized, snr_db, clipping_adc, signal_lna, signal_vga
    
    def get_max_sampling_rate(self) -> float:
        """
        Returns maximum sampling rate.
        
        Returns:
            Sampling frequency, Hz
        """
        return self.fs

