"""
ENOB Calculator - calculation of Effective Number of Bits (ENOB) for acoustic sonar path.

Based on formulas from README.md section 12.9.
"""

import numpy as np
from typing import Dict


class ENOBCalculator:
    """
    Calculator for Effective Number of Bits (ENOB) in acoustic sonar signal path.
    
    Calculates ENOB at different stages:
    - Analog ENOB (at VGA/PGA output, before ADC)
    - Digital ENOB (after correlation/matched filtering)
    """
    
    def calculate_enob(self, 
                      signal_input_voltage: float,
                      bandwidth: float,
                      lna_params: Dict,
                      vga_params: Dict,
                      adc_params: Dict,
                      vga_gain: float,
                      chirp_duration: float,
                      sample_rate: float) -> Dict:
        """
        Calculates ENOB for the complete signal path.
        
        Args:
            signal_input_voltage: Input signal voltage at hydrophone, V
            bandwidth: Signal bandwidth, Hz
            lna_params: LNA parameters (G_LNA, NF_LNA, input_noise, etc.)
            vga_params: VGA parameters (G_min, G_max, input_noise, etc.)
            adc_params: ADC parameters (N, V_FS, etc.)
            vga_gain: Current VGA gain, dB (AFTER optimization/simulation)
            chirp_duration: CHIRP pulse duration, seconds (AFTER optimization)
            sample_rate: ADC sampling rate, Hz
        
        Returns:
            Dictionary with ENOB calculation results
        """
        # Get parameters
        G_LNA_db = lna_params.get('G_LNA', lna_params.get('G', 20.0))
        G_LNA_linear = 10 ** (G_LNA_db / 20.0)
        G_VGA_linear = 10 ** (vga_gain / 20.0)
        
        # Get noise parameters
        e_n_lna = lna_params.get('input_noise', None)
        if e_n_lna is None:
            NF_LNA = lna_params.get('NF_LNA', lna_params.get('NF', 2.0))
            e_n_lna = 0.9e-9 * (10 ** (NF_LNA / 20.0))
        
        e_n_vga = vga_params.get('input_noise_0dB', None)
        if e_n_vga is None:
            e_n_vga = vga_params.get('input_noise_80dB', None)
        if e_n_vga is None:
            e_n_vga = vga_params.get('input_noise', None)
        if e_n_vga is None:
            e_n_vga = 1.0e-9
        
        # ADC parameters
        N_bits = adc_params.get('N', 12)
        V_FS = adc_params.get('V_FS', 2.0)
        adc_enob_nominal = adc_params.get('ENOB', N_bits - 1)
        
        # Calculate noise
        noise_lna_rms = e_n_lna * np.sqrt(bandwidth)
        noise_lna_out = noise_lna_rms * G_LNA_linear
        noise_vga_rms = e_n_vga * np.sqrt(bandwidth)
        noise_vga_out = noise_vga_rms * G_VGA_linear
        noise_total_rms = np.sqrt((noise_lna_out * G_VGA_linear) ** 2 + (noise_vga_out) ** 2)
        
        # Calculate signal
        signal_output_v = signal_input_voltage * G_LNA_linear * G_VGA_linear
        
        # Analog SNR and ENOB
        # NOTE: Analog SNR is measured at VGA OUTPUT (before ADC quantization)
        # This is different from SNR_ADC in receiver_model.py which is measured
        # at ADC OUTPUT (after quantization).
        # 
        # Differences:
        # 1. Measurement point: VGA output (analog) vs ADC output (digital)
        # 2. Noise model: input_noise (e_n * sqrt(BW)) vs thermal noise (k_B*T*BW*(10^(NF/10)-1))
        # 3. ADC quantization: NOT included in analog SNR, included in SNR_ADC
        # 
        # Typical difference: Analog SNR is usually lower than SNR_ADC because:
        # - ADC quantization noise may be smaller than analog noise for well-designed systems
        # - Different noise models may give different results
        if noise_total_rms > 0:
            snr_analog_linear = signal_output_v / noise_total_rms
            snr_analog_db = 20.0 * np.log10(snr_analog_linear)
        else:
            snr_analog_db = np.inf
        
        if np.isfinite(snr_analog_db):
            enob_analog = (snr_analog_db - 1.76) / 6.02
            enob_analog = max(0.0, min(enob_analog, N_bits))
        else:
            enob_analog = N_bits
        
        # Digital SNR and ENOB (after correlation)
        ns_samples = int(chirp_duration * sample_rate)
        correlation_gain_db = 10.0 * np.log10(ns_samples) if ns_samples > 0 else 0.0
        snr_digital_db = snr_analog_db + correlation_gain_db
        
        if np.isfinite(snr_digital_db):
            enob_digital = (snr_digital_db - 1.76) / 6.02
            enob_digital = max(0.0, min(enob_digital, N_bits))
        else:
            enob_digital = N_bits
        
        return {
            'signal_input_v': signal_input_voltage,
            'signal_output_v': signal_output_v,
            'noise_lna_rms': noise_lna_rms,
            'noise_lna_out': noise_lna_out,
            'noise_vga_rms': noise_vga_rms,
            'noise_vga_out': noise_vga_out,
            'noise_total_rms': noise_total_rms,
            'snr_analog_db': snr_analog_db,
            'enob_analog': enob_analog,
            'ns_samples': ns_samples,
            'correlation_gain_db': correlation_gain_db,
            'snr_digital_db': snr_digital_db,
            'enob_digital': enob_digital,
            'adc_n_bits': N_bits,
            'adc_enob_nominal': adc_enob_nominal,
            'bandwidth_hz': bandwidth,
            'bandwidth_khz': bandwidth / 1000.0,
            'chirp_duration_s': chirp_duration,
            'chirp_duration_us': chirp_duration * 1e6,
            'sample_rate_hz': sample_rate,
            'g_lna_db': G_LNA_db,
            'g_vga_db': vga_gain,
            'e_n_lna_v_per_sqrt_hz': e_n_lna,
            'e_n_vga_v_per_sqrt_hz': e_n_vga,
        }
    
    def format_enob_report(self, enob_results: Dict) -> str:
        """Formats ENOB calculation results as a readable report."""
        lines = []
        lines.append("=" * 70)
        lines.append("ENOB (Effective Number of Bits) Calculation")
        lines.append("=" * 70)
        lines.append("")
        lines.append("IMPORTANT: SNR Measurement Points")
        lines.append("-" * 70)
        lines.append("  Analog SNR (below): Measured at VGA OUTPUT (before ADC quantization)")
        lines.append("  - Noise model: input_noise (e_n * sqrt(BW))")
        lines.append("  - Does NOT include ADC quantization noise")
        lines.append("")
        lines.append("  Measured SNR (in recommendations): Measured at ADC OUTPUT (after quantization)")
        lines.append("  - Noise model: thermal noise (k_B*T*BW*(10^(NF/10)-1))")
        lines.append("  - Includes ADC quantization noise")
        lines.append("  - These two SNRs may differ by several dB due to different")
        lines.append("    measurement points and noise models")
        lines.append("")
        lines.append("-" * 70)
        lines.append("")
        
        lines.append("Signal Path Parameters:")
        lines.append(f"  Input signal voltage: {enob_results['signal_input_v']*1e6:.3f} µV")
        lines.append(f"  Bandwidth: {enob_results['bandwidth_khz']:.2f} kHz")
        lines.append(f"  CHIRP duration: {enob_results['chirp_duration_us']:.2f} µs")
        lines.append(f"  Sample rate: {enob_results['sample_rate_hz']/1e6:.2f} MSPS")
        lines.append(f"  Number of samples: {enob_results['ns_samples']}")
        lines.append("")
        
        lines.append("Amplifier Parameters:")
        lines.append(f"  LNA gain: {enob_results['g_lna_db']:.1f} dB")
        lines.append(f"  LNA noise: {enob_results['e_n_lna_v_per_sqrt_hz']*1e9:.2f} nV/√Hz")
        lines.append(f"  VGA gain: {enob_results['g_vga_db']:.1f} dB")
        lines.append(f"  VGA noise: {enob_results['e_n_vga_v_per_sqrt_hz']*1e9:.2f} nV/√Hz")
        lines.append("")
        
        lines.append("Noise Calculations:")
        lines.append(f"  LNA RMS noise: {enob_results['noise_lna_rms']*1e6:.3f} µV")
        lines.append(f"  LNA output noise: {enob_results['noise_lna_out']*1e3:.3f} mV")
        lines.append(f"  VGA RMS noise: {enob_results['noise_vga_rms']*1e6:.3f} µV")
        lines.append(f"  VGA output noise: {enob_results['noise_vga_out']*1e3:.3f} mV")
        lines.append(f"  Total RMS noise (VGA output): {enob_results['noise_total_rms']*1e3:.3f} mV")
        lines.append("")
        
        lines.append("Signal Calculations:")
        lines.append(f"  Signal at VGA output: {enob_results['signal_output_v']*1e3:.3f} mV")
        lines.append("")
        
        lines.append("Analog (Before Correlation):")
        lines.append("  Measurement point: VGA OUTPUT (before ADC quantization)")
        lines.append("  Noise model: input_noise (e_n * sqrt(BW))")
        lines.append("  NOTE: This SNR is different from 'Measured SNR' which is")
        lines.append("        measured at ADC OUTPUT (after quantization)")
        lines.append(f"  SNR: {enob_results['snr_analog_db']:.2f} dB")
        lines.append(f"  ENOB: {enob_results['enob_analog']:.2f} bits")
        lines.append("")
        
        lines.append("Digital (After Correlation/Matched Filtering):")
        lines.append(f"  Correlation gain: {enob_results.get('correlation_gain_db', 0.0):.2f} dB")
        lines.append(f"  SNR: {enob_results['snr_digital_db']:.2f} dB")
        lines.append(f"  ENOB: {enob_results['enob_digital']:.2f} bits")
        lines.append("")
        
        lines.append("ADC Comparison:")
        lines.append(f"  ADC nominal resolution: {enob_results['adc_n_bits']} bits")
        lines.append(f"  ADC nominal ENOB: {enob_results['adc_enob_nominal']:.2f} bits")
        lines.append(f"  Effective ENOB (analog): {enob_results['enob_analog']:.2f} bits")
        lines.append(f"  Effective ENOB (digital): {enob_results['enob_digital']:.2f} bits")
        lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)

