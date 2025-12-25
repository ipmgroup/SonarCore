"""
SignalCalculator - calculation of optimal CHIRP signal parameters.
"""

import numpy as np
from typing import Dict, Tuple
from .water_model import WaterModel
from .signal_model import SignalModel


class SignalCalculator:
    """
    Signal parameter calculator.
    
    Calculates optimal CHIRP parameters based on:
    - Transducer parameters
    - Minimum range
    - Environment parameters
    """
    
    def __init__(self):
        """Initialize calculator."""
        self.water_model = WaterModel()
    
    def calculate_min_pulse_duration(self, D_min: float, T: float, S: float, z: float) -> float:
        """
        Calculates maximum allowed pulse duration based on minimum range.
        
        Criterion: transmission time must be 80% of round-trip time
        (i.e. Tp = 0.8 * TOF) to allow signal reception.
        
        Formula: Tp_max = 0.8 * (2 * D_min / c)
        where c is sound speed in water
        
        Args:
            D_min: Minimum range, m
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
        
        Returns:
            Maximum allowed pulse duration, µs
        """
        # Calculate sound speed
        P = self.water_model.calculate_pressure(z)
        c = self.water_model.calculate_sound_speed(T, S, P)
        
        # Calculate propagation time to minimum range (round-trip)
        # TOF = 2 * D_min / c (seconds)
        TOF_min_sec = 2.0 * D_min / c
        
        # Transmission time must be 80% of round-trip time
        # Tp_max = 0.8 * TOF (80% of round-trip time)
        pulse_duration_factor = 0.8
        Tp_max_sec = TOF_min_sec * pulse_duration_factor
        
        # Convert to microseconds
        Tp_max_us = Tp_max_sec * 1e6
        
        # Limit to reasonable values (50-5000 µs)
        Tp_max_us = max(50.0, min(Tp_max_us, 5000.0))
        
        return Tp_max_us
    
    def _center_frequencies_around_f0(self, f_min: float, f_max: float, f_0: float,
                                      used_bw: float) -> Tuple[float, float]:
        """
        Centers frequency range around central frequency f_0.
        
        Common logic for frequency calculation methods.
        Ensures the resulting bandwidth matches used_bw exactly (if possible within limits).
        
        Args:
            f_min: Minimum frequency, Hz
            f_max: Maximum frequency, Hz
            f_0: Central frequency, Hz
            used_bw: Bandwidth to use, Hz
        
        Returns:
            Tuple (f_start, f_end) in Hz
        """
        # Limit used_bw to available range
        max_available_bw = f_max - f_min
        if used_bw > max_available_bw:
            used_bw = max_available_bw
        
        # Center bandwidth around transducer central frequency
        f_start = max(f_min, f_0 - used_bw / 2)
        f_end = min(f_max, f_0 + used_bw / 2)
        
        # Calculate actual bandwidth after centering
        actual_bw = f_end - f_start
        
        # If actual bandwidth is less than requested, try to extend
        # but only if we haven't hit both boundaries
        if actual_bw < used_bw:
            if f_start == f_min and f_end < f_max:
                # Hit lower boundary, extend upward
                f_end = min(f_max, f_start + used_bw)
            elif f_end == f_max and f_start > f_min:
                # Hit upper boundary, extend downward
                f_start = max(f_min, f_end - used_bw)
            elif f_start > f_min and f_end < f_max:
                # No boundaries hit, extend both ways equally
                remaining_bw = used_bw - actual_bw
                f_start = max(f_min, f_start - remaining_bw / 2)
                f_end = min(f_max, f_end + remaining_bw / 2)
        
        # Final safety check: ensure f_end > f_start
        if f_end <= f_start:
            # If still invalid, use maximum available bandwidth
            f_start = f_min
            f_end = min(f_max, f_start + used_bw)
        
        return f_start, f_end
    
    def calculate_optimal_pulse_duration(self, D_target: float, T: float, S: float, z: float,
                                        min_tp: float = None) -> float:
        """
        Calculates optimal pulse duration for given distance.
        
        To achieve good SNR at different distances, duration
        should vary with distance. Longer pulses provide better SNR,
        but are limited by round-trip time.
        
        Args:
            D_target: Target range, m
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
            min_tp: Minimum duration (from D_min), µs
        
        Returns:
            Optimal pulse duration, µs
        """
        # Calculate sound speed
        P = self.water_model.calculate_pressure(z)
        c = self.water_model.calculate_sound_speed(T, S, P)
        
        # Calculate propagation time to target range (round-trip)
        TOF_target_sec = 2.0 * D_target / c
        
        # Maximum duration for this distance (80% of TOF)
        Tp_max_us = TOF_target_sec * 0.8 * 1e6
        
        # Optimal duration: use maximum for good SNR
        # (can use less, but for maximum SNR use maximum)
        # but not less than minimum (if specified)
        Tp_optimal_us = Tp_max_us
        
        if min_tp is not None:
            Tp_optimal_us = max(min_tp, Tp_optimal_us)
        
        # Limit to reasonable values
        Tp_optimal_us = max(50.0, min(Tp_optimal_us, 5000.0))
        
        return Tp_optimal_us
    
    def adjust_chirp_to_transducer(self, transducer_params: Dict, 
                                   bandwidth_ratio: float = 0.8) -> Tuple[float, float]:
        """
        Adjusts CHIRP parameters to transducer.
        
        Args:
            transducer_params: Transducer parameters (f_min, f_max, f_0, B_tr)
            bandwidth_ratio: Bandwidth utilization ratio (0.0-1.0)
        
        Returns:
            Tuple (f_start, f_end) in Hz
        """
        f_min = transducer_params.get('f_min', 150000)
        f_max = transducer_params.get('f_max', 250000)
        f_0 = transducer_params.get('f_0', (f_min + f_max) / 2)
        B_tr = transducer_params.get('B_tr', f_max - f_min)
        
        # Use specified percentage of transducer bandwidth
        used_bw = min(B_tr * bandwidth_ratio, f_max - f_min)
        
        # Center CHIRP around central frequency using common method
        return self._center_frequencies_around_f0(f_min, f_max, f_0, used_bw)
    
    def validate_chirp_parameters(self, f_start: float, f_end: float, Tp: float,
                                 transducer_params: Dict, D_min: float,
                                 T: float, S: float, z: float) -> Tuple[bool, str]:
        """
        Validates CHIRP parameters.
        
        Args:
            f_start: Start frequency, Hz
            f_end: End frequency, Hz
            Tp: Pulse duration, µs
            transducer_params: Transducer parameters
            D_min: Minimum range, m
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
        
        Returns:
            Tuple (validity, error message)
        """
        errors = []
        
        # Frequency check
        f_min = transducer_params.get('f_min', 0)
        f_max = transducer_params.get('f_max', float('inf'))
        
        if f_start < f_min:
            errors.append(f"f_start ({f_start} Hz) is less than transducer minimum frequency ({f_min} Hz)")
        
        if f_end > f_max:
            errors.append(f"f_end ({f_end} Hz) is greater than transducer maximum frequency ({f_max} Hz)")
        
        if f_start >= f_end:
            errors.append(f"f_start ({f_start} Hz) must be less than f_end ({f_end} Hz)")
        
        # Bandwidth check
        bandwidth = SignalModel.get_bandwidth(f_start, f_end)
        B_tr = transducer_params.get('B_tr', f_max - f_min)
        if bandwidth > B_tr:
            errors.append(f"CHIRP bandwidth ({bandwidth} Hz) exceeds transducer bandwidth ({B_tr} Hz)")
        
        # Maximum allowed duration check
        # Tp must not exceed 80% of round-trip time at minimum distance
        Tp_max = self.calculate_min_pulse_duration(D_min, T, S, z)
        if Tp > Tp_max:
            errors.append(f"Tp ({Tp} µs) exceeds maximum allowed duration ({Tp_max:.1f} µs) for D_min={D_min} m. "
                        f"Transmission time must be 80% of round-trip time (0.8 * TOF)")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, ""
    
    def calculate_recommended_accuracy(self, transducer_params: Dict, 
                                      f_start: float, f_end: float,
                                      Tp: float, T: float, S: float, z: float) -> float:
        """
        Calculates recommended measurement accuracy based on parameters.
        
        Accuracy depends on:
        - CHIRP signal bandwidth
        - Pulse duration
        - Environment parameters
        
        Args:
            transducer_params: Transducer parameters
            f_start: CHIRP start frequency, Hz
            f_end: CHIRP end frequency, Hz
            Tp: Pulse duration, µs
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
        
        Returns:
            Recommended accuracy (σ_D), m
        """
        # Calculate sound speed
        P = self.water_model.calculate_pressure(z)
        c = self.water_model.calculate_sound_speed(T, S, P)
        
        # Effective CHIRP bandwidth
        bandwidth = SignalModel.get_bandwidth(f_start, f_end)
        
        # Pulse duration in seconds
        Tp_sec = Tp * 1e-6
        
        # Effective bandwidth considering duration
        # For CHIRP: effective bandwidth ≈ 1 / Tp (approximately)
        BW_eff = max(bandwidth, 1.0 / Tp_sec) if Tp_sec > 0 else bandwidth
        
        # Cramér-Rao lower bound for range measurement accuracy
        # σ_D ≈ c / (2 * BW_eff * sqrt(SNR))
        # For estimation use typical SNR = 20 dB (100 in linear scale)
        typical_snr_linear = 100.0
        
        # Minimum range uncertainty
        sigma_D = c / (2.0 * BW_eff * np.sqrt(typical_snr_linear))
        
        # Limit to reasonable values (from 1 mm to 1 m)
        sigma_D = max(0.001, min(sigma_D, 1.0))
        
        return sigma_D
    
    def calculate_required_bandwidth(self, target_sigma_D: float, 
                                     T: float, S: float, z: float,
                                     snr_db: float = 20.0, Tp: float = None) -> float:
        """
        Calculates required CHIRP bandwidth to achieve target accuracy.
        
        Note: Effective bandwidth = max(bandwidth, 1/Tp), so if Tp is provided,
        the required bandwidth may be adjusted to account for pulse duration.
        
        Args:
            target_sigma_D: Target accuracy (σ_D), m
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
            snr_db: Expected SNR, dB
            Tp: Pulse duration, µs (optional, for more accurate calculation)
        
        Returns:
            Required CHIRP bandwidth, Hz
        """
        # Calculate sound speed
        P = self.water_model.calculate_pressure(z)
        c = self.water_model.calculate_sound_speed(T, S, P)
        
        # SNR in linear scale
        snr_linear = 10 ** (snr_db / 10)
        
        # From Cramér-Rao: σ_D = c / (2 * BW_eff * sqrt(SNR))
        # Where BW_eff = max(bandwidth, 1/Tp)
        # Therefore: BW_eff = c / (2 * σ_D * sqrt(SNR))
        if target_sigma_D > 0:
            required_BW_eff = c / (2.0 * target_sigma_D * np.sqrt(snr_linear))
        else:
            required_BW_eff = float('inf')
        
        # If Tp is provided, account for pulse duration contribution
        if Tp is not None and Tp > 0:
            Tp_sec = Tp * 1e-6
            bw_from_tp = 1.0 / Tp_sec
            
            # Effective bandwidth = max(bandwidth, 1/Tp)
            # If 1/Tp >= required_BW_eff, then any bandwidth is sufficient
            # If 1/Tp < required_BW_eff, then bandwidth must be >= required_BW_eff
            if bw_from_tp >= required_BW_eff:
                # Pulse duration alone provides sufficient effective bandwidth
                # Minimum bandwidth is 0, but we use a small value for practical purposes
                required_bw = max(1000.0, required_BW_eff - bw_from_tp)
            else:
                # Need bandwidth to make up the difference
                required_bw = required_BW_eff
        else:
            # No Tp provided, use simple formula
            required_bw = required_BW_eff
        
        return required_bw
    
    def get_recommended_frequencies_from_transducer(self, transducer_params: Dict,
                                                   bandwidth_ratio: float = 0.8) -> Tuple[float, float]:
        """
        Calculates recommended CHIRP frequencies from transducer central frequency and bandwidth.
        
        Recommended frequencies are always calculated from f_0 (central frequency) and B_tr
        (manufacturer's bandwidth), centered around f_0.
        
        Args:
            transducer_params: Transducer parameters from manufacturer data
            bandwidth_ratio: Bandwidth utilization ratio (0.0-1.0)
        
        Returns:
            Tuple (f_start_recommended, f_end_recommended) in Hz
        """
        f_min = transducer_params.get('f_min', 150000)
        f_max = transducer_params.get('f_max', 250000)
        f_0 = transducer_params.get('f_0', (f_min + f_max) / 2)
        B_tr = transducer_params.get('B_tr', f_max - f_min)
        
        # Always calculate from f_0 and B_tr (manufacturer's bandwidth)
        # Use bandwidth_ratio of transducer bandwidth centered around f_0
        used_bw = B_tr * bandwidth_ratio
        return self._center_frequencies_around_f0(f_min, f_max, f_0, used_bw)
    
    def calculate_user_frequencies_from_accuracy(self, transducer_params: Dict,
                                                 target_sigma_D: float,
                                                 T: float, S: float, z: float,
                                                 bandwidth_ratio: float = 0.8,
                                                 Tp: float = None) -> Tuple[float, float]:
        """
        Calculates user-defined CHIRP frequencies based on target accuracy.
        
        User frequencies are calculated from the target accuracy specified by the user.
        The bandwidth is determined solely by the target accuracy requirement.
        
        Args:
            transducer_params: Transducer parameters
            target_sigma_D: Target accuracy (σ_D) specified by user, m
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
            bandwidth_ratio: Not used for user frequencies (kept for compatibility)
            Tp: Pulse duration, µs (optional, for more accurate bandwidth calculation)
        
        Returns:
            Tuple (f_start_user, f_end_user) in Hz
        """
        f_min = transducer_params.get('f_min', 150000)
        f_max = transducer_params.get('f_max', 250000)
        f_0 = transducer_params.get('f_0', (f_min + f_max) / 2)
        B_tr = transducer_params.get('B_tr', f_max - f_min)
        
        # Calculate required bandwidth for target accuracy (accounting for Tp if provided)
        required_bw = self.calculate_required_bandwidth(target_sigma_D, T, S, z, Tp=Tp)
        
        # For user frequencies, use exactly the required bandwidth
        # (not limited by bandwidth_ratio, as user wants specific accuracy)
        # But ensure it doesn't exceed transducer limits
        max_possible_bw = min(B_tr, f_max - f_min)
        used_bw = min(required_bw, max_possible_bw)
        
        # Center bandwidth around transducer central frequency using common method
        return self._center_frequencies_around_f0(f_min, f_max, f_0, used_bw)
    
    def calculate_optimal_tp_with_constraints(self, D_min: float, D_max: float,
                                               T: float, S: float, z: float) -> float:
        """
        Calculates optimal pulse duration considering all constraints.
        
        Args:
            D_min: Minimum range, m
            D_max: Maximum range, m
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
        
        Returns:
            Optimal pulse duration, µs
        """
        # Calculate maximum allowed duration (from D_min)
        Tp_max_us = self.calculate_min_pulse_duration(D_min, T, S, z)
        
        # Calculate optimal duration for average distance
        D_target = (D_min + D_max) / 2
        Tp_optimal_us = self.calculate_optimal_pulse_duration(
            D_target, T, S, z, min_tp=None
        )
        
        # Use optimal duration, but not more than maximum
        Tp_recommended_us = min(Tp_optimal_us, Tp_max_us)
        
        return Tp_recommended_us
    
    def get_transducer_info(self, transducer_params: Dict) -> Dict:
        """
        Gets formatted transducer information for display.
        
        Args:
            transducer_params: Transducer parameters
        
        Returns:
            Dictionary with formatted transducer information
        """
        B_tr = transducer_params.get('B_tr', 0)
        f_min = transducer_params.get('f_min', 0)
        f_max = transducer_params.get('f_max', 0)
        f_0 = transducer_params.get('f_0', (f_min + f_max) / 2 if f_min > 0 and f_max > 0 else 0)
        
        # Calculate bandwidth if not provided
        if B_tr <= 0:
            B_tr = abs(f_max - f_min) if f_max > f_min else 0
        
        return {
            'bandwidth': B_tr,
            'bandwidth_khz': B_tr / 1000.0,
            'f_0': f_0,
            'f_0_khz': f_0 / 1000.0,
            'f_min': f_min,
            'f_min_khz': f_min / 1000.0,
            'f_max': f_max,
            'f_max_khz': f_max / 1000.0,
            'bandwidth_estimated': B_tr <= 0
        }
    
    
    def calculate_target_range(self, D_min: float, D_max: float) -> float:
        """
        Calculates target range (average of min and max).
        
        Args:
            D_min: Minimum range, m
            D_max: Maximum range, m
        
        Returns:
            Target range, m
        """
        return (D_min + D_max) / 2
    
    def check_bandwidth_sufficiency(self, current_bw: float, required_bw: float) -> Dict:
        """
        Checks if current bandwidth is sufficient for required bandwidth.
        
        Args:
            current_bw: Current bandwidth, Hz
            required_bw: Required bandwidth, Hz
        
        Returns:
            Dictionary with check results: {'sufficient': bool, 'current_bw': float, 'required_bw': float}
        """
        sufficient = current_bw >= required_bw
        return {
            'sufficient': sufficient,
            'current_bw': current_bw,
            'required_bw': required_bw,
            'current_bw_khz': current_bw / 1000.0,
            'required_bw_khz': required_bw / 1000.0
        }
    
    def calculate_optimal_tp_for_snr(self, D_target: float, target_snr_db: float,
                                     transducer_params: Dict, hardware_params: Dict,
                                     T: float, S: float, z: float,
                                     f_start: float, f_end: float,
                                     tx_voltage: float = 100.0) -> float:
        """
        Calculates optimal pulse duration to achieve target SNR considering attenuation.
        
        SNR depends on:
        - Signal energy (proportional to Tp)
        - Transmission loss (spreading + absorption)
        - Receiver gain (S_RX + LNA + VGA)
        - Noise level
        
        Formula: SNR = Signal_power - Noise_power - Transmission_loss + RX_gain
        Signal energy ∝ Tp, so longer Tp → better SNR
        
        Args:
            D_target: Target range, m
            target_snr_db: Target SNR at receiver, dB
            transducer_params: Transducer parameters (S_TX, S_RX, etc.)
            hardware_params: Hardware parameters (LNA gain, VGA gain, etc.)
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
            f_start: CHIRP start frequency, Hz
            f_end: CHIRP end frequency, Hz
            tx_voltage: Transmitter voltage, V
        
        Returns:
            Optimal pulse duration, µs
        """
        # Calculate sound speed
        P = self.water_model.calculate_pressure(z)
        c = self.water_model.calculate_sound_speed(T, S, P)
        
        # Maximum allowed duration (80% of round-trip time)
        TOF_sec = 2.0 * D_target / c
        Tp_max_us = TOF_sec * 0.8 * 1e6
        
        # Get parameters
        S_TX = transducer_params.get('S_TX', 170)  # dB re 1µPa/V @ 1m
        S_RX = transducer_params.get('S_RX', -193)  # dB re 1V/µPa
        lna_gain = hardware_params.get('lna_gain', 20)  # dB
        vga_gain = hardware_params.get('vga_gain', 30)  # dB
        bottom_reflection = hardware_params.get('bottom_reflection', -15)  # dB
        
        # Calculate transmission loss for round trip
        f_center = (f_start + f_end) / 2
        spreading_loss = self.water_model.calculate_spreading_loss(D_target) * 2  # Round trip
        absorption_loss = self.water_model.calculate_absorption_loss(D_target, f_center, T, S, z) * 2  # Round trip
        total_path_loss = spreading_loss + absorption_loss + bottom_reflection
        
        # Calculate total RX gain
        total_rx_gain = S_RX + lna_gain + vga_gain
        
        # Calculate TX signal level
        tx_spl_db = S_TX + 20 * np.log10(tx_voltage)  # dB re 1µPa @ 1m
        
        # Signal level at receiver (before noise)
        signal_level_db = tx_spl_db - total_path_loss + total_rx_gain
        
        # For CHIRP signals, SNR improvement is approximately 10*log10(Tp * BW)
        # where BW is the CHIRP bandwidth
        bandwidth = abs(f_end - f_start)
        
        # Estimate noise level (typical thermal noise + receiver noise)
        # Typical noise floor: -120 to -140 dB re 1µPa (depends on bandwidth)
        # Simplified: noise_floor ≈ -130 dB + 10*log10(BW/1kHz)
        noise_floor_db = -130 + 10 * np.log10(bandwidth / 1000.0) if bandwidth > 0 else -130
        
        # Current SNR (without Tp contribution)
        # SNR = Signal - Noise + Processing_gain
        # Processing gain for CHIRP ≈ 10*log10(Tp * BW)
        # We need: SNR_current + 10*log10(Tp * BW) >= target_snr_db
        
        # Estimate current SNR (for reference Tp = 1 ms)
        Tp_ref_us = 1000.0  # Reference: 1 ms
        Tp_ref_sec = Tp_ref_us * 1e-6
        processing_gain_ref_db = 10 * np.log10(Tp_ref_sec * bandwidth) if bandwidth > 0 else 0
        snr_ref_db = signal_level_db - noise_floor_db + processing_gain_ref_db
        
        # Calculate required processing gain
        required_processing_gain_db = target_snr_db - snr_ref_db
        
        # Processing gain = 10*log10(Tp * BW)
        # Therefore: Tp = 10^(required_gain/10) / BW
        if required_processing_gain_db > 0 and bandwidth > 0:
            Tp_required_sec = (10 ** (required_processing_gain_db / 10.0)) / bandwidth
            Tp_required_us = Tp_required_sec * 1e6
        else:
            # If SNR is already sufficient, use minimum reasonable Tp
            Tp_required_us = 100.0
        
        # Use maximum of required Tp and minimum Tp, but not more than maximum allowed
        Tp_optimal_us = max(100.0, min(Tp_required_us, Tp_max_us))
        
        # Limit to reasonable range
        Tp_optimal_us = max(50.0, min(Tp_optimal_us, 5000.0))
        
        return Tp_optimal_us

