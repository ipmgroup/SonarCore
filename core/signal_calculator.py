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
    
    def _calculate_max_tp_from_distance(self, D: float, T: float, S: float, z: float) -> float:
        """
        Common helper: calculates maximum allowed pulse duration (80% of round-trip time) for given distance.
        
        Formula: Tp_max = 0.8 * (2 * D / c)
        where c is sound speed in water
        
        Args:
            D: Range/distance, m
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
        
        Returns:
            Maximum allowed pulse duration, µs
        """
        # Calculate sound speed
        P = self.water_model.calculate_pressure(z)
        c = self.water_model.calculate_sound_speed(T, S, P)
        
        # DEBUG: Log calculation parameters
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"_calculate_max_tp_from_distance: D={D:.2f}m, T={T:.1f}°C, S={S:.1f}PSU, z={z:.1f}m, P={P:.2f}Pa, c={c:.2f}m/s")
        
        # Calculate propagation time (round-trip)
        # TOF = 2 * D / c (seconds)
        TOF_sec = 2.0 * D / c
        
        # Transmission time must be 80% of round-trip time
        # Tp_max = 0.8 * TOF (80% of round-trip time)
        pulse_duration_factor = 0.8
        Tp_max_sec = TOF_sec * pulse_duration_factor
        
        # Convert to microseconds
        Tp_max_us = Tp_max_sec * 1e6
        
        logger.debug(f"_calculate_max_tp_from_distance: TOF={TOF_sec*1000:.2f}ms, Tp_max={Tp_max_us:.2f}µs")
        
        # Ensure minimum 1 µs to avoid numerical issues
        Tp_max_us = max(1.0, Tp_max_us)
        
        return Tp_max_us
    
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
        return self._calculate_max_tp_from_distance(D_min, T, S, z)
    
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
        # Maximum duration for this distance (80% of TOF)
        Tp_max_us = self._calculate_max_tp_from_distance(D_target, T, S, z)
        
        # Optimal duration: use maximum for good SNR
        # (can use less, but for maximum SNR use maximum)
        # but not less than minimum (if specified)
        Tp_optimal_us = Tp_max_us
        
        if min_tp is not None:
            Tp_optimal_us = max(min_tp, Tp_optimal_us)
        
        # Ensure minimum 1 µs to avoid numerical issues
        Tp_optimal_us = max(1.0, Tp_optimal_us)
        
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
        
        NOTE: This returns the MINIMUM pulse duration for MINIMUM distance (D_min).
        This is the maximum allowed Tp based on physical constraint:
        Tp cannot exceed 80% of round-trip time at D_min to allow signal reception.
        
        For target distance (D_target), use calculate_optimal_pulse_duration(D_target, ...)
        which may return a larger value if D_target > D_min.
        
        Args:
            D_min: Minimum range, m
            D_max: Maximum range, m
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
        
        Returns:
            Minimum pulse duration for minimum distance, µs
            (This is the maximum allowed Tp based on D_min constraint)
        """
        # Calculate maximum allowed duration (from D_min)
        # This is the MINIMUM pulse duration for MINIMUM distance
        # Formula: Tp_max = 0.8 * (2 * D_min / c) where c is sound speed
        Tp_max_us = self.calculate_min_pulse_duration(D_min, T, S, z)
        
        # Calculate optimal duration for average distance
        D_target = (D_min + D_max) / 2
        Tp_optimal_us = self.calculate_optimal_pulse_duration(
            D_target, T, S, z, min_tp=None
        )
        
        # Use optimal duration, but not more than maximum
        # This ensures Tp doesn't exceed physical constraint from D_min
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
    
    def calculate_tp_reduction_for_snr(self, current_tp: float, current_snr_db: float, 
                                       target_snr_db: float, D_target: float,
                                       T: float, S: float, z: float) -> float:
        """
        Calculates reduced Tp when current SNR is above target SNR.
        
        SNR scales with 10*log10(Tp), so to reduce SNR by X dB, reduce Tp by factor 10^(-X/10).
        Formula: Tp_new = Tp_current * 10^((target_snr - SNR_current) / 10)
        
        Args:
            current_tp: Current pulse duration, µs
            current_snr_db: Current SNR, dB
            target_snr_db: Target SNR, dB
            D_target: Target distance, m
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
        
        Returns:
            Reduced pulse duration, µs (constrained by physical limits)
        """
        # Calculate reduction factor
        snr_reduction_needed = current_snr_db - target_snr_db
        tp_reduction_factor = 10 ** (-snr_reduction_needed / 10.0)
        optimal_tp = current_tp * tp_reduction_factor
        
        # Check physical constraint: Tp must not exceed 80% of round-trip time at D_target
        Tp_max_us = self._calculate_max_tp_from_distance(D_target, T, S, z)
        optimal_tp = min(optimal_tp, Tp_max_us)
        
        # Ensure minimum 1 µs
        optimal_tp = max(1.0, optimal_tp)
        
        return optimal_tp
    
    def _calculate_snr_from_sonar_equation(self, D: float, Tp_us: float,
                                          transducer_params: Dict, hardware_params: Dict,
                                          T: float, S: float, z: float,
                                          f_start: float, f_end: float,
                                          tx_voltage: float = 100.0) -> float:
        """
        Calculates SNR using active sonar equation.
        
        Common function used by both calculate_optimal_tp_for_snr and calculate_max_depth_for_snr_tp.
        
        Uses the active sonar equation:
        SNR = SL + TS + DI - NL - 2TL + 10*log10(BT)
        
        Args:
            D: Range/depth, m
            Tp_us: Pulse duration, µs
            transducer_params: Transducer parameters
            hardware_params: Hardware parameters
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
            f_start: CHIRP start frequency, Hz
            f_end: CHIRP end frequency, Hz
            tx_voltage: Transmitter voltage, V
        
        Returns:
            SNR, dB
        """
        # Get parameters
        S_TX = transducer_params.get('S_TX', 170)  # dB re 1µPa/V @ 1m
        bottom_reflection = hardware_params.get('bottom_reflection', -15)  # dB (reflection loss, negative)
        
        # Calculate bandwidth
        bandwidth = abs(f_end - f_start)  # Hz
        f_center = (f_start + f_end) / 2
        Tp_sec = Tp_us * 1e-6  # Convert to seconds
        
        # === ACTIVE SONAR EQUATION ===
        # SNR = SL + TS + DI - NL - 2TL + 10*log10(BT)
        
        # 1. Source Level (SL)
        SL = S_TX + 20 * np.log10(tx_voltage)  # dB re 1µPa @ 1m
        
        # 2. Target Strength (TS)
        TS = -bottom_reflection  # dB (target strength, positive)
        
        # 3. Directivity Index (DI)
        DI = 0.0  # dB
        
        # 4. Noise Level (NL)
        base_noise_spectrum_level_db = 120.0  # dB re 1µPa @ 1kHz (spectrum level)
        bandwidth_khz = bandwidth / 1000.0 if bandwidth > 0 else 1.0
        NL = base_noise_spectrum_level_db + 10 * np.log10(bandwidth_khz)
        lna_nf = hardware_params.get('lna_nf', 2.0)  # dB
        if lna_nf > 1.0:
            NL += (lna_nf - 1.0) * 0.2  # Small contribution
        
        # 5. Transmission Loss (TL) - one-way
        spreading_loss_one_way = self.water_model.calculate_spreading_loss(D)  # One-way
        absorption_loss_one_way = self.water_model.calculate_absorption_loss(D, f_center, T, S, z)  # One-way
        TL = spreading_loss_one_way + absorption_loss_one_way  # One-way TL
        
        # 6. Pulse Compression Gain: 10*log10(BT)
        BT_product = bandwidth * Tp_sec
        PG_db = 10 * np.log10(BT_product) if BT_product > 0 else 0
        
        # Calculate SNR using active sonar equation
        SNR = SL + TS + DI - NL - 2*TL + PG_db
        
        return SNR
    
    def calculate_optimal_tp_for_snr(self, D_target: float, target_snr_db: float,
                                     transducer_params: Dict, hardware_params: Dict,
                                     T: float, S: float, z: float,
                                     f_start: float, f_end: float,
                                     tx_voltage: float = 100.0) -> float:
        """
        Calculates optimal pulse duration to achieve target SNR using active sonar equation.
        
        Uses the active sonar equation:
        SNR = SL + TS + DI - NL - 2TL + 10*log10(BT)
        
        where:
        - SL = Source Level (transmitted signal level at 1m) = S_TX + 20*log10(V)
        - TS = Target Strength (reflection coefficient, positive for bottom)
        - DI = Directivity Index (assumed 0 for omnidirectional)
        - NL = Noise Level (ambient + receiver noise, positive dB)
        - TL = Transmission Loss (one-way path loss) = spreading + absorption
        - 2TL = round-trip path loss
        - BT = bandwidth * time (pulse compression product)
        - 10*log10(BT) = pulse compression gain
        
        For CHIRP signals, pulse compression gain is: 10*log10(BT)
        where B = bandwidth (Hz), T = pulse duration (seconds).
        
        Rearranging to find Tp:
        10*log10(BT) = SNR_target - (SL + TS + DI - NL - 2TL)
        BT = 10^((SNR_target - (SL + TS + DI - NL - 2TL)) / 10)
        T = BT / B
        Tp = (BT / B) * 1e6  (convert to microseconds)
        
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
        # Maximum allowed duration (80% of round-trip time)
        Tp_max_us = self._calculate_max_tp_from_distance(D_target, T, S, z)
        
        # Get parameters
        S_TX = transducer_params.get('S_TX', 170)  # dB re 1µPa/V @ 1m
        bottom_reflection = hardware_params.get('bottom_reflection', -15)  # dB (reflection loss, negative)
        
        # Calculate bandwidth
        bandwidth = abs(f_end - f_start)  # Hz
        f_center = (f_start + f_end) / 2
        
        # Calculate components needed for Tp calculation
        # Use same formulas as in _calculate_snr_from_sonar_equation
        SL = S_TX + 20 * np.log10(tx_voltage)
        TS = -bottom_reflection
        DI = 0.0
        
        # Noise Level
        base_noise_spectrum_level_db = 120.0
        bandwidth_khz = bandwidth / 1000.0 if bandwidth > 0 else 1.0
        NL = base_noise_spectrum_level_db + 10 * np.log10(bandwidth_khz)
        lna_nf = hardware_params.get('lna_nf', 2.0)
        if lna_nf > 1.0:
            NL += (lna_nf - 1.0) * 0.2
        
        # Transmission Loss (TL) - one-way
        spreading_loss_one_way = self.water_model.calculate_spreading_loss(D_target)
        absorption_loss_one_way = self.water_model.calculate_absorption_loss(D_target, f_center, T, S, z)
        TL = spreading_loss_one_way + absorption_loss_one_way
        
        # === CALCULATE REQUIRED Tp ===
        # From sonar equation: SNR = SL + TS + DI - NL - 2TL + 10*log10(BT)
        # Rearranging to find BT:
        # 10*log10(BT) = SNR_target - (SL + TS + DI - NL - 2TL)
        # BT = 10^((SNR_target - (SL + TS + DI - NL - 2TL)) / 10)
        # T = BT / B
        # Tp = (BT / B) * 1e6  (convert to microseconds)
        
        # Calculate required pulse compression product (BT)
        required_PG_db = target_snr_db - (SL + TS + DI - NL - 2*TL)
        
        # Calculate required Tp from BT product
        if bandwidth > 0:
            # BT = 10^(PG / 10)
            BT_product = 10 ** (required_PG_db / 10.0)
            # T = BT / B
            Tp_required_sec = BT_product / bandwidth
            Tp_required_us = Tp_required_sec * 1e6
        else:
            # Fallback if bandwidth is 0 - use minimum physical constraint
            Tp_required_us = 1.0
        
        # Apply only physical constraint: Tp must not exceed 80% of round-trip time
        # No artificial minimum or maximum limits
        Tp_optimal_us = min(Tp_required_us, Tp_max_us)
        
        # Ensure minimum 1 µs to avoid numerical issues
        Tp_optimal_us = max(1.0, Tp_optimal_us)
        
        return Tp_optimal_us
    
    def calculate_max_depth_for_snr_tp(self, target_snr_db: float, Tp_us: float,
                                      transducer_params: Dict, hardware_params: Dict,
                                      T: float, S: float, z: float,
                                      f_start: float, f_end: float,
                                      tx_voltage: float = 100.0,
                                      D_min: float = 0.1, D_max: float = 1000.0) -> float:
        """
        Calculates maximum depth/range for given SNR and pulse duration.
        
        Uses active sonar equation and solves for maximum D where SNR >= target_SNR.
        Uses iterative approach (binary search) to find maximum depth.
        
        Args:
            target_snr_db: Target SNR, dB
            Tp_us: Pulse duration, µs
            transducer_params: Transducer parameters
            hardware_params: Hardware parameters
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m (used for pressure calculation)
            f_start: CHIRP start frequency, Hz
            f_end: CHIRP end frequency, Hz
            tx_voltage: Transmitter voltage, V
            D_min: Minimum depth to search, m
            D_max: Maximum depth to search, m
        
        Returns:
            Maximum depth/range, m (or 0 if not achievable)
        """
        # Binary search for maximum depth
        # We need to find maximum D where SNR(D) >= target_snr_db
        # Use common function _calculate_snr_from_sonar_equation
        left = D_min
        right = D_max
        max_valid_D = 0.0
        tolerance = 0.1  # 10 cm tolerance
        
        for _ in range(50):  # Max 50 iterations
            D_test = (left + right) / 2.0
            
            # Calculate SNR for this depth using common function
            SNR_calc = self._calculate_snr_from_sonar_equation(
                D_test, Tp_us, transducer_params, hardware_params,
                T, S, z, f_start, f_end, tx_voltage
            )
            
            if SNR_calc >= target_snr_db:
                # This depth is valid, try larger depth
                max_valid_D = D_test
                left = D_test
            else:
                # SNR too low, try smaller depth
                right = D_test
            
            if right - left < tolerance:
                break
        
        return max_valid_D
    
    def calculate_depth_vs_tp_curve(self, target_snr_db: float,
                                    transducer_params: Dict, hardware_params: Dict,
                                    T: float, S: float, z: float,
                                    f_start: float, f_end: float,
                                    tx_voltage: float = 100.0,
                                    Tp_min_us: float = 50.0, Tp_max_us: float = 5000.0,
                                    num_points: int = 100,
                                    D_max_search: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates depth vs pulse duration curve for given SNR.
        
        Args:
            target_snr_db: Target SNR, dB
            transducer_params: Transducer parameters
            hardware_params: Hardware parameters
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
            f_start: CHIRP start frequency, Hz
            f_end: CHIRP end frequency, Hz
            tx_voltage: Transmitter voltage, V
            Tp_min_us: Minimum pulse duration, µs
            Tp_max_us: Maximum pulse duration, µs
            num_points: Number of points in curve
            D_max_search: Maximum depth to search, m (default: 10000.0)
        
        Returns:
            Tuple (Tp_array, Depth_array) - Tp_array in µs, Depth_array in m
        """
        Tp_array = np.linspace(Tp_min_us, Tp_max_us, num_points)
        Depth_array = np.zeros_like(Tp_array)
        
        # Use provided D_max_search to find maximum achievable depth
        # Don't limit by physical constraint of Tp - we want to see what depth is achievable for given SNR
        
        for i, Tp_us in enumerate(Tp_array):
            # Use D_max_search directly - don't limit by physical constraint
            # This shows maximum depth achievable for given SNR, regardless of Tp physical limit
            Depth_array[i] = self.calculate_max_depth_for_snr_tp(
                target_snr_db, Tp_us,
                transducer_params, hardware_params,
                T, S, z, f_start, f_end, tx_voltage,
                D_min=0.1, D_max=D_max_search
            )
        
        return Tp_array, Depth_array
    
    def calculate_tgc_curve(self, D_min: float, D_max: float, target_snr_db: float,
                           transducer_params: Dict, hardware_params: Dict,
                           adc_params: Dict, vga_params: Dict,
                           T: float, S: float, z: float,
                           f_start: float, f_end: float, Tp_us: float,
                           tx_voltage: float = 100.0,
                           num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates Time Gain Control (TGC) curve: optimal VGA gain vs distance.
        
        For each distance, calculates VGA gain that:
        1. Achieves target SNR or better
        2. Avoids ADC clipping
        
        Args:
            D_min: Minimum distance, m
            D_max: Maximum distance, m
            target_snr_db: Target SNR, dB
            transducer_params: Transducer parameters
            hardware_params: Hardware parameters (LNA gain, LNA NF, etc.)
            adc_params: ADC parameters (V_FS, bits, etc.)
            vga_params: VGA parameters (G_min, G_max, etc.)
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
            f_start: CHIRP start frequency, Hz
            f_end: CHIRP end frequency, Hz
            Tp_us: Pulse duration, µs
            tx_voltage: Transmitter voltage, V
            num_points: Number of points in curve
        
        Returns:
            Tuple (Distance_array, VGA_gain_array) - Distance_array in m, VGA_gain_array in dB
        """
        Distance_array = np.linspace(D_min, D_max, num_points)
        VGA_gain_array = np.zeros_like(Distance_array)
        
        # Get parameters
        S_TX = transducer_params.get('S_TX', 170)  # dB re 1µPa/V @ 1m
        S_RX = transducer_params.get('S_RX', -193)  # dB re 1V/µPa
        bottom_reflection = hardware_params.get('bottom_reflection', -15)  # dB
        
        lna_gain = hardware_params.get('lna_gain', 20.0)  # dB
        lna_nf = hardware_params.get('lna_nf', 2.0)  # dB
        
        vga_gain_min = vga_params.get('G_min', 0.0)  # dB
        vga_gain_max = vga_params.get('G_max', 60.0)  # dB
        
        adc_v_fs = adc_params.get('V_FS', adc_params.get('V_ref', 2.5))  # V (full scale)
        adc_bits = adc_params.get('N', adc_params.get('bits', 16))
        adc_max_voltage = adc_v_fs / 2.0  # Maximum voltage without clipping, V
        
        # Calculate bandwidth and center frequency
        bandwidth = abs(f_end - f_start)  # Hz
        f_center = (f_start + f_end) / 2
        Tp_sec = Tp_us * 1e-6  # Convert to seconds
        
        # Calculate Source Level
        SL = S_TX + 20 * np.log10(tx_voltage)  # dB re 1µPa @ 1m
        
        # Calculate Target Strength
        TS = -bottom_reflection  # dB
        
        # Calculate Directivity Index
        DI = 0.0  # dB
        
        # Calculate Noise Level (constant for all distances)
        base_noise_spectrum_level_db = 120.0  # dB re 1µPa @ 1kHz
        bandwidth_khz = bandwidth / 1000.0 if bandwidth > 0 else 1.0
        NL = base_noise_spectrum_level_db + 10 * np.log10(bandwidth_khz)
        if lna_nf > 1.0:
            NL += (lna_nf - 1.0) * 0.2
        
        # Pulse Compression Gain (constant for all distances)
        BT_product = bandwidth * Tp_sec
        PG_db = 10 * np.log10(BT_product) if BT_product > 0 else 0
        
        for i, D in enumerate(Distance_array):
            # Calculate Transmission Loss (one-way)
            spreading_loss_one_way = self.water_model.calculate_spreading_loss(D)
            absorption_loss_one_way = self.water_model.calculate_absorption_loss(D, f_center, T, S, z)
            TL = spreading_loss_one_way + absorption_loss_one_way  # One-way TL
            
            # Calculate SNR without VGA gain (at LNA output)
            # SNR = SL + TS + DI - NL - 2TL + PG_db
            # But we need to account for VGA gain contribution
            # SNR_with_VGA = SNR_at_LNA + VGA_gain (approximately, for signal)
            # But noise also gets amplified, so it's more complex
            
            # Calculate SNR at LNA output (without VGA)
            snr_at_lna = SL + TS + DI - NL - 2*TL + PG_db
            
            # Calculate signal level at LNA output (in dB re 1µPa)
            signal_level_at_lna_db = SL + TS - 2*TL + lna_gain  # dB re 1µPa
            
            # Convert to voltage at LNA output
            # S_RX is sensitivity: dB re 1V/µPa
            # signal_voltage = signal_pressure * 10^(S_RX/20)
            signal_pressure_at_lna = 10 ** ((signal_level_at_lna_db) / 20)  # µPa (linear)
            signal_voltage_at_lna = signal_pressure_at_lna * 10 ** (S_RX / 20)  # V (linear)
            
            # Calculate required VGA gain for target SNR
            # SNR scales with signal power, which scales with VGA gain squared
            # But noise also gets amplified, so net effect: SNR increases with VGA gain
            # Approximate: SNR_VGA ≈ SNR_LNA + G_VGA (for high SNR cases)
            # More accurate: SNR_VGA = SNR_LNA + G_VGA - noise_contribution
            # For TGC, we use simplified model: SNR_VGA ≈ SNR_LNA + G_VGA
            snr_deficit = target_snr_db - snr_at_lna
            vga_gain_for_snr = max(0.0, snr_deficit)  # Only increase if needed
            
            # Calculate maximum VGA gain to avoid clipping
            # Signal voltage after VGA: V_VGA = V_LNA * 10^(G_VGA/20)
            # To avoid clipping: V_VGA < ADC_max_voltage
            # G_VGA_max = 20*log10(ADC_max_voltage / V_LNA)
            if signal_voltage_at_lna > 0:
                vga_gain_max_for_no_clip = 20 * np.log10(adc_max_voltage / signal_voltage_at_lna)
            else:
                vga_gain_max_for_no_clip = vga_gain_max
            
            # TGC principle: compensate for transmission loss to maintain constant signal level
            # Ideal TGC: G_VGA = 2*TL (compensate round-trip loss)
            # But also need to ensure SNR >= target and no clipping
            tgc_compensation = 2 * TL  # Compensate for round-trip loss
            
            # Choose optimal VGA gain:
            # 1. Must be >= gain for SNR
            # 2. Must be <= gain for no clipping
            # 3. Should be close to TGC compensation (2*TL) if possible
            # 4. Must be within VGA limits
            optimal_vga_gain = max(vga_gain_for_snr, tgc_compensation)  # At least compensate for loss or meet SNR
            optimal_vga_gain = min(optimal_vga_gain, vga_gain_max_for_no_clip, vga_gain_max)  # But not too high
            
            # Ensure it's within VGA limits
            optimal_vga_gain = max(vga_gain_min, optimal_vga_gain)
            
            VGA_gain_array[i] = optimal_vga_gain
        
        return Distance_array, VGA_gain_array

