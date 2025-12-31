"""
SBP Profile Generator - generates sub-bottom profile from sediment model.

Implements profile formation algorithm:
1. Calculate time delays for each interface
2. Calculate reflection amplitudes with attenuation
3. Sum all reflections to form profile
4. Apply TVG compensation
5. Process with matched filter
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import signal as scipy_signal
from .sediment_model import SedimentModel, SedimentLayer
from .signal_model import SignalModel
from .water_model import WaterModel
from .dsp_model import DSPModel


class SBPProfileGenerator:
    """
    Sub-bottom profile generator.
    
    Generates profile from multi-layer sediment model.
    """
    
    def __init__(self, water_model: WaterModel, dsp_model=None):
        """
        Initialize profile generator.
        
        Args:
            water_model: WaterModel instance for sound speed calculation
            dsp_model: Optional DSPModel instance for matched filtering.
                      If None, will be created when needed (requires fs).
        """
        self.water_model = water_model
        self.dsp_model = dsp_model
    
    def generate_profile(self, 
                        reference_signal: np.ndarray,
                        t_reference: np.ndarray,
                        sediment_model: SedimentModel,
                        water_depth: float,
                        T: float,
                        S: float,
                        z: float,
                        fs: float,
                        f_center: float = 200e3,
                        apply_tvg: bool = True,
                        tvg_g0: float = 0.0,
                        tvg_alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
        """
        Generate sub-bottom profile.
        
        Formula: s_rx(t) = sum_i R_i * s_tx(t - τ_i) * A_i
        
        where:
        - R_i: reflection coefficient at interface i
        - τ_i: time delay to interface i
        - A_i: attenuation factor for round-trip through layers
        
        Args:
            reference_signal: Transmitted CHIRP signal
            t_reference: Time axis for reference signal, s
            sediment_model: SedimentModel instance
            water_depth: Water depth, m
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
            fs: Sampling frequency, Hz
            apply_tvg: If True, apply Time Variant Gain compensation
            tvg_g0: TVG initial gain, dB
            tvg_alpha: TVG attenuation coefficient, dB/m (if None, uses average from sediment)
        
        Returns:
            Tuple (profile_amplitudes, profile_depths, interface_depths, interface_amplitudes)
        """
        # Calculate sound speed in water
        P = self.water_model.calculate_pressure(z)
        c_water = self.water_model.calculate_sound_speed(T, S, P)
        
        # Water impedance (for reflection coefficient calculation)
        rho_water = 1025.0  # kg/m³ (seawater)
        water_impedance = rho_water * c_water
        
        # Get number of interfaces
        num_interfaces = sediment_model.get_num_interfaces()
        
        # Calculate time delays and amplitudes for each interface
        interface_delays = []
        interface_amplitudes = []
        interface_depths = []
        
        for i in range(num_interfaces):
            # Get interface depth in sediment (from bottom, not from surface)
            # This is the depth of the interface below the seafloor
            interface_depth_in_sediment = sediment_model.get_interface_depth(i)
            # For display: total depth from surface = water_depth + depth_in_sediment
            interface_depth_from_surface = water_depth + interface_depth_in_sediment
            interface_depths.append(interface_depth_in_sediment)  # Store depth in sediment for profile
            
            # Calculate time of flight to interface
            # Path: surface -> bottom (water) -> interface (sediment)
            # Round-trip: surface -> bottom -> interface -> bottom -> surface
            tof = sediment_model.get_time_of_flight_to_interface(
                i, water_depth, c_water
            )
            interface_delays.append(tof)
            
            # Get reflection coefficient at interface i
            R_linear, R_dB = sediment_model.get_reflection_coefficient(
                i, water_impedance
            )
            
            # Calculate cumulative transmission coefficient (product of T for all layers before interface)
            # T_i represents energy entering layer i from previous layer
            # For signal to reach interface i, it must pass through all previous interfaces
            # Formula: T_cumulative = T_0 * T_1 * ... * T_{i-1}
            # where T_j is transmission coefficient at interface j (entering layer j+1)
            T_cumulative_linear = 1.0  # Start with unity
            for j in range(i):
                # Get transmission coefficient at interface j (entering layer j+1 from layer j)
                # For j=0: water -> layer 0
                # For j>0: layer j-1 -> layer j
                T_j_linear, _ = sediment_model.get_transmission_coefficient(j, water_impedance)
                T_cumulative_linear *= abs(T_j_linear)
            
            # Calculate attenuation in sediment (round-trip from bottom to interface)
            # This is the path: bottom -> interface -> bottom (in sediment only)
            # Use frequency-dependent attenuation: α(f) = α_0 * f^n
            sediment_attenuation_db = sediment_model.calculate_attenuation_to_interface(i, f_center)
            
            # Calculate water attenuation (round-trip through water column)
            # Path: surface -> bottom -> surface (always the same for all interfaces)
            # Distance: 2 * water_depth (round-trip)
            water_attenuation_db = self.water_model.calculate_absorption_loss(
                2 * water_depth, f_center, T, S, z
            )
            
            # Spreading loss: for SBP, spreading occurs in water (spherical)
            # After reflection from bottom, signal propagates in sediment (cylindrical, less spreading)
            # For simplicity, we use spreading loss for water path only (to bottom and back)
            # Total path to interface: water (2*water_depth) + sediment (2*interface_depth_in_sediment)
            # But spreading is mainly in water, so we use water_depth for spreading
            water_spreading_db = self.water_model.calculate_spreading_loss(
                2 * water_depth
            )
            
            # Note: For more accurate model, spreading loss could be calculated for total path
            # (water + sediment), but in practice, spreading in sediment is much less
            # due to guided propagation, so we use water path only for spreading
            
            total_water_tl = water_attenuation_db + water_spreading_db
            
            # Total attenuation: water (to bottom and back) + sediment (from bottom to interface and back)
            # Formula: TL_total = TL_water(2*water_depth) + TL_sediment(2*interface_depth_in_sediment)
            total_tl_db = total_water_tl + sediment_attenuation_db
            
            # Amplitude calculation according to physics:
            # A(z) ~ T_cumulative * e^(-2αz) * R_i
            # where:
            # - T_cumulative: product of transmission coefficients (energy entering each layer)
            # - e^(-2αz): attenuation during round-trip in sediment
            # - R_i: reflection coefficient at interface i
            # Formula: A = T_cumulative * |R| * 10^(-TL/20)
            amplitude = T_cumulative_linear * abs(R_linear) * (10 ** (-total_tl_db / 20))
            interface_amplitudes.append(amplitude)
        
        # Create time axis for profile (long enough to capture all reflections)
        max_delay = max(interface_delays) if interface_delays else 0.0
        profile_duration = max_delay + len(reference_signal) / fs + 0.01  # Add 10ms margin
        num_samples = int(profile_duration * fs)
        t_profile = np.arange(num_samples) / fs
        
        # Initialize profile signal
        profile_signal = np.zeros(num_samples)
        
        # Sum all reflections
        for i in range(num_interfaces):
            delay = interface_delays[i]
            amplitude = interface_amplitudes[i]
            
            # Find delay in samples
            delay_samples = int(delay * fs)
            
            # Place reference signal at delay position
            if delay_samples + len(reference_signal) <= len(profile_signal):
                profile_signal[delay_samples:delay_samples + len(reference_signal)] += (
                    reference_signal * amplitude
                )
        
        # Apply TVG if requested
        if apply_tvg:
            # TVG formula: G(t) = G0 + 20*log10(t) + 2*α*z(t)
            # where z(t) = c_sediment * t / 2
            
            # Use average sound speed in sediment
            avg_sound_speed = np.mean([layer.sound_speed for layer in sediment_model.layers])
            
            # Use average attenuation at reference frequency if not specified
            if tvg_alpha is None:
                tvg_alpha = np.mean([layer.attenuation_ref for layer in sediment_model.layers])
            
            # Calculate TVG gain for each time sample
            tvg_gain_db = np.zeros(len(t_profile))
            for j, t in enumerate(t_profile):
                if t > 0:
                    # Depth corresponding to time t (in sediment, after water)
                    # Time in sediment = t - tof_water
                    tof_water = 2 * water_depth / c_water
                    t_sediment = max(0.0, t - tof_water)
                    z_sediment = avg_sound_speed * t_sediment / 2
                    
                    # TVG gain
                    tvg_gain_db[j] = tvg_g0 + 20 * np.log10(t) + 2 * tvg_alpha * z_sediment
                else:
                    tvg_gain_db[j] = tvg_g0
            
            # Convert to linear scale and apply
            tvg_gain_linear = 10 ** (tvg_gain_db / 20)
            profile_signal = profile_signal * tvg_gain_linear
        
        # Convert time axis to depth axis (depth from surface, including water)
        # First echo (water-bottom) should be at water_depth
        # Layer interfaces should be at water_depth + interface_depth_in_sediment
        avg_sound_speed = np.mean([layer.sound_speed for layer in sediment_model.layers])
        
        # Calculate water round-trip time
        tof_water_round_trip = 2 * water_depth / c_water
        
        # Convert time to depth from surface
        # For first echo (water-bottom): t = tof_water_round_trip, depth = water_depth
        # For layer interfaces: t = tof_water_round_trip + t_sediment, depth = water_depth + depth_in_sediment
        profile_depths = np.zeros(len(t_profile))
        for j, t in enumerate(t_profile):
            if t <= tof_water_round_trip:
                # In water column: depth from surface = c_water * t / 2 (round-trip)
                profile_depths[j] = c_water * t / 2
            else:
                # After reaching bottom: depth = water_depth + depth_in_sediment
                t_sediment = t - tof_water_round_trip
                depth_in_sediment = avg_sound_speed * t_sediment / 2
                profile_depths[j] = water_depth + depth_in_sediment
        
        return profile_signal, profile_depths, interface_depths, interface_amplitudes
    
    def process_profile_with_matched_filter(self,
                                           profile_signal: np.ndarray,
                                           reference_signal: np.ndarray,
                                           fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process profile with matched filter (correlation).
        
        Optimal processing for CHIRP in additive noise is the matched filter:
        
        y(t) = ∫ x(τ) * s_ref*(τ - t) dτ
        
        where:
        - x(t) = profile_signal: received signal (sum of reflections)
        - s_ref(t) = reference_signal: reference CHIRP signal
        
        REQUIREMENTS for reference signal:
        The reference CHIRP signal MUST have:
        1. Same pulse duration T (Tp)
        2. Same frequency sweep (f_start, f_end)
        3. Same window function (Tukey or Hanning - mandatory!)
        4. Same sampling rate (fs)
        
        Example reference signal formation:
        s_ref(t) = w(t) * cos(2π * (f0*t + (B/(2*T))*t²))
        
        where:
        - w(t) is Tukey or Hanning window (required to suppress side lobes)
        - f0 = f_start (start frequency)
        - B = f_end - f_start (bandwidth)
        - T = Tp (pulse duration)
        
        Uses mode='full' for correlation (as in user's example) to get full correlation
        with correct lag alignment. Lags range from -len(reference) + 1 to len(profile_signal) - 1.
        
        Args:
            profile_signal: Received signal x(t) (profile signal with reflections)
            reference_signal: Reference CHIRP signal s_ref(t) (must match transmitted signal)
            fs: Sampling frequency, Hz (must match reference signal sampling rate)
        
        Returns:
            Tuple (processed_signal, envelope, correlation_lags) where:
            - processed_signal: Correlation output y(t) (length = len(profile_signal) + len(reference) - 1)
            - envelope: |Hilbert(y(t))| - envelope detection to remove oscillations
            - correlation_lags: Lag indices (delay in samples) for correlation alignment
        """
        # Use DSPModel.matched_filter for correlation (single correlation function for all tasks in core)
        # Create DSPModel if not provided (requires fs)
        if self.dsp_model is None:
            self.dsp_model = DSPModel(fs)
        
        # Matched filter: correlation between received signal (profile_signal) and reference signal
        # DSPModel.matched_filter handles time-reversal and uses mode='full' internally
        correlated = self.dsp_model.matched_filter(profile_signal, reference_signal)
        
        # DSPModel.matched_filter normalizes the result, but we may need unnormalized for envelope calculation
        # Store normalization factor to potentially restore original scale if needed
        # For now, use normalized correlation (consistent with other uses in core)
        
        # NOTE: Oscillations in correlation appear because:
        # 1. profile_signal is a SUM of multiple reflections (not a single CHIRP):
        #    profile_signal(t) = Σ A_i * reference(t - τ_i)
        #    where: A_i = amplitude, τ_i = delay of i-th interface reflection
        # 2. When correlating SUM with reference, we get interference between reflections:
        #    correlate(Σ A_i * ref(t-τ_i), ref(t)) = Σ A_i * correlate(ref(t-τ_i), ref(t))
        #    - Each reflection creates its own correlation peak
        #    - Peaks interfere with each other → oscillations between peaks
        #    - Overlapping correlation responses create interference patterns
        # 3. Note: Correlation of two IDENTICAL CHIRP signals gives smooth peak (no oscillations)
        #    But here: received = sum of delayed/attenuated copies → oscillations appear
        # 4. Envelope detection removes these oscillations and extracts peak amplitudes
        
        # Calculate envelope using Hilbert transform
        # Envelope removes high-frequency oscillations and extracts the amplitude envelope
        analytic_signal = scipy_signal.hilbert(correlated)
        envelope = np.abs(analytic_signal)
        
        # Calculate lags (delay indices) for correlation, matching user's example:
        # lags = np.arange(-len(reference) + 1, len(profile_signal))
        # This gives correct time alignment: lag=0 means reference[0] aligns with profile_signal[0]
        correlation_lags = np.arange(-len(reference_signal) + 1, len(profile_signal))
        
        return correlated, envelope, correlation_lags
    
    def find_all_echoes_from_correlation(self, correlation: np.ndarray,
                                         profile_depths: np.ndarray,
                                         min_height: float = 0.1,
                                         min_distance: Optional[int] = None,
                                         use_envelope: bool = True) -> Tuple[List[float], List[float]]:
        """
        Find all echoes (interfaces) from correlation peaks using envelope detection.
        
        All distances to echoes are determined by correlation between received signal
        and reference signal. Each peak in correlation indicates a position where
        received signal best matches the reference signal (echo from an interface).
        
        Correlation is computed as: correlation(t) = ∫ profile_signal(τ) * reference(t - τ) dτ
        
        Envelope detection (|Hilbert(correlation)|) removes oscillations and provides
        smooth peaks for better echo detection.
        
        Args:
            correlation: Correlation signal (matched filter output)
            profile_depths: Depth axis corresponding to correlation, m
            min_height: Minimum peak height (relative to maximum, 0.0-1.0)
            min_distance: Minimum distance between peaks (in samples). If None, uses 1% of signal length
            use_envelope: If True, use envelope detection (|Hilbert(correlation)|) for peak detection.
                         If False, use absolute correlation |correlation|
        
        Returns:
            Tuple (echo_depths, echo_amplitudes) - lists of depths and amplitudes for all detected echoes
        """
        if len(correlation) == 0 or len(profile_depths) == 0:
            return [], []
        
        if use_envelope:
            # Use envelope detection: |Hilbert(correlation)|
            # This removes oscillations and provides smooth peaks
            analytic_signal = scipy_signal.hilbert(correlation)
            envelope = np.abs(analytic_signal)
            signal_for_peaks = envelope
        else:
            # Use absolute correlation: |correlation|
            signal_for_peaks = np.abs(correlation)
        
        max_val = np.max(signal_for_peaks)
        
        if max_val <= 0:
            return [], []
        
        # Set minimum distance between peaks if not provided
        if min_distance is None:
            min_distance = max(1, int(len(signal_for_peaks) * 0.01))  # 1% of signal length
        
        # Find all peaks using scipy's find_peaks
        # Height threshold: min_height * max_val (relative to maximum)
        peaks, properties = scipy_signal.find_peaks(
            signal_for_peaks,
            height=min_height * max_val,
            distance=min_distance
        )
        
        # Extract depths and amplitudes for detected peaks
        echo_depths = [profile_depths[p] for p in peaks if p < len(profile_depths)]
        echo_amplitudes = [signal_for_peaks[p] for p in peaks if p < len(signal_for_peaks)]
        
        return echo_depths, echo_amplitudes
    
    def find_first_echo_from_correlation(self, correlation: np.ndarray, 
                                         profile_depths: np.ndarray,
                                         threshold: float = 0.1,
                                         use_envelope: bool = True) -> Optional[float]:
        """
        Find first echo (water-bottom) distance from correlation peak using envelope detection.
        
        This is a convenience method that calls find_all_echoes_from_correlation
        and returns only the first (closest) echo.
        
        Uses envelope detection (|Hilbert(correlation)|) to remove oscillations
        and provide smooth peak for better echo detection.
        
        Args:
            correlation: Correlation signal (matched filter output)
            profile_depths: Depth axis corresponding to correlation, m
            threshold: Minimum correlation threshold (relative to max, 0.0-1.0)
            use_envelope: If True, use envelope detection for peak finding
        
        Returns:
            Distance to first echo (water-bottom depth), m, or None if not found
        """
        echo_depths, _ = self.find_all_echoes_from_correlation(
            correlation, profile_depths, min_height=threshold, use_envelope=use_envelope
        )
        
        if len(echo_depths) > 0:
            return echo_depths[0]  # Return first (closest) echo
        
        return None
    
    def calculate_vertical_resolution(self, bandwidth: float, 
                                     sound_speed: float = 1600.0) -> float:
        """
        Calculate vertical resolution.
        
        Formula: Δz = c / (2*B)
        
        Args:
            bandwidth: CHIRP bandwidth, Hz
            sound_speed: Sound speed in sediment, m/s
        
        Returns:
            Vertical resolution, m
        """
        if bandwidth <= 0:
            return float('inf')
        
        delta_z = sound_speed / (2 * bandwidth)
        return delta_z
    
    def detect_interfaces(self, 
                         profile_envelope: np.ndarray,
                         profile_depths: np.ndarray,
                         min_snr: float = 10.0,
                         noise_level: Optional[float] = None) -> Tuple[List[float], List[float]]:
        """
        Detect interfaces in profile using peak detection.
        
        Args:
            profile_envelope: Profile envelope (amplitude vs depth)
            profile_depths: Depth axis, m
            min_snr: Minimum SNR for detection, dB
            noise_level: Noise level (if None, estimated from signal)
        
        Returns:
            Tuple (detected_depths, detected_amplitudes)
        """
        if len(profile_envelope) == 0:
            return [], []
        
        # Estimate noise level if not provided
        if noise_level is None:
            # Use median of lower 10% of amplitudes as noise estimate
            sorted_amplitudes = np.sort(profile_envelope)
            noise_level = np.median(sorted_amplitudes[:max(1, len(sorted_amplitudes) // 10)])
        
        # Find peaks (local maxima)
        # Use scipy's find_peaks
        peaks, properties = scipy_signal.find_peaks(
            profile_envelope,
            height=noise_level * (10 ** (min_snr / 20)),  # Minimum height in linear scale
            distance=int(len(profile_envelope) * 0.01)  # Minimum distance between peaks (1% of signal length)
        )
        
        # Extract depths and amplitudes for detected peaks
        detected_depths = [profile_depths[p] for p in peaks]
        detected_amplitudes = [profile_envelope[p] for p in peaks]
        
        return detected_depths, detected_amplitudes

