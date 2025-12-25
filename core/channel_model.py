"""
ChannelModel - hydroacoustic channel modeling.
"""

import numpy as np
from .water_model import WaterModel


class ChannelModel:
    """
    Hydroacoustic channel model.
    
    Accounts for:
    - Transmission Loss
    - Water attenuation
    - Geometric spreading
    """
    
    def __init__(self, water_model: WaterModel):
        """
        Initialize channel model.
        
        Args:
            water_model: WaterModel instance
        """
        self.water_model = water_model
    
    def calculate_transmission_loss(self, D: float, f: float, 
                                   T: float, S: float, z: float) -> float:
        """
        Calculates transmission loss.
        
        Args:
            D: Range, m
            f: Frequency, Hz
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
        
        Returns:
            Transmission loss, dB
        """
        return self.water_model.calculate_transmission_loss(D, f, T, S, z)
    
    def apply_channel_response(self, signal: np.ndarray, D: float, 
                              f_center: float, T: float, S: float, z: float,
                              fs: float, round_trip: bool = True) -> np.ndarray:
        """
        Applies channel response to signal.
        
        Args:
            signal: Input signal
            D: Range, m (one-way)
            f_center: Central frequency, Hz
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
            fs: Sampling frequency, Hz
            round_trip: If True, applies attenuation for round trip (2*D),
                       if False, only forward (D)
        
        Returns:
            Signal after passing through channel
        """
        # Calculate transmission loss
        # If round_trip=True, attenuation for 2*D (round trip)
        # If round_trip=False, attenuation only for D (forward)
        distance_for_loss = 2 * D if round_trip else D
        TL = self.calculate_transmission_loss(distance_for_loss, f_center, T, S, z)
        
        # Convert dB to linear scale
        attenuation_linear = 10 ** (-TL / 20)
        
        # Apply attenuation
        signal_attenuated = signal * attenuation_linear
        
        # Propagation delay (TOF)
        c = self.water_model.calculate_sound_speed(T, S, 
                                                   self.water_model.calculate_pressure(z))
        # TOF depends on round_trip
        # D is one-way distance, so for round_trip: TOF = 2*D/c
        tof = (2 * D / c) if round_trip else (D / c)
        delay_samples = int(tof * fs)
        
        # Log for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"ChannelModel.apply_channel_response: D={D:.2f}m (one-way), round_trip={round_trip}, "
                   f"tof={tof*1e3:.3f}ms, c={c:.2f}m/s, delay_samples={delay_samples}, fs={fs:.0f}Hz, "
                   f"expected_round_trip_distance={2*D:.2f}m if round_trip=True")
        
        # Add delay (create array with delay)
        if delay_samples > 0:
            signal_delayed = np.zeros(len(signal) + delay_samples)
            signal_delayed[delay_samples:] = signal_attenuated
        else:
            signal_delayed = signal_attenuated
        
        return signal_delayed
    
    def apply_attenuation_only(self, signal: np.ndarray, D: float, 
                               f_center: float, T: float, S: float, z: float,
                               round_trip: bool = False,
                               f_start: float = None, f_end: float = None,
                               t: np.ndarray = None, Tp: float = None,
                               absorption_only: bool = False) -> np.ndarray:
        """
        Applies only attenuation (without delay) to signal.
        
        For CHIRP signals, applies frequency-dependent attenuation.
        For constant frequency signals, uses f_center.
        
        Used for visualization where delay is not needed.
        
        Args:
            signal: Input signal
            D: Distance, m (one-way)
            f_center: Central frequency, Hz (used if f_start/f_end not provided)
            T: Temperature, °C
            S: Salinity, PSU
            z: Depth, m
            round_trip: If True, applies attenuation for round trip (2*D),
                       if False, only forward (D)
            f_start: Start frequency for CHIRP, Hz (optional)
            f_end: End frequency for CHIRP, Hz (optional)
            t: Time axis, s (optional, required for CHIRP)
            Tp: Pulse duration, µs (optional, required for CHIRP)
            absorption_only: If True, apply only absorption (without spreading) for visualization
        
        Returns:
            Signal after attenuation (same length as input)
        """
        distance_for_loss = 2 * D if round_trip else D
        
        # Spreading loss depends ONLY on distance, NOT on frequency
        # Formula: Spreading Loss = 20 * log10(D)
        # This is geometric spreading (spherical wave propagation)
        # For CHIRP signals, spreading is the same for all frequencies
        if absorption_only:
            # For visualization: apply only absorption (spreading is constant anyway)
            spreading_loss = 0.0  # No spreading loss for visualization
            spreading_attenuation = 1.0  # No spreading attenuation
        else:
            spreading_loss = self.water_model.calculate_spreading_loss(distance_for_loss)
            spreading_attenuation = 10 ** (-spreading_loss / 20)
        
        # Check if this is a CHIRP signal (frequency-dependent absorption)
        if f_start is not None and f_end is not None and t is not None and Tp is not None:
            # CHIRP signal: apply frequency-dependent absorption
            from .signal_model import SignalModel
            
            # Calculate instantaneous frequency for each time sample
            f_inst = SignalModel.get_instantaneous_frequency(t, f_start, f_end, Tp)
            
            # Pre-calculate pressure (same for all frequencies, depends only on z)
            # This avoids recalculating it in the loop inside calculate_absorption_loss
            P = self.water_model.calculate_pressure(z)
            
            # Calculate absorption loss for each frequency (absorption depends on frequency)
            # Vectorized calculation
            # Note: calculate_absorption_loss will still calculate pressure internally,
            # but we could optimize further by calling calculate_attenuation directly
            absorption_loss_array = np.array([self.water_model.calculate_absorption_loss(
                distance_for_loss, f, T, S, z) for f in f_inst])
            
            # Convert absorption loss to linear scale for all frequencies
            absorption_attenuation = 10 ** (-absorption_loss_array / 20)
            
            # Debug: print absorption info for first and last frequency
            if len(f_inst) > 0:
                print(f"DEBUG Absorption: f_start={f_inst[0]/1000:.1f} kHz, abs_loss={absorption_loss_array[0]:.6f} dB, atten={absorption_attenuation[0]:.6f}")
                print(f"DEBUG Absorption: f_end={f_inst[-1]/1000:.1f} kHz, abs_loss={absorption_loss_array[-1]:.6f} dB, atten={absorption_attenuation[-1]:.6f}")
                print(f"DEBUG Spreading: loss={spreading_loss:.2f} dB, atten={spreading_attenuation:.6f}")
                print(f"DEBUG Total atten ratio (high/low): {absorption_attenuation[-1]/absorption_attenuation[0]:.6f}")
            
            # Total attenuation = spreading (constant) * absorption (frequency-dependent)
            total_attenuation = spreading_attenuation * absorption_attenuation
            
            # Apply frequency-dependent attenuation
            signal_attenuated = signal * total_attenuation
        else:
            # Constant frequency signal: use f_center
            absorption_loss = self.water_model.calculate_absorption_loss(
                distance_for_loss, f_center, T, S, z)
            absorption_attenuation = 10 ** (-absorption_loss / 20)
            
            # Total attenuation = spreading * absorption
            total_attenuation = spreading_attenuation * absorption_attenuation
            signal_attenuated = signal * total_attenuation
        
        return signal_attenuated

