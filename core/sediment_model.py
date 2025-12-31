"""
SedimentModel - model for sub-bottom sediment layers.

Implements multi-layer sediment model with:
- Acoustic impedance calculation
- Reflection coefficient calculation
- Attenuation in sediment layers
- Sub-bottom profile formation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class SedimentLayer:
    """Single sediment layer."""
    
    def __init__(self, thickness: float, density: float, sound_speed: float, 
                 attenuation: float, name: str = "", attenuation_exponent: float = 1.2):
        """
        Initialize sediment layer.
        
        Args:
            thickness: Layer thickness, m
            density: Density, kg/m³
            sound_speed: Sound speed in layer, m/s
            attenuation: Attenuation coefficient at reference frequency (typically 1 kHz), dB/m
            name: Layer name (optional)
            attenuation_exponent: Exponent n in frequency-dependent attenuation α(f) = α_0 * f^n (default: 1.2)
        """
        self.thickness = thickness  # m
        self.density = density  # kg/m³
        self.sound_speed = sound_speed  # m/s
        self.attenuation_ref = attenuation  # dB/m at reference frequency (typically 1 kHz)
        self.attenuation_exponent = attenuation_exponent  # n in α(f) = α_0 * f^n
        self.name = name
        
        # For backward compatibility: keep old attribute name
        # It will return attenuation at reference frequency (1 kHz)
        self.attenuation = attenuation  # dB/m at reference frequency
        
        # Calculate acoustic impedance: Z = ρ * c
        self.impedance = density * sound_speed  # Pa·s/m (kg/(m²·s))
    
    def get_attenuation(self, frequency: float, ref_frequency: float = 1000.0) -> float:
        """
        Calculate frequency-dependent attenuation coefficient.
        
        Formula: α(f) = α_0 * (f/f_ref)^n
        
        According to TZ (SBP.md, line 317): α(f) = α_0 * f^n
        where n ≈ 1-1.5 for most sediments.
        
        Args:
            frequency: Frequency, Hz
            ref_frequency: Reference frequency for attenuation_ref, Hz (default: 1 kHz)
        
        Returns:
            Attenuation coefficient at given frequency, dB/m
        """
        if frequency <= 0:
            return 0.0
        
        # Normalize to reference frequency
        freq_ratio = frequency / ref_frequency
        
        # α(f) = α_0 * (f/f_ref)^n
        attenuation_at_freq = self.attenuation_ref * (freq_ratio ** self.attenuation_exponent)
        
        return attenuation_at_freq
    
    def __repr__(self):
        return f"SedimentLayer(name={self.name}, z={self.thickness}m, ρ={self.density}kg/m³, c={self.sound_speed}m/s, α={self.attenuation}dB/m)"


class SedimentModel:
    """
    Multi-layer sediment model for sub-bottom profiling.
    
    Models sediment as sequence of horizontal layers with different
    acoustic properties (density, sound speed, attenuation).
    """
    
    def __init__(self, layers: List[SedimentLayer]):
        """
        Initialize sediment model.
        
        Args:
            layers: List of sediment layers (from top to bottom)
        """
        self.layers = layers
        self._validate_layers()
    
    def _validate_layers(self):
        """Validate layer parameters."""
        for i, layer in enumerate(self.layers):
            if layer.thickness <= 0:
                raise ValueError(f"Layer {i} thickness must be > 0")
            if layer.density <= 0:
                raise ValueError(f"Layer {i} density must be > 0")
            if layer.sound_speed <= 0:
                raise ValueError(f"Layer {i} sound_speed must be > 0")
            if layer.attenuation < 0:
                raise ValueError(f"Layer {i} attenuation must be >= 0")
    
    def get_num_interfaces(self) -> int:
        """
        Get number of interfaces (boundaries between layers).
        
        Returns:
            Number of interfaces
        """
        return len(self.layers)
    
    def get_interface_depth(self, interface_idx: int) -> float:
        """
        Get depth of interface in sediment (cumulative depth from seafloor).
        
        Args:
            interface_idx: Interface index 
                - 0 = water-bottom interface (seafloor, depth = 0)
                - 1 = interface between layer 1 and layer 2 (depth = thickness of layer 1)
                - 2 = interface between layer 2 and layer 3 (depth = thickness of layer 1 + layer 2)
                - etc.
        
        Returns:
            Depth of interface in sediment from seafloor, m
        """
        if interface_idx < 0 or interface_idx >= len(self.layers):
            raise ValueError(f"Interface index {interface_idx} out of range [0, {len(self.layers)-1}]")
        
        # Interface 0 is water-bottom (seafloor), depth = 0 in sediment
        if interface_idx == 0:
            return 0.0
        
        # For interfaces > 0, sum thicknesses of layers up to (interface_idx - 1)
        # Interface i is between layers i-1 and i, so depth = sum of layers 0 to i-1
        depth = 0.0
        for i in range(interface_idx):
            depth += self.layers[i].thickness
        
        return depth
    
    def get_reflection_coefficient(self, interface_idx: int, 
                                  water_impedance: float = 1.5e6) -> Tuple[float, float]:
        """
        Calculate reflection coefficient at interface.
        
        Formula: R = (Z2 - Z1) / (Z2 + Z1)
        
        Args:
            interface_idx: Interface index (0 = water-sediment, 1 = layer1-layer2, etc.)
            water_impedance: Acoustic impedance of water, Pa·s/m (default: 1.5e6 for seawater)
        
        Returns:
            Tuple (R_linear, R_dB) - reflection coefficient in linear and dB scale
        """
        if interface_idx < 0 or interface_idx >= len(self.layers):
            raise ValueError(f"Interface index {interface_idx} out of range")
        
        # First interface: water -> first layer
        if interface_idx == 0:
            Z1 = water_impedance
            Z2 = self.layers[0].impedance
        else:
            # Interface between layers
            Z1 = self.layers[interface_idx - 1].impedance
            Z2 = self.layers[interface_idx].impedance
        
        # Reflection coefficient: R = (Z2 - Z1) / (Z2 + Z1)
        R_linear = (Z2 - Z1) / (Z2 + Z1)
        
        # In dB: R_dB = 20 * log10(|R|)
        R_dB = 20 * np.log10(abs(R_linear)) if abs(R_linear) > 1e-10 else -np.inf
        
        return R_linear, R_dB
    
    def get_transmission_coefficient(self, interface_idx: int,
                                    water_impedance: float = 1.5e6) -> Tuple[float, float]:
        """
        Calculate transmission coefficient at interface.
        
        Formula: T = 1 + R = 2*Z2 / (Z2 + Z1)
        
        Args:
            interface_idx: Interface index
            water_impedance: Acoustic impedance of water
        
        Returns:
            Tuple (T_linear, T_dB) - transmission coefficient
        """
        R_linear, _ = self.get_reflection_coefficient(interface_idx, water_impedance)
        T_linear = 1 + R_linear  # T = 1 + R
        
        T_dB = 20 * np.log10(abs(T_linear)) if abs(T_linear) > 1e-10 else -np.inf
        
        return T_linear, T_dB
    
    def calculate_attenuation_to_interface(self, interface_idx: int, frequency: float = None) -> float:
        """
        Calculate total attenuation (in dB) in sediment from bottom to interface (round-trip).
        
        Formula: TL = 2 * sum(α_i(f) * z_i) for round-trip in sediment
        
        This calculates attenuation ONLY in sediment layers, not in water.
        Water attenuation must be added separately.
        
        Args:
            interface_idx: Interface index (0 = bottom, 1 = first layer boundary, etc.)
            frequency: Frequency for frequency-dependent attenuation, Hz.
                      If None, uses reference frequency (backward compatibility)
        
        Returns:
            Total attenuation in sediment, dB (round-trip from bottom to interface)
        """
        if interface_idx < 0 or interface_idx >= len(self.layers):
            raise ValueError(f"Interface index {interface_idx} out of range")
        
        total_attenuation = 0.0
        
        # Interface 0 is water-bottom (no sediment layers to traverse)
        if interface_idx == 0:
            return 0.0
        
        # Sum attenuation through sediment layers up to (but not including) interface
        # Path: bottom -> interface -> bottom (round-trip in sediment)
        # Interface i is at the boundary between layers i-1 and i
        # So we sum attenuation only through layers 0 to i-1 (not including layer i)
        # Example: Interface 1 (between layers 0 and 1) -> sum only layer 0
        #          Interface 2 (between layers 1 and 2) -> sum layers 0 and 1
        for i in range(interface_idx):
            layer = self.layers[i]
            # Get frequency-dependent attenuation
            if frequency is not None:
                layer_alpha = layer.get_attenuation(frequency)
            else:
                # Backward compatibility: use reference attenuation
                layer_alpha = layer.attenuation_ref
            # Round-trip: signal goes down and back up through layer
            # Attenuation: 2 * α(f) * z (dB)
            total_attenuation += 2 * layer_alpha * layer.thickness
        
        return total_attenuation
    
    def get_time_of_flight_to_interface(self, interface_idx: int,
                                       water_depth: float = 0.0,
                                       water_sound_speed: float = 1500.0) -> float:
        """
        Calculate time of flight (TOF) to interface.
        
        Formula: TOF = 2 * (sum(z_i / c_i) + water_depth / c_water)
        
        Args:
            interface_idx: Interface index
            water_depth: Depth of water column, m
            water_sound_speed: Sound speed in water, m/s
        
        Returns:
            Time of flight (round-trip), s
        """
        if interface_idx < 0 or interface_idx >= len(self.layers):
            raise ValueError(f"Interface index {interface_idx} out of range")
        
        # Time in water (round-trip)
        tof_water = 2 * water_depth / water_sound_speed if water_depth > 0 else 0.0
        
        # Time in sediment layers (round-trip)
        # For interface 0 (water-bottom), no sediment time (interface is at water-bottom boundary)
        # For interface i > 0, add time through layers up to interface i-1 (interface is between layers i-1 and i)
        tof_sediment = 0.0
        if interface_idx == 0:
            # Interface 0 is water-bottom boundary, no sediment time
            tof_sediment = 0.0
        else:
            # Interface i > 0: time through layers up to layer i-1 (interface is between layers i-1 and i)
            for i in range(interface_idx):
                layer = self.layers[i]
                # One-way time: z / c
                # Round-trip: 2 * z / c
                tof_sediment += 2 * layer.thickness / layer.sound_speed
        
        total_tof = tof_water + tof_sediment
        
        return total_tof
    
    def get_max_penetration_depth(self, source_level: float, noise_level: float,
                                 processing_gain: float, water_tl: float,
                                 min_snr: float = 10.0) -> float:
        """
        Estimate maximum penetration depth based on sonar equation.
        
        Formula: z_max = (SL - TL_water + R - NL + PG - SNR_min) / (2*α)
        
        Args:
            source_level: Source level, dB re 1µPa @ 1m
            noise_level: Noise level, dB
            processing_gain: Processing gain (from matched filter), dB
            water_tl: Transmission loss in water (to bottom), dB
            min_snr: Minimum required SNR, dB
        
        Returns:
            Maximum penetration depth, m
        """
        # Use average attenuation at reference frequency of all layers
        avg_attenuation = np.mean([layer.attenuation_ref for layer in self.layers])
        
        if avg_attenuation <= 0:
            return float('inf')  # No attenuation limit
        
        # Use average reflection coefficient
        avg_reflection = np.mean([abs(self.get_reflection_coefficient(i)[0]) 
                                 for i in range(len(self.layers))])
        R_dB = 20 * np.log10(avg_reflection) if avg_reflection > 0 else -20.0
        
        # Sonar equation for SBP:
        # SNR(z) = SL - TL_water - TL_sediment(z) + R - NL + PG
        # At maximum depth: SNR(z_max) = SNR_min
        # Solving for z_max:
        # z_max = (SL - TL_water + R - NL + PG - SNR_min) / (2*α)
        
        numerator = source_level - water_tl + R_dB - noise_level + processing_gain - min_snr
        z_max = numerator / (2 * avg_attenuation)
        
        return max(0.0, z_max)
    
    @staticmethod
    def create_typical_profile(profile_type: str = "clay_silt_sand") -> 'SedimentModel':
        """
        Create typical sediment profile.
        
        Args:
            profile_type: Profile type ("clay_silt_sand", "sand_gravel", etc.)
        
        Returns:
            SedimentModel instance
        """
        if profile_type == "clay_silt_sand":
            layers = [
                SedimentLayer(2.0, 1600, 1550, 1.0, "Clay"),
                SedimentLayer(3.0, 1800, 1600, 2.0, "Silt"),
                SedimentLayer(5.0, 2000, 1700, 3.0, "Sand"),
            ]
        elif profile_type == "sand_gravel":
            layers = [
                SedimentLayer(1.0, 1900, 1650, 2.5, "Sand"),
                SedimentLayer(4.0, 2100, 1900, 4.0, "Gravel"),
            ]
        elif profile_type == "mud_clay":
            layers = [
                SedimentLayer(3.0, 1400, 1500, 0.8, "Mud"),
                SedimentLayer(7.0, 1600, 1550, 1.2, "Clay"),
            ]
        else:
            # Default: single layer
            layers = [
                SedimentLayer(10.0, 1800, 1600, 2.0, "Sediment"),
            ]
        
        return SedimentModel(layers)

