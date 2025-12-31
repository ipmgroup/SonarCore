"""
DTO (Data Transfer Objects) for data transfer between modules.
"""

from typing import Dict, Optional, List, Any
from pydantic import BaseModel, Field, validator


class HardwareDTO(BaseModel):
    """Hardware parameters."""
    transducer_id: str = Field(..., description="Transducer ID")
    lna_id: str = Field(..., description="LNA ID")
    vga_id: str = Field(..., description="VGA ID")
    adc_id: str = Field(..., description="ADC ID")


class SignalDTO(BaseModel):
    """CHIRP signal parameters."""
    f_start: float = Field(..., gt=0, description="Start frequency, Hz")
    f_end: float = Field(..., gt=0, description="End frequency, Hz")
    Tp: float = Field(..., gt=0, description="Pulse duration, µs")
    window: str = Field(default="Hann", description="Window function: Rect/Hann/Tukey")
    sample_rate: float = Field(..., gt=0, description="Sampling frequency for signal generation, Hz")
    
    @validator('window')
    def validate_window(cls, v):
        allowed = ['Rect', 'Hann', 'Tukey']
        if v not in allowed:
            raise ValueError(f"Window must be one of {allowed}")
        return v
    
    @validator('f_end')
    def validate_f_end(cls, v, values):
        if 'f_start' in values and v <= values['f_start']:
            raise ValueError("f_end must be greater than f_start")
        return v
    
    @validator('sample_rate')
    def validate_sample_rate(cls, v, values):
        if 'f_end' in values and v < 2 * values['f_end']:
            raise ValueError(f"sample_rate ({v} Hz) must be >= 2*f_end ({2*values['f_end']} Hz) to satisfy Nyquist criterion")
        return v


class SedimentLayerDTO(BaseModel):
    """Single sediment layer parameters."""
    thickness: float = Field(..., gt=0, description="Layer thickness, m")
    density: float = Field(..., gt=0, description="Density, kg/m³")
    sound_speed: float = Field(..., gt=0, description="Sound speed in layer, m/s")
    attenuation: float = Field(..., ge=0, description="Attenuation coefficient, dB/m")
    name: str = Field(default="", description="Layer name (optional)")


class SedimentProfileDTO(BaseModel):
    """Sediment profile (multi-layer model)."""
    layers: List[SedimentLayerDTO] = Field(default_factory=list, description="List of sediment layers (from top to bottom)")
    water_depth: float = Field(default=0.0, ge=0, description="Water depth above sediment, m")
    
    @validator('layers')
    def validate_layers(cls, v):
        if len(v) == 0:
            raise ValueError("At least one sediment layer is required")
        return v


class EnvironmentDTO(BaseModel):
    """Environment parameters."""
    T: float = Field(..., ge=0, le=30, description="Temperature, °C")
    S: float = Field(..., ge=0, le=35, description="Salinity, PSU")
    z: float = Field(..., ge=0.2, le=300, description="Depth, m")
    bottom_reflection: float = Field(
        default=-15.0, 
        ge=-30, 
        le=0, 
        description="Bottom reflection coefficient, dB. Typical values: Mud -10 to -20 dB, Sand -6 to -12 dB, Gravel -3 to -8 dB, Rock/Concrete -1 to -3 dB"
    )


class RangeDTO(BaseModel):
    """Range parameters."""
    D_min: float = Field(..., gt=0, description="Minimum range, m")
    D_max: float = Field(..., gt=0, description="Maximum range, m")
    D_target: Optional[float] = Field(default=None, gt=0, description="Target range for simulation, m (if None, uses average of D_min and D_max)")
    
    @validator('D_max')
    def validate_range(cls, v, values):
        if 'D_min' in values and v <= values['D_min']:
            raise ValueError("D_max must be greater than D_min")
        return v
    
    @validator('D_target')
    def validate_d_target(cls, v, values):
        if v is not None:
            if 'D_min' in values and v < values['D_min']:
                raise ValueError("D_target must be >= D_min")
            if 'D_max' in values and v > values['D_max']:
                raise ValueError("D_target must be <= D_max")
        return v


class InputDTO(BaseModel):
    """Input DTO for simulation."""
    hardware: HardwareDTO
    signal: SignalDTO
    environment: EnvironmentDTO
    range: RangeDTO
    target_snr: Optional[float] = Field(default=20.0, ge=10, le=40, description="Target SNR at ADC output, dB (range: 10-40)")
    optimization_strategy: Optional[str] = Field(default="max_tp_min_vga", description="Optimization strategy: 'max_tp_min_vga' (use maximum Tp and find minimum VGA Gain) or 'min_tp_for_snr' (find minimum Tp for target SNR)")
    limit_tp_for_fast_calculation: Optional[bool] = Field(default=False, description="Limit Tp to 1 second for fast calculation when Tp > 1s")
    # Sub-bottom profiling parameters
    sediment_profile: Optional[SedimentProfileDTO] = Field(default=None, description="Sediment profile for SBP (if None, uses simple bottom reflection)")
    enable_sbp: Optional[bool] = Field(default=False, description="Enable sub-bottom profiling mode")
    snr_pre: Optional[float] = Field(default=-5.0, ge=-20, le=10, description="Pre-correlation SNR for SBP (typical: -5 to 0 dB for sediment), dB")


class RecommendationsDTO(BaseModel):
    """Optimization recommendations."""
    increase_Tp: bool = False
    decrease_Tp: bool = False
    aggressive_Tp_increase: bool = False  # Flag for more aggressive Tp increase when VGA is at max
    increase_f_start: bool = False
    decrease_f_start: bool = False
    increase_f_end: bool = False
    decrease_f_end: bool = False
    increase_G_VGA: bool = False
    decrease_G_VGA: bool = False
    suggested_vga_gain: Optional[float] = Field(default=None, description="Suggested VGA gain value, dB")
    warn_unachievable_snr: bool = False  # Warning flag if target SNR may be unachievable
    change_transducer: bool = False
    change_lna: bool = False
    change_adc: bool = False
    message: str = ""
    suggested_changes: Dict = Field(default_factory=dict, description="Suggested parameter values (Tp, f_start, f_end)")


class OutputDTO(BaseModel):
    """Output DTO with simulation results."""
    D_measured: float = Field(..., description="Measured range, m")
    sigma_D: float = Field(..., description="Range standard deviation, m")
    SNR_ADC: float = Field(..., description="SNR at ADC output, dB")
    clipping_flags: bool = Field(default=False, description="ADC clipping flag")
    success: bool = Field(default=True, description="Simulation success")
    recommendations: RecommendationsDTO = Field(default_factory=RecommendationsDTO)
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    errors: List[str] = Field(default_factory=list, description="Errors")
    # Receiver gain values
    lna_gain: Optional[float] = Field(default=None, description="LNA gain used in simulation, dB")
    vga_gain: Optional[float] = Field(default=None, description="VGA gain used in simulation, dB")
    vga_gain_max: Optional[float] = Field(default=None, description="VGA maximum gain, dB")
    # ADC parameters
    adc_full_scale: Optional[float] = Field(default=None, description="ADC full scale voltage (V_FS), V")
    adc_range: Optional[float] = Field(default=None, description="ADC input range (±V_FS/2), V")
    adc_bits: Optional[int] = Field(default=None, description="ADC resolution (bits)")
    adc_dynamic_range: Optional[float] = Field(default=None, description="ADC dynamic range, dB")
    # Signal data for visualization (stored as lists for JSON serialization)
    tx_signal: Optional[List[float]] = Field(default=None, description="Transmitted signal (after TX transducer)")
    signal_at_bottom: Optional[List[float]] = Field(default=None, description="Signal at bottom (after channel, before reflection)")
    received_signal: Optional[List[float]] = Field(default=None, description="Received signal (after RX sensitivity, before receiver)")
    signal_after_lna: Optional[List[float]] = Field(default=None, description="Signal after LNA (before VGA)")
    signal_after_vga: Optional[List[float]] = Field(default=None, description="Signal after VGA (before ADC)")
    time_axis: Optional[List[float]] = Field(default=None, description="Time axis for signals, seconds")
    # Signal attenuation values (calculated in Core for visualization)
    attenuation_at_bottom_db: Optional[float] = Field(default=None, description="Signal attenuation at bottom relative to TX, dB (calculated in Core)")
    attenuation_received_db: Optional[float] = Field(default=None, description="Signal attenuation at receiver relative to TX, dB (calculated in Core)")
    # ENOB calculation results (calculated AFTER optimization with recommended parameters)
    enob_results: Optional[Dict[str, Any]] = Field(default=None, description="ENOB calculation results dictionary")
    # Sub-bottom profiling results
    profile_amplitudes: Optional[List[float]] = Field(default=None, description="Sub-bottom profile amplitudes vs depth (envelope after matched filter), V")
    profile_depths: Optional[List[float]] = Field(default=None, description="Sub-bottom profile depths, m")
    profile_time_axis: Optional[List[float]] = Field(default=None, description="Sub-bottom profile time axis, s")
    profile_raw_signal: Optional[List[float]] = Field(default=None, description="Raw sub-bottom profile signal (before matched filter), V")
    profile_correlation: Optional[List[float]] = Field(default=None, description="Sub-bottom profile correlation signal (matched filter output), V")
    profile_correlation_depths: Optional[List[float]] = Field(default=None, description="Depth axis for correlation signal (matches correlation length), m")
    profile_water_depth: Optional[float] = Field(default=None, description="Water depth used for profile generation, m")
    interface_depths: Optional[List[float]] = Field(default=None, description="Detected interface depths, m")
    interface_amplitudes: Optional[List[float]] = Field(default=None, description="Detected interface reflection amplitudes, V")
    max_penetration_depth: Optional[float] = Field(default=None, description="Maximum penetration depth in sediment, m")
    vertical_resolution: Optional[float] = Field(default=None, description="Vertical resolution (minimum distinguishable layer thickness), m")
    # Signal path data (from SignalPathCalculator)
    signal_path: Optional[Dict[str, Any]] = Field(default=None, description="Complete signal path data with all losses (from SignalPathCalculator)")

