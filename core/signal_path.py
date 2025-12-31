"""
SignalPathCalculator - calculation of signal path through the system.
"""

import numpy as np
from typing import Dict, Optional
from .water_model import WaterModel
from .dto import InputDTO


class SignalPathCalculator:
    """
    Signal path calculator.
    
    Calculates signal parameters at each stage:
    - Transmitter (TX)
    - Water (forward)
    - Bottom
    - Water (backward)
    - Receiver (RX)
    """
    
    def __init__(self, data_provider=None):
        """
        Initialize calculator.
        
        Args:
            data_provider: Data provider for hardware parameters
        """
        self.water_model = WaterModel()
        self.data_provider = data_provider
    
    def calculate_signal_path(self, input_dto: InputDTO,
                            tx_voltage: Optional[float] = None, 
                            lna_gain: Optional[float] = None,
                            lna_nf: Optional[float] = None,
                            vga_gain: Optional[float] = None) -> Dict:
        """
        Calculates signal parameters at each path stage.
        
        Args:
            input_dto: InputDTO with simulation parameters
            tx_voltage: Transmitter voltage, V (if None, taken from transducer data or default 100 V)
            lna_gain: LNA gain, dB (if None, taken from data)
            lna_nf: LNA noise figure, dB (if None, taken from data)
            vga_gain: VGA gain, dB (if None, taken from data)
        
        Returns:
            Dictionary with parameters for each stage
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Get hardware parameters from data_provider
            if self.data_provider is None:
                raise ValueError("data_provider is not set. Use constructor with data_provider.")
            
            logger.debug(f"Loading transducer: {input_dto.hardware.transducer_id}")
            transducer_params = self.data_provider.get_transducer(input_dto.hardware.transducer_id)
            logger.debug(f"Transducer loaded: {list(transducer_params.keys())}")
            
            logger.debug(f"Loading LNA: {input_dto.hardware.lna_id}")
            lna_params = self.data_provider.get_lna(input_dto.hardware.lna_id)
            logger.debug(f"LNA loaded: {list(lna_params.keys())}")
            
            logger.debug(f"Loading VGA: {input_dto.hardware.vga_id}")
            vga_params = self.data_provider.get_vga(input_dto.hardware.vga_id)
            logger.debug(f"VGA loaded: {list(vga_params.keys())}")
            
            logger.debug(f"Loading ADC: {input_dto.hardware.adc_id}")
            adc_params = self.data_provider.get_adc(input_dto.hardware.adc_id)
            logger.debug(f"ADC loaded: {list(adc_params.keys())}")
        except Exception as e:
            logger.error(f"Error loading hardware parameters: {e}", exc_info=True)
            raise
        
        # Use target range for signal path calculation
        # If D_target is specified, use it; otherwise use average of D_min and D_max
        # Path: TX -> water (D_target) -> bottom -> water (D_target) -> RX
        # D_target is one-way distance to target, signal travels this distance twice
        # This matches the distance used in signal visualization graphs
        # IMPORTANT: z (depth) is NOT added to D_target - they are separate parameters
        # z is used only for pressure calculation (affects sound speed and attenuation)
        # D_target is the horizontal distance to target
        if input_dto.range.D_target is not None:
            D_target = input_dto.range.D_target
        else:
            D_target = (input_dto.range.D_min + input_dto.range.D_max) / 2
        D_one_way = D_target  # One-way distance to target (don't divide by 2!)
        
        # Central frequency
        f_center = (input_dto.signal.f_start + input_dto.signal.f_end) / 2
        
        # Environment parameters
        T = input_dto.environment.T
        S = input_dto.environment.S
        z = input_dto.environment.z
        P = self.water_model.calculate_pressure(z)
        
        # Log for debugging - ensure z and D_target are NOT summed
        # IMPORTANT: z (depth) is NOT added to D_target - they are separate parameters
        # z is used only for pressure calculation (affects sound speed and attenuation)
        # D_target is the horizontal distance to target
        logger.debug(f"SignalPathCalculator: D_target={D_target:.2f}m, z={z:.2f}m, "
                    f"D_one_way={D_one_way:.2f}m (z and D_target are NOT summed)")
        
        # === STAGE 1: TRANSMITTER (TX) ===
        # Transmitter parameters from transducer data
        S_TX = transducer_params.get('S_TX', 170)  # dB re 1µPa/V @ 1m
        Z_tx = transducer_params.get('Z', 100)  # Impedance, Ohms
        
        # Voltage: always take from transducer data if not explicitly specified
        # Transducer data may have V_max (maximum) or V_nominal (nominal)
        if tx_voltage is None:
            # First try to get from transducer data
            tx_voltage = transducer_params.get('V_max')
            if tx_voltage is None:
                tx_voltage = transducer_params.get('V_nominal')
            if tx_voltage is None:
                tx_voltage = transducer_params.get('V')
            # If not in data, use default 100 V
            if tx_voltage is None:
                tx_voltage = 100.0
        
        # Transmit power (approximately: P = V^2 / Z)
        tx_power_watts = (tx_voltage ** 2) / Z_tx
        tx_power_dbm = 10 * np.log10(tx_power_watts * 1000)  # dBm
        
        # Sound pressure level (SPL)
        # S_TX already accounts for voltage to pressure conversion
        tx_spl_db = S_TX + 20 * np.log10(tx_voltage)  # dB re 1µPa @ 1m
        
        tx_stage = {
            'voltage': tx_voltage,
            'voltage_unit': 'V',
            'power': tx_power_watts,
            'power_dbm': tx_power_dbm,
            'spl': tx_spl_db,
            'spl_unit': 'dB re 1µPa @ 1m',
            'sensitivity': S_TX,
            'sensitivity_unit': 'dB re 1µPa/V @ 1m',
            'impedance': Z_tx,
            'impedance_unit': 'Ω'
        }
        
        # === STAGE 2: WATER (FORWARD) ===
        # Split losses into two components: spreading and absorption
        spreading_forward = self.water_model.calculate_spreading_loss(D_one_way)
        absorption_forward = self.water_model.calculate_absorption_loss(D_one_way, f_center, T, S, z)
        TL_forward = spreading_forward + absorption_forward
        
        # Attenuation coefficient
        alpha = self.water_model.calculate_attenuation(f_center, T, S, P)
        
        water_forward = {
            'distance': D_one_way,
            'distance_unit': 'm',
            'temperature': T,
            'temperature_unit': '°C',
            'salinity': S,
            'salinity_unit': 'PSU',
            'depth': z,
            'depth_unit': 'm',
            'spreading_loss': spreading_forward,
            'spreading_loss_unit': 'dB',
            'absorption_loss': absorption_forward,
            'absorption_loss_unit': 'dB',
            'total_attenuation': TL_forward,
            'total_attenuation_unit': 'dB',
            'attenuation_coeff': alpha,
            'attenuation_coeff_unit': 'dB/m',
            'frequency': f_center,
            'frequency_unit': 'Hz'
        }
        
        # === STAGE 3: BOTTOM ===
        # Reflection coefficient from environment parameters
        # Typical values: -10...-20 dB, depends on bottom type
        bottom_reflection_db = input_dto.environment.bottom_reflection
        
        bottom_stage = {
            'reflection_loss': bottom_reflection_db,
            'reflection_loss_unit': 'dB',
            'description': 'Bottom reflection coefficient'
        }
        
        # === STAGE 4: WATER (BACKWARD) ===
        # Backward losses are same as forward (spreading and absorption calculated separately)
        spreading_backward = spreading_forward  # Same distance
        absorption_backward = absorption_forward  # Same conditions
        TL_backward = spreading_backward + absorption_backward
        
        water_backward = {
            'distance': D_one_way,
            'distance_unit': 'm',
            'temperature': T,
            'temperature_unit': '°C',
            'salinity': S,
            'salinity_unit': 'PSU',
            'depth': z,
            'depth_unit': 'm',
            'spreading_loss': spreading_backward,
            'spreading_loss_unit': 'dB',
            'absorption_loss': absorption_backward,
            'absorption_loss_unit': 'dB',
            'total_attenuation': TL_backward,
            'total_attenuation_unit': 'dB',
            'attenuation_coeff': alpha,
            'attenuation_coeff_unit': 'dB/m',
            'frequency': f_center,
            'frequency_unit': 'Hz'
        }
        
        # === STAGE 5: RECEIVER (RX) ===
        S_RX = transducer_params.get('S_RX', -193)  # dB re 1V/µPa
        
        # LNA parameters
        if lna_gain is not None:
            lna_gain_value = lna_gain
        else:
            # Try G_LNA first (used by ReceiverModel), then G, then default
            lna_gain_value = lna_params.get('G_LNA', lna_params.get('G', 20))  # dB
        
        if lna_nf is not None:
            lna_nf_value = lna_nf
        else:
            # Try NF_LNA first (used by ReceiverModel), then NF, then default
            lna_nf_value = lna_params.get('NF_LNA', lna_params.get('NF', 2))  # dB (noise figure)
        
        # VGA parameters
        if vga_gain is not None:
            vga_gain_value = vga_gain
        else:
            vga_gain_value = vga_params.get('G', 30)  # dB
        vga_gain_min = vga_params.get('G_min', 0)
        vga_gain_max = vga_params.get('G_max', 60)
        
        # ADC parameters
        adc_bits = adc_params.get('bits', 16)
        adc_fs = adc_params.get('f_s', 1000000)
        adc_vref = adc_params.get('V_ref', 2.5)  # V
        
        # ADC dynamic range
        adc_dynamic_range_db = 20 * np.log10(2 ** adc_bits)  # dB
        
        rx_stage = {
            'transducer_sensitivity': S_RX,
            'transducer_sensitivity_unit': 'dB re 1V/µPa',
            'lna_gain': lna_gain_value,
            'lna_gain_unit': 'dB',
            'lna_noise_factor': lna_nf_value,
            'lna_noise_factor_unit': 'dB',
            'vga_gain': vga_gain_value,
            'vga_gain_unit': 'dB',
            'vga_gain_range': f'{vga_gain_min}-{vga_gain_max}',
            'vga_gain_range_unit': 'dB',
            'adc_bits': adc_bits,
            'adc_bits_unit': 'bits',
            'adc_sample_rate': adc_fs,
            'adc_sample_rate_unit': 'Hz',
            'adc_vref': adc_vref,
            'adc_vref_unit': 'V',
            'adc_dynamic_range': adc_dynamic_range_db,
            'adc_dynamic_range_unit': 'dB'
        }
        
        # === TOTAL SIGNAL LEVEL ===
        # All calculations are done in core - GUI only receives ready data
        # 
        # Path structure: TX -> water (forward, one-way) -> bottom -> water (backward, one-way) -> RX
        # 
        # Losses breakdown:
        # - water_forward_loss: Losses for forward path (TX -> bottom), one-way, includes spreading + absorption
        # - bottom_loss: Reflection loss at bottom (from environment parameters)
        # - water_backward_loss: Losses for backward path (bottom -> RX), one-way, includes spreading + absorption
        # - total_path_loss: Sum of all losses (forward + bottom + backward) for complete round-trip
        #
        # Note: Each water path (forward/backward) is calculated separately for one-way distance D_one_way
        # Total round-trip distance = 2 * D_one_way, but losses are calculated separately for each direction
        
        # Calculate total path loss (all losses combined)
        # This is the complete round-trip loss: forward + bottom + backward
        # All losses are positive values (they reduce signal), so we sum them
        # bottom_reflection_db is already negative (e.g., -15 dB), so we need to make it positive for sum
        total_path_loss = TL_forward + abs(bottom_reflection_db) + TL_backward
        
        # Total receiver gain
        total_rx_gain = S_RX + lna_gain_value + vga_gain_value
        
        # Final signal level at ADC input
        signal_at_adc_db = tx_spl_db - total_path_loss + total_rx_gain
        
        # Return structured data: all calculations done in core
        # GUI receives ready data - no calculations in GUI
        result = {
            'tx': tx_stage,
            'water_forward': water_forward,  # Forward path: one-way losses
            'bottom': bottom_stage,  # Bottom reflection
            'water_backward': water_backward,  # Backward path: one-way losses
            'rx': rx_stage,
            'summary': {
                'distance': D_target,  # Average distance (one-way)
                'distance_unit': 'm',
                # Signal parameters
                'f_start': input_dto.signal.f_start,
                'f_end': input_dto.signal.f_end,
                'bandwidth': input_dto.signal.f_end - input_dto.signal.f_start,
                'Tp': input_dto.signal.Tp * 1e-6,  # Convert from µs to seconds
                'Tp_us': input_dto.signal.Tp,  # Keep in µs for display
                'window': input_dto.signal.window,
                # Forward path losses (one-way: TX -> bottom)
                # For display: negative values (losses reduce signal)
                'water_forward_loss': -TL_forward,  # One-way loss: spreading + absorption (negative for display)
                'water_forward_loss_unit': 'dB',
                # Bottom reflection loss (from environment, already negative)
                'bottom_loss': bottom_reflection_db,
                'bottom_loss_unit': 'dB',
                # Backward path losses (one-way: bottom -> RX)
                # For display: negative values (losses reduce signal)
                'water_backward_loss': -TL_backward,  # One-way loss: spreading + absorption (negative for display)
                'water_backward_loss_unit': 'dB',
                # Total round-trip loss (forward + bottom + backward)
                # For display: sum of all negative losses
                'total_path_loss': -TL_forward + bottom_reflection_db - TL_backward,  # All negative: -forward + bottom -backward
                'total_path_loss_unit': 'dB',
                # Receiver gain
                'total_rx_gain': total_rx_gain,
                'total_rx_gain_unit': 'dB',
                # Final signal level
                'signal_at_adc': signal_at_adc_db,
                'signal_at_adc_unit': 'dB',
                # Path structure: explicit separation of forward and backward paths
                'path_structure': {
                    'forward': {
                        'distance': D_one_way,
                        'loss': TL_forward,
                        'description': 'One-way path: TX -> bottom'
                    },
                    'backward': {
                        'distance': D_one_way,
                        'loss': TL_backward,
                        'description': 'One-way path: bottom -> RX'
                    },
                    'round_trip': {
                        'total_distance': 2 * D_one_way,
                        'total_loss': total_path_loss,
                        'description': 'Complete round-trip: TX -> bottom -> RX'
                    }
                }
            }
        }
        
        # Log result for debugging
        logger.info(f"SignalPathCalculator: Returning path_data with keys: {list(result.keys())}")
        logger.debug(f"SignalPathCalculator: summary keys: {list(result.get('summary', {}).keys())}")
        
        return result

