"""
Optimizer - parameter optimization when constraints are violated.
"""

import numpy as np
from typing import Dict, List, Optional
from .dto import InputDTO, OutputDTO, RecommendationsDTO
from .optimizer_strategy import apply_max_tp_min_vga_strategy, apply_min_tp_for_snr_strategy


class Optimizer:
    """
    Parameter optimizer.
    
    Analyzes simulation results and generates recommendations
    for parameter changes to satisfy constraints.
    """
    
    def __init__(self, data_provider=None):
        """
        Initialize optimizer.
        
        Args:
            data_provider: Data provider for accessing transducer parameters (optional)
        """
        self.constraints = {
            'sigma_D_max': 0.01,  # 1 cm
            'SNR_min': 20.0,  # dB (default, will be overridden by target_snr from input_dto)
            'clipping_allowed': False,
        }
        self.data_provider = data_provider
    
    def analyze_results(self, input_dto: InputDTO, output_dto: OutputDTO) -> RecommendationsDTO:
        """
        Analyzes results and generates recommendations.
        
        Args:
            input_dto: Input DTO
            output_dto: Output DTO with results
        
        Returns:
            Optimization recommendations
        """
        recommendations = RecommendationsDTO()
        errors = []  # Critical errors (simulation failed)
        warnings = []  # Warnings (targets not met, but simulation succeeded)
        
        # Use target_snr from input_dto if available, otherwise use default
        target_snr = input_dto.target_snr if hasattr(input_dto, 'target_snr') and input_dto.target_snr is not None else self.constraints['SNR_min']
        
        # Get optimization strategy
        optimization_strategy = getattr(input_dto, 'optimization_strategy', 'max_tp_min_vga')
        
        D_target = input_dto.range.D_target if input_dto.range.D_target is not None else (input_dto.range.D_min + input_dto.range.D_max) / 2
        
        # Get transducer parameters for TX and RX signal level information
        tx_info_text = ""
        rx_info_text = ""
        if self.data_provider:
            try:
                transducer_params = self.data_provider.get_transducer(input_dto.hardware.transducer_id)
                S_TX = transducer_params.get('S_TX', 0.0)
                S_RX = transducer_params.get('S_RX', -193.0)  # dB re 1V/µPa
                V_max = transducer_params.get('V_max')
                if V_max is None:
                    V_max = transducer_params.get('V_nominal')
                if V_max is None:
                    V_max = transducer_params.get('V', 100.0)
                
                # Calculate SPL at maximum voltage
                tx_spl_max_db = S_TX + 20 * np.log10(V_max)  # dB re 1µPa @ 1m
                tx_info_text = f"\nTransducer TX Signal Level:\n"
                tx_info_text += f"- TX Sensitivity (S_TX): {S_TX:.1f} dB re 1µPa/V @ 1m\n"
                tx_info_text += f"- Maximum Voltage (V_max): {V_max:.1f} V\n"
                tx_info_text += f"- Sound Pressure Level at V_max: {tx_spl_max_db:.1f} dB re 1µPa @ 1m\n"
                tx_info_text += f"  (This is the maximum signal level your transducer can transmit)\n"
                
                # RX sensitivity information
                # Get LNA and VGA gains from output_dto (used in simulation) - this is the ACTUAL value used
                # output_dto.vga_gain contains the value that was actually used in the simulation
                # IMPORTANT: Always use output_dto.vga_gain if available, as it reflects the actual value used
                # If output_dto.vga_gain is None, it means the simulation didn't set it properly
                # In that case, we should NOT use vga_params.get('G') as fallback, because that's the default
                # value from the VGA file, not the value actually used in simulation
                # Instead, we should use None and handle it gracefully, or try to get it from input_dto if available
                lna_gain = output_dto.lna_gain if output_dto.lna_gain is not None else None
                vga_gain = output_dto.vga_gain if output_dto.vga_gain is not None else None
                
                # If not in output_dto, try to get from data_provider (fallback)
                # This should rarely happen if simulation ran correctly
                if lna_gain is None:
                    try:
                        lna_params = self.data_provider.get_lna(input_dto.hardware.lna_id)
                        lna_gain = lna_params.get('G_LNA', lna_params.get('G', 20))
                    except:
                        lna_gain = 20.0  # Default
                
                if vga_gain is None:
                    # This should not happen if simulation ran correctly
                    # output_dto.vga_gain should always be set by simulator.simulate()
                    # If it's None, something went wrong, but we need a fallback
                    # Use the default from VGA params, but this may not match what was actually used
                    try:
                        vga_params = self.data_provider.get_vga(input_dto.hardware.vga_id)
                        vga_gain = vga_params.get('G', 30)  # This is the default from VGA file, may not match actual usage
                    except:
                        vga_gain = 30.0  # Default
                
                # If there's a recommended VGA gain and it's different from current, show it
                # This helps user see what the optimizer suggests
                if (hasattr(recommendations, 'suggested_vga_gain') and 
                    recommendations.suggested_vga_gain is not None and
                    abs(recommendations.suggested_vga_gain - vga_gain) > 0.01):  # Tolerance for floating point comparison
                    vga_gain_display = recommendations.suggested_vga_gain
                    vga_gain_note = f" (recommended: {vga_gain_display:.1f} dB, current: {vga_gain:.1f} dB)"
                else:
                    vga_gain_display = vga_gain
                    vga_gain_note = ""
                
                # Calculate total RX gain
                total_rx_gain = S_RX + lna_gain + vga_gain
                
                # Get ADC parameters for compatibility analysis
                adc_info_text = ""
                try:
                    adc_params = self.data_provider.get_adc(input_dto.hardware.adc_id)
                    adc_bits = adc_params.get('N', adc_params.get('bits', 12))
                    adc_vfs = adc_params.get('V_FS', adc_params.get('V_ref', 3.3) * 2)  # Full scale voltage
                    adc_vref = adc_params.get('V_ref', adc_params.get('V_FS', 3.3) / 2)  # Reference voltage
                    
                    # ADC dynamic range (theoretical)
                    adc_dynamic_range_db = 20 * np.log10(2 ** adc_bits)  # dB
                    
                    # ADC input range (maximum signal level before clipping)
                    adc_max_voltage = adc_vfs / 2  # ±V_FS/2
                    adc_max_voltage_db = 20 * np.log10(adc_max_voltage) if adc_max_voltage > 0 else -np.inf  # dBV
                    
                    # Calculate what input pressure level would saturate ADC
                    # V_ADC = P_input * 10^(S_RX/20) * 10^(LNA/20) * 10^(VGA/20)
                    # For saturation: V_ADC = V_FS/2
                    # P_saturate = (V_FS/2) / (10^(S_RX/20) * 10^(LNA/20) * 10^(VGA/20))
                    # In dB: P_saturate_dB = 20*log10(V_FS/2) - S_RX - LNA - VGA
                    total_gain_linear = 10 ** ((lna_gain + vga_gain) / 20)  # LNA + VGA gain in linear scale
                    s_rx_linear = 10 ** (S_RX / 20)  # S_RX in linear scale (V/µPa)
                    
                    # Maximum input pressure that would saturate ADC
                    if s_rx_linear > 0 and total_gain_linear > 0:
                        p_saturate_upa = adc_max_voltage / (s_rx_linear * total_gain_linear)  # µPa
                        p_saturate_db = 20 * np.log10(p_saturate_upa) if p_saturate_upa > 0 else -np.inf  # dB re 1µPa
                    else:
                        p_saturate_upa = float('inf')
                        p_saturate_db = float('inf')
                    
                    # Minimum detectable pressure (quantization noise limited)
                    # Quantization step = V_FS / (2^bits)
                    quantization_step = adc_vfs / (2 ** adc_bits)  # V
                    # Minimum detectable voltage at ADC input
                    v_min_adc = quantization_step  # V
                    # Minimum detectable pressure at transducer input
                    if s_rx_linear > 0 and total_gain_linear > 0:
                        p_min_upa = v_min_adc / (s_rx_linear * total_gain_linear)  # µPa
                        p_min_db = 20 * np.log10(p_min_upa) if p_min_upa > 0 else -np.inf  # dB re 1µPa
                    else:
                        p_min_upa = float('inf')
                        p_min_db = float('inf')
                    
                    # Calculate effective dynamic range at transducer input
                    # This is the range of pressures that ADC can digitize
                    if p_saturate_db != float('inf') and p_min_db != float('inf'):
                        effective_dynamic_range_db = p_saturate_db - p_min_db  # dB
                    else:
                        effective_dynamic_range_db = adc_dynamic_range_db  # Fallback to theoretical
                    
                    # Store ADC info data for later use (will be updated with recommended VGA gain)
                    self._adc_info_data = {
                        'adc_bits': adc_bits,
                        'adc_max_voltage': adc_max_voltage,
                        'adc_vfs': adc_vfs,
                        'adc_dynamic_range_db': adc_dynamic_range_db,
                        'S_RX': S_RX,
                        'lna_gain': lna_gain,
                        'vga_gain': vga_gain,
                        'total_rx_gain': total_rx_gain,
                        'p_saturate_db': p_saturate_db,
                        'p_saturate_upa': p_saturate_upa,
                        'p_min_db': p_min_db,
                        'p_min_upa': p_min_upa,
                        'effective_dynamic_range_db': effective_dynamic_range_db
                    }
                    
                    # Initialize adc_info_text (will be updated later with recommended VGA gain)
                    adc_info_text = ""
                except Exception as e:
                    # If can't get ADC data, skip this information
                    pass
                
                # Store VGA gain values for later use (will be updated after recommendations are calculated)
                # We'll update rx_info_text at the end after suggested_vga_gain is set
                self._rx_info_data = {
                    'S_RX': S_RX,
                    'lna_gain': lna_gain,
                    'vga_gain': vga_gain,
                    'total_rx_gain': S_RX + lna_gain + vga_gain
                }
                
                # Initialize rx_info_text (will be updated later after suggested_vga_gain is calculated)
                rx_info_text = ""
            except Exception:
                # If can't get transducer data, skip this information
                pass
        
        # Add help text explaining what Simulator and Optimizer do
        help_text = f"""=== HELP: Simulator and Optimizer ===

SIMULATOR:
The simulator models a complete hydroacoustic sonar system:
1. Signal Generation: Creates CHIRP signal with specified frequencies and duration
2. Transmission: Models signal transmission through transducer (TX sensitivity)
3. Channel: Models signal propagation through water (attenuation, spreading, delay)
4. Reception: Models signal reception through transducer (RX sensitivity)
5. Receiver Chain: Models LNA (amplification), VGA (variable gain), ADC (digitization)
6. DSP Processing: Performs matched filtering and calculates Time of Flight (TOF)
7. Range Estimation: Calculates measured range (D_measured) and uncertainty (σ_D)

OPTIMIZER:
The optimizer analyzes simulation results and checks if they meet quality constraints:
- Range accuracy (σ_D ≤ 0.01 m = 1 cm)
- Signal quality (SNR ≥ Target SNR = {target_snr:.1f} dB, set in GUI)
- No clipping (ADC must not saturate)
- Range validity (D_measured within [D_min, D_max])

When constraints are violated, the optimizer suggests parameter changes:
- Increase Tp: Improves accuracy and SNR (longer pulse = more energy)
- Increase VGA gain: Improves SNR (more amplification)
- Adjust frequencies: Affects range resolution and minimum/maximum range
- Decrease VGA gain: Eliminates clipping (reduces signal level)

========================================

"""
        # Start with help text
        recommendations.message = help_text
        
        # Build recommendations text separately
        recommendations_text = ""
        
        # Check σ_D (warning, not error - target not met but simulation succeeded)
        # Note: sigma_D check is done before strategy application
        # Strategy will handle Tp recommendations based on its logic
        # We only set increase_Tp here if strategy doesn't handle it
        if output_dto.sigma_D > self.constraints['sigma_D_max']:
            warnings.append(f"σ_D ({output_dto.sigma_D:.4f} m) exceeds target ({self.constraints['sigma_D_max']} m)")
            # Don't set increase_Tp here - let strategy handle it
            # If strategy is "max_tp_min_vga", it will already maximize Tp
            # If strategy is "min_tp_for_snr", it will optimize Tp for SNR
            recommendations_text += "Consider increasing Tp to improve accuracy (if not already at maximum). "
        
        # Check for very long Tp that may require parameter adjustments
        # For large distances, suggest using lower frequency transducer and lower sample rate
        LONG_TP_THRESHOLD_US = 1_000_000  # 1 second
        LARGE_DEPTH_THRESHOLD_M = 1000  # 1 km
        if input_dto.signal.Tp > LONG_TP_THRESHOLD_US or D_target > LARGE_DEPTH_THRESHOLD_M:
            recommendations_text += f"NOTE: "
            if input_dto.signal.Tp > LONG_TP_THRESHOLD_US:
                recommendations_text += f"Tp ({input_dto.signal.Tp:.1f} µs = {input_dto.signal.Tp/1e6:.2f} s) is very long. "
            if D_target > LARGE_DEPTH_THRESHOLD_M:
                recommendations_text += f"Large depth ({D_target:.0f} m) detected. "
            recommendations_text += f"For large distances, consider:\n"
            recommendations_text += f"  • Using lower frequency transducer (reduces absorption loss)\n"
            recommendations_text += f"  • Reducing ADC sample_rate (reduces memory usage, processing time, and number of samples)\n"
            recommendations_text += f"    Note: Sample rate must still satisfy Nyquist criterion (≥ 2*f_end)\n"
            recommendations_text += f"  • Accepting lower accuracy (increases σ_D tolerance)\n"
            recommendations_text += f"  • Using Strategy 2 (Min Tp for SNR) instead of Strategy 1\n"
        
        # Check SNR (warning, not error - target not met but simulation succeeded)
        # Case 1: SNR is below target
        if output_dto.SNR_ADC < target_snr:
            warnings.append(f"SNR ({output_dto.SNR_ADC:.2f} dB) is below target ({target_snr:.2f} dB)")
            
            try:
                from .signal_calculator import SignalCalculator
                calculator = SignalCalculator()
                
                if optimization_strategy == "max_tp_min_vga":
                    # Strategy 1: Use maximum Tp (80% TOF for D_target), then find minimum VGA Gain
                    optimal_tp, recommendations_text = apply_max_tp_min_vga_strategy(
                        input_dto, output_dto, target_snr, recommendations, recommendations_text,
                        self.data_provider, calculator
                    )
                    
                    # Calculate minimum VGA Gain needed for target SNR at optimal Tp
                    recommendations_text = self._calculate_min_vga_for_snr(
                        input_dto, output_dto, target_snr, optimal_tp,
                        recommendations, recommendations_text
                    )
                
                elif optimization_strategy == "min_tp_for_snr":
                    # Strategy 2: Find minimum Tp needed to achieve target SNR at D_target
                    optimal_tp, recommendations_text = apply_min_tp_for_snr_strategy(
                        input_dto, output_dto, target_snr, recommendations, recommendations_text,
                        self.data_provider, calculator
                    )
                    
                    if input_dto.signal.Tp < optimal_tp:
                        recommendations.increase_Tp = True
                        recommendations._optimal_tp_for_snr = optimal_tp
                        recommendations_text += f"Strategy 2 (Min Tp for SNR): Minimum Tp for target SNR ({target_snr:.2f} dB) at D_target: {optimal_tp:.1f} µs. "
                        recommendations_text += f"Current Tp ({input_dto.signal.Tp:.1f} µs) < minimum ({optimal_tp:.1f} µs). "
                    else:
                        recommendations_text += f"Strategy 2 (Min Tp for SNR): Current Tp ({input_dto.signal.Tp:.1f} µs) is sufficient for target SNR. "
                    
                    # For Strategy 2, also calculate minimum VGA gain needed at optimal Tp
                    # This helps optimize VGA gain even when focusing on minimizing Tp
                    recommendations_text = self._calculate_min_vga_for_snr(
                        input_dto, output_dto, target_snr, optimal_tp,
                        recommendations, recommendations_text
                    )
            except Exception:
                # If calculation fails, use fallback logic
                recommendations.increase_Tp = True
                recommendations.increase_G_VGA = True
                recommendations_text += f"Increase Tp and VGA gain to improve SNR (target: {target_snr:.2f} dB). "
        
        # Case 2: SNR is above target
        elif output_dto.SNR_ADC > target_snr:
            try:
                from .signal_calculator import SignalCalculator
                calculator = SignalCalculator()
                
                if optimization_strategy == "max_tp_min_vga":
                    # Strategy 1: Use maximum Tp (80% TOF for D_target), then find minimum VGA Gain
                    optimal_tp, recommendations_text = apply_max_tp_min_vga_strategy(
                        input_dto, output_dto, target_snr, recommendations, recommendations_text,
                        self.data_provider, calculator
                    )
                    
                    # Calculate minimum VGA Gain needed for target SNR at optimal Tp
                    recommendations_text = self._calculate_min_vga_for_snr(
                        input_dto, output_dto, target_snr, optimal_tp,
                        recommendations, recommendations_text
                    )
                
                elif optimization_strategy == "min_tp_for_snr":
                    # Strategy 2: Reduce Tp to minimum needed for target SNR
                    snr_excess = output_dto.SNR_ADC - target_snr
                    if snr_excess > 1.0:  # Only suggest if significantly above (1 dB margin)
                        optimal_tp, recommendations_text = apply_min_tp_for_snr_strategy(
                            input_dto, output_dto, target_snr, recommendations, recommendations_text,
                            self.data_provider, calculator
                        )
                        
                        # Check if we can reduce Tp
                        if optimal_tp < input_dto.signal.Tp:
                            recommendations.decrease_Tp = True
                            recommendations._optimal_tp_for_snr = optimal_tp
                            recommendations_text += f"Strategy 2 (Min Tp for SNR): SNR ({output_dto.SNR_ADC:.2f} dB) is above target ({target_snr:.2f} dB). "
                            recommendations_text += f"Minimum Tp for target SNR at D_target: {optimal_tp:.1f} µs. "
                            recommendations_text += f"Current Tp ({input_dto.signal.Tp:.1f} µs) can be reduced to {optimal_tp:.1f} µs. "
                        else:
                            recommendations_text += f"Strategy 2 (Min Tp for SNR): Current Tp ({input_dto.signal.Tp:.1f} µs) is already at minimum for target SNR. "
                        
                        # For Strategy 2, also calculate minimum VGA gain needed at optimal Tp
                        # This helps optimize VGA gain even when focusing on minimizing Tp
                        recommendations_text = self._calculate_min_vga_for_snr(
                            input_dto, output_dto, target_snr, optimal_tp,
                            recommendations, recommendations_text
                        )
            except Exception:
                # If calculation fails, skip optimization
                pass
            
        
        # Check clipping (error - critical problem)
        if output_dto.clipping_flags:
            errors.append("ADC clipping detected")
            recommendations.decrease_G_VGA = True
            recommendations_text += "Decrease VGA gain to eliminate clipping. "
        
        # Check range (error if significantly out of bounds, warning if slightly out)
        # Allow small deviations due to measurement uncertainty (typically 1-2% of range)
        range_tolerance = max(0.01 * input_dto.range.D_max, 0.1)  # 1% of D_max or 0.1m, whichever is larger
        
        if output_dto.D_measured < (input_dto.range.D_min - range_tolerance):
            errors.append(f"Measured range ({output_dto.D_measured:.2f} m) is significantly less than minimum ({input_dto.range.D_min:.2f} m)")
            recommendations.increase_Tp = True
            recommendations.increase_f_start = True
            recommendations_text += "Increase Tp or f_start to improve minimum range. "
        elif output_dto.D_measured < input_dto.range.D_min:
            warnings.append(f"Measured range ({output_dto.D_measured:.2f} m) is slightly less than minimum ({input_dto.range.D_min:.2f} m)")
        
        if output_dto.D_measured > (input_dto.range.D_max + range_tolerance):
            errors.append(f"Measured range ({output_dto.D_measured:.2f} m) is significantly greater than maximum ({input_dto.range.D_max:.2f} m)")
            recommendations.decrease_Tp = True
            recommendations.decrease_f_end = True
            recommendations_text += "Decrease Tp or f_end to improve maximum range. "
        elif output_dto.D_measured > input_dto.range.D_max:
            warnings.append(f"Measured range ({output_dto.D_measured:.2f} m) is slightly greater than maximum ({input_dto.range.D_max:.2f} m)")
        
        # Calculate suggested parameter changes
        suggested_changes = self.suggest_parameter_changes(input_dto, recommendations)
        
        # If we calculated optimal Tp for SNR reduction, use it instead of heuristic
        # Only set suggested_changes['Tp'] if we have a valid recommendation
        # For Strategy 1 (Max Tp + Min VGA), _optimal_tp_for_snr contains gradual increase (20% or 50%)
        # For Strategy 2 (Min Tp for SNR), _optimal_tp_for_snr contains minimum Tp for SNR
        if recommendations.decrease_Tp and hasattr(recommendations, '_optimal_tp_for_snr') and recommendations._optimal_tp_for_snr is not None:
            suggested_changes['Tp'] = recommendations._optimal_tp_for_snr
        elif recommendations.increase_Tp and hasattr(recommendations, '_optimal_tp_for_snr') and recommendations._optimal_tp_for_snr is not None:
            # For Strategy 1, _optimal_tp_for_snr is the maximum Tp (from calculate_optimal_pulse_duration)
            # For Strategy 2, _optimal_tp_for_snr is the minimum Tp for SNR (from calculate_optimal_tp_for_snr)
            # We need to verify which strategy was used and recalculate if needed
            suggested_tp = recommendations._optimal_tp_for_snr
            current_tp = input_dto.signal.Tp
            
            # Get optimization strategy to verify
            optimization_strategy = getattr(input_dto, 'optimization_strategy', 'max_tp_min_vga')
            
            # For Strategy 1, _optimal_tp_for_snr now contains gradual increase (50%), not maximum
            # Use it directly, but ensure it doesn't exceed physical maximum
            if optimization_strategy == "max_tp_min_vga":
                from .signal_calculator import SignalCalculator
                calculator = SignalCalculator()
                D_target = input_dto.range.D_target if input_dto.range.D_target is not None else (input_dto.range.D_min + input_dto.range.D_max) / 2
                Tp_max_physical = calculator.calculate_optimal_pulse_duration(
                    D_target,
                    input_dto.environment.T,
                    input_dto.environment.S,
                    input_dto.environment.z,
                    min_tp=None
                )
                
                # Log for debugging
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"suggest_parameter_changes: Strategy 1, current_tp={current_tp:.2f} µs, _optimal_tp_for_snr={suggested_tp:.2f} µs, Tp_max_physical={Tp_max_physical:.2f} µs, D_target={D_target:.2f}m")
                
                # Ensure suggested_tp doesn't exceed physical maximum
                if suggested_tp > Tp_max_physical:
                    logger.info(f"suggest_parameter_changes: Suggested Tp ({suggested_tp:.2f} µs) exceeds physical maximum ({Tp_max_physical:.2f} µs). Clamping to maximum.")
                    suggested_tp = Tp_max_physical
            
            suggested_changes['Tp'] = suggested_tp
        
        # Add VGA gain to suggested_changes if there's a recommendation
        if hasattr(recommendations, 'suggested_vga_gain') and recommendations.suggested_vga_gain is not None:
            suggested_changes['VGA_gain'] = recommendations.suggested_vga_gain
        
        # Store suggested changes in recommendations for GUI to apply
        recommendations.suggested_changes = suggested_changes
        
        # Now update rx_info_text with recommended VGA gain if available
        # This must be done AFTER suggested_vga_gain is set in _calculate_min_vga_for_snr
        if hasattr(self, '_rx_info_data') and self._rx_info_data:
            rx_data = self._rx_info_data
            # Check if there's a recommended VGA gain and it's different from current
            if (hasattr(recommendations, 'suggested_vga_gain') and 
                recommendations.suggested_vga_gain is not None and
                abs(recommendations.suggested_vga_gain - rx_data['vga_gain']) > 0.01):  # Tolerance for floating point comparison
                vga_gain_display = recommendations.suggested_vga_gain
                vga_gain_note = f" (recommended: {vga_gain_display:.1f} dB, current: {rx_data['vga_gain']:.1f} dB)"
            else:
                vga_gain_display = rx_data['vga_gain']
                vga_gain_note = ""
            
            # Calculate total RX gain with displayed VGA gain
            total_rx_gain_display = rx_data['S_RX'] + rx_data['lna_gain'] + vga_gain_display
            
            rx_info_text = f"\nTransducer RX Signal Level:\n"
            rx_info_text += f"- RX Sensitivity (S_RX): {rx_data['S_RX']:.1f} dB re 1V/µPa\n"
            rx_info_text += f"  (Higher S_RX = better sensitivity = stronger received signal)\n"
            rx_info_text += f"- LNA Gain: {rx_data['lna_gain']:.1f} dB\n"
            rx_info_text += f"- VGA Gain: {vga_gain_display:.1f} dB{vga_gain_note}\n"
            rx_info_text += f"- Total RX Gain: {total_rx_gain_display:.1f} dB (S_RX + LNA + VGA)\n"
            rx_info_text += f"  (This is the total amplification of received signal before ADC)\n"
        
        # Update adc_info_text with recommended VGA gain if available
        if hasattr(self, '_adc_info_data') and self._adc_info_data:
            adc_data = self._adc_info_data
            # Use recommended VGA gain if available, otherwise use current
            if hasattr(recommendations, 'suggested_vga_gain') and recommendations.suggested_vga_gain is not None:
                vga_gain_for_adc = recommendations.suggested_vga_gain
            else:
                vga_gain_for_adc = adc_data['vga_gain']
            
            # Recalculate ADC analysis with recommended VGA gain
            total_gain_linear = 10 ** ((adc_data['lna_gain'] + vga_gain_for_adc) / 20)
            s_rx_linear = 10 ** (adc_data['S_RX'] / 20)
            
            if s_rx_linear > 0 and total_gain_linear > 0:
                p_saturate_upa = adc_data['adc_max_voltage'] / (s_rx_linear * total_gain_linear)
                p_saturate_db = 20 * np.log10(p_saturate_upa) if p_saturate_upa > 0 else -np.inf
                quantization_step = adc_data['adc_vfs'] / (2 ** adc_data['adc_bits'])
                v_min_adc = quantization_step
                p_min_upa = v_min_adc / (s_rx_linear * total_gain_linear)
                p_min_db = 20 * np.log10(p_min_upa) if p_min_upa > 0 else -np.inf
                effective_dynamic_range_db = p_saturate_db - p_min_db if (p_saturate_db != float('inf') and p_min_db != float('inf')) else adc_data['adc_dynamic_range_db']
            else:
                p_saturate_db = adc_data['p_saturate_db']
                p_saturate_upa = adc_data['p_saturate_upa']
                p_min_db = adc_data['p_min_db']
                p_min_upa = adc_data['p_min_upa']
                effective_dynamic_range_db = adc_data['effective_dynamic_range_db']
            
            total_rx_gain_display = adc_data['S_RX'] + adc_data['lna_gain'] + vga_gain_for_adc
            
            adc_info_text = f"\nADC Compatibility Analysis:\n"
            adc_info_text += f"- ADC Resolution: {adc_data['adc_bits']} bits\n"
            adc_info_text += f"- ADC Full Scale: ±{adc_data['adc_max_voltage']:.3f} V (V_FS = {adc_data['adc_vfs']:.3f} V)\n"
            adc_info_text += f"- ADC Dynamic Range: {adc_data['adc_dynamic_range_db']:.1f} dB (theoretical)\n"
            adc_info_text += f"\nReceiver Chain Analysis (S_RX + LNA + VGA):\n"
            adc_info_text += f"- Total RX Gain: {total_rx_gain_display:.1f} dB (S_RX={adc_data['S_RX']:.1f} + LNA={adc_data['lna_gain']:.1f} + VGA={vga_gain_for_adc:.1f})\n"
            adc_info_text += f"- Maximum Input Pressure (saturation): {p_saturate_db:.1f} dB re 1µPa ({p_saturate_upa:.2e} µPa)\n"
            adc_info_text += f"- Minimum Detectable Pressure: {p_min_db:.1f} dB re 1µPa ({p_min_upa:.2e} µPa)\n"
            adc_info_text += f"- Effective Dynamic Range: {effective_dynamic_range_db:.1f} dB\n"
            adc_info_text += f"\nInterpretation:\n"
            if p_saturate_db < 200:
                adc_info_text += f"  • Input pressure > {p_saturate_db:.1f} dB re 1µPa will saturate ADC\n"
            if p_min_db > -200:
                adc_info_text += f"  • Input pressure < {p_min_db:.1f} dB re 1µPa may be below quantization noise\n"
            adc_info_text += f"  • Optimal input range: {p_min_db:.1f} to {p_saturate_db:.1f} dB re 1µPa\n"
            adc_info_text += f"  • Current combination provides {effective_dynamic_range_db:.1f} dB usable range\n"
        
        # Build optimized parameters section
        # Show section if there are suggested changes OR VGA gain recommendations
        has_vga_recommendation = recommendations.increase_G_VGA or recommendations.decrease_G_VGA
        optimized_params_text = ""
        if suggested_changes or has_vga_recommendation:
            optimized_params_text += "\n\n=== OPTIMIZED PARAMETERS ===\n\n"
            optimized_params_text += "Current values -> Suggested values:\n\n"
            
            # Add each optimized parameter with current and suggested values
            if 'Tp' in suggested_changes:
                current = input_dto.signal.Tp
                suggested = suggested_changes['Tp']
                change_pct = ((suggested - current) / current) * 100
                if recommendations.aggressive_Tp_increase:
                    optimized_params_text += f"Tp (pulse duration): {current:.2f} µs -> {suggested:.2f} µs ({change_pct:+.1f}% - AGGRESSIVE increase, VGA at max)\n"
                else:
                    optimized_params_text += f"Tp (pulse duration): {current:.2f} µs -> {suggested:.2f} µs ({change_pct:+.1f}%)\n"
                
                # Add warning for Strategy 1 about repeated optimization
                optimization_strategy = getattr(input_dto, 'optimization_strategy', 'max_tp_min_vga')
                if optimization_strategy == "max_tp_min_vga" and recommendations.increase_Tp:
                    optimized_params_text += "\n⚠️ WARNING: Strategy 1 (Max Tp + Min VGA) will continue to suggest Tp increases until maximum is reached.\n"
                    optimized_params_text += "   Do NOT run optimization repeatedly - apply recommendations once and let optimization converge gradually.\n"
                    optimized_params_text += "   Running optimization again immediately will suggest another increase, as Tp is still below maximum.\n"
            # Removed: artificial maximum limit check (4900 µs)
            # Tp can now be increased without artificial limits (only physical constraint applies)
            
            if 'f_start' in suggested_changes:
                current = input_dto.signal.f_start
                suggested = suggested_changes['f_start']
                change_pct = ((suggested - current) / current) * 100
                optimized_params_text += f"f_start (start frequency): {current:.2f} Hz -> {suggested:.2f} Hz ({change_pct:+.1f}%)\n"
            
            if 'f_end' in suggested_changes:
                current = input_dto.signal.f_end
                suggested = suggested_changes['f_end']
                change_pct = ((suggested - current) / current) * 100
                optimized_params_text += f"f_end (end frequency): {current:.2f} Hz -> {suggested:.2f} Hz ({change_pct:+.1f}%)\n"
            
            # Add VGA gain with current value
            if has_vga_recommendation:
                current_vga = output_dto.vga_gain if output_dto.vga_gain is not None else 0.0
                vga_max = output_dto.vga_gain_max if output_dto.vga_gain_max is not None else None
                if recommendations.increase_G_VGA:
                    if vga_max is not None and (vga_max - current_vga) < 1.0:
                        optimized_params_text += f"VGA gain: {current_vga:.1f} dB (MAX: {vga_max:.1f} dB) - Already at maximum!\n"
                    elif recommendations.suggested_vga_gain is not None:
                        # Show specific suggested value
                        suggested_vga = recommendations.suggested_vga_gain
                        change = suggested_vga - current_vga
                        optimized_params_text += f"VGA gain: {current_vga:.1f} dB -> {suggested_vga:.1f} dB (+{change:.1f} dB)\n"
                    else:
                        optimized_params_text += f"VGA gain: {current_vga:.1f} dB -> Increase (adjust manually in GUI)\n"
                elif recommendations.decrease_G_VGA:
                    optimized_params_text += f"VGA gain: {current_vga:.1f} dB -> Decrease (adjust manually in GUI)\n"
            
            optimized_params_text += "\n"
        
        # Finalize message: help text + recommendations + optimized parameters
        if recommendations_text.strip():
            # Has recommendations - show recommendations and optimized parameters
            final_message = help_text + recommendations_text.strip() + optimized_params_text
            if tx_info_text:
                final_message += tx_info_text
            if rx_info_text:
                final_message += rx_info_text
            if adc_info_text:
                final_message += adc_info_text
            recommendations.message = final_message
        elif optimized_params_text.strip():
            # Only optimized parameters (shouldn't happen, but just in case)
            final_message = help_text + optimized_params_text
            if tx_info_text:
                final_message += tx_info_text
            if rx_info_text:
                final_message += rx_info_text
            if adc_info_text:
                final_message += adc_info_text
            recommendations.message = final_message
        else:
            # No recommendations - show current parameter values for reference
            current_params_text = "\n\n=== CURRENT PARAMETERS ===\n\n"
            current_params_text += f"Tp (pulse duration): {input_dto.signal.Tp:.2f} µs\n"
            current_params_text += f"f_start (start frequency): {input_dto.signal.f_start:.2f} Hz\n"
            current_params_text += f"f_end (end frequency): {input_dto.signal.f_end:.2f} Hz\n"
            if output_dto.lna_gain is not None:
                current_params_text += f"LNA gain: {output_dto.lna_gain:.1f} dB\n"
            if output_dto.vga_gain is not None:
                current_params_text += f"VGA gain: {output_dto.vga_gain:.1f} dB\n"
            current_params_text += f"Target SNR: {target_snr:.1f} dB\n"
            current_params_text += f"Measured SNR: {output_dto.SNR_ADC:.2f} dB (at ADC OUTPUT, after quantization)\n"
            current_params_text += f"Range accuracy (σ_D): {output_dto.sigma_D:.4f} m\n"
            current_params_text += f"Measured range: {output_dto.D_measured:.2f} m\n"
            
            # Add TX, RX, and ADC signal level information if available
            if tx_info_text:
                current_params_text += tx_info_text
            if rx_info_text:
                current_params_text += rx_info_text
            if adc_info_text:
                current_params_text += adc_info_text
            
            recommendations.message = help_text + "\n\nNo issues detected. All constraints are satisfied." + current_params_text
        
        # Set errors and warnings
        # Only critical errors mark simulation as unsuccessful
        # Warnings indicate targets not met, but simulation succeeded
        if errors:
            output_dto.success = False
            output_dto.errors.extend(errors)
        else:
            output_dto.success = True
        
        if warnings:
            output_dto.warnings.extend(warnings)
        
        return recommendations
    
    def _calculate_min_vga_for_snr(self, input_dto: InputDTO, output_dto: OutputDTO,
                                    target_snr: float, optimal_tp: float,
                                    recommendations: RecommendationsDTO, recommendations_text: str) -> str:
        """
        Calculate minimum VGA Gain needed for target SNR at given Tp.
        
        Args:
            input_dto: Input DTO
            output_dto: Output DTO
            target_snr: Target SNR in dB
            optimal_tp: Optimal Tp in microseconds
            recommendations: Recommendations DTO to update
            recommendations_text: Text to append recommendations to
        
        Returns:
            Updated recommendations_text
        """
        try:
            if self.data_provider:
                transducer_params = self.data_provider.get_transducer(input_dto.hardware.transducer_id)
                lna_params = self.data_provider.get_lna(input_dto.hardware.lna_id)
                vga_params = self.data_provider.get_vga(input_dto.hardware.vga_id)
                
                # Get current VGA Gain
                current_vga_gain = output_dto.vga_gain if output_dto.vga_gain is not None else vga_params.get('G', 30)
                vga_gain_min = vga_params.get('G_min', 0)
                vga_gain_max = output_dto.vga_gain_max if output_dto.vga_gain_max is not None else vga_params.get('G_max', 60)
                
                # Calculate SNR at optimal Tp using sonar equation (more accurate for large distances)
                # Use SignalCalculator to get accurate SNR calculation
                from .signal_calculator import SignalCalculator
                calculator = SignalCalculator()
                
                D_target = input_dto.range.D_target if input_dto.range.D_target is not None else (input_dto.range.D_min + input_dto.range.D_max) / 2
                
                # Prepare hardware params for SNR calculation
                hardware_params_for_snr = {
                    'lna_gain': lna_params.get('G_LNA', lna_params.get('G', 20)),
                    'lna_nf': lna_params.get('NF_LNA', lna_params.get('NF', 2.0)),
                    'vga_gain': current_vga_gain,  # Use current VGA gain for baseline SNR
                    'bottom_reflection': input_dto.environment.bottom_reflection
                }
                
                # Get TX voltage
                tx_voltage = transducer_params.get('V_max', transducer_params.get('V_nominal', 100.0))
                
                # Calculate SNR at optimal Tp using sonar equation
                snr_at_optimal_tp = calculator._calculate_snr_from_sonar_equation(
                    D=D_target,
                    Tp_us=optimal_tp,
                    transducer_params=transducer_params,
                    hardware_params=hardware_params_for_snr,
                    T=input_dto.environment.T,
                    S=input_dto.environment.S,
                    z=input_dto.environment.z,
                    f_start=input_dto.signal.f_start,
                    f_end=input_dto.signal.f_end,
                    tx_voltage=tx_voltage
                )
                
                # VGA gain affects SNR approximately 1:1 (signal and noise both amplified)
                # But for large distances, we need to account for the fact that VGA gain
                # increases signal power more than noise power (noise is dominated by ambient)
                # Approximate: SNR_VGA ≈ SNR_LNA + G_VGA (for high SNR cases)
                # For low SNR cases, the relationship is more complex
                # Use simplified model: each dB of VGA gain adds approximately 1 dB to SNR
                
                # Calculate required VGA Gain change to achieve target SNR
                snr_deficit_at_optimal_tp = target_snr - snr_at_optimal_tp
                
                if snr_deficit_at_optimal_tp > 0:
                    # Need to increase VGA Gain
                    required_vga_gain = current_vga_gain + snr_deficit_at_optimal_tp
                    # Limit to maximum
                    required_vga_gain = min(required_vga_gain, vga_gain_max)
                    
                    # Only set suggested_vga_gain if it's different from current
                    if required_vga_gain != current_vga_gain:
                        recommendations.increase_G_VGA = True
                        recommendations.suggested_vga_gain = required_vga_gain
                        recommendations_text += f"At Tp={optimal_tp:.1f} µs, estimated SNR={snr_at_optimal_tp:.2f} dB. "
                        recommendations_text += f"Minimum VGA Gain for target SNR ({target_snr:.2f} dB): {required_vga_gain:.1f} dB (current: {current_vga_gain:.1f} dB). "
                        if required_vga_gain >= vga_gain_max:
                            recommendations_text += f"WARNING: Required VGA Gain ({required_vga_gain:.1f} dB) is at maximum ({vga_gain_max:.1f} dB). Target SNR may not be achievable. "
                            recommendations.warn_unachievable_snr = True
                    else:
                        # VGA gain is already optimal
                        recommendations_text += f"At Tp={optimal_tp:.1f} µs, estimated SNR={snr_at_optimal_tp:.2f} dB. "
                        recommendations_text += f"VGA Gain is already optimal at {current_vga_gain:.1f} dB for target SNR ({target_snr:.2f} dB). "
                else:
                    # SNR at optimal Tp is already at or above target - can reduce VGA Gain
                    # Calculate minimum VGA gain needed: start from minimum and find what's needed
                    # Try with minimum VGA gain first to see if it's sufficient
                    hardware_params_min_vga = hardware_params_for_snr.copy()
                    hardware_params_min_vga['vga_gain'] = vga_gain_min
                    
                    snr_at_optimal_tp_min_vga = calculator._calculate_snr_from_sonar_equation(
                        D=D_target,
                        Tp_us=optimal_tp,
                        transducer_params=transducer_params,
                        hardware_params=hardware_params_min_vga,
                        T=input_dto.environment.T,
                        S=input_dto.environment.S,
                        z=input_dto.environment.z,
                        f_start=input_dto.signal.f_start,
                        f_end=input_dto.signal.f_end,
                        tx_voltage=tx_voltage
                    )
                    
                    if snr_at_optimal_tp_min_vga >= target_snr:
                        # Even with minimum VGA gain, SNR is above target
                        # But don't use absolute minimum (0 dB) - calculate what's actually needed
                        # Calculate the exact VGA gain needed to achieve target SNR
                        # SNR excess above target
                        snr_excess = snr_at_optimal_tp_min_vga - target_snr
                        # Each dB of VGA gain adds approximately 1 dB to SNR
                        # So we can reduce VGA gain by the excess SNR amount
                        required_vga_gain = vga_gain_min + snr_excess
                        # But ensure we don't go below minimum (shouldn't happen, but safety check)
                        required_vga_gain = max(required_vga_gain, vga_gain_min)
                    else:
                        # Need some VGA gain - calculate exactly what's needed
                        # SNR deficit from minimum VGA to target
                        snr_deficit_from_min = target_snr - snr_at_optimal_tp_min_vga
                        # Each dB of VGA gain adds approximately 1 dB to SNR
                        required_vga_gain = vga_gain_min + snr_deficit_from_min
                        # Limit to maximum (shouldn't happen, but just in case)
                        required_vga_gain = min(required_vga_gain, vga_gain_max)
                    
                    # Ensure it's at least minimum
                    required_vga_gain = max(required_vga_gain, vga_gain_min)
                    
                    if required_vga_gain < current_vga_gain:
                        recommendations.decrease_G_VGA = True
                        recommendations.suggested_vga_gain = required_vga_gain
                        recommendations_text += f"At Tp={optimal_tp:.1f} µs, estimated SNR={snr_at_optimal_tp:.2f} dB. "
                        recommendations_text += f"SNR is above target. Can reduce VGA Gain to {required_vga_gain:.1f} dB (current: {current_vga_gain:.1f} dB) while maintaining target SNR. "
                    elif required_vga_gain == current_vga_gain:
                        recommendations_text += f"At Tp={optimal_tp:.1f} µs, estimated SNR={snr_at_optimal_tp:.2f} dB. "
                        recommendations_text += f"SNR is at target. VGA Gain is already optimal at {current_vga_gain:.1f} dB. "
                    else:
                        # This shouldn't happen if logic is correct, but handle it
                        recommendations_text += f"At Tp={optimal_tp:.1f} µs, estimated SNR={snr_at_optimal_tp:.2f} dB. "
                        recommendations_text += f"SNR is above target, but VGA Gain calculation suggests increase (this may indicate a calculation error). "
        except Exception as e:
            # If calculation fails, skip VGA optimization
            pass
        
        return recommendations_text
    
    def suggest_parameter_changes(self, input_dto: InputDTO, 
                                  recommendations: RecommendationsDTO) -> Dict:
        """
        Suggests specific parameter changes.
        
        Args:
            input_dto: Current input DTO
            recommendations: Recommendations
        
        Returns:
            Dictionary with suggested changes
        """
        changes = {}
        
        # Handle Tp changes - decrease has priority over increase
        # (if both are set, decrease is more important when SNR is above target)
        if recommendations.decrease_Tp:
            # Use calculated optimal Tp if available, otherwise use heuristic (20% decrease)
            if hasattr(recommendations, '_optimal_tp_for_snr') and recommendations._optimal_tp_for_snr is not None:
                changes['Tp'] = recommendations._optimal_tp_for_snr
            else:
                changes['Tp'] = input_dto.signal.Tp * 0.8  # Decrease by 20%
        elif recommendations.increase_Tp:
            # Only increase if not decreasing
            # For Strategy 1 (Max Tp + Min VGA), _optimal_tp_for_snr contains gradual increase (20% or 50%)
            # For Strategy 2 (Min Tp for SNR), _optimal_tp_for_snr contains minimum Tp for SNR
            if hasattr(recommendations, '_optimal_tp_for_snr') and recommendations._optimal_tp_for_snr is not None:
                # Use the calculated optimal Tp (gradual increase for Strategy 1, minimum for Strategy 2)
                suggested_tp = recommendations._optimal_tp_for_snr
                
                # Safety check: verify that suggested_tp is reasonable
                # It should not exceed physical maximum (80% TOF for D_target)
                # and should not be more than 10x current Tp (to prevent huge jumps)
                D_target = input_dto.range.D_target if input_dto.range.D_target is not None else (input_dto.range.D_min + input_dto.range.D_max) / 2
                from .signal_calculator import SignalCalculator
                calculator = SignalCalculator()
                Tp_max_physical = calculator.calculate_optimal_pulse_duration(
                    D_target,
                    input_dto.environment.T,
                    input_dto.environment.S,
                    input_dto.environment.z,
                    min_tp=None
                )
                
                current_tp = input_dto.signal.Tp
                max_reasonable_increase = current_tp * 10.0  # Don't increase more than 10x
                
                # Clamp suggested_tp to reasonable limits
                if suggested_tp > Tp_max_physical:
                    # Suggested Tp exceeds physical maximum - use physical maximum instead
                    suggested_tp = Tp_max_physical
                elif suggested_tp > max_reasonable_increase:
                    # Suggested Tp is too large compared to current - limit to 10x
                    suggested_tp = max_reasonable_increase
                
                changes['Tp'] = suggested_tp
            else:
                # Fallback: use percentage increase if optimal_tp not available
                # Check if this is for large distance
                D_target = input_dto.range.D_target if input_dto.range.D_target is not None else (input_dto.range.D_min + input_dto.range.D_max) / 2
                is_large_distance = D_target > 200.0
                
                if recommendations.aggressive_Tp_increase:
                    # More aggressive increase when VGA is at maximum (50-100% increase)
                    # No artificial maximum limit - only physical constraint applies
                    suggested_tp = input_dto.signal.Tp * 2.0  # Double Tp
                    if suggested_tp > input_dto.signal.Tp:
                        changes['Tp'] = suggested_tp
                else:
                    if is_large_distance:
                        # For large distances, more aggressive Tp increase (50% instead of 20%)
                        suggested_tp = input_dto.signal.Tp * 1.5
                    else:
                        suggested_tp = input_dto.signal.Tp * 1.2  # Increase by 20%
                    
                    # No artificial maximum limit - only physical constraint applies
                    changes['Tp'] = suggested_tp
        
        if recommendations.increase_f_start:
            changes['f_start'] = input_dto.signal.f_start * 1.1
        
        if recommendations.decrease_f_start:
            changes['f_start'] = input_dto.signal.f_start * 0.9
        
        if recommendations.increase_f_end:
            changes['f_end'] = input_dto.signal.f_end * 1.1
        
        if recommendations.decrease_f_end:
            changes['f_end'] = input_dto.signal.f_end * 0.9
        
        return changes

