"""
Optimizer - parameter optimization when constraints are violated.
"""

import numpy as np
from typing import Dict, List, Optional
from .dto import InputDTO, OutputDTO, RecommendationsDTO


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
                # Get LNA and VGA gains from output_dto (used in simulation) or from data
                lna_gain = output_dto.lna_gain if output_dto.lna_gain is not None else None
                vga_gain = output_dto.vga_gain if output_dto.vga_gain is not None else None
                
                # If not in output_dto, try to get from data_provider
                if lna_gain is None:
                    try:
                        lna_params = self.data_provider.get_lna(input_dto.hardware.lna_id)
                        lna_gain = lna_params.get('G_LNA', lna_params.get('G', 20))
                    except:
                        lna_gain = 20.0  # Default
                
                if vga_gain is None:
                    try:
                        vga_params = self.data_provider.get_vga(input_dto.hardware.vga_id)
                        vga_gain = vga_params.get('G', 30)
                    except:
                        vga_gain = 30.0  # Default
                
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
                    
                    adc_info_text = f"\nADC Compatibility Analysis:\n"
                    adc_info_text += f"- ADC Resolution: {adc_bits} bits\n"
                    adc_info_text += f"- ADC Full Scale: ±{adc_max_voltage:.3f} V (V_FS = {adc_vfs:.3f} V)\n"
                    adc_info_text += f"- ADC Dynamic Range: {adc_dynamic_range_db:.1f} dB (theoretical)\n"
                    adc_info_text += f"\nReceiver Chain Analysis (S_RX + LNA + VGA):\n"
                    adc_info_text += f"- Total RX Gain: {total_rx_gain:.1f} dB (S_RX={S_RX:.1f} + LNA={lna_gain:.1f} + VGA={vga_gain:.1f})\n"
                    adc_info_text += f"- Maximum Input Pressure (saturation): {p_saturate_db:.1f} dB re 1µPa ({p_saturate_upa:.2e} µPa)\n"
                    adc_info_text += f"- Minimum Detectable Pressure: {p_min_db:.1f} dB re 1µPa ({p_min_upa:.2e} µPa)\n"
                    adc_info_text += f"- Effective Dynamic Range: {effective_dynamic_range_db:.1f} dB\n"
                    adc_info_text += f"\nInterpretation:\n"
                    if p_saturate_db < 200:  # Reasonable saturation level
                        adc_info_text += f"  • Input pressure > {p_saturate_db:.1f} dB re 1µPa will saturate ADC\n"
                    if p_min_db > -200:  # Reasonable minimum
                        adc_info_text += f"  • Input pressure < {p_min_db:.1f} dB re 1µPa may be below quantization noise\n"
                    adc_info_text += f"  • Optimal input range: {p_min_db:.1f} to {p_saturate_db:.1f} dB re 1µPa\n"
                    adc_info_text += f"  • Current combination provides {effective_dynamic_range_db:.1f} dB usable range\n"
                except Exception as e:
                    # If can't get ADC data, skip this information
                    pass
                
                rx_info_text = f"\nTransducer RX Signal Level:\n"
                rx_info_text += f"- RX Sensitivity (S_RX): {S_RX:.1f} dB re 1V/µPa\n"
                rx_info_text += f"  (Higher S_RX = better sensitivity = stronger received signal)\n"
                rx_info_text += f"- LNA Gain: {lna_gain:.1f} dB\n"
                rx_info_text += f"- VGA Gain: {vga_gain:.1f} dB\n"
                rx_info_text += f"- Total RX Gain: {total_rx_gain:.1f} dB (S_RX + LNA + VGA)\n"
                rx_info_text += f"  (This is the total amplification of received signal before ADC)\n"
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
        if output_dto.sigma_D > self.constraints['sigma_D_max']:
            warnings.append(f"σ_D ({output_dto.sigma_D:.4f} m) exceeds target ({self.constraints['sigma_D_max']} m)")
            # Increase Tp for better resolution
            recommendations.increase_Tp = True
            recommendations_text += "Increase Tp to improve accuracy. "
        
        # Check SNR (warning, not error - target not met but simulation succeeded)
        if output_dto.SNR_ADC < target_snr:
            warnings.append(f"SNR ({output_dto.SNR_ADC:.2f} dB) is below target ({target_snr:.2f} dB)")
            # Check if VGA is already at maximum
            vga_at_max = False
            if output_dto.vga_gain is not None and output_dto.vga_gain_max is not None:
                # Consider VGA at max if within 1 dB of maximum (accounting for rounding)
                vga_at_max = (output_dto.vga_gain_max - output_dto.vga_gain) < 1.0
            
            if vga_at_max:
                # VGA is at maximum - need more aggressive Tp increase
                recommendations.increase_Tp = True
                recommendations.aggressive_Tp_increase = True  # Flag for more aggressive increase
                recommendations_text += f"VGA gain is at maximum ({output_dto.vga_gain:.1f} dB). "
                recommendations_text += f"Significantly increase Tp to improve SNR (target: {target_snr:.2f} dB). "
                # Also suggest checking if target SNR is achievable
                snr_deficit = target_snr - output_dto.SNR_ADC
                if snr_deficit > 10:
                    recommendations_text += f"WARNING: Large SNR deficit ({snr_deficit:.1f} dB). "
                    recommendations_text += "Consider reducing target SNR or increasing distance may not be achievable. "
            else:
                # VGA can still be increased
                # Calculate how much VGA gain increase is needed
                snr_deficit = target_snr - output_dto.SNR_ADC
                
                # For large distances, be more aggressive
                D_target = input_dto.range.D_target if input_dto.range.D_target is not None else (input_dto.range.D_min + input_dto.range.D_max) / 2
                is_large_distance = D_target > 200.0  # Consider >200m as large distance
                
                # Check if Tp is at or near maximum (cannot be increased much)
                Tp_at_max = input_dto.signal.Tp >= 4900.0  # Consider >=4900 µs as at maximum
                
                # Calculate suggested VGA gain increase
                # VGA gain directly affects SNR (1 dB VGA = 1 dB SNR, approximately)
                # But in practice, the relationship may not be 1:1 due to noise floor
                # Use a more conservative estimate: 0.8-0.9 dB SNR per 1 dB VGA gain
                vga_to_snr_ratio = 0.85  # Conservative estimate
                
                if Tp_at_max:
                    # If Tp is at maximum, we need to compensate more through VGA gain
                    # Calculate required VGA gain increase to cover the deficit
                    # Required VGA increase = SNR deficit / vga_to_snr_ratio
                    required_vga_increase = snr_deficit / vga_to_snr_ratio
                    # Use up to 95% of available VGA range (leave some margin)
                    available_vga_range = output_dto.vga_gain_max - output_dto.vga_gain if output_dto.vga_gain_max and output_dto.vga_gain else 100.0
                    vga_gain_increase = min(required_vga_increase, available_vga_range * 0.95)
                elif is_large_distance:
                    # For large distances, use more VGA gain (up to 70% of deficit)
                    vga_gain_increase = min(snr_deficit * 0.7 / vga_to_snr_ratio, output_dto.vga_gain_max - output_dto.vga_gain if output_dto.vga_gain_max else 20.0)
                else:
                    # For normal distances, use less VGA gain (up to 50% of deficit)
                    vga_gain_increase = min(snr_deficit * 0.5 / vga_to_snr_ratio, output_dto.vga_gain_max - output_dto.vga_gain if output_dto.vga_gain_max else 20.0)
                
                # Round to reasonable step (e.g., 1 dB), but ensure at least 1 dB increase
                vga_gain_increase = max(1.0, round(vga_gain_increase))
                
                # Store suggested VGA gain value
                suggested_vga = None
                if output_dto.vga_gain is not None and output_dto.vga_gain_max is not None:
                    suggested_vga = min(output_dto.vga_gain + vga_gain_increase, output_dto.vga_gain_max)
                    recommendations.suggested_vga_gain = suggested_vga
                    
                    # Check if we're suggesting maximum VGA gain
                    vga_at_suggested_max = (output_dto.vga_gain_max - suggested_vga) < 1.0
                    if vga_at_suggested_max and snr_deficit > 2.0:
                        # If suggesting max VGA and still large deficit, warn about feasibility
                        recommendations.warn_unachievable_snr = True
                
                recommendations.increase_G_VGA = True
                
                # Only suggest Tp increase if not at maximum
                if not Tp_at_max:
                    recommendations.increase_Tp = True
                
                # Build recommendation text
                if Tp_at_max:
                    # Tp is at maximum, focus on VGA gain
                    if suggested_vga is not None:
                        if suggested_vga >= output_dto.vga_gain_max - 1.0:
                            recommendations_text += f"Tp is at maximum ({input_dto.signal.Tp:.0f} µs). "
                            recommendations_text += f"Increase VGA gain to maximum ({suggested_vga:.1f} dB) to improve SNR (target: {target_snr:.2f} dB). "
                            if is_large_distance:
                                recommendations_text += f"WARNING: For large distance ({D_target:.0f} m), achieving target SNR may require maximum VGA gain. "
                        else:
                            # Estimate expected SNR after VGA gain increase
                            estimated_snr_after = output_dto.SNR_ADC + (suggested_vga - output_dto.vga_gain) * vga_to_snr_ratio
                            recommendations_text += f"Tp is at maximum ({input_dto.signal.Tp:.0f} µs). "
                            recommendations_text += f"Significantly increase VGA gain (suggested: {suggested_vga:.1f} dB) to improve SNR (target: {target_snr:.2f} dB). "
                            recommendations_text += f"Estimated SNR after change: {estimated_snr_after:.1f} dB. "
                            if estimated_snr_after < target_snr * 0.9:  # Still below 90% of target
                                if output_dto.vga_gain_max:
                                    recommendations_text += f"May need to increase VGA gain further (up to {output_dto.vga_gain_max:.1f} dB max). "
                                if recommendations.warn_unachievable_snr:
                                    recommendations_text += f"WARNING: Target SNR ({target_snr:.1f} dB) may not be achievable even with maximum VGA gain. "
                                    recommendations_text += f"Consider reducing target SNR or distance. "
                    else:
                        recommendations_text += f"Tp is at maximum ({input_dto.signal.Tp:.0f} µs). "
                        recommendations_text += f"Significantly increase VGA gain to improve SNR (target: {target_snr:.2f} dB). "
                elif is_large_distance:
                    if suggested_vga is not None:
                        recommendations_text += f"For large distance ({D_target:.0f} m), significantly increase VGA gain (suggested: {suggested_vga:.1f} dB) and Tp to improve SNR (target: {target_snr:.2f} dB). "
                    else:
                        recommendations_text += f"For large distance ({D_target:.0f} m), significantly increase VGA gain and Tp to improve SNR (target: {target_snr:.2f} dB). "
                else:
                    if suggested_vga is not None:
                        recommendations_text += f"Increase VGA gain (suggested: {suggested_vga:.1f} dB) or Tp to improve SNR (target: {target_snr:.2f} dB). "
                    else:
                        recommendations_text += f"Increase VGA gain or Tp to improve SNR (target: {target_snr:.2f} dB). "
        
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
        # Store suggested changes in recommendations for GUI to apply
        recommendations.suggested_changes = suggested_changes
        
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
            elif recommendations.increase_Tp and input_dto.signal.Tp >= 4900.0:
                # Tp is at maximum, cannot be increased
                optimized_params_text += f"Tp (pulse duration): {input_dto.signal.Tp:.2f} µs (at maximum, cannot increase)\n"
            
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
            current_params_text += f"Measured SNR: {output_dto.SNR_ADC:.2f} dB\n"
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
        
        if recommendations.increase_Tp:
            # Check if this is for large distance
            D_target = input_dto.range.D_target if input_dto.range.D_target is not None else (input_dto.range.D_min + input_dto.range.D_max) / 2
            is_large_distance = D_target > 200.0
            
            if recommendations.aggressive_Tp_increase:
                # More aggressive increase when VGA is at maximum (50-100% increase)
                # But limit to reasonable maximum (e.g., 5000 µs)
                suggested_tp = min(input_dto.signal.Tp * 2.0, 5000.0)  # Double or max 5000 µs
                # Don't suggest if already at or near maximum
                if suggested_tp <= input_dto.signal.Tp:
                    # Already at maximum, don't suggest increase
                    pass
                else:
                    changes['Tp'] = suggested_tp
            else:
                if is_large_distance:
                    # For large distances, more aggressive Tp increase (50% instead of 20%)
                    suggested_tp = min(input_dto.signal.Tp * 1.5, 5000.0)
                else:
                    suggested_tp = input_dto.signal.Tp * 1.2  # Increase by 20%
                
                # Limit to maximum 5000 µs
                if suggested_tp > 5000.0:
                    # If current Tp is already at or near max, don't suggest increase
                    if input_dto.signal.Tp < 4900.0:
                        changes['Tp'] = 5000.0  # Suggest maximum
                    # Otherwise, don't suggest (already at max)
                else:
                    changes['Tp'] = suggested_tp
        
        if recommendations.decrease_Tp:
            changes['Tp'] = input_dto.signal.Tp * 0.8  # Decrease by 20%
        
        if recommendations.increase_f_start:
            changes['f_start'] = input_dto.signal.f_start * 1.1
        
        if recommendations.decrease_f_start:
            changes['f_start'] = input_dto.signal.f_start * 0.9
        
        if recommendations.increase_f_end:
            changes['f_end'] = input_dto.signal.f_end * 1.1
        
        if recommendations.decrease_f_end:
            changes['f_end'] = input_dto.signal.f_end * 0.9
        
        return changes

