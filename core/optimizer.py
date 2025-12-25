"""
Optimizer - parameter optimization when constraints are violated.
"""

from typing import Dict, List
from .dto import InputDTO, OutputDTO, RecommendationsDTO


class Optimizer:
    """
    Parameter optimizer.
    
    Analyzes simulation results and generates recommendations
    for parameter changes to satisfy constraints.
    """
    
    def __init__(self):
        """Initialize optimizer."""
        self.constraints = {
            'sigma_D_max': 0.01,  # 1 cm
            'SNR_min': 20.0,  # dB (default, will be overridden by target_snr from input_dto)
            'clipping_allowed': False,
        }
    
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
        messages = []
        
        # Use target_snr from input_dto if available, otherwise use default
        target_snr = input_dto.target_snr if hasattr(input_dto, 'target_snr') and input_dto.target_snr is not None else self.constraints['SNR_min']
        
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
        
        # Check σ_D
        if output_dto.sigma_D > self.constraints['sigma_D_max']:
            messages.append(f"σ_D ({output_dto.sigma_D:.4f} m) exceeds allowed value ({self.constraints['sigma_D_max']} m)")
            # Increase Tp for better resolution
            recommendations.increase_Tp = True
            recommendations_text += "Increase Tp to improve accuracy. "
        
        # Check SNR (use target_snr from input_dto)
        if output_dto.SNR_ADC < target_snr:
            messages.append(f"SNR ({output_dto.SNR_ADC:.2f} dB) is below target ({target_snr:.2f} dB)")
            # Increase VGA gain or pulse duration
            recommendations.increase_G_VGA = True
            recommendations.increase_Tp = True
            recommendations_text += f"Increase VGA gain or Tp to improve SNR (target: {target_snr:.2f} dB). "
        
        # Check clipping
        if output_dto.clipping_flags:
            messages.append("ADC clipping detected")
            recommendations.decrease_G_VGA = True
            recommendations_text += "Decrease VGA gain to eliminate clipping. "
        
        # Check range (with tolerance for measurement uncertainty)
        # Allow small deviations due to measurement uncertainty (typically 1-2% of range)
        range_tolerance = max(0.01 * input_dto.range.D_max, 0.1)  # 1% of D_max or 0.1m, whichever is larger
        
        if output_dto.D_measured < (input_dto.range.D_min - range_tolerance):
            messages.append(f"Measured range ({output_dto.D_measured:.2f} m) is significantly less than minimum ({input_dto.range.D_min} m)")
            recommendations.increase_Tp = True
            recommendations.increase_f_start = True
            recommendations_text += "Increase Tp or f_start to improve minimum range. "
        
        if output_dto.D_measured > (input_dto.range.D_max + range_tolerance):
            messages.append(f"Measured range ({output_dto.D_measured:.2f} m) is significantly greater than maximum ({input_dto.range.D_max} m)")
            recommendations.decrease_Tp = True
            recommendations.decrease_f_end = True
            recommendations_text += "Decrease Tp or f_end to improve maximum range. "
        
        # Calculate suggested parameter changes
        suggested_changes = self.suggest_parameter_changes(input_dto, recommendations)
        
        # Build optimized parameters section
        optimized_params_text = ""
        if suggested_changes:
            optimized_params_text += "\n\n=== OPTIMIZED PARAMETERS ===\n\n"
            optimized_params_text += "Current values -> Suggested values:\n\n"
            
            # Add each optimized parameter with current and suggested values
            if 'Tp' in suggested_changes:
                current = input_dto.signal.Tp
                suggested = suggested_changes['Tp']
                change_pct = ((suggested - current) / current) * 100
                optimized_params_text += f"Tp (pulse duration): {current:.2f} µs -> {suggested:.2f} µs ({change_pct:+.1f}%)\n"
            
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
            
            # Add VGA gain if needed
            if recommendations.increase_G_VGA or recommendations.decrease_G_VGA:
                if recommendations.increase_G_VGA:
                    optimized_params_text += "\nVGA gain: Increase (adjust manually in GUI)\n"
                elif recommendations.decrease_G_VGA:
                    optimized_params_text += "\nVGA gain: Decrease (adjust manually in GUI)\n"
            
            optimized_params_text += "\n"
        
        # Finalize message: help text + recommendations + optimized parameters
        if recommendations_text.strip():
            # Has recommendations - show recommendations and optimized parameters
            recommendations.message = help_text + recommendations_text.strip() + optimized_params_text
        elif optimized_params_text.strip():
            # Only optimized parameters (shouldn't happen, but just in case)
            recommendations.message = help_text + optimized_params_text
        else:
            # No recommendations - show current parameter values for reference
            current_params_text = "\n\n=== CURRENT PARAMETERS ===\n\n"
            current_params_text += f"Tp (pulse duration): {input_dto.signal.Tp:.2f} µs\n"
            current_params_text += f"f_start (start frequency): {input_dto.signal.f_start:.2f} Hz\n"
            current_params_text += f"f_end (end frequency): {input_dto.signal.f_end:.2f} Hz\n"
            current_params_text += f"Target SNR: {target_snr:.1f} dB\n"
            current_params_text += f"Measured SNR: {output_dto.SNR_ADC:.2f} dB\n"
            current_params_text += f"Range accuracy (σ_D): {output_dto.sigma_D:.4f} m\n"
            current_params_text += f"Measured range: {output_dto.D_measured:.2f} m\n"
            recommendations.message = help_text + "\n\nNo issues detected. All constraints are satisfied." + current_params_text
        
        # If there are problems, mark as unsuccessful simulation
        if messages:
            output_dto.success = False
            output_dto.errors.extend(messages)
        else:
            output_dto.success = True
        
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
            changes['Tp'] = input_dto.signal.Tp * 1.2  # Increase by 20%
        
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

