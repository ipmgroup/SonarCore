"""
Helper functions for optimization strategies.
"""
import numpy as np
from typing import Tuple
from .dto import InputDTO, OutputDTO, RecommendationsDTO


def apply_max_tp_min_vga_strategy(
    input_dto: InputDTO,
    output_dto: OutputDTO,
    target_snr: float,
    recommendations: RecommendationsDTO,
    recommendations_text: str,
    data_provider,
    calculator
) -> Tuple[float, str]:
    """
    Strategy 1: Use maximum Tp (80% TOF for D_target), then find minimum VGA Gain.
    
    Returns:
        tuple: (optimal_tp, updated_recommendations_text)
    """
    D_target = input_dto.range.D_target if input_dto.range.D_target is not None else (input_dto.range.D_min + input_dto.range.D_max) / 2
    
    # Calculate maximum Tp (80% of round-trip time for D_target)
    try:
        Tp_max_for_D_target = calculator.calculate_min_pulse_duration(
            D_target,
            input_dto.environment.T,
            input_dto.environment.S,
            input_dto.environment.z
        )
        optimal_tp = Tp_max_for_D_target
        
        # Check if current Tp is less than maximum
        if input_dto.signal.Tp < optimal_tp:
            # Recommend increasing Tp to maximum
            recommendations.increase_Tp = True
            recommendations._optimal_tp_for_snr = optimal_tp
            recommendations_text += f"Strategy 1 (Max Tp + Min VGA): Set Tp to maximum (80% TOF for D_target = {optimal_tp:.1f} µs) to maximize signal energy. "
            recommendations_text += f"Current Tp ({input_dto.signal.Tp:.1f} µs) < maximum ({optimal_tp:.1f} µs). "
        else:
            # Tp is already at or above maximum - use current Tp
            optimal_tp = input_dto.signal.Tp
        
        return optimal_tp, recommendations_text
    except Exception:
        return input_dto.signal.Tp, recommendations_text


def apply_min_tp_for_snr_strategy(
    input_dto: InputDTO,
    output_dto: OutputDTO,
    target_snr: float,
    recommendations: RecommendationsDTO,
    recommendations_text: str,
    data_provider,
    calculator
) -> Tuple[float, str]:
    """
    Strategy 2: Find minimum Tp needed to achieve target SNR at D_target.
    
    Returns:
        tuple: (optimal_tp, updated_recommendations_text)
    """
    D_target = input_dto.range.D_target if input_dto.range.D_target is not None else (input_dto.range.D_min + input_dto.range.D_max) / 2
    
    try:
        # Get transducer and hardware parameters
        transducer_params = None
        lna_params = None
        vga_params = None
        if data_provider:
            transducer_params = data_provider.get_transducer(input_dto.hardware.transducer_id)
            lna_params = data_provider.get_lna(input_dto.hardware.lna_id)
            vga_params = data_provider.get_vga(input_dto.hardware.vga_id)
        
        if transducer_params:
            hardware_params = {
                'lna_gain': lna_params.get('G_LNA', lna_params.get('G', 20)) if lna_params else 20.0,
                'vga_gain': output_dto.vga_gain if output_dto.vga_gain is not None else (vga_params.get('G', 30) if vga_params else 30.0),
                'lna_nf': lna_params.get('NF_LNA', lna_params.get('NF', 2.0)) if lna_params else 2.0,
                'bottom_reflection': input_dto.environment.bottom_reflection
            }
            
            tx_voltage = transducer_params.get('V_max', transducer_params.get('V_nominal', 100.0))
            
            # Calculate minimum Tp for target SNR
            optimal_tp = calculator.calculate_optimal_tp_for_snr(
                D_target=D_target,
                target_snr_db=target_snr,
                transducer_params=transducer_params,
                hardware_params=hardware_params,
                T=input_dto.environment.T,
                S=input_dto.environment.S,
                z=input_dto.environment.z,
                tx_voltage=tx_voltage
            )
            
            # Apply physical constraints
            Tp_max_physical_D_min = calculator.calculate_min_pulse_duration(
                input_dto.range.D_min,
                input_dto.environment.T,
                input_dto.environment.S,
                input_dto.environment.z
            )
            Tp_max_physical_D_target = calculator.calculate_min_pulse_duration(
                D_target,
                input_dto.environment.T,
                input_dto.environment.S,
                input_dto.environment.z
            )
            
            # Ensure optimal_tp is within physical limits
            if optimal_tp > Tp_max_physical_D_target:
                optimal_tp = Tp_max_physical_D_target
                recommendations_text += f"WARNING: Calculated Tp exceeds maximum (80% TOF for D_target). Using maximum: {optimal_tp:.1f} µs. "
            
            if optimal_tp < Tp_max_physical_D_min:
                # For D_min, Tp cannot exceed Tp_max_physical_D_min
                # But for D_target, we can use larger Tp
                if D_target > input_dto.range.D_min:
                    # D_target > D_min, so we can use larger Tp
                    pass
                else:
                    optimal_tp = Tp_max_physical_D_min
                    recommendations_text += f"WARNING: Calculated Tp below minimum (80% TOF for D_min). Using minimum: {optimal_tp:.1f} µs. "
            
            return optimal_tp, recommendations_text
        else:
            return input_dto.signal.Tp, recommendations_text
    except Exception:
        return input_dto.signal.Tp, recommendations_text

