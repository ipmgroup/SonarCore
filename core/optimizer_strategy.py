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
        # DEBUG: Log input parameters
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"apply_max_tp_min_vga_strategy: D_target={D_target:.2f}m, T={input_dto.environment.T:.1f}°C, S={input_dto.environment.S:.1f}PSU, z={input_dto.environment.z:.1f}m")
        logger.info(f"apply_max_tp_min_vga_strategy: current Tp={input_dto.signal.Tp:.2f} µs")
        
        Tp_max_for_D_target = calculator.calculate_optimal_pulse_duration(
            D_target,
            input_dto.environment.T,
            input_dto.environment.S,
            input_dto.environment.z,
            min_tp=None
        )
        
        logger.info(f"apply_max_tp_min_vga_strategy: calculated Tp_max_for_D_target={Tp_max_for_D_target:.2f} µs")
        
        # Safety check: ensure Tp_max is reasonable (not NaN or infinite)
        if not np.isfinite(Tp_max_for_D_target) or Tp_max_for_D_target <= 0:
            # Fallback to current Tp if calculation failed
            logger.error(f"apply_max_tp_min_vga_strategy: Invalid Tp_max_for_D_target={Tp_max_for_D_target}, using current Tp")
            return input_dto.signal.Tp, recommendations_text + "WARNING: Failed to calculate maximum Tp, keeping current value. "
        
        # Additional check: Tp_max should be reasonable for given distance
        # For D_target in meters, Tp_max should be approximately: 0.8 * (2 * D_target / 1500) * 1e6 µs
        # Rough check: Tp_max should be between 1000 * D_target / 1500 and 2000 * D_target / 1500 µs
        expected_tp_min = (D_target / 1500.0) * 0.8 * 1e6 * 0.5  # At least 50% of expected
        expected_tp_max = (D_target / 1500.0) * 0.8 * 1e6 * 2.0  # At most 200% of expected
        if Tp_max_for_D_target < expected_tp_min or Tp_max_for_D_target > expected_tp_max:
            logger.warning(f"apply_max_tp_min_vga_strategy: Tp_max_for_D_target={Tp_max_for_D_target:.2f} µs seems unreasonable for D_target={D_target:.2f}m (expected range: {expected_tp_min:.2f}-{expected_tp_max:.2f} µs)")
        
        optimal_tp = Tp_max_for_D_target
        
        # Tolerance for comparison: 2% or 50 µs, whichever is larger
        # This prevents recommending Tp increase when current Tp is already close to maximum
        # Increased tolerance to handle rounding errors and prevent infinite optimization loops
        tolerance = max(optimal_tp * 0.02, 50.0)
        
        # Check if current Tp is significantly less than maximum
        # Also check if current Tp is already at or above maximum (with tolerance)
        current_tp = input_dto.signal.Tp
        
        if current_tp >= optimal_tp - tolerance:
            # Tp is already at or close to maximum (or even above) - use current Tp
            optimal_tp = current_tp
            # Explicitly clear increase_Tp flag since Tp is already at maximum
            recommendations.increase_Tp = False
            # Clear _optimal_tp_for_snr to prevent suggesting changes
            if hasattr(recommendations, '_optimal_tp_for_snr'):
                recommendations._optimal_tp_for_snr = None
            recommendations_text += f"Strategy 1 (Max Tp + Min VGA): Current Tp ({current_tp:.1f} µs) is already at or close to maximum ({Tp_max_for_D_target:.1f} µs, tolerance: {tolerance:.1f} µs). "
            recommendations_text += f"No Tp increase needed. "
        elif current_tp < optimal_tp - tolerance:
            # Current Tp is significantly less than maximum - recommend gradual increase
            # Use same logic as fallback: 20% for normal distances, 50% for large distances (>200m)
            # This prevents huge jumps and allows optimization to converge gradually
            is_large_distance = D_target > 200.0
            if is_large_distance:
                recommended_tp_increase = current_tp * 1.5  # 50% increase for large distances
                increase_pct = 50
            else:
                recommended_tp_increase = current_tp * 1.2  # 20% increase for normal distances
                increase_pct = 20
            
            # Don't exceed maximum Tp
            if recommended_tp_increase > optimal_tp:
                recommended_tp_increase = optimal_tp
                increase_pct = ((recommended_tp_increase - current_tp) / current_tp) * 100  # Recalculate actual percentage
            
            recommendations.increase_Tp = True
            recommendations._optimal_tp_for_snr = recommended_tp_increase
            logger.info(f"apply_max_tp_min_vga_strategy: Setting gradual increase: current_tp={current_tp:.2f} µs, recommended_tp_increase={recommended_tp_increase:.2f} µs ({increase_pct:.1f}%), optimal_tp={optimal_tp:.2f} µs")
            recommendations_text += f"Strategy 1 (Max Tp + Min VGA): Increase Tp gradually ({increase_pct:.1f}% increase) to maximize signal energy. "
            recommendations_text += f"Current Tp ({current_tp:.1f} µs) -> suggested ({recommended_tp_increase:.1f} µs), maximum possible ({optimal_tp:.1f} µs). "
            recommendations_text += f"\n\n⚠️ IMPORTANT: Strategy 1 will continue to suggest Tp increases until maximum is reached. "
            recommendations_text += f"Do not run optimization repeatedly - apply recommendations once and let optimization converge gradually over multiple iterations. "
            recommendations_text += f"Running optimization again immediately will suggest another increase, as Tp is still below maximum. "
        
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
            Tp_max_physical_D_target = calculator.calculate_optimal_pulse_duration(
                D_target,
                input_dto.environment.T,
                input_dto.environment.S,
                input_dto.environment.z,
                min_tp=None
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

