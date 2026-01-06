"""
BVD (Butterworth-Van Dyke) Model Calculator
Functions for calculating BVD parameters from admittance data
"""

import numpy as np
from scipy.optimize import minimize


def bvd_admittance(freq, C0, R1, L1, C1):
    """
    Calculate Admittance from BVD model
    
    Args:
        freq: frequency array (Hz)
        C0: static capacitance (F)
        R1: series resistance (Ohm)
        L1: series inductance (H)
        C1: series capacitance (F)
    
    Returns:
        Complex admittance array (S)
    """
    omega = 2 * np.pi * freq
    
    # Series branch impedance
    Z_series = R1 + 1j * (omega * L1 - 1 / (omega * C1))
    
    # Series branch admittance
    Y_series = 1 / Z_series
    
    # C0 admittance
    Y_C0 = 1j * omega * C0
    
    # Total admittance
    Y_total = Y_series + Y_C0
    
    return Y_total


def calculate_bvd_parameters(freq_g, g_values_S, freq_b, b_values_S, C0):
    """
    Calculate BVD parameters from experimental admittance data
    
    Args:
        freq_g: frequency array for conductance (Hz)
        g_values_S: conductance values (S)
        freq_b: frequency array for susceptance (Hz)
        b_values_S: susceptance values (S)
        C0: static capacitance (F)
    
    Returns:
        Dictionary with BVD parameters:
        {
            'C0': C0 (F),
            'fs': series resonant frequency (Hz),
            'fp': parallel resonant frequency (Hz),
            'R1': series resistance (Ohm),
            'L1': series inductance (H),
            'C1': series capacitance (F),
            'Qm': mechanical Q-factor,
            'k': electromechanical coupling coefficient
        }
    """
    # Find resonant frequency from Conductance maximum
    g_max_idx = np.argmax(g_values_S)
    fs_measured = freq_g[g_max_idx]
    g_max = g_values_S[g_max_idx]
    
    # Find antiresonant frequency
    # fp is the frequency of |Y| minimum after resonance
    # Approximately: where Susceptance crosses zero after resonance
    zero_crossings = np.where(np.diff(np.sign(b_values_S)))[0]
    fp_measured = fs_measured
    
    for idx in zero_crossings:
        if freq_b[idx] > fs_measured:
            fp_measured = freq_b[idx]
            break
    
    # If not found, use approximate formula
    if fp_measured == fs_measured:
        # Typical ratio fp/fs ≈ 1.05-1.15 for piezoceramics
        fp_measured = fs_measured * 1.1
    
    # Initial BVD parameters (improved calculation)
    # R1 from peak conductance (more accurate)
    R1_init = 1.0 / (g_max + 1e-10)  # Avoid division by zero
    
    # C1 and L1 from resonance frequencies
    freq_ratio = (fp_measured / fs_measured) ** 2
    if freq_ratio > 1.01:  # Ensure reasonable ratio
        C1_init = C0 / (freq_ratio - 1)
    else:
        # Fallback: use typical ratio for piezoceramics
        C1_init = C0 * 0.1
    
    # Ensure C1 is reasonable
    if C1_init <= 0 or C1_init > C0 * 10:
        C1_init = C0 * 0.1
    
    L1_init = 1.0 / (4 * np.pi**2 * fs_measured**2 * C1_init)
    
    # Ensure L1 is reasonable (typical range: 0.01 mH to 100 mH)
    if L1_init <= 0 or L1_init > 0.1:  # > 100 mH is unusual
        # Recalculate C1 to get reasonable L1
        L1_init = 0.01  # 10 mH as default
        C1_init = 1.0 / (4 * np.pi**2 * fs_measured**2 * L1_init)
    
    # Optimize BVD parameters to fit experimental data
    # Interpolate experimental data to common frequency grid
    freq_min = max(freq_g.min(), freq_b.min())
    freq_max = min(freq_g.max(), freq_b.max())
    freq_common = np.linspace(freq_min, freq_max, min(500, len(freq_g)))
    
    # Interpolate G and B to common frequencies
    g_interp = np.interp(freq_common, freq_g, g_values_S)
    b_interp = np.interp(freq_common, freq_b, b_values_S)
    
    # Find resonance region for weighting (where G is near maximum)
    g_max = np.max(g_interp)
    g_threshold = g_max * 0.5  # Points where G > 50% of max get higher weight
    resonance_weight = np.where(g_interp > g_threshold, 3.0, 1.0)  # 3x weight near resonance
    
    # Optimization function with normalized error and resonance weighting
    def bvd_error(params):
        """Calculate normalized error between model and experimental data"""
        R1_opt, L1_opt, C1_opt = params
        
        # Avoid invalid parameters
        if R1_opt <= 0 or L1_opt <= 0 or C1_opt <= 0:
            return 1e10
        
        # Calculate model admittance
        Y_model = bvd_admittance(freq_common, C0, R1_opt, L1_opt, C1_opt)
        g_model = np.real(Y_model)
        b_model = np.imag(Y_model)
        
        # Use combination of absolute and relative errors for better fit
        # Absolute error (important for small values)
        abs_error_g = np.abs(g_model - g_interp)
        abs_error_b = np.abs(b_model - b_interp)
        
        # Relative error (important for large values)
        g_magnitude = np.abs(g_interp) + 1e-10
        b_magnitude = np.abs(b_interp) + 1e-10
        rel_error_g = abs_error_g / g_magnitude
        rel_error_b = abs_error_b / b_magnitude
        
        # Combined error: weighted combination of absolute and relative
        # Normalize absolute errors by typical values
        g_typical = np.max(g_interp) + 1e-10
        b_typical = np.max(np.abs(b_interp)) + 1e-10
        
        # Weighted errors (resonance region gets 3x weight)
        # Give much higher weight to Conductance to improve its fit
        error_g = np.sum(resonance_weight * (0.5 * (abs_error_g / g_typical)**2 + 0.5 * rel_error_g**2))
        error_b = np.sum((abs_error_b / b_typical)**2 + rel_error_b**2)
        
        # Increase weight for Conductance (5x) since it's fitting poorly
        return 5.0 * error_g + error_b
    
    # Optimize parameters
    try:
        initial_params = [R1_init, L1_init, C1_init]
        # Wider bounds to allow more exploration
        bounds = [
            (R1_init * 0.01, R1_init * 100),  # R1 bounds (wider range)
            (L1_init * 0.01, L1_init * 100),  # L1 bounds (wider range)
            (C1_init * 0.01, C1_init * 100)   # C1 bounds (wider range)
        ]
        
        # Try multiple starting points for better global search
        initial_guesses = [
            initial_params,
            [R1_init * 0.5, L1_init, C1_init],
            [R1_init * 2.0, L1_init, C1_init],
            [R1_init, L1_init * 0.5, C1_init],
            [R1_init, L1_init * 2.0, C1_init],
            [R1_init, L1_init, C1_init * 0.5],
            [R1_init, L1_init, C1_init * 2.0],
        ]
        
        # Try multiple optimization methods and starting points for better results
        best_result = None
        best_error = 1e10
        
        # Try each starting point with L-BFGS-B
        for start_params in initial_guesses:
            try:
                result = minimize(bvd_error, start_params, method='L-BFGS-B', bounds=bounds, 
                                options={'maxiter': 2000, 'ftol': 1e-8})
                if result.success and result.fun < best_error:
                    best_result = result
                    best_error = result.fun
            except:
                continue
        
        # Try SLSQP with best starting point
        if best_result is not None:
            best_start = best_result.x
        else:
            best_start = initial_params
            
        try:
            result2 = minimize(bvd_error, best_start, method='SLSQP', bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-8})
            if result2.success and result2.fun < best_error:
                best_result = result2
                best_error = result2.fun
        except:
            pass
        
        # Try differential evolution for global optimization (slower but more thorough)
        try:
            from scipy.optimize import differential_evolution
            result3 = differential_evolution(bvd_error, bounds, seed=42, 
                                           maxiter=200, popsize=20, atol=1e-8, polish=True)
            if result3.success and result3.fun < best_error:
                best_result = result3
                best_error = result3.fun
        except:
            pass
        
        if best_result is not None:
            R1, L1, C1 = best_result.x
        else:
            # Fallback to initial values
            R1, L1, C1 = R1_init, L1_init, C1_init
    except Exception:
        # Fallback to initial values
        R1, L1, C1 = R1_init, L1_init, C1_init
    
    # Recalculate fs and fp from optimized parameters
    # fs = 1 / (2*pi*sqrt(L1*C1))
    fs_optimized = 1.0 / (2 * np.pi * np.sqrt(L1 * C1))
    
    # fp from admittance minimum (antiresonance)
    # Use wider range and more points for better accuracy
    freq_test = np.linspace(fs_optimized, fs_optimized * 1.5, 1000)
    Y_test = bvd_admittance(freq_test, C0, R1, L1, C1)
    Y_mag_test = np.abs(Y_test)
    
    # Find minimum after fs (antiresonance)
    # Look for minimum in the range after fs
    fs_idx = np.argmin(np.abs(freq_test - fs_optimized))
    if fs_idx < len(Y_mag_test) - 10:
        # Search after fs
        search_range = Y_mag_test[fs_idx:]
        search_freq = freq_test[fs_idx:]
        fp_idx_local = np.argmin(search_range)
        fp_optimized = search_freq[fp_idx_local]
    else:
        # Fallback: use minimum in entire range
        fp_idx = np.argmin(Y_mag_test)
        fp_optimized = freq_test[fp_idx]
    
    # Ensure fp > fs (antiresonance must be after resonance)
    if fp_optimized <= fs_optimized:
        # Use theoretical formula: fp ≈ fs * sqrt(1 + C1/C0)
        if C1 > 0 and C0 > 0:
            fp_optimized = fs_optimized * np.sqrt(1 + C1 / C0)
        else:
            fp_optimized = fs_optimized * 1.1  # Fallback
    
    # Use optimized frequencies if they're reasonable
    if 0.8 * fs_measured < fs_optimized < 1.2 * fs_measured:
        fs_measured = fs_optimized
    if fp_optimized > fs_measured and 0.8 * fp_measured < fp_optimized < 1.5 * fp_measured:
        fp_measured = fp_optimized
    elif fp_measured <= fs_measured:
        # Ensure fp > fs
        fp_measured = fs_measured * 1.1
    
    # Mechanical Q-factor
    omega_s = 2 * np.pi * fs_measured
    Qm = omega_s * L1 / R1
    
    # Electromechanical coupling coefficient
    k = np.sqrt(1 - (fs_measured / fp_measured)**2) if fp_measured > fs_measured else 0.0
    
    return {
        'C0': C0,
        'fs': fs_measured,
        'fp': fp_measured,
        'R1': R1,
        'L1': L1,
        'C1': C1,
        'Qm': Qm,
        'k': k
    }


def calculate_model_curves(freq_model, bvd_params):
    """
    Calculate model curves for plotting
    
    Args:
        freq_model: frequency array for model (Hz)
        bvd_params: dictionary with BVD parameters
    
    Returns:
        Dictionary with model data:
        {
            'freq': frequency array (kHz),
            'g': conductance (mS),
            'b': susceptance (mS),
            'magnitude': |Y| (mS),
            'phase': phase (degrees)
        }
    """
    Y_model = bvd_admittance(
        freq_model,
        bvd_params['C0'],
        bvd_params['R1'],
        bvd_params['L1'],
        bvd_params['C1']
    )
    
    g_model = np.real(Y_model) * 1e3  # S -> mS
    b_model = np.imag(Y_model) * 1e3  # S -> mS
    y_mag_model = np.abs(Y_model) * 1e3  # mS
    y_phase_model = np.angle(Y_model, deg=True)  # degrees
    
    return {
        'freq': freq_model * 1e-3,  # Convert to kHz
        'g': g_model,
        'b': b_model,
        'magnitude': y_mag_model,
        'phase': y_phase_model
    }
