"""
Alternative Transducer Models for Piezoelectric Transducers
Provides more accurate models than basic BVD
"""

import numpy as np
from scipy.optimize import minimize


def mbvd_admittance(freq, C0, R0, R1, L1, C1):
    """
    Calculate Admittance from Modified BVD (MBVD) model
    
    Modified BVD adds loss resistance R0 in parallel with C0
    to account for dielectric losses.
    
    Args:
        freq: frequency array (Hz)
        C0: static capacitance (F)
        R0: parallel loss resistance (Ohm) - accounts for dielectric losses
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
    
    # C0 with parallel loss resistance
    Y_C0 = 1j * omega * C0 + 1 / R0
    
    # Total admittance
    Y_total = Y_series + Y_C0
    
    return Y_total


def ebvd_admittance(freq, C0, R1, L1, C1, R2=None, L2=None, C2=None, R3=None, L3=None, C3=None):
    """
    Calculate Admittance from Extended BVD (EBVD) model
    
    Extended BVD adds additional resonant branches for harmonics.
    
    Args:
        freq: frequency array (Hz)
        C0: static capacitance (F)
        R1, L1, C1: fundamental resonance branch
        R2, L2, C2: first harmonic branch (optional)
        R3, L3, C3: second harmonic branch (optional)
    
    Returns:
        Complex admittance array (S)
    """
    omega = 2 * np.pi * freq
    
    # Fundamental series branch
    Z1 = R1 + 1j * (omega * L1 - 1 / (omega * C1))
    Y_total = 1 / Z1
    
    # First harmonic branch (if provided)
    if R2 is not None and L2 is not None and C2 is not None:
        Z2 = R2 + 1j * (omega * L2 - 1 / (omega * C2))
        Y_total += 1 / Z2
    
    # Second harmonic branch (if provided)
    if R3 is not None and L3 is not None and C3 is not None:
        Z3 = R3 + 1j * (omega * L3 - 1 / (omega * C3))
        Y_total += 1 / Z3
    
    # C0 admittance
    Y_C0 = 1j * omega * C0
    
    # Total admittance
    Y_total += Y_C0
    
    return Y_total


def calculate_mbvd_parameters(freq_g, g_values_S, freq_b, b_values_S, C0):
    """
    Calculate Modified BVD (MBVD) parameters from experimental data
    
    MBVD adds R0 (parallel loss resistance) to account for dielectric losses.
    
    Args:
        freq_g: frequency array for conductance (Hz)
        g_values_S: conductance values (S)
        freq_b: frequency array for susceptance (Hz)
        b_values_S: susceptance values (S)
        C0: static capacitance (F)
    
    Returns:
        Dictionary with MBVD parameters
    """
    # Find resonant frequency from Conductance maximum
    g_max_idx = np.argmax(g_values_S)
    fs_measured = freq_g[g_max_idx]
    g_max = g_values_S[g_max_idx]
    
    # Find antiresonant frequency
    zero_crossings = np.where(np.diff(np.sign(b_values_S)))[0]
    fp_measured = fs_measured
    
    for idx in zero_crossings:
        if freq_b[idx] > fs_measured:
            fp_measured = freq_b[idx]
            break
    
    if fp_measured == fs_measured:
        fp_measured = fs_measured * 1.1
    
    # Initial parameters (similar to BVD)
    R1_init = 1.0 / g_max
    freq_ratio = (fp_measured / fs_measured) ** 2
    C1_init = C0 / (freq_ratio - 1) if freq_ratio > 1 else C0 * 0.1
    L1_init = 1.0 / (4 * np.pi**2 * fs_measured**2 * C1_init)
    
    # Initial R0 - estimate from low frequency conductance
    # At low frequencies, Y ≈ 1/R0 + j*ω*C0
    low_freq_idx = len(g_values_S) // 10  # Use first 10% of data
    g_low = np.mean(g_values_S[:low_freq_idx])
    R0_init = 1.0 / (g_low + 1e-10) if g_low > 0 else 1e6
    
    # Optimize parameters
    freq_min = max(freq_g.min(), freq_b.min())
    freq_max = min(freq_g.max(), freq_b.max())
    freq_common = np.linspace(freq_min, freq_max, min(500, len(freq_g)))
    
    g_interp = np.interp(freq_common, freq_g, g_values_S)
    b_interp = np.interp(freq_common, freq_b, b_values_S)
    
    # Find resonance region for weighting (where G is near maximum)
    g_max = np.max(g_interp)
    g_threshold = g_max * 0.5  # Points where G > 50% of max get higher weight
    resonance_weight = np.where(g_interp > g_threshold, 3.0, 1.0)  # 3x weight near resonance
    
    def mbvd_error(params):
        """Calculate normalized error between MBVD model and experimental data"""
        R0_opt, R1_opt, L1_opt, C1_opt = params
        
        if R0_opt <= 0 or R1_opt <= 0 or L1_opt <= 0 or C1_opt <= 0:
            return 1e10
        
        Y_model = mbvd_admittance(freq_common, C0, R0_opt, R1_opt, L1_opt, C1_opt)
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
    
    try:
        initial_params = [R0_init, R1_init, L1_init, C1_init]
        # Wider bounds to allow more exploration
        bounds = [
            (R0_init * 0.01, R0_init * 1000),  # R0 bounds (very wide range)
            (R1_init * 0.01, R1_init * 100),  # R1 bounds (wider range)
            (L1_init * 0.01, L1_init * 100),  # L1 bounds (wider range)
            (C1_init * 0.01, C1_init * 100)   # C1 bounds (wider range)
        ]
        
        # Try multiple starting points for better global search
        initial_guesses = [
            initial_params,
            [R0_init * 0.5, R1_init, L1_init, C1_init],
            [R0_init * 2.0, R1_init, L1_init, C1_init],
            [R0_init, R1_init * 0.5, L1_init, C1_init],
            [R0_init, R1_init * 2.0, L1_init, C1_init],
            [R0_init, R1_init, L1_init * 0.5, C1_init],
            [R0_init, R1_init, L1_init * 2.0, C1_init],
            [R0_init, R1_init, L1_init, C1_init * 0.5],
            [R0_init, R1_init, L1_init, C1_init * 2.0],
        ]
        
        # Try multiple optimization methods and starting points for better results
        best_result = None
        best_error = 1e10
        
        # Try each starting point with L-BFGS-B
        for start_params in initial_guesses:
            try:
                result = minimize(mbvd_error, start_params, method='L-BFGS-B', bounds=bounds,
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
            result2 = minimize(mbvd_error, best_start, method='SLSQP', bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-8})
            if result2.success and result2.fun < best_error:
                best_result = result2
                best_error = result2.fun
        except:
            pass
        
        # Try differential evolution for global optimization (slower but more thorough)
        try:
            from scipy.optimize import differential_evolution
            result3 = differential_evolution(mbvd_error, bounds, seed=42,
                                           maxiter=200, popsize=20, atol=1e-8, polish=True)
            if result3.success and result3.fun < best_error:
                best_result = result3
                best_error = result3.fun
        except:
            pass
        
        if best_result is not None:
            R0, R1, L1, C1 = best_result.x
        else:
            R0, R1, L1, C1 = R0_init, R1_init, L1_init, C1_init
    except Exception:
        R0, R1, L1, C1 = R0_init, R1_init, L1_init, C1_init
    
    # Recalculate frequencies
    fs_optimized = 1.0 / (2 * np.pi * np.sqrt(L1 * C1))
    
    freq_test = np.linspace(fs_optimized, fs_optimized * 1.2, 200)
    Y_test = mbvd_admittance(freq_test, C0, R0, R1, L1, C1)
    Y_mag_test = np.abs(Y_test)
    fp_idx = np.argmin(Y_mag_test)
    fp_optimized = freq_test[fp_idx]
    
    if 0.8 * fs_measured < fs_optimized < 1.2 * fs_measured:
        fs_measured = fs_optimized
    if 0.8 * fp_measured < fp_optimized < 1.2 * fp_measured:
        fp_measured = fp_optimized
    
    omega_s = 2 * np.pi * fs_measured
    Qm = omega_s * L1 / R1
    
    # Dielectric loss factor
    tan_delta = 1.0 / (omega_s * C0 * R0) if R0 > 0 else 0.0
    
    k = np.sqrt(1 - (fs_measured / fp_measured)**2) if fp_measured > fs_measured else 0.0
    
    return {
        'model_type': 'MBVD',
        'C0': C0,
        'R0': R0,
        'fs': fs_measured,
        'fp': fp_measured,
        'R1': R1,
        'L1': L1,
        'C1': C1,
        'Qm': Qm,
        'k': k,
        'tan_delta': tan_delta  # Dielectric loss tangent
    }


def calculate_model_curves_mbvd(freq_model, mbvd_params):
    """
    Calculate MBVD model curves for plotting
    
    Args:
        freq_model: frequency array for model (Hz)
        mbvd_params: dictionary with MBVD parameters
    
    Returns:
        Dictionary with model data
    """
    Y_model = mbvd_admittance(
        freq_model,
        mbvd_params['C0'],
        mbvd_params['R0'],
        mbvd_params['R1'],
        mbvd_params['L1'],
        mbvd_params['C1']
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
