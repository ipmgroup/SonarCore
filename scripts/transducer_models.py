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


def calculate_ebvd_parameters(freq_g, g_values_S, freq_b, b_values_S, C0, use_harmonic=True):
    """
    Calculate Extended BVD (EBVD) parameters from experimental data
    
    EBVD adds harmonic branches (R2, L2, C2) for better fit of complex resonances.
    
    Args:
        freq_g: frequency array for conductance (Hz)
        g_values_S: conductance values (S)
        freq_b: frequency array for susceptance (Hz)
        b_values_S: susceptance values (S)
        C0: static capacitance (F)
        use_harmonic: if True, include first harmonic branch (R2, L2, C2)
    
    Returns:
        Dictionary with EBVD parameters
    """
    # First, calculate MBVD parameters as starting point
    mbvd_params = calculate_mbvd_parameters(freq_g, g_values_S, freq_b, b_values_S, C0)
    
    # Extract fundamental parameters
    R0 = mbvd_params.get('R0', 1e6)
    R1 = mbvd_params['R1']
    L1 = mbvd_params['L1']
    C1 = mbvd_params['C1']
    fs_measured = mbvd_params['fs']
    fp_measured = mbvd_params['fp']
    
    if not use_harmonic:
        # Return MBVD-like structure but marked as EBVD
        return {
            'model_type': 'EBVD',
            'C0': C0,
            'R0': R0,
            'fs': fs_measured,
            'fp': fp_measured,
            'R1': R1,
            'L1': L1,
            'C1': C1,
            'Qm': mbvd_params['Qm'],
            'k': mbvd_params['k'],
            'tan_delta': mbvd_params.get('tan_delta', 0.0)
        }
    
    # Estimate harmonic parameters (first harmonic at ~2*fs)
    fs_harmonic = fs_measured * 2.0  # First harmonic typically at 2x fundamental
    
    # Initial harmonic parameters (scaled from fundamental)
    R2_init = R1 * 2.0  # Harmonic typically has higher resistance
    C2_init = C1 * 0.25  # Harmonic capacitance typically smaller
    L2_init = 1.0 / (4 * np.pi**2 * fs_harmonic**2 * C2_init)
    
    # Optimize parameters
    freq_min = max(freq_g.min(), freq_b.min())
    freq_max = min(freq_g.max(), freq_b.max())
    freq_common = np.linspace(freq_min, freq_max, min(500, len(freq_g)))
    
    g_interp = np.interp(freq_common, freq_g, g_values_S)
    b_interp = np.interp(freq_common, freq_b, b_values_S)
    
    # Find resonance region for weighting
    g_max = np.max(g_interp)
    g_threshold = g_max * 0.5
    resonance_weight = np.where(g_interp > g_threshold, 3.0, 1.0)
    
    def ebvd_error(params):
        """Calculate normalized error between EBVD model and experimental data"""
        R0_opt, R1_opt, L1_opt, C1_opt, R2_opt, L2_opt, C2_opt = params
        
        if (R0_opt <= 0 or R1_opt <= 0 or L1_opt <= 0 or C1_opt <= 0 or
            R2_opt <= 0 or L2_opt <= 0 or C2_opt <= 0):
            return 1e10
        
        Y_model = ebvd_admittance(freq_common, C0, R1_opt, L1_opt, C1_opt,
                                  R2_opt, L2_opt, C2_opt)
        g_model = np.real(Y_model)
        b_model = np.imag(Y_model)
        
        # Use combination of absolute and relative errors
        abs_error_g = np.abs(g_model - g_interp)
        abs_error_b = np.abs(b_model - b_interp)
        
        g_magnitude = np.abs(g_interp) + 1e-10
        b_magnitude = np.abs(b_interp) + 1e-10
        rel_error_g = abs_error_g / g_magnitude
        rel_error_b = abs_error_b / b_magnitude
        
        g_typical = np.max(g_interp) + 1e-10
        b_typical = np.max(np.abs(b_interp)) + 1e-10
        
        # Weighted errors (resonance region gets 3x weight)
        error_g = np.sum(resonance_weight * (0.5 * (abs_error_g / g_typical)**2 + 0.5 * rel_error_g**2))
        error_b = np.sum((abs_error_b / b_typical)**2 + rel_error_b**2)
        
        return 5.0 * error_g + error_b
    
    try:
        initial_params = [R0, R1, L1, C1, R2_init, L2_init, C2_init]
        bounds = [
            (R0 * 0.1, R0 * 1000),
            (R1 * 0.01, R1 * 100),
            (L1 * 0.01, L1 * 100),
            (C1 * 0.01, C1 * 100),
            (R2_init * 0.01, R2_init * 100),
            (L2_init * 0.01, L2_init * 100),
            (C2_init * 0.01, C2_init * 100)
        ]
        
        # Try multiple optimization methods
        best_result = None
        best_error = 1e10
        
        # Try L-BFGS-B
        try:
            result = minimize(ebvd_error, initial_params, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 2000, 'ftol': 1e-8})
            if result.success and result.fun < best_error:
                best_result = result
                best_error = result.fun
        except:
            pass
        
        # Try SLSQP
        try:
            result2 = minimize(ebvd_error, initial_params, method='SLSQP', bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-8})
            if result2.success and result2.fun < best_error:
                best_result = result2
                best_error = result2.fun
        except:
            pass
        
        if best_result is not None:
            R0, R1, L1, C1, R2, L2, C2 = best_result.x
        else:
            # Fallback to initial values
            R2, L2, C2 = R2_init, L2_init, C2_init
    except Exception:
        R2, L2, C2 = R2_init, L2_init, C2_init
    
    # Recalculate frequencies
    fs_optimized = 1.0 / (2 * np.pi * np.sqrt(L1 * C1))
    
    freq_test = np.linspace(fs_optimized, fs_optimized * 1.5, 1000)
    Y_test = ebvd_admittance(freq_test, C0, R1, L1, C1, R2, L2, C2)
    Y_mag_test = np.abs(Y_test)
    
    fs_idx = np.argmin(np.abs(freq_test - fs_optimized))
    if fs_idx < len(Y_mag_test) - 10:
        search_range = Y_mag_test[fs_idx:]
        search_freq = freq_test[fs_idx:]
        fp_idx_local = np.argmin(search_range)
        fp_optimized = search_freq[fp_idx_local]
    else:
        fp_idx = np.argmin(Y_mag_test)
        fp_optimized = freq_test[fp_idx]
    
    if fp_optimized <= fs_optimized:
        if C1 > 0 and C0 > 0:
            fp_optimized = fs_optimized * np.sqrt(1 + C1 / C0)
        else:
            fp_optimized = fs_optimized * 1.1
    
    if 0.8 * fs_measured < fs_optimized < 1.2 * fs_measured:
        fs_measured = fs_optimized
    if fp_optimized > fs_measured and 0.8 * fp_measured < fp_optimized < 1.5 * fp_measured:
        fp_measured = fp_optimized
    elif fp_measured <= fs_measured:
        fp_measured = fs_measured * 1.1
    
    omega_s = 2 * np.pi * fs_measured
    Qm = omega_s * L1 / R1
    tan_delta = 1.0 / (omega_s * C0 * R0) if R0 > 0 else 0.0
    k = np.sqrt(1 - (fs_measured / fp_measured)**2) if fp_measured > fs_measured else 0.0
    
    return {
        'model_type': 'EBVD',
        'C0': C0,
        'R0': R0,
        'fs': fs_measured,
        'fp': fp_measured,
        'R1': R1,
        'L1': L1,
        'C1': C1,
        'R2': R2,
        'L2': L2,
        'C2': C2,
        'Qm': Qm,
        'k': k,
        'tan_delta': tan_delta
    }


def calculate_model_curves_ebvd(freq_model, ebvd_params):
    """
    Calculate EBVD model curves for plotting
    
    Args:
        freq_model: frequency array for model (Hz)
        ebvd_params: dictionary with EBVD parameters
    
    Returns:
        Dictionary with model data
    """
    # Check if harmonic parameters are present
    if 'R2' in ebvd_params and 'L2' in ebvd_params and 'C2' in ebvd_params:
        Y_model = ebvd_admittance(
            freq_model,
            ebvd_params['C0'],
            ebvd_params['R1'],
            ebvd_params['L1'],
            ebvd_params['C1'],
            ebvd_params['R2'],
            ebvd_params['L2'],
            ebvd_params['C2']
        )
    else:
        # Fallback to MBVD if no harmonics (use mbvd_admittance directly)
        Y_model = mbvd_admittance(
            freq_model,
            ebvd_params['C0'],
            ebvd_params.get('R0', 1e6),
            ebvd_params['R1'],
            ebvd_params['L1'],
            ebvd_params['C1']
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


def mason_admittance(freq, C0, k_t, Z_a, t, A, alpha=0.0, R_m=None):
    """
    Calculate Admittance from Mason Model
    
    Mason model is a physical model based on acoustic wave propagation.
    It describes the piezoelectric element as an acoustic transmission line.
    
    Args:
        freq: frequency array (Hz)
        C0: static capacitance (F) - electrical capacitance at low frequency
        k_t: electromechanical coupling coefficient (thickness mode)
        Z_a: acoustic impedance (kg/(m²·s)) = ρ·c where ρ is density, c is sound speed
        t: thickness of piezoelectric element (m)
        A: area of piezoelectric element (m²)
        alpha: acoustic attenuation coefficient (Np/m), optional
        R_m: mechanical loss resistance (Ohm), optional - adds real part to admittance
    
    Returns:
        Complex admittance array (S)
    """
    omega = 2 * np.pi * freq
    
    # Acoustic wave number
    # Estimate sound speed from acoustic impedance (Z_a = ρ·c)
    # Approximate density for common materials
    if Z_a < 1e7:
        rho_est = 2650  # Typical for ceramics
    else:
        rho_est = 7800  # Typical for PZT
    
    c = Z_a / rho_est  # Sound speed estimate
    k = omega / c  # Wave number
    
    # Complex wave number with attenuation
    k_complex = k - 1j * alpha
    
    # Mason model admittance (simplified for thickness mode)
    # Y = j*omega*C0 * [1 - k_t² * tan(kl/2) / (kl/2)]
    
    kl_half = k_complex * t / 2.0
    
    # Avoid division by zero and handle large values
    kl_half_safe = np.where(np.abs(kl_half) < 1e-10, 1e-10, kl_half)
    
    # Mason model admittance
    # Use tan(x)/x approximation for small x to avoid numerical issues
    tan_term = np.where(
        np.abs(kl_half_safe) < 0.1,
        1.0 + (kl_half_safe**2) / 3.0 + 2 * (kl_half_safe**4) / 15.0,  # Taylor expansion
        np.tan(kl_half_safe) / kl_half_safe
    )
    
    # Main Mason admittance
    Y_mason = 1j * omega * C0 * (1 - k_t**2 * tan_term)
    
    # Add mechanical losses if provided
    # Mechanical losses add real part to admittance, especially near resonance
    # Use more accurate model based on equivalent BVD R1
    if R_m is not None and R_m > 0:
        # Frequency-dependent mechanical loss contribution
        # Peak at resonance frequency
        fs_est = c / (2 * t) if t > 0 else 25000
        omega_s = 2 * np.pi * fs_est
        freq_ratio = omega / omega_s
        
        # More accurate mechanical loss model
        # Use equivalent mechanical L and C for resonance
        # Estimate mechanical inductance and capacitance from acoustic parameters
        L_m_est = rho_est * t * A / (c**2) if c > 0 and A > 0 else 1e-3
        C_m_est = 1.0 / (omega_s**2 * L_m_est) if L_m_est > 0 and omega_s > 0 else 1e-9
        
        # Mechanical impedance: Z_m = R_m + j*(ω*L_m - 1/(ω*C_m))
        X_m = omega * L_m_est - 1.0 / (omega * C_m_est + 1e-12)
        Z_m = R_m + 1j * X_m
        
        # Mechanical admittance (real part contributes to Conductance)
        Y_m = 1.0 / (Z_m + 1e-12)
        G_m = np.real(Y_m)
        
        # Scale by coupling coefficient for proper contribution
        Y_mason += G_m * k_t**2
    
    return Y_mason


def calculate_mason_parameters(freq_g, g_values_S, freq_b, b_values_S, C0, 
                                t=None, A=None, rho=None, c=None):
    """
    Calculate Mason Model parameters from experimental data
    
    Mason model requires physical parameters. This function:
    1. Estimates physical parameters from BVD/MBVD fit
    2. Optimizes to match experimental data
    
    Args:
        freq_g: frequency array for conductance (Hz)
        g_values_S: conductance values (S)
        freq_b: frequency array for susceptance (Hz)
        b_values_S: susceptance values (S)
        C0: static capacitance (F)
        t: thickness (m), optional - will be estimated if not provided
        A: area (m²), optional - will be estimated if not provided
        rho: density (kg/m³), optional - default ~7800 for PZT
        c: sound speed (m/s), optional - default ~4000 for PZT
    
    Returns:
        Dictionary with Mason parameters
    """
    # First, get MBVD parameters as starting point
    mbvd_params = calculate_mbvd_parameters(freq_g, g_values_S, freq_b, b_values_S, C0)
    
    # Extract fundamental parameters
    R0 = mbvd_params.get('R0', 1e6)
    R1 = mbvd_params['R1']
    L1 = mbvd_params['L1']
    C1 = mbvd_params['C1']
    fs_measured = mbvd_params['fs']
    fp_measured = mbvd_params['fp']
    k = mbvd_params['k']
    
    # Default material properties (PZT-5A typical values)
    if rho is None:
        rho = 7800  # kg/m³
    if c is None:
        c = 4000  # m/s (typical for PZT)
    
    # Estimate thickness from resonance frequency
    if t is None:
        t = c / (2 * fs_measured)  # Half-wavelength resonance
    
    # Estimate area from capacitance
    # C0 = ε_r * ε_0 * A / t
    epsilon_r = 2000  # Typical value for PZT
    epsilon_0 = 8.854e-12  # F/m
    if A is None:
        A = C0 * t / (epsilon_r * epsilon_0)
    
    # Sanity check: ensure reasonable physical dimensions
    # Typical PZT thickness: 0.1-50 mm, area: 1-10000 mm²
    t_max = 50e-3  # 50 mm maximum
    t_min = 0.1e-3  # 0.1 mm minimum
    if t > t_max:
        t = t_max
    if t < t_min:
        t = t_min
    
    A_max = 10000e-6  # 10000 mm² maximum
    A_min = 1e-6  # 1 mm² minimum
    if A > A_max:
        A = A_max
        # Recalculate t to maintain C0
        t = C0 * A / (epsilon_r * epsilon_0)
    if A < A_min:
        A = A_min
        t = C0 * A / (epsilon_r * epsilon_0)
    
    # Acoustic impedance
    Z_a = rho * c  # kg/(m²·s)
    
    # Electromechanical coupling coefficient
    k_t = k  # Use from MBVD fit
    
    # Optimize parameters
    freq_min = max(freq_g.min(), freq_b.min())
    freq_max = min(freq_g.max(), freq_b.max())
    freq_common = np.linspace(freq_min, freq_max, min(500, len(freq_g)))
    
    g_interp = np.interp(freq_common, freq_g, g_values_S)
    b_interp = np.interp(freq_common, freq_b, b_values_S)
    
    # Find resonance region for weighting
    g_max = np.max(g_interp)
    g_threshold = g_max * 0.5
    resonance_weight = np.where(g_interp > g_threshold, 3.0, 1.0)
    
    # Define reasonable physical limits
    t_min = 0.1e-3  # 0.1 mm
    t_max = 50e-3   # 50 mm
    A_min = 1e-6    # 1 mm²
    A_max = 10000e-6  # 10000 mm²
    
    def mason_error(params):
        """Calculate normalized error between Mason model and experimental data"""
        k_t_opt, Z_a_opt, t_opt, alpha_opt, R_m_opt = params
        
        if (k_t_opt <= 0 or k_t_opt >= 1 or Z_a_opt <= 0 or 
            t_opt < t_min or t_opt > t_max or alpha_opt < 0 or R_m_opt <= 0):
            return 1e10
        
        # Recalculate A to maintain C0, but constrain to reasonable values
        A_opt = C0 * t_opt / (epsilon_r * epsilon_0)
        
        # Constrain A to reasonable physical values
        if A_opt > A_max:
            A_opt = A_max
            # Recalculate t to maintain C0
            t_opt = C0 * A_opt / (epsilon_r * epsilon_0)
            # Check if t is still in bounds
            if t_opt < t_min or t_opt > t_max:
                return 1e10
        elif A_opt < A_min:
            A_opt = A_min
            t_opt = C0 * A_opt / (epsilon_r * epsilon_0)
            if t_opt < t_min or t_opt > t_max:
                return 1e10
        
        Y_model = mason_admittance(freq_common, C0, k_t_opt, Z_a_opt, t_opt, A_opt, alpha_opt, R_m_opt)
        
        # Add dielectric losses (R0 in parallel)
        if R0 > 0:
            Y_model += 1.0 / R0
        
        # Hybrid approach: add MBVD series branch for better Conductance fit
        # This combines Mason structure with MBVD loss modeling
        if R1 > 0 and L1 > 0 and C1 > 0:
            omega = 2 * np.pi * freq_common
            Z_series = R1 + 1j * (omega * L1 - 1 / (omega * C1))
            Y_series = 1 / (Z_series + 1e-12)
            G_mbvd = np.real(Y_series)
            # Blend: use MBVD losses (30%) with Mason structure (70%)
            # This improves Conductance fit while keeping Mason physics
            Y_model = Y_model + 0.3 * G_mbvd
        
        g_model = np.real(Y_model)
        b_model = np.imag(Y_model)
        
        # Use combination of absolute and relative errors
        abs_error_g = np.abs(g_model - g_interp)
        abs_error_b = np.abs(b_model - b_interp)
        
        g_magnitude = np.abs(g_interp) + 1e-10
        b_magnitude = np.abs(b_interp) + 1e-10
        rel_error_g = abs_error_g / g_magnitude
        rel_error_b = abs_error_b / b_magnitude
        
        g_typical = np.max(g_interp) + 1e-10
        b_typical = np.max(np.abs(b_interp)) + 1e-10
        
        # Weighted errors
        error_g = np.sum(resonance_weight * (0.5 * (abs_error_g / g_typical)**2 + 0.5 * rel_error_g**2))
        error_b = np.sum((abs_error_b / b_typical)**2 + rel_error_b**2)
        
        return 5.0 * error_g + error_b
    
    try:
        # Initial parameters
        k_t_init = k
        Z_a_init = Z_a
        t_init = t
        alpha_init = 0.0  # Start with no attenuation
        # Estimate R_m from MBVD R1 (mechanical losses)
        # R_m represents mechanical loss resistance
        R_m_init = R1 if R1 > 0 else 1000.0
        
        initial_params = [k_t_init, Z_a_init, t_init, alpha_init, R_m_init]
        # Reasonable bounds for physical parameters
        t_min = 0.1e-3  # 0.1 mm
        t_max = 50e-3   # 50 mm
        bounds = [
            (k_t_init * 0.5, min(0.99, k_t_init * 1.5)),  # k_t < 1
            (Z_a_init * 0.1, Z_a_init * 10),
            (max(t_min, t_init * 0.1), min(t_max, t_init * 10)),  # Constrain thickness
            (0.0, 100.0),  # Attenuation in Np/m
            (R_m_init * 0.01, R_m_init * 100)  # Mechanical loss resistance
        ]
        
        # Try optimization
        best_result = None
        best_error = 1e10
        
        try:
            result = minimize(mason_error, initial_params, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 2000, 'ftol': 1e-8})
            if result.success and result.fun < best_error:
                best_result = result
                best_error = result.fun
        except:
            pass
        
        try:
            result2 = minimize(mason_error, initial_params, method='SLSQP', bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-8})
            if result2.success and result2.fun < best_error:
                best_result = result2
                best_error = result2.fun
        except:
            pass
        
        if best_result is not None:
            k_t_opt, Z_a_opt, t_opt, alpha_opt, R_m_opt = best_result.x
        else:
            k_t_opt, Z_a_opt, t_opt, alpha_opt, R_m_opt = k_t_init, Z_a_init, t_init, alpha_init, R_m_init
    except Exception:
        k_t_opt, Z_a_opt, t_opt, alpha_opt, R_m_opt = k_t_init, Z_a_init, t_init, alpha_init, R_m_init
    
    # Recalculate A
    A_opt = C0 * t_opt / (epsilon_r * epsilon_0)
    
    # Calculate equivalent frequencies
    c_opt = Z_a_opt / rho
    fs_mason = c_opt / (2 * t_opt) if t_opt > 0 else fs_measured
    fp_mason = fs_mason / np.sqrt(1 - k_t_opt**2) if k_t_opt < 1 else fs_mason * 1.1
    
    # Use measured frequencies if reasonable
    if 0.8 * fs_measured < fs_mason < 1.2 * fs_measured:
        fs_measured = fs_mason
    if fp_mason > fs_measured and 0.8 * fp_measured < fp_mason < 1.5 * fp_measured:
        fp_measured = fp_mason
    
    omega_s = 2 * np.pi * fs_measured
    Qm = omega_s * L1 / R1 if R1 > 0 else 10.0
    
    return {
        'model_type': 'Mason',
        'C0': C0,
        'R0': R0,
        'k_t': k_t_opt,
        'Z_a': Z_a_opt,
        't': t_opt,
        'A': A_opt,
        'rho': rho,
        'c': c_opt,
        'alpha': alpha_opt,
        'R_m': R_m_opt,  # Mechanical loss resistance
        'fs': fs_measured,
        'fp': fp_measured,
        'R1': R1,
        'L1': L1,
        'C1': C1,
        'Qm': Qm,
        'k': k_t_opt,
        'tan_delta': mbvd_params.get('tan_delta', 0.0)
    }


def calculate_model_curves_mason(freq_model, mason_params):
    """
    Calculate Mason model curves for plotting
    
    Args:
        freq_model: frequency array for model (Hz)
        mason_params: dictionary with Mason parameters
    
    Returns:
        Dictionary with model data
    """
    # Use Mason admittance function
    Y_model = mason_admittance(
        freq_model,
        mason_params['C0'],
        mason_params['k_t'],
        mason_params['Z_a'],
        mason_params['t'],
        mason_params['A'],
        mason_params.get('alpha', 0.0),
        mason_params.get('R_m', None)
    )
    
    # Add dielectric losses (R0 in parallel)
    if 'R0' in mason_params and mason_params['R0'] > 0:
        Y_model += 1.0 / mason_params['R0']
    
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
