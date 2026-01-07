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
    R2_init = R1 * 2.0  # First harmonic typically has higher resistance
    C2_init = C1 * 0.25  # First harmonic capacitance typically smaller
    L2_init = 1.0 / (4 * np.pi**2 * fs_harmonic**2 * C2_init)
    
    # High resolution interpolation for better optimization (like improved MBVD/EBVD)
    freq_min = max(freq_g.min(), freq_b.min())
    freq_max = min(freq_g.max(), freq_b.max())
    # Use cubic spline interpolation for smoother data
    from scipy.interpolate import interp1d
    num_points = min(1000, max(len(freq_g), len(freq_b)) * 2)
    freq_common = np.linspace(freq_min, freq_max, num_points)
    
    # Use cubic interpolation for smoother curves
    if len(freq_g) > 3:
        g_interp_func = interp1d(freq_g, g_values_S, kind='cubic', bounds_error=False, fill_value='extrapolate')
        g_interp = g_interp_func(freq_common)
    else:
        g_interp = np.interp(freq_common, freq_g, g_values_S)
    
    if len(freq_b) > 3:
        b_interp_func = interp1d(freq_b, b_values_S, kind='cubic', bounds_error=False, fill_value='extrapolate')
        b_interp = b_interp_func(freq_common)
    else:
        b_interp = np.interp(freq_common, freq_b, b_values_S)
    
    # Enhanced resonance weighting (like improved MBVD)
    g_max = np.max(g_interp)
    g_min = np.min(g_interp)
    g_range = g_max - g_min if g_max > g_min else 1.0
    
    # Find peaks for better harmonic detection
    try:
        from scipy.signal import find_peaks
        # Use higher resolution for peak detection
        freq_peaks = np.linspace(freq_min, freq_max, min(2000, len(freq_g) * 10))
        if len(freq_g) > 3:
            g_peaks_func = interp1d(freq_g, g_values_S, kind='cubic', bounds_error=False, fill_value='extrapolate')
            g_peaks = g_peaks_func(freq_peaks)
        else:
            g_peaks = np.interp(freq_peaks, freq_g, g_values_S)
        
        peaks, properties = find_peaks(g_peaks, height=g_max * 0.3, distance=len(freq_peaks) // 20)
        peak_freqs = freq_peaks[peaks] if len(peaks) > 0 else [fs_measured]
    except:
        peak_freqs = [fs_measured]
    
    # Enhanced resonance weighting
    resonance_weight = np.ones_like(freq_common)
    for peak_freq in peak_freqs:
        freq_distance = np.abs(freq_common - peak_freq)
        freq_range = freq_max - freq_min
        resonance_proximity = 1.0 - np.clip(freq_distance / (freq_range * 0.2), 0.0, 1.0)
        resonance_weight += 2.0 * resonance_proximity
    
    resonance_weight[g_interp > g_max * 0.5] *= 2.0
    resonance_weight[g_interp > g_max * 0.8] *= 2.0
    
    # Store adaptive weights (will be updated in Stage 2)
    adaptive_weights_g_ebvd = np.ones_like(freq_common)
    
    def ebvd_error(params, adaptive_weights_g_inner=None):
        """Calculate normalized error for EBVD model - optimized for Mean Relative Error"""
        R0_opt, R1_opt, L1_opt, C1_opt, R2_opt, L2_opt, C2_opt = params
        
        if (R0_opt <= 0 or R1_opt <= 0 or L1_opt <= 0 or C1_opt <= 0 or
            R2_opt <= 0 or L2_opt <= 0 or C2_opt <= 0):
            return 1e10
        
        Y_model = ebvd_admittance(freq_common, C0, R1_opt, L1_opt, C1_opt,
                                  R2_opt, L2_opt, C2_opt)
        g_model = np.real(Y_model)
        b_model = np.imag(Y_model)
        
        # Direct Mean Relative Error minimization (like improved MBVD/EBVD)
        abs_error_g = np.abs(g_model - g_interp)
        abs_error_b = np.abs(b_model - b_interp)
        
        g_magnitude = np.abs(g_interp) + 1e-10
        b_magnitude = np.abs(b_interp) + 1e-10
        rel_error_g = abs_error_g / g_magnitude
        rel_error_b = abs_error_b / b_magnitude
        
        # Combine resonance weight with adaptive weight
        if adaptive_weights_g_inner is not None:
            combined_weight_g = resonance_weight * adaptive_weights_g_inner
        else:
            combined_weight_g = resonance_weight
        
        rel_error_g_power = np.abs(rel_error_g) ** 1.5
        error_g_mean_rel = np.mean(combined_weight_g * rel_error_g_power)
        error_g_rel_sq = np.mean(combined_weight_g * rel_error_g**2)
        error_g = 0.8 * error_g_mean_rel + 0.2 * error_g_rel_sq
        
        b_typical = np.max(np.abs(b_interp)) + 1e-10
        error_b = np.mean((abs_error_b / b_typical)**2 + rel_error_b**2)
        
        return 20.0 * error_g + error_b
    
    # Multi-stage optimization (same as improved MBVD/Mason/KLM - 7 stages)
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
    
    best_result = None
    best_error = 1e10
    
    # Multiple starting points (more variations)
    initial_guesses = [
        initial_params,
        [R0 * 0.8, R1, L1, C1, R2_init, L2_init, C2_init],
        [R0 * 1.2, R1, L1, C1, R2_init, L2_init, C2_init],
        [R0, R1 * 0.5, L1, C1, R2_init, L2_init, C2_init],
        [R0, R1 * 2.0, L1, C1, R2_init, L2_init, C2_init],
        [R0, R1, L1 * 0.8, C1, R2_init, L2_init, C2_init],
        [R0, R1, L1 * 1.2, C1, R2_init, L2_init, C2_init],
        [R0, R1, L1, C1 * 0.8, R2_init, L2_init, C2_init],
        [R0, R1, L1, C1 * 1.2, R2_init, L2_init, C2_init],
        [R0, R1, L1, C1, R2_init * 0.5, L2_init, C2_init],
        [R0, R1, L1, C1, R2_init * 2.0, L2_init, C2_init],
    ]
    
    # Stage 1: Initial optimization with multiple starting points
    for start_params in initial_guesses:
        try:
            result = minimize(ebvd_error, start_params, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.success and result.fun < best_error:
                best_result = result
                best_error = result.fun
        except:
            continue
    
    # Stage 2: Adaptive weighting optimization
    if best_result is not None:
        # Evaluate model with best parameters
        A_temp = best_result.x
        Y_temp = ebvd_admittance(freq_common, C0, A_temp[1], A_temp[2], A_temp[3],
                                 A_temp[4], A_temp[5], A_temp[6])
        g_temp = np.real(Y_temp)
        
        # Calculate local relative errors
        rel_error_g_local = np.abs(g_temp - g_interp) / (np.abs(g_interp) + 1e-10)
        
        # Create adaptive weights: much higher weight for points with high relative error
        threshold_g = 0.05  # 5% relative error threshold
        adaptive_weights_g = np.ones_like(rel_error_g_local)
        high_error_mask_g = rel_error_g_local > threshold_g
        adaptive_weights_g[high_error_mask_g] = 1.0 + 9.0 * np.exp(
            np.clip((rel_error_g_local[high_error_mask_g] - threshold_g) / threshold_g, 0, 2)
        )
        
        # Additional weighting: penalize points with relative error > 7%
        very_high_error_mask_g = rel_error_g_local > 0.07
        adaptive_weights_g[very_high_error_mask_g] *= 2.0
        
        # Update adaptive weights
        adaptive_weights_g_ebvd = adaptive_weights_g.copy()
        
        # Create error function with adaptive weights
        def ebvd_error_adaptive(params):
            return ebvd_error(params, adaptive_weights_g_ebvd)
        
        # Re-optimize with adaptive weights
        try:
            result_adaptive = minimize(ebvd_error_adaptive, best_result.x, method='L-BFGS-B', bounds=bounds,
                                     options={'maxiter': 3000, 'ftol': 1e-11, 'gtol': 1e-9})
            if result_adaptive.success and result_adaptive.fun < best_error:
                best_result = result_adaptive
                best_error = result_adaptive.fun
        except:
            pass
    
    # Stage 3: SLSQP refinement
    if best_result is not None:
        best_start = best_result.x
    else:
        best_start = initial_params
        
    try:
        result2 = minimize(ebvd_error, best_start, method='SLSQP', bounds=bounds,
                         options={'maxiter': 5000, 'ftol': 1e-11})
        if result2.success and result2.fun < best_error:
            best_result = result2
            best_error = result2.fun
    except:
        pass
    
    # Stage 4: Differential evolution for global optimization
    try:
        from scipy.optimize import differential_evolution
        result3 = differential_evolution(ebvd_error, bounds, seed=42,
                                       maxiter=500, popsize=40, atol=1e-11, polish=True,
                                       workers=1, updating='immediate')
        if result3.success and result3.fun < best_error:
            best_result = result3
            best_error = result3.fun
    except:
        pass
    
    # Stage 5: Iterative refinement with narrowed bounds
    if best_result is not None:
        narrowed_bounds = [
            (best_result.x[0] * 0.8, best_result.x[0] * 1.2),
            (best_result.x[1] * 0.8, best_result.x[1] * 1.2),
            (best_result.x[2] * 0.8, best_result.x[2] * 1.2),
            (best_result.x[3] * 0.8, best_result.x[3] * 1.2),
            (best_result.x[4] * 0.8, best_result.x[4] * 1.2),
            (best_result.x[5] * 0.8, best_result.x[5] * 1.2),
            (best_result.x[6] * 0.8, best_result.x[6] * 1.2),
        ]
        
        try:
            result5 = minimize(ebvd_error, best_result.x, method='L-BFGS-B', bounds=narrowed_bounds,
                              options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-11})
            if result5.success and result5.fun < best_error:
                best_result = result5
                best_error = result5.fun
        except:
            pass
        
        # Final ultra-precise polish
        try:
            result6 = minimize(ebvd_error, best_result.x, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            if result6.success and result6.fun < best_error:
                best_result = result6
                best_error = result6.fun
        except:
            pass
    
    # Stage 6: Direct Mean Relative Error minimization - Conductance only
    def ebvd_mre_error_g_only(params):
        R0_mre, R1_mre, L1_mre, C1_mre, R2_mre, L2_mre, C2_mre = params
        if (R0_mre <= 0 or R1_mre <= 0 or L1_mre <= 0 or C1_mre <= 0 or
            R2_mre <= 0 or L2_mre <= 0 or C2_mre <= 0):
            return 1e10
        
        Y_mre = ebvd_admittance(freq_common, C0, R1_mre, L1_mre, C1_mre,
                                R2_mre, L2_mre, C2_mre)
        g_mre = np.real(Y_mre)
        rel_error_g_mre = np.abs(g_mre - g_interp) / (np.abs(g_interp) + 1e-10)
        mre_g = np.mean(rel_error_g_mre)
        high_error_penalty = np.mean(np.maximum(0, rel_error_g_mre - 0.10)) * 10.0
        return mre_g + high_error_penalty
    
    # Optimize with Conductance-only MRE-focused function (multiple attempts)
    if best_result is not None:
        for attempt in range(3):
            try:
                if attempt == 0:
                    start_mre = best_result.x
                elif attempt == 1:
                    start_mre = best_result.x * 1.05
                else:
                    start_mre = best_result.x * 0.95
                
                result7 = minimize(ebvd_mre_error_g_only, start_mre, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-11})
                if result7.success:
                    Y_test = ebvd_admittance(freq_common, C0, result7.x[1], result7.x[2], result7.x[3],
                                            result7.x[4], result7.x[5], result7.x[6])
                    g_test = np.real(Y_test)
                    mre_test = np.mean(np.abs(g_test - g_interp) / (np.abs(g_interp) + 1e-10))
                    
                    Y_prev = ebvd_admittance(freq_common, C0, best_result.x[1], best_result.x[2], best_result.x[3],
                                            best_result.x[4], best_result.x[5], best_result.x[6])
                    g_prev = np.real(Y_prev)
                    mre_prev = np.mean(np.abs(g_prev - g_interp) / (np.abs(g_interp) + 1e-10))
                    
                    # Also check that Susceptance doesn't degrade too much
                    b_test = np.imag(Y_test)
                    b_prev = np.imag(Y_prev)
                    mre_b_test = np.mean(np.abs(b_test - b_interp) / (np.abs(b_interp) + 1e-10))
                    mre_b_prev = np.mean(np.abs(b_prev - b_interp) / (np.abs(b_interp) + 1e-10))
                    
                    if mre_test < mre_prev and (mre_b_test - mre_b_prev) < 0.005:
                        best_result = result7
                        best_error = ebvd_mre_error_g_only(result7.x)
                        break
            except:
                continue
    
    # Stage 7: Final combined optimization with very high weight on Conductance MRE
    def ebvd_mre_error_combined(params):
        R0_mre, R1_mre, L1_mre, C1_mre, R2_mre, L2_mre, C2_mre = params
        if (R0_mre <= 0 or R1_mre <= 0 or L1_mre <= 0 or C1_mre <= 0 or
            R2_mre <= 0 or L2_mre <= 0 or C2_mre <= 0):
            return 1e10
        
        Y_mre = ebvd_admittance(freq_common, C0, R1_mre, L1_mre, C1_mre,
                                R2_mre, L2_mre, C2_mre)
        g_mre = np.real(Y_mre)
        b_mre = np.imag(Y_mre)
        
        # Conductance: Mean Relative Error (direct target)
        rel_error_g_mre = np.abs(g_mre - g_interp) / (np.abs(g_interp) + 1e-10)
        mre_g = np.mean(rel_error_g_mre)
        
        # Susceptance: regular error (already good)
        abs_error_b = np.abs(b_mre - b_interp)
        b_magnitude = np.abs(b_interp) + 1e-10
        rel_error_b = abs_error_b / b_magnitude
        b_typical = np.max(np.abs(b_interp)) + 1e-10
        error_b = np.mean((abs_error_b / b_typical)**2 + rel_error_b**2)
        
        # Very high weight on Conductance MRE (100x)
        return 100.0 * mre_g + error_b
    
    # Final optimization with combined function
    if best_result is not None:
        try:
            result8 = minimize(ebvd_mre_error_combined, best_result.x, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 4000, 'ftol': 1e-13, 'gtol': 1e-11})
            if result8.success:
                Y_test_final = ebvd_admittance(freq_common, C0, result8.x[1], result8.x[2], result8.x[3],
                                              result8.x[4], result8.x[5], result8.x[6])
                g_test_final = np.real(Y_test_final)
                mre_final = np.mean(np.abs(g_test_final - g_interp) / (np.abs(g_interp) + 1e-10))
                
                Y_prev = ebvd_admittance(freq_common, C0, best_result.x[1], best_result.x[2], best_result.x[3],
                                        best_result.x[4], best_result.x[5], best_result.x[6])
                g_prev = np.real(Y_prev)
                mre_prev = np.mean(np.abs(g_prev - g_interp) / (np.abs(g_interp) + 1e-10))
                
                if mre_final < mre_prev:
                    best_result = result8
                    best_error = ebvd_mre_error_combined(result8.x)
        except:
            pass
    
    if best_result is not None:
        R0, R1, L1, C1, R2, L2, C2 = best_result.x
    else:
        # Fallback to initial values
        R2, L2, C2 = R2_init, L2_init, C2_init
    
    # Recalculate frequencies
    fs_optimized = 1.0 / (2 * np.pi * np.sqrt(L1 * C1))
    
    freq_test = np.linspace(fs_optimized, fs_optimized * 1.5, 1000)
    Y_test = ebvd_admittance(freq_test, C0, R1, L1, C1,
                             R2, L2, C2)
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
        
        # Multi-stage optimization (same as EBVD - 7 stages)
        best_result = None
        best_error = 1e10
        
        # Multiple starting points (more variations)
        initial_guesses = [
            initial_params,
            [k_t_init * 0.8, Z_a_init, t_init, alpha_init, R_m_init],
            [k_t_init * 1.2, Z_a_init, t_init, alpha_init, R_m_init],
            [k_t_init, Z_a_init * 0.5, t_init, alpha_init, R_m_init],
            [k_t_init, Z_a_init * 2.0, t_init, alpha_init, R_m_init],
            [k_t_init, Z_a_init, t_init * 0.8, alpha_init, R_m_init],
            [k_t_init, Z_a_init, t_init * 1.2, alpha_init, R_m_init],
            [k_t_init, Z_a_init, t_init, alpha_init, R_m_init * 0.5],
            [k_t_init, Z_a_init, t_init, alpha_init, R_m_init * 2.0],
        ]
        
        # Stage 1: Initial optimization with multiple starting points
        for start_params in initial_guesses:
            try:
                result = minimize(mason_error, start_params, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
                if result.success and result.fun < best_error:
                    best_result = result
                    best_error = result.fun
            except:
                continue
        
        # Stage 2: Adaptive weighting optimization
        if best_result is not None:
            # Evaluate model with best parameters
            A_temp = C0 * best_result.x[2] / (epsilon_r * epsilon_0)
            A_temp = np.clip(A_temp, A_min, A_max)
            Y_temp = mason_admittance(freq_common, C0, best_result.x[0], best_result.x[1], best_result.x[2],
                                     A_temp, best_result.x[3], best_result.x[4])
            if R0 > 0:
                Y_temp += 1.0 / R0
            if R1 > 0 and L1 > 0 and C1 > 0:
                omega = 2 * np.pi * freq_common
                Z_series = R1 + 1j * (omega * L1 - 1 / (omega * C1))
                Y_series = 1 / (Z_series + 1e-12)
                G_mbvd = np.real(Y_series)
                Y_temp = Y_temp + 0.3 * G_mbvd
            
            g_temp = np.real(Y_temp)
            rel_error_g_local = np.abs(g_temp - g_interp) / (np.abs(g_interp) + 1e-10)
            
            # Create adaptive weights
            threshold_g = 0.05  # 5% relative error threshold
            adaptive_weights_g_mason = np.ones_like(rel_error_g_local)
            high_error_mask_g = rel_error_g_local > threshold_g
            adaptive_weights_g_mason[high_error_mask_g] = 1.0 + 9.0 * np.exp(
                np.clip((rel_error_g_local[high_error_mask_g] - threshold_g) / threshold_g, 0, 2)
            )
            very_high_error_mask_g = rel_error_g_local > 0.07
            adaptive_weights_g_mason[very_high_error_mask_g] *= 2.0
            
            # Create error function with adaptive weights
            def mason_error_adaptive(params):
                k_t_ma, Z_a_ma, t_ma, alpha_ma, R_m_ma = params
                if (k_t_ma <= 0 or k_t_ma >= 1 or Z_a_ma <= 0 or 
                    t_ma < t_min or t_ma > t_max or alpha_ma < 0 or R_m_ma <= 0):
                    return 1e10
                
                A_ma = C0 * t_ma / (epsilon_r * epsilon_0)
                A_ma = np.clip(A_ma, A_min, A_max)
                
                Y_ma = mason_admittance(freq_common, C0, k_t_ma, Z_a_ma, t_ma, A_ma, alpha_ma, R_m_ma)
                if R0 > 0:
                    Y_ma += 1.0 / R0
                if R1 > 0 and L1 > 0 and C1 > 0:
                    omega = 2 * np.pi * freq_common
                    Z_series = R1 + 1j * (omega * L1 - 1 / (omega * C1))
                    Y_series = 1 / (Z_series + 1e-12)
                    G_mbvd = np.real(Y_series)
                    Y_ma = Y_ma + 0.3 * G_mbvd
                
                g_ma = np.real(Y_ma)
                b_ma = np.imag(Y_ma)
                
                abs_error_g = np.abs(g_ma - g_interp)
                abs_error_b = np.abs(b_ma - b_interp)
                g_magnitude = np.abs(g_interp) + 1e-10
                b_magnitude = np.abs(b_interp) + 1e-10
                rel_error_g = abs_error_g / g_magnitude
                rel_error_b = abs_error_b / b_magnitude
                
                combined_weight_g = resonance_weight * adaptive_weights_g_mason
                rel_error_g_power = np.abs(rel_error_g) ** 1.5
                error_g_mean_rel = np.mean(combined_weight_g * rel_error_g_power)
                error_g_rel_sq = np.mean(combined_weight_g * rel_error_g**2)
                error_g = 0.9 * error_g_mean_rel + 0.1 * error_g_rel_sq
                
                b_typical = np.max(np.abs(b_interp)) + 1e-10
                error_b = np.mean((abs_error_b / b_typical)**2 + rel_error_b**2)
                
                return 30.0 * error_g + error_b
            
            # Re-optimize with adaptive weights
            try:
                result_adaptive = minimize(mason_error_adaptive, best_result.x, method='L-BFGS-B', bounds=bounds,
                                         options={'maxiter': 3000, 'ftol': 1e-11, 'gtol': 1e-9})
                if result_adaptive.success and result_adaptive.fun < best_error:
                    best_result = result_adaptive
                    best_error = result_adaptive.fun
            except:
                pass
        
        # Stage 3: SLSQP refinement
        if best_result is not None:
            best_start = best_result.x
        else:
            best_start = initial_params
            
        try:
            result2 = minimize(mason_error, best_start, method='SLSQP', bounds=bounds,
                             options={'maxiter': 5000, 'ftol': 1e-11})
            if result2.success and result2.fun < best_error:
                best_result = result2
                best_error = result2.fun
        except:
            pass
        
        # Stage 4: Differential evolution for global optimization
        try:
            from scipy.optimize import differential_evolution
            result3 = differential_evolution(mason_error, bounds, seed=42,
                                           maxiter=500, popsize=40, atol=1e-11, polish=True,
                                           workers=1, updating='immediate')
            if result3.success and result3.fun < best_error:
                best_result = result3
                best_error = result3.fun
        except:
            pass
        
        # Stage 5: Iterative refinement with narrowed bounds
        if best_result is not None:
            narrowed_bounds = [
                (max(k_t_init * 0.5, best_result.x[0] * 0.8), min(0.99, best_result.x[0] * 1.2)),
                (best_result.x[1] * 0.8, best_result.x[1] * 1.2),
                (max(t_min, best_result.x[2] * 0.8), min(t_max, best_result.x[2] * 1.2)),
                (best_result.x[3] * 0.8, min(100.0, best_result.x[3] * 1.2)),
                (best_result.x[4] * 0.8, best_result.x[4] * 1.2),
            ]
            
            try:
                result5 = minimize(mason_error, best_result.x, method='L-BFGS-B', bounds=narrowed_bounds,
                                  options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-11})
                if result5.success and result5.fun < best_error:
                    best_result = result5
                    best_error = result5.fun
            except:
                pass
            
            # Final ultra-precise polish
            try:
                result6 = minimize(mason_error, best_result.x, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
                if result6.success and result6.fun < best_error:
                    best_result = result6
                    best_error = result6.fun
            except:
                pass
        
        # Stage 6: Direct Mean Relative Error minimization - Conductance only
        def mason_mre_error_g_only(params):
            k_t_mre, Z_a_mre, t_mre, alpha_mre, R_m_mre = params
            if (k_t_mre <= 0 or k_t_mre >= 1 or Z_a_mre <= 0 or
                t_mre < t_min or t_mre > t_max or alpha_mre < 0 or R_m_mre <= 0):
                return 1e10
            
            A_mre = np.clip(C0 * t_mre / (epsilon_r * epsilon_0), A_min, A_max)
            Y_mre = mason_admittance(freq_common, C0, k_t_mre, Z_a_mre, t_mre, A_mre, alpha_mre, R_m_mre)
            if R0 > 0:
                Y_mre += 1.0 / R0
            if R1 > 0 and L1 > 0 and C1 > 0:
                omega = 2 * np.pi * freq_common
                Z_series = R1 + 1j * (omega * L1 - 1 / (omega * C1))
                Y_series = 1 / (Z_series + 1e-12)
                G_mbvd = np.real(Y_series)
                Y_mre = Y_mre + 0.3 * G_mbvd
            
            g_mre = np.real(Y_mre)
            rel_error_g_mre = np.abs(g_mre - g_interp) / (np.abs(g_interp) + 1e-10)
            mre_g = np.mean(rel_error_g_mre)
            high_error_penalty = np.mean(np.maximum(0, rel_error_g_mre - 0.10)) * 10.0
            return mre_g + high_error_penalty
        
        # Optimize with Conductance-only MRE-focused function (multiple attempts)
        if best_result is not None:
            for attempt in range(3):
                try:
                    if attempt == 0:
                        start_mre = best_result.x
                    elif attempt == 1:
                        start_mre = best_result.x * 1.05
                    else:
                        start_mre = best_result.x * 0.95
                    
                    result7 = minimize(mason_mre_error_g_only, start_mre, method='L-BFGS-B', bounds=bounds,
                                      options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-11})
                    if result7.success:
                        A_test = np.clip(C0 * result7.x[2] / (epsilon_r * epsilon_0), A_min, A_max)
                        Y_test = mason_admittance(freq_common, C0, result7.x[0], result7.x[1], result7.x[2],
                                                 A_test, result7.x[3], result7.x[4])
                        if R0 > 0:
                            Y_test += 1.0 / R0
                        if R1 > 0 and L1 > 0 and C1 > 0:
                            omega = 2 * np.pi * freq_common
                            Z_series = R1 + 1j * (omega * L1 - 1 / (omega * C1))
                            Y_series = 1 / (Z_series + 1e-12)
                            G_mbvd = np.real(Y_series)
                            Y_test = Y_test + 0.3 * G_mbvd
                        
                        g_test = np.real(Y_test)
                        mre_test = np.mean(np.abs(g_test - g_interp) / (np.abs(g_interp) + 1e-10))
                        
                        A_prev = np.clip(C0 * best_result.x[2] / (epsilon_r * epsilon_0), A_min, A_max)
                        Y_prev = mason_admittance(freq_common, C0, best_result.x[0], best_result.x[1], best_result.x[2],
                                                 A_prev, best_result.x[3], best_result.x[4])
                        if R0 > 0:
                            Y_prev += 1.0 / R0
                        if R1 > 0 and L1 > 0 and C1 > 0:
                            omega = 2 * np.pi * freq_common
                            Z_series = R1 + 1j * (omega * L1 - 1 / (omega * C1))
                            Y_series = 1 / (Z_series + 1e-12)
                            G_mbvd = np.real(Y_series)
                            Y_prev = Y_prev + 0.3 * G_mbvd
                        
                        g_prev = np.real(Y_prev)
                        mre_prev = np.mean(np.abs(g_prev - g_interp) / (np.abs(g_interp) + 1e-10))
                        
                        b_test = np.imag(Y_test)
                        b_prev = np.imag(Y_prev)
                        mre_b_test = np.mean(np.abs(b_test - b_interp) / (np.abs(b_interp) + 1e-10))
                        mre_b_prev = np.mean(np.abs(b_prev - b_interp) / (np.abs(b_interp) + 1e-10))
                        
                        if mre_test < mre_prev and (mre_b_test - mre_b_prev) < 0.005:
                            best_result = result7
                            best_error = mason_mre_error_g_only(result7.x)
                            break
                except:
                    continue
        
        # Stage 7: Final combined optimization with very high weight on Conductance MRE
        def mason_mre_error_combined(params):
            k_t_mre, Z_a_mre, t_mre, alpha_mre, R_m_mre = params
            if (k_t_mre <= 0 or k_t_mre >= 1 or Z_a_mre <= 0 or
                t_mre < t_min or t_mre > t_max or alpha_mre < 0 or R_m_mre <= 0):
                return 1e10
            
            A_mre = np.clip(C0 * t_mre / (epsilon_r * epsilon_0), A_min, A_max)
            Y_mre = mason_admittance(freq_common, C0, k_t_mre, Z_a_mre, t_mre, A_mre, alpha_mre, R_m_mre)
            if R0 > 0:
                Y_mre += 1.0 / R0
            if R1 > 0 and L1 > 0 and C1 > 0:
                omega = 2 * np.pi * freq_common
                Z_series = R1 + 1j * (omega * L1 - 1 / (omega * C1))
                Y_series = 1 / (Z_series + 1e-12)
                G_mbvd = np.real(Y_series)
                Y_mre = Y_mre + 0.3 * G_mbvd
            
            g_mre = np.real(Y_mre)
            b_mre = np.imag(Y_mre)
            
            rel_error_g_mre = np.abs(g_mre - g_interp) / (np.abs(g_interp) + 1e-10)
            mre_g = np.mean(rel_error_g_mre)
            
            abs_error_b = np.abs(b_mre - b_interp)
            b_magnitude = np.abs(b_interp) + 1e-10
            rel_error_b = abs_error_b / b_magnitude
            b_typical = np.max(np.abs(b_interp)) + 1e-10
            error_b = np.mean((abs_error_b / b_typical)**2 + rel_error_b**2)
            
            return 100.0 * mre_g + error_b
        
        # Final optimization with combined function
        if best_result is not None:
            try:
                result8 = minimize(mason_mre_error_combined, best_result.x, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 4000, 'ftol': 1e-13, 'gtol': 1e-11})
                if result8.success:
                    A_test_final = np.clip(C0 * result8.x[2] / (epsilon_r * epsilon_0), A_min, A_max)
                    Y_test_final = mason_admittance(freq_common, C0, result8.x[0], result8.x[1], result8.x[2],
                                                   A_test_final, result8.x[3], result8.x[4])
                    if R0 > 0:
                        Y_test_final += 1.0 / R0
                    if R1 > 0 and L1 > 0 and C1 > 0:
                        omega = 2 * np.pi * freq_common
                        Z_series = R1 + 1j * (omega * L1 - 1 / (omega * C1))
                        Y_series = 1 / (Z_series + 1e-12)
                        G_mbvd = np.real(Y_series)
                        Y_test_final = Y_test_final + 0.3 * G_mbvd
                    
                    g_test_final = np.real(Y_test_final)
                    mre_final = np.mean(np.abs(g_test_final - g_interp) / (np.abs(g_interp) + 1e-10))
                    
                    A_prev = np.clip(C0 * best_result.x[2] / (epsilon_r * epsilon_0), A_min, A_max)
                    Y_prev = mason_admittance(freq_common, C0, best_result.x[0], best_result.x[1], best_result.x[2],
                                             A_prev, best_result.x[3], best_result.x[4])
                    if R0 > 0:
                        Y_prev += 1.0 / R0
                    if R1 > 0 and L1 > 0 and C1 > 0:
                        omega = 2 * np.pi * freq_common
                        Z_series = R1 + 1j * (omega * L1 - 1 / (omega * C1))
                        Y_series = 1 / (Z_series + 1e-12)
                        G_mbvd = np.real(Y_series)
                        Y_prev = Y_prev + 0.3 * G_mbvd
                    
                    g_prev = np.real(Y_prev)
                    mre_prev = np.mean(np.abs(g_prev - g_interp) / (np.abs(g_interp) + 1e-10))
                    
                    if mre_final < mre_prev:
                        best_result = result8
                        best_error = mason_mre_error_combined(result8.x)
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


def klm_admittance(freq, C0, k_t, Z_a, t, A, Z_load=None, alpha=0.0, R_m=None, R0=None):
    """
    Calculate Admittance from KLM (Krimholtz-Leedom-Matthaei) Model
    
    KLM model uses a transformer to couple electrical and mechanical parts.
    It's particularly accurate for hydroacoustic transducers.
    
    Args:
        freq: frequency array (Hz)
        C0: static capacitance (F)
        k_t: electromechanical coupling coefficient (thickness mode)
        Z_a: acoustic impedance (kg/(m²·s)) = ρ·c
        t: thickness of piezoelectric element (m)
        A: area of piezoelectric element (m²)
        Z_load: acoustic load impedance (kg/(m²·s)), optional - default is water (1.5e6)
        alpha: acoustic attenuation coefficient (Np/m), optional
        R_m: mechanical loss resistance (Ohm), optional
        R0: dielectric loss resistance (Ohm), optional
    
    Returns:
        Complex admittance array (S)
    """
    omega = 2 * np.pi * freq
    
    # Estimate density and sound speed
    if Z_a < 1e7:
        rho = 2650  # Ceramics
    else:
        rho = 7800  # PZT
    
    c = Z_a / rho  # Sound speed
    
    # Acoustic load (default to water)
    if Z_load is None:
        Z_load = 1.5e6  # Water acoustic impedance (kg/(m²·s))
    
    # Wave number with attenuation
    k = omega / c
    k_complex = k - 1j * alpha
    
    # KLM model parameters
    # Transformer turns ratio: n = k_t * sqrt(C0 * Z_a / (A * t))
    n = k_t * np.sqrt(C0 * Z_a / (A * t + 1e-12))
    
    # Characteristic impedance of transmission line
    Z_0 = Z_a * A  # Acoustic impedance scaled by area
    
    # Electrical length: beta * l = k * t
    beta_l = k_complex * t
    
    # KLM equivalent circuit admittance
    beta_l_safe = np.where(np.abs(beta_l) < 1e-10, 1e-10, beta_l)
    tan_bl = np.tan(beta_l_safe)
    
    # Mechanical impedance (transmission line terminated by Z_load)
    Z_mech_num = Z_load + 1j * Z_0 * tan_bl
    Z_mech_den = Z_0 + 1j * Z_load * tan_bl
    Z_mech = Z_0 * Z_mech_num / (Z_mech_den + 1e-12)
    
    # Add mechanical losses
    if R_m is not None and R_m > 0:
        Z_mech = Z_mech + R_m
    
    # Mechanical admittance
    Y_mech = 1.0 / (Z_mech + 1e-12)
    
    # Transform to electrical domain: Y_elec = n² * Y_mech
    Y_mech_transformed = n**2 * Y_mech
    
    # Total electrical admittance
    Y_klm = 1j * omega * C0 + Y_mech_transformed
    
    # Add dielectric losses (R0 in parallel)
    if R0 is not None and R0 > 0:
        Y_klm += 1.0 / R0
    
    return Y_klm


def calculate_klm_parameters(freq_g, g_values_S, freq_b, b_values_S, C0,
                             t=None, A=None, rho=None, c=None, Z_load=None):
    """
    Calculate KLM Model parameters from experimental data
    
    KLM model is particularly accurate for hydroacoustic transducers.
    Uses transformer coupling between electrical and mechanical domains.
    
    Args:
        freq_g: frequency array for conductance (Hz)
        g_values_S: conductance values (S)
        freq_b: frequency array for susceptance (Hz)
        b_values_S: susceptance values (S)
        C0: static capacitance (F)
        t: thickness (m), optional
        A: area (m²), optional
        rho: density (kg/m³), optional - default ~7800 for PZT
        c: sound speed (m/s), optional - default ~4000 for PZT
        Z_load: acoustic load impedance (kg/(m²·s)), optional - default water (1.5e6)
    
    Returns:
        Dictionary with KLM parameters
    """
    # Start with MBVD for initial estimates
    mbvd_params = calculate_mbvd_parameters(freq_g, g_values_S, freq_b, b_values_S, C0)
    
    R0 = mbvd_params.get('R0', 1e6)
    R1 = mbvd_params['R1']
    L1 = mbvd_params['L1']
    C1 = mbvd_params['C1']
    fs_measured = mbvd_params['fs']
    fp_measured = mbvd_params['fp']
    k = mbvd_params['k']
    
    # Default material properties
    if rho is None:
        rho = 7800  # kg/m³ (PZT)
    if c is None:
        c = 4000  # m/s (PZT)
    if Z_load is None:
        Z_load = 1.5e6  # Water acoustic impedance
    
    # Estimate thickness and area
    if t is None:
        t = c / (2 * fs_measured)
    epsilon_r = 2000
    epsilon_0 = 8.854e-12
    if A is None:
        A = C0 * t / (epsilon_r * epsilon_0)
    
    # Physical constraints
    t_min, t_max = 0.1e-3, 50e-3
    A_min, A_max = 1e-6, 10000e-6
    t = np.clip(t, t_min, t_max)
    A = np.clip(A, A_min, A_max)
    
    Z_a = rho * c
    k_t = k
    
    # High resolution interpolation (like EBVD)
    freq_min = max(freq_g.min(), freq_b.min())
    freq_max = min(freq_g.max(), freq_b.max())
    n_points = min(2000, max(1000, len(freq_g) * 3))
    freq_common = np.linspace(freq_min, freq_max, n_points)
    
    from scipy.interpolate import interp1d
    g_interp_func = interp1d(freq_g, g_values_S, kind='cubic', bounds_error=False, fill_value='extrapolate')
    b_interp_func = interp1d(freq_b, b_values_S, kind='cubic', bounds_error=False, fill_value='extrapolate')
    g_interp = g_interp_func(freq_common)
    b_interp = b_interp_func(freq_common)
    
    # Adaptive weighting (like EBVD)
    g_max = np.max(g_interp)
    g_max_idx = np.argmax(g_interp)
    fs_approx = freq_common[g_max_idx]
    
    freq_distance = np.abs(freq_common - fs_approx)
    freq_range = freq_max - freq_min
    resonance_proximity = 1.0 - np.clip(freq_distance / (freq_range * 0.2), 0.0, 1.0)
    
    resonance_weight = np.ones_like(g_interp)
    resonance_weight += 2.0 * resonance_proximity
    resonance_weight[g_interp > g_max * 0.5] *= 2.0
    resonance_weight[g_interp > g_max * 0.8] *= 2.0
    
    # Store adaptive weights (will be updated in Stage 2)
    adaptive_weights_g_klm = np.ones_like(freq_common)
    
    def klm_error(params, adaptive_weights_g_inner=None):
        """Calculate normalized error for KLM model - optimized for Mean Relative Error"""
        k_t_opt, Z_a_opt, t_opt, alpha_opt, R_m_opt = params
        
        if (k_t_opt <= 0 or k_t_opt >= 1 or Z_a_opt <= 0 or
            t_opt < t_min or t_opt > t_max or alpha_opt < 0 or R_m_opt <= 0):
            return 1e10
        
        A_opt = C0 * t_opt / (epsilon_r * epsilon_0)
        A_opt = np.clip(A_opt, A_min, A_max)
        
        Y_model = klm_admittance(freq_common, C0, k_t_opt, Z_a_opt, t_opt, A_opt,
                                 Z_load, alpha_opt, R_m_opt, R0)
        g_model = np.real(Y_model)
        b_model = np.imag(Y_model)
        
        # Direct Mean Relative Error minimization (like EBVD)
        abs_error_g = np.abs(g_model - g_interp)
        abs_error_b = np.abs(b_model - b_interp)
        
        g_magnitude = np.abs(g_interp) + 1e-10
        b_magnitude = np.abs(b_interp) + 1e-10
        rel_error_g = abs_error_g / g_magnitude
        rel_error_b = abs_error_b / b_magnitude
        
        # Combine resonance weight with adaptive weight
        if adaptive_weights_g_inner is not None:
            combined_weight_g = resonance_weight * adaptive_weights_g_inner
        else:
            combined_weight_g = resonance_weight
        
        rel_error_g_power = np.abs(rel_error_g) ** 1.5
        error_g_mean_rel = np.mean(combined_weight_g * rel_error_g_power)
        error_g_rel_sq = np.mean(combined_weight_g * rel_error_g**2)
        error_g = 0.9 * error_g_mean_rel + 0.1 * error_g_rel_sq
        
        b_typical = np.max(np.abs(b_interp)) + 1e-10
        error_b = np.mean((abs_error_b / b_typical)**2 + rel_error_b**2)
        
        return 30.0 * error_g + error_b
    
    # Store for adaptive weighting
    adaptive_weights_g = np.ones_like(freq_common)
    
    # Multi-stage optimization (same as EBVD - 7 stages)
    initial_params = [k_t, Z_a, t, 0.0, R1]
    bounds = [
        (k_t * 0.5, min(0.99, k_t * 1.5)),
        (Z_a * 0.1, Z_a * 10),
        (max(t_min, t * 0.1), min(t_max, t * 10)),
        (0.0, 100.0),
        (R1 * 0.01, R1 * 100)
    ]
    
    best_result = None
    best_error = 1e10
    
    # Multiple starting points (more variations)
    initial_guesses = [
        initial_params,
        [k_t * 0.8, Z_a, t, 0.0, R1],
        [k_t * 1.2, Z_a, t, 0.0, R1],
        [k_t, Z_a * 0.5, t, 0.0, R1],
        [k_t, Z_a * 2.0, t, 0.0, R1],
        [k_t, Z_a, t * 0.8, 0.0, R1],
        [k_t, Z_a, t * 1.2, 0.0, R1],
        [k_t, Z_a, t, 0.0, R1 * 0.5],
        [k_t, Z_a, t, 0.0, R1 * 2.0],
    ]
    
    # Stage 1: Initial optimization with multiple starting points
    for start_params in initial_guesses:
        try:
            result = minimize(klm_error, start_params, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.success and result.fun < best_error:
                best_result = result
                best_error = result.fun
        except:
            continue
    
    # Stage 2: Adaptive weighting optimization
    if best_result is not None:
        # Evaluate model with best parameters
        A_temp = np.clip(C0 * best_result.x[2] / (epsilon_r * epsilon_0), A_min, A_max)
        Y_temp = klm_admittance(freq_common, C0, best_result.x[0], best_result.x[1], best_result.x[2],
                               A_temp, Z_load, best_result.x[3], best_result.x[4], R0)
        g_temp = np.real(Y_temp)
        
        # Calculate local relative errors
        rel_error_g_local = np.abs(g_temp - g_interp) / (np.abs(g_interp) + 1e-10)
        
        # Create adaptive weights: much higher weight for points with high relative error
        threshold_g = 0.05  # 5% relative error threshold
        adaptive_weights_g = np.ones_like(rel_error_g_local)
        high_error_mask_g = rel_error_g_local > threshold_g
        adaptive_weights_g[high_error_mask_g] = 1.0 + 9.0 * np.exp(
            np.clip((rel_error_g_local[high_error_mask_g] - threshold_g) / threshold_g, 0, 2)
        )
        
        # Additional weighting: penalize points with relative error > 7%
        very_high_error_mask_g = rel_error_g_local > 0.07
        adaptive_weights_g[very_high_error_mask_g] *= 2.0
        
        # Update adaptive weights
        adaptive_weights_g_klm = adaptive_weights_g.copy()
        
        # Create error function with adaptive weights
        def klm_error_adaptive(params):
            return klm_error(params, adaptive_weights_g_klm)
        
        # Re-optimize with adaptive weights
        try:
            result_adaptive = minimize(klm_error_adaptive, best_result.x, method='L-BFGS-B', bounds=bounds,
                                     options={'maxiter': 3000, 'ftol': 1e-11, 'gtol': 1e-9})
            if result_adaptive.success and result_adaptive.fun < best_error:
                best_result = result_adaptive
                best_error = result_adaptive.fun
        except:
            pass
    
    # Stage 3: SLSQP refinement
    if best_result is not None:
        best_start = best_result.x
    else:
        best_start = initial_params
        
    try:
        result2 = minimize(klm_error, best_start, method='SLSQP', bounds=bounds,
                         options={'maxiter': 5000, 'ftol': 1e-11})
        if result2.success and result2.fun < best_error:
            best_result = result2
            best_error = result2.fun
    except:
        pass
    
    # Stage 4: Differential evolution for global optimization
    try:
        from scipy.optimize import differential_evolution
        result3 = differential_evolution(klm_error, bounds, seed=42,
                                       maxiter=500, popsize=40, atol=1e-11, polish=True,
                                       workers=1, updating='immediate')
        if result3.success and result3.fun < best_error:
            best_result = result3
            best_error = result3.fun
    except:
        pass
    
    # Stage 5: Iterative refinement with narrowed bounds
    if best_result is not None:
        narrowed_bounds = [
            (max(k_t * 0.5, best_result.x[0] * 0.8), min(0.99, best_result.x[0] * 1.2)),
            (best_result.x[1] * 0.8, best_result.x[1] * 1.2),
            (max(t_min, best_result.x[2] * 0.8), min(t_max, best_result.x[2] * 1.2)),
            (best_result.x[3] * 0.8, min(100.0, best_result.x[3] * 1.2)),
            (best_result.x[4] * 0.8, best_result.x[4] * 1.2),
        ]
        
        try:
            result5 = minimize(klm_error, best_result.x, method='L-BFGS-B', bounds=narrowed_bounds,
                              options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-11})
            if result5.success and result5.fun < best_error:
                best_result = result5
                best_error = result5.fun
        except:
            pass
        
        # Final ultra-precise polish
        try:
            result6 = minimize(klm_error, best_result.x, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            if result6.success and result6.fun < best_error:
                best_result = result6
                best_error = result6.fun
        except:
            pass
    
    # Stage 6: Direct Mean Relative Error minimization - Conductance only
    def klm_mre_error_g_only(params):
        k_t_mre, Z_a_mre, t_mre, alpha_mre, R_m_mre = params
        if (k_t_mre <= 0 or k_t_mre >= 1 or Z_a_mre <= 0 or
            t_mre < t_min or t_mre > t_max or alpha_mre < 0 or R_m_mre <= 0):
            return 1e10
        
        A_mre = np.clip(C0 * t_mre / (epsilon_r * epsilon_0), A_min, A_max)
        Y_mre = klm_admittance(freq_common, C0, k_t_mre, Z_a_mre, t_mre, A_mre,
                              Z_load, alpha_mre, R_m_mre, R0)
        g_mre = np.real(Y_mre)
        rel_error_g_mre = np.abs(g_mre - g_interp) / (np.abs(g_interp) + 1e-10)
        mre_g = np.mean(rel_error_g_mre)
        high_error_penalty = np.mean(np.maximum(0, rel_error_g_mre - 0.10)) * 10.0
        return mre_g + high_error_penalty
    
    # Optimize with Conductance-only MRE-focused function (multiple attempts)
    if best_result is not None:
        for attempt in range(3):
            try:
                if attempt == 0:
                    start_params = best_result.x
                elif attempt == 1:
                    start_params = best_result.x * 1.05
                else:
                    start_params = best_result.x * 0.95
                
                result7 = minimize(klm_mre_error_g_only, start_params, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-11})
                if result7.success:
                    A_test = np.clip(C0 * result7.x[2] / (epsilon_r * epsilon_0), A_min, A_max)
                    Y_test = klm_admittance(freq_common, C0, result7.x[0], result7.x[1], result7.x[2],
                                           A_test, Z_load, result7.x[3], result7.x[4], R0)
                    g_test = np.real(Y_test)
                    mre_test = np.mean(np.abs(g_test - g_interp) / (np.abs(g_interp) + 1e-10))
                    
                    A_prev = np.clip(C0 * best_result.x[2] / (epsilon_r * epsilon_0), A_min, A_max)
                    Y_prev = klm_admittance(freq_common, C0, best_result.x[0], best_result.x[1], best_result.x[2],
                                           A_prev, Z_load, best_result.x[3], best_result.x[4], R0)
                    g_prev = np.real(Y_prev)
                    mre_prev = np.mean(np.abs(g_prev - g_interp) / (np.abs(g_interp) + 1e-10))
                    
                    # Also check that Susceptance doesn't degrade too much
                    b_test = np.imag(Y_test)
                    b_prev = np.imag(Y_prev)
                    mre_b_test = np.mean(np.abs(b_test - b_interp) / (np.abs(b_interp) + 1e-10))
                    mre_b_prev = np.mean(np.abs(b_prev - b_interp) / (np.abs(b_interp) + 1e-10))
                    
                    if mre_test < mre_prev and (mre_b_test - mre_b_prev) < 0.005:
                        best_result = result7
                        best_error = klm_mre_error_g_only(result7.x)
                        break
            except:
                continue
    
    # Stage 7: Final combined optimization with very high weight on Conductance MRE
    def klm_mre_error_combined(params):
        k_t_mre, Z_a_mre, t_mre, alpha_mre, R_m_mre = params
        if (k_t_mre <= 0 or k_t_mre >= 1 or Z_a_mre <= 0 or
            t_mre < t_min or t_mre > t_max or alpha_mre < 0 or R_m_mre <= 0):
            return 1e10
        
        A_mre = np.clip(C0 * t_mre / (epsilon_r * epsilon_0), A_min, A_max)
        Y_mre = klm_admittance(freq_common, C0, k_t_mre, Z_a_mre, t_mre, A_mre,
                              Z_load, alpha_mre, R_m_mre, R0)
        g_mre = np.real(Y_mre)
        b_mre = np.imag(Y_mre)
        
        # Conductance: Mean Relative Error (direct target)
        rel_error_g_mre = np.abs(g_mre - g_interp) / (np.abs(g_interp) + 1e-10)
        mre_g = np.mean(rel_error_g_mre)
        
        # Susceptance: regular error (already good)
        abs_error_b = np.abs(b_mre - b_interp)
        b_magnitude = np.abs(b_interp) + 1e-10
        rel_error_b = abs_error_b / b_magnitude
        b_typical = np.max(np.abs(b_interp)) + 1e-10
        error_b = np.mean((abs_error_b / b_typical)**2 + rel_error_b**2)
        
        # Very high weight on Conductance MRE (100x)
        return 100.0 * mre_g + error_b
    
    # Final optimization with combined function
    if best_result is not None:
        try:
            result8 = minimize(klm_mre_error_combined, best_result.x, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 4000, 'ftol': 1e-13, 'gtol': 1e-11})
            if result8.success:
                A_test_final = np.clip(C0 * result8.x[2] / (epsilon_r * epsilon_0), A_min, A_max)
                Y_test_final = klm_admittance(freq_common, C0, result8.x[0], result8.x[1], result8.x[2],
                                             A_test_final, Z_load, result8.x[3], result8.x[4], R0)
                g_test_final = np.real(Y_test_final)
                mre_final = np.mean(np.abs(g_test_final - g_interp) / (np.abs(g_interp) + 1e-10))
                
                A_prev = np.clip(C0 * best_result.x[2] / (epsilon_r * epsilon_0), A_min, A_max)
                Y_prev = klm_admittance(freq_common, C0, best_result.x[0], best_result.x[1], best_result.x[2],
                                       A_prev, Z_load, best_result.x[3], best_result.x[4], R0)
                g_prev = np.real(Y_prev)
                mre_prev = np.mean(np.abs(g_prev - g_interp) / (np.abs(g_interp) + 1e-10))
                
                if mre_final < mre_prev:
                    best_result = result8
                    best_error = klm_mre_error_combined(result8.x)
        except:
            pass
    
    if best_result is not None:
        k_t_opt, Z_a_opt, t_opt, alpha_opt, R_m_opt = best_result.x
    else:
        k_t_opt, Z_a_opt, t_opt, alpha_opt, R_m_opt = k_t, Z_a, t, 0.0, R1
    
    A_opt = np.clip(C0 * t_opt / (epsilon_r * epsilon_0), A_min, A_max)
    
    # Recalculate frequencies
    c_opt = Z_a_opt / rho
    fs_klm = c_opt / (2 * t_opt) if t_opt > 0 else fs_measured
    fp_klm = fs_klm / np.sqrt(1 - k_t_opt**2) if k_t_opt < 1 else fs_klm * 1.1
    
    if 0.8 * fs_measured < fs_klm < 1.2 * fs_measured:
        fs_measured = fs_klm
    if fp_klm > fs_measured and 0.8 * fp_measured < fp_klm < 1.5 * fp_measured:
        fp_measured = fp_klm
    
    omega_s = 2 * np.pi * fs_measured
    Qm = omega_s * L1 / R1 if R1 > 0 else 10.0
    
    return {
        'model_type': 'KLM',
        'C0': C0,
        'R0': R0,
        'k_t': k_t_opt,
        'Z_a': Z_a_opt,
        't': t_opt,
        'A': A_opt,
        'rho': rho,
        'c': c_opt,
        'alpha': alpha_opt,
        'R_m': R_m_opt,
        'Z_load': Z_load,
        'fs': fs_measured,
        'fp': fp_measured,
        'R1': R1,
        'L1': L1,
        'C1': C1,
        'Qm': Qm,
        'k': k_t_opt,
        'tan_delta': mbvd_params.get('tan_delta', 0.0)
    }


def calculate_model_curves_klm(freq_model, klm_params):
    """
    Calculate KLM model curves for plotting
    
    Args:
        freq_model: frequency array for model (Hz)
        klm_params: dictionary with KLM parameters
    
    Returns:
        Dictionary with model data
    """
    Y_model = klm_admittance(
        freq_model,
        klm_params['C0'],
        klm_params['k_t'],
        klm_params['Z_a'],
        klm_params['t'],
        klm_params['A'],
        klm_params.get('Z_load', 1.5e6),
        klm_params.get('alpha', 0.0),
        klm_params.get('R_m', None),
        klm_params.get('R0', None)
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
