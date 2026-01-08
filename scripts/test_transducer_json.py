#!/usr/bin/env python3
"""
Test program for visualizing transducer data from JSON file.

Loads JSON file exported from GUI and displays 4 graphs:
1. Conductance (G)
2. Susceptance (B)
3. RX Sensitivity
4. TX Sensitivity
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import UnivariateSpline
from numpy.polynomial import Polynomial


# This function is not used but kept for reference


def reconstruct_polynomial(coefficients, x_min=None, x_max=None):
    """
    Reconstructs a polynomial function from coefficients.
    
    Args:
        coefficients: list of coefficients [a0, a1, a2, ...] for y = a0 + a1*x + a2*x² + ...
        x_min: minimum x value (optional)
        x_max: maximum x value (optional)
    
    Returns:
        Function that takes x and returns y
    """
    poly = Polynomial(coefficients)
    return poly


def reconstruct_function(fitting_params):
    """
    Reconstructs a function from fitting parameters.
    
    Args:
        fitting_params: dictionary with fitting parameters
    
    Returns:
        Function f(x) and frequency range
    """
    fitting_type = fitting_params.get('fitting_type', 'spline')
    freq_range = fitting_params.get('frequency_range', {})
    x_min = freq_range.get('min', 0)
    x_max = freq_range.get('max', 100)
    
    if fitting_type == 'spline':
        knots = fitting_params.get('spline_knots')
        coeffs = fitting_params.get('spline_coefficients')
        degree = fitting_params.get('spline_degree', 3)
        smoothing = fitting_params.get('smoothing', 0.0)
        
        if knots and coeffs:
            knots = np.array(knots)
            coeffs = np.array(coeffs)
            
            # UnivariateSpline uses B-spline basis, but its coefficients
            # are not direct control points for BSpline.
            # Use simplified approach: interpolation through knots and coeffs
            
            try:
                from scipy.interpolate import interp1d, UnivariateSpline
                
                # Method 1: Try using knots as x and coeffs as y
                # This is an approximate method but works for visualization
                if len(knots) == len(coeffs):
                    # If counts match, use direct interpolation
                    x_interp = knots
                    y_interp = coeffs
                elif len(knots) > len(coeffs):
                    # If knots are more, use first len(coeffs) knots
                    x_interp = knots[:len(coeffs)]
                    y_interp = coeffs
                else:
                    # If coeffs are more, use all knots and first len(knots) coeffs
                    x_interp = knots
                    y_interp = coeffs[:len(knots)]
                
                # Use cubic interpolation for smoothness
                if len(x_interp) >= 4:
                    interp_func = interp1d(x_interp, y_interp, kind='cubic',
                                         bounds_error=False, fill_value='extrapolate')
                elif len(x_interp) >= 2:
                    interp_func = interp1d(x_interp, y_interp, kind='linear',
                                         bounds_error=False, fill_value='extrapolate')
                else:
                    # If not enough points, return constant
                    const_value = coeffs[0] if len(coeffs) > 0 else 0.0
                    def func(x):
                        return np.full_like(x, const_value)
                    return func, (x_min, x_max)
                
                def func(x):
                    x_clipped = np.clip(x, x_min, x_max)
                    return interp_func(x_clipped)
                
                return func, (x_min, x_max)
                    
            except Exception as e:
                print(f"Error reconstructing spline for {fitting_params.get('name')}: {e}")
                return None, None
        else:
            print(f"Warning: Missing spline parameters for {fitting_params.get('name')}")
            return None, None
    
    elif fitting_type == 'polynomial':
        coeffs = fitting_params.get('polynomial_coefficients')
        
        if coeffs:
            poly = Polynomial(coeffs)
            
            def func(x):
                return poly(x)
            
            return func, (x_min, x_max)
        else:
            print(f"Warning: Missing polynomial coefficients for {fitting_params.get('name')}")
            return None, None
    
    return None, None


def load_and_plot(json_path, show=True, output_file=None):
    """
    Loads JSON file and plots 4 graphs.
    
    Args:
        json_path: path to JSON file
        show: whether to show the graph (True) or only save (False)
        output_file: output file name (if None, uses <json_file>.png)
    """
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded transducer data: {data.get('model', 'Unknown')}")
    print(f"BVD Model Type: {data.get('bvd_model_type', 'N/A')}")
    
    # Получаем fitted functions
    fitted_functions = data.get('fitted_functions', {})
    
    if not fitted_functions:
        print("No fitted functions found in JSON")
        return
    
    print(f"\nFound fitted functions:")
    for func_name in ['conductance', 'susceptance', 'rx_sensitivity', 'tx_sensitivity']:
        if func_name in fitted_functions:
            func_data = fitted_functions[func_name]
            if isinstance(func_data, list):
                print(f"  - {func_name}: {len(func_data)} function(s)")
            else:
                print(f"  - {func_name}: 1 function")
        else:
            print(f"  - {func_name}: not found")
    
    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Transducer: {data.get('model', 'Unknown')}", fontsize=16, fontweight='bold')
    
    # Plot 1: Conductance
    ax1 = axes[0, 0]
    if 'conductance' in fitted_functions:
        conductance = fitted_functions['conductance']
        func, (x_min, x_max) = reconstruct_function(conductance)
        if func:
            x = np.linspace(x_min, x_max, 200)
            y = func(x)
            ax1.plot(x, y, 'b-', linewidth=2, label=conductance.get('name', 'Conductance'))
            ax1.set_xlabel('Frequency (kHz)')
            ax1.set_ylabel('Conductance (mS)')
            ax1.set_title('Conductance (G)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'Cannot reconstruct\nConductance function', 
                    ha='center', va='center', transform=ax1.transAxes)
    else:
        ax1.text(0.5, 0.5, 'No Conductance data', 
                ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: Susceptance
    ax2 = axes[0, 1]
    if 'susceptance' in fitted_functions:
        susceptance = fitted_functions['susceptance']
        func, (x_min, x_max) = reconstruct_function(susceptance)
        if func:
            x = np.linspace(x_min, x_max, 200)
            y = func(x)
            ax2.plot(x, y, 'r-', linewidth=2, label=susceptance.get('name', 'Susceptance'))
            ax2.set_xlabel('Frequency (kHz)')
            ax2.set_ylabel('Susceptance (mS)')
            ax2.set_title('Susceptance (B)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Cannot reconstruct\nSusceptance function', 
                    ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'No Susceptance data', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # Plot 3: RX Sensitivity
    ax3 = axes[1, 0]
    if 'rx_sensitivity' in fitted_functions:
        rx_data = fitted_functions['rx_sensitivity']
        # RX can be a single object or a list
        if isinstance(rx_data, list):
            rx_funcs = rx_data
        else:
            rx_funcs = [rx_data]
        
        for rx_func in rx_funcs:
            func, (x_min, x_max) = reconstruct_function(rx_func)
            if func:
                x = np.linspace(x_min, x_max, 200)
                y = func(x)
                ax3.plot(x, y, 'g-', linewidth=2, label=rx_func.get('name', 'RX'))
        
        if any(reconstruct_function(rx_func)[0] for rx_func in rx_funcs):
            ax3.set_xlabel('Frequency (kHz)')
            ax3.set_ylabel('RX Sensitivity (dB)')
            ax3.set_title('RX Sensitivity')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Cannot reconstruct\nRX Sensitivity function', 
                    ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No RX Sensitivity data', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: TX Sensitivity
    ax4 = axes[1, 1]
    if 'tx_sensitivity' in fitted_functions:
        tx_data = fitted_functions['tx_sensitivity']
        # TX can be a single object or a list
        if isinstance(tx_data, list):
            tx_funcs = tx_data
        else:
            tx_funcs = [tx_data]
        
        for tx_func in tx_funcs:
            func, (x_min, x_max) = reconstruct_function(tx_func)
            if func:
                x = np.linspace(x_min, x_max, 200)
                y = func(x)
                ax4.plot(x, y, 'm-', linewidth=2, label=tx_func.get('name', 'TX'))
        
        if any(reconstruct_function(tx_func)[0] for tx_func in tx_funcs):
            ax4.set_xlabel('Frequency (kHz)')
            ax4.set_ylabel('TX Sensitivity (dB)')
            ax4.set_title('TX Sensitivity')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Cannot reconstruct\nTX Sensitivity function', 
                    ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No TX Sensitivity data', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # Save graph to file
    if output_file is None:
        output_file = json_path.with_suffix('.png')
    else:
        output_file = Path(output_file)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nGraph saved to: {output_file}")
    
    # Show graph if needed
    if show:
        print("Displaying graphs... (close the window to continue)")
        plt.show()
    else:
        plt.close()
        print("Graph saved (not displayed)")
    
    # Print parameter information
    print("\n" + "="*60)
    print("Transducer Parameters:")
    print("="*60)
    for key in ['f_0', 'f_min', 'f_max', 'B_tr', 'S_TX', 'S_RX', 'Theta_BW', 'Q', 'Z', 'V_max']:
        if key in data:
            print(f"  {key}: {data[key]}")
    
    if 'bvd_model' in data:
        print("\nBVD Model Parameters:")
        for key, value in data['bvd_model'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize transducer data from JSON file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_transducer_json.py scripts/T257_24.json
  python test_transducer_json.py scripts/T257_24.json --no-show
  python test_transducer_json.py scripts/T257_24.json --output custom_name.png
        """
    )
    parser.add_argument('json_file', type=str, help='Path to JSON file with transducer data')
    parser.add_argument('--no-show', action='store_true', 
                       help='Save graph without displaying (useful for automation)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output PNG file name (default: <json_file>.png)')
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    try:
        load_and_plot(json_path, show=not args.no_show, output_file=args.output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
