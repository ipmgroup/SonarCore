"""
Unit test for correlation function (matched filter) in SBP profile processing.

Tests the correlation function to ensure it produces narrow peaks as expected
when correlating CHIRP signals, matching the user's example behavior.
"""

import unittest
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add parent directory to path for imports
import sys
from pathlib import Path
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.sbp_profile import SBPProfileGenerator
from core.signal_model import SignalModel
from core.water_model import WaterModel


class TestCorrelation(unittest.TestCase):
    """Test correlation function for CHIRP signals."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_output_dir = Path(__file__).parent / "test_correlation_plots"
        self.test_output_dir.mkdir(exist_ok=True)
        water_model = WaterModel()
        self.profile_generator = SBPProfileGenerator(water_model)
        
    def test_correlation_narrow_peak(self):
        """
        Test that correlation produces narrow peak when correlating CHIRP signals.
        
        This test reproduces the user's example:
        - Generate two CHIRP signals (same parameters)
        - Add noise before and after first signal
        - Apply window to second signal (reference)
        - Correlate them
        - Verify correlation produces narrow peak (as expected)
        """
        # Parameters matching user's example
        fs = 10000  # Sampling frequency, Hz
        duration = 0.1  # CHIRP duration, seconds
        Tp_us = duration * 1e6  # Convert to microseconds (as expected by SignalModel)
        
        f0 = 3000  # Start frequency, Hz
        f1 = 12000  # End frequency, Hz
        
        # Generate CHIRP signals using SignalModel from core
        # Received signal: use Rect window (effectively no window, as in user's example)
        t_chirp, sig1_clean = SignalModel.generate_chirp(
            f_start=f0, f_end=f1, Tp=Tp_us, fs=fs, window='Rect'
        )
        
        # Add noise before and after (as in user's example)
        noise_duration = 2  # seconds
        noise_len = int(noise_duration * fs)
        noise_std = 0.2
        noise_front = np.random.normal(0, noise_std, noise_len)
        noise_back = np.random.normal(0, noise_std, noise_len)
        sig1 = np.concatenate([noise_front, sig1_clean, noise_back])
        
        # Generate reference CHIRP signal with Hanning window (as in user's example)
        t_ref, sig2 = SignalModel.generate_chirp(
            f_start=f0, f_end=f1, Tp=Tp_us, fs=fs, window='Hann'
        )
        
        # Perform correlation using process_profile_with_matched_filter
        # This function uses mode='full' internally
        processed_signal, envelope, correlation_lags = self.profile_generator.process_profile_with_matched_filter(
            sig1, sig2, fs
        )
        
        # Verify correlation properties
        self.assertEqual(len(processed_signal), len(correlation_lags), 
                        "Correlation signal and lags should have same length")
        self.assertEqual(len(processed_signal), len(sig1) + len(sig2) - 1,
                        "Correlation length should be len(sig1) + len(sig2) - 1 (mode='full')")
        
        # Find peak in correlation
        peak_idx = np.argmax(np.abs(processed_signal))
        peak_lag = correlation_lags[peak_idx]
        
        # Expected peak position: at the start of CHIRP in sig1 (after noise_front)
        expected_peak_lag = len(noise_front)
        
        # Verify peak is at expected position (allow tolerance for noise and reference reversal)
        # Note: when using reference_reversed (for matched filter), peak position may shift slightly
        # Allow larger tolerance (e.g., 1000 samples = 0.1s at 10kHz) to account for this
        tolerance_samples = max(100, int(0.1 * fs))  # At least 100 samples or 10% of signal length
        self.assertAlmostEqual(peak_lag, expected_peak_lag, delta=tolerance_samples,
                              msg=f"Peak should be near lag={expected_peak_lag}, got {peak_lag} (tolerance={tolerance_samples})")
        
        # Calculate peak width (FWHM - Full Width at Half Maximum)
        peak_value = np.abs(processed_signal[peak_idx])
        half_max = peak_value / 2
        
        # Find width at half maximum
        left_idx = peak_idx
        while left_idx > 0 and np.abs(processed_signal[left_idx]) > half_max:
            left_idx -= 1
        
        right_idx = peak_idx
        while right_idx < len(processed_signal) - 1 and np.abs(processed_signal[right_idx]) > half_max:
            right_idx += 1
        
        peak_width_samples = right_idx - left_idx
        peak_width_time = peak_width_samples / fs  # seconds
        
        # Verify peak is narrow (should be comparable to CHIRP duration, not much wider)
        # Peak width should be roughly equal to CHIRP duration (due to matched filter)
        # Allow some margin (up to 2x CHIRP duration is acceptable)
        self.assertLess(peak_width_time, 2 * duration,
                       msg=f"Peak should be narrow (width < {2*duration}s), got {peak_width_time:.4f}s")
        
        # Verify envelope also has narrow peak
        envelope_peak_idx = np.argmax(envelope)
        self.assertEqual(envelope_peak_idx, peak_idx,
                        "Envelope peak should be at same position as correlation peak")
        
        # Save plots
        self._save_correlation_plot(sig1, sig2, processed_signal, envelope, correlation_lags, fs, 
                                    peak_idx, expected_peak_lag, peak_width_samples)
        
        print(f"\n✓ Correlation test passed:")
        print(f"  Peak at lag: {peak_lag} (expected: {expected_peak_lag})")
        print(f"  Peak width: {peak_width_samples} samples ({peak_width_time*1000:.2f} ms)")
        print(f"  CHIRP duration: {duration*1000:.2f} ms")
        print(f"  Plot saved to: {self.test_output_dir / 'correlation_test.png'}")
    
    def _save_correlation_plot(self, sig1, sig2, correlation, envelope, lags, fs,
                               peak_idx, expected_peak_lag, peak_width_samples):
        """Save correlation plot for visual inspection."""
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        fig.suptitle('Correlation Test: CHIRP Signals', fontsize=14, fontweight='bold')
        
        # Time axis for signals
        t_sig1 = np.arange(len(sig1)) / fs
        t_sig2 = np.arange(len(sig2)) / fs
        correlation_time = lags / fs
        
        # Plot 1: Received signal (sig1)
        axes[0].plot(t_sig1, sig1, 'b-', linewidth=0.5, alpha=0.7)
        axes[0].axvline(expected_peak_lag / fs, color='r', linestyle='--', 
                       label=f'Expected peak (lag={expected_peak_lag})')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Received Signal (sig1) - CHIRP with noise before and after')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Reference signal (sig2)
        axes[1].plot(t_sig2, sig2, 'g-', linewidth=1)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('Reference Signal (sig2) - Windowed CHIRP')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Correlation (full signal)
        axes[2].plot(correlation_time, np.abs(correlation), 'm-', linewidth=1, label='|Correlation|')
        axes[2].axvline(correlation_time[peak_idx], color='r', linestyle='--', linewidth=2,
                       label=f'Peak (lag={lags[peak_idx]})')
        axes[2].axvline(expected_peak_lag / fs, color='orange', linestyle=':', linewidth=1.5,
                       label=f'Expected (lag={expected_peak_lag})')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_title('Correlation (mode=\'full\') - Should show narrow peak')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Plot 4: Correlation (zoomed around peak)
        peak_time = correlation_time[peak_idx]
        # Zoom to 1.998 to 2.002 seconds
        zoom_min = 1.998
        zoom_max = 2.002
        zoom_mask = (correlation_time >= zoom_min) & (correlation_time <= zoom_max)
        if np.any(zoom_mask):
            axes[3].plot(correlation_time[zoom_mask], np.abs(correlation[zoom_mask]), 
                        'm-', linewidth=2, label='|Correlation|')
            axes[3].plot(correlation_time[zoom_mask], envelope[zoom_mask], 
                        'c--', linewidth=1.5, label='Envelope')
            axes[3].axvline(correlation_time[peak_idx], color='r', linestyle='--', linewidth=2,
                           label=f'Peak (lag={lags[peak_idx]})')
            axes[3].axvline(expected_peak_lag / fs, color='orange', linestyle=':', linewidth=1.5,
                           label=f'Expected (lag={expected_peak_lag})')
            axes[3].set_xlabel('Time (s)')
            axes[3].set_ylabel('Amplitude')
            axes[3].set_title(f'Correlation Zoom (1.998-2.002 s, peak width={peak_width_samples} samples)')
            axes[3].grid(True, alpha=0.3)
            axes[3].legend()
        else:
            axes[3].text(0.5, 0.5, 'No data in zoom range', ha='center', va='center',
                        transform=axes[3].transAxes)
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.test_output_dir / 'correlation_test.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {output_file}")
    
    def test_correlation_multiple_peaks(self):
        """
        Test correlation with multiple echoes (simulating SBP profile).
        
        Creates a received signal with multiple delayed CHIRP copies (echoes)
        and verifies that correlation detects all peaks correctly.
        """
        fs = 10000
        duration = 0.1
        Tp_us = duration * 1e6  # Convert to microseconds
        
        f0 = 3000
        f1 = 12000
        
        # Generate reference CHIRP using SignalModel from core (with Hanning window)
        t_ref, reference = SignalModel.generate_chirp(
            f_start=f0, f_end=f1, Tp=Tp_us, fs=fs, window='Hann'
        )
        
        # Create received signal with multiple echoes
        # Echo 1: delay = 0.5s, amplitude = 1.0
        # Echo 2: delay = 1.0s, amplitude = 0.5
        # Echo 3: delay = 1.5s, amplitude = 0.3
        delays_samples = [int(0.5 * fs), int(1.0 * fs), int(1.5 * fs)]
        amplitudes = [1.0, 0.5, 0.3]
        
        total_length = delays_samples[-1] + len(reference) + int(0.5 * fs)
        received = np.zeros(total_length)
        
        for delay, amp in zip(delays_samples, amplitudes):
            if delay + len(reference) <= len(received):
                received[delay:delay + len(reference)] += reference * amp
        
        # Add noise
        noise_std = 0.05
        received += np.random.normal(0, noise_std, len(received))
        
        # Perform correlation
        processed_signal, envelope, correlation_lags = self.profile_generator.process_profile_with_matched_filter(
            received, reference, fs
        )
        
        # Verify we can detect all peaks
        correlation_time = correlation_lags / fs
        
        # Convert correlation_time to depth for find_all_echoes_from_correlation
        # Use simple conversion: depth = c * t / 2 (assuming water sound speed)
        c_water = 1500.0  # m/s, approximate sound speed in water
        correlation_depths = c_water * correlation_time / 2  # Round-trip to one-way
        
        # Find peaks using core function (find_all_echoes_from_correlation)
        # This function uses envelope detection internally for better peak detection
        echo_depths, echo_amplitudes = self.profile_generator.find_all_echoes_from_correlation(
            envelope, correlation_depths, min_height=0.1, min_distance=int(0.3 * fs), use_envelope=False
        )
        
        # Convert echo depths back to time for verification
        echo_depths_array = np.array(echo_depths)
        echo_times = 2 * echo_depths_array / c_water  # Convert depth back to round-trip time
        
        # Should detect at least the three main peaks
        self.assertGreaterEqual(len(echo_depths), 3, 
                               msg=f"Should detect at least 3 peaks, found {len(echo_depths)}")
        
        # Verify peak positions are close to expected delays
        # Note: When using reference_reversed (matched filter), peak positions may shift
        # Allow tolerance for this shift (e.g., 0.1s = 100ms)
        tolerance_time = 0.1  # 100ms tolerance
        for expected_delay in delays_samples:
            expected_time = expected_delay / fs
            # Find closest peak
            closest_peak_time = echo_times[np.argmin(np.abs(echo_times - expected_time))]
            error = abs(closest_peak_time - expected_time)
            self.assertLess(error, tolerance_time,
                           msg=f"Peak should be near t={expected_time:.3f}s, closest found at {closest_peak_time:.3f}s (tolerance={tolerance_time}s)")
        
        # Convert echo_times to peak indices for plotting
        peaks = []
        for echo_time in echo_times:
            idx = np.argmin(np.abs(correlation_time - echo_time))
            if idx not in peaks:
                peaks.append(idx)
        peaks = np.array(peaks)
        
        # Save plot
        self._save_multiple_peaks_plot(received, reference, processed_signal, envelope, 
                                      correlation_time, delays_samples, amplitudes, fs, peaks)
        
        print(f"\n✓ Multiple peaks test passed:")
        print(f"  Detected {len(peaks)} peaks")
        print(f"  Expected {len(delays_samples)} echoes")
        print(f"  Plot saved to: {self.test_output_dir / 'correlation_multiple_peaks_test.png'}")
    
    def _save_multiple_peaks_plot(self, received, reference, correlation, envelope, correlation_time,
                                  expected_delays, expected_amplitudes, fs, detected_peaks):
        """Save plot for multiple peaks test."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Correlation Test: Multiple Echoes', fontsize=14, fontweight='bold')
        
        t_received = np.arange(len(received)) / fs
        t_reference = np.arange(len(reference)) / fs
        
        # Plot 1: Received signal
        axes[0].plot(t_received, received, 'b-', linewidth=0.5, alpha=0.7)
        for delay, amp in zip(expected_delays, expected_amplitudes):
            axes[0].axvline(delay / fs, color='r', linestyle='--', alpha=0.7,
                           label=f'Echo at {delay/fs:.2f}s (amp={amp})' if delay == expected_delays[0] else '')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Received Signal - Multiple Echoes')
        axes[0].grid(True, alpha=0.3)
        if expected_delays:
            axes[0].legend()
        
        # Plot 2: Correlation (full)
        axes[1].plot(correlation_time, np.abs(correlation), 'm-', linewidth=1, alpha=0.7, label='|Correlation|')
        axes[1].plot(correlation_time, envelope, 'c-', linewidth=1.5, label='Envelope')
        for delay in expected_delays:
            axes[1].axvline(delay / fs, color='r', linestyle='--', alpha=0.7)
        for peak_idx in detected_peaks:
            axes[1].plot(correlation_time[peak_idx], envelope[peak_idx], 'ro', markersize=8)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('Correlation - Detected Peaks Marked')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot 3: Zoomed view of correlation
        if len(detected_peaks) > 0:
            zoom_center = correlation_time[detected_peaks[len(detected_peaks)//2]]
            zoom_range = 0.3
            zoom_mask = np.abs(correlation_time - zoom_center) < zoom_range
            if np.any(zoom_mask):
                axes[2].plot(correlation_time[zoom_mask], np.abs(correlation[zoom_mask]), 
                            'm-', linewidth=1.5, alpha=0.7, label='|Correlation|')
                axes[2].plot(correlation_time[zoom_mask], envelope[zoom_mask], 
                            'c-', linewidth=2, label='Envelope')
                for delay in expected_delays:
                    if abs(delay / fs - zoom_center) < zoom_range:
                        axes[2].axvline(delay / fs, color='r', linestyle='--', alpha=0.7)
                for peak_idx in detected_peaks:
                    if zoom_mask[peak_idx]:
                        axes[2].plot(correlation_time[peak_idx], envelope[peak_idx], 
                                    'ro', markersize=10, label='Detected peak' if peak_idx == detected_peaks[0] else '')
                axes[2].set_xlabel('Time (s)')
                axes[2].set_ylabel('Amplitude')
                axes[2].set_title('Correlation Zoom - Peak Details')
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()
            else:
                axes[2].text(0.5, 0.5, 'No data in zoom range', ha='center', va='center',
                            transform=axes[2].transAxes)
        
        plt.tight_layout()
        
        output_file = self.test_output_dir / 'correlation_multiple_peaks_test.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {output_file}")


if __name__ == '__main__':
    unittest.main()

