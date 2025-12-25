"""
Simulator - main simulator class.
"""

import logging
import numpy as np
from typing import Optional
from scipy import signal as scipy_signal
from .dto import InputDTO, OutputDTO, RecommendationsDTO
from .water_model import WaterModel
from .transducer_model import TransducerModel
from .signal_model import SignalModel
from .channel_model import ChannelModel
from .receiver_model import ReceiverModel
from .dsp_model import DSPModel
from .range_estimator import RangeEstimator
from .optimizer import Optimizer


class Simulator:
    """
    Main hydroacoustic sonar simulator class.
    
    Coordinates all subsystems.
    """
    
    # Fixed sampling frequency for visualization (to avoid signal shape changes when ADC changes)
    # Use high frequency (10 MHz) to ensure smooth visualization regardless of ADC selection
    VISUALIZATION_FS = 10e6  # Hz
    
    def __init__(self, data_provider):
        """
        Initialize simulator.
        
        Args:
            data_provider: Data provider (DATA module)
        """
        self.data_provider = data_provider
        self.water_model = WaterModel()
        self.optimizer = Optimizer(data_provider)  # Pass data_provider to optimizer
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Cache for models
        self._transducer_cache = {}
        self._lna_cache = {}
        self._vga_cache = {}
        self._adc_cache = {}
    
    def _get_transducer(self, transducer_id: str) -> TransducerModel:
        """Get transducer model."""
        if transducer_id not in self._transducer_cache:
            params = self.data_provider.get_transducer(transducer_id)
            self._transducer_cache[transducer_id] = TransducerModel(params)
        return self._transducer_cache[transducer_id]
    
    def _get_lna(self, lna_id: str) -> dict:
        """Get LNA parameters."""
        if lna_id not in self._lna_cache:
            self._lna_cache[lna_id] = self.data_provider.get_lna(lna_id)
        return self._lna_cache[lna_id]
    
    def _get_vga(self, vga_id: str) -> dict:
        """Get VGA parameters."""
        if vga_id not in self._vga_cache:
            self._vga_cache[vga_id] = self.data_provider.get_vga(vga_id)
        return self._vga_cache[vga_id]
    
    def _get_adc(self, adc_id: str) -> dict:
        """Get ADC parameters."""
        if adc_id not in self._adc_cache:
            self._adc_cache[adc_id] = self.data_provider.get_adc(adc_id)
        return self._adc_cache[adc_id]
    
    def _validate_input(self, input_dto: InputDTO) -> list:
        """
        Validates input data.
        
        Returns:
            List of errors (empty if OK)
        """
        errors = []
        
        # Get hardware parameters
        try:
            transducer = self._get_transducer(input_dto.hardware.transducer_id)
            lna = self._get_lna(input_dto.hardware.lna_id)
            vga = self._get_vga(input_dto.hardware.vga_id)
            adc = self._get_adc(input_dto.hardware.adc_id)
        except Exception as e:
            errors.append(f"Error loading hardware parameters: {e}")
            return errors
        
        # CHIRP frequency check
        if not transducer.validate_frequency(input_dto.signal.f_start):
            errors.append(f"f_start ({input_dto.signal.f_start} Hz) is outside transducer range [{transducer.f_min}, {transducer.f_max}] Hz")
        
        if not transducer.validate_frequency(input_dto.signal.f_end):
            errors.append(f"f_end ({input_dto.signal.f_end} Hz) is outside transducer range [{transducer.f_min}, {transducer.f_max}] Hz")
        
        # CHIRP bandwidth check
        bandwidth = SignalModel.get_bandwidth(input_dto.signal.f_start, input_dto.signal.f_end)
        if bandwidth > transducer.B_tr:
            errors.append(f"CHIRP bandwidth ({bandwidth} Hz) exceeds transducer bandwidth ({transducer.B_tr} Hz)")
        
        # Sampling frequency check (signal sample_rate vs ADC f_s)
        if input_dto.signal.sample_rate < 2 * input_dto.signal.f_end:
            errors.append(f"Signal sampling frequency ({input_dto.signal.sample_rate} Hz) must be >= 2*f_end ({2*input_dto.signal.f_end} Hz)")
        
        # ADC sampling frequency check (should match or be compatible with signal sample_rate)
        if adc['f_s'] < 2 * input_dto.signal.f_end:
            errors.append(f"ADC sampling frequency ({adc['f_s']} Hz) must be >= 2*f_end ({2*input_dto.signal.f_end} Hz)")
        
        return errors
    
    def simulate(self, input_dto: InputDTO, iteration: int = 0, absorption_only: bool = True, 
                 vga_gain: Optional[float] = None) -> OutputDTO:
        """
        Performs simulation.
        
        Args:
            input_dto: Input DTO
            iteration: Iteration number (for logging)
        
        Returns:
            Output DTO with results
        """
        self.logger.info(f"Starting simulation, iteration {iteration}")
        self.logger.info(f"Input parameters: {input_dto.model_dump_json()}")
        
        # Validation
        errors = self._validate_input(input_dto)
        if errors:
            output_dto = OutputDTO(
                D_measured=0.0,
                sigma_D=0.0,
                SNR_ADC=0.0,
                success=False,
                errors=errors
            )
            return output_dto
        
        try:
            # Get hardware parameters
            transducer = self._get_transducer(input_dto.hardware.transducer_id)
            lna_params = self._get_lna(input_dto.hardware.lna_id)
            vga_params = self._get_vga(input_dto.hardware.vga_id)
            adc_params = self._get_adc(input_dto.hardware.adc_id)
            
            # Initialize models
            receiver = ReceiverModel(lna_params, vga_params, adc_params)
            # DSPModel must use ADC f_s because:
            # - In real system, ADC digitizes signal at its sampling rate
            # - DSP processes already digitized signal at ADC sampling rate
            # - Signal generation and reception can work at different frequencies
            # - We will resample received_signal and reference_signal to ADC f_s before DSP processing
            dsp = DSPModel(adc_params['f_s'])
            channel = ChannelModel(self.water_model)
            range_estimator = RangeEstimator(dsp, self.water_model)
            
            # Generate CHIRP signal for visualization with fixed sampling frequency
            # This ensures signal shape doesn't change when ADC is changed
            # ADC only affects quantization, not signal generation for visualization
            t_ref_vis, reference_signal_vis = SignalModel.generate_chirp(
                input_dto.signal.f_start,
                input_dto.signal.f_end,
                input_dto.signal.Tp,
                self.VISUALIZATION_FS,  # Fixed frequency for visualization
                input_dto.signal.window
            )
            
            # Generate CHIRP signal for actual simulation with signal sample_rate
            # This is used for range estimation and receiver processing
            t_ref, reference_signal = SignalModel.generate_chirp(
                input_dto.signal.f_start,
                input_dto.signal.f_end,
                input_dto.signal.Tp,
                input_dto.signal.sample_rate,  # Signal sampling frequency from input
                input_dto.signal.window
            )
            
            # Model transmission through transducer
            # Simplified: apply TX sensitivity
            f_center = (input_dto.signal.f_start + input_dto.signal.f_end) / 2
            S_TX = transducer.get_tx_sensitivity(f_center)
            tx_signal = reference_signal * (10 ** (S_TX / 20))
            
            # Model channel (for target range)
            # Use D_target if specified, otherwise use average of D_min and D_max
            if input_dto.range.D_target is not None:
                D_target = input_dto.range.D_target
            else:
                D_target = (input_dto.range.D_min + input_dto.range.D_max) / 2
            
            # Validate D_target is within [D_min, D_max]
            if D_target < input_dto.range.D_min:
                self.logger.warning(f"D_target ({D_target}) < D_min ({input_dto.range.D_min}). Using D_min.")
                D_target = input_dto.range.D_min
            elif D_target > input_dto.range.D_max:
                self.logger.warning(f"D_target ({D_target}) > D_max ({input_dto.range.D_max}). Using D_max.")
                D_target = input_dto.range.D_max
            
            self.logger.info(f"Simulation: D_min={input_dto.range.D_min}, D_max={input_dto.range.D_max}, D_target={D_target}")
            
            # For visualization: three stages of signal propagation
            # Use fixed sampling frequency (VISUALIZATION_FS) to ensure signal shape doesn't change
            # when ADC is changed. ADC only affects quantization, not signal generation.
            # Stage 1: Original CHIRP signal (reference_signal_vis) - will be saved as tx_signal for display
            # Stage 2: Signal after passing through water forward (TX -> target) - only attenuation, no delay
            # Stage 3: Signal after passing through water backward (target -> RX) - only attenuation, no delay
            # All signals have the same length for visualization (same time axis)
            
            # Stage 2: Signal after passing through water forward (one-way, attenuation only)
            # Apply frequency-dependent attenuation for CHIRP signal
            # Use absorption_only parameter to control visualization mode
            signal_after_water_forward = channel.apply_attenuation_only(
                reference_signal_vis,  # Use visualization signal (fixed f_s)
                D_target,  # Distance to target
                f_center,
                input_dto.environment.T,
                input_dto.environment.S,
                input_dto.environment.z,
                round_trip=False,  # Only forward: TX -> target
                f_start=input_dto.signal.f_start,  # CHIRP start frequency
                f_end=input_dto.signal.f_end,  # CHIRP end frequency
                t=t_ref_vis,  # Time axis (visualization)
                Tp=input_dto.signal.Tp,  # Pulse duration
                absorption_only=absorption_only  # Show only absorption or full attenuation
            )
            
            # Stage 3: Signal after passing through water backward (one-way, attenuation only)
            # Take signal from stage 2 and pass it through water again (backward)
            # Apply frequency-dependent attenuation for CHIRP signal
            signal_after_water_backward = channel.apply_attenuation_only(
                signal_after_water_forward,  # Signal from stage 2
                D_target,  # Distance back from target
                f_center,
                input_dto.environment.T,
                input_dto.environment.S,
                input_dto.environment.z,
                round_trip=False,  # Only backward: target -> RX
                f_start=input_dto.signal.f_start,  # CHIRP start frequency
                f_end=input_dto.signal.f_end,  # CHIRP end frequency
                t=t_ref_vis,  # Time axis (visualization)
                Tp=input_dto.signal.Tp,  # Pulse duration
                absorption_only=absorption_only  # Show only absorption or full attenuation
            )
            
            # For actual simulation: use signal with TX and RX sensitivity
            # Signal at bottom (after channel, before reflection) - for path calculation
            signal_at_bottom = channel.apply_channel_response(
                tx_signal,
                input_dto.range.D_max,  # Distance to bottom (one-way)
                f_center,
                input_dto.environment.T,
                input_dto.environment.S,
                input_dto.environment.z,
                input_dto.signal.sample_rate,  # Use signal sample_rate
                round_trip=False  # Only one-way: TX -> bottom
            )
            
            # Received signal (after round trip: to bottom and back)
            # D_target is one-way distance, round_trip=True means signal travels 2*D_target
            self.logger.info(f"Creating received_signal: D_target={D_target:.2f}m, round_trip=True, "
                           f"expected round_trip_distance={2*D_target:.2f}m")
            received_signal = channel.apply_channel_response(
                tx_signal,
                D_target,
                f_center,
                input_dto.environment.T,
                input_dto.environment.S,
                input_dto.environment.z,
                input_dto.signal.sample_rate,  # Use signal sample_rate
                round_trip=True  # Round trip: TX -> target -> RX
            )
            
            # Apply RX sensitivity
            S_RX = transducer.get_rx_sensitivity(f_center)
            received_signal = received_signal * (10 ** (S_RX / 20))
            
            # Resample signals to ADC sampling rate
            # In real system, ADC digitizes signal at its sampling rate (adc.f_s)
            # Signal generation (signal.sample_rate) and ADC sampling (adc.f_s) can be different
            adc_fs = adc_params['f_s']
            signal_fs = input_dto.signal.sample_rate
            
            if signal_fs != adc_fs:
                # Resample received_signal to ADC sampling rate
                # Calculate number of samples at ADC rate
                t_received = len(received_signal) / signal_fs  # Time duration
                num_samples_adc = int(t_received * adc_fs)
                received_signal_adc = scipy_signal.resample(received_signal, num_samples_adc)
                
                # Resample reference_signal to ADC sampling rate for DSP
                t_reference = len(reference_signal) / signal_fs  # Time duration
                num_samples_ref_adc = int(t_reference * adc_fs)
                reference_signal_adc = scipy_signal.resample(reference_signal, num_samples_ref_adc)
                
                self.logger.info(f"Resampling: signal_fs={signal_fs:.0f}Hz -> adc_fs={adc_fs:.0f}Hz, "
                               f"received: {len(received_signal)} -> {len(received_signal_adc)} samples, "
                               f"reference: {len(reference_signal)} -> {len(reference_signal_adc)} samples")
            else:
                # No resampling needed
                received_signal_adc = received_signal
                reference_signal_adc = reference_signal
            
            # Receiver processing (on signal at ADC sampling rate)
            # Set VGA gain from parameter or use average
            if vga_gain is not None:
                receiver.set_vga_gain(vga_gain)
            else:
                # Default: use average of min and max
                receiver.set_vga_gain((vga_params['G_min'] + vga_params['G_max']) / 2)
            
            digital_signal, snr_adc, clipping, signal_after_lna, signal_after_vga = receiver.process_signal(received_signal_adc, add_noise=True)
            
            # Range estimation (using signals at ADC sampling rate)
            self.logger.info(f"Estimating range: received_signal length={len(digital_signal)}, "
                           f"reference_signal length={len(reference_signal_adc)}, "
                           f"expected D_target={D_target:.2f}m, adc_fs={adc_fs:.0f}Hz")
            D_measured, sigma_D = range_estimator.estimate_range(
                digital_signal,
                reference_signal_adc,
                input_dto.environment.T,
                input_dto.environment.S,
                input_dto.environment.z,
                snr_adc
            )
            if D_measured is not None:
                self.logger.info(f"Range estimation result: D_measured={D_measured:.2f}m, "
                               f"expected={D_target:.2f}m, difference={abs(D_measured-D_target):.2f}m")
            
            if D_measured is None:
                D_measured = 0.0
                sigma_D = 0.0
            
            # Create output DTO with signal data for visualization
            # Five stages for visualization:
            # 1. tx_signal: Original CHIRP signal (reference_signal_vis - fixed f_s for visualization)
            # 2. signal_at_bottom: Signal after passing through water forward
            # 3. received_signal: Signal after passing through water backward (after RX sensitivity)
            # 4. signal_after_lna: Signal after LNA (resampled to visualization f_s)
            # 5. signal_after_vga: Signal after VGA (resampled to visualization f_s)
            # All signals use fixed sampling frequency (VISUALIZATION_FS) for consistent visualization
            ref_len_vis = len(reference_signal_vis)
            
            # Resample signals after LNA and VGA to visualization sampling frequency
            # They were generated with signal sample_rate, so we need to resample them
            if len(signal_after_lna) > 0:
                # Resample from signal sample_rate to visualization f_s
                num_samples_vis = int(len(signal_after_lna) * self.VISUALIZATION_FS / input_dto.signal.sample_rate)
                signal_after_lna_vis = scipy_signal.resample(signal_after_lna, num_samples_vis)
                # Trim to reference length
                signal_after_lna_vis = signal_after_lna_vis[-ref_len_vis:] if len(signal_after_lna_vis) >= ref_len_vis else signal_after_lna_vis
            else:
                signal_after_lna_vis = signal_after_lna
                
            if len(signal_after_vga) > 0:
                # Resample from signal sample_rate to visualization f_s
                num_samples_vis = int(len(signal_after_vga) * self.VISUALIZATION_FS / input_dto.signal.sample_rate)
                signal_after_vga_vis = scipy_signal.resample(signal_after_vga, num_samples_vis)
                # Trim to reference length
                signal_after_vga_vis = signal_after_vga_vis[-ref_len_vis:] if len(signal_after_vga_vis) >= ref_len_vis else signal_after_vga_vis
            else:
                signal_after_vga_vis = signal_after_vga
            
            # Calculate signal attenuation values (all calculations in Core)
            # Attenuation at bottom (relative to TX signal)
            reference_max = np.max(np.abs(reference_signal_vis)) if len(reference_signal_vis) > 0 else 1.0
            epsilon = 1e-10
            attenuation_at_bottom_db = None
            if len(signal_after_water_forward) > 0:
                signal_max_bottom = np.max(np.abs(signal_after_water_forward))
                if signal_max_bottom > epsilon and reference_max > epsilon:
                    attenuation_at_bottom_db = 20 * np.log10(signal_max_bottom / reference_max)
            
            # Attenuation at receiver (relative to TX signal)
            attenuation_received_db = None
            if len(signal_after_water_backward) > 0:
                signal_max_received = np.max(np.abs(signal_after_water_backward))
                if signal_max_received > epsilon and reference_max > epsilon:
                    attenuation_received_db = 20 * np.log10(signal_max_received / reference_max)
            
            # Convert numpy arrays to lists for JSON serialization
            output_dto = OutputDTO(
                D_measured=D_measured,
                sigma_D=sigma_D,
                SNR_ADC=snr_adc,
                clipping_flags=clipping,
                lna_gain=receiver.G_LNA,  # LNA gain used in simulation
                vga_gain=receiver.current_G_VGA,  # VGA gain used in simulation
                vga_gain_max=receiver.G_VGA_max,  # VGA maximum gain
                tx_signal=reference_signal_vis.tolist() if len(reference_signal_vis) > 0 else None,  # Stage 1: Original CHIRP (visualization)
                signal_at_bottom=signal_after_water_forward.tolist() if len(signal_after_water_forward) > 0 else None,  # Stage 2: After water forward
                received_signal=signal_after_water_backward.tolist() if len(signal_after_water_backward) > 0 else None,  # Stage 3: After water backward
                signal_after_lna=signal_after_lna_vis.tolist() if len(signal_after_lna_vis) > 0 else None,  # Stage 4: After LNA (resampled)
                signal_after_vga=signal_after_vga_vis.tolist() if len(signal_after_vga_vis) > 0 else None,  # Stage 5: After VGA (resampled)
                time_axis=t_ref_vis.tolist() if len(t_ref_vis) > 0 else None,  # Visualization time axis
                attenuation_at_bottom_db=attenuation_at_bottom_db,  # Calculated in Core
                attenuation_received_db=attenuation_received_db  # Calculated in Core
            )
            
            # Analyze results and generate recommendations
            recommendations = self.optimizer.analyze_results(input_dto, output_dto)
            output_dto.recommendations = recommendations
            
            self.logger.info(f"Simulation results: D={D_measured:.2f} m, σ_D={sigma_D:.4f} m, SNR={snr_adc:.2f} dB")
            
            return output_dto
            
        except Exception as e:
            self.logger.error(f"Simulation error: {e}", exc_info=True)
            output_dto = OutputDTO(
                D_measured=0.0,
                sigma_D=0.0,
                SNR_ADC=0.0,
                success=False,
                errors=[f"Simulation error: {str(e)}"]
            )
            return output_dto

