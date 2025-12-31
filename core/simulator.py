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
from .enob_calculator import ENOBCalculator
from .sediment_model import SedimentModel, SedimentLayer
from .sbp_profile import SBPProfileGenerator
from .signal_path import SignalPathCalculator


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
        self.signal_path_calculator = SignalPathCalculator(data_provider)
        
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
            
            # Get limit_tp_for_fast_calculation flag from input_dto
            limit_tp = getattr(input_dto, 'limit_tp_for_fast_calculation', False)
            
            # Log if limit is applied (for debugging)
            if limit_tp and input_dto.signal.Tp > 1_000_000:  # Tp > 1 second
                original_tp_sec = input_dto.signal.Tp / 1e6
                self.logger.info(f"[FAST CALC] Tp limited to 1.0 s (original: {original_tp_sec:.2f} s). "
                                f"Signal samples reduced from ~{int(input_dto.signal.sample_rate * original_tp_sec):,} "
                                f"to ~{int(input_dto.signal.sample_rate * 1.0):,} samples.")
            
            # Generate CHIRP signal for visualization with fixed sampling frequency
            # This ensures signal shape doesn't change when ADC is changed
            # ADC only affects quantization, not signal generation for visualization
            t_ref_vis, reference_signal_vis = SignalModel.generate_chirp(
                input_dto.signal.f_start,
                input_dto.signal.f_end,
                input_dto.signal.Tp,
                self.VISUALIZATION_FS,  # Fixed frequency for visualization
                input_dto.signal.window,
                limit_tp_for_fast_calc=limit_tp
            )
            
            # Generate CHIRP signal for actual simulation with signal sample_rate
            # This is used for range estimation and receiver processing
            t_ref, reference_signal = SignalModel.generate_chirp(
                input_dto.signal.f_start,
                input_dto.signal.f_end,
                input_dto.signal.Tp,
                input_dto.signal.sample_rate,  # Signal sampling frequency from input
                input_dto.signal.window,
                limit_tp_for_fast_calc=limit_tp
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
            
            # Get ADC parameters for output (reuse adc_params from earlier if available, otherwise get it)
            # Note: adc_params is already loaded above when creating receiver, but we reload here for clarity
            adc_params_output = self._get_adc(input_dto.hardware.adc_id)
            adc_full_scale = adc_params_output.get('V_FS', 2.0)
            adc_range = adc_full_scale / 2.0  # ±V_FS/2
            # ADC resolution: try 'N' first (used by ReceiverModel), then 'bits'
            adc_bits = adc_params_output.get('N', adc_params_output.get('bits', 16))
            adc_dynamic_range_db = 20 * np.log10(2 ** adc_bits)  # dB
            
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
            # They were processed at ADC sampling rate, so we need to resample them
            # Find echo start (TOF) and show only echo, not echo with delay
            # Calculate TOF for D_target (round trip)
            c = self.water_model.calculate_sound_speed(
                input_dto.environment.T,
                input_dto.environment.S,
                self.water_model.calculate_pressure(input_dto.environment.z)
            )
            tof_round_trip = 2 * D_target / c  # Round trip time of flight
            # Calculate delay in samples at ADC sampling rate (where signals were processed)
            delay_samples_adc = int(tof_round_trip * adc_fs)  # Delay in ADC samples
            Tp_samples_adc = int(input_dto.signal.Tp * 1e-6 * adc_fs)  # Echo duration in ADC samples
            
            if len(signal_after_lna) > 0:
                # Extract only echo (from delay_samples_adc to delay_samples_adc + Tp_samples_adc)
                # This is done BEFORE resampling to avoid interpolation artifacts
                if delay_samples_adc < len(signal_after_lna):
                    echo_end = min(delay_samples_adc + Tp_samples_adc, len(signal_after_lna))
                    signal_after_lna_echo = signal_after_lna[delay_samples_adc:echo_end]
                    # Now resample the echo to visualization f_s
                    num_samples_vis = int(len(signal_after_lna_echo) * self.VISUALIZATION_FS / adc_fs)
                    if num_samples_vis > 0:
                        signal_after_lna_vis = scipy_signal.resample(signal_after_lna_echo, num_samples_vis)
                    else:
                        signal_after_lna_vis = np.array([])
                else:
                    # If delay is beyond signal length, return empty
                    signal_after_lna_vis = np.array([])
            else:
                signal_after_lna_vis = signal_after_lna
                
            if len(signal_after_vga) > 0:
                # Extract only echo (from delay_samples_adc to delay_samples_adc + Tp_samples_adc)
                # This is done BEFORE resampling to avoid interpolation artifacts
                if delay_samples_adc < len(signal_after_vga):
                    echo_end = min(delay_samples_adc + Tp_samples_adc, len(signal_after_vga))
                    signal_after_vga_echo = signal_after_vga[delay_samples_adc:echo_end]
                    # Now resample the echo to visualization f_s
                    num_samples_vis = int(len(signal_after_vga_echo) * self.VISUALIZATION_FS / adc_fs)
                    if num_samples_vis > 0:
                        signal_after_vga_vis = scipy_signal.resample(signal_after_vga_echo, num_samples_vis)
                    else:
                        signal_after_vga_vis = np.array([])
                else:
                    # If delay is beyond signal length, return empty
                    signal_after_vga_vis = np.array([])
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
            
            # Generate SBP profile if enabled
            profile_amplitudes = None
            profile_depths = None
            profile_time_axis = None
            profile_raw_signal = None  # Raw profile signal before matched filter
            profile_correlation = None  # Correlation signal (matched filter output)
            profile_water_depth = None  # Water depth for filtering
            interface_depths = None
            interface_amplitudes = None
            max_penetration_depth = None
            vertical_resolution = None
            sediment_model = None  # Initialize for use in signal_path calculation
            
            if getattr(input_dto, 'enable_sbp', False) and input_dto.sediment_profile:
                try:
                    # Create sediment model from DTO
                    sediment_layers = []
                    for layer_dto in input_dto.sediment_profile.layers:
                        layer = SedimentLayer(
                            thickness=layer_dto.thickness,
                            density=layer_dto.density,
                            sound_speed=layer_dto.sound_speed,
                            attenuation=layer_dto.attenuation,
                            name=layer_dto.name
                        )
                        sediment_layers.append(layer)
                    
                    sediment_model = SedimentModel(sediment_layers)
                    
                    # Create profile generator with DSPModel for matched filtering
                    # Use the same DSPModel that was created earlier for consistency
                    profile_generator = SBPProfileGenerator(self.water_model, dsp_model=dsp)
                    
                    # Generate profile
                    water_depth = input_dto.sediment_profile.water_depth if input_dto.sediment_profile.water_depth > 0 else input_dto.environment.z
                    profile_water_depth = water_depth  # Store for filtering in GUI
                    
                    # Calculate center frequency for profile generation
                    f_center_profile = (input_dto.signal.f_start + input_dto.signal.f_end) / 2
                    
                    profile_signal, profile_depths_array, interface_depths_list, interface_amps_list = profile_generator.generate_profile(
                        reference_signal=reference_signal,
                        t_reference=t_ref,
                        sediment_model=sediment_model,
                        water_depth=water_depth,
                        T=input_dto.environment.T,
                        S=input_dto.environment.S,
                        z=input_dto.environment.z,
                        fs=input_dto.signal.sample_rate,
                        f_center=f_center_profile,
                        apply_tvg=True
                    )
                    
                    # Log theoretical interface amplitudes for debugging
                    self.logger.info(f"SBP theoretical interfaces: {len(interface_depths_list)} interfaces")
                    if len(interface_amps_list) > 0:
                        max_theoretical_amp = max(interface_amps_list)
                        for i, (depth, amp) in enumerate(zip(interface_depths_list, interface_amps_list)):
                            amp_relative = amp / max_theoretical_amp if max_theoretical_amp > 0 else 0.0
                            full_depth = water_depth + depth
                            # Check if amplitude is above detection threshold (1% of max)
                            detectable = amp_relative >= 0.01
                            self.logger.info(f"  Interface {i}: depth_in_sediment={depth:.2f}m (full_depth={full_depth:.2f}m), "
                                           f"amplitude={amp:.6e}, relative_to_max={amp_relative:.4f} ({amp_relative*100:.2f}%), "
                                           f"detectable={'YES' if detectable else 'NO (below 1% threshold)'}")
                    
                    # Apply receiver chain to profile signal (same as regular echosounder)
                    # profile_signal is in relative amplitude units (after all losses)
                    # Need to convert to pressure, then to voltage, then through receiver chain
                    
                    # Step 1: Scale to source pressure level
                    # Get TX sensitivity and voltage to calculate source level
                    S_TX = transducer.get_tx_sensitivity(f_center_profile)
                    # Get transducer parameters from data provider
                    transducer_params = self.data_provider.get_transducer(input_dto.hardware.transducer_id)
                    tx_voltage = transducer_params.get('V_max')
                    if tx_voltage is None:
                        tx_voltage = transducer_params.get('V_nominal')
                    if tx_voltage is None:
                        tx_voltage = transducer_params.get('V', 100.0)
                    # Source Level (SL) = S_TX + 20*log10(V)
                    # At 1m: pressure = 10^((S_TX + 20*log10(V))/20) = V * 10^(S_TX/20) (linear)
                    source_pressure_at_1m = tx_voltage * (10 ** (S_TX / 20))  # Relative units (normalized to reference)
                    
                    # profile_signal is already scaled by losses, so multiply by source level
                    # to get pressure at receiver (in relative units, normalized to 1µPa reference)
                    profile_signal_pressure = profile_signal * source_pressure_at_1m
                    
                    # Step 2: Apply RX sensitivity to convert pressure to voltage
                    S_RX = transducer.get_rx_sensitivity(f_center_profile)
                    profile_signal_voltage = profile_signal_pressure * (10 ** (S_RX / 20))
                    
                    # Step 3: Resample to ADC sampling rate (same as regular echosounder)
                    adc_fs = adc_params['f_s']
                    signal_fs = input_dto.signal.sample_rate
                    
                    if signal_fs != adc_fs:
                        t_profile = len(profile_signal_voltage) / signal_fs
                        num_samples_adc = int(t_profile * adc_fs)
                        profile_signal_adc = scipy_signal.resample(profile_signal_voltage, num_samples_adc)
                        # Also resample profile_depths_array to match
                        profile_depths_adc = np.interp(
                            np.linspace(0, len(profile_depths_array)-1, num_samples_adc),
                            np.arange(len(profile_depths_array)),
                            profile_depths_array
                        )
                    else:
                        profile_signal_adc = profile_signal_voltage
                        profile_depths_adc = profile_depths_array
                    
                    # Step 4: Process through receiver chain (LNA, VGA, ADC)
                    # Set VGA gain from parameter or use average
                    if vga_gain is not None:
                        receiver.set_vga_gain(vga_gain)
                    else:
                        receiver.set_vga_gain((vga_params['G_min'] + vga_params['G_max']) / 2)
                    
                    # Process signal through receiver
                    profile_signal_digital, snr_adc_profile, clipping_profile, signal_after_lna_profile, signal_after_vga_profile = receiver.process_signal(
                        profile_signal_adc, add_noise=True
                    )
                    
                    # For display: use signal at ADC input (signal_after_vga) - this is what ADC sees
                    profile_signal_at_adc = signal_after_vga_profile
                    
                    # Resample back to original sampling rate for display (if needed)
                    if signal_fs != adc_fs:
                        num_samples_orig = len(profile_signal)
                        profile_signal_at_adc_display = scipy_signal.resample(profile_signal_at_adc, num_samples_orig)
                        profile_depths_display = profile_depths_array  # Use original depths
                    else:
                        profile_signal_at_adc_display = profile_signal_at_adc
                        profile_depths_display = profile_depths_array
                    
                    # Process with matched filter (on signal at ADC input)
                    # Resample reference_signal to match profile_signal_at_adc_display sampling rate
                    if signal_fs != adc_fs:
                        t_ref_len = len(reference_signal) / signal_fs
                        num_samples_ref_display = int(t_ref_len * signal_fs)
                        reference_signal_display = scipy_signal.resample(reference_signal, num_samples_ref_display)
                    else:
                        reference_signal_display = reference_signal
                    
                    # Log sampling rates for debugging correlation depth issues
                    self.logger.info(f"SBP correlation: signal_fs={signal_fs:.0f}Hz, adc_fs={adc_fs:.0f}Hz, "
                                   f"profile_signal_at_adc_display length={len(profile_signal_at_adc_display)}, "
                                   f"reference_signal_display length={len(reference_signal_display)}")
                    
                    processed_signal, envelope, correlation_lags = profile_generator.process_profile_with_matched_filter(
                        profile_signal_at_adc_display, reference_signal_display, signal_fs
                    )
                    
                    # Convert correlation lags to time axis, then to depth axis
                    # correlation_lags are lag indices (shifts in samples)
                    # lag = k means reference[0] aligns with profile_signal[k]
                    # Time for correlation: t = lag / fs (where lag is the shift)
                    # NOTE: Using signal_fs because correlation was performed on signals at signal_fs
                    # (after resampling back from adc_fs if needed)
                    correlation_time = correlation_lags / signal_fs
                    
                    # Log for debugging correlation depth calculation
                    if len(correlation_time) > 0:
                        self.logger.info(f"SBP correlation time: min={np.min(correlation_time):.6f}s, "
                                       f"max={np.max(correlation_time):.6f}s, "
                                       f"using signal_fs={signal_fs:.0f}Hz for time calculation")
                    
                    # Convert time to depth using water sound speed
                    # Use c_water for correlation depths (correlation happens in water column)
                    c_water = self.water_model.calculate_sound_speed(
                        input_dto.environment.T, 
                        input_dto.environment.S, 
                        self.water_model.calculate_pressure(input_dto.environment.z)
                    )
                    
                    # Convert correlation_time to depths: depth = c * t / 2 (round-trip to one-way)
                    # For negative times (negative lags), set depth to 0
                    # Use np.where to handle negative times, then clamp to ensure no negative depths
                    correlation_depths = np.where(
                        correlation_time >= 0,
                        c_water * correlation_time / 2,  # Round-trip to one-way, using water sound speed
                        0.0  # Negative times correspond to reference starting before profile
                    )
                    # Clamp to ensure no negative depths (handle any numerical precision issues)
                    correlation_depths = np.maximum(correlation_depths, 0.0)
                    
                    # Log correlation depths range for debugging
                    if len(correlation_depths) > 0:
                        self.logger.info(f"SBP correlation_depths range: min={np.min(correlation_depths):.2f}m, "
                                       f"max={np.max(correlation_depths):.2f}m, "
                                       f"expected water_depth={water_depth:.2f}m, c_water={c_water:.2f}m/s")
                    
                    # Find all echoes (interfaces) from correlation peaks
                    # All distances to echoes are determined by correlation between received signal and reference signal
                    # Each peak in correlation indicates best match position = echo from an interface
                    # IMPORTANT: Use correlation_depths (not profile_depths_display) because correlation has different length (mode='full')
                    # Use lower threshold (0.01 = 1%) to detect weak sediment layer reflections
                    # Sediment layer reflections are much weaker than water-bottom echo due to attenuation
                    detected_echo_depths, detected_echo_amplitudes = profile_generator.find_all_echoes_from_correlation(
                        processed_signal, correlation_depths, min_height=0.01
                    )
                    
                    # Log detected echo details for debugging
                    if len(detected_echo_depths) > 0:
                        self.logger.info(f"SBP detected echoes: {len(detected_echo_depths)} echoes found")
                        for i, (depth, amp) in enumerate(zip(detected_echo_depths, detected_echo_amplitudes)):
                            # Calculate expected depth for each interface for comparison
                            if i == 0:
                                expected_depth = water_depth
                                expected_interface = "water-bottom (interface 0)"
                            else:
                                # Expected depths: interface 1 at 2m, interface 2 at 6m in sediment
                                expected_depths = [water_depth + 2.0, water_depth + 6.0]  # Mud-Gravel, Gravel-Sand
                                if i - 1 < len(expected_depths):
                                    expected_depth = expected_depths[i-1]
                                    expected_interface = f"interface {i} (expected at {expected_depths[i-1]:.2f}m)"
                                else:
                                    expected_depth = None
                                    expected_interface = f"unknown (possibly artifact)"
                            
                            if expected_depth is not None:
                                depth_diff = abs(depth - expected_depth)
                                self.logger.info(f"  Echo {i}: depth={depth:.2f}m (expected {expected_interface}: {expected_depth:.2f}m, diff={depth_diff:.2f}m), amplitude={amp:.6f}")
                            else:
                                self.logger.info(f"  Echo {i}: depth={depth:.2f}m, amplitude={amp:.6f} (possibly artifact)")
                    
                    if len(detected_echo_depths) > 0:
                        # Use detected depths and amplitudes from correlation
                        # IMPORTANT: The first detected echo (index 0) may NOT be the water-bottom echo!
                        # Artifacts from correlation can appear before the actual first echo.
                        # The water-bottom echo should have the maximum amplitude (strongest reflection).
                        # Find the echo with maximum amplitude - this should be the water-bottom echo
                        max_amp_idx = np.argmax(detected_echo_amplitudes)
                        detected_water_depth = detected_echo_depths[max_amp_idx]
                        profile_water_depth = detected_water_depth
                        
                        self.logger.info(f"SBP water-bottom detection: Echo {max_amp_idx} at depth {detected_water_depth:.2f}m "
                                       f"has maximum amplitude {detected_echo_amplitudes[max_amp_idx]:.6f} "
                                       f"(expected water depth: {water_depth:.2f}m)")
                        
                        # Convert full depths to depths in sediment (relative to water-bottom)
                        # The echo with maximum amplitude (max_amp_idx) is at water-bottom, so depth_in_sediment = 0
                        # Other echoes are at full_depth - water_depth
                        interface_depths = [0.0]  # First interface (water-bottom) is at 0 in sediment
                        interface_amplitudes = [detected_echo_amplitudes[max_amp_idx]] if len(detected_echo_amplitudes) > max_amp_idx else [0.0]
                        
                        # Add other echoes (excluding the water-bottom echo)
                        for i in range(len(detected_echo_depths)):
                            if i != max_amp_idx:  # Skip the water-bottom echo (already added)
                                depth_in_sediment = detected_echo_depths[i] - detected_water_depth
                                if depth_in_sediment > 0:  # Only add if in sediment
                                    interface_depths.append(depth_in_sediment)
                                    if i < len(detected_echo_amplitudes):
                                        interface_amplitudes.append(detected_echo_amplitudes[i])
                                    else:
                                        interface_amplitudes.append(0.0)
                        
                        self.logger.info(f"Detected {len(detected_echo_depths)} echoes from correlation: "
                                       f"water-bottom at {detected_water_depth:.2f} m, "
                                       f"{len(interface_depths)-1} layer interfaces")
                    else:
                        # Fallback: use theoretical positions if no echoes detected
                        detected_water_depth = water_depth
                        profile_water_depth = detected_water_depth
                        interface_depths = interface_depths_list  # Depths in sediment (from bottom)
                        interface_amplitudes = interface_amps_list  # Theoretical amplitudes
                        self.logger.warning(f"No echoes detected in correlation, using theoretical depths")
                    
                    # Optionally detect interfaces in processed signal for validation
                    # detected_depths, detected_amplitudes = profile_generator.detect_interfaces(
                    #     envelope, profile_depths_array
                    # )
                    
                    # Calculate vertical resolution
                    bandwidth = SignalModel.get_bandwidth(input_dto.signal.f_start, input_dto.signal.f_end)
                    avg_sound_speed = np.mean([layer.sound_speed for layer in sediment_layers])
                    vertical_resolution = profile_generator.calculate_vertical_resolution(bandwidth, avg_sound_speed)
                    
                    # Calculate max penetration depth (simplified)
                    # Use sonar equation
                    source_level_db = S_TX + 20 * np.log10(tx_voltage)  # dB re 1µPa @ 1m
                    noise_level = 80.0  # Typical noise level, dB
                    processing_gain = 10 * np.log10(bandwidth * input_dto.signal.Tp * 1e-6)
                    water_tl = self.water_model.calculate_transmission_loss(
                        2 * water_depth, f_center_profile, 
                        input_dto.environment.T, 
                        input_dto.environment.S, 
                        input_dto.environment.z
                    )
                    max_penetration_depth = sediment_model.get_max_penetration_depth(
                        source_level_db, noise_level, processing_gain, water_tl
                    )
                    
                    # NOTE: correlation_depths was already computed above (after correlation processing)
                    # and used for find_all_echoes_from_correlation. We reuse the same correlation_depths
                    # for profile_correlation_depths (no need to recompute).
                    
                    # Convert to lists for JSON serialization
                    profile_amplitudes = envelope.tolist()  # Envelope for main profile display (smooth, no oscillations)
                    profile_depths = profile_depths_display.tolist()  # Full depths from surface (water_depth + depth_in_sediment for sediment layers)
                    profile_time_axis = (np.arange(len(profile_signal_at_adc_display)) / signal_fs).tolist()
                    # Raw signal contains all reflections: first echo (water-bottom) + all layer interfaces
                    # profile_signal_at_adc_display is signal at ADC input (after receiver chain: LNA, VGA)
                    profile_raw_signal = np.abs(profile_signal_at_adc_display).tolist()  # Signal at ADC input (all reflections before matched filter)
                    # Correlation signal (matched filter output - correlation of received signal with reference CHIRP)
                    # Note: correlation has length = len(profile_signal) + len(reference) - 1 (mode='full')
                    # correlation_depths provides the correct depth axis for correlation
                    # This is the raw correlation output (can have oscillations), not envelope
                    profile_correlation = processed_signal.tolist()  # Correlation signal (matched filter output)
                    profile_correlation_depths = correlation_depths.tolist()  # Depth axis for correlation
                    
                    self.logger.info(f"SBP profile generated: {len(interface_depths)} interfaces detected "
                                   f"(first echo: water-bottom at {interface_depths[0]:.2f}m, "
                                   f"then {len(interface_depths)-1} layer interfaces), "
                                   f"max penetration: {max_penetration_depth:.2f} m, "
                                   f"vertical resolution: {vertical_resolution:.4f} m")
                    
                except Exception as e:
                    self.logger.error(f"SBP profile generation failed: {e}", exc_info=True)
                    # Continue without profile data
            
            # Calculate complete signal path using SignalPathCalculator (from core)
            # All calculations are done in core - GUI only receives ready data
            try:
                # For SBP, use reflection coefficient from first interface (water-sediment)
                # This is calculated from impedances, not from environment.bottom_reflection
                # First interface is water-sediment boundary (first layer)
                if getattr(input_dto, 'enable_sbp', False) and input_dto.sediment_profile and sediment_model is not None:
                    # Calculate reflection coefficient from first layer (water-sediment interface)
                    # Water impedance
                    P_water = self.water_model.calculate_pressure(input_dto.environment.z)
                    c_water = self.water_model.calculate_sound_speed(
                        input_dto.environment.T,
                        input_dto.environment.S,
                        P_water
                    )
                    rho_water = 1025.0  # kg/m³
                    water_impedance = rho_water * c_water
                    
                    # First sediment layer impedance
                    first_layer = sediment_model.layers[0]  # Get first layer from layers list
                    sediment_impedance = first_layer.impedance
                    
                    # Reflection coefficient at water-sediment interface
                    R_linear = (sediment_impedance - water_impedance) / (sediment_impedance + water_impedance)
                    R_dB = 20 * np.log10(abs(R_linear)) if abs(R_linear) > 1e-10 else -np.inf
                    
                    # Update input_dto temporarily to use calculated reflection coefficient
                    original_bottom_reflection = input_dto.environment.bottom_reflection
                    input_dto.environment.bottom_reflection = R_dB
                    
                    signal_path_data = self.signal_path_calculator.calculate_signal_path(
                        input_dto,
                        lna_gain=receiver.G_LNA,
                        vga_gain=receiver.current_G_VGA
                    )
                    
                    # Restore original value
                    input_dto.environment.bottom_reflection = original_bottom_reflection
                else:
                    # For regular sonar, use bottom_reflection from environment
                    signal_path_data = self.signal_path_calculator.calculate_signal_path(
                        input_dto,
                        lna_gain=receiver.G_LNA,
                        vga_gain=receiver.current_G_VGA
                    )
            except Exception as e:
                self.logger.warning(f"Failed to calculate signal path: {e}", exc_info=True)
                signal_path_data = None
            
            # Convert numpy arrays to lists for JSON serialization
            output_dto = OutputDTO(
                D_measured=D_measured,
                sigma_D=sigma_D,
                SNR_ADC=snr_adc,
                clipping_flags=clipping,
                lna_gain=receiver.G_LNA,  # LNA gain used in simulation
                vga_gain=receiver.current_G_VGA,  # VGA gain used in simulation
                vga_gain_max=receiver.G_VGA_max,  # VGA maximum gain
                adc_full_scale=adc_full_scale,  # ADC full scale voltage (V_FS), V
                adc_range=adc_range,  # ADC input range (±V_FS/2), V
                adc_bits=adc_bits,  # ADC resolution (bits)
                adc_dynamic_range=adc_dynamic_range_db,  # ADC dynamic range, dB
                tx_signal=reference_signal_vis.tolist() if len(reference_signal_vis) > 0 else None,  # Stage 1: Original CHIRP (visualization)
                signal_at_bottom=signal_after_water_forward.tolist() if len(signal_after_water_forward) > 0 else None,  # Stage 2: After water forward
                received_signal=signal_after_water_backward.tolist() if len(signal_after_water_backward) > 0 else None,  # Stage 3: After water backward
                signal_after_lna=signal_after_lna_vis.tolist() if len(signal_after_lna_vis) > 0 else None,  # Stage 4: After LNA (resampled)
                signal_after_vga=signal_after_vga_vis.tolist() if len(signal_after_vga_vis) > 0 else None,  # Stage 5: After VGA (resampled)
                time_axis=t_ref_vis.tolist() if len(t_ref_vis) > 0 else None,  # Visualization time axis
                attenuation_at_bottom_db=attenuation_at_bottom_db,  # Calculated in Core
                attenuation_received_db=attenuation_received_db,  # Calculated in Core
                # SBP profile data
                profile_amplitudes=profile_amplitudes,
                profile_depths=profile_depths,
                profile_time_axis=profile_time_axis,
                profile_raw_signal=profile_raw_signal,
                profile_correlation=profile_correlation if 'profile_correlation' in locals() else None,
                profile_correlation_depths=profile_correlation_depths if 'profile_correlation_depths' in locals() else None,
                profile_water_depth=profile_water_depth,
                interface_depths=interface_depths,
                interface_amplitudes=interface_amplitudes,
                max_penetration_depth=max_penetration_depth,
                vertical_resolution=vertical_resolution,
                signal_path=signal_path_data  # Complete signal path data
            )
            
            # Analyze results and generate recommendations
            recommendations = self.optimizer.analyze_results(input_dto, output_dto)
            output_dto.recommendations = recommendations
            
            # Calculate ENOB with parameters AFTER optimization
            # Use recommended values if available, otherwise use current values
            #
            # NOTE: ENOB calculation uses Analog SNR measured at VGA OUTPUT (before ADC).
            # This is different from SNR_ADC (Measured SNR) which is measured at ADC OUTPUT (after quantization).
            # See enob_calculator.py and receiver_model.py for detailed explanations of the differences.
            enob_calculator = ENOBCalculator()
            enob_results = None
            
            try:
                # Get parameters AFTER optimization
                if recommendations.suggested_changes:
                    # Use recommended values
                    optimized_tp = recommendations.suggested_changes.get('Tp', input_dto.signal.Tp)
                    optimized_f_start = recommendations.suggested_changes.get('f_start', input_dto.signal.f_start)
                    optimized_f_end = recommendations.suggested_changes.get('f_end', input_dto.signal.f_end)
                else:
                    # Use current values
                    optimized_tp = input_dto.signal.Tp
                    optimized_f_start = input_dto.signal.f_start
                    optimized_f_end = input_dto.signal.f_end
                
                # VGA gain from simulation (after optimization)
                vga_gain_used = receiver.current_G_VGA
                
                # Calculate signal RMS (use the signal from simulation)
                if len(received_signal) > 0:
                    signal_rms = np.sqrt(np.mean(received_signal ** 2))
                    
                    # Calculate bandwidth with optimized frequencies
                    bandwidth = SignalModel.get_bandwidth(optimized_f_start, optimized_f_end)
                    chirp_duration = optimized_tp * 1e-6  # Convert from µs to seconds
                    
                    # Calculate ENOB with optimized parameters
                    enob_results = enob_calculator.calculate_enob(
                        signal_input_voltage=signal_rms,
                        bandwidth=bandwidth,
                        lna_params=lna_params,
                        vga_params=vga_params,
                        adc_params=adc_params,
                        vga_gain=vga_gain_used,
                        chirp_duration=chirp_duration,
                        sample_rate=adc_fs
                    )
                    
            except Exception as e:
                self.logger.warning(f"ENOB calculation failed: {e}", exc_info=True)
            
            # Update output_dto with ENOB results
            output_dto.enob_results = enob_results
            
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

