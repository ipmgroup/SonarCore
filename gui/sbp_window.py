"""
SBP Window - GUI for Sub-Bottom Profiler simulation and optimization.

Features:
- Dark theme
- PyQtGraph for plotting
- Profile visualization (A-scan, B-scan)
- Parameter input and optimization
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, List
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QComboBox,
                             QGroupBox, QFormLayout, QTextEdit, QFileDialog,
                             QMessageBox, QTabWidget, QSpinBox, QDoubleSpinBox,
                             QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QScrollArea, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette

# Add root directory to path for imports
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.dto import (InputDTO, OutputDTO, HardwareDTO, SignalDTO, EnvironmentDTO,
                      RangeDTO, SedimentLayerDTO, SedimentProfileDTO)
from core.simulator import Simulator
from core.sediment_model import SedimentModel, SedimentLayer
from core.signal_calculator import SignalCalculator
from data.data_provider import DataProvider


# Configure PyQtGraph for dark theme
pg.setConfigOption('background', 'k')  # Black background
pg.setConfigOption('foreground', 'w')  # White foreground


class SBPSimulationThread(QThread):
    """Thread for SBP simulation execution."""
    
    finished = pyqtSignal(object, object)  # input_dto, output_dto
    
    def __init__(self, simulator, input_dto, vga_gain=None):
        super().__init__()
        self.simulator = simulator
        self.input_dto = input_dto
        self.vga_gain = vga_gain
    
    def run(self):
        """Executes simulation."""
        output_dto = self.simulator.simulate(self.input_dto, vga_gain=self.vga_gain)
        self.finished.emit(self.input_dto, output_dto)


class SBPWindow(QMainWindow):
    """Main window for SBP simulation."""
    
    def __init__(self):
        super().__init__()
        
        # Setup logging first (needed by other initialization methods)
        self.logger = logging.getLogger(__name__)
        
        self.data_provider = DataProvider()
        self.simulator = Simulator(self.data_provider)
        self.signal_calculator = SignalCalculator()
        self.current_output: Optional[OutputDTO] = None
        self.current_optimal_tp_ms: Optional[float] = None  # Store current optimal Tp in ms
        
        # Flags for settings loading/saving
        self._loading_settings = False
        self._gui_settings_loaded = False
        
        # Setup dark theme
        self._setup_dark_theme()
        
        # Initialize UI
        self._init_ui()
        
        # Load saved settings after UI is created
        self._load_gui_settings()
        self._gui_settings_loaded = True
    
    def _setup_dark_theme(self):
        """Setup dark theme for the application."""
        dark_palette = QPalette()
        
        # Window colors
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        
        # Base colors
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        
        # Text colors
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        
        # Button colors
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        
        # Highlight colors
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.white)
        
        self.setPalette(dark_palette)
    
    def _init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("Sub-Bottom Profiler Simulator")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Parameters
        left_panel = self._create_parameters_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # Right panel: Visualization
        right_panel = self._create_visualization_panel()
        main_layout.addWidget(right_panel, stretch=2)
    
    def _create_parameters_panel(self) -> QWidget:
        """Create parameters input panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Hardware selection
        hw_group = QGroupBox("Hardware")
        hw_layout = QFormLayout()
        
        self.transducer_combo = QComboBox()
        self.lna_combo = QComboBox()
        self.vga_combo = QComboBox()
        self.adc_combo = QComboBox()
        
        # Populate combos
        self._populate_hardware_combos()
        
        hw_layout.addRow("Transducer:", self.transducer_combo)
        hw_layout.addRow("LNA:", self.lna_combo)
        hw_layout.addRow("VGA:", self.vga_combo)
        hw_layout.addRow("ADC:", self.adc_combo)
        hw_group.setLayout(hw_layout)
        scroll_layout.addWidget(hw_group)
        
        # Signal parameters
        signal_group = QGroupBox("CHIRP Signal")
        signal_layout = QFormLayout()
        
        self.f_start_spin = QDoubleSpinBox()
        self.f_start_spin.setRange(1000, 1000000)
        self.f_start_spin.setValue(100000)
        self.f_start_spin.setSuffix(" Hz")
        self.f_start_spin.valueChanged.connect(self._save_gui_settings)
        self.f_start_spin.valueChanged.connect(self._on_frequency_changed)
        
        self.f_end_spin = QDoubleSpinBox()
        self.f_end_spin.setRange(1000, 1000000)
        self.f_end_spin.setValue(300000)
        self.f_end_spin.setSuffix(" Hz")
        self.f_end_spin.valueChanged.connect(self._save_gui_settings)
        self.f_end_spin.valueChanged.connect(self._on_frequency_changed)
        
        self.tp_spin = QDoubleSpinBox()
        self.tp_spin.setRange(1, 10000000)
        self.tp_spin.setValue(1000)
        self.tp_spin.setSuffix(" µs")
        self.tp_spin.valueChanged.connect(self._save_gui_settings)
        
        self.window_combo = QComboBox()
        self.window_combo.addItems(["Rect", "Hann", "Tukey"])
        self.window_combo.currentTextChanged.connect(self._save_gui_settings)
        
        signal_layout.addRow("f_start:", self.f_start_spin)
        signal_layout.addRow("f_end:", self.f_end_spin)
        signal_layout.addRow("Tp:", self.tp_spin)
        signal_layout.addRow("Window:", self.window_combo)
        
        # SNR pre-correlation input (for SBP optimization)
        self.snr_pre_spin = QDoubleSpinBox()
        self.snr_pre_spin.setRange(-20, 10)
        self.snr_pre_spin.setValue(-5.0)
        self.snr_pre_spin.setSuffix(" dB")
        self.snr_pre_spin.setToolTip("Pre-correlation SNR (typical: -5 to 0 dB for sediment)")
        self.snr_pre_spin.valueChanged.connect(self._on_snr_pre_changed)
        self.snr_pre_spin.valueChanged.connect(self._save_gui_settings)
        signal_layout.addRow("SNR pre (SBP):", self.snr_pre_spin)
        
        # Optimal Tp display (read-only label) with apply button
        optimal_tp_layout = QHBoxLayout()
        self.optimal_tp_label = QLabel("N/A")
        self.optimal_tp_label.setToolTip("Optimal pulse duration calculated from SNR requirements")
        optimal_tp_layout.addWidget(self.optimal_tp_label)
        optimal_tp_layout.addStretch()
        self.apply_optimal_tp_btn = QPushButton("Apply")
        self.apply_optimal_tp_btn.setToolTip("Apply optimal pulse duration to Tp field")
        self.apply_optimal_tp_btn.setEnabled(False)  # Disabled until optimal Tp is calculated
        self.apply_optimal_tp_btn.clicked.connect(self._apply_optimal_tp)
        optimal_tp_layout.addWidget(self.apply_optimal_tp_btn)
        optimal_tp_widget = QWidget()
        optimal_tp_widget.setLayout(optimal_tp_layout)
        signal_layout.addRow("Optimal Tp (SNR):", optimal_tp_widget)
        
        # Theoretical resolution display (read-only label)
        self.theoretical_resolution_label = QLabel("N/A")
        self.theoretical_resolution_label.setToolTip("Theoretical vertical resolution: Δz_min = c / (2*B), where c=1500 m/s")
        signal_layout.addRow("Theoretical Resolution:", self.theoretical_resolution_label)
        
        signal_group.setLayout(signal_layout)
        scroll_layout.addWidget(signal_group)
        
        # Environment parameters
        env_group = QGroupBox("Environment")
        env_layout = QFormLayout()
        
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0, 30)
        self.temp_spin.setValue(15)
        self.temp_spin.setSuffix(" °C")
        self.temp_spin.valueChanged.connect(self._save_gui_settings)
        
        self.salinity_spin = QDoubleSpinBox()
        self.salinity_spin.setRange(0, 35)
        self.salinity_spin.setValue(35)
        self.salinity_spin.setSuffix(" PSU")
        self.salinity_spin.valueChanged.connect(self._save_gui_settings)
        
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.2, 300)
        self.depth_spin.setValue(100)
        self.depth_spin.setSuffix(" m")
        self.depth_spin.valueChanged.connect(self._save_gui_settings)
        
        env_layout.addRow("Temperature:", self.temp_spin)
        env_layout.addRow("Salinity:", self.salinity_spin)
        env_layout.addRow("Depth:", self.depth_spin)
        env_group.setLayout(env_layout)
        scroll_layout.addWidget(env_group)
        
        # Receiver parameters
        receiver_group = QGroupBox("Receiver")
        receiver_layout = QFormLayout()
        
        self.vga_gain_spin = QDoubleSpinBox()
        self.vga_gain_spin.setRange(0, 100)  # Will be updated based on selected VGA
        self.vga_gain_spin.setValue(30)
        self.vga_gain_spin.setSuffix(" dB")
        self.vga_gain_spin.setToolTip("VGA gain (Variable Gain Amplifier)")
        self.vga_gain_spin.valueChanged.connect(self._save_gui_settings)
        receiver_layout.addRow("VGA Gain:", self.vga_gain_spin)
        
        receiver_group.setLayout(receiver_layout)
        scroll_layout.addWidget(receiver_group)
        
        # Sediment profile
        sediment_group = QGroupBox("Sediment Profile")
        sediment_layout = QVBoxLayout()
        
        self.sediment_table = QTableWidget()
        self.sediment_table.setColumnCount(5)
        self.sediment_table.setHorizontalHeaderLabels(["Name", "Thickness (m)", "Density (kg/m³)", "Sound Speed (m/s)", "Attenuation (dB/m)"])
        self.sediment_table.horizontalHeader().setStretchLastSection(True)
        self.sediment_table.setRowCount(3)
        # Connect table changes to save settings
        self.sediment_table.itemChanged.connect(self._save_gui_settings)
        
        # Default layers
        default_layers = [
            ("Clay", 2.0, 1600, 1550, 1.0),
            ("Silt", 3.0, 1800, 1600, 2.0),
            ("Sand", 5.0, 2000, 1700, 3.0),
        ]
        
        for i, (name, thickness, density, speed, atten) in enumerate(default_layers):
            self.sediment_table.setItem(i, 0, QTableWidgetItem(name))
            self.sediment_table.setItem(i, 1, QTableWidgetItem(str(thickness)))
            self.sediment_table.setItem(i, 2, QTableWidgetItem(str(density)))
            self.sediment_table.setItem(i, 3, QTableWidgetItem(str(speed)))
            self.sediment_table.setItem(i, 4, QTableWidgetItem(str(atten)))
        
        sediment_layout.addWidget(self.sediment_table)
        
        # Buttons for sediment table
        sediment_btn_layout = QHBoxLayout()
        
        # Layer type selector for adding new layers
        layer_type_label = QLabel("Layer Type:")
        self.layer_type_combo = QComboBox()
        self.layer_type_combo.addItem("Custom", "")
        # Populate with available layer types
        try:
            layer_types = self.data_provider.list_sediment_layers()
            for layer_type_id in sorted(layer_types):
                try:
                    layer_data = self.data_provider.get_sediment_layer(layer_type_id)
                    display_name = layer_data.get('name', layer_type_id)
                    self.layer_type_combo.addItem(display_name, layer_type_id)
                except Exception as e:
                    self.logger.warning(f"Failed to load layer type {layer_type_id}: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to list layer types: {e}")
        
        add_layer_btn = QPushButton("Add Layer")
        remove_layer_btn = QPushButton("Remove Layer")
        add_layer_btn.clicked.connect(self._add_sediment_layer)
        remove_layer_btn.clicked.connect(self._remove_sediment_layer)
        sediment_btn_layout.addWidget(layer_type_label)
        sediment_btn_layout.addWidget(self.layer_type_combo)
        sediment_btn_layout.addWidget(add_layer_btn)
        sediment_btn_layout.addWidget(remove_layer_btn)
        sediment_btn_layout.addStretch()
        sediment_layout.addLayout(sediment_btn_layout)
        
        sediment_group.setLayout(sediment_layout)
        scroll_layout.addWidget(sediment_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self._run_simulation)
        self.optimize_btn = QPushButton("Optimize")
        self.optimize_btn.clicked.connect(self._run_optimization)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.optimize_btn)
        scroll_layout.addLayout(btn_layout)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return panel
    
    def _create_visualization_panel(self) -> QWidget:
        """Create visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different views
        tabs = QTabWidget()
        
        # Profile view (A-scan)
        profile_tab = QWidget()
        profile_layout = QVBoxLayout(profile_tab)
        
        # Controls row: Mode selection (RAW/Ground) and checkboxes (Signal/Correlation)
        controls_row = QHBoxLayout()
        
        # Mode selection: RAW / Ground
        mode_label = QLabel("Mode:")
        controls_row.addWidget(mode_label)
        
        self.profile_mode_group = QButtonGroup()
        self.raw_mode_radio = QRadioButton("RAW")
        self.ground_mode_radio = QRadioButton("Ground")
        self.raw_mode_radio.setChecked(True)  # Default: RAW mode
        self.profile_mode_group.addButton(self.raw_mode_radio, 0)
        self.profile_mode_group.addButton(self.ground_mode_radio, 1)
        self.profile_mode_group.buttonClicked.connect(self._on_profile_mode_changed)
        
        controls_row.addWidget(self.raw_mode_radio)
        controls_row.addWidget(self.ground_mode_radio)
        controls_row.addSpacing(20)  # Spacing between mode and checkboxes
        
        # Checkbox to show signal
        self.show_signal_checkbox = QCheckBox("Signal")
        self.show_signal_checkbox.setChecked(True)  # Default: show signal
        self.show_signal_checkbox.stateChanged.connect(self._on_signal_checkbox_changed)
        controls_row.addWidget(self.show_signal_checkbox)
        
        # Checkbox to show correlation signal
        self.show_correlation_checkbox = QCheckBox("Correlation")
        self.show_correlation_checkbox.setChecked(True)  # Default: show correlation
        self.show_correlation_checkbox.stateChanged.connect(self._on_correlation_checkbox_changed)
        controls_row.addWidget(self.show_correlation_checkbox)
        
        controls_row.addStretch()  # Push all controls to the left
        
        profile_layout.addLayout(controls_row)
        
        self.profile_plot = pg.PlotWidget(title="Sub-Bottom Profile (A-scan)")
        self.profile_plot.setLabel('left', 'Amplitude', units='V')
        self.profile_plot.setLabel('bottom', 'Depth', units='m')
        self.profile_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Add secondary Y-axis for correlation signal (can have negative values)
        self.correlation_axis = pg.ViewBox()
        self.profile_plot.scene().addItem(self.correlation_axis)
        self.profile_plot.getAxis('right').linkToView(self.correlation_axis)
        self.correlation_axis.setXLink(self.profile_plot)
        
        # Update correlation axis when main view changes
        def update_views():
            self.correlation_axis.setGeometry(self.profile_plot.getViewBox().sceneBoundingRect())
        self.profile_plot.getViewBox().sigResized.connect(update_views)
        update_views()  # Initial update
        
        profile_layout.addWidget(self.profile_plot)
        
        tabs.addTab(profile_tab, "Profile")
        
        # Signal Path tab
        signal_path_tab = QWidget()
        signal_path_layout = QVBoxLayout(signal_path_tab)
        
        # Use SignalPathDiagram from signal_path_widget (same as main_window.py)
        from gui.signal_path_widget import SignalPathDiagram
        self.signal_path_diagram = SignalPathDiagram()
        # Note: SignalPathDiagram uses light theme by default (white background)
        # This matches the visual style from main_window.py
        signal_path_layout.addWidget(self.signal_path_diagram)
        
        tabs.addTab(signal_path_tab, "Signal Path")
        
        # Results text
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        tabs.addTab(results_tab, "Results")
        
        layout.addWidget(tabs)
        
        return panel
    
    def _populate_hardware_combos(self):
        """Populate hardware combo boxes."""
        try:
            transducers = self.data_provider.list_transducers()
            self.transducer_combo.addItems(transducers)
            
            # Connect transducer change signal to update CHIRP frequencies
            # _on_transducer_changed will handle saving settings after frequencies are updated
            self.transducer_combo.currentTextChanged.connect(self._on_transducer_changed)
            # Save settings when other hardware changes
            self.lna_combo.currentTextChanged.connect(self._save_gui_settings)
            # VGA combo: connect to _on_vga_changed which updates range and saves settings
            self.vga_combo.currentTextChanged.connect(self._on_vga_changed)
            self.adc_combo.currentTextChanged.connect(self._save_gui_settings)
            
            lnas = self.data_provider.list_lna()
            self.lna_combo.addItems(lnas)
            
            vgas = self.data_provider.list_vga()
            self.vga_combo.addItems(vgas)
            
            adcs = self.data_provider.list_adc()
            self.adc_combo.addItems(adcs)
            
            # Set initial frequencies based on first transducer
            if transducers:
                self._on_transducer_changed(transducers[0])
        except Exception as e:
            self.logger.error(f"Error populating hardware combos: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load hardware data: {e}")
    
    def _on_vga_changed(self, vga_id: str):
        """
        Handle VGA selection change.
        Updates VGA gain spinbox range based on selected VGA parameters.
        """
        try:
            if not vga_id:
                return
            
            # Get VGA parameters to determine gain range
            vga_params = self.data_provider.get_vga(vga_id)
            g_min = vga_params.get('G_min', 0.0)
            g_max = vga_params.get('G_max', 100.0)
            
            # Update VGA gain spinbox range
            if hasattr(self, 'vga_gain_spin'):
                current_value = self.vga_gain_spin.value()
                self.vga_gain_spin.setRange(g_min, g_max)
                # Keep current value if it's within new range, otherwise set to middle
                if g_min <= current_value <= g_max:
                    self.vga_gain_spin.setValue(current_value)
                else:
                    self.vga_gain_spin.setValue((g_min + g_max) / 2.0)
                
                self.logger.info(f"VGA changed to {vga_id}: gain range {g_min:.1f} - {g_max:.1f} dB")
                # Save settings after updating VGA range
                self._save_gui_settings()
        except Exception as e:
            self.logger.error(f"Error updating VGA gain range for {vga_id}: {e}")
    
    def _on_transducer_changed(self, transducer_id: str):
        """
        Handle transducer selection change.
        Automatically sets CHIRP frequencies based on transducer parameters.
        """
        try:
            if not transducer_id:
                return
            
            # Get transducer parameters
            transducer_params = self.data_provider.get_transducer(transducer_id)
            
            f_min = transducer_params.get('f_min')
            f_max = transducer_params.get('f_max')
            
            if f_min and f_max:
                # Block signals to prevent saving during frequency update
                # (will be saved after both frequencies are set)
                self.f_start_spin.blockSignals(True)
                self.f_end_spin.blockSignals(True)
                
                # Set CHIRP frequencies to match transducer range
                # Use f_min as f_start and f_max as f_end
                self.f_start_spin.setValue(f_min)
                self.f_end_spin.setValue(f_max)
                
                # Unblock signals
                self.f_start_spin.blockSignals(False)
                self.f_end_spin.blockSignals(False)
                
                # Save settings after frequencies are updated
                # (only if not currently loading settings)
                if not getattr(self, '_loading_settings', False):
                    self._save_gui_settings()
                
                self.logger.info(f"Transducer changed to {transducer_id}: "
                               f"CHIRP frequencies set to {f_min/1000:.1f}-{f_max/1000:.1f} kHz")
            
            # Update optimal Tp calculation after frequencies change
            if hasattr(self, 'signal_calculator'):
                self._calculate_optimal_tp()
        except Exception as e:
            self.logger.error(f"Error updating CHIRP frequencies for transducer {transducer_id}: {e}")
            # Don't show error message to user, just log it
    
    def _on_snr_pre_changed(self):
        """Handle SNR pre-correlation value change."""
        if hasattr(self, 'signal_calculator'):
            self._calculate_optimal_tp()
    
    def _on_frequency_changed(self):
        """Handle frequency (f_start or f_end) change."""
        if hasattr(self, 'signal_calculator'):
            self._calculate_optimal_tp()
    
    def _calculate_optimal_tp(self):
        """Calculate optimal pulse duration based on SNR requirements and theoretical resolution."""
        try:
            f_start = self.f_start_spin.value()
            f_end = self.f_end_spin.value()
            snr_pre = self.snr_pre_spin.value()
            snr_post_min = 10.0  # Minimum required post-correlation SNR (from section 23.5)
            
            # Calculate optimal Tp
            optimal_tp_ms = self.signal_calculator.calculate_optimal_tp_for_sbp_snr(
                f_start=f_start,
                f_end=f_end,
                snr_pre=snr_pre,
                snr_post_min=snr_post_min
            )
            
            if optimal_tp_ms != float('inf') and optimal_tp_ms > 0:
                # Store current optimal Tp value for apply button
                self.current_optimal_tp_ms = optimal_tp_ms
                # Convert to microseconds for display (to match Tp spinbox)
                optimal_tp_us = optimal_tp_ms * 1000.0
                self.optimal_tp_label.setText(f"{optimal_tp_ms:.2f} ms ({optimal_tp_us:.1f} µs)")
                self.optimal_tp_label.setStyleSheet("color: lightgreen;")
                # Enable apply button if value is valid
                if hasattr(self, 'apply_optimal_tp_btn'):
                    self.apply_optimal_tp_btn.setEnabled(True)
            else:
                self.current_optimal_tp_ms = None
                self.optimal_tp_label.setText("N/A")
                self.optimal_tp_label.setStyleSheet("")
                # Disable apply button if value is not available
                if hasattr(self, 'apply_optimal_tp_btn'):
                    self.apply_optimal_tp_btn.setEnabled(False)
            
            # Calculate theoretical resolution: Δz_min = c / (2*B)
            # Formula from SBP.md section 23.3
            from core.signal_model import SignalModel
            bandwidth = SignalModel.get_bandwidth(f_start, f_end)
            c = 1500.0  # Sound speed in water, m/s (from section 23.2)
            
            if bandwidth > 0:
                theoretical_resolution_m = c / (2.0 * bandwidth)
                theoretical_resolution_cm = theoretical_resolution_m * 100.0
                self.theoretical_resolution_label.setText(f"{theoretical_resolution_m:.3f} m ({theoretical_resolution_cm:.1f} cm)")
                self.theoretical_resolution_label.setStyleSheet("color: lightblue;")
            else:
                self.theoretical_resolution_label.setText("N/A")
                self.theoretical_resolution_label.setStyleSheet("")
                
        except Exception as e:
            self.logger.error(f"Error calculating optimal Tp/resolution: {e}")
            self.current_optimal_tp_ms = None
            self.optimal_tp_label.setText("Error")
            self.optimal_tp_label.setStyleSheet("color: red;")
            self.theoretical_resolution_label.setText("Error")
            self.theoretical_resolution_label.setStyleSheet("color: red;")
            # Disable apply button on error
            if hasattr(self, 'apply_optimal_tp_btn'):
                self.apply_optimal_tp_btn.setEnabled(False)
    
    def _apply_optimal_tp(self):
        """Apply optimal pulse duration to Tp field."""
        try:
            # Use stored optimal Tp value (more reliable than parsing label text)
            if self.current_optimal_tp_ms is None or self.current_optimal_tp_ms <= 0:
                QMessageBox.warning(self, "Warning", "Optimal Tp is not available. Please check signal parameters.")
                return
            
            # Convert to microseconds for tp_spin (which uses µs)
            optimal_tp_us = self.current_optimal_tp_ms * 1000.0
            
            # Get current Tp value before applying
            old_tp_us = self.tp_spin.value()
            
            # Set the value in tp_spin
            self.tp_spin.blockSignals(True)
            self.tp_spin.setValue(optimal_tp_us)
            self.tp_spin.blockSignals(False)
            
            # Verify that the value was set correctly
            actual_tp_us = self.tp_spin.value()
            actual_tp_ms = actual_tp_us / 1000.0
            
            # Log the change
            self.logger.info(f"Applied optimal Tp: {self.current_optimal_tp_ms:.2f} ms ({optimal_tp_us:.1f} µs). "
                           f"Previous value: {old_tp_us:.1f} µs, New value: {actual_tp_us:.1f} µs ({actual_tp_ms:.3f} ms)")
            
            # Save settings with new Tp value
            self._save_gui_settings()
            
            # Show confirmation message with details
            QMessageBox.information(
                self, 
                "Optimal Tp Applied", 
                f"Optimal pulse duration applied:\n\n"
                f"Previous: {old_tp_us:.1f} µs ({old_tp_us/1000.0:.3f} ms)\n"
                f"New: {actual_tp_us:.1f} µs ({actual_tp_ms:.3f} ms)\n\n"
                f"Target: {self.current_optimal_tp_ms:.2f} ms\n\n"
                f"CHIRP length: {actual_tp_ms:.3f} ms"
            )
        except Exception as e:
            self.logger.error(f"Error applying optimal Tp: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to apply optimal Tp: {e}")
    
    def _add_sediment_layer(self):
        """Add new sediment layer to table."""
        layer_type_id = self.layer_type_combo.currentData()
        
        # Block signals to prevent saving during table update
        self.sediment_table.blockSignals(True)
        
        row = self.sediment_table.rowCount()
        self.sediment_table.insertRow(row)
        
        if layer_type_id:
            # Load layer type properties from JSON
            try:
                layer_data = self.data_provider.get_sediment_layer(layer_type_id)
                name = layer_data.get('name', layer_type_id)
                thickness = layer_data.get('default_thickness', 2.0)
                density = layer_data.get('density', 1800)
                sound_speed = layer_data.get('sound_speed', 1600)
                attenuation = layer_data.get('attenuation', 2.0)
            except Exception as e:
                self.logger.warning(f"Failed to load layer type {layer_type_id}: {e}, using defaults")
                name = "Layer"
                thickness = 2.0
                density = 1800
                sound_speed = 1600
                attenuation = 2.0
        else:
            # Custom layer - use default values
            name = "Layer"
            thickness = 2.0
            density = 1800
            sound_speed = 1600
            attenuation = 2.0
        
        self.sediment_table.setItem(row, 0, QTableWidgetItem(name))
        self.sediment_table.setItem(row, 1, QTableWidgetItem(str(thickness)))
        self.sediment_table.setItem(row, 2, QTableWidgetItem(str(density)))
        self.sediment_table.setItem(row, 3, QTableWidgetItem(str(sound_speed)))
        self.sediment_table.setItem(row, 4, QTableWidgetItem(str(attenuation)))
        
        # Unblock signals
        self.sediment_table.blockSignals(False)
        
        # Save settings after adding layer
        self._save_gui_settings()
    
    def _remove_sediment_layer(self):
        """Remove selected sediment layer."""
        current_row = self.sediment_table.currentRow()
        if current_row >= 0:
            self.sediment_table.removeRow(current_row)
            # Save settings after removing layer
            self._save_gui_settings()
    
    def _get_sediment_profile(self) -> Optional[SedimentProfileDTO]:
        """Get sediment profile from table."""
        layers = []
        for row in range(self.sediment_table.rowCount()):
            name_item = self.sediment_table.item(row, 0)
            thickness_item = self.sediment_table.item(row, 1)
            density_item = self.sediment_table.item(row, 2)
            speed_item = self.sediment_table.item(row, 3)
            atten_item = self.sediment_table.item(row, 4)
            
            if all([name_item, thickness_item, density_item, speed_item, atten_item]):
                try:
                    layer = SedimentLayerDTO(
                        name=name_item.text() or f"Layer {row+1}",
                        thickness=float(thickness_item.text()),
                        density=float(density_item.text()),
                        sound_speed=float(speed_item.text()),
                        attenuation=float(atten_item.text())
                    )
                    layers.append(layer)
                except ValueError as e:
                    QMessageBox.warning(self, "Error", f"Invalid layer data at row {row+1}: {e}")
                    return None
        
        if len(layers) == 0:
            return None
        
        return SedimentProfileDTO(layers=layers)
    
    def _create_input_dto(self) -> Optional[InputDTO]:
        """Create InputDTO from UI parameters."""
        try:
            # Hardware
            hardware = HardwareDTO(
                transducer_id=self.transducer_combo.currentText(),
                lna_id=self.lna_combo.currentText(),
                vga_id=self.vga_combo.currentText(),
                adc_id=self.adc_combo.currentText()
            )
            
            # Signal
            signal = SignalDTO(
                f_start=self.f_start_spin.value(),
                f_end=self.f_end_spin.value(),
                Tp=self.tp_spin.value(),
                window=self.window_combo.currentText(),
                sample_rate=2e6  # Default sample rate
            )
            
            # Environment
            environment = EnvironmentDTO(
                T=self.temp_spin.value(),
                S=self.salinity_spin.value(),
                z=self.depth_spin.value()
            )
            
            # Range (for SBP, use depth as range)
            range_dto = RangeDTO(
                D_min=0.1,
                D_max=self.depth_spin.value() + 50.0,  # Extend beyond water depth
                D_target=self.depth_spin.value()
            )
            
            # Sediment profile
            sediment_profile = self._get_sediment_profile()
            
            input_dto = InputDTO(
                hardware=hardware,
                signal=signal,
                environment=environment,
                range=range_dto,
                enable_sbp=True,
                sediment_profile=sediment_profile,
                snr_pre=self.snr_pre_spin.value() if hasattr(self, 'snr_pre_spin') else -5.0
            )
            
            return input_dto
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create input DTO: {e}")
            return None
    
    def _run_simulation(self):
        """Run simulation."""
        input_dto = self._create_input_dto()
        if input_dto is None:
            return
        
        # Get VGA gain from input field
        vga_gain = self.vga_gain_spin.value() if hasattr(self, 'vga_gain_spin') else None
        
        # Disable buttons
        self.run_btn.setEnabled(False)
        self.optimize_btn.setEnabled(False)
        
        # Run in thread
        self.sim_thread = SBPSimulationThread(self.simulator, input_dto, vga_gain=vga_gain)
        self.sim_thread.finished.connect(self._on_simulation_finished)
        self.sim_thread.start()
    
    def _run_optimization(self):
        """Run optimization."""
        # TODO: Implement optimization
        QMessageBox.information(self, "Info", "Optimization not yet implemented")
    
    def _on_simulation_finished(self, input_dto: InputDTO, output_dto: OutputDTO):
        """Handle simulation finished."""
        self.current_output = output_dto
        
        # Enable buttons
        self.run_btn.setEnabled(True)
        self.optimize_btn.setEnabled(True)
        
        # Update visualization
        self._update_visualization(output_dto)
        
        # Update signal path diagram (from core calculations - no calculations in GUI)
        self._update_signal_path_diagram(output_dto)
        
        # Update results text
        self._update_results_text(output_dto)
    
    def _update_visualization(self, output_dto: OutputDTO):
        """Update profile visualization."""
        self.profile_plot.clear()
        self.correlation_axis.clear()  # Clear secondary axis as well
        self.current_output = output_dto  # Store for mode callback
        
        if output_dto.profile_amplitudes and output_dto.profile_depths:
            depths = np.array(output_dto.profile_depths)
            
            # Get selected mode (radio buttons: RAW or Ground)
            show_raw_mode = self.raw_mode_radio.isChecked()
            show_signal = self.show_signal_checkbox.isChecked()
            show_correlation = self.show_correlation_checkbox.isChecked()
            
            if show_raw_mode:
                # Mode 1: RAW - Show all reflections including water-bottom echo
                # Full distance X from 0
                
                # Show signal if checkbox is checked
                if show_signal and output_dto.profile_raw_signal:
                    raw_signal = np.array(output_dto.profile_raw_signal)
                    
                    # Interpolate raw signal to match depths array length if needed
                    if len(raw_signal) != len(depths):
                        if len(raw_signal) > len(depths):
                            raw_signal = raw_signal[:len(depths)]
                        else:
                            padded = np.zeros(len(depths))
                            padded[:len(raw_signal)] = raw_signal
                            raw_signal = padded
                    
                    # Plot raw signal (all reflections: water-bottom echo + all layer interfaces)
                    self.profile_plot.plot(depths, raw_signal, pen='c', name="All Reflections (RAW)")
                
                # Show correlation signal if checkbox is checked (on secondary Y-axis)
                if show_correlation and output_dto.profile_correlation and output_dto.profile_correlation_depths:
                    # Clear correlation axis before adding new data
                    self.correlation_axis.clear()
                    
                    correlation_signal = np.array(output_dto.profile_correlation)
                    correlation_depths = np.array(output_dto.profile_correlation_depths)
                    
                    # Filter out negative depths (from mode='full' correlation with negative lags)
                    # Use > 0.0 (strict) to exclude zero and negative depths completely
                    # Also handle any numerical precision issues by using a small threshold
                    mask_positive = correlation_depths > 1e-10  # Strict positive check with small threshold
                    correlation_signal = correlation_signal[mask_positive]
                    correlation_depths = correlation_depths[mask_positive]
                    
                    # Additional safety check: ensure no negative depths remain
                    if np.any(correlation_depths < 0):
                        # If any negative values remain, filter them out again
                        mask_positive_strict = correlation_depths >= 0
                        correlation_signal = correlation_signal[mask_positive_strict]
                        correlation_depths = correlation_depths[mask_positive_strict]
                    
                    if len(correlation_signal) > 0 and len(correlation_depths) > 0:
                        # Verify no negative depths before plotting
                        assert np.all(correlation_depths >= 0), f"Negative depths detected: {correlation_depths[correlation_depths < 0]}"
                        
                        # Plot correlation on secondary Y-axis (only positive values)
                        correlation_positive = np.abs(correlation_signal)
                        correlation_item = pg.PlotDataItem(correlation_depths, correlation_positive, pen='m', name="Correlation")
                        self.correlation_axis.addItem(correlation_item)
                        # Set label for secondary axis
                        self.profile_plot.setLabel('right', 'Correlation', units='V')
                        # Auto-range secondary axis (only positive values, starting from 0)
                        self.correlation_axis.setYRange(0, np.max(correlation_positive) * 1.1 if len(correlation_positive) > 0 else 1.0, padding=0.05)
                else:
                    # Clear correlation axis if checkbox is unchecked
                    self.correlation_axis.clear()
                    self.profile_plot.setLabel('right', '')
                
                # Mark detected interfaces (full depths from surface)
                # For RAW mode, use amplitudes from raw_signal at interface positions (if signal is shown)
                if show_signal and output_dto.interface_depths and output_dto.profile_water_depth is not None and output_dto.profile_raw_signal:
                    raw_signal = np.array(output_dto.profile_raw_signal)
                    interface_depths_raw = np.array(output_dto.interface_depths)
                    water_depth = output_dto.profile_water_depth
                    
                    # interface_depths are in sediment (from bottom), convert to full depth from surface
                    interface_full_depths = []
                    interface_raw_amps = []
                    
                    for i, depth_in_sediment in enumerate(interface_depths_raw):
                        if i == 0:
                            # First interface (water-bottom) is at water_depth
                            interface_full_depth = water_depth
                        else:
                            # Other interfaces are at water_depth + depth_in_sediment
                            interface_full_depth = water_depth + depth_in_sediment
                        
                        interface_full_depths.append(interface_full_depth)
                        
                        # Find closest depth index in depths array
                        depth_idx = np.argmin(np.abs(depths - interface_full_depth))
                        # Get amplitude from raw signal at this position
                        if depth_idx < len(raw_signal):
                            interface_raw_amps.append(raw_signal[depth_idx])
                        else:
                            interface_raw_amps.append(0.0)
                    
                    interface_full_depths = np.array(interface_full_depths)
                    interface_raw_amps = np.array(interface_raw_amps)
                    
                    self.profile_plot.plot(interface_full_depths, interface_raw_amps, 
                                          pen=None, symbol='o', symbolSize=10,
                                          symbolBrush='r', name="Interfaces")
                elif output_dto.interface_depths and output_dto.interface_amplitudes:
                    # Fallback: use theoretical amplitudes if raw signal not available
                    interface_depths_raw = np.array(output_dto.interface_depths)
                    interface_amps_raw = np.array(output_dto.interface_amplitudes)
                    
                    self.profile_plot.plot(interface_depths_raw, interface_amps_raw, 
                                          pen=None, symbol='o', symbolSize=10,
                                          symbolBrush='r', name="Interfaces")
            else:  # show_raw_mode == False: Ground mode - same data as RAW but starting from first echo
                # Ground mode: Show same data as RAW (raw_signal and correlation), but starting from water_depth (first echo)
                water_depth = output_dto.profile_water_depth if output_dto.profile_water_depth is not None else 0.0
                
                # Show signal if checkbox is checked (use raw_signal like in RAW mode)
                if show_signal and output_dto.profile_raw_signal:
                    raw_signal = np.array(output_dto.profile_raw_signal)
                    
                    # Interpolate raw signal to match depths array length if needed
                    if len(raw_signal) != len(depths):
                        if len(raw_signal) > len(depths):
                            raw_signal = raw_signal[:len(depths)]
                        else:
                            padded = np.zeros(len(depths))
                            padded[:len(raw_signal)] = raw_signal
                            raw_signal = padded
                    
                    # Filter: show only from water_depth (first echo) to end
                    mask = depths >= water_depth
                    sediment_depths = depths[mask]
                    sediment_raw_signal = raw_signal[mask]
                    
                    if len(sediment_depths) > 0:
                        # Plot raw signal starting from water_depth (same as RAW mode, just filtered)
                        self.profile_plot.plot(sediment_depths, sediment_raw_signal, pen='c', name="Ground (from first echo)")
                
                # Show correlation signal if checkbox is checked (filtered for sediment layers only, on secondary Y-axis)
                # Correlation is independent of show_signal checkbox
                if show_correlation and output_dto.profile_correlation and output_dto.profile_correlation_depths:
                    # Clear correlation axis before adding new data
                    self.correlation_axis.clear()
                    
                    correlation_signal = np.array(output_dto.profile_correlation)
                    correlation_depths = np.array(output_dto.profile_correlation_depths)
                    
                    # First filter out negative depths (from mode='full' correlation with negative lags)
                    # Use > 0.0 (strict) to exclude zero and negative depths completely
                    # Also handle any numerical precision issues by using a small threshold
                    mask_positive = correlation_depths > 1e-10  # Strict positive check with small threshold
                    correlation_signal = correlation_signal[mask_positive]
                    correlation_depths = correlation_depths[mask_positive]
                    
                    # Additional safety check: ensure no negative depths remain
                    if np.any(correlation_depths < 0):
                        # If any negative values remain, filter them out again
                        mask_positive_strict = correlation_depths >= 0
                        correlation_signal = correlation_signal[mask_positive_strict]
                        correlation_depths = correlation_depths[mask_positive_strict]
                    
                    # Then filter correlation_depths for sediment layers only
                    if len(correlation_depths) > 0:
                        # Use >= to include water-bottom interface (at water_depth) and all deeper sediment layers
                        sediment_correlation_mask = correlation_depths >= water_depth
                        sediment_correlation = correlation_signal[sediment_correlation_mask]
                        sediment_correlation_depths = correlation_depths[sediment_correlation_mask]
                    else:
                        sediment_correlation = np.array([])
                        sediment_correlation_depths = np.array([])
                    
                    if len(sediment_correlation) > 0:
                        # Verify no negative depths before plotting
                        assert np.all(sediment_correlation_depths >= 0), f"Negative depths detected: {sediment_correlation_depths[sediment_correlation_depths < 0]}"
                        
                        # Plot correlation on secondary Y-axis (only positive values)
                        # Correlation is cross-correlation (convolution) of CHIRP signal
                        # Take absolute value to show only positive correlation
                        sediment_correlation_positive = np.abs(sediment_correlation)
                        correlation_item = pg.PlotDataItem(sediment_correlation_depths, sediment_correlation_positive, pen='m', name="Correlation")
                        self.correlation_axis.addItem(correlation_item)
                        # Set label for secondary axis
                        self.profile_plot.setLabel('right', 'Correlation', units='V')
                        # Auto-range secondary axis (only positive values, starting from 0)
                        self.correlation_axis.setYRange(0, np.max(sediment_correlation_positive) * 1.1 if len(sediment_correlation_positive) > 0 else 1.0, padding=0.05)
                else:
                    # Clear correlation axis if checkbox is unchecked
                    self.correlation_axis.clear()
                    self.profile_plot.setLabel('right', '')
                
                # Mark detected interfaces (same as RAW mode, but only from water_depth)
                if show_signal and output_dto.interface_depths and output_dto.profile_water_depth is not None and output_dto.profile_raw_signal:
                    raw_signal = np.array(output_dto.profile_raw_signal)
                    interface_depths_raw = np.array(output_dto.interface_depths)
                    
                    # interface_depths are in sediment (from bottom), convert to full depth from surface
                    interface_full_depths = []
                    interface_raw_amps = []
                    
                    for i, depth_in_sediment in enumerate(interface_depths_raw):
                        if i == 0:
                            # First interface (water-bottom) is at water_depth
                            interface_full_depth = water_depth
                        else:
                            # Other interfaces are at water_depth + depth_in_sediment
                            interface_full_depth = water_depth + depth_in_sediment
                        
                        # Only include interfaces >= water_depth (starting from first echo)
                        if interface_full_depth >= water_depth:
                            interface_full_depths.append(interface_full_depth)
                            
                            # Find closest depth index in depths array
                            depth_idx = np.argmin(np.abs(depths - interface_full_depth))
                            # Get amplitude from raw signal at this position
                            if depth_idx < len(raw_signal):
                                interface_raw_amps.append(raw_signal[depth_idx])
                            else:
                                interface_raw_amps.append(0.0)
                    
                    if len(interface_full_depths) > 0:
                        interface_full_depths = np.array(interface_full_depths)
                        interface_raw_amps = np.array(interface_raw_amps)
                        
                        self.profile_plot.plot(interface_full_depths, interface_raw_amps, 
                                              pen=None, symbol='o', symbolSize=10,
                                              symbolBrush='r', name="Interfaces")
            
            # Set X-range based on mode (RAW or Ground)
            # correlation_axis is X-linked to profile_plot, so setting X-range on profile_plot controls both
            if show_raw_mode:
                # RAW mode: X-axis from 0 to maximum depth (including water column + all sediment layers)
                if output_dto.profile_depths:
                    all_depths = np.array(output_dto.profile_depths)
                    # Also consider interface depths for maximum extent
                    max_profile_depth = np.max(all_depths) if len(all_depths) > 0 else 0.0
                    if output_dto.interface_depths and output_dto.profile_water_depth is not None:
                        water_depth = output_dto.profile_water_depth
                        interface_depths = np.array(output_dto.interface_depths)
                        # Convert interface depths (in sediment) to full depths (from surface)
                        interface_full_depths = water_depth + interface_depths
                        max_interface_depth = np.max(interface_full_depths) if len(interface_full_depths) > 0 else max_profile_depth
                        x_max = max(max_profile_depth, max_interface_depth)
                    else:
                        x_max = max_profile_depth
                    
                    x_min = 0.0  # Always start from 0 in RAW mode
                    self.profile_plot.setXRange(x_min, x_max, padding=0.05)
                else:
                    self.profile_plot.autoRange()
            else:
                # Ground mode: X-axis from water_depth to maximum depth (sediment layers only)
                if output_dto.profile_water_depth is not None and output_dto.profile_depths:
                    water_depth = output_dto.profile_water_depth
                    all_depths = np.array(output_dto.profile_depths)
                    # Use >= to include water-bottom interface (at water_depth) and all deeper sediment layers
                    sediment_depths = all_depths[all_depths >= water_depth]
                    
                    # Also consider interface depths (in sediment) for maximum extent
                    max_sediment_depth = np.max(sediment_depths) if len(sediment_depths) > 0 else water_depth
                    if output_dto.interface_depths:
                        interface_depths = np.array(output_dto.interface_depths)
                        # interface_depths are in sediment (from bottom), convert to full depths
                        interface_full_depths = water_depth + interface_depths
                        # Filter to only sediment interfaces (exclude water-bottom at index 0)
                        if len(interface_full_depths) > 1:
                            sediment_interface_depths = interface_full_depths[1:]  # Skip water-bottom
                            max_interface_depth = np.max(sediment_interface_depths) if len(sediment_interface_depths) > 0 else max_sediment_depth
                            x_max = max(max_sediment_depth, max_interface_depth)
                        else:
                            x_max = max_sediment_depth
                    else:
                        x_max = max_sediment_depth
                    
                    x_min = water_depth  # Start from water_depth (first echo = beginning of sediment)
                    self.profile_plot.setXRange(x_min, x_max, padding=0.05)
                else:
                    self.profile_plot.autoRange()
            
            # Y-axis auto-range for both modes
            self.profile_plot.getViewBox().enableAutoRange(axis='y', enable=True)
    
    def _on_profile_mode_changed(self):
        """Handle profile mode change."""
        # Reset X-axis label to default
        self.profile_plot.setLabel('bottom', 'Depth', units='m')
        if self.current_output:
            self._update_visualization(self.current_output)
    
    def _on_signal_checkbox_changed(self):
        """Handle signal checkbox state change."""
        if self.current_output:
            self._update_visualization(self.current_output)
    
    def _on_correlation_checkbox_changed(self):
        """Handle correlation checkbox change."""
        if self.current_output:
            self._update_visualization(self.current_output)
    
    def _update_results_text(self, output_dto: OutputDTO):
        """Update results text."""
        text = "=== Simulation Results ===\n\n"
        
        # Receiver gain information
        text += "=== Receiver Gain ===\n"
        if output_dto.lna_gain is not None:
            text += f"LNA Gain: {output_dto.lna_gain:.1f} dB\n"
        if output_dto.vga_gain is not None:
            text += f"VGA Gain: {output_dto.vga_gain:.1f} dB"
            if output_dto.vga_gain_max is not None:
                text += f" (max: {output_dto.vga_gain_max:.1f} dB)"
            text += "\n"
        text += "\n"
        
        # ADC parameters
        text += "=== ADC Parameters ===\n"
        if output_dto.adc_full_scale is not None:
            text += f"Full Scale (V_FS): {output_dto.adc_full_scale:.3f} V\n"
        if output_dto.adc_range is not None:
            text += f"Input Range: ±{output_dto.adc_range:.3f} V\n"
        if output_dto.adc_bits is not None:
            text += f"Resolution: {output_dto.adc_bits} bits\n"
        if output_dto.adc_dynamic_range is not None:
            text += f"Dynamic Range: {output_dto.adc_dynamic_range:.1f} dB\n"
        
        # Clipping status
        if output_dto.clipping_flags:
            text += f"⚠️ ADC Clipping: DETECTED (signal exceeds ±{output_dto.adc_range:.3f} V)\n"
        else:
            text += f"✓ ADC Clipping: No clipping detected\n"
        text += "\n"
        
        # SNR and range
        text += "=== Signal Quality ===\n"
        text += f"SNR (ADC): {output_dto.SNR_ADC:.2f} dB\n"
        text += f"Range accuracy (σ_D): {output_dto.sigma_D:.4f} m\n"
        text += "\n"
        
        # SBP profile results
        if output_dto.max_penetration_depth:
            text += "=== Sub-Bottom Profile ===\n"
            text += f"Max penetration depth: {output_dto.max_penetration_depth:.2f} m\n"
        
        if output_dto.vertical_resolution:
            text += f"Vertical resolution: {output_dto.vertical_resolution:.4f} m\n"
        
        if output_dto.interface_depths:
            text += f"\nDetected interfaces: {len(output_dto.interface_depths)}\n"
            for i, (depth, amp) in enumerate(zip(output_dto.interface_depths, 
                                                  output_dto.interface_amplitudes or [])):
                text += f"  Interface {i+1}: {depth:.2f} m (amplitude: {amp:.4f} V)\n"
        
        if output_dto.errors:
            text += f"\n=== Errors ===\n"
            for error in output_dto.errors:
                text += f"  - {error}\n"
        
        if output_dto.warnings:
            text += f"\n=== Warnings ===\n"
            for warning in output_dto.warnings:
                text += f"  - {warning}\n"
        
        self.results_text.setText(text)
    
    def _update_signal_path_diagram(self, output_dto: OutputDTO):
        """Update signal path diagram with data from core (no calculations in GUI)."""
        if not output_dto.signal_path:
            # Diagram will show "No data to display" message
            self.signal_path_diagram.set_path_data(None)
            return
        
        # Set path data to diagram - all calculations are done in core
        path_data = output_dto.signal_path
        self.signal_path_diagram.set_path_data(path_data)
    
    def _get_gui_settings_path(self) -> Path:
        """Get path to GUI settings file."""
        return Path(__file__).parent.parent / 'sbp_config.json'
    
    def _get_default_gui_settings(self) -> dict:
        """Returns default GUI settings."""
        return {
            'signal': {
                'f_start': 3000.0,
                'f_end': 12000.0,
                'Tp': 1000.0,
                'window': 'Tukey',  # Tukey (α=0.25) recommended for SBP (section 23.9) for symmetric correlation response
                'sample_rate': 2000000.0,
                'snr_pre': -5.0
            },
            'environment': {
                'T': 15.0,
                'S': 35.0,
                'z': 50.0
            },
            'receiver': {
                'vga_gain': 30.0
            },
            'hardware': {
                'transducer_id': '',
                'lna_id': '',
                'vga_id': '',
                'adc_id': ''
            },
            'sediment_profile': {
                'layers': [
                    {'name': 'Clay', 'thickness': 2.0, 'density': 1600, 'sound_speed': 1550, 'attenuation': 1.0},
                    {'name': 'Silt', 'thickness': 3.0, 'density': 1800, 'sound_speed': 1600, 'attenuation': 2.0},
                    {'name': 'Sand', 'thickness': 5.0, 'density': 2000, 'sound_speed': 1700, 'attenuation': 3.0}
                ],
                'water_depth': 0.0
            }
        }
    
    def _save_gui_settings(self):
        """Saves current GUI settings to sbp_config.json file."""
        if not hasattr(self, 'f_start_spin'):
            # UI not created yet, skip saving
            return
        
        # Don't save during loading to prevent overwriting loaded values
        if getattr(self, '_loading_settings', False):
            return
        
        # Don't save before GUI settings are loaded
        if not getattr(self, '_gui_settings_loaded', False):
            return
        
        try:
            # Get sediment profile
            sediment_layers = []
            for row in range(self.sediment_table.rowCount()):
                name_item = self.sediment_table.item(row, 0)
                thickness_item = self.sediment_table.item(row, 1)
                density_item = self.sediment_table.item(row, 2)
                speed_item = self.sediment_table.item(row, 3)
                atten_item = self.sediment_table.item(row, 4)
                
                if all([name_item, thickness_item, density_item, speed_item, atten_item]):
                    try:
                        layer = {
                            'name': name_item.text() or f"Layer {row+1}",
                            'thickness': float(thickness_item.text()),
                            'density': float(density_item.text()),
                            'sound_speed': float(speed_item.text()),
                            'attenuation': float(atten_item.text())
                        }
                        sediment_layers.append(layer)
                    except ValueError:
                        continue
            
            settings = {
                'signal': {
                    'f_start': self.f_start_spin.value(),
                    'f_end': self.f_end_spin.value(),
                    'Tp': self.tp_spin.value(),
                    'window': self.window_combo.currentText(),
                    'sample_rate': 2e6,  # Default
                    'snr_pre': self.snr_pre_spin.value() if hasattr(self, 'snr_pre_spin') else -5.0
                },
                'environment': {
                    'T': self.temp_spin.value(),
                    'S': self.salinity_spin.value(),
                    'z': self.depth_spin.value()
                },
                'receiver': {
                    'vga_gain': self.vga_gain_spin.value() if hasattr(self, 'vga_gain_spin') else 30.0
                },
                'hardware': {
                    'transducer_id': self.transducer_combo.currentText(),
                    'lna_id': self.lna_combo.currentText(),
                    'vga_id': self.vga_combo.currentText(),
                    'adc_id': self.adc_combo.currentText()
                },
                'sediment_profile': {
                    'layers': sediment_layers,
                    'water_depth': 0.0
                }
            }
            
            settings_path = self._get_gui_settings_path()
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved SBP GUI settings to {settings_path}")
        except Exception as e:
            self.logger.error(f"Error saving GUI settings: {e}", exc_info=True)
    
    def _load_gui_settings(self):
        """Loads GUI settings from sbp_config.json file."""
        settings_path = self._get_gui_settings_path()
        
        try:
            if settings_path.exists():
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                self.logger.info(f"Loaded SBP GUI settings from {settings_path}")
            else:
                # Create default settings file
                settings = self._get_default_gui_settings()
                with open(settings_path, 'w', encoding='utf-8') as f:
                    json.dump(settings, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Created default SBP GUI settings file: {settings_path}")
        except Exception as e:
            self.logger.error(f"Error loading GUI settings: {e}", exc_info=True)
            settings = self._get_default_gui_settings()
        
        # Apply settings to UI
        self._loading_settings = True
        
        try:
            # Load signal parameters
            if 'signal' in settings:
                sig = settings['signal']
                if hasattr(self, 'f_start_spin'):
                    self.f_start_spin.blockSignals(True)
                    self.f_start_spin.setValue(sig.get('f_start', 3000.0))
                    self.f_start_spin.blockSignals(False)
                if hasattr(self, 'f_end_spin'):
                    self.f_end_spin.blockSignals(True)
                    self.f_end_spin.setValue(sig.get('f_end', 12000.0))
                    self.f_end_spin.blockSignals(False)
                if hasattr(self, 'tp_spin'):
                    self.tp_spin.blockSignals(True)
                    self.tp_spin.setValue(sig.get('Tp', 1000.0))
                    self.tp_spin.blockSignals(False)
                if hasattr(self, 'window_combo'):
                    window = sig.get('window', 'Hann')
                    index = self.window_combo.findText(window)
                    if index >= 0:
                        self.window_combo.blockSignals(True)
                        self.window_combo.setCurrentIndex(index)
                        self.window_combo.blockSignals(False)
                if hasattr(self, 'snr_pre_spin'):
                    self.snr_pre_spin.blockSignals(True)
                    self.snr_pre_spin.setValue(sig.get('snr_pre', -5.0))
                    self.snr_pre_spin.blockSignals(False)
            
            # Load environment parameters
            if 'environment' in settings:
                env = settings['environment']
                if hasattr(self, 'temp_spin'):
                    self.temp_spin.blockSignals(True)
                    self.temp_spin.setValue(env.get('T', 15.0))
                    self.temp_spin.blockSignals(False)
                if hasattr(self, 'salinity_spin'):
                    self.salinity_spin.blockSignals(True)
                    self.salinity_spin.setValue(env.get('S', 35.0))
                    self.salinity_spin.blockSignals(False)
                if hasattr(self, 'depth_spin'):
                    self.depth_spin.blockSignals(True)
                    self.depth_spin.setValue(env.get('z', 50.0))
                    self.depth_spin.blockSignals(False)
            
            # Load hardware selection (must be before receiver to set VGA range)
            if 'hardware' in settings:
                hw = settings['hardware']
                if hasattr(self, 'transducer_combo') and hw.get('transducer_id'):
                    transducer_id = hw.get('transducer_id')
                    index = self.transducer_combo.findText(transducer_id)
                    if index >= 0:
                        self.transducer_combo.blockSignals(True)
                        self.transducer_combo.setCurrentIndex(index)
                        self.transducer_combo.blockSignals(False)
                        # Update CHIRP frequencies for selected transducer
                        self._on_transducer_changed(transducer_id)
                
                if hasattr(self, 'lna_combo') and hw.get('lna_id'):
                    lna_id = hw.get('lna_id')
                    index = self.lna_combo.findText(lna_id)
                    if index >= 0:
                        self.lna_combo.blockSignals(True)
                        self.lna_combo.setCurrentIndex(index)
                        self.lna_combo.blockSignals(False)
                
                if hasattr(self, 'vga_combo') and hw.get('vga_id'):
                    vga_id = hw.get('vga_id')
                    index = self.vga_combo.findText(vga_id)
                    if index >= 0:
                        self.vga_combo.blockSignals(True)
                        self.vga_combo.setCurrentIndex(index)
                        self.vga_combo.blockSignals(False)
                        # Update VGA gain range after VGA is selected
                        self._on_vga_changed(vga_id)
                
                if hasattr(self, 'adc_combo') and hw.get('adc_id'):
                    adc_id = hw.get('adc_id')
                    index = self.adc_combo.findText(adc_id)
                    if index >= 0:
                        self.adc_combo.blockSignals(True)
                        self.adc_combo.setCurrentIndex(index)
                        self.adc_combo.blockSignals(False)
            
            # Load receiver parameters (after hardware to ensure VGA range is set)
            if 'receiver' in settings:
                receiver = settings['receiver']
                if hasattr(self, 'vga_gain_spin'):
                    self.vga_gain_spin.blockSignals(True)
                    self.vga_gain_spin.setValue(receiver.get('vga_gain', 30.0))
                    self.vga_gain_spin.blockSignals(False)
            
            # Load sediment profile
            if 'sediment_profile' in settings:
                profile = settings['sediment_profile']
                layers = profile.get('layers', [])
                if hasattr(self, 'sediment_table') and layers:
                    # Clear existing rows
                    self.sediment_table.setRowCount(len(layers))
                    
                    for row, layer in enumerate(layers):
                        self.sediment_table.setItem(row, 0, QTableWidgetItem(layer.get('name', f'Layer {row+1}')))
                        self.sediment_table.setItem(row, 1, QTableWidgetItem(str(layer.get('thickness', 1.0))))
                        self.sediment_table.setItem(row, 2, QTableWidgetItem(str(layer.get('density', 1800))))
                        self.sediment_table.setItem(row, 3, QTableWidgetItem(str(layer.get('sound_speed', 1600))))
                        self.sediment_table.setItem(row, 4, QTableWidgetItem(str(layer.get('attenuation', 2.0))))
        except Exception as e:
            self.logger.error(f"Error applying GUI settings: {e}", exc_info=True)
        finally:
            self._loading_settings = False
            # Calculate optimal Tp after settings are loaded
            if hasattr(self, 'signal_calculator') and hasattr(self, 'snr_pre_spin'):
                self._calculate_optimal_tp()
    
    def closeEvent(self, event):
        """Handle window close event - save settings before closing."""
        self._save_gui_settings()
        event.accept()


def main():
    """Main entry point."""
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = SBPWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

