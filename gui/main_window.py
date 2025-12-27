"""
Main application window.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QComboBox,
                             QGroupBox, QFormLayout, QTextEdit, QFileDialog,
                             QMessageBox, QTabWidget, QSpinBox, QDoubleSpinBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import sys
from pathlib import Path

# Add root directory to path for imports
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.dto import InputDTO, OutputDTO
from core.simulator import Simulator
from core.signal_calculator import SignalCalculator
from core.signal_path import SignalPathCalculator
from core.enob_calculator import ENOBCalculator
from data.data_provider import DataProvider


class SimulationThread(QThread):
    """Thread for simulation execution."""
    
    finished = pyqtSignal(object, object)  # input_dto, output_dto
    
    def __init__(self, simulator, input_dto, absorption_only=True, vga_gain=None):
        super().__init__()
        self.simulator = simulator
        self.input_dto = input_dto
        self.absorption_only = absorption_only
        self.vga_gain = vga_gain
    
    def run(self):
        """Executes simulation."""
        output_dto = self.simulator.simulate(self.input_dto, absorption_only=self.absorption_only, vga_gain=self.vga_gain)
        self.finished.emit(self.input_dto, output_dto)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hydroacoustic Sonar Simulation")
        self.setGeometry(100, 100, 1400, 1000)
        
        # Инициализация
        self.data_provider = DataProvider()
        self.simulator = Simulator(self.data_provider)
        self.signal_calculator = SignalCalculator()
        self.signal_path_calculator = SignalPathCalculator(self.data_provider)
        self.simulation_history = []  # Iteration history
        
        # Flag to prevent overwriting values during loading
        self._loading_settings = False
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Create UI
        self._create_ui()
        
        # Load data first (populate combo boxes)
        self._load_data()
        
        # Load GUI settings from gui.json will be done in showEvent
        # after all widgets are fully displayed
        # This prevents parameters from being overwritten during initialization
        self._gui_settings_loaded = False
        
        # Note: Don't save immediately after loading - this would overwrite user's manual changes
        # Settings will be saved automatically when user changes any parameter
    
    def _setup_logging(self):
        """Setup logging."""
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'simulator.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def _create_ui(self):
        """Creates user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Top part - parameters (left), results (center), signal path (right)
        top_layout = QHBoxLayout()
        
        # Left panel - parameters
        left_panel = self._create_parameters_panel()
        top_layout.addWidget(left_panel, 1)
        
        # Center panel - results
        right_panel = self._create_results_panel()
        top_layout.addWidget(right_panel, 1)
        
        # Right panel - signal path and optimization strategy
        from gui.signal_path_widget import SignalPathWidget
        self.signal_path_widget = SignalPathWidget()
        self.signal_path_widget.lna_gain_changed.connect(self._on_signal_path_param_changed)
        self.signal_path_widget.lna_nf_changed.connect(self._on_signal_path_param_changed)
        self.signal_path_widget.vga_gain_changed.connect(self._on_signal_path_param_changed)
        # Bottom reflection is now in environment parameters, not in signal_path_widget
        
        # Create vertical container for Signal Path and Optimization Strategy
        right_panel_container = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_container)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        
        signal_path_group = QGroupBox("Signal Path")
        signal_path_layout = QVBoxLayout()
        signal_path_layout.addWidget(self.signal_path_widget)
        signal_path_group.setLayout(signal_path_layout)
        # Set maximum width to limit Signal Path width
        signal_path_group.setMaximumWidth(680)
        right_panel_layout.addWidget(signal_path_group, 1)  # Allow Signal Path to expand
        
        # Optimization strategy selection (under Signal Path)
        strategy_group = QGroupBox("Optimization Strategy")
        strategy_layout = QVBoxLayout()
        strategy_layout.setContentsMargins(5, 5, 5, 5)
        strategy_layout.setSpacing(5)
        
        self.optimization_strategy_combo = QComboBox()
        self.optimization_strategy_combo.addItem("Maximum Tp + Minimum VGA Gain", "max_tp_min_vga")
        self.optimization_strategy_combo.addItem("Minimum Tp for Target SNR", "min_tp_for_snr")
        self.optimization_strategy_combo.setToolTip(
            "Strategy 1 (Max Tp + Min VGA): Set Tp to maximum (80% TOF for D_target), then find minimum VGA Gain for target SNR.\n"
            "Strategy 2 (Min Tp for SNR): Find minimum Tp needed to achieve target SNR at D_target."
        )
        strategy_layout.addWidget(QLabel("Strategy:"))
        strategy_layout.addWidget(self.optimization_strategy_combo)
        strategy_group.setLayout(strategy_layout)
        strategy_group.setMaximumWidth(680)
        # Set size policy to prevent vertical expansion
        from PyQt5.QtWidgets import QSizePolicy
        size_policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        strategy_group.setSizePolicy(size_policy)
        right_panel_layout.addWidget(strategy_group, 0)  # Set stretch factor to 0 - no expansion
        
        # Add stretch at the end to push everything up
        right_panel_layout.addStretch()
        
        # Add the container to top layout
        top_layout.addWidget(right_panel_container, 0)
        
        main_layout.addLayout(top_layout, 1)
    
    def _create_parameters_panel(self) -> QWidget:
        """Creates parameters panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Hardware
        hardware_group = QGroupBox("Hardware")
        hardware_layout = QFormLayout()
        
        self.transducer_combo = QComboBox()
        self.transducer_combo.currentTextChanged.connect(self._on_transducer_changed)
        self.transducer_combo.currentTextChanged.connect(self._save_gui_settings)
        hardware_layout.addRow("Transducer:", self.transducer_combo)
        
        # Transducer information (read-only)
        self.transducer_info_label = QLabel("")
        self.transducer_info_label.setWordWrap(True)
        self.transducer_info_label.setStyleSheet("color: blue; font-size: 9pt;")
        hardware_layout.addRow("", self.transducer_info_label)
        
        self.lna_combo = QComboBox()
        self.lna_combo.currentTextChanged.connect(self._save_gui_settings)
        hardware_layout.addRow("LNA:", self.lna_combo)
        
        self.vga_combo = QComboBox()
        self.vga_combo.currentTextChanged.connect(self._save_gui_settings)
        hardware_layout.addRow("VGA:", self.vga_combo)
        
        self.adc_combo = QComboBox()
        self.adc_combo.currentTextChanged.connect(self._save_gui_settings)
        hardware_layout.addRow("ADC:", self.adc_combo)
        
        hardware_group.setLayout(hardware_layout)
        layout.addWidget(hardware_group)
        
        # Signal
        signal_group = QGroupBox("CHIRP Signal")
        signal_layout = QFormLayout()
        
        # Checkbox to choose between recommended and user-defined frequencies
        self.use_recommended_freq_checkbox = QCheckBox("Use recommended frequencies (from manufacturer data)")
        self.use_recommended_freq_checkbox.setChecked(False)
        self.use_recommended_freq_checkbox.stateChanged.connect(self._on_frequency_mode_changed)
        signal_layout.addRow("", self.use_recommended_freq_checkbox)
        
        # Recommended frequencies (from manufacturer data, read-only)
        self.f_start_recommended_label = QLabel("—")
        self.f_start_recommended_label.setStyleSheet("color: blue; font-weight: bold;")
        signal_layout.addRow("f_start (recommended, from manufacturer):", self.f_start_recommended_label)
        
        self.f_end_recommended_label = QLabel("—")
        self.f_end_recommended_label.setStyleSheet("color: blue; font-weight: bold;")
        signal_layout.addRow("f_end (recommended, from manufacturer):", self.f_end_recommended_label)
        
        # User-defined frequencies
        self.f_start_spin = QDoubleSpinBox()
        self.f_start_spin.setRange(1000, 1000000)
        self.f_start_spin.setValue(150000)
        self.f_start_spin.setSuffix(" Hz")
        self.f_start_spin.valueChanged.connect(self._on_chirp_frequency_changed)
        self.f_start_spin.valueChanged.connect(self._save_gui_settings)
        signal_layout.addRow("f_start (user-defined):", self.f_start_spin)
        
        self.f_end_spin = QDoubleSpinBox()
        self.f_end_spin.setRange(1000, 1000000)
        self.f_end_spin.setValue(250000)
        self.f_end_spin.setSuffix(" Hz")
        self.f_end_spin.valueChanged.connect(self._on_chirp_frequency_changed)
        self.f_end_spin.valueChanged.connect(self._save_gui_settings)
        signal_layout.addRow("f_end (user-defined):", self.f_end_spin)
        
        self.Tp_spin = QDoubleSpinBox()
        # No artificial maximum limit - only physical constraint (80% of round-trip time) applies
        # Set a very large initial maximum (will be updated based on D_min)
        self.Tp_spin.setRange(1.0, 10000000.0)  # 1 µs to 10 seconds (10,000,000 µs)
        self.Tp_spin.setValue(500)
        self.Tp_spin.setSuffix(" µs")
        self.Tp_spin.valueChanged.connect(self._on_tp_changed)
        self.Tp_spin.valueChanged.connect(self._save_gui_settings)
        signal_layout.addRow("Tp:", self.Tp_spin)
        
        # Tp_min: Minimum pulse duration for minimum distance (D_min)
        # This is the maximum allowed Tp based on physical constraint
        self.Tp_min_label = QLabel("—")
        self.Tp_min_label.setStyleSheet("color: orange; font-weight: bold;")
        signal_layout.addRow("Tp_min (for D_min, physical constraint):", self.Tp_min_label)
        
        # Tp_optimal: Optimal pulse duration for target distance (D_target)
        # This can be larger than Tp_min if D_target > D_min
        self.Tp_optimal_label = QLabel("—")
        self.Tp_optimal_label.setStyleSheet("color: green; font-weight: bold;")
        signal_layout.addRow("Tp_optimal (for D_target):", self.Tp_optimal_label)
        
        self.window_combo = QComboBox()
        self.window_combo.addItems(["Rect", "Hann", "Tukey"])
        self.window_combo.currentTextChanged.connect(self._save_gui_settings)
        signal_layout.addRow("Window:", self.window_combo)
        
        self.sample_rate_spin = QDoubleSpinBox()
        self.sample_rate_spin.setRange(100000, 10000000)  # 100 kHz to 10 MHz
        self.sample_rate_spin.setValue(2000000)  # Default 2 MHz
        self.sample_rate_spin.setSuffix(" Hz")
        self.sample_rate_spin.setToolTip("Sampling frequency for signal generation. Must be >= 2*f_end (Nyquist criterion)")
        self.sample_rate_spin.valueChanged.connect(self._on_sample_rate_changed)
        self.sample_rate_spin.valueChanged.connect(self._save_gui_settings)
        signal_layout.addRow("Sample Rate:", self.sample_rate_spin)
        
        signal_group.setLayout(signal_layout)
        layout.addWidget(signal_group)
        
        # Environment
        environment_group = QGroupBox("Environment Parameters")
        environment_layout = QFormLayout()
        
        self.T_spin = QDoubleSpinBox()
        self.T_spin.setRange(0, 30)
        self.T_spin.setValue(25)  # Default will be overridden by gui.json if exists
        self.T_spin.setSuffix(" °C")
        self.T_spin.valueChanged.connect(self._on_environment_changed)
        self.T_spin.valueChanged.connect(self._save_gui_settings)
        environment_layout.addRow("Temperature:", self.T_spin)
        
        self.S_spin = QDoubleSpinBox()
        self.S_spin.setRange(0, 35)
        self.S_spin.setValue(35)  # Default will be overridden by gui.json if exists
        self.S_spin.setSuffix(" PSU")
        self.S_spin.valueChanged.connect(self._on_environment_changed)
        self.S_spin.valueChanged.connect(self._save_gui_settings)
        environment_layout.addRow("Salinity:", self.S_spin)
        
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(0.2, 300)
        self.z_spin.setValue(100)  # Default will be overridden by gui.json if exists
        self.z_spin.setSuffix(" m")
        self.z_spin.setToolTip(
            "Water depth (vertical distance from surface to bottom).\n"
            "Affects:\n"
            "  • Water pressure (P = 1.0 + 0.1*z dBar)\n"
            "  • Sound speed (increases with pressure)\n"
            "  • Attenuation coefficient (depends on pressure)\n"
            "Note: This is different from 'Target Range' - depth is vertical, range is distance to target.\n"
            "If 'Link to Target Range' is enabled, changing depth will update target range."
        )
        self.z_spin.valueChanged.connect(self._on_environment_changed)
        self.z_spin.valueChanged.connect(self._save_gui_settings)
        environment_layout.addRow("Water Depth (z):", self.z_spin)
        
        # Option to link depth to target range (if target is on bottom)
        self.link_depth_to_range_checkbox = QCheckBox("Link Depth to Target Range")
        self.link_depth_to_range_checkbox.setToolTip(
            "If enabled, Target Range will automatically equal Water Depth.\n"
            "Useful when target is on the bottom."
        )
        self.link_depth_to_range_checkbox.setChecked(False)
        self.link_depth_to_range_checkbox.stateChanged.connect(self._on_link_depth_to_range_changed)
        self.link_depth_to_range_checkbox.stateChanged.connect(self._save_gui_settings)
        environment_layout.addRow("", self.link_depth_to_range_checkbox)
        
        self.bottom_reflection_spin = QDoubleSpinBox()
        self.bottom_reflection_spin.setRange(-30, 0)
        self.bottom_reflection_spin.setValue(-15.0)
        self.bottom_reflection_spin.setSuffix(" dB")
        self.bottom_reflection_spin.setToolTip(
            "Bottom reflection loss (single reflection, not round-trip).\n"
            "Typical values:\n"
            "  • Mud: -10 to -20 dB\n"
            "  • Sand: -6 to -12 dB\n"
            "  • Gravel: -3 to -8 dB\n"
            "  • Rock/Concrete: -1 to -3 dB"
        )
        self.bottom_reflection_spin.valueChanged.connect(self._on_environment_changed)
        self.bottom_reflection_spin.valueChanged.connect(self._save_gui_settings)
        environment_layout.addRow("Bottom Reflection:", self.bottom_reflection_spin)
        
        environment_group.setLayout(environment_layout)
        layout.addWidget(environment_group)
        
        # Range
        range_group = QGroupBox("Range")
        range_layout = QFormLayout()
        
        self.D_min_spin = QDoubleSpinBox()
        self.D_min_spin.setRange(0.1, 1000)
        self.D_min_spin.setValue(0.5)  # Default will be overridden by gui.json if exists
        self.D_min_spin.setSuffix(" m")
        self.D_min_spin.valueChanged.connect(self._on_d_min_changed)
        self.D_min_spin.valueChanged.connect(self._save_gui_settings)
        range_layout.addRow("D_min:", self.D_min_spin)
        
        self.D_max_spin = QDoubleSpinBox()
        self.D_max_spin.setRange(0.1, 1000)
        self.D_max_spin.setValue(100.0)  # Default will be overridden by gui.json if exists
        self.D_max_spin.setSuffix(" m")
        self.D_max_spin.valueChanged.connect(self._on_d_max_changed)
        self.D_max_spin.valueChanged.connect(self._save_gui_settings)
        range_layout.addRow("D_max:", self.D_max_spin)
        
        # Target distance for simulation
        self.D_target_spin = QDoubleSpinBox()
        self.D_target_spin.setRange(0.1, 1000)
        self.D_target_spin.setValue(100.0)  # Default will be overridden by gui.json if exists
        self.D_target_spin.setSuffix(" m")
        self.D_target_spin.setToolTip(
            "Target range (distance from sonar to target).\n"
            "Used for:\n"
            "  • Signal simulation at this distance\n"
            "  • Optimal pulse duration (Tp) calculation\n"
            "  • Comparison with measured range\n"
            "Note: This is the distance to target (can be any direction).\n"
            "Different from 'Water Depth' which is vertical depth affecting water properties.\n"
            "If 'Link Depth to Target Range' is enabled, this will be set equal to Water Depth."
        )
        self.D_target_spin.valueChanged.connect(self._on_d_target_changed)
        # Connect directly to _save_gui_settings with logging
        self.D_target_spin.valueChanged.connect(lambda v: self._log_and_save('D_target', v))
        range_layout.addRow("Target Range (D_target):", self.D_target_spin)
        
        range_group.setLayout(range_layout)
        layout.addWidget(range_group)
        
        # Tp Optimization
        optimization_group = QGroupBox("Pulse Duration Optimization")
        optimization_layout = QFormLayout()
        
        # Target SNR
        self.target_snr_spin = QDoubleSpinBox()
        self.target_snr_spin.setRange(10, 40)  # Realistic range for hydroacoustic systems: 10-40 dB
        self.target_snr_spin.setValue(20.0)
        self.target_snr_spin.setSuffix(" dB")
        self.target_snr_spin.setToolTip("Target SNR at receiver (considers attenuation). Range: 10-40 dB")
        self.target_snr_spin.valueChanged.connect(self._save_gui_settings)
        optimization_layout.addRow("Target SNR:", self.target_snr_spin)
        
        # Calculate optimal Tp button
        from PyQt5.QtWidgets import QPushButton
        self.calculate_tp_button = QPushButton("Calculate Optimal Tp")
        self.calculate_tp_button.setToolTip(
            "Calculates optimal pulse duration considering:\n"
            "  • Target distance\n"
            "  • Required SNR\n"
            "  • Water attenuation (spreading + absorption)\n"
            "  • Bottom reflection\n"
            "  • Receiver gain"
        )
        self.calculate_tp_button.clicked.connect(self._calculate_optimal_tp)
        optimization_layout.addRow("", self.calculate_tp_button)
        
        optimization_group.setLayout(optimization_layout)
        layout.addWidget(optimization_group)
        
        # Measurement Accuracy
        accuracy_group = QGroupBox("Measurement Accuracy")
        accuracy_layout = QFormLayout()
        
        # Transducer bandwidth (read-only)
        self.bandwidth_label = QLabel("—")
        self.bandwidth_label.setStyleSheet("font-weight: bold;")
        accuracy_layout.addRow("Transducer Bandwidth (B_tr):", self.bandwidth_label)
        
        # Recommended accuracy (read-only)
        self.recommended_accuracy_label = QLabel("—")
        self.recommended_accuracy_label.setStyleSheet("color: green; font-weight: bold;")
        accuracy_layout.addRow("Recommended Accuracy (σ_D):", self.recommended_accuracy_label)
        
        # User-defined target accuracy
        self.target_sigma_spin = QDoubleSpinBox()
        self.target_sigma_spin.setRange(0.0001, 1.0)
        self.target_sigma_spin.setValue(0.01)
        self.target_sigma_spin.setDecimals(4)
        self.target_sigma_spin.setSuffix(" m")
        self.target_sigma_spin.setToolTip("Target measurement accuracy (standard deviation)")
        self.target_sigma_spin.valueChanged.connect(self._on_target_accuracy_changed)
        self.target_sigma_spin.valueChanged.connect(self._save_gui_settings)
        accuracy_layout.addRow("Target Accuracy (σ_D):", self.target_sigma_spin)
        
        # Required bandwidth for target accuracy
        self.required_bandwidth_label = QLabel("—")
        self.required_bandwidth_label.setStyleSheet("color: orange;")
        accuracy_layout.addRow("Required CHIRP Bandwidth:", self.required_bandwidth_label)
        
        accuracy_group.setLayout(accuracy_layout)
        layout.addWidget(accuracy_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.simulate_btn = QPushButton("Run Simulation")
        self.simulate_btn.clicked.connect(self._run_simulation)
        button_layout.addWidget(self.simulate_btn)
        
        self.optimize_btn = QPushButton("Optimize")
        self.optimize_btn.clicked.connect(self._run_optimization)
        button_layout.addWidget(self.optimize_btn)
        
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._export_results)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        return panel
    
    def _create_results_panel(self) -> QWidget:
        """Creates results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tabs
        tabs = QTabWidget()
        
        # Results
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        tabs.addTab(results_tab, "Results")
        
        # Plots
        plots_tab = QWidget()
        plots_layout = QVBoxLayout(plots_tab)
        
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        plots_layout.addWidget(self.canvas)
        
        tabs.addTab(plots_tab, "Plots")
        
        # Signals tab - three graphs: transmitted, at bottom, received
        signals_tab = QWidget()
        signals_layout = QVBoxLayout(signals_tab)
        
        # Checkbox for absorption_only mode
        self.absorption_only_checkbox = QCheckBox("Show only absorption (without spreading)")
        self.absorption_only_checkbox.setChecked(False)  # Default: show full attenuation
        self.absorption_only_checkbox.setToolTip("If checked, shows only frequency-dependent absorption.\n"
                                                 "If unchecked, shows full attenuation (spreading + absorption).")
        self.absorption_only_checkbox.stateChanged.connect(self._on_absorption_only_changed)
        signals_layout.addWidget(self.absorption_only_checkbox)
        
        # Create 3 pyqtgraph PlotWidgets
        self.signals_plot_tx = pg.PlotWidget(title="Graph 1: Original CHIRP Signal")
        self.signals_plot_tx.setLabel('left', 'Amplitude')
        self.signals_plot_tx.setLabel('bottom', 'Time', units='ms')
        self.signals_plot_tx.showGrid(x=True, y=True, alpha=0.3)
        signals_layout.addWidget(self.signals_plot_tx)
        
        self.signals_plot_bottom = pg.PlotWidget(title="Graph 2: Signal After Water Forward (TX -> Target)")
        self.signals_plot_bottom.setLabel('left', 'Amplitude')
        self.signals_plot_bottom.setLabel('bottom', 'Time', units='ms')
        self.signals_plot_bottom.showGrid(x=True, y=True, alpha=0.3)
        signals_layout.addWidget(self.signals_plot_bottom)
        
        self.signals_plot_rx = pg.PlotWidget(title="Graph 3: Signal After Water Backward (Target -> RX)")
        self.signals_plot_rx.setLabel('left', 'Amplitude')
        self.signals_plot_rx.setLabel('bottom', 'Time', units='ms')
        self.signals_plot_rx.showGrid(x=True, y=True, alpha=0.3)
        signals_layout.addWidget(self.signals_plot_rx)
        
        tabs.addTab(signals_tab, "Signals")
        
        # Signals RX tab - graphs 4 and 5: after LNA and after VGA
        signals_rx_tab = QWidget()
        signals_rx_layout = QVBoxLayout(signals_rx_tab)
        
        self.signals_plot_lna = pg.PlotWidget(title="Graph 4: Signal After LNA")
        self.signals_plot_lna.setLabel('left', 'Amplitude')
        self.signals_plot_lna.setLabel('bottom', 'Time', units='ms')
        self.signals_plot_lna.showGrid(x=True, y=True, alpha=0.3)
        signals_rx_layout.addWidget(self.signals_plot_lna)
        
        self.signals_plot_vga = pg.PlotWidget(title="Graph 5: Signal After VGA")
        self.signals_plot_vga.setLabel('left', 'Amplitude')
        self.signals_plot_vga.setLabel('bottom', 'Time', units='ms')
        self.signals_plot_vga.showGrid(x=True, y=True, alpha=0.3)
        signals_rx_layout.addWidget(self.signals_plot_vga)
        
        tabs.addTab(signals_rx_tab, "Signals RX")
        
        # Recommendations
        recommendations_tab = QWidget()
        recommendations_layout = QVBoxLayout(recommendations_tab)
        
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        recommendations_layout.addWidget(self.recommendations_text)
        
        # Button to apply recommendations
        self.apply_recommendations_button = QPushButton("Apply Recommendations")
        self.apply_recommendations_button.setToolTip("Apply suggested parameter changes from optimizer")
        self.apply_recommendations_button.clicked.connect(self._apply_recommendations)
        recommendations_layout.addWidget(self.apply_recommendations_button)
        
        tabs.addTab(recommendations_tab, "Recommendations")
        
        # ENOB tab
        enob_tab = QWidget()
        enob_layout = QVBoxLayout(enob_tab)
        
        self.enob_text = QTextEdit()
        self.enob_text.setReadOnly(True)
        self.enob_text.setFontFamily("Courier")
        enob_layout.addWidget(self.enob_text)
        
        tabs.addTab(enob_tab, "ENOB")
        
        layout.addWidget(tabs)
        
        return panel
    
    def _get_gui_settings_path(self) -> Path:
        """Returns path to gui.json settings file."""
        return Path(__file__).parent.parent / 'gui.json'
    
    def _get_default_gui_settings(self) -> dict:
        """Returns default GUI settings."""
        return {
            'signal': {
                'f_start': 150000.0,
                'f_end': 250000.0,
                'Tp': 500.0,
                'window': 'Hann',
                'sample_rate': 2000000.0
            },
            'environment': {
                'T': 25.0,  # Temperature, °C
                'S': 35.0,  # Salinity, PSU
                'z': 100.0,  # Water Depth, m
                'bottom_reflection': -15.0
            },
            'range': {
                'D_min': 0.5,  # m
                'D_max': 100.0,  # m
                'D_target': 100.0  # m
            },
            'target': {
                'target_snr': 20.0,
                'target_sigma': 0.01
            },
            'hardware': {
                'transducer_id': '',
                'lna_id': '',
                'vga_id': '',
                'adc_id': ''
            },
            'options': {
                'link_depth_to_range': False
            }
        }
    
    def _load_gui_settings(self):
        """Loads GUI settings from gui.json file."""
        settings_path = self._get_gui_settings_path()
        self.logger.info(f"_load_gui_settings: Attempting to load from {settings_path}")
        
        try:
            if settings_path.exists():
                self.logger.info(f"_load_gui_settings: File exists, reading...")
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                self.logger.info(f"Loaded GUI settings from {settings_path}")
                self.logger.debug(f"Settings keys: {list(settings.keys())}")
            else:
                # Create default settings file
                self.logger.warning(f"_load_gui_settings: File does not exist, creating defaults")
                settings = self._get_default_gui_settings()
                # Save default settings directly to file (bypass _save_gui_settings to avoid flags check)
                try:
                    with open(settings_path, 'w', encoding='utf-8') as f:
                        json.dump(settings, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"Created default GUI settings file: {settings_path}")
                except Exception as e:
                    self.logger.error(f"Error creating default GUI settings file: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error loading GUI settings: {e}", exc_info=True)
            settings = self._get_default_gui_settings()
        
        # Apply settings to UI (only if widgets exist)
        # This will be called after _create_ui, so widgets should exist
        # Block all signals during loading to prevent unwanted saves
        try:
            if 'signal' in settings:
                sig = settings['signal']
                self.logger.info(f"Loading signal parameters: f_start={sig.get('f_start')}, f_end={sig.get('f_end')}, Tp={sig.get('Tp')}, sample_rate={sig.get('sample_rate')}")
                if hasattr(self, 'f_start_spin'):
                    self.f_start_spin.blockSignals(True)
                    f_start = sig.get('f_start', 150000.0)
                    self.f_start_spin.setValue(f_start)
                    self.f_start_spin.blockSignals(False)
                    self.logger.info(f"Set f_start to {f_start}")
                if hasattr(self, 'f_end_spin'):
                    self.f_end_spin.blockSignals(True)
                    self.f_end_spin.setValue(sig.get('f_end', 250000.0))
                    self.f_end_spin.blockSignals(False)
                if hasattr(self, 'Tp_spin'):
                    self.Tp_spin.blockSignals(True)
                    self.Tp_spin.setValue(sig.get('Tp', 500.0))
                    self.Tp_spin.blockSignals(False)
                if hasattr(self, 'window_combo'):
                    window = sig.get('window', 'Hann')
                    index = self.window_combo.findText(window)
                    if index >= 0:
                        self.window_combo.blockSignals(True)
                        self.window_combo.setCurrentIndex(index)
                        self.window_combo.blockSignals(False)
                if hasattr(self, 'sample_rate_spin'):
                    self.sample_rate_spin.blockSignals(True)
                    self.sample_rate_spin.setValue(sig.get('sample_rate', 2000000.0))
                    self.sample_rate_spin.blockSignals(False)
            
            if 'environment' in settings:
                env = settings['environment']
                self.logger.info(f"Loading environment parameters: T={env.get('T')}, S={env.get('S')}, z={env.get('z')}, bottom_reflection={env.get('bottom_reflection')}")
                if hasattr(self, 'T_spin'):
                    self.T_spin.blockSignals(True)
                    T = env.get('T', 25.0)
                    self.T_spin.setValue(T)
                    self.T_spin.blockSignals(False)
                    self.logger.info(f"Set T to {T}")
                if hasattr(self, 'S_spin'):
                    self.S_spin.blockSignals(True)
                    self.S_spin.setValue(env.get('S', 35.0))
                    self.S_spin.blockSignals(False)
                if hasattr(self, 'z_spin'):
                    self.z_spin.blockSignals(True)
                    self.z_spin.setValue(env.get('z', 100.0))
                    self.z_spin.blockSignals(False)
                if hasattr(self, 'bottom_reflection_spin'):
                    self.bottom_reflection_spin.blockSignals(True)
                    self.bottom_reflection_spin.setValue(env.get('bottom_reflection', -15.0))
                    self.bottom_reflection_spin.blockSignals(False)
            
            if 'range' in settings:
                rng = settings['range']
                # Load D_min first
                if hasattr(self, 'D_min_spin'):
                    self.D_min_spin.blockSignals(True)
                    D_min = rng.get('D_min', 0.5)
                    self.D_min_spin.setValue(D_min)
                    self.D_min_spin.blockSignals(False)
                    self.logger.info(f"Loaded D_min from gui.json: {D_min} m")
                else:
                    D_min = 0.5
                
                # Load D_max second
                if hasattr(self, 'D_max_spin'):
                    self.D_max_spin.blockSignals(True)
                    D_max = rng.get('D_max', 100.0)
                    # Ensure D_max > D_min
                    if D_max <= D_min:
                        old_D_max = D_max
                        D_max = D_min + 0.1
                        self.logger.warning(f"D_max ({old_D_max}) <= D_min ({D_min}). Adjusted to {D_max}")
                    self.D_max_spin.setValue(D_max)
                    self.D_max_spin.blockSignals(False)
                    self.logger.info(f"Loaded D_max from gui.json: {D_max} m")
                else:
                    D_max = 100.0
                
                # Load D_target last, with validation
                if hasattr(self, 'D_target_spin'):
                    self.D_target_spin.blockSignals(True)
                    D_target_original = rng.get('D_target', 100.0)
                    D_target = D_target_original
                    # Validate D_target: must be within [D_min, D_max]
                    if D_target < D_min:
                        D_target = D_min
                        self.logger.warning(f"D_target ({D_target_original}) < D_min ({D_min}). Adjusted to {D_min}")
                    elif D_target > D_max:
                        D_target = D_max
                        self.logger.warning(f"D_target ({D_target_original}) > D_max ({D_max}). Adjusted to {D_max}")
                    self.D_target_spin.setValue(D_target)
                    self.D_target_spin.blockSignals(False)
                    self.logger.info(f"Loaded D_target from gui.json: {D_target_original} -> {D_target} m (D_min={D_min}, D_max={D_max})")
            
            if 'target' in settings:
                tgt = settings['target']
                if hasattr(self, 'target_snr_spin'):
                    self.target_snr_spin.blockSignals(True)
                    target_snr_value = tgt.get('target_snr', 20.0)
                    # Ensure value is within valid range [10, 40]
                    if target_snr_value < 10:
                        target_snr_value = 10.0
                        self.logger.warning(f"Target SNR ({tgt.get('target_snr', 20.0)}) < 10. Adjusted to 10.0")
                    elif target_snr_value > 40:
                        target_snr_value = 40.0
                        self.logger.warning(f"Target SNR ({tgt.get('target_snr', 20.0)}) > 40. Adjusted to 40.0")
                    self.target_snr_spin.setValue(target_snr_value)
                    self.target_snr_spin.blockSignals(False)
                if hasattr(self, 'target_sigma_spin'):
                    self.target_sigma_spin.blockSignals(True)
                    self.target_sigma_spin.setValue(tgt.get('target_sigma', 0.01))
                    self.target_sigma_spin.blockSignals(False)
            
            if 'hardware' in settings:
                hw = settings['hardware']
                self.logger.info(f"Loading hardware parameters: transducer_id={hw.get('transducer_id')}, lna_id={hw.get('lna_id')}, vga_id={hw.get('vga_id')}, adc_id={hw.get('adc_id')}")
                # Load hardware selections (must be after combo boxes are populated in _load_data)
                # Block all signals to prevent handlers from overwriting loaded values
                if hasattr(self, 'transducer_combo') and hw.get('transducer_id'):
                    index = self.transducer_combo.findText(hw['transducer_id'])
                    if index >= 0:
                        self.transducer_combo.blockSignals(True)
                        self.transducer_combo.setCurrentIndex(index)
                        self.transducer_combo.blockSignals(False)
                        self.logger.info(f"Set transducer to {hw['transducer_id']} (index {index})")
                        # Update frequency ranges and recommended frequencies after loading transducer
                        # This must be done after all settings are loaded to ensure correct values
                        # We'll call _update_accuracy_info in _update_signal_path_after_load
                    else:
                        self.logger.warning(f"Transducer {hw['transducer_id']} not found in combo box")
                        # Don't trigger transducer change handler during loading
                        # This would overwrite loaded values from gui.json
                        # We'll manually update frequency ranges if needed, but not change frequencies
                        # Only update ranges to match transducer capabilities
                        try:
                            transducer_data = self.data_provider.get_transducer(hw['transducer_id'])
                            f_min = transducer_data.get('f_min', 150000)
                            f_max = transducer_data.get('f_max', 250000)
                            # Update ranges only, don't change values
                            if hasattr(self, 'f_start_spin') and hasattr(self, 'f_end_spin'):
                                self.f_start_spin.blockSignals(True)
                                self.f_end_spin.blockSignals(True)
                                self.f_start_spin.setRange(f_min, f_max)
                                self.f_end_spin.setRange(f_min, f_max)
                                self.f_start_spin.blockSignals(False)
                                self.f_end_spin.blockSignals(False)
                        except Exception as e:
                            self.logger.warning(f"Error updating transducer ranges: {e}")
                if hasattr(self, 'lna_combo') and hw.get('lna_id'):
                    index = self.lna_combo.findText(hw['lna_id'])
                    if index >= 0:
                        self.lna_combo.blockSignals(True)
                        self.lna_combo.setCurrentIndex(index)
                        self.lna_combo.blockSignals(False)
                if hasattr(self, 'vga_combo') and hw.get('vga_id'):
                    index = self.vga_combo.findText(hw['vga_id'])
                    if index >= 0:
                        self.vga_combo.blockSignals(True)
                        self.vga_combo.setCurrentIndex(index)
                        self.vga_combo.blockSignals(False)
                if hasattr(self, 'adc_combo') and hw.get('adc_id'):
                    index = self.adc_combo.findText(hw['adc_id'])
                    if index >= 0:
                        self.adc_combo.blockSignals(True)
                        self.adc_combo.setCurrentIndex(index)
                        self.adc_combo.blockSignals(False)
            
            if 'options' in settings:
                opts = settings['options']
                if hasattr(self, 'link_depth_to_range_checkbox'):
                    self.link_depth_to_range_checkbox.blockSignals(True)
                    self.link_depth_to_range_checkbox.setChecked(opts.get('link_depth_to_range', False))
                    self.link_depth_to_range_checkbox.blockSignals(False)
        except Exception as e:
            self.logger.warning(f"Error applying GUI settings: {e}")
    
    def _log_and_save(self, param_name: str, value):
        """Logs parameter change and calls _save_gui_settings."""
        self.logger.info(f"Parameter changed: {param_name} = {value}, calling _save_gui_settings")
        self._save_gui_settings()
    
    def _save_gui_settings(self):
        """Saves current GUI settings to gui.json file."""
        # Log all calls for debugging
        import traceback
        stack = traceback.extract_stack()
        caller = stack[-2].name if len(stack) > 1 else "unknown"
        caller_line = stack[-2].lineno if len(stack) > 1 else 0
        _loading_settings = getattr(self, '_loading_settings', False)
        _gui_settings_loaded = getattr(self, '_gui_settings_loaded', False)
        
        # Check if signals are blocked (for debugging)
        signals_blocked = {}
        if hasattr(self, 'f_start_spin'):
            signals_blocked['f_start'] = self.f_start_spin.signalsBlocked()
            signals_blocked['D_target'] = self.D_target_spin.signalsBlocked() if hasattr(self, 'D_target_spin') else False
        
        self.logger.info(f"_save_gui_settings called from {caller}:{caller_line}, _loading_settings={_loading_settings}, _gui_settings_loaded={_gui_settings_loaded}, signals_blocked={signals_blocked}")
        
        if not hasattr(self, 'f_start_spin'):
            # UI not created yet, skip saving
            self.logger.info("_save_gui_settings: UI not created yet, skipping")
            return
        
        # Don't save during loading to prevent overwriting loaded values
        if _loading_settings:
            self.logger.info(f"_save_gui_settings: Skipping save - currently loading settings (_loading_settings={_loading_settings})")
            return
        
        # Don't save before GUI settings are loaded (during initialization)
        # This prevents saving default values before loading from gui.json
        if not _gui_settings_loaded:
            self.logger.info(f"_save_gui_settings: Skipping save - GUI settings not loaded yet (_gui_settings_loaded={_gui_settings_loaded})")
            return
        
        try:
            self.logger.info("_save_gui_settings: Saving GUI settings to gui.json")
            settings = {
                'signal': {
                    'f_start': self.f_start_spin.value(),
                    'f_end': self.f_end_spin.value(),
                    'Tp': self.Tp_spin.value(),
                    'window': self.window_combo.currentText(),
                    'sample_rate': self.sample_rate_spin.value()
                },
                'environment': {
                    'T': self.T_spin.value(),
                    'S': self.S_spin.value(),
                    'z': self.z_spin.value(),
                    'bottom_reflection': self.bottom_reflection_spin.value()
                },
                'range': {
                    'D_min': self.D_min_spin.value(),
                    'D_max': self.D_max_spin.value(),
                    # Validate D_target before saving
                    'D_target': max(self.D_min_spin.value(), 
                                  min(self.D_target_spin.value(), 
                                      self.D_max_spin.value()))
                },
                'target': {
                    'target_snr': self.target_snr_spin.value(),
                    'target_sigma': self.target_sigma_spin.value()
                },
                'hardware': {
                    'transducer_id': self.transducer_combo.currentText() if hasattr(self, 'transducer_combo') else '',
                    'lna_id': self.lna_combo.currentText() if hasattr(self, 'lna_combo') else '',
                    'vga_id': self.vga_combo.currentText() if hasattr(self, 'vga_combo') else '',
                    'adc_id': self.adc_combo.currentText() if hasattr(self, 'adc_combo') else ''
                },
                'options': {
                    'link_depth_to_range': self.link_depth_to_range_checkbox.isChecked() if hasattr(self, 'link_depth_to_range_checkbox') else False
                }
            }
            
            settings_path = self._get_gui_settings_path()
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved GUI settings to {settings_path}")
            self.logger.debug(f"Saved settings: signal={settings.get('signal', {})}, "
                           f"environment={settings.get('environment', {})}, "
                           f"range={settings.get('range', {})}, "
                           f"hardware={settings.get('hardware', {})}")
        except Exception as e:
            self.logger.error(f"Error saving GUI settings: {e}", exc_info=True)
    
    def _load_data(self):
        """Loads data from DATA module."""
        try:
            # Don't update D_target here - it will be loaded from gui.json
            # If not in gui.json, default value (100.0) is already set in UI creation
            
            # Block signals during data loading to prevent premature saves
            self.transducer_combo.blockSignals(True)
            self.lna_combo.blockSignals(True)
            self.vga_combo.blockSignals(True)
            self.adc_combo.blockSignals(True)
            
            # Load lists
            transducers = self.data_provider.list_transducers()
            if not transducers:
                transducers = ['example_transducer']
            self.transducer_combo.addItems(transducers)
            
            lna_list = self.data_provider.list_lna()
            if not lna_list:
                lna_list = ['example_lna']
            self.lna_combo.addItems(lna_list)
            
            vga_list = self.data_provider.list_vga()
            if not vga_list:
                vga_list = ['example_vga']
            self.vga_combo.addItems(vga_list)
            
            adc_list = self.data_provider.list_adc()
            if not adc_list:
                adc_list = ['example_adc']
            self.adc_combo.addItems(adc_list)
            
            # Unblock signals after loading
            self.transducer_combo.blockSignals(False)
            self.lna_combo.blockSignals(False)
            self.vga_combo.blockSignals(False)
            self.adc_combo.blockSignals(False)
            
            # Don't initialize sample_rate from first ADC here
            # It will be loaded from gui.json or use default value
            # This prevents overwriting saved settings
            
            # Don't auto-select first transducer here - let _load_gui_settings do it
            # This allows saved settings to override default selection
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load data: {e}")
    
    def _on_transducer_changed(self, transducer_id: str):
        """Handler for transducer change - adjusts CHIRP parameters."""
        self.logger.info(f"_on_transducer_changed: transducer_id={transducer_id}, signalsBlocked={self.transducer_combo.signalsBlocked()}")
        if not transducer_id:
            return
        
        try:
            # Load transducer parameters
            transducer_data = self.data_provider.get_transducer(transducer_id)
            
            f_min = transducer_data.get('f_min', 150000)
            f_max = transducer_data.get('f_max', 250000)
            
            # Update frequency ranges
            self.f_start_spin.blockSignals(True)
            self.f_end_spin.blockSignals(True)
            
            self.f_start_spin.setRange(f_min, f_max)
            self.f_end_spin.setRange(f_min, f_max)
            
            # If frequencies are out of range, adjust automatically
            current_f_start = self.f_start_spin.value()
            current_f_end = self.f_end_spin.value()
            
            if current_f_start < f_min or current_f_start > f_max or \
               current_f_end < f_min or current_f_end > f_max:
                # Use CORE to adjust parameters
                f_start, f_end = self.signal_calculator.adjust_chirp_to_transducer(
                    transducer_data, bandwidth_ratio=0.8
                )
                self.f_start_spin.setValue(f_start)
                self.f_end_spin.setValue(f_end)
            
            self.f_start_spin.blockSignals(False)
            self.f_end_spin.blockSignals(False)
            
            # Recalculate minimum signal duration
            self._update_min_tp()
            
            # Update Tp labels (will be updated in _update_min_tp, but ensure it's called)
            self._update_tp_labels()
            
            # Always update accuracy information (this will also update recommended frequencies)
            # This must be done even during loading to show correct recommended frequencies
            self._update_accuracy_info()
            
            # Don't adjust frequencies if we're loading settings - they should come from gui.json
            if not self._loading_settings:
                # Initialize frequencies with correct values based on checkbox state
                # This ensures f_start and f_end (user-defined) are set correctly
                if self.use_recommended_freq_checkbox.isChecked():
                    # Use recommended frequencies
                    self._apply_recommended_frequencies()
                else:
                    # Use user-defined frequencies (calculated from target accuracy)
                    self._apply_user_frequencies()
            
            # Always update signal path (but don't overwrite loaded values)
            if not self._loading_settings:
                self._update_signal_path()
            
        except Exception as e:
            self.logger.warning(f"Error adjusting transducer parameters: {e}")
    
    def _on_chirp_frequency_changed(self):
        """Handler for CHIRP frequency change - recalculates accuracy and updates sample rate minimum."""
        f_start = self.f_start_spin.value()
        f_end = self.f_end_spin.value()
        self.logger.info(f"_on_chirp_frequency_changed: f_start={f_start}, f_end={f_end}, signalsBlocked(f_start)={self.f_start_spin.signalsBlocked()}, signalsBlocked(f_end)={self.f_end_spin.signalsBlocked()}")
        # If using recommended frequencies, don't allow manual changes
        if self.use_recommended_freq_checkbox.isChecked():
            # Revert to recommended values from manufacturer data
            self._apply_recommended_frequencies()
        else:
            # Update sample rate minimum based on f_end (Nyquist: f_s >= 2*f_end)
            f_end = self.f_end_spin.value()
            min_sample_rate = 2 * f_end
            current_sample_rate = self.sample_rate_spin.value()
            
            # Update minimum sample rate
            self.sample_rate_spin.setMinimum(min_sample_rate)
            
            # If current sample rate is below minimum, adjust it
            if current_sample_rate < min_sample_rate:
                self.sample_rate_spin.blockSignals(True)
                self.sample_rate_spin.setValue(min_sample_rate)
                self.sample_rate_spin.blockSignals(False)
            
            # User can manually change frequencies, but we can also recalculate from target accuracy
            # For now, just update accuracy info
            self._update_accuracy_info()
    
    def _on_target_accuracy_changed(self):
        """Handler for target accuracy change - recalculates user frequencies."""
        target_sigma = self.target_sigma_spin.value()
        self.logger.info(f"_on_target_accuracy_changed: target_sigma={target_sigma}, signalsBlocked={self.target_sigma_spin.signalsBlocked()}")
        self._update_accuracy_info()
        # If using user-defined frequencies, update them based on target accuracy
        if not self.use_recommended_freq_checkbox.isChecked():
            self._apply_user_frequencies()
    
    def _on_absorption_only_changed(self):
        """Handler for absorption_only checkbox change."""
        # If simulation was already run, re-run it with new setting
        if hasattr(self, 'simulation_history') and len(self.simulation_history) > 0:
            # Get last simulation input
            last_input, _ = self.simulation_history[-1]
            # Re-run simulation with new absorption_only setting
            absorption_only = self.absorption_only_checkbox.isChecked()
            self.sim_thread = SimulationThread(self.simulator, last_input, absorption_only=absorption_only)
            self.sim_thread.finished.connect(self._on_simulation_finished)
            self.sim_thread.start()
    
    def _on_frequency_mode_changed(self):
        """Handler for frequency mode change (recommended vs user-defined)."""
        if self.use_recommended_freq_checkbox.isChecked():
            # Switch to recommended frequencies (from manufacturer data)
            self._apply_recommended_frequencies()
            # Disable user-defined spinboxes
            self.f_start_spin.setEnabled(False)
            self.f_end_spin.setEnabled(False)
        else:
            # Switch to user-defined frequencies (calculated from target accuracy)
            self._apply_user_frequencies()
            # Enable user-defined spinboxes
            self.f_start_spin.setEnabled(True)
            self.f_end_spin.setEnabled(True)
            self._update_accuracy_info()
    
    def _apply_user_frequencies(self):
        """Applies user-defined frequencies calculated from target accuracy to the spinboxes."""
        try:
            transducer_id = self.transducer_combo.currentText()
            if not transducer_id:
                return
            
            transducer_data = self.data_provider.get_transducer(transducer_id)
            target_sigma = self.target_sigma_spin.value()
            T = self.T_spin.value()
            S = self.S_spin.value()
            z = self.z_spin.value()
            
            # Calculate user frequencies from target accuracy
            Tp = self.Tp_spin.value()
            f_start_user, f_end_user = self.signal_calculator.calculate_user_frequencies_from_accuracy(
                transducer_data, target_sigma, T, S, z, Tp=Tp
            )
            
            # Update spinboxes (block signals to avoid recursion)
            self.f_start_spin.blockSignals(True)
            self.f_end_spin.blockSignals(True)
            self.f_start_spin.setValue(f_start_user)
            self.f_end_spin.setValue(f_end_user)
            self.f_start_spin.blockSignals(False)
            self.f_end_spin.blockSignals(False)
            
            # Update accuracy info
            self._update_accuracy_info()
        except Exception as e:
            self.logger.warning(f"Error applying user frequencies: {e}")
    
    def _apply_recommended_frequencies(self):
        """Applies recommended frequencies from transducer data to the spinboxes."""
        try:
            transducer_id = self.transducer_combo.currentText()
            if not transducer_id:
                return
            
            transducer_data = self.data_provider.get_transducer(transducer_id)
            
            # Get recommended frequencies from manufacturer data
            f_start_rec, f_end_rec = self.signal_calculator.get_recommended_frequencies_from_transducer(
                transducer_data
            )
            
            # Update spinboxes (block signals to avoid recursion)
            self.f_start_spin.blockSignals(True)
            self.f_end_spin.blockSignals(True)
            self.f_start_spin.setValue(f_start_rec)
            self.f_end_spin.setValue(f_end_rec)
            self.f_start_spin.blockSignals(False)
            self.f_end_spin.blockSignals(False)
            
            # Update accuracy info
            self._update_accuracy_info()
        except Exception as e:
            self.logger.warning(f"Error applying recommended frequencies: {e}")
    
    def _update_accuracy_info(self):
        """Updates accuracy and bandwidth information."""
        try:
            transducer_id = self.transducer_combo.currentText()
            if not transducer_id:
                return
            
            # Load transducer parameters
            transducer_data = self.data_provider.get_transducer(transducer_id)
            
            # Get transducer information from Core
            transducer_info = self.signal_calculator.get_transducer_info(transducer_data)
            
            # Display transducer bandwidth
            if transducer_info['bandwidth_estimated']:
                self.bandwidth_label.setText(f"{transducer_info['bandwidth_khz']:.1f} kHz (estimated)")
            else:
                self.bandwidth_label.setText(f"{transducer_info['bandwidth_khz']:.1f} kHz ({transducer_info['bandwidth']:.0f} Hz)")
            
            # Display transducer information
            info_text = f"f₀={transducer_info['f_0_khz']:.1f} kHz, range: {transducer_info['f_min_khz']:.1f}-{transducer_info['f_max_khz']:.1f} kHz"
            self.transducer_info_label.setText(info_text)
            
            # Get environment parameters
            T = self.T_spin.value()
            S = self.S_spin.value()
            z = self.z_spin.value()
            Tp = self.Tp_spin.value()
            
            # === RECOMMENDED FREQUENCIES (from manufacturer data) ===
            # Get recommended frequencies from transducer manufacturer data
            f_start_rec, f_end_rec = self.signal_calculator.get_recommended_frequencies_from_transducer(
                transducer_data
            )
            self.f_start_recommended_label.setText(f"{f_start_rec/1000:.1f} kHz ({f_start_rec:.0f} Hz)")
            self.f_end_recommended_label.setText(f"{f_end_rec/1000:.1f} kHz ({f_end_rec:.0f} Hz)")
            
            # Calculate recommended accuracy from recommended frequencies
            recommended_sigma = self.signal_calculator.calculate_recommended_accuracy(
                transducer_data, f_start_rec, f_end_rec, Tp, T, S, z
            )
            self.recommended_accuracy_label.setText(
                f"{recommended_sigma:.4f} m"
            )
            
            # === USER FREQUENCIES (calculated from target accuracy) ===
            # Calculate user frequencies from target accuracy
            target_sigma = self.target_sigma_spin.value()
            f_start_user, f_end_user = self.signal_calculator.calculate_user_frequencies_from_accuracy(
                transducer_data, target_sigma, T, S, z, Tp=Tp
            )
            
            # Calculate required bandwidth for target accuracy
            required_bw = self.signal_calculator.calculate_required_bandwidth(
                target_sigma, T, S, z, Tp=Tp
            )
            
            # Get transducer bandwidth (B_tr) for comparison
            B_tr = transducer_data.get('B_tr', transducer_data.get('f_max', 250000) - transducer_data.get('f_min', 150000))
            
            if required_bw < float('inf'):
                self.required_bandwidth_label.setText(
                    f"{required_bw/1000:.1f} kHz ({required_bw:.0f} Hz)"
                )
                
                # Determine color based on comparison with transducer bandwidth:
                # RED: only if required_bw > B_tr (exceeds transducer capability)
                # GREEN: if required_bw <= B_tr (within transducer limits)
                if required_bw > B_tr:
                    self.required_bandwidth_label.setStyleSheet("color: red; font-weight: bold;")
                    self.required_bandwidth_label.setToolTip(
                        f"Required bandwidth ({required_bw/1000:.1f} kHz) exceeds transducer bandwidth ({B_tr/1000:.1f} kHz)!"
                    )
                else:
                    self.required_bandwidth_label.setStyleSheet("color: green;")
                    self.required_bandwidth_label.setToolTip(
                        f"Required bandwidth ({required_bw/1000:.1f} kHz) is within transducer limits ({B_tr/1000:.1f} kHz)"
                    )
            else:
                self.required_bandwidth_label.setText("—")
                self.required_bandwidth_label.setStyleSheet("")
                
        except Exception as e:
            self.logger.warning(f"Error updating accuracy information: {e}")
    
    def _on_d_min_changed(self, value: float):
        """Handler for minimum range change - recalculates Tp."""
        self.logger.info(f"_on_d_min_changed: D_min={value}, signalsBlocked={self.D_min_spin.signalsBlocked()}")
        # Ensure D_target is still within valid range
        D_target = self.D_target_spin.value()
        if D_target < value:
            # If D_target is now less than D_min, adjust it
            self.D_target_spin.blockSignals(True)
            self.D_target_spin.setValue(value)
            self.D_target_spin.blockSignals(False)
        
        self._update_min_tp()
        self._update_accuracy_info()
        self._update_signal_path()
    
    def _on_d_max_changed(self, value: float):
        """Handler for maximum range change - recalculates optimal Tp."""
        self.logger.info(f"_on_d_max_changed: D_max={value}, signalsBlocked={self.D_max_spin.signalsBlocked()}")
        try:
            # Ensure D_max > D_min
            D_min = self.D_min_spin.value()
            if value <= D_min:
                # If D_max <= D_min, adjust D_min first
                self.D_min_spin.blockSignals(True)
                self.D_min_spin.setValue(value - 0.1)  # Set D_min slightly less than D_max
                self.D_min_spin.blockSignals(False)
            
            # Ensure D_target is still within valid range
            D_target = self.D_target_spin.value()
            if D_target > value:
                # If D_target is now greater than D_max, adjust it
                self.D_target_spin.blockSignals(True)
                self.D_target_spin.setValue(value)
                self.D_target_spin.blockSignals(False)
            
            self._update_min_tp()
            self._update_accuracy_info()
            self._update_signal_path()
        except Exception as e:
            self.logger.warning(f"Error handling D_max change: {e}")
    
    def _on_d_target_changed(self, value: float):
        """Handler for target distance change."""
        self.logger.info(f"_on_d_target_changed called with value={value}, signalsBlocked={self.D_target_spin.signalsBlocked()}")
        # Ensure D_target is within D_min and D_max
        D_min = self.D_min_spin.value()
        D_max = self.D_max_spin.value()
        if value < D_min:
            self.logger.info(f"D_target ({value}) < D_min ({D_min}), adjusting to D_min")
            self.D_target_spin.setValue(D_min)
        elif value > D_max:
            self.logger.info(f"D_target ({value}) > D_max ({D_max}), adjusting to D_max")
            self.D_target_spin.setValue(D_max)
        else:
            # If depth is linked to target range, update depth
            if self.link_depth_to_range_checkbox.isChecked():
                self.z_spin.blockSignals(True)
                self.z_spin.setValue(value)
                self.z_spin.blockSignals(False)
            # Update signal path when D_target changes
            self._update_signal_path()
            # Update Tp_optimal label when D_target changes
            self._update_tp_labels()
    
    def _on_link_depth_to_range_changed(self, state: int):
        """Handler for link depth to range checkbox change."""
        checked = state == Qt.Checked
        self.logger.info(f"_on_link_depth_to_range_changed: link_depth_to_range={checked}, signalsBlocked={self.link_depth_to_range_checkbox.signalsBlocked()}")
        if state == Qt.Checked:
            # Link is enabled: set D_target = depth
            depth = self.z_spin.value()
            D_min = self.D_min_spin.value()
            D_max = self.D_max_spin.value()
            D_target = max(D_min, min(depth, D_max))
            self.D_target_spin.blockSignals(True)
            self.D_target_spin.setValue(D_target)
            self.D_target_spin.blockSignals(False)
            # Make D_target read-only when linked (optional - can be editable)
            # self.D_target_spin.setEnabled(False)
        else:
            # Link is disabled: allow independent editing
            # self.D_target_spin.setEnabled(True)
            pass
    
    def _update_d_target(self):
        """Updates target distance to average of D_min and D_max."""
        try:
            D_min = self.D_min_spin.value()
            D_max = self.D_max_spin.value()
            
            # Ensure D_max > D_min
            if D_max <= D_min:
                return  # Skip update if invalid range
            
            D_target = (D_min + D_max) / 2
            # Ensure D_target is within valid range
            D_target = max(D_min, min(D_target, D_max))
            
            # Block signals to avoid recursion
            self.D_target_spin.blockSignals(True)
            self.D_target_spin.setValue(D_target)
            self.D_target_spin.blockSignals(False)
        except Exception as e:
            self.logger.warning(f"Error updating D_target: {e}")
    
    def _calculate_optimal_tp(self):
        """Calculates optimal pulse duration considering distance, SNR, and attenuation."""
        try:
            from core.signal_calculator import SignalCalculator
            from data.data_provider import DataProvider
            
            # Get current parameters
            D_target = self.D_target_spin.value()
            target_snr = self.target_snr_spin.value()
            
            # Log for debugging
            self.logger.info(f"_calculate_optimal_tp: D_target={D_target:.1f}m, target_snr={target_snr:.1f}dB")
            
            # Get hardware parameters
            transducer_id = self.transducer_combo.currentText()
            lna_id = self.lna_combo.currentText()
            vga_id = self.vga_combo.currentText()
            
            if not transducer_id or not lna_id or not vga_id:
                QMessageBox.warning(self, "Error", "Please select all hardware components first.")
                return
            
            # Get parameters from data provider
            transducer_params = self.data_provider.get_transducer(transducer_id)
            lna_params = self.data_provider.get_lna(lna_id)
            vga_params = self.data_provider.get_vga(vga_id)
            
            # Get signal parameters
            f_start = self.f_start_spin.value()
            f_end = self.f_end_spin.value()
            
            # Get environment parameters
            T = self.T_spin.value()
            S = self.S_spin.value()
            z = self.z_spin.value()
            bottom_reflection = self.bottom_reflection_spin.value()
            
            # Get TX voltage (from transducer or default)
            tx_voltage = transducer_params.get('V_max', transducer_params.get('V_nominal', 100.0))
            
            # Prepare hardware params
            hardware_params = {
                'lna_gain': lna_params.get('G_LNA', lna_params.get('G', 20)),
                'vga_gain': vga_params.get('G', 30),
                'lna_nf': lna_params.get('NF_LNA', lna_params.get('NF', 2.0)),
                'bottom_reflection': bottom_reflection
            }
            
            # Get current Tp
            current_tp = self.Tp_spin.value()
            
            # Check if we have results from last simulation
            # If current SNR is above target, we need to REDUCE Tp
            # If current SNR is below target or unknown, we calculate minimum Tp needed
            current_snr = None
            if hasattr(self, 'last_output_dto') and self.last_output_dto:
                current_snr = self.last_output_dto.SNR_ADC
            
            calculator = SignalCalculator()
            
            # If we have current SNR and it's above target, use reduction formula
            if current_snr is not None and current_snr > target_snr * 1.05:  # 5% margin
                # SNR scales with 10*log10(Tp), so to reduce SNR by X dB, reduce Tp by factor 10^(-X/10)
                # Formula: Tp_new = Tp_current * 10^((target_snr - SNR_current) / 10)
                snr_reduction_needed = current_snr - target_snr
                tp_reduction_factor = 10 ** (-snr_reduction_needed / 10.0)
                optimal_tp = current_tp * tp_reduction_factor
                
                # Check physical constraint: Tp must not exceed 80% of round-trip time at D_target
                # NOTE: Use D_target, not D_min, for physical constraint in this calculation
                Tp_max_us = calculator.calculate_optimal_pulse_duration(D_target, T, S, z, min_tp=None)
                optimal_tp = min(optimal_tp, Tp_max_us)
                
                # Ensure minimum 1 µs
                optimal_tp = max(1.0, optimal_tp)
                
                calculation_method = "Reduction from current Tp (SNR above target)"
            else:
                # Calculate minimum Tp needed to achieve target SNR (from scratch)
                optimal_tp = calculator.calculate_optimal_tp_for_snr(
                    D_target=D_target,
                    target_snr_db=target_snr,
                    transducer_params=transducer_params,
                    hardware_params=hardware_params,
                    T=T,
                    S=S,
                    z=z,
                    f_start=f_start,
                    f_end=f_end,
                    tx_voltage=tx_voltage
                )
                calculation_method = "Minimum Tp to achieve target SNR"
            
            # Update Tp spinbox
            self.Tp_spin.setValue(optimal_tp)
            
            # Calculate Tp_optimal for comparison (80% of TOF for D_target)
            Tp_optimal_for_target = calculator.calculate_optimal_pulse_duration(D_target, T, S, z, min_tp=None)
            
            # Show message
            message = f"Optimal pulse duration: {optimal_tp:.1f} µs\n\n"
            message += f"Calculated for:\n"
            message += f"  • Distance: {D_target:.1f} m\n"
            message += f"  • Target SNR: {target_snr:.1f} dB\n"
            if current_snr is not None:
                message += f"  • Current SNR: {current_snr:.2f} dB\n"
            message += f"  • Current Tp: {current_tp:.1f} µs\n"
            message += f"  • Method: {calculation_method}\n"
            message += f"\nNote:\n"
            message += f"  • This is the MINIMUM Tp to achieve target SNR\n"
            message += f"  • Tp_optimal (for D_target, 80% of TOF): {Tp_optimal_for_target:.1f} µs\n"
            message += f"  • These values differ because:\n"
            message += f"    - Calculate Optimal Tp: minimum Tp for target SNR\n"
            message += f"    - Tp_optimal label: maximum Tp (80% of round-trip time)\n"
            message += f"\n  • Attenuation: considered (spreading + absorption)\n"
            message += f"  • Bottom reflection: {bottom_reflection:.1f} dB"
            
            QMessageBox.information(
                self,
                "Optimal Tp Calculated",
                message
            )
            
            # Update Tp labels after calculation
            self._update_tp_labels()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate optimal Tp:\n{str(e)}")
            self.logger.error(f"Error calculating optimal Tp: {e}", exc_info=True)
    
    def _update_min_tp(self):
        """Updates maximum allowed signal duration based on D_min."""
        try:
            D_min = self.D_min_spin.value()
            D_max = self.D_max_spin.value()
            T = self.T_spin.value()
            S = self.S_spin.value()
            z = self.z_spin.value()
            
            # Tp_min: Minimum pulse duration for minimum distance (D_min)
            # This is the maximum allowed Tp based on physical constraint:
            # Tp cannot exceed 80% of round-trip time at D_min to allow signal reception
            # Formula: Tp_max = 0.8 * (2 * D_min / c) where c is sound speed
            Tp_min_us = self.signal_calculator.calculate_min_pulse_duration(D_min, T, S, z)
            
            # Tp_optimal: Optimal pulse duration for target distance (D_target)
            # This is calculated for the target distance to achieve good SNR
            # It can be larger than Tp_min if D_target > D_min
            D_target = self.D_target_spin.value() if hasattr(self, 'D_target_spin') else (D_min + D_max) / 2
            Tp_optimal_us = self.signal_calculator.calculate_optimal_pulse_duration(
                D_target, T, S, z, min_tp=None
            )
            
            # Use CORE to calculate recommended Tp with all constraints
            # This returns the optimal Tp considering both D_min and D_max constraints
            # NOTE: This is the minimum duration for minimum distance, not for target distance
            Tp_recommended_us = self.signal_calculator.calculate_optimal_tp_with_constraints(
                D_min, D_max, T, S, z
            )
            
            # Calculate maximum allowed Tp from D_min (for spinbox maximum)
            # This is the physical constraint limit
            Tp_max_us = Tp_min_us
            
            # Save old maximum value and current Tp
            old_max = self.Tp_spin.maximum()
            current_tp = self.Tp_spin.value()
            
            # Update maximum spinbox value (cannot be greater than Tp_max)
            self.Tp_spin.setMaximum(Tp_max_us)
            
            # Update Tp if:
            # 1. Current value is greater than new maximum (must update)
            # 2. Or maximum changed (D_min changed) - always update to optimal
            # 3. Or optimal value changed (D_max changed) - update if current was close to optimal
            should_update = False
            if current_tp > Tp_max_us:
                # Must update if greater than maximum
                should_update = True
                Tp_recommended_us = Tp_max_us
            elif abs(old_max - Tp_max_us) > 1.0:
                # Maximum changed (D_min changed) - always update
                should_update = True
            elif abs(current_tp - Tp_recommended_us) < 10.0:
                # Current value is close to optimal - update when optimal changes
                should_update = True
            
            # Update labels for Tp_min and Tp_optimal
            self._update_tp_labels(Tp_min_us, Tp_optimal_us, D_min, D_target)
            
            if should_update:
                self.Tp_spin.blockSignals(True)
                self.Tp_spin.setValue(Tp_recommended_us)
                self.Tp_spin.blockSignals(False)
                self.logger.info(f"_update_min_tp: Tp_min={Tp_min_us:.1f} µs (for D_min={D_min}m), Tp_optimal={Tp_optimal_us:.1f} µs (for D_target={D_target:.1f}m), Tp_recommended={Tp_recommended_us:.1f} µs")
            
        except Exception as e:
            self.logger.warning(f"Error calculating Tp duration: {e}")
    
    def _update_tp_labels(self, Tp_min_us=None, Tp_optimal_us=None, D_min=None, D_target=None):
        """Updates Tp_min and Tp_optimal labels."""
        try:
            # Check if labels exist
            if not hasattr(self, 'Tp_min_label') or not hasattr(self, 'Tp_optimal_label'):
                return
            
            if Tp_min_us is None or D_min is None:
                D_min = self.D_min_spin.value()
                T = self.T_spin.value()
                S = self.S_spin.value()
                z = self.z_spin.value()
                Tp_min_us = self.signal_calculator.calculate_min_pulse_duration(D_min, T, S, z)
            
            if Tp_optimal_us is None or D_target is None:
                # Get D_target - use D_target_spin if available, otherwise use average of D_min and D_max
                if hasattr(self, 'D_target_spin') and self.D_target_spin is not None:
                    D_target = self.D_target_spin.value()
                else:
                    D_min_val = self.D_min_spin.value() if hasattr(self, 'D_min_spin') else 0.5
                    D_max_val = self.D_max_spin.value() if hasattr(self, 'D_max_spin') else 500.0
                    D_target = (D_min_val + D_max_val) / 2
                
                # Tp_optimal should show the same value as "Calculate Optimal Tp"
                # This is the minimum Tp needed to achieve target SNR (using calculate_optimal_tp_for_snr)
                try:
                    # Get hardware parameters
                    transducer_id = self.transducer_combo.currentText()
                    lna_id = self.lna_combo.currentText()
                    vga_id = self.vga_combo.currentText()
                    target_snr = self.target_snr_spin.value()
                    
                    if transducer_id and lna_id and vga_id:
                        transducer_params = self.data_provider.get_transducer(transducer_id)
                        lna_params = self.data_provider.get_lna(lna_id)
                        vga_params = self.data_provider.get_vga(vga_id)
                        
                        # Get signal parameters
                        f_start = self.f_start_spin.value()
                        f_end = self.f_end_spin.value()
                        
                        # Get environment parameters
                        T = self.T_spin.value()
                        S = self.S_spin.value()
                        z = self.z_spin.value()
                        bottom_reflection = self.bottom_reflection_spin.value()
                        
                        # Get TX voltage
                        tx_voltage = transducer_params.get('V_max', transducer_params.get('V_nominal', 100.0))
                        
                        # Prepare hardware params
                        # Use VGA gain from signal_path_widget if available (user may have adjusted it)
                        # Otherwise use from VGA params
                        if hasattr(self, 'signal_path_widget') and self.signal_path_widget:
                            try:
                                vga_gain = self.signal_path_widget.get_vga_gain()
                            except Exception:
                                vga_gain = vga_params.get('G', 30)
                        else:
                            vga_gain = vga_params.get('G', 30)
                        
                        hardware_params = {
                            'lna_gain': lna_params.get('G_LNA', lna_params.get('G', 20)),
                            'vga_gain': vga_gain,
                            'lna_nf': lna_params.get('NF_LNA', lna_params.get('NF', 2.0)),
                            'bottom_reflection': bottom_reflection
                        }
                        
                        # Calculate optimal Tp using the same method as "Calculate Optimal Tp"
                        Tp_optimal_us = self.signal_calculator.calculate_optimal_tp_for_snr(
                            D_target=D_target,
                            target_snr_db=target_snr,
                            transducer_params=transducer_params,
                            hardware_params=hardware_params,
                            T=T,
                            S=S,
                            z=z,
                            f_start=f_start,
                            f_end=f_end,
                            tx_voltage=tx_voltage
                        )
                    else:
                        # Fallback: use calculate_optimal_pulse_duration if hardware not selected
                        # This returns 80% of TOF for D_target, not optimal for SNR
                        T = self.T_spin.value()
                        S = self.S_spin.value()
                        z = self.z_spin.value()
                        Tp_optimal_us = self.signal_calculator.calculate_optimal_pulse_duration(
                            D_target, T, S, z, min_tp=None
                        )
                        self.logger.warning(f"_update_tp_labels: Hardware not fully selected, using fallback calculation (80% TOF): {Tp_optimal_us:.1f} µs")
                except Exception as e:
                    # Fallback: use calculate_optimal_pulse_duration if calculation fails
                    # This returns 80% of TOF for D_target, not optimal for SNR
                    self.logger.warning(f"Error calculating Tp_optimal with calculate_optimal_tp_for_snr: {e}, using fallback", exc_info=True)
                    T = self.T_spin.value()
                    S = self.S_spin.value()
                    z = self.z_spin.value()
                    Tp_optimal_us = self.signal_calculator.calculate_optimal_pulse_duration(
                        D_target, T, S, z, min_tp=None
                    )
                    self.logger.warning(f"_update_tp_labels: Using fallback calculation (80% TOF): {Tp_optimal_us:.1f} µs")
            
            # Update labels
            self.Tp_min_label.setText(f"{Tp_min_us:.1f} µs (D_min={D_min:.1f} m)")
            self.Tp_optimal_label.setText(f"{Tp_optimal_us:.1f} µs (D_target={D_target:.1f} m, for target SNR)")
            
            self.logger.info(f"_update_tp_labels: Tp_min={Tp_min_us:.1f} µs (D_min={D_min:.1f}m), Tp_optimal={Tp_optimal_us:.1f} µs (D_target={D_target:.1f}m, for target SNR)")
        except Exception as e:
            self.logger.warning(f"Error updating Tp labels: {e}", exc_info=True)
    
    def _on_tp_changed(self):
        """Handler for pulse duration change - recalculates accuracy."""
        Tp = self.Tp_spin.value()
        self.logger.info(f"_on_tp_changed: Tp={Tp}, signalsBlocked={self.Tp_spin.signalsBlocked()}")
        self._update_accuracy_info()
    
    def _on_sample_rate_changed(self, value: float):
        """Handler for sample rate change - validates against f_end."""
        self.logger.info(f"_on_sample_rate_changed: sample_rate={value}, signalsBlocked={self.sample_rate_spin.signalsBlocked()}")
        f_end = self.f_end_spin.value()
        min_sample_rate = 2 * f_end
        if value < min_sample_rate:
            # Auto-adjust to minimum required
            self.sample_rate_spin.blockSignals(True)
            self.sample_rate_spin.setValue(min_sample_rate)
            self.sample_rate_spin.blockSignals(False)
            QMessageBox.warning(self, "Warning", 
                             f"Sample rate must be >= 2*f_end ({min_sample_rate:.0f} Hz). "
                             f"Adjusted to {min_sample_rate:.0f} Hz.")
    
    def _on_environment_changed(self):
        """Handler for environment parameters change - recalculates Tp and accuracy."""
        T = self.T_spin.value()
        S = self.S_spin.value()
        z = self.z_spin.value()
        bottom_reflection = self.bottom_reflection_spin.value()
        self.logger.info(f"_on_environment_changed: T={T}, S={S}, z={z}, bottom_reflection={bottom_reflection}, "
                        f"signalsBlocked(T)={self.T_spin.signalsBlocked()}, signalsBlocked(S)={self.S_spin.signalsBlocked()}, "
                        f"signalsBlocked(z)={self.z_spin.signalsBlocked()}, signalsBlocked(bottom_reflection)={self.bottom_reflection_spin.signalsBlocked()}")
        # If depth is linked to target range, update D_target
        if self.link_depth_to_range_checkbox.isChecked():
            depth = self.z_spin.value()
            # Ensure D_target is within valid range
            D_min = self.D_min_spin.value()
            D_max = self.D_max_spin.value()
            D_target = max(D_min, min(depth, D_max))
            self.D_target_spin.blockSignals(True)
            self.D_target_spin.setValue(D_target)
            self.D_target_spin.blockSignals(False)
        
        self._update_min_tp()
        self._update_accuracy_info()
        # If using recommended frequencies, update them
        if self.use_recommended_freq_checkbox.isChecked():
            self._apply_recommended_frequencies()
        self._update_signal_path()
        # Update Tp labels when environment changes
        self._update_tp_labels()
    
    def _get_input_dto(self) -> Optional[InputDTO]:
        """Creates InputDTO from UI parameters."""
        try:
            from core.dto import HardwareDTO, SignalDTO, EnvironmentDTO, RangeDTO
            
            # Get IDs from combo boxes, check if they are not empty
            transducer_id = self.transducer_combo.currentText()
            lna_id = self.lna_combo.currentText()
            vga_id = self.vga_combo.currentText()
            adc_id = self.adc_combo.currentText()
            
            # Validate that all IDs are set
            if not transducer_id or not lna_id or not vga_id or not adc_id:
                return None
            
            hardware = HardwareDTO(
                transducer_id=transducer_id,
                lna_id=lna_id,
                vga_id=vga_id,
                adc_id=adc_id
            )
            
            signal = SignalDTO(
                f_start=self.f_start_spin.value(),
                f_end=self.f_end_spin.value(),
                Tp=self.Tp_spin.value(),
                window=self.window_combo.currentText(),
                sample_rate=self.sample_rate_spin.value()
            )
            
            environment = EnvironmentDTO(
                T=self.T_spin.value(),
                S=self.S_spin.value(),
                z=self.z_spin.value(),
                bottom_reflection=self.bottom_reflection_spin.value()
            )
            
            # Get range values and validate before creating RangeDTO
            D_min = self.D_min_spin.value()
            D_max = self.D_max_spin.value()
            D_target = self.D_target_spin.value()
            
            # Log current values for debugging
            self.logger.info(f"Creating InputDTO: D_min={D_min}, D_max={D_max}, D_target={D_target}")
            
            # Ensure D_max > D_min (fix if invalid)
            if D_max <= D_min:
                # Adjust D_min to be slightly less than D_max
                old_D_min = D_min
                D_min = max(0.1, D_max - 0.1)
                self.D_min_spin.blockSignals(True)
                self.D_min_spin.setValue(D_min)
                self.D_min_spin.blockSignals(False)
                self.logger.warning(f"D_max ({D_max}) <= D_min ({old_D_min}). Adjusted D_min to {D_min}")
            
            # Ensure D_target is within valid range [D_min, D_max]
            D_target_original = D_target
            if D_target < D_min:
                D_target = D_min
                self.logger.warning(f"D_target ({D_target_original}) < D_min ({D_min}). Adjusted to {D_min}")
                self.D_target_spin.blockSignals(True)
                self.D_target_spin.setValue(D_target)
                self.D_target_spin.blockSignals(False)
            elif D_target > D_max:
                D_target = D_max
                self.logger.warning(f"D_target ({D_target_original}) > D_max ({D_max}). Adjusted to {D_max}")
                self.D_target_spin.blockSignals(True)
                self.D_target_spin.setValue(D_target)
                self.D_target_spin.blockSignals(False)
            
            # Log final values
            if D_target_original != D_target:
                self.logger.info(f"D_target adjusted: {D_target_original} -> {D_target}")
            
            # Ensure D_target is within valid range
            if D_target is not None:
                if D_target < D_min:
                    D_target = D_min
                    self.D_target_spin.blockSignals(True)
                    self.D_target_spin.setValue(D_target)
                    self.D_target_spin.blockSignals(False)
                elif D_target > D_max:
                    D_target = D_max
                    self.D_target_spin.blockSignals(True)
                    self.D_target_spin.setValue(D_target)
                    self.D_target_spin.blockSignals(False)
            
            # Log final values before creating RangeDTO
            self.logger.info(f"Creating RangeDTO: D_min={D_min}, D_max={D_max}, D_target={D_target}")
            
            range_dto = RangeDTO(
                D_min=D_min,
                D_max=D_max,
                D_target=D_target
            )
            
            # Get target_snr from GUI
            target_snr = self.target_snr_spin.value() if hasattr(self, 'target_snr_spin') else 20.0
            # Ensure value is within valid range [10, 40]
            if target_snr < 10:
                target_snr = 10.0
                self.logger.warning(f"Target SNR ({self.target_snr_spin.value() if hasattr(self, 'target_snr_spin') else 20.0}) < 10. Using 10.0")
            elif target_snr > 40:
                target_snr = 40.0
                self.logger.warning(f"Target SNR ({self.target_snr_spin.value() if hasattr(self, 'target_snr_spin') else 20.0}) > 40. Using 40.0")
            
            # Get optimization strategy
            optimization_strategy = self.optimization_strategy_combo.currentData() if hasattr(self, 'optimization_strategy_combo') else "max_tp_min_vga"
            
            return InputDTO(
                hardware=hardware,
                signal=signal,
                environment=environment,
                range=range_dto,
                target_snr=target_snr,
                optimization_strategy=optimization_strategy
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating input data: {e}")
            return None
    
    def _run_simulation(self):
        """Runs simulation."""
        input_dto = self._get_input_dto()
        if input_dto is None:
            return
        
        self.simulate_btn.setEnabled(False)
        self.simulate_btn.setText("Running...")
        
        # Get absorption_only setting from checkbox
        absorption_only = self.absorption_only_checkbox.isChecked()
        
        # Get VGA gain from signal_path_widget
        vga_gain = self.signal_path_widget.get_vga_gain() if hasattr(self, 'signal_path_widget') else None
        
        # Run in separate thread
        self.sim_thread = SimulationThread(self.simulator, input_dto, absorption_only=absorption_only, vga_gain=vga_gain)
        self.sim_thread.finished.connect(self._on_simulation_finished)
        self.sim_thread.start()
    
    def _on_simulation_finished(self, input_dto: InputDTO, output_dto: OutputDTO):
        """Handler for simulation completion."""
        self.simulate_btn.setEnabled(True)
        self.simulate_btn.setText("Run Simulation")
        
        # Save to history
        self.simulation_history.append((input_dto, output_dto))
        
        # Display results
        self._display_results(input_dto, output_dto)
        self._display_recommendations(output_dto)
        self._display_enob(output_dto)
        self._display_signals(input_dto, output_dto)
        self._plot_results(input_dto, output_dto)
        self._display_signal_path(input_dto)
    
    def _display_results(self, input_dto: InputDTO, output_dto: OutputDTO):
        """Displays results."""
        text = f"""
=== Simulation Results ===

Measured Range: {output_dto.D_measured:.4f} m
Standard Deviation: {output_dto.sigma_D:.6f} m
SNR ADC: {output_dto.SNR_ADC:.2f} dB
Clipping: {'Yes' if output_dto.clipping_flags else 'No'}
Success: {'Yes' if output_dto.success else 'No'}

=== Errors ===
"""
        if output_dto.errors:
            text += "\n".join(f"- {e}" for e in output_dto.errors)
        else:
            text += "No errors"
        
        text += "\n\n=== Warnings ===\n"
        if output_dto.warnings:
            text += "\n".join(f"- {w}" for w in output_dto.warnings)
        else:
            text += "No warnings"
        
        self.results_text.setPlainText(text)
    
    def _display_recommendations(self, output_dto: OutputDTO):
        """Displays recommendations."""
        rec = output_dto.recommendations
        
        # Log for debugging
        self.logger.info(f"_display_recommendations: rec={rec}, has_message={rec and rec.message and rec.message.strip()}, message_length={len(rec.message) if rec and rec.message else 0}")
        
        # Use message from optimizer if available (contains help text and optimized parameters)
        if rec and rec.message and rec.message.strip():
            # Optimizer message contains complete information: help text + recommendations + optimized parameters
            text = rec.message
            self.logger.info(f"_display_recommendations: Using optimizer message (length={len(text)})")
        else:
            # Fallback to old format if message is empty
            text = "=== Optimization Recommendations ===\n\n"
            
            recommendations_list = []
            if rec and rec.increase_Tp:
                recommendations_list.append("Increase Tp")
            if rec and rec.decrease_Tp:
                recommendations_list.append("Decrease Tp")
            if rec and rec.increase_f_start:
                recommendations_list.append("Increase f_start")
            if rec and rec.decrease_f_start:
                recommendations_list.append("Decrease f_start")
            if rec and rec.increase_f_end:
                recommendations_list.append("Increase f_end")
            if rec and rec.decrease_f_end:
                recommendations_list.append("Decrease f_end")
            if rec and rec.increase_G_VGA:
                recommendations_list.append("Increase VGA gain")
            if rec and rec.decrease_G_VGA:
                recommendations_list.append("Decrease VGA gain")
            if rec and rec.change_transducer:
                recommendations_list.append("Consider different transducer")
            if rec and rec.change_lna:
                recommendations_list.append("Consider different LNA")
            if rec and rec.change_adc:
                recommendations_list.append("Consider different ADC")
            
            if recommendations_list:
                text += "\n".join(f"- {r}" for r in recommendations_list)
            else:
                text += "No recommendations - all parameters are normal"
        
        self.recommendations_text.setPlainText(text)
        
        # Store last output_dto for applying recommendations
        self.last_output_dto = output_dto
    
    def _display_enob(self, output_dto: OutputDTO):
        """Displays ENOB calculation results."""
        if output_dto.enob_results is None:
            self.enob_text.setPlainText("ENOB calculation not available.\nRun simulation to calculate ENOB.")
            return
        
        # Format ENOB report using ENOBCalculator
        enob_calculator = ENOBCalculator()
        text = enob_calculator.format_enob_report(output_dto.enob_results)
        self.enob_text.setPlainText(text)
    
    def _apply_recommendations(self):
        """Applies optimizer recommendations to GUI parameters."""
        if not hasattr(self, 'last_output_dto') or not self.last_output_dto:
            QMessageBox.warning(self, "No Recommendations", "No recommendations available. Run simulation first.")
            return
        
        recommendations = self.last_output_dto.recommendations
        if not recommendations:
            QMessageBox.warning(self, "No Recommendations", "No recommendations available. Run simulation first.")
            return
        
        # Check if suggested_changes exists and is not empty
        if not hasattr(recommendations, 'suggested_changes') or not recommendations.suggested_changes:
            QMessageBox.information(self, "No Changes", "No parameter changes suggested. All parameters are optimal.")
            self.logger.info(f"_apply_recommendations: No suggested_changes. recommendations.decrease_Tp={getattr(recommendations, 'decrease_Tp', False)}, recommendations.increase_Tp={getattr(recommendations, 'increase_Tp', False)}")
            return
        
        # Apply suggested changes
        changes_applied = []
        
        if 'Tp' in recommendations.suggested_changes:
            suggested_tp = recommendations.suggested_changes['Tp']
            # Ensure within valid range
            suggested_tp = max(self.Tp_spin.minimum(), min(suggested_tp, self.Tp_spin.maximum()))
            self.Tp_spin.blockSignals(True)
            self.Tp_spin.setValue(suggested_tp)
            self.Tp_spin.blockSignals(False)
            changes_applied.append(f"Tp: {suggested_tp:.2f} µs")
        
        if 'f_start' in recommendations.suggested_changes:
            suggested_f_start = recommendations.suggested_changes['f_start']
            # Ensure within valid range
            suggested_f_start = max(self.f_start_spin.minimum(), min(suggested_f_start, self.f_start_spin.maximum()))
            self.f_start_spin.blockSignals(True)
            self.f_start_spin.setValue(suggested_f_start)
            self.f_start_spin.blockSignals(False)
            changes_applied.append(f"f_start: {suggested_f_start:.2f} Hz")
        
        if 'f_end' in recommendations.suggested_changes:
            suggested_f_end = recommendations.suggested_changes['f_end']
            # Ensure within valid range
            suggested_f_end = max(self.f_end_spin.minimum(), min(suggested_f_end, self.f_end_spin.maximum()))
            self.f_end_spin.blockSignals(True)
            self.f_end_spin.setValue(suggested_f_end)
            self.f_end_spin.blockSignals(False)
            changes_applied.append(f"f_end: {suggested_f_end:.2f} Hz")
        
        # VGA gain changes need to be applied manually (user adjusts in Signal Path widget)
        if recommendations.increase_G_VGA or recommendations.decrease_G_VGA:
            if recommendations.increase_G_VGA:
                changes_applied.append("VGA gain: Increase manually in Signal Path widget")
            else:
                changes_applied.append("VGA gain: Decrease manually in Signal Path widget")
        
        if changes_applied:
            message = "Applied recommendations:\n" + "\n".join(f"  • {c}" for c in changes_applied)
            QMessageBox.information(self, "Recommendations Applied", message)
            self.logger.info(f"Applied recommendations: {changes_applied}")
            
            # Update Tp labels after applying all changes
            # (Tp_optimal may depend on f_start, f_end, and other parameters)
            self._update_tp_labels()
            
            # Trigger save
            self._save_gui_settings()
        else:
            QMessageBox.information(self, "No Changes", "No applicable parameter changes.")
    
    def _plot_results(self, input_dto: InputDTO, output_dto: OutputDTO):
        """Plots graphs."""
        self.figure.clear()
        
        # Plot 1: Iteration history
        ax1 = self.figure.add_subplot(2, 2, 1)
        if len(self.simulation_history) > 1:
            iterations = range(1, len(self.simulation_history) + 1)
            sigma_D_values = [out.sigma_D for _, out in self.simulation_history]
            ax1.plot(iterations, sigma_D_values, 'o-')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('σ_D, m')
            ax1.set_title('σ_D History')
            ax1.grid(True)
        
        # Plot 2: SNR
        ax2 = self.figure.add_subplot(2, 2, 2)
        if len(self.simulation_history) > 1:
            snr_values = [out.SNR_ADC for _, out in self.simulation_history]
            ax2.plot(iterations, snr_values, 'o-')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('SNR, dB')
            ax2.set_title('SNR History')
            ax2.grid(True)
        
        # Plot 3: D_measured vs D_target
        ax3 = self.figure.add_subplot(2, 2, 3)
        D_target = self.signal_calculator.calculate_target_range(
            input_dto.range.D_min, input_dto.range.D_max
        )
        ax3.bar(['Target', 'Measured'], [D_target, output_dto.D_measured])
        ax3.set_ylabel('Range, m')
        ax3.set_title('Range Comparison')
        ax3.grid(True, axis='y')
        
        # Plot 4: Metrics
        ax4 = self.figure.add_subplot(2, 2, 4)
        metrics = ['σ_D (m)', 'SNR (dB)']
        values = [output_dto.sigma_D, output_dto.SNR_ADC]
        colors = ['green' if output_dto.sigma_D <= 0.01 else 'red',
                 'green' if output_dto.SNR_ADC >= 20 else 'red']
        ax4.barh(metrics, values, color=colors)
        ax4.set_xlabel('Value')
        ax4.set_title('Quality Metrics')
        ax4.grid(True, axis='x')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def showEvent(self, event):
        """Called when window is shown - load settings and update signal path."""
        super().showEvent(event)
        
        # Load GUI settings from gui.json after all widgets are fully displayed
        # This ensures that all widgets exist and are ready
        if not self._gui_settings_loaded:
            self.logger.info("showEvent: Loading GUI settings from gui.json")
            self._loading_settings = True
            self._load_gui_settings()
            self._loading_settings = False
            self._gui_settings_loaded = True
            self.logger.info(f"showEvent: GUI settings loaded, _gui_settings_loaded={self._gui_settings_loaded}, updating signal path")
            
            # Update signal path after loading settings
            self._update_signal_path_after_load()
            
            # Update Tp labels after loading settings
            self._update_tp_labels()
        else:
            self.logger.debug(f"showEvent: GUI settings already loaded (_gui_settings_loaded={self._gui_settings_loaded}), skipping")
    
    def _on_signal_path_param_changed(self):
        """Handler for signal path parameter change."""
        self._update_signal_path()
    
    def _update_signal_path_after_load(self):
        """Updates signal path after loading settings - with additional checks."""
        try:
            # Check if all required widgets exist
            if not hasattr(self, 'transducer_combo') or not hasattr(self, 'signal_path_widget'):
                self.logger.warning("Required widgets not initialized yet, skipping signal path update")
                return
            
            # Check if combo boxes have items
            if self.transducer_combo.count() == 0:
                self.logger.warning("Transducer combo box is empty, skipping signal path update")
                return
            
            # Check if a transducer is selected
            if not self.transducer_combo.currentText():
                self.logger.warning("No transducer selected, skipping signal path update")
                return
            
            # Update accuracy info first to update recommended frequencies
            # This must be done after transducer is loaded
            self.logger.info("_update_signal_path_after_load: Updating accuracy info to refresh recommended frequencies")
            self._update_accuracy_info()
            
            # Now update signal path
            self._update_signal_path()
        except Exception as e:
            self.logger.warning(f"Error updating signal path after load: {e}", exc_info=True)
    
    def _update_signal_path(self):
        """Updates signal path."""
        try:
            # Validate range before creating input_dto
            D_min = self.D_min_spin.value()
            D_max = self.D_max_spin.value()
            if D_max <= D_min:
                # Invalid range, skip update
                return
            
            input_dto = self._get_input_dto()
            if input_dto is None:
                return
            
            # Get parameters from widget (user can change them)
            lna_gain = self.signal_path_widget.get_lna_gain()
            lna_nf = self.signal_path_widget.get_lna_nf()
            vga_gain = self.signal_path_widget.get_vga_gain()
            # Bottom reflection is now in environment parameters, not in signal_path_widget
            
            # Log input parameters for debugging
            self.logger.info(f"Signal path calculation: D_target={input_dto.range.D_target}, "
                           f"z={input_dto.environment.z}, D_min={input_dto.range.D_min}, "
                           f"D_max={input_dto.range.D_max}")
            
            # Check if signal_path_calculator is initialized
            if not hasattr(self, 'signal_path_calculator') or self.signal_path_calculator is None:
                self.logger.error("signal_path_calculator is not initialized")
                return
            
            # Calculate signal path in core module
            # All calculations and data retrieval from data_provider happens in core
            # Bottom reflection is taken from input_dto.environment.bottom_reflection
            self.logger.info(f"Calculating signal path for transducer: {input_dto.hardware.transducer_id}")
            path_data = self.signal_path_calculator.calculate_signal_path(
                input_dto,
                lna_gain=lna_gain, lna_nf=lna_nf, vga_gain=vga_gain
            )
            
            # Check if path_data is valid
            if path_data is None:
                self.logger.error("Signal path calculation returned None")
                return
            
            if not isinstance(path_data, dict):
                self.logger.error(f"Signal path calculation returned invalid type: {type(path_data)}")
                return
            
            # Log path_data keys for debugging
            self.logger.info(f"Signal path data keys: {list(path_data.keys())}")
            if 'summary' in path_data:
                self.logger.info(f"Signal path summary keys: {list(path_data['summary'].keys())}")
            
            # Update diagram
            self.signal_path_widget.set_path_data(path_data)
            self.logger.info("Signal path data set to widget")
            
        except Exception as e:
            self.logger.error(f"Error updating signal path: {e}", exc_info=True)
    
    def _display_signals(self, input_dto: InputDTO, output_dto: OutputDTO):
        """Displays five signal graphs: transmitted, at bottom, received, after LNA, after VGA."""
        try:
            # Clear previous plots
            self.signals_plot_tx.clear()
            self.signals_plot_bottom.clear()
            self.signals_plot_rx.clear()
            self.signals_plot_lna.clear()
            self.signals_plot_vga.clear()
            
            # Check if signal data is available
            if (output_dto.tx_signal is None or output_dto.signal_at_bottom is None or 
                output_dto.received_signal is None or output_dto.time_axis is None):
                # No signal data available - show message
                self.signals_plot_tx.addItem(pg.TextItem('No signal data available.\nRun simulation to see signals.', 
                                                         anchor=(0.5, 0.5)))
                return
            
            # Get sampling frequency from ADC
            adc_params = self.data_provider.get_adc(input_dto.hardware.adc_id)
            fs = adc_params.get('f_s', 1e6)  # Default 1 MHz if not found
            
            # Convert lists back to numpy arrays
            t = np.array(output_dto.time_axis)
            tx_signal = np.array(output_dto.tx_signal)
            signal_at_bottom = np.array(output_dto.signal_at_bottom)
            received_signal = np.array(output_dto.received_signal)
            signal_after_lna = np.array(output_dto.signal_after_lna) if output_dto.signal_after_lna is not None else None
            signal_after_vga = np.array(output_dto.signal_after_vga) if output_dto.signal_after_vga is not None else None
            
            # Debug: print signal info
            print(f"DEBUG: tx_signal: len={len(tx_signal)}, min={np.min(tx_signal):.6e}, max={np.max(tx_signal):.6e}")
            print(f"DEBUG: signal_at_bottom: len={len(signal_at_bottom)}, min={np.min(signal_at_bottom):.6e}, max={np.max(signal_at_bottom):.6e}")
            print(f"DEBUG: received_signal: len={len(received_signal)}, min={np.min(received_signal):.6e}, max={np.max(received_signal):.6e}")
            print(f"DEBUG: time_axis: len={len(t)}, min={np.min(t):.6e}, max={np.max(t):.6e}")
            print(f"DEBUG: fs={fs:.0f} Hz")
            
            # Plot 1: Transmitted signal
            if len(t) == len(tx_signal) and len(tx_signal) > 0:
                t_ms = t * 1000  # Convert to milliseconds
                # Check for valid data
                if np.any(np.isfinite(tx_signal)) and np.any(np.isfinite(t_ms)):
                    self.signals_plot_tx.plot(t_ms, tx_signal, pen=pg.mkPen('b', width=2))
                    # Show first 5 ms or full signal, whichever is smaller
                    max_time = min(t[-1] * 1000, t[0] * 1000 + 5)
                    self.signals_plot_tx.setXRange(t[0] * 1000, max_time)
                    # Auto-scale Y axis for this signal
                    y_min, y_max = np.min(tx_signal), np.max(tx_signal)
                    y_range = y_max - y_min
                    if y_range > 1e-10:  # Check for non-zero range
                        # Add 10% padding
                        padding = y_range * 0.1
                        self.signals_plot_tx.setYRange(y_min - padding, y_max + padding)
                    else:
                        # If signal is constant, set range around the value
                        center = (y_min + y_max) / 2
                        self.signals_plot_tx.setYRange(center - 0.1, center + 0.1)
                    # Force update
                    self.signals_plot_tx.repaint()
                else:
                    self.signals_plot_tx.addItem(pg.TextItem('Invalid TX signal data\n(non-finite values)', 
                                                             anchor=(0.5, 0.5)))
            else:
                self.signals_plot_tx.addItem(pg.TextItem(f'Invalid TX signal data\nlen(t)={len(t)}, len(signal)={len(tx_signal)}', 
                                                         anchor=(0.5, 0.5)))
            
            # Plot 2: Signal after water forward
            # Use same time axis as graph 1 (no delay for visualization)
            # Display original signal values (no conversions)
            # All calculations done in Core - use values from OutputDTO
            if len(t) == len(signal_at_bottom) and len(signal_at_bottom) > 0:
                t_ms = t * 1000  # Convert to milliseconds (same as graph 1)
                # Check for valid data
                if np.any(np.isfinite(signal_at_bottom)) and np.any(np.isfinite(t_ms)):
                    self.signals_plot_bottom.plot(t_ms, signal_at_bottom, pen=pg.mkPen('g', width=2))
                    
                    # Use attenuation value calculated in Core (from OutputDTO)
                    if output_dto.attenuation_at_bottom_db is not None:
                        attenuation_db = output_dto.attenuation_at_bottom_db
                        # Only show text if attenuation is significant (not zero)
                        if abs(attenuation_db) > 0.1:  # More than 0.1 dB
                            # Add text label with attenuation info (no white background)
                            text_item = pg.TextItem(f'Signal attenuated by {abs(attenuation_db):.1f} dB', 
                                                   anchor=(0.5, 1), color='g')
                            # Position at top center of plot
                            max_time = min(t[-1] * 1000, t[0] * 1000 + 5)
                            text_x = (t[0] * 1000 + max_time) / 2
                            y_min, y_max = np.min(signal_at_bottom), np.max(signal_at_bottom)
                            y_range = y_max - y_min
                            if y_range > 1e-10:
                                text_y = y_max + y_range * 0.05
                            else:
                                text_y = y_max + abs(y_max) * 0.1
                            text_item.setPos(text_x, text_y)
                            self.signals_plot_bottom.addItem(text_item)
                    
                    # Use same X range as graph 1
                    max_time = min(t[-1] * 1000, t[0] * 1000 + 5)
                    self.signals_plot_bottom.setXRange(t[0] * 1000, max_time)
                    # Auto-scale Y axis for this signal
                    y_min, y_max = np.min(signal_at_bottom), np.max(signal_at_bottom)
                    y_range = y_max - y_min
                    if y_range > 1e-10:  # Check for non-zero range
                        # Add 10% padding
                        padding = y_range * 0.1
                        self.signals_plot_bottom.setYRange(y_min - padding, y_max + padding)
                    elif abs(y_max) > 1e-10:  # If signal is non-zero but constant
                        center = (y_min + y_max) / 2
                        self.signals_plot_bottom.setYRange(center - abs(center) * 0.1, center + abs(center) * 0.1)
                    else:
                        # If signal is near zero, set small range
                        self.signals_plot_bottom.setYRange(-0.001, 0.001)
                    # Force update
                    self.signals_plot_bottom.repaint()
                else:
                    self.signals_plot_bottom.addItem(pg.TextItem('Invalid signal data\n(non-finite values)', 
                                                                 anchor=(0.5, 0.5)))
            else:
                self.signals_plot_bottom.addItem(pg.TextItem(f'Invalid signal data\nlen(t)={len(t)}, len(signal)={len(signal_at_bottom)}', 
                                                             anchor=(0.5, 0.5)))
            
            # Plot 3: Signal after water backward
            # Use same time axis as graph 1 (no delay for visualization)
            # Display original signal values (no conversions)
            # All calculations done in Core - use values from OutputDTO
            if len(t) == len(received_signal) and len(received_signal) > 0:
                t_ms = t * 1000  # Convert to milliseconds (same as graph 1)
                # Check for valid data
                if np.any(np.isfinite(received_signal)) and np.any(np.isfinite(t_ms)):
                    self.signals_plot_rx.plot(t_ms, received_signal, pen=pg.mkPen('r', width=2))
                    
                    # Use attenuation value calculated in Core (from OutputDTO)
                    if output_dto.attenuation_received_db is not None:
                        attenuation_db = output_dto.attenuation_received_db
                        # Only show text if attenuation is significant (not zero)
                        if abs(attenuation_db) > 0.1:  # More than 0.1 dB
                            # Add text label with attenuation info (no white background)
                            text_item = pg.TextItem(f'Signal attenuated by {abs(attenuation_db):.1f} dB', 
                                                   anchor=(0.5, 1), color='r')
                            # Position at top center of plot
                            max_time = min(t[-1] * 1000, t[0] * 1000 + 5)
                            text_x = (t[0] * 1000 + max_time) / 2
                            y_min, y_max = np.min(received_signal), np.max(received_signal)
                            y_range = y_max - y_min
                            if y_range > 1e-10:
                                text_y = y_max + y_range * 0.05
                            else:
                                text_y = y_max + abs(y_max) * 0.1
                            text_item.setPos(text_x, text_y)
                            self.signals_plot_rx.addItem(text_item)
                    
                    # Use same X range as graph 1
                    max_time = min(t[-1] * 1000, t[0] * 1000 + 5)
                    self.signals_plot_rx.setXRange(t[0] * 1000, max_time)
                    # Auto-scale Y axis for this signal
                    y_min, y_max = np.min(received_signal), np.max(received_signal)
                    y_range = y_max - y_min
                    if y_range > 1e-10:  # Check for non-zero range
                        # Add 10% padding
                        padding = y_range * 0.1
                        self.signals_plot_rx.setYRange(y_min - padding, y_max + padding)
                    elif abs(y_max) > 1e-10:  # If signal is non-zero but constant
                        center = (y_min + y_max) / 2
                        self.signals_plot_rx.setYRange(center - abs(center) * 0.1, center + abs(center) * 0.1)
                    else:
                        # If signal is near zero, set small range
                        self.signals_plot_rx.setYRange(-0.001, 0.001)
                    # Force update
                    self.signals_plot_rx.repaint()
                else:
                    self.signals_plot_rx.addItem(pg.TextItem('Invalid signal data\n(non-finite values)', 
                                                             anchor=(0.5, 0.5)))
            else:
                self.signals_plot_rx.addItem(pg.TextItem(f'Invalid signal data\nlen(t)={len(t)}, len(signal)={len(received_signal)}', 
                                                         anchor=(0.5, 0.5)))
            
            # Plot 4: Signal after LNA
            if signal_after_lna is not None and len(signal_after_lna) > 0:
                # Use same time axis as graph 1
                if len(t) == len(signal_after_lna):
                    t_ms = t * 1000  # Convert to milliseconds
                    # Check for valid data
                    if np.any(np.isfinite(signal_after_lna)) and np.any(np.isfinite(t_ms)):
                        self.signals_plot_lna.plot(t_ms, signal_after_lna, pen=pg.mkPen('m', width=2))
                        
                        # Get LNA gain from input_dto
                        try:
                            lna_data = self.data_provider.get_lna(input_dto.hardware.lna_id)
                            lna_gain = lna_data.get('G_LNA', lna_data.get('G', 20))  # dB
                            # Add text label with gain info (no white background)
                            text_item = pg.TextItem(f'Gain {lna_gain:.1f} dB', 
                                                   anchor=(0.5, 1), color='m')
                            # Position at top center of plot
                            max_time = min(t[-1] * 1000, t[0] * 1000 + 5)
                            text_x = (t[0] * 1000 + max_time) / 2
                            y_min, y_max = np.min(signal_after_lna), np.max(signal_after_lna)
                            y_range = y_max - y_min
                            if y_range > 1e-10:
                                text_y = y_max + y_range * 0.05
                            else:
                                text_y = y_max + abs(y_max) * 0.1
                            text_item.setPos(text_x, text_y)
                            self.signals_plot_lna.addItem(text_item)
                        except Exception:
                            pass  # If can't get LNA gain, just skip the label
                        
                        # Use same X range as graph 1
                        max_time = min(t[-1] * 1000, t[0] * 1000 + 5)
                        self.signals_plot_lna.setXRange(t[0] * 1000, max_time)
                        # Auto-scale Y axis for this signal
                        y_min, y_max = np.min(signal_after_lna), np.max(signal_after_lna)
                        y_range = y_max - y_min
                        if y_range > 1e-10:  # Check for non-zero range
                            # Add 10% padding
                            padding = y_range * 0.1
                            self.signals_plot_lna.setYRange(y_min - padding, y_max + padding)
                        elif abs(y_max) > 1e-10:  # If signal is non-zero but constant
                            center = (y_min + y_max) / 2
                            self.signals_plot_lna.setYRange(center - abs(center) * 0.1, center + abs(center) * 0.1)
                        else:
                            # If signal is near zero, set small range
                            self.signals_plot_lna.setYRange(-0.001, 0.001)
                        # Force update
                        self.signals_plot_lna.repaint()
                    else:
                        self.signals_plot_lna.addItem(pg.TextItem('Invalid signal data\n(non-finite values)', 
                                                                 anchor=(0.5, 0.5)))
                else:
                    self.signals_plot_lna.addItem(pg.TextItem(f'Invalid signal data\nlen(t)={len(t)}, len(signal)={len(signal_after_lna)}', 
                                                             anchor=(0.5, 0.5)))
            else:
                self.signals_plot_lna.addItem(pg.TextItem('No LNA signal data available', 
                                                         anchor=(0.5, 0.5)))
            
            # Plot 5: Signal after VGA
            if signal_after_vga is not None and len(signal_after_vga) > 0:
                # Use same time axis as graph 1
                if len(t) == len(signal_after_vga):
                    t_ms = t * 1000  # Convert to milliseconds
                    # Check for valid data
                    if np.any(np.isfinite(signal_after_vga)) and np.any(np.isfinite(t_ms)):
                        self.signals_plot_vga.plot(t_ms, signal_after_vga, pen=pg.mkPen('c', width=2))
                        
                        # Get VGA gain from input_dto
                        try:
                            vga_data = self.data_provider.get_vga(input_dto.hardware.vga_id)
                            # VGA gain might be set dynamically, try to get current value from signal_path_widget
                            vga_gain = self.signal_path_widget.get_vga_gain() if hasattr(self, 'signal_path_widget') else vga_data.get('G', 30)
                            # Add text label with gain info (no white background)
                            text_item = pg.TextItem(f'Gain {vga_gain:.1f} dB', 
                                                   anchor=(0.5, 1), color='c')
                            # Position at top center of plot
                            max_time = min(t[-1] * 1000, t[0] * 1000 + 5)
                            text_x = (t[0] * 1000 + max_time) / 2
                            y_min, y_max = np.min(signal_after_vga), np.max(signal_after_vga)
                            y_range = y_max - y_min
                            if y_range > 1e-10:
                                text_y = y_max + y_range * 0.05
                            else:
                                text_y = y_max + abs(y_max) * 0.1
                            text_item.setPos(text_x, text_y)
                            self.signals_plot_vga.addItem(text_item)
                        except Exception:
                            pass  # If can't get VGA gain, just skip the label
                        
                        # Use same X range as graph 1
                        max_time = min(t[-1] * 1000, t[0] * 1000 + 5)
                        self.signals_plot_vga.setXRange(t[0] * 1000, max_time)
                        # Auto-scale Y axis for this signal
                        y_min, y_max = np.min(signal_after_vga), np.max(signal_after_vga)
                        y_range = y_max - y_min
                        if y_range > 1e-10:  # Check for non-zero range
                            # Add 10% padding
                            padding = y_range * 0.1
                            self.signals_plot_vga.setYRange(y_min - padding, y_max + padding)
                        elif abs(y_max) > 1e-10:  # If signal is non-zero but constant
                            center = (y_min + y_max) / 2
                            self.signals_plot_vga.setYRange(center - abs(center) * 0.1, center + abs(center) * 0.1)
                        else:
                            # If signal is near zero, set small range
                            self.signals_plot_vga.setYRange(-0.001, 0.001)
                        # Force update
                        self.signals_plot_vga.repaint()
                    else:
                        self.signals_plot_vga.addItem(pg.TextItem('Invalid signal data\n(non-finite values)', 
                                                                 anchor=(0.5, 0.5)))
                else:
                    self.signals_plot_vga.addItem(pg.TextItem(f'Invalid signal data\nlen(t)={len(t)}, len(signal)={len(signal_after_vga)}', 
                                                             anchor=(0.5, 0.5)))
            else:
                self.signals_plot_vga.addItem(pg.TextItem('No VGA signal data available', 
                                                         anchor=(0.5, 0.5)))
            
        except Exception as e:
            self.logger.warning(f"Error displaying signals: {e}")
            # Clear plots and show error
            self.signals_plot_tx.clear()
            self.signals_plot_bottom.clear()
            self.signals_plot_rx.clear()
            self.signals_plot_lna.clear()
            self.signals_plot_vga.clear()
            self.signals_plot_tx.addItem(pg.TextItem(f'Error displaying signals:\n{str(e)}', 
                                                    anchor=(0.5, 0.5), color='r'))
    
    def _display_signal_path(self, input_dto: InputDTO):
        """Displays signal path as diagram."""
        # Update widget parameters from real data (only on initial load, not after simulation)
        # Don't reset user-modified values after simulation
        try:
            lna_data = self.data_provider.get_lna(input_dto.hardware.lna_id)
            vga_data = self.data_provider.get_vga(input_dto.hardware.vga_id)
            
            # Only set values if they haven't been modified by user
            # For LNA, we can update from data (these are usually fixed)
            if lna_data:
                # Only update if current value is default or close to default
                current_lna_gain = self.signal_path_widget.get_lna_gain()
                default_lna_gain = lna_data.get('G_LNA', lna_data.get('G', 20))
                if abs(current_lna_gain - default_lna_gain) < 0.1:  # Close to default
                    self.signal_path_widget.set_lna_gain(default_lna_gain)
                
                current_lna_nf = self.signal_path_widget.get_lna_nf()
                default_lna_nf = lna_data.get('NF_LNA', lna_data.get('NF', 2))
                if abs(current_lna_nf - default_lna_nf) < 0.1:  # Close to default
                    self.signal_path_widget.set_lna_nf(default_lna_nf)
            
            # Don't reset VGA gain - it's user-controlled and should persist
            # VGA gain is set by user in signal_path_widget and should not be overwritten
        except Exception as e:
            self.logger.warning(f"Error loading LNA/VGA parameters: {e}")
        
        # Update diagram
        self._update_signal_path()
    
    def _run_optimization(self):
        """Runs iterative optimization."""
        input_dto = self._get_input_dto()
        if input_dto is None:
            return
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            output_dto = self.simulator.simulate(input_dto, iteration=iteration, absorption_only=True)
            self.simulation_history.append((input_dto, output_dto))
            
            if output_dto.success:
                QMessageBox.information(self, "Success", 
                                       f"Optimization completed in {iteration + 1} iterations")
                self._display_results(input_dto, output_dto)
                self._display_recommendations(output_dto)
                self._plot_results(input_dto, output_dto)
                return
            
            # Apply recommendations
            rec = output_dto.recommendations
            changes = self.simulator.optimizer.suggest_parameter_changes(input_dto, rec)
            
            # Update parameters
            if 'Tp' in changes:
                self.Tp_spin.setValue(changes['Tp'])
            if 'f_start' in changes:
                self.f_start_spin.setValue(changes['f_start'])
            if 'f_end' in changes:
                self.f_end_spin.setValue(changes['f_end'])
            
            # Create new DTO
            input_dto = self._get_input_dto()
            if input_dto is None:
                return
            
            iteration += 1
        
        QMessageBox.warning(self, "Warning", 
                           f"Maximum number of iterations reached ({max_iterations})")
        self._display_results(input_dto, output_dto)
        self._display_recommendations(output_dto)
        self._plot_results(input_dto, output_dto)
        self._display_signal_path(input_dto)
    
    def _export_results(self):
        """Exports results."""
        if not self.simulation_history:
            QMessageBox.warning(self, "Warning", "No data to export")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "JSON Files (*.json);;CSV Files (*.csv)")
        
        if not filename:
            return
        
        try:
            if filename.endswith('.json'):
                data = []
                for input_dto, output_dto in self.simulation_history:
                    data.append({
                        'input': input_dto.model_dump(),
                        'output': output_dto.model_dump()
                    })
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(self, "Success", "Data exported")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export error: {e}")


def main():
    """Entry point for GUI."""
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

