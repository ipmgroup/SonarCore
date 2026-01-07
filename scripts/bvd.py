"""
Graph Extractor - Application for extracting data from graphs
Uses PyQt6, pyqtgraph for interactive data extraction
"""

import sys
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QFileDialog,
                              QLineEdit, QTableWidget, QTableWidgetItem, 
                              QTabWidget, QGroupBox, QMessageBox, QSplitter,
                              QComboBox, QSpinBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import pyqtgraph as pg
from scipy.interpolate import UnivariateSpline, interp1d
from numpy.polynomial import Polynomial
import csv
import json
from bvd_model import calculate_bvd_parameters, bvd_admittance, calculate_model_curves
from transducer_models import (
    calculate_mbvd_parameters, mbvd_admittance, calculate_model_curves_mbvd,
    calculate_ebvd_parameters, ebvd_admittance, calculate_model_curves_ebvd,
    calculate_mason_parameters, mason_admittance, calculate_model_curves_mason,
    calculate_klm_parameters, klm_admittance, calculate_model_curves_klm
)

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"bvd_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("PyMuPDF is not installed. Install: pip install PyMuPDF")

from PIL import Image
import io


class ImageWidget(QLabel):
    """Widget for displaying images and handling clicks"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.points = []
        self.click_callback = None
        self.original_pixmap = None
        self.scale_factor = 1.0
        
        # Zooming
        self.zoom_rect = None  # (x1, y1, x2, y2) in original image coordinates
        self.is_selecting = False
        self.selection_start = None
        self.selection_current = None
        self.selection_mode = False  # Area selection mode (otherwise - point clicking)
        self.zoom_button = None  # Reference to zoom button for synchronization
        
        # Auto color detection
        self.color_picking_mode = False
        self.target_color = None
        self.calibration_mode = False  # Calibration mode - cursor always cross
        
    def setImage(self, pixmap):
        self.original_pixmap = pixmap
        self.updateDisplay()
        
    def updateDisplay(self):
        if self.original_pixmap:
            # Apply zoom if present
            if self.zoom_rect:
                x1, y1, x2, y2 = self.zoom_rect
                cropped = self.original_pixmap.copy(int(x1), int(y1), int(x2-x1), int(y2-y1))
                scaled = cropped.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            else:
                scaled = self.original_pixmap.scaled(
                    self.size(), 
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            
            # Draw points
            painter = QPainter(scaled)
            for i, point in enumerate(self.points):
                # Scale point coordinates considering zoom
                if self.zoom_rect:
                    x1, y1, x2, y2 = self.zoom_rect
                    # Check if point is within zoomed area
                    if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                        x = (point[0] - x1) * scaled.width() / (x2 - x1)
                        y = (point[1] - y1) * scaled.height() / (y2 - y1)
                    else:
                        continue  # Point outside zoomed area
                else:
                    x = point[0] * scaled.width() / self.original_pixmap.width()
                    y = point[1] * scaled.height() / self.original_pixmap.height()
                
                if i == 0:
                    painter.setPen(QPen(QColor(33, 150, 243), 3))
                    painter.setBrush(QColor(33, 150, 243))
                elif i < 3:
                    painter.setPen(QPen(QColor(76, 175, 80), 3))
                    painter.setBrush(QColor(76, 175, 80))
                else:
                    painter.setPen(QPen(QColor(255, 87, 34), 3))
                    painter.setBrush(QColor(255, 87, 34))
                    
                painter.drawEllipse(QPointF(x, y), 6, 6)
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.drawText(int(x - 5), int(y + 5), str(i + 1))
            
            # Draw selection rectangle
            if self.is_selecting and self.selection_start and self.selection_current:
                painter.setPen(QPen(QColor(255, 165, 0), 2, Qt.PenStyle.DashLine))
                painter.setBrush(QColor(255, 165, 0, 50))
                x1, y1 = self.selection_start
                x2, y2 = self.selection_current
                painter.drawRect(int(min(x1, x2)), int(min(y1, y2)), int(abs(x2-x1)), int(abs(y2-y1)))
            
            painter.end()
            self.setPixmap(scaled)
    
    def mousePressEvent(self, event):
        if self.original_pixmap:
            pixmap = self.pixmap()
            if pixmap:
                x_offset = (self.width() - pixmap.width()) / 2
                y_offset = (self.height() - pixmap.height()) / 2
                x = event.pos().x() - x_offset
                y = event.pos().y() - y_offset
                
                if 0 <= x <= pixmap.width() and 0 <= y <= pixmap.height():
                    if self.color_picking_mode:
                        # Color selection mode
                        orig_x, orig_y = self.displayToOriginal(x, y)
                        self.pickColorAt(int(orig_x), int(orig_y))
                        self.color_picking_mode = False
                        self.setCursor(Qt.CursorShape.ArrowCursor)
                    elif self.selection_mode:
                        # Area selection mode
                        self.is_selecting = True
                        self.selection_start = (x, y)
                        self.selection_current = (x, y)
                    elif self.click_callback:
                        # Point clicking mode
                        orig_x, orig_y = self.displayToOriginal(x, y)
                        self.click_callback(orig_x, orig_y)
    
    def mouseMoveEvent(self, event):
        if self.is_selecting:
            pixmap = self.pixmap()
            if pixmap:
                x_offset = (self.width() - pixmap.width()) / 2
                y_offset = (self.height() - pixmap.height()) / 2
                x = event.pos().x() - x_offset
                y = event.pos().y() - y_offset
                
                # Limit coordinates to image size
                x = max(0, min(x, pixmap.width()))
                y = max(0, min(y, pixmap.height()))
                
                self.selection_current = (x, y)
                self.updateDisplay()
    
    def mouseReleaseEvent(self, event):
        if self.is_selecting:
            self.is_selecting = False
            if self.selection_start and self.selection_current:
                x1, y1 = self.selection_start
                x2, y2 = self.selection_current
                
                # Minimum selection size
                if abs(x2 - x1) > 20 and abs(y2 - y1) > 20:
                    # Convert to original image coordinates
                    orig_x1, orig_y1 = self.displayToOriginal(min(x1, x2), min(y1, y2))
                    orig_x2, orig_y2 = self.displayToOriginal(max(x1, x2), max(y1, y2))
                    self.zoom_rect = (orig_x1, orig_y1, orig_x2, orig_y2)
                    
                    # Automatically disable selection mode after zoom
                    self.setSelectionMode(False)
                
                self.selection_start = None
                self.selection_current = None
                self.updateDisplay()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateDisplay()
    
    def clearPoints(self):
        self.points = []
        self.updateDisplay()
    
    def addPoint(self, x, y):
        self.points.append((x, y))
        self.updateDisplay()
    
    def removeLastPoint(self):
        if self.points:
            self.points.pop()
            self.updateDisplay()
    
    def displayToOriginal(self, disp_x, disp_y):
        """Convert display image coordinates to original coordinates"""
        pixmap = self.pixmap()
        if not pixmap:
            return disp_x, disp_y
        
        if self.zoom_rect:
            x1, y1, x2, y2 = self.zoom_rect
            orig_x = x1 + disp_x * (x2 - x1) / pixmap.width()
            orig_y = y1 + disp_y * (y2 - y1) / pixmap.height()
        else:
            orig_x = disp_x * self.original_pixmap.width() / pixmap.width()
            orig_y = disp_y * self.original_pixmap.height() / pixmap.height()
        
        return orig_x, orig_y
    
    def setSelectionMode(self, enabled):
        """Toggle area selection mode"""
        self.selection_mode = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            # If calibration mode, keep cross cursor
            if self.calibration_mode:
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # Synchronize state with button if attached
        if hasattr(self, 'zoom_button') and self.zoom_button:
            self.zoom_button.setChecked(enabled)
    
    def resetZoom(self):
        """Reset zoom to full view"""
        self.zoom_rect = None
        self.updateDisplay()
    
    def pickColorAt(self, x, y):
        """Get color at specified point"""
        if not self.original_pixmap:
            return
        
        image = self.original_pixmap.toImage()
        if 0 <= x < image.width() and 0 <= y < image.height():
            color = QColor(image.pixel(x, y))
            self.target_color = (color.red(), color.green(), color.blue())
            if hasattr(self, 'color_callback') and self.color_callback:
                self.color_callback(self.target_color)
    
    def startColorPicking(self):
        """Enable color picking mode"""
        self.color_picking_mode = True
        self.setCursor(Qt.CursorShape.CrossCursor)


class GraphExtractorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graph Extractor - BVD Parameters")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data
        self.image_path = None
        self.coord_points = []  # [(x, y), ...] - 3 points for calibration
        self.data_points = []   # [(x, y), ...] - data points
        self.calibration = None  # Coordinate calibration
        self.extracted_data = []  # [(freq, value), ...]
        self.all_functions = []  # All extracted functions
        
        # Storage for additional axes for multiple functions
        self.extra_axes = []
        # Storage for additional ViewBoxes for real Y range mode
        self.extra_viewboxes = []
        
        # PDF document storage
        self.pdf_doc = None  # PyMuPDF document object
        self.pdf_path = None  # Path to PDF file
        self.pdf_current_page = 0  # Current page index (0-based)
        self.pdf_total_pages = 0  # Total number of pages
        
        self.initUI()
        
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Tabs for different stages
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.onTabChanged)
        layout.addWidget(self.tabs)
        
        # Step 1: Load
        self.tab_load = self.createLoadTab()
        self.tabs.addTab(self.tab_load, "1. Load")
        
        # Step 2: Coordinate calibration
        self.tab_calib = self.createCalibrationTab()
        self.tabs.addTab(self.tab_calib, "2. Coordinates")
        
        # Step 3: Data extraction
        self.tab_extract = self.createExtractionTab()
        self.tabs.addTab(self.tab_extract, "3. Data Points")
        
        # Step 4: Results
        self.tab_results = self.createResultsTab()
        self.tabs.addTab(self.tab_results, "4. Results")
        
        # Step 5: BVD model
        self.tab_bvd = self.createBVDTab()
        self.tabs.addTab(self.tab_bvd, "5. BVD Model")
        
        # Step 6: RX/TX Sensitivity
        self.tab_rxtx = self.createRXTXTab()
        self.tabs.addTab(self.tab_rxtx, "6. RX/TX Sensitivity")
        
        # Disable tabs until image is loaded
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)
        self.tabs.setTabEnabled(4, False)
        self.tabs.setTabEnabled(5, False)
        
    def createLoadTab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Instructions
        info = QLabel("""
        <b style='font-size: 18px;'>📁 STEP 1: Load Data</b><br><br>
        <b>Workflow:</b><br>
        1️⃣ Load graph image/PDF <b>OR</b> import data from CSV/JSON files<br>
        2️⃣ Calibrate coordinate axes (3 points) - only for images<br>
        3️⃣ Extract data from graph - only for images<br>
        4️⃣ View results and export<br>
        5️⃣ (Optional) Calculate BVD parameters
        """)
        info.setStyleSheet("font-size: 14px; padding: 20px; background: #e3f2fd; border-radius: 8px;")
        layout.addWidget(info)
        
        # Load buttons
        btn_layout = QVBoxLayout()
        
        # Image/PDF loading section
        image_group = QGroupBox("Load from Image/PDF")
        image_layout = QVBoxLayout()
        
        # Buttons row
        buttons_row = QHBoxLayout()
        btn_image = QPushButton("📷 Load Image")
        btn_image.clicked.connect(self.loadImage)
        buttons_row.addWidget(btn_image)
        
        if PDF_AVAILABLE:
            btn_pdf = QPushButton("📄 Load PDF")
            btn_pdf.clicked.connect(lambda: self.loadPDFFromButton())
            buttons_row.addWidget(btn_pdf)
        buttons_row.addStretch()
        image_layout.addLayout(buttons_row)
        
        # PDF page selection (only visible when PDF is loaded)
        if PDF_AVAILABLE:
            pdf_page_layout = QHBoxLayout()
            pdf_page_layout.addWidget(QLabel("PDF Page:"))
            self.pdf_page_spinbox = QSpinBox()
            self.pdf_page_spinbox.setMinimum(1)
            self.pdf_page_spinbox.setMaximum(1)
            self.pdf_page_spinbox.setValue(1)
            self.pdf_page_spinbox.setEnabled(False)  # Disabled until PDF is loaded
            self.pdf_page_spinbox.setToolTip("Select page number from PDF document")
            self.pdf_page_spinbox.valueChanged.connect(self.onPDFPageChanged)
            pdf_page_layout.addWidget(self.pdf_page_spinbox)
            self.pdf_page_total_label = QLabel("of 1")
            pdf_page_layout.addWidget(self.pdf_page_total_label)
            pdf_page_layout.addStretch()
            image_layout.addLayout(pdf_page_layout)
        
        image_group.setLayout(image_layout)
        btn_layout.addWidget(image_group)
        
        # Data file loading section
        data_group = QGroupBox("Load from Data Files")
        data_layout = QHBoxLayout()
        btn_csv = QPushButton("📥 Load CSV")
        btn_csv.clicked.connect(self.loadDataFromCSV)
        data_layout.addWidget(btn_csv)
        
        btn_json = QPushButton("📥 Load JSON")
        btn_json.clicked.connect(self.loadDataFromJSON)
        data_layout.addWidget(btn_json)
        data_layout.addStretch()
        data_group.setLayout(data_layout)
        btn_layout.addWidget(data_group)
        
        # PDF zoom settings
        if PDF_AVAILABLE:
            zoom_layout = QHBoxLayout()
            zoom_layout.addWidget(QLabel("PDF Zoom (quality):"))
            self.pdf_zoom_input = QLineEdit("3.0")
            self.pdf_zoom_input.setMaximumWidth(100)
            self.pdf_zoom_input.setToolTip("Resolution multiplier (1.0-5.0). Higher = better quality, but slower")
            zoom_layout.addWidget(self.pdf_zoom_input)
            zoom_layout.addStretch()
            btn_layout.addLayout(zoom_layout)
        
        layout.addLayout(btn_layout)
        
        # Image preview
        self.preview_label = QLabel("Image not loaded")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setStyleSheet("border: 2px dashed #ccc; background: #f5f5f5;")
        layout.addWidget(self.preview_label)
        
        layout.addStretch()
        return widget
    
    def createBVDTab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # BVD model information
        info = QLabel("""
        <b>BVD (Butterworth-Van Dyke) model of piezoelectric transducer</b><br>
        Equivalent circuit: C₀ in parallel with series circuit R₁-L₁-C₁<br>
        Parameter calculation from Admittance data (Conductance and Susceptance)
        """)
        info.setStyleSheet("padding: 15px; background: #e8f5e9; border-radius: 8px;")
        layout.addWidget(info)
        
        # Note: Function assignment is done in TAB 4 (Results tab)
        info_bvd = QLabel("""
        <b>📋 Workflow:</b><br>
        1️⃣ Assign Conductance and Susceptance functions in TAB 4 (Results tab)<br>
        2️⃣ Select model type below and enter transducer parameters<br>
        3️⃣ Click "Calculate Model" button below to create model from your experimental data
        """)
        info_bvd.setStyleSheet("padding: 10px; background: #e3f2fd; border-radius: 5px;")
        layout.addWidget(info_bvd)
        
        # Model selection
        model_group = QGroupBox("🔬 Model Selection")
        model_layout = QVBoxLayout()
        
        info_model = QLabel("""
        <b>Select model type:</b> The model will be created from your extracted Conductance and Susceptance data.
        """)
        info_model.setStyleSheet("padding: 5px; font-size: 11px;")
        model_layout.addWidget(info_model)
        
        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("Transducer Model:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "BVD (Basic)",
            "MBVD (Modified BVD - Recommended)",
            "EBVD (Extended BVD with Harmonics - Best for dual peaks)",
            "Mason (Physical Model - Advanced)",
            "KLM (Krimholtz-Leedom-Matthaei - Hydroacoustic)"
        ])
        self.model_type_combo.setCurrentIndex(2)  # Default to EBVD (best for dual-peak Conductance)
        self.model_type_combo.setToolTip(
            "BVD: Basic 4-parameter model (C0, R1, L1, C1)\n"
            "MBVD: Modified BVD with dielectric losses (R0) - usually more accurate\n"
            "EBVD: Extended BVD with harmonic branches (R2, L2, C2) - best for complex resonances\n"
            "Mason: Physical model based on acoustic waves - for ultrasonic transducers\n"
            "KLM: Transformer-coupled model - particularly accurate for hydroacoustic transducers\n\n"
            "The model will be fitted to your experimental data using optimization."
        )
        combo_layout.addWidget(self.model_type_combo)
        combo_layout.addStretch()
        model_layout.addLayout(combo_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Transducer parameters
        params_group = QGroupBox("Transducer Parameters")
        params_layout = QHBoxLayout()
        
        params_layout.addWidget(QLabel("Static capacitance C₀ (nF):"))
        self.input_c0 = QLineEdit("12")
        self.input_c0.setMaximumWidth(150)
        params_layout.addWidget(self.input_c0)
        
        params_layout.addWidget(QLabel("Resonant frequency fs (kHz):"))
        self.input_fs = QLineEdit("25")
        self.input_fs.setMaximumWidth(150)
        params_layout.addWidget(self.input_fs)
        
        params_layout.addStretch()
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Calculated parameters group
        self.params_group = QGroupBox("Calculated Model Parameters")
        params_layout = QVBoxLayout()
        
        # Parameters table
        self.bvd_params_table = QTableWidget()
        self.bvd_params_table.setColumnCount(3)
        self.bvd_params_table.setHorizontalHeaderLabels(['Parameter', 'Value', 'Unit'])
        self.bvd_params_table.setRowCount(15)  # Can accommodate EBVD (up to 12 params), MBVD (9 params) or BVD (8 params)
        
        params = [
            ('C₀ (static capacitance)', '', 'nF'),
            ('fs (resonant frequency)', '', 'kHz'),
            ('fp (antiresonant frequency)', '', 'kHz'),
            ('R₁ (loss resistance)', '', 'Ω'),
            ('L₁ (dynamic inductance)', '', 'mH'),
            ('C₁ (dynamic capacitance)', '', 'nF'),
            ('Qm (mechanical Q-factor)', '', ''),
            ('k (coupling coefficient)', '', '')
        ]
        
        for i, (param, val, unit) in enumerate(params):
            self.bvd_params_table.setItem(i, 0, QTableWidgetItem(param))
            self.bvd_params_table.setItem(i, 1, QTableWidgetItem(val))
            self.bvd_params_table.setItem(i, 2, QTableWidgetItem(unit))
        
        self.bvd_params_table.resizeColumnsToContents()
        params_layout.addWidget(self.bvd_params_table)
        
        self.params_group.setLayout(params_layout)
        layout.addWidget(self.params_group)
        
        # Comparison plots - 4 plots in 2x2 grid
        compare_group = QGroupBox("Comparison: Experimental (PDF) vs BVD Model")
        compare_layout = QVBoxLayout()
        
        # Top row: Conductance and Susceptance
        top_layout = QHBoxLayout()
        
        self.bvd_plot_g = pg.PlotWidget(title="Conductance (G)")
        self.bvd_plot_g.setBackground('w')
        self.bvd_plot_g.setLabel('left', 'G (mS)')
        self.bvd_plot_g.setLabel('bottom', 'Frequency (kHz)')
        self.bvd_plot_g.showGrid(x=True, y=True, alpha=0.3)
        self.bvd_plot_g.addLegend()
        top_layout.addWidget(self.bvd_plot_g)
        
        self.bvd_plot_b = pg.PlotWidget(title="Susceptance (B)")
        self.bvd_plot_b.setBackground('w')
        self.bvd_plot_b.setLabel('left', 'B (mS)')
        self.bvd_plot_b.setLabel('bottom', 'Frequency (kHz)')
        self.bvd_plot_b.showGrid(x=True, y=True, alpha=0.3)
        self.bvd_plot_b.addLegend()
        top_layout.addWidget(self.bvd_plot_b)
        
        compare_layout.addLayout(top_layout)
        
        # Bottom row: |Y| and Phase
        bottom_layout = QHBoxLayout()
        
        self.bvd_plot_mag = pg.PlotWidget(title="Admittance Magnitude |Y|")
        self.bvd_plot_mag.setBackground('w')
        self.bvd_plot_mag.setLabel('left', '|Y| (mS)')
        self.bvd_plot_mag.setLabel('bottom', 'Frequency (kHz)')
        self.bvd_plot_mag.showGrid(x=True, y=True, alpha=0.3)
        self.bvd_plot_mag.addLegend()
        bottom_layout.addWidget(self.bvd_plot_mag)
        
        self.bvd_plot_phase = pg.PlotWidget(title="Admittance Phase")
        self.bvd_plot_phase.setBackground('w')
        self.bvd_plot_phase.setLabel('left', 'Phase (degrees)')
        self.bvd_plot_phase.setLabel('bottom', 'Frequency (kHz)')
        self.bvd_plot_phase.showGrid(x=True, y=True, alpha=0.3)
        self.bvd_plot_phase.addLegend()
        bottom_layout.addWidget(self.bvd_plot_phase)
        
        compare_layout.addLayout(bottom_layout)
        
        compare_group.setLayout(compare_layout)
        layout.addWidget(compare_group)
        
        # Fit quality assessment
        self.fit_quality_label = QLabel("")
        self.fit_quality_label.setStyleSheet("padding: 10px; font-size: 14px;")
        layout.addWidget(self.fit_quality_label)
        
        # Main button to create model (in TAB 5)
        btn_create_model = QPushButton("🔬 Create Model")
        btn_create_model.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 15px 30px;
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        btn_create_model.setToolTip(
            "Create transducer model from your experimental data.\n"
            "Make sure:\n"
            "1. Conductance and Susceptance are assigned in TAB 4\n"
            "2. Model type is selected above\n"
            "3. Transducer parameters (C₀, fs) are entered"
        )
        btn_create_model.clicked.connect(self.calculateBVD)
        layout.addWidget(btn_create_model)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        btn_export_bvd = QPushButton("💾 Export BVD Parameters")
        btn_export_bvd.clicked.connect(self.exportBVDParams)
        btn_layout.addWidget(btn_export_bvd)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return widget
    
    def createRXTXTab(self):
        """Create TAB 6 for RX/TX Sensitivity graphs"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel("📊 RX/TX Sensitivity Graphs")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Info label
        info_label = QLabel(
            "This tab automatically displays RX and TX sensitivity graphs if functions with 'RX' or 'TX' in their names are found.\n"
            "Graphs are updated automatically when functions are added or modified."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("padding: 5px; color: #666;")
        layout.addWidget(info_label)
        
        # Splitter for two graphs side by side
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # RX Graph
        rx_group = QGroupBox("📡 RX Sensitivity")
        rx_layout = QVBoxLayout()
        self.rx_plot = pg.PlotWidget()
        self.rx_plot.setBackground('w')
        self.rx_plot.setLabel('left', 'Sensitivity', units='dB')
        self.rx_plot.setLabel('bottom', 'Frequency', units='kHz')
        self.rx_plot.addLegend()
        self.rx_plot.showGrid(x=True, y=True, alpha=0.3)
        rx_layout.addWidget(self.rx_plot)
        rx_group.setLayout(rx_layout)
        splitter.addWidget(rx_group)
        
        # TX Graph
        tx_group = QGroupBox("📡 TX Sensitivity")
        tx_layout = QVBoxLayout()
        self.tx_plot = pg.PlotWidget()
        self.tx_plot.setBackground('w')
        self.tx_plot.setLabel('left', 'Sensitivity', units='dB')
        self.tx_plot.setLabel('bottom', 'Frequency', units='kHz')
        self.tx_plot.addLegend()
        self.tx_plot.showGrid(x=True, y=True, alpha=0.3)
        tx_layout.addWidget(self.tx_plot)
        tx_group.setLayout(tx_layout)
        splitter.addWidget(tx_group)
        
        # Set equal sizes for splitter
        splitter.setSizes([500, 500])
        layout.addWidget(splitter)
        
        # Status label
        self.rxtx_status_label = QLabel("No RX/TX functions found. Load functions with 'RX' or 'TX' in their names.")
        self.rxtx_status_label.setStyleSheet("padding: 10px; color: #666; font-style: italic;")
        layout.addWidget(self.rxtx_status_label)
        
        return widget
    
    def updateRXTXGraphs(self):
        """Update RX/TX graphs based on function names - shows fitted functions"""
        if not hasattr(self, 'rx_plot') or not hasattr(self, 'tx_plot'):
            return
        
        # Clear existing plots
        self.rx_plot.clear()
        self.tx_plot.clear()
        
        # Find RX and TX functions
        rx_functions = []
        tx_functions = []
        
        for func in self.all_functions:
            if not func.get('visible', True):
                continue
            name_lower = func['name'].lower()
            if 'rx' in name_lower and 'tx' not in name_lower:
                rx_functions.append(func)
            elif 'tx' in name_lower and 'rx' not in name_lower:
                tx_functions.append(func)
        
        # Plot RX functions (using fitted data)
        if rx_functions:
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, func in enumerate(rx_functions):
                # Ensure fitting is calculated
                if not func.get('data') and func.get('original_points'):
                    self.recalculateFittingForFunction(func)
                
                if not func.get('data'):
                    continue
                
                # Use fitted data (func['data'] contains fitted curve with ~200 points)
                data = np.array(func['data'])
                if len(data) == 0:
                    continue
                
                freq = data[:, 0]
                values = data[:, 1]
                color = colors[i % len(colors)]
                
                # Plot fitted curve (smooth line, no symbols for fitted data)
                self.rx_plot.plot(freq, values, pen=pg.mkPen(color, width=2), 
                                 name=func['name'])
                
                # Optionally show original points as small markers
                if func.get('original_points'):
                    orig = np.array(func['original_points'])
                    if len(orig) > 0:
                        orig_freq = orig[:, 0]
                        orig_values = orig[:, 1]
                        # Show original points as small transparent markers
                        if len(orig) <= 100:  # Only show markers if not too many points
                            self.rx_plot.plot(orig_freq, orig_values, 
                                             pen=None, symbol='o', 
                                             symbolBrush=pg.mkBrush(color, alpha=100),
                                             symbolSize=4,
                                             name=None)  # Don't add to legend
            
            self.rx_plot.setTitle(f"RX Sensitivity ({len(rx_functions)} function(s))")
        else:
            self.rx_plot.setTitle("RX Sensitivity (No data)")
        
        # Plot TX functions (using fitted data)
        if tx_functions:
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, func in enumerate(tx_functions):
                # Ensure fitting is calculated
                if not func.get('data') and func.get('original_points'):
                    self.recalculateFittingForFunction(func)
                
                if not func.get('data'):
                    continue
                
                # Use fitted data (func['data'] contains fitted curve with ~200 points)
                data = np.array(func['data'])
                if len(data) == 0:
                    continue
                
                freq = data[:, 0]
                values = data[:, 1]
                color = colors[i % len(colors)]
                
                # Plot fitted curve (smooth line, no symbols for fitted data)
                self.tx_plot.plot(freq, values, pen=pg.mkPen(color, width=2), 
                                 name=func['name'])
                
                # Optionally show original points as small markers
                if func.get('original_points'):
                    orig = np.array(func['original_points'])
                    if len(orig) > 0:
                        orig_freq = orig[:, 0]
                        orig_values = orig[:, 1]
                        # Show original points as small transparent markers
                        if len(orig) <= 100:  # Only show markers if not too many points
                            self.tx_plot.plot(orig_freq, orig_values, 
                                             pen=None, symbol='o', 
                                             symbolBrush=pg.mkBrush(color, alpha=100),
                                             symbolSize=4,
                                             name=None)  # Don't add to legend
            
            self.tx_plot.setTitle(f"TX Sensitivity ({len(tx_functions)} function(s))")
        else:
            self.tx_plot.setTitle("TX Sensitivity (No data)")
        
        # Update status label
        if rx_functions or tx_functions:
            status_text = f"Found: {len(rx_functions)} RX function(s), {len(tx_functions)} TX function(s)"
            self.rxtx_status_label.setText(status_text)
            self.rxtx_status_label.setStyleSheet("padding: 10px; color: #006400; font-weight: bold;")
        else:
            self.rxtx_status_label.setText("No RX/TX functions found. Load functions with 'RX' or 'TX' in their names.")
            self.rxtx_status_label.setStyleSheet("padding: 10px; color: #666; font-style: italic;")
    
    def updateBVDFunctionLists(self):
        """Update function selection combo boxes in BVD tab (deprecated - use TAB 4 instead)"""
        # This method is kept for backward compatibility but function assignment
        # should be done in TAB 4 (Results tab)
        pass
    
    def calculateBVD(self):
        """Calculate BVD parameters from Admittance data"""
        try:
            # Get assigned functions from TAB 4 (bvd_type field)
            conductance_func = None
            susceptance_func = None
            
            for func in self.all_functions:
                if func.get('bvd_type') == 'conductance':
                    conductance_func = func
                elif func.get('bvd_type') == 'susceptance':
                    susceptance_func = func
            
            if not conductance_func or not susceptance_func:
                QMessageBox.warning(
                    self, "Error", 
                    "Please assign Conductance and Susceptance functions in TAB 4 (Results tab).\n"
                    "Use the 'BVD Model Assignment' section to select which function is Conductance and which is Susceptance."
                )
                return
            
            if conductance_func == susceptance_func:
                QMessageBox.warning(
                    self, "Error", 
                    "Conductance and Susceptance must be different functions"
                )
                return
            
            # Get C₀ and fs
            C0_nF = float(self.input_c0.text())
            fs_kHz = float(self.input_fs.text())
            
            # Convert to SI units
            C0 = C0_nF * 1e-9  # nF -> F
            fs = fs_kHz * 1e3  # kHz -> Hz
            
            # Use fitted function data (not original points)
            # Conductance data from fitted function
            g_data = conductance_func['data']  # This is the fitted data (spline/polynomial)
            # Data is stored as [freq_kHz, value_mS], so we need to convert
            freq_g = np.array([p[0] * 1e3 for p in g_data])  # kHz -> Hz
            g_values = np.array([p[1] for p in g_data])  # Keep in mS for now
            
            # Susceptance data from fitted function
            b_data = susceptance_func['data']  # This is the fitted data (spline/polynomial)
            freq_b = np.array([p[0] * 1e3 for p in b_data])  # kHz -> Hz
            b_values = np.array([p[1] for p in b_data])  # Keep in mS for now
            
            # Convert to S for calculations
            g_values_S = g_values * 1e-3  # mS -> S
            b_values_S = b_values * 1e-3  # mS -> S
            
            # Get selected model type
            model_type = self.model_type_combo.currentText()
            use_mbvd = "MBVD" in model_type
            use_ebvd = "EBVD" in model_type
            use_mason = "Mason" in model_type
            use_klm = "KLM" in model_type
            
            # Calculate parameters using selected model
            if use_klm:
                logger.info("Starting KLM (Krimholtz-Leedom-Matthaei) parameter calculation...")
                logger.info(f"Input parameters: C0={C0_nF} nF, fs={fs_kHz} kHz (initial)")
                logger.info(f"Conductance data: {len(g_values)} points, range: {g_values.min():.4f} - {g_values.max():.4f} mS")
                logger.info(f"Susceptance data: {len(b_values)} points, range: {b_values.min():.4f} - {b_values.max():.4f} mS")
                
                self.bvd_params = calculate_klm_parameters(
                    freq_g, g_values_S, freq_b, b_values_S, C0
                )
                self.bvd_params['model_type'] = 'KLM'
                
                logger.info("KLM parameter calculation completed successfully")
                logger.info(f"Calculated KLM parameters: k_t={self.bvd_params.get('k_t', 0):.4f}, t={self.bvd_params.get('t', 0)*1e3:.4f} mm")
                logger.info(f"Physical parameters: Z_a={self.bvd_params.get('Z_a', 0):.2e} kg/(m²·s), A={self.bvd_params.get('A', 0)*1e6:.4f} mm²")
                logger.info(f"Acoustic load: Z_load={self.bvd_params.get('Z_load', 0):.2e} kg/(m²·s)")
                
                # Update group title
                self.params_group.setTitle("Calculated KLM Model Parameters")
                
                # Update table for KLM (physical parameters similar to Mason)
                param_data = [
                    ('C₀', f"{self.bvd_params['C0']*1e9:.2f}", 'nF'),
                    ('R₀', f"{self.bvd_params.get('R0', 0):.2f}", 'Ohm'),
                    ('R_m', f"{self.bvd_params.get('R_m', 0):.2f}", 'Ohm'),
                    ('k_t', f"{self.bvd_params.get('k_t', 0):.4f}", ''),
                    ('Z_a', f"{self.bvd_params.get('Z_a', 0):.2e}", 'kg/(m²·s)'),
                    ('Z_load', f"{self.bvd_params.get('Z_load', 0):.2e}", 'kg/(m²·s)'),
                    ('t', f"{self.bvd_params.get('t', 0)*1e3:.4f}", 'mm'),
                    ('A', f"{self.bvd_params.get('A', 0)*1e6:.4f}", 'mm²'),
                    ('ρ', f"{self.bvd_params.get('rho', 0):.0f}", 'kg/m³'),
                    ('c', f"{self.bvd_params.get('c', 0):.0f}", 'm/s'),
                    ('α', f"{self.bvd_params.get('alpha', 0):.4f}", 'Np/m'),
                    ('fs', f"{self.bvd_params['fs']*1e-3:.4f}", 'kHz'),
                    ('fp', f"{self.bvd_params['fp']*1e-3:.4f}", 'kHz'),
                    ('Qm', f"{self.bvd_params['Qm']:.2f}", ''),
                    ('k', f"{self.bvd_params['k']:.4f}", ''),
                    ('tan δ', f"{self.bvd_params.get('tan_delta', 0.0):.6f}", '')
                ]
                
                self.bvd_params_table.setRowCount(len(param_data))
                for i, (name, val, unit) in enumerate(param_data):
                    self.bvd_params_table.setItem(i, 0, QTableWidgetItem(name))
                    self.bvd_params_table.setItem(i, 1, QTableWidgetItem(val))
                    self.bvd_params_table.setItem(i, 2, QTableWidgetItem(unit))
            elif use_mason:
                logger.info("Starting Mason (Physical Model) parameter calculation...")
                logger.info(f"Input parameters: C0={C0_nF} nF, fs={fs_kHz} kHz (initial)")
                logger.info(f"Conductance data: {len(g_values)} points, range: {g_values.min():.4f} - {g_values.max():.4f} mS")
                logger.info(f"Susceptance data: {len(b_values)} points, range: {b_values.min():.4f} - {b_values.max():.4f} mS")
                
                self.bvd_params = calculate_mason_parameters(
                    freq_g, g_values_S, freq_b, b_values_S, C0
                )
                self.bvd_params['model_type'] = 'Mason'
                
                logger.info("Mason parameter calculation completed successfully")
                logger.info(f"Calculated Mason parameters: k_t={self.bvd_params.get('k_t', 0):.4f}, t={self.bvd_params.get('t', 0)*1e3:.4f} mm")
                logger.info(f"Physical parameters: Z_a={self.bvd_params.get('Z_a', 0):.2e} kg/(m²·s), A={self.bvd_params.get('A', 0)*1e6:.4f} mm²")
                
                # Update group title
                self.params_group.setTitle("Calculated Mason Model Parameters")
                
                # Update table for Mason (physical parameters)
                param_data = [
                    ('C₀', f"{self.bvd_params['C0']*1e9:.2f}", 'nF'),
                    ('R₀', f"{self.bvd_params.get('R0', 0):.2f}", 'Ohm'),
                    ('R_m', f"{self.bvd_params.get('R_m', 0):.2f}", 'Ohm'),
                    ('k_t', f"{self.bvd_params.get('k_t', 0):.4f}", ''),
                    ('Z_a', f"{self.bvd_params.get('Z_a', 0):.2e}", 'kg/(m²·s)'),
                    ('t', f"{self.bvd_params.get('t', 0)*1e3:.4f}", 'mm'),
                    ('A', f"{self.bvd_params.get('A', 0)*1e6:.4f}", 'mm²'),
                    ('ρ', f"{self.bvd_params.get('rho', 0):.0f}", 'kg/m³'),
                    ('c', f"{self.bvd_params.get('c', 0):.0f}", 'm/s'),
                    ('α', f"{self.bvd_params.get('alpha', 0):.4f}", 'Np/m'),
                    ('fs', f"{self.bvd_params['fs']*1e-3:.4f}", 'kHz'),
                    ('fp', f"{self.bvd_params['fp']*1e-3:.4f}", 'kHz'),
                    ('Qm', f"{self.bvd_params['Qm']:.2f}", ''),
                    ('k', f"{self.bvd_params['k']:.4f}", ''),
                    ('tan δ', f"{self.bvd_params.get('tan_delta', 0.0):.6f}", '')
                ]
                
                self.bvd_params_table.setRowCount(len(param_data))
                for i, (name, val, unit) in enumerate(param_data):
                    self.bvd_params_table.setItem(i, 0, QTableWidgetItem(name))
                    self.bvd_params_table.setItem(i, 1, QTableWidgetItem(val))
                    self.bvd_params_table.setItem(i, 2, QTableWidgetItem(unit))
            elif use_ebvd:
                logger.info("Starting EBVD (Extended BVD) parameter calculation...")
                logger.info(f"Input parameters: C0={C0_nF} nF, fs={fs_kHz} kHz (initial)")
                logger.info(f"Conductance data: {len(g_values)} points, range: {g_values.min():.4f} - {g_values.max():.4f} mS")
                logger.info(f"Susceptance data: {len(b_values)} points, range: {b_values.min():.4f} - {b_values.max():.4f} mS")
                
                self.bvd_params = calculate_ebvd_parameters(
                    freq_g, g_values_S, freq_b, b_values_S, C0, use_harmonic=True
                )
                self.bvd_params['model_type'] = 'EBVD'
                
                logger.info("EBVD parameter calculation completed successfully")
                logger.info(f"Calculated EBVD parameters: R0={self.bvd_params.get('R0', 0):.2f} Ohm, R1={self.bvd_params.get('R1', 0):.2f} Ohm")
                if 'R2' in self.bvd_params:
                    logger.info(f"Harmonic branch: R2={self.bvd_params['R2']:.2f} Ohm, L2={self.bvd_params['L2']*1e3:.4f} mH, C2={self.bvd_params['C2']*1e9:.4f} nF")
            
                # Update group title
                self.params_group.setTitle("Calculated EBVD Parameters")
                
                # Update table for EBVD (has R0 and optionally R2, L2, C2)
                param_data = [
                    ('C₀', f"{self.bvd_params['C0']*1e9:.2f}", 'nF'),
                    ('R₀', f"{self.bvd_params.get('R0', 0):.2f}", 'Ohm'),
                    ('fs', f"{self.bvd_params['fs']*1e-3:.4f}", 'kHz'),
                    ('fp', f"{self.bvd_params['fp']*1e-3:.4f}", 'kHz'),
                    ('R₁', f"{self.bvd_params['R1']:.2f}", 'Ohm'),
                    ('L₁', f"{self.bvd_params['L1']*1e3:.4f}", 'mH'),
                    ('C₁', f"{self.bvd_params['C1']*1e9:.4f}", 'nF'),
                ]
                
                # Add harmonic branch if present
                if 'R2' in self.bvd_params:
                    param_data.extend([
                        ('R₂', f"{self.bvd_params['R2']:.2f}", 'Ohm'),
                        ('L₂', f"{self.bvd_params['L2']*1e3:.4f}", 'mH'),
                        ('C₂', f"{self.bvd_params['C2']*1e9:.4f}", 'nF'),
                    ])
                
                param_data.extend([
                    ('Qm', f"{self.bvd_params['Qm']:.2f}", ''),
                    ('k', f"{self.bvd_params['k']:.4f}", ''),
                    ('tan δ', f"{self.bvd_params.get('tan_delta', 0.0):.6f}", '')
                ])
                
                self.bvd_params_table.setRowCount(len(param_data))
                for i, (name, val, unit) in enumerate(param_data):
                    self.bvd_params_table.setItem(i, 0, QTableWidgetItem(name))
                    self.bvd_params_table.setItem(i, 1, QTableWidgetItem(val))
                    self.bvd_params_table.setItem(i, 2, QTableWidgetItem(unit))
            elif use_mbvd:
                logger.info("Starting MBVD (Modified BVD) parameter calculation...")
                logger.info(f"Input parameters: C0={C0_nF} nF, fs={fs_kHz} kHz (initial)")
                logger.info(f"Conductance data: {len(g_values)} points, range: {g_values.min():.4f} - {g_values.max():.4f} mS")
                logger.info(f"Susceptance data: {len(b_values)} points, range: {b_values.min():.4f} - {b_values.max():.4f} mS")
                
                self.bvd_params = calculate_mbvd_parameters(
                    freq_g, g_values_S, freq_b, b_values_S, C0
                )
                
                # Ensure model_type is set
                self.bvd_params['model_type'] = 'MBVD'
                
                logger.info("MBVD parameter calculation completed successfully")
                logger.info(f"Calculated MBVD parameters: R0={self.bvd_params['R0']:.2f} Ohm, R1={self.bvd_params['R1']:.2f} Ohm")
                
                # Update group title
                self.params_group.setTitle("Calculated MBVD Parameters")
                
                # Update table for MBVD (has R0)
                param_data = [
                    ('C₀', f"{self.bvd_params['C0']*1e9:.2f}", 'nF'),
                    ('R₀', f"{self.bvd_params['R0']:.2f}", 'Ohm'),
                    ('fs', f"{self.bvd_params['fs']*1e-3:.4f}", 'kHz'),
                    ('fp', f"{self.bvd_params['fp']*1e-3:.4f}", 'kHz'),
                    ('R₁', f"{self.bvd_params['R1']:.2f}", 'Ohm'),
                    ('L₁', f"{self.bvd_params['L1']*1e3:.4f}", 'mH'),
                    ('C₁', f"{self.bvd_params['C1']*1e9:.4f}", 'nF'),
                    ('Qm', f"{self.bvd_params['Qm']:.2f}", ''),
                    ('k', f"{self.bvd_params['k']:.4f}", ''),
                    ('tan δ', f"{self.bvd_params.get('tan_delta', 0.0):.6f}", '')
                ]
                self.bvd_params_table.setRowCount(len(param_data))
                for i, (name, val, unit) in enumerate(param_data):
                    self.bvd_params_table.setItem(i, 0, QTableWidgetItem(name))
                    self.bvd_params_table.setItem(i, 1, QTableWidgetItem(val))
                    self.bvd_params_table.setItem(i, 2, QTableWidgetItem(unit))
            else:
                logger.info("Starting BVD parameter calculation...")
                logger.info(f"Input parameters: C0={C0_nF} nF, fs={fs_kHz} kHz (initial)")
                logger.info(f"Conductance data: {len(g_values)} points, range: {g_values.min():.4f} - {g_values.max():.4f} mS")
                logger.info(f"Susceptance data: {len(b_values)} points, range: {b_values.min():.4f} - {b_values.max():.4f} mS")
                
                self.bvd_params = calculate_bvd_parameters(
                    freq_g, g_values_S, freq_b, b_values_S, C0
                )
                self.bvd_params['model_type'] = 'BVD'
                
                logger.info("BVD parameter calculation completed successfully")
                
                # Update group title
                self.params_group.setTitle("Calculated BVD Parameters")
                
                # Update table for BVD
                param_data = [
                    ('C₀', f"{self.bvd_params['C0']*1e9:.2f}", 'nF'),
                    ('fs', f"{self.bvd_params['fs']*1e-3:.4f}", 'kHz'),
                    ('fp', f"{self.bvd_params['fp']*1e-3:.4f}", 'kHz'),
                    ('R₁', f"{self.bvd_params['R1']:.2f}", 'Ohm'),
                    ('L₁', f"{self.bvd_params['L1']*1e3:.4f}", 'mH'),
                    ('C₁', f"{self.bvd_params['C1']*1e9:.4f}", 'nF'),
                    ('Qm', f"{self.bvd_params['Qm']:.2f}", ''),
                    ('k', f"{self.bvd_params['k']:.4f}", '')
                ]
                self.bvd_params_table.setRowCount(len(param_data))
                for i, (name, val, unit) in enumerate(param_data):
                    self.bvd_params_table.setItem(i, 0, QTableWidgetItem(name))
                    self.bvd_params_table.setItem(i, 1, QTableWidgetItem(val))
                    self.bvd_params_table.setItem(i, 2, QTableWidgetItem(unit))
            
            # Log model creation
            if use_klm:
                model_name = "KLM"
            elif use_mason:
                model_name = "Mason"
            elif use_ebvd:
                model_name = "EBVD"
            elif use_mbvd:
                model_name = "MBVD"
            else:
                model_name = "BVD"
            logger.info(f"=== {model_name} Model Created Successfully ===")
            logger.info(f"Model type: {self.bvd_params.get('model_type', 'BVD')}")
            if use_klm:
                logger.info(f"KLM model: k_t={self.bvd_params.get('k_t', 0):.4f}, t={self.bvd_params.get('t', 0)*1e3:.4f} mm")
                logger.info(f"KLM acoustic load: Z_load={self.bvd_params.get('Z_load', 0):.2e} kg/(m²·s)")
            elif use_mason:
                logger.info(f"Mason model: k_t={self.bvd_params.get('k_t', 0):.4f}, t={self.bvd_params.get('t', 0)*1e3:.4f} mm")
            elif use_ebvd:
                logger.info(f"EBVD model: R0={self.bvd_params.get('R0', 0):.2f} Ohm, R1={self.bvd_params.get('R1', 0):.2f} Ohm")
                if 'R2' in self.bvd_params:
                    logger.info(f"EBVD harmonic: R2={self.bvd_params['R2']:.2f} Ohm")
            elif use_mbvd:
                logger.info(f"MBVD model: R0={self.bvd_params.get('R0', 0):.2f} Ohm, R1={self.bvd_params.get('R1', 0):.2f} Ohm")
        
            # Build model curves (pass values in mS for plotting)
            self.plotBVDComparison(freq_g, g_values, freq_b, b_values)
            
            # Enable BVD tab
            self.tabs.setTabEnabled(4, True)
            self.tabs.setCurrentIndex(4)
            
            # Show success message
            model_desc = {
                'BVD': 'Basic BVD (4 parameters)',
                'MBVD': 'Modified BVD with dielectric losses (5 parameters)',
                'EBVD': 'Extended BVD with harmonic branches (7+ parameters)',
                'Mason': 'Physical model based on acoustic waves (physical parameters)',
                'KLM': 'Transformer-coupled model for hydroacoustic transducers (physical parameters)'
            }
            desc = model_desc.get(self.bvd_params.get('model_type', 'BVD'), model_name)
            QMessageBox.information(
                self, f"{model_name} Model Created", 
                f"{model_name} model has been successfully created from your data!\n\n"
                f"Model: {desc}\n"
                f"Parameters are displayed in the table above.\n"
                f"Check the graphs below to see the model fit quality.\n\n"
                f"Model type: {self.bvd_params.get('model_type', 'BVD')}"
            )
            
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Parameter error: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"BVD calculation error: {e}")
    
    def plotBVDComparison(self, freq_g, g_exp, freq_b, b_exp):
        """Plot comparison of experimental and model data
        Args:
            freq_g: frequency array for conductance (Hz)
            g_exp: conductance values (mS)
            freq_b: frequency array for susceptance (Hz)
            b_exp: susceptance values (mS)
        """
        
        # Generate frequency grid
        freq_model = np.linspace(min(freq_g.min(), freq_b.min()), 
                                 max(freq_g.max(), freq_b.max()), 500)
        
        # Calculate model curves using the selected model
        model_type = self.bvd_params.get('model_type', 'BVD')
        logger.info(f"Plotting model curves using {model_type} model")
        
        if model_type == 'KLM':
            logger.info(f"Using KLM model with k_t={self.bvd_params.get('k_t', 0):.4f}, t={self.bvd_params.get('t', 0)*1e3:.4f} mm")
            logger.info(f"KLM acoustic load: Z_load={self.bvd_params.get('Z_load', 0):.2e} kg/(m²·s)")
            model_curves = calculate_model_curves_klm(freq_model, self.bvd_params)
        elif model_type == 'Mason':
            logger.info(f"Using Mason model with k_t={self.bvd_params.get('k_t', 0):.4f}, t={self.bvd_params.get('t', 0)*1e3:.4f} mm")
            model_curves = calculate_model_curves_mason(freq_model, self.bvd_params)
        elif model_type == 'EBVD':
            logger.info(f"Using EBVD model with R0={self.bvd_params.get('R0', 0):.2f} Ohm, R1={self.bvd_params['R1']:.2f} Ohm")
            if 'R2' in self.bvd_params:
                logger.info(f"EBVD harmonic branch: R2={self.bvd_params['R2']:.2f} Ohm")
            model_curves = calculate_model_curves_ebvd(freq_model, self.bvd_params)
        elif model_type == 'MBVD' and 'R0' in self.bvd_params:
            logger.info(f"Using MBVD model with R0={self.bvd_params['R0']:.2f} Ohm, R1={self.bvd_params['R1']:.2f} Ohm")
            model_curves = calculate_model_curves_mbvd(freq_model, self.bvd_params)
        else:
            logger.info(f"Using BVD model with R1={self.bvd_params['R1']:.2f} Ohm")
            model_curves = calculate_model_curves(freq_model, self.bvd_params)
        
        g_model = model_curves['g']
        b_model = model_curves['b']
        y_mag_model = model_curves['magnitude']
        y_phase_model = model_curves['phase']
        freq_model_kHz = model_curves['freq']
        
        # Experimental magnitude and phase
        # Interpolate susceptance to conductance frequencies
        # Convert to S for calculations, then back to mS
        g_exp_S = g_exp * 1e-3  # mS -> S
        b_exp_S = b_exp * 1e-3  # mS -> S
        b_exp_interp = np.interp(freq_g, freq_b, b_exp_S)  # Interpolate in S
        Y_exp_g = g_exp_S + 1j * b_exp_interp
        y_mag_exp = np.abs(Y_exp_g) * 1e3  # S -> mS
        y_phase_exp = np.angle(Y_exp_g, deg=True)  # degrees
        
        # === Plot 1: Conductance ===
        self.bvd_plot_g.clear()
        self.bvd_plot_g.plot(
            freq_g * 1e-3, g_exp,  # g_exp already in mS
            pen=None, symbol='o', symbolBrush=(0, 100, 255, 150), symbolSize=6,
            name='PDF data (fitted)'
        )
        self.bvd_plot_g.plot(
            freq_model_kHz, g_model,
            pen=pg.mkPen((255, 0, 0), width=2),
            name='BVD model'
        )
        
        # Mark resonance
        fs_kHz = self.bvd_params['fs'] * 1e-3
        self.bvd_plot_g.addLine(x=fs_kHz, pen=pg.mkPen('g', style=Qt.PenStyle.DashLine, width=2))
        
        # === Plot 2: Susceptance ===
        self.bvd_plot_b.clear()
        self.bvd_plot_b.plot(
            freq_b * 1e-3, b_exp,  # b_exp already in mS
            pen=None, symbol='o', symbolBrush=(0, 100, 255, 150), symbolSize=6,
            name='PDF data (fitted)'
        )
        self.bvd_plot_b.plot(
            freq_model_kHz, b_model,
            pen=pg.mkPen((255, 0, 0), width=2),
            name='BVD model'
        )
        
        # Mark resonance and antiresonance
        fp_kHz = self.bvd_params['fp'] * 1e-3
        self.bvd_plot_b.addLine(x=fs_kHz, pen=pg.mkPen('g', style=Qt.PenStyle.DashLine, width=2))
        self.bvd_plot_b.addLine(x=fp_kHz, pen=pg.mkPen('m', style=Qt.PenStyle.DashLine, width=2))
        self.bvd_plot_b.addLine(y=0, pen=pg.mkPen('k', style=Qt.PenStyle.DotLine, width=1))
        
        # === Plot 3: Magnitude |Y| ===
        self.bvd_plot_mag.clear()
        self.bvd_plot_mag.plot(
            freq_g * 1e-3, y_mag_exp,
            pen=None, symbol='o', symbolBrush=(0, 100, 255, 150), symbolSize=6,
            name='PDF data (G²+B²)^0.5'
        )
        self.bvd_plot_mag.plot(
            freq_model_kHz, y_mag_model,
            pen=pg.mkPen((255, 0, 0), width=2),
            name='BVD model'
        )
        
        # Mark fs (|Y| maximum) and fp (|Y| minimum)
        self.bvd_plot_mag.addLine(x=fs_kHz, pen=pg.mkPen('g', style=Qt.PenStyle.DashLine, width=2))
        self.bvd_plot_mag.addLine(x=fp_kHz, pen=pg.mkPen('m', style=Qt.PenStyle.DashLine, width=2))
        
        # Add annotations
        text_fs = pg.TextItem(f"fs={fs_kHz:.2f} kHz", color='g', anchor=(0, 1))
        text_fs.setPos(fs_kHz, y_mag_model.max() * 0.9)
        self.bvd_plot_mag.addItem(text_fs)
        
        text_fp = pg.TextItem(f"fp={fp_kHz:.2f} kHz", color='m', anchor=(0, 1))
        text_fp.setPos(fp_kHz, y_mag_model.min() * 1.5)
        self.bvd_plot_mag.addItem(text_fp)
        
        # === Plot 4: Phase ===
        self.bvd_plot_phase.clear()
        self.bvd_plot_phase.plot(
            freq_g * 1e-3, y_phase_exp,
            pen=None, symbol='o', symbolBrush=(0, 100, 255, 150), symbolSize=6,
            name='PDF data'
        )
        self.bvd_plot_phase.plot(
            freq_model_kHz, y_phase_model,
            pen=pg.mkPen((255, 0, 0), width=2),
            name='BVD model'
        )
        
        # Mark 0° and ±90°
        self.bvd_plot_phase.addLine(y=0, pen=pg.mkPen('k', style=Qt.PenStyle.DotLine, width=1))
        self.bvd_plot_phase.addLine(y=90, pen=pg.mkPen('gray', style=Qt.PenStyle.DotLine, width=1))
        self.bvd_plot_phase.addLine(y=-90, pen=pg.mkPen('gray', style=Qt.PenStyle.DotLine, width=1))
        
        self.bvd_plot_phase.addLine(x=fs_kHz, pen=pg.mkPen('g', style=Qt.PenStyle.DashLine, width=2))
        self.bvd_plot_phase.addLine(x=fp_kHz, pen=pg.mkPen('m', style=Qt.PenStyle.DashLine, width=2))
        
        # === Fit quality assessment ===
        from scipy.interpolate import interp1d
        
        g_model_interp = interp1d(freq_model, g_model, 
                                  bounds_error=False, fill_value='extrapolate')
        b_model_interp = interp1d(freq_model, b_model,
                                  bounds_error=False, fill_value='extrapolate')
        
        g_model_at_exp = g_model_interp(freq_g)
        b_model_at_exp = b_model_interp(freq_b)
        
        # RMSE (both g_exp and g_model_at_exp are already in mS)
        rmse_g = np.sqrt(np.mean((g_exp - g_model_at_exp)**2))
        rmse_b = np.sqrt(np.mean((b_exp - b_model_at_exp)**2))
        
        # R² (coefficient of determination)
        ss_res_g = np.sum((g_exp - g_model_at_exp)**2)
        ss_tot_g = np.sum((g_exp - np.mean(g_exp))**2)
        r2_g = 1 - (ss_res_g / ss_tot_g) if ss_tot_g > 0 else 0
        
        ss_res_b = np.sum((b_exp - b_model_at_exp)**2)
        ss_tot_b = np.sum((b_exp - np.mean(b_exp))**2)
        r2_b = 1 - (ss_res_b / ss_tot_b) if ss_tot_b > 0 else 0
        
        # Maximum deviations
        max_error_g = np.max(np.abs(g_exp - g_model_at_exp))
        max_error_b = np.max(np.abs(b_exp - b_model_at_exp))
        
        # Mean relative errors
        mean_rel_error_g = np.mean(np.abs((g_exp - g_model_at_exp) / (np.abs(g_exp) + 1e-10))) * 100
        mean_rel_error_b = np.mean(np.abs((b_exp - b_model_at_exp) / (np.abs(b_exp) + 1e-10))) * 100
        
        # Log metrics to file and console
        logger.info("=" * 60)
        logger.info("BVD Model Fit Quality Metrics")
        logger.info("=" * 60)
        logger.info(f"Conductance (G) Metrics:")
        logger.info(f"  RMSE: {rmse_g:.6f} mS")
        logger.info(f"  R²: {r2_g:.6f}")
        logger.info(f"  Max Error: {max_error_g:.6f} mS")
        logger.info(f"  Mean Relative Error: {mean_rel_error_g:.2f}%")
        logger.info(f"Susceptance (B) Metrics:")
        logger.info(f"  RMSE: {rmse_b:.6f} mS")
        logger.info(f"  R²: {r2_b:.6f}")
        logger.info(f"  Max Error: {max_error_b:.6f} mS")
        logger.info(f"  Mean Relative Error: {mean_rel_error_b:.2f}%")
        avg_r2 = (r2_g + r2_b) / 2
        logger.info(f"Average R²: {avg_r2:.6f}")
        logger.info(f"BVD Parameters:")
        logger.info(f"  C0: {self.bvd_params['C0']*1e9:.4f} nF")
        logger.info(f"  fs: {self.bvd_params['fs']*1e-3:.4f} kHz")
        logger.info(f"  fp: {self.bvd_params['fp']*1e-3:.4f} kHz")
        logger.info(f"  R1: {self.bvd_params['R1']:.4f} Ohm")
        logger.info(f"  L1: {self.bvd_params['L1']*1e3:.4f} mH")
        logger.info(f"  C1: {self.bvd_params['C1']*1e9:.4f} nF")
        logger.info(f"  Qm: {self.bvd_params['Qm']:.4f}")
        logger.info(f"  k: {self.bvd_params['k']:.6f}")
        logger.info(f"  Δf: {(self.bvd_params['fp']*1e-3 - self.bvd_params['fs']*1e-3):.4f} kHz")
        logger.info("=" * 60)
        
        # Display detailed metrics
        quality_text = f"""
        <b style='font-size: 16px;'>📊 BVD Model Fit Quality</b><br><br>
        
        <table style='width: 100%; border-collapse: collapse;'>
        <tr style='background: #e3f2fd;'>
            <th style='padding: 8px; border: 1px solid #ddd;'>Metric</th>
            <th style='padding: 8px; border: 1px solid #ddd;'>Conductance (G)</th>
            <th style='padding: 8px; border: 1px solid #ddd;'>Susceptance (B)</th>
        </tr>
        <tr>
            <td style='padding: 8px; border: 1px solid #ddd;'><b>RMSE (mS)</b></td>
            <td style='padding: 8px; border: 1px solid #ddd;'>{rmse_g:.4f}</td>
            <td style='padding: 8px; border: 1px solid #ddd;'>{rmse_b:.4f}</td>
        </tr>
        <tr style='background: #f5f5f5;'>
            <td style='padding: 8px; border: 1px solid #ddd;'><b>R² (coefficient of determination)</b></td>
            <td style='padding: 8px; border: 1px solid #ddd;'>{r2_g:.4f}</td>
            <td style='padding: 8px; border: 1px solid #ddd;'>{r2_b:.4f}</td>
        </tr>
        <tr>
            <td style='padding: 8px; border: 1px solid #ddd;'><b>Max error (mS)</b></td>
            <td style='padding: 8px; border: 1px solid #ddd;'>{max_error_g:.4f}</td>
            <td style='padding: 8px; border: 1px solid #ddd;'>{max_error_b:.4f}</td>
        </tr>
        <tr style='background: #f5f5f5;'>
            <td style='padding: 8px; border: 1px solid #ddd;'><b>Mean rel. error (%)</b></td>
            <td style='padding: 8px; border: 1px solid #ddd;'>{mean_rel_error_g:.2f}%</td>
            <td style='padding: 8px; border: 1px solid #ddd;'>{mean_rel_error_b:.2f}%</td>
        </tr>
        </table>
        <br>
        """
        
        # Overall assessment
        avg_r2 = (r2_g + r2_b) / 2
        if avg_r2 > 0.95:
            quality_text += "<span style='color: green; font-size: 16px; font-weight: bold;'>✓ Excellent match! Model accurately describes the transducer.</span>"
        elif avg_r2 > 0.85:
            quality_text += "<span style='color: orange; font-size: 16px; font-weight: bold;'>⚠ Acceptable match. Small errors possible.</span>"
        else:
            quality_text += "<span style='color: red; font-size: 16px; font-weight: bold;'>✗ Poor match. Check extracted data or C₀ and fs parameters.</span>"
        
        quality_text += f"""
        <br><br>
        <b>Key frequencies:</b><br>
        • fs (resonance) = {fs_kHz:.3f} kHz - |Y| maximum, phase ≈ 0°<br>
        • fp (antiresonance) = {fp_kHz:.3f} kHz - |Y| minimum, B = 0<br>
        • Δf = {(fp_kHz - fs_kHz):.3f} kHz ({((fp_kHz/fs_kHz - 1)*100):.2f}%)
        """
        
        self.fit_quality_label.setText(quality_text)
    
    def exportBVDParams(self):
        """Export BVD parameters"""
        if not hasattr(self, 'bvd_params'):
            QMessageBox.warning(self, "Error", "Calculate BVD parameters first")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save BVD Parameters", "", "JSON Files (*.json);;Text Files (*.txt)"
        )
        
        if filename:
            try:
                export_data = {
                    'C0_nF': self.bvd_params['C0'] * 1e9,
                    'fs_kHz': self.bvd_params['fs'] * 1e-3,
                    'fp_kHz': self.bvd_params['fp'] * 1e-3,
                    'R1_Ohm': self.bvd_params['R1'],
                    'L1_mH': self.bvd_params['L1'] * 1e3,
                    'C1_nF': self.bvd_params['C1'] * 1e9,
                    'Qm': self.bvd_params['Qm'],
                    'k': self.bvd_params['k']
                }
                
                if filename.endswith('.json'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2)
                else:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("BVD Transducer Parameters\n")
                        f.write("=" * 40 + "\n\n")
                        for key, val in export_data.items():
                            f.write(f"{key}: {val}\n")
                
                QMessageBox.information(self, "Success", "BVD parameters exported")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")
    
    def exportForSimulator(self):
        """Export BVD/MBVD parameters in SonarCore simulator format"""
        if not hasattr(self, 'bvd_params') or not self.bvd_params:
            QMessageBox.warning(
                self, "Error", 
                "Please calculate model parameters first.\n"
                "Go to TAB 5 and click 'Calculate Model'."
            )
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Transducer for Simulator", "", 
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if not filename:
            return
        
        try:
            # Get model type
            model_type = self.bvd_params.get('model_type', 'BVD')
            
            # Get basic parameters
            fs_kHz = self.bvd_params['fs'] * 1e-3
            fp_kHz = self.bvd_params['fp'] * 1e-3
            
            # Estimate frequency range
            f_min = fs_kHz * 0.8  # 80% of fs
            f_max = fp_kHz * 1.2  # 120% of fp
            f_0 = fs_kHz  # Center frequency
            
            # Create simulator-compatible JSON
            transducer_data = {
                "model": f"Extracted {model_type} Model",
                "f_min": f_min * 1000,  # Convert to Hz
                "f_max": f_max * 1000,
                "f_0": f_0 * 1000,
                "B_tr": (f_max - f_min) * 1000,
                "S_TX": 180,  # Default, should be measured
                "S_RX": -180,  # Default, should be measured
                "Theta_BW": 10,  # Default beam width
                "Q": self.bvd_params.get('Qm', 10.0),
                "T_rd": 0,  # Ring-down time
                "Z": 50,  # Default impedance
                "source": "BVD Extractor",
                "version": "1.0",
                
                # BVD/MBVD parameters for simulator
                "bvd": {
                    "R_s": self.bvd_params['R1'],  # Series resistance
                    "L_s": self.bvd_params['L1'],  # Series inductance (H)
                    "C_s": self.bvd_params['C1'],  # Series capacitance (F)
                    "C_p": self.bvd_params['C0'],  # Parallel capacitance (F)
                    "k": self.bvd_params.get('k', 0.0)  # Coupling coefficient
                },
                
                "resonance": {
                    "f_s": fs_kHz * 1000,  # Series resonance (Hz)
                    "f_p": fp_kHz * 1000   # Parallel resonance (Hz)
                }
            }
            
            # Add MBVD-specific parameters if available
            if model_type == 'MBVD' and 'R0' in self.bvd_params:
                transducer_data["bvd"]["R_p"] = self.bvd_params['R0']  # Parallel loss resistance
                transducer_data["bvd"]["tan_delta"] = self.bvd_params.get('tan_delta', 0.0)
            
            # Write JSON file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(transducer_data, f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(
                self, "Success", 
                f"Transducer model exported successfully!\n\n"
                f"Model type: {model_type}\n"
                f"File: {filename}\n\n"
                f"This file can be used in SonarCore simulator.\n"
                f"Place it in data/transducers/ directory."
            )
            
            logger.info(f"Exported transducer model for simulator: {filename}")
            logger.info(f"Model type: {model_type}, fs={fs_kHz:.2f} kHz, fp={fp_kHz:.2f} kHz")
            
        except Exception as e:
            logger.error(f"Failed to export for simulator: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to export: {e}")
    
    def createCalibrationTab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Instructions
        info = QLabel("""
        <b>STEP 2: Coordinate System Setup</b><br>
        <b style='color: #f44336;'>⚠ Required step before data extraction!</b><br><br>
        Click on 3 points in order:<br>
        1. <span style='color: #2196f3;'>●</span> Origin (set X₀, Y₀ values below)<br>
        2. <span style='color: #4caf50;'>●</span> X-axis end (known point)<br>
        3. <span style='color: #4caf50;'>●</span> Y-axis end (known point)
        """)
        info.setStyleSheet("padding: 15px; background: #e3f2fd; border-radius: 8px;")
        layout.addWidget(info)
        
        # Image for calibration
        self.calib_image = ImageWidget()
        self.calib_image.click_callback = self.onCalibClick
        self.calib_image.setMinimumSize(600, 400)
        self.calib_image.calibration_mode = True  # Enable calibration mode
        self.calib_image.setCursor(Qt.CursorShape.CrossCursor)  # Set cross cursor by default
        layout.addWidget(self.calib_image)
        
        # Value input fields
        input_layout = QVBoxLayout()
        
        # Axis start values
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("X₀ (X-axis start):"))
        self.input_x_min = QLineEdit("0")
        self.input_x_min.setPlaceholderText("e.g.: 20")
        self.input_x_min.textChanged.connect(self.checkAxisInputs)
        start_layout.addWidget(self.input_x_min)
        
        start_layout.addWidget(QLabel("Y₀ (Y-axis start):"))
        self.input_y_min = QLineEdit("0")
        self.input_y_min.setPlaceholderText("e.g.: -10")
        self.input_y_min.textChanged.connect(self.checkAxisInputs)
        start_layout.addWidget(self.input_y_min)
        input_layout.addLayout(start_layout)
        
        # Axis end values
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("X at X-axis end:"))
        self.input_x_max = QLineEdit()
        self.input_x_max.setPlaceholderText("e.g.: 50")
        self.input_x_max.textChanged.connect(self.checkAxisInputs)
        end_layout.addWidget(self.input_x_max)
        
        end_layout.addWidget(QLabel("Y at Y-axis end:"))
        self.input_y_max = QLineEdit()
        self.input_y_max.setPlaceholderText("e.g.: 100")
        self.input_y_max.textChanged.connect(self.checkAxisInputs)
        end_layout.addWidget(self.input_y_max)
        input_layout.addLayout(end_layout)
        
        layout.addLayout(input_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        # Zoom controls
        self.btn_calib_zoom = QPushButton("🔍 Zoom (select area)")
        self.btn_calib_zoom.setCheckable(True)
        self.btn_calib_zoom.clicked.connect(lambda checked: self.calib_image.setSelectionMode(checked))
        self.calib_image.zoom_button = self.btn_calib_zoom  # Link button to widget
        btn_layout.addWidget(self.btn_calib_zoom)
        
        btn_zoom_reset = QPushButton("↺ Reset zoom")
        btn_zoom_reset.clicked.connect(self.calib_image.resetZoom)
        btn_layout.addWidget(btn_zoom_reset)
        
        btn_reset_calib = QPushButton("🔄 Reset points")
        btn_reset_calib.clicked.connect(self.resetCalibration)
        btn_layout.addWidget(btn_reset_calib)
        
        btn_layout.addStretch()
        
        self.btn_calib_next = QPushButton("Next ► (Extract points)")
        self.btn_calib_next.clicked.connect(self.moveToExtraction)
        self.btn_calib_next.setEnabled(False)
        btn_layout.addWidget(self.btn_calib_next)
        
        layout.addLayout(btn_layout)
        
        return widget
    
    def createExtractionTab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Instructions
        info = QLabel("""
        <b>STEP 3: Data Point Extraction</b><br>
        <b style='color: #ff9800;'>Available only after calibration (Step 2)</b><br><br>
        Method 1: Click along the graph line from left to right (minimum 3 points)<br>
        Method 2: Use automatic color-based extraction
        """)
        info.setStyleSheet("padding: 15px; background: #e3f2fd; border-radius: 8px;")
        layout.addWidget(info)
        
        # Function name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Function name:"))
        self.input_func_name = QLineEdit("Function 1")
        name_layout.addWidget(self.input_func_name)
        layout.addLayout(name_layout)
        
        # Automatic extraction
        auto_group = QGroupBox("⚡ Automatic Color-Based Extraction")
        auto_layout = QHBoxLayout()
        
        self.btn_pick_color = QPushButton("🎨 Pick Graph Color")
        self.btn_pick_color.clicked.connect(self.startColorPicking)
        auto_layout.addWidget(self.btn_pick_color)
        
        self.color_display = QLabel("Color not selected")
        self.color_display.setMinimumWidth(150)
        self.color_display.setStyleSheet("padding: 5px; border: 1px solid #ccc; background: #f5f5f5;")
        auto_layout.addWidget(self.color_display)
        
        auto_layout.addWidget(QLabel("Tolerance:"))
        self.color_tolerance = QLineEdit("30")
        self.color_tolerance.setMaximumWidth(50)
        self.color_tolerance.setToolTip("Color comparison tolerance (0-255)")
        auto_layout.addWidget(self.color_tolerance)
        
        self.btn_auto_extract = QPushButton("🤖 Auto Extract Points")
        self.btn_auto_extract.clicked.connect(self.autoExtractPoints)
        self.btn_auto_extract.setEnabled(False)
        auto_layout.addWidget(self.btn_auto_extract)
        
        auto_group.setLayout(auto_layout)
        layout.addWidget(auto_group)
        
        # Image for data extraction
        self.extract_image = ImageWidget()
        self.extract_image.click_callback = self.onDataClick
        self.extract_image.setMinimumSize(600, 400)
        layout.addWidget(self.extract_image)
        
        # Point counter
        self.label_point_count = QLabel("Points collected: 0")
        self.label_point_count.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.label_point_count)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        # Zoom controls
        self.btn_extract_zoom = QPushButton("🔍 Zoom (select area)")
        self.btn_extract_zoom.setCheckable(True)
        self.btn_extract_zoom.clicked.connect(lambda checked: self.extract_image.setSelectionMode(checked))
        self.extract_image.zoom_button = self.btn_extract_zoom  # Link button to widget
        btn_layout.addWidget(self.btn_extract_zoom)
        
        btn_zoom_reset = QPushButton("↺ Reset zoom")
        btn_zoom_reset.clicked.connect(self.extract_image.resetZoom)
        btn_layout.addWidget(btn_zoom_reset)
        
        btn_undo = QPushButton("↶ Undo")
        btn_undo.clicked.connect(self.undoPoint)
        btn_layout.addWidget(btn_undo)
        
        btn_clear = QPushButton("🗑️ Clear")
        btn_clear.clicked.connect(self.clearPoints)
        btn_layout.addWidget(btn_clear)
        
        btn_layout.addStretch()
        
        self.btn_extract_next = QPushButton("Next ►")
        self.btn_extract_next.clicked.connect(self.processExtractedData)
        self.btn_extract_next.setEnabled(False)
        btn_layout.addWidget(self.btn_extract_next)
        
        layout.addLayout(btn_layout)
        
        return widget
    
    def createResultsTab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Splitter for plot and table
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Container for plot and settings
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        # Fitting settings
        fitting_group = QGroupBox("📊 Fitting Options")
        fitting_layout = QVBoxLayout()
        
        # First row: Fitting type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Fitting type:"))
        
        from PyQt6.QtWidgets import QComboBox
        self.fitting_type_combo = QComboBox()
        self.fitting_type_combo.addItems(["Spline", "Polynomial"])
        self.fitting_type_combo.currentTextChanged.connect(self.onFittingTypeChanged)
        type_layout.addWidget(self.fitting_type_combo)
        
        # Polynomial degree (only visible for polynomial)
        type_layout.addWidget(QLabel("  Polynomial degree:"))
        self.poly_degree_spin = QSpinBox()
        self.poly_degree_spin.setMinimum(1)
        self.poly_degree_spin.setMaximum(20)
        self.poly_degree_spin.setValue(3)
        self.poly_degree_spin.setEnabled(False)  # Disabled by default (spline selected)
        self.poly_degree_spin.valueChanged.connect(self.onFittingParamsChanged)
        type_layout.addWidget(self.poly_degree_spin)
        
        type_layout.addStretch()
        fitting_layout.addLayout(type_layout)
        
        # Second row: Spline smoothing (only visible for spline)
        spline_layout = QHBoxLayout()
        spline_layout.addWidget(QLabel("Spline smoothing (0=exact, 1.0=strong):"))
        
        from PyQt6.QtWidgets import QSlider, QDoubleSpinBox
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setMinimum(0)
        self.smoothing_slider.setMaximum(100)  # 0-100 for slider, will convert to 0.0-1.0
        self.smoothing_slider.setValue(0)
        self.smoothing_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.smoothing_slider.setTickInterval(10)
        self.smoothing_slider.valueChanged.connect(self.updateSmoothing)
        spline_layout.addWidget(self.smoothing_slider)
        
        self.smoothing_value_label = QLabel("0.00")
        self.smoothing_value_label.setMinimumWidth(50)
        self.smoothing_value_label.setStyleSheet("font-weight: bold; padding: 5px;")
        spline_layout.addWidget(self.smoothing_value_label)
        
        spline_layout.addWidget(QLabel("  Fine tuning:"))
        self.smoothing_fine = QLineEdit("0.0")
        self.smoothing_fine.setMaximumWidth(80)
        self.smoothing_fine.setToolTip("Smoothing value: 0.0 (exact fit) to 1.0 (strong smoothing)")
        self.smoothing_fine.returnPressed.connect(self.updateSmoothingFromInput)
        spline_layout.addWidget(self.smoothing_fine)
        
        btn_apply_smoothing = QPushButton("Apply")
        btn_apply_smoothing.clicked.connect(self.updateSmoothingFromInput)
        spline_layout.addWidget(btn_apply_smoothing)
        
        btn_reset_smoothing = QPushButton("Reset (0.0)")
        btn_reset_smoothing.clicked.connect(lambda: self.smoothing_slider.setValue(0))
        spline_layout.addWidget(btn_reset_smoothing)
        
        spline_layout.addStretch()
        fitting_layout.addLayout(spline_layout)
        
        fitting_group.setLayout(fitting_layout)
        plot_layout.addWidget(fitting_group)
        
        # Function management
        functions_group = QGroupBox("📋 Function Management")
        functions_layout = QHBoxLayout()
        
        # Function list
        from PyQt6.QtWidgets import QListWidget
        self.functions_list = QListWidget()
        self.functions_list.setMaximumHeight(100)
        self.functions_list.itemSelectionChanged.connect(self.onFunctionSelected)
        functions_layout.addWidget(self.functions_list)
        
        # Management buttons
        func_buttons_layout = QVBoxLayout()
        
        btn_rename_func = QPushButton("✏️ Rename")
        btn_rename_func.clicked.connect(self.renameFunction)
        func_buttons_layout.addWidget(btn_rename_func)
        
        btn_delete_func = QPushButton("🗑️ Delete")
        btn_delete_func.clicked.connect(self.deleteFunction)
        func_buttons_layout.addWidget(btn_delete_func)
        
        btn_toggle_func = QPushButton("👁️ Show/Hide")
        btn_toggle_func.clicked.connect(self.toggleFunction)
        func_buttons_layout.addWidget(btn_toggle_func)
        
        func_buttons_layout.addStretch()
        functions_layout.addLayout(func_buttons_layout)
        
        functions_group.setLayout(functions_layout)
        plot_layout.addWidget(functions_group)
        
        # BVD function assignment
        bvd_assignment_group = QGroupBox("🔬 BVD Model Assignment")
        bvd_assignment_layout = QVBoxLayout()
        
        # Conductance function selection
        conductance_layout = QHBoxLayout()
        conductance_layout.addWidget(QLabel("Conductance (G) function:"))
        self.conductance_combo_tab4 = QComboBox()
        self.conductance_combo_tab4.setMinimumWidth(200)
        self.conductance_combo_tab4.currentIndexChanged.connect(self.onBVDAssignmentChanged)
        conductance_layout.addWidget(self.conductance_combo_tab4)
        conductance_layout.addStretch()
        bvd_assignment_layout.addLayout(conductance_layout)
        
        # Susceptance function selection
        susceptance_layout = QHBoxLayout()
        susceptance_layout.addWidget(QLabel("Susceptance (B) function:"))
        self.susceptance_combo_tab4 = QComboBox()
        self.susceptance_combo_tab4.setMinimumWidth(200)
        self.susceptance_combo_tab4.currentIndexChanged.connect(self.onBVDAssignmentChanged)
        susceptance_layout.addWidget(self.susceptance_combo_tab4)
        susceptance_layout.addStretch()
        bvd_assignment_layout.addLayout(susceptance_layout)
        
        bvd_assignment_group.setLayout(bvd_assignment_layout)
        plot_layout.addWidget(bvd_assignment_group)
        
        # Single button in TAB 4: Go to TAB 5 for model calculation
        btn_go_to_tab5 = QPushButton("→ Go to TAB 5 (Model Calculation)")
        btn_go_to_tab5.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                padding: 12px;
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        btn_go_to_tab5.setToolTip(
            "After assigning Conductance and Susceptance functions above,\n"
            "click here to go to TAB 5 where you will:\n"
            "1. Select model type (BVD/MBVD)\n"
            "2. Enter transducer parameters (C₀, fs)\n"
            "3. Click 'Create Model' to calculate the model from your data."
        )
        btn_go_to_tab5.clicked.connect(lambda: self.tabs.setCurrentIndex(4))  # Go to TAB 5
        plot_layout.addWidget(btn_go_to_tab5)
        
        # Results plot
        self.result_plot = pg.PlotWidget()
        self.result_plot.setBackground('w')
        self.result_plot.setLabel('left', 'Value')
        self.result_plot.setLabel('bottom', 'Frequency (kHz)')
        self.result_plot.showGrid(x=True, y=True, alpha=0.3)
        plot_layout.addWidget(self.result_plot)
        
        splitter.addWidget(plot_container)
        
        # Data table - dynamic columns for each function
        self.result_table = QTableWidget()
        # Will set columns dynamically based on number of functions
        splitter.addWidget(self.result_table)
        
        layout.addWidget(splitter)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        btn_import_csv = QPushButton("📥 Import CSV")
        btn_import_csv.clicked.connect(self.importCSV)
        btn_layout.addWidget(btn_import_csv)
        
        btn_import_json = QPushButton("📥 Import JSON")
        btn_import_json.clicked.connect(self.importJSON)
        btn_layout.addWidget(btn_import_json)
        
        btn_export_csv = QPushButton("💾 Export CSV")
        btn_export_csv.clicked.connect(self.exportCSV)
        btn_layout.addWidget(btn_export_csv)
        
        btn_export_json = QPushButton("💾 Export JSON")
        btn_export_json.clicked.connect(self.exportJSON)
        btn_layout.addWidget(btn_export_json)
        
        btn_export_simulator = QPushButton("🚀 Export for Simulator")
        btn_export_simulator.setToolTip(
            "Export BVD/MBVD parameters in format compatible with SonarCore simulator.\n"
            "Creates a transducer JSON file that can be used in the simulator."
        )
        btn_export_simulator.clicked.connect(self.exportForSimulator)
        btn_layout.addWidget(btn_export_simulator)
        
        btn_layout.addStretch()
        
        btn_new_func = QPushButton("➕ New Function")
        btn_new_func.clicked.connect(self.startNewFunction)
        btn_layout.addWidget(btn_new_func)
        
        btn_reset = QPushButton("🔄 Start Over")
        btn_reset.clicked.connect(self.reset)
        btn_layout.addWidget(btn_reset)
        
        layout.addLayout(btn_layout)
        
        return widget
    
    def loadImage(self):
        if PDF_AVAILABLE:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Select Image or PDF", "", 
                "All Files (*.png *.jpg *.jpeg *.bmp *.pdf);;Images (*.png *.jpg *.jpeg *.bmp);;PDF Files (*.pdf);;All Files (*.*)"
            )
        else:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", 
                "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)"
            )
        if filename:
            # Determine file type
            if filename.lower().endswith('.pdf'):
                try:
                    zoom = float(self.pdf_zoom_input.text()) if hasattr(self, 'pdf_zoom_input') else 3.0
                    zoom = max(1.0, min(5.0, zoom))
                except:
                    zoom = 3.0
                self.loadPDFFile(filename, zoom)
            else:
                self.loadImageFile(filename)
    
    def onTabChanged(self, index):
        """Handle tab change - update RX/TX graphs when TAB 6 is selected"""
        if index == 5 and hasattr(self, 'rx_plot') and hasattr(self, 'tx_plot'):
            self.updateRXTXGraphs()
    
    def loadImageFile(self, filename):
        """Load image"""
        # Ask if user wants to keep existing functions
        keep_existing = False
        if self.all_functions:
            reply = QMessageBox.question(
                self, "Load Image",
                f"Found {len(self.all_functions)} existing function(s).\n"
                "Keep existing functions and add new data, or replace everything?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            keep_existing = (reply == QMessageBox.StandardButton.Yes)
        
        # Close PDF if open (only if not keeping existing data)
        if not keep_existing and self.pdf_doc is not None:
            self.pdf_doc.close()
            self.pdf_doc = None
            self.pdf_path = None
            # Disable PDF page selection
            if PDF_AVAILABLE and hasattr(self, 'pdf_page_spinbox'):
                self.pdf_page_spinbox.setEnabled(False)
                self.pdf_page_spinbox.setMaximum(1)
                self.pdf_page_spinbox.setValue(1)
                if hasattr(self, 'pdf_page_total_label'):
                    self.pdf_page_total_label.setText("of 1")
        
        pixmap = QPixmap(filename)
        self.preview_label.setPixmap(pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        
        self.calib_image.setImage(pixmap)
        self.extract_image.setImage(pixmap)
        
        # Reset calibration and data points for new image
        # But keep existing functions if requested
        self.coord_points = []
        self.data_points = []
        self.calibration = None
        self.extracted_data = []
        self.calib_image.clearPoints()
        self.extract_image.clearPoints()
        
        # Clear functions only if not keeping existing
        if not keep_existing:
            self.all_functions = []
            logger.info("Cleared all existing functions (replacing with new image)")
        else:
            logger.info(f"Keeping {len(self.all_functions)} existing function(s), adding new image data")
        
        # Set calibration mode and cross cursor
        self.calib_image.calibration_mode = True
        self.calib_image.setCursor(Qt.CursorShape.CrossCursor)
        
        # Enable only calibration
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)
        self.tabs.setTabEnabled(4, False)
        self.tabs.setCurrentIndex(1)
    
    def loadPDFFile(self, filename, zoom_factor=3.0):
        """Load PDF file with adjustable zoom"""
        if not PDF_AVAILABLE:
            QMessageBox.warning(self, "Error", "PyMuPDF is not installed. Install: pip install PyMuPDF")
            return
        
        try:
            # Close previous PDF if open
            if self.pdf_doc is not None:
                self.pdf_doc.close()
            
            # Open new PDF document
            self.pdf_doc = fitz.open(filename)
            self.pdf_path = filename
            self.pdf_total_pages = len(self.pdf_doc)
            
            # Set initial page
            if self.pdf_total_pages > 1:
                from PyQt6.QtWidgets import QInputDialog
                page_num, ok = QInputDialog.getInt(
                    self, "Page Selection", 
                    f"Document contains {self.pdf_total_pages} pages.\nSelect page to load:",
                    1, 1, self.pdf_total_pages, 1
                )
                if not ok:
                    self.pdf_doc.close()
                    self.pdf_doc = None
                    self.pdf_path = None
                    return
                self.pdf_current_page = page_num - 1  # Convert to 0-index
            else:
                self.pdf_current_page = 0
            
            # Update page selection UI
            if PDF_AVAILABLE and hasattr(self, 'pdf_page_spinbox'):
                self.pdf_page_spinbox.setMaximum(self.pdf_total_pages)
                self.pdf_page_spinbox.setValue(self.pdf_current_page + 1)
                self.pdf_page_spinbox.setEnabled(True)
                self.pdf_page_total_label.setText(f"of {self.pdf_total_pages}")
            
            # Load the selected page
            self.loadPDFPage(self.pdf_current_page, zoom_factor)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load PDF: {e}")
            if self.pdf_doc is not None:
                self.pdf_doc.close()
                self.pdf_doc = None
                self.pdf_path = None
    
    def loadPDFPage(self, page_index, zoom_factor=3.0):
        """Load a specific page from the currently open PDF"""
        if not PDF_AVAILABLE or self.pdf_doc is None:
            return
        
        try:
            if page_index < 0 or page_index >= self.pdf_total_pages:
                return
            
            self.pdf_current_page = page_index
            page = self.pdf_doc[page_index]
            
            # Convert to high resolution image
            mat = fitz.Matrix(zoom_factor, zoom_factor)  # Adjustable scale for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to QPixmap
            img_data = pix.tobytes("png")
            qimg = QImage.fromData(img_data)
            pixmap = QPixmap.fromImage(qimg)
            
            # Update preview
            self.preview_label.setPixmap(pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
            # Update calibration and extraction images
            self.calib_image.setImage(pixmap)
            self.extract_image.setImage(pixmap)
            
            # Clear previous calibration and data points when switching pages
            # Ask if user wants to keep existing functions
            keep_existing = False
            if self.all_functions:
                reply = QMessageBox.question(
                    self, "Load PDF Page",
                    f"Found {len(self.all_functions)} existing function(s).\n"
                    "Keep existing functions and add new data, or replace everything?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                keep_existing = (reply == QMessageBox.StandardButton.Yes)
            
            self.coord_points = []
            self.data_points = []
            self.calibration = None
            self.extracted_data = []
            
            # Clear functions only if not keeping existing
            if not keep_existing:
                self.all_functions = []
                logger.info("Cleared all existing functions (replacing with new PDF page)")
            else:
                logger.info(f"Keeping {len(self.all_functions)} existing function(s), adding new PDF page data")
            
            # Enable tabs
            self.tabs.setTabEnabled(1, True)
            self.tabs.setTabEnabled(2, False)
            self.tabs.setTabEnabled(3, False)
            self.tabs.setTabEnabled(4, False)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load PDF page: {e}")
    
    def onPDFPageChanged(self, page_num):
        """Handle PDF page selection change"""
        if self.pdf_doc is None:
            return
        
        page_index = page_num - 1  # Convert from 1-based to 0-based
        if page_index == self.pdf_current_page:
            return  # Already on this page
        
        try:
            zoom = float(self.pdf_zoom_input.text()) if hasattr(self, 'pdf_zoom_input') else 3.0
            zoom = max(1.0, min(5.0, zoom))  # Limit 1.0-5.0
        except:
            zoom = 3.0
        
        self.loadPDFPage(page_index, zoom)
    
    def loadPDFFromButton(self):
        """Load PDF via button with adjustable zoom"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select PDF", "", "PDF Files (*.pdf)"
        )
        if filename:
            try:
                zoom = float(self.pdf_zoom_input.text())
                zoom = max(1.0, min(5.0, zoom))  # Limit 1.0-5.0
            except:
                zoom = 3.0
            self.loadPDFFile(filename, zoom)
    
    def loadPDF(self):
        """Deprecated method - now uses loadImage with auto-detection"""
        self.loadPDFFromButton()
    
    def onCalibClick(self, x, y):
        if len(self.coord_points) < 3:
            self.coord_points.append((x, y))
            self.calib_image.addPoint(x, y)
            
            if len(self.coord_points) == 3:
                # Calibration complete - disable calibration mode
                self.calib_image.calibration_mode = False
                self.calib_image.setCursor(Qt.CursorShape.ArrowCursor)
                self.calculateCalibration()
                # Automatically set zoom to graph area in extraction tab
                self.setExtractionZoomToGraphArea()
    
    def checkAxisInputs(self):
        """Check axis field completion and automatic recalculation of calibration"""
        # If already have 3 points and all fields filled, recalculate calibration
        if len(self.coord_points) == 3:
            try:
                # Check that all fields are filled with valid numbers
                float(self.input_x_min.text())
                float(self.input_y_min.text())
                float(self.input_x_max.text())
                float(self.input_y_max.text())
                
                # All values entered - recalculate calibration
                self.calculateCalibration()
            except ValueError:
                # Not all fields filled correctly - disable button
                self.btn_calib_next.setEnabled(False)
    
    def calculateCalibration(self):
        try:
            x_min = float(self.input_x_min.text())
            y_min = float(self.input_y_min.text())
            x_max = float(self.input_x_max.text())
            y_max = float(self.input_y_max.text())
            
            origin = self.coord_points[0]
            x_end = self.coord_points[1]
            y_end = self.coord_points[2]
            
            # Convert coordinates to float
            origin_x = float(origin[0])
            origin_y = float(origin[1])
            x_end_x = float(x_end[0])
            y_end_y = float(y_end[1])
            
            # Calibration: pixels -> real coordinates
            # X axis: from x_min to x_max
            # Y axis: from y_min to y_max (inverted)
            self.calibration = {
                'origin': (origin_x, origin_y),
                'x_scale': (x_max - x_min) / (x_end_x - origin_x),
                'y_scale': (y_max - y_min) / (origin_y - y_end_y),  # Y inverted
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max
            }
            
            self.btn_calib_next.setEnabled(True)
            self.tabs.setTabEnabled(2, True)
            
        except ValueError:
            self.btn_calib_next.setEnabled(False)
    
    def moveToExtraction(self):
        """Move to data extraction with readiness check"""
        if not self.calibration:
            QMessageBox.warning(
                self, "Calibration Not Complete",
                "First place 3 calibration points and enter axis values!"
            )
            return
        self.tabs.setCurrentIndex(2)
    
    def setExtractionZoomToGraphArea(self):
        """Automatic zoom to graph area in data extraction tab"""
        if not self.coord_points or len(self.coord_points) < 3:
            return
        
        origin = self.coord_points[0]
        x_end = self.coord_points[1]
        y_end = self.coord_points[2]
        
        # Define graph area rectangle with small margin
        margin = 10  # pixels margin
        
        x1 = origin[0] - margin
        y1 = y_end[1] - margin
        x2 = x_end[0] + margin
        y2 = origin[1] + margin
        
        # Set zoom
        self.extract_image.zoom_rect = (x1, y1, x2, y2)
        self.extract_image.updateDisplay()
    
    def resetCalibration(self):
        self.coord_points = []
        self.calibration = None
        self.calib_image.clearPoints()
        self.btn_calib_next.setEnabled(False)
        
        # Set calibration mode and cross cursor for new calibration
        self.calib_image.calibration_mode = True
        self.calib_image.setCursor(Qt.CursorShape.CrossCursor)
        
        # Disable next tabs when resetting calibration
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)
        self.tabs.setTabEnabled(4, False)
    
    def onDataClick(self, x, y):
        if not self.calibration:
            QMessageBox.warning(
                self, "Sequence Error",
                "First complete coordinate calibration in step 2!"
            )
            return
            
        self.data_points.append((x, y))
        self.extract_image.addPoint(x, y)
        self.label_point_count.setText(f"Points collected: {len(self.data_points)}")
        
        if len(self.data_points) >= 3:
            self.btn_extract_next.setEnabled(True)
    
    def undoPoint(self):
        if self.data_points:
            self.data_points.pop()
            self.extract_image.removeLastPoint()
            self.label_point_count.setText(f"Points collected: {len(self.data_points)}")
            
            if len(self.data_points) < 3:
                self.btn_extract_next.setEnabled(False)
    
    def clearPoints(self):
        self.data_points = []
        self.extract_image.clearPoints()
        self.label_point_count.setText("Points collected: 0")
        self.btn_extract_next.setEnabled(False)
    
    def startColorPicking(self):
        """Start graph color selection"""
        self.extract_image.color_callback = self.onColorPicked
        self.extract_image.startColorPicking()
        self.btn_pick_color.setText("👆 Click on graph line (within axes)...")
    
    def onColorPicked(self, color):
        """Handle selected color"""
        r, g, b = color
        self.color_display.setText(f"RGB({r}, {g}, {b})")
        self.color_display.setStyleSheet(f"padding: 5px; border: 2px solid rgb({r},{g},{b}); background: rgb({r},{g},{b}); color: {'white' if (r+g+b) < 384 else 'black'};")
        self.btn_pick_color.setText("🎨 Pick Graph Color")
        self.btn_auto_extract.setEnabled(True)
        self.extract_image.target_color = color
    
    def autoExtractPoints(self):
        """Automatic point extraction by color"""
        if not self.extract_image.target_color or not self.calibration:
            QMessageBox.warning(self, "Error", "First select graph color and calibrate coordinates")
            return
        
        try:
            tolerance = int(self.color_tolerance.text())
        except:
            tolerance = 30
        
        # Get image
        image = self.extract_image.original_pixmap.toImage()
        target_r, target_g, target_b = self.extract_image.target_color
        
        # Define search area - ONLY within coordinate axes
        origin = self.coord_points[0]
        x_end = self.coord_points[1]
        y_end = self.coord_points[2]
        
        # Graph area (between calibration points)
        search_x1 = int(origin[0])
        search_y1 = int(y_end[1])
        search_x2 = int(x_end[0])
        search_y2 = int(origin[1])
        
        # Dictionary to store points by X coordinate
        points_by_x = {}
        
        # Scan image ONLY within axes
        for x in range(search_x1, search_x2):
            for y in range(search_y1, search_y2):
                color = QColor(image.pixel(x, y))
                r, g, b = color.red(), color.green(), color.blue()
                
                # Check color match with tolerance
                if (abs(r - target_r) <= tolerance and 
                    abs(g - target_g) <= tolerance and 
                    abs(b - target_b) <= tolerance):
                    
                    # Save average Y for each X
                    if x not in points_by_x:
                        points_by_x[x] = []
                    points_by_x[x].append(y)
        
        if not points_by_x:
            QMessageBox.warning(self, "Not Found", "No points found with selected color within coordinate axes. Try increasing tolerance or selecting different color.")
            return
        
        # Clear old points
        self.data_points = []
        self.extract_image.clearPoints()
        
        # Average Y for each X and add points
        temp_points = []
        for x in sorted(points_by_x.keys()):
            y_avg = sum(points_by_x[x]) / len(points_by_x[x])
            temp_points.append((float(x), float(y_avg)))
        
        # Thin points (take every Nth)
        if len(temp_points) > 500:
            step = len(temp_points) // 500
            temp_points = [temp_points[i] for i in range(0, len(temp_points), step)]
        
        # Add points to data_points and to image
        for x, y in temp_points:
            self.data_points.append((x, y))
            self.extract_image.addPoint(x, y)
        
        self.label_point_count.setText(f"Points collected: {len(self.data_points)} (automatic)")
        self.btn_extract_next.setEnabled(True)
        
        QMessageBox.information(self, "Success", f"Automatically extracted {len(self.data_points)} points from graph area!")
    
    def pixelToReal(self, px, py):
        """Convert pixel coordinates to real coordinates"""
        if not self.calibration:
            return None, None
            
        origin = self.calibration['origin']
        origin_x = float(origin[0])
        origin_y = float(origin[1])
        
        x = float(self.calibration['x_min']) + (float(px) - origin_x) * float(self.calibration['x_scale'])
        y = float(self.calibration['y_min']) + (origin_y - float(py)) * float(self.calibration['y_scale'])
        
        return x, y
    
    def processExtractedData(self):
        """Process extracted points and build spline"""
        if len(self.data_points) < 3:
            QMessageBox.warning(self, "Error", "Need at least 3 points")
            return
        
        # Convert to real coordinates
        real_points = []
        try:
            for i, point in enumerate(self.data_points):
                # Ensure point is a tuple of two numbers
                if isinstance(point, (tuple, list)) and len(point) == 2:
                    px, py = float(point[0]), float(point[1])
                else:
                    print(f"Skipping invalid point {i}: {point}, type: {type(point)}")
                    continue
                    
                x, y = self.pixelToReal(px, py)
                if x is not None and y is not None:
                    real_points.append((float(x), float(y)))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error converting coordinates: {str(e)}\nError type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return
        
        if len(real_points) < 3:
            QMessageBox.warning(self, "Error", f"Insufficient valid points after conversion: {len(real_points)}")
            return
        
        # Sort by X
        real_points.sort(key=lambda p: p[0])
        
        try:
            x_data = np.array([float(p[0]) for p in real_points], dtype=np.float64)
            y_data = np.array([float(p[1]) for p in real_points], dtype=np.float64)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating data arrays: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        # Create initial fitting (spline by default)
        try:
            # By default no smoothing, user will adjust in results
            smoothing = 0.0
            
            spline = UnivariateSpline(x_data, y_data, s=smoothing, k=min(3, len(x_data)-1))
            
            # Generate dense grid for visualization
            x_dense = np.linspace(x_data.min(), x_data.max(), 200)
            y_dense = spline(x_dense)
            
            # Save data
            self.extracted_data = list(zip(x_dense, y_dense))
            
            # Save function
            func_name = str(self.input_func_name.text())
            
            # Save Y range from TAB 2 for this function
            y_range_tab2 = None
            if hasattr(self, 'input_y_min') and hasattr(self, 'input_y_max'):
                y_min_text = self.input_y_min.text().strip()
                y_max_text = self.input_y_max.text().strip()
                if y_min_text and y_max_text:
                    try:
                        y_range_tab2 = (float(y_min_text), float(y_max_text))
                    except ValueError:
                        pass
            
            self.all_functions.append({
                'name': func_name,
                'data': self.extracted_data.copy(),
                'original_points': real_points,
                'visible': True,  # Visibility flag
                'fitting_type': 'spline',  # Default fitting type
                'smoothing': 0.0,  # Spline smoothing parameter (0.0-1.0)
                'poly_degree': 3,  # Polynomial degree (if polynomial fitting)
                'y_range_tab2': y_range_tab2  # Y range from TAB 2 for this function
            })
            
            # Update UI to match first function's fitting type
            if len(self.all_functions) == 1:
                self.fitting_type_combo.setCurrentText('Spline')
                self.smoothing_slider.setValue(0)
                self.smoothing_fine.setText("0.0")
                self.smoothing_value_label.setText("0.00")
            
            # Show results
            self.showResults()
            self.tabs.setTabEnabled(3, True)
            # Enable TAB 6 (RX/TX) if there are functions
            if hasattr(self, 'tab_rxtx'):
                self.tabs.setTabEnabled(5, True)
            self.tabs.setCurrentIndex(3)
            
            # Update BVD function assignment lists in TAB 4
            self.updateBVDAssignmentLists()
            
            # Update BVD function lists if BVD tab exists
            if hasattr(self, 'conductance_combo'):
                self.updateBVDFunctionLists()
            
            QMessageBox.information(
                self, "Data Extracted", 
                f"Extracted {len(x_data)} points!\nGo to step 4 to view results."
            )
            
        except Exception as e:
            error_msg = f"Error building spline: {str(e)}\nType: {type(e).__name__}"
            QMessageBox.critical(self, "Error", error_msg)
            import traceback
            traceback.print_exc()
    
    def onFunctionSelected(self):
        """Handle function selection from list"""
        current_row = self.functions_list.currentRow()
        if current_row >= 0 and current_row < len(self.all_functions):
            func = self.all_functions[current_row]
            # Update UI to match selected function's fitting parameters
            fitting_type = func.get('fitting_type', 'spline')
            if fitting_type == 'polynomial':
                self.fitting_type_combo.setCurrentText('Polynomial')
                self.poly_degree_spin.setValue(func.get('poly_degree', 3))
            else:
                self.fitting_type_combo.setCurrentText('Spline')
                smoothing = func.get('smoothing', 0.0)
                self.smoothing_slider.setValue(int(smoothing * 100))
                self.smoothing_fine.setText(f"{smoothing:.2f}")
                self.smoothing_value_label.setText(f"{smoothing:.2f}")
    
    def renameFunction(self):
        """Rename selected function"""
        current_row = self.functions_list.currentRow()
        if current_row < 0 or current_row >= len(self.all_functions):
            QMessageBox.warning(self, "Error", "Select function to rename")
            return
        
        from PyQt6.QtWidgets import QInputDialog
        old_name = self.all_functions[current_row]['name']
        new_name, ok = QInputDialog.getText(
            self, "Rename Function",
            "Enter new name:",
            text=old_name
        )
        
        if ok and new_name:
            self.all_functions[current_row]['name'] = new_name
            self.showResults()
            
            # Update BVD assignment lists to reflect new name
            # This will preserve the selection based on bvd_type
            self.updateBVDAssignmentLists()
    
    def deleteFunction(self):
        """Delete selected function"""
        current_row = self.functions_list.currentRow()
        if current_row < 0 or current_row >= len(self.all_functions):
            QMessageBox.warning(self, "Error", "Select function to delete")
            return
        
        func_name = self.all_functions[current_row]['name']
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Delete function '{func_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            del self.all_functions[current_row]
            self.showResults()
            
            if not self.all_functions:
                QMessageBox.information(self, "Information", "All functions deleted. Can extract new data.")
    
    def updateBVDAssignmentLists(self):
        """Update BVD function assignment combo boxes in TAB 4"""
        if not hasattr(self, 'conductance_combo_tab4'):
            return
        
        if not self.all_functions:
            return
        
        # Clear existing items
        self.conductance_combo_tab4.clear()
        self.susceptance_combo_tab4.clear()
        
        # Add all functions
        for func in self.all_functions:
            func_name = func['name']
            self.conductance_combo_tab4.addItem(func_name)
            self.susceptance_combo_tab4.addItem(func_name)
        
        logger.info(f"Updating BVD assignment lists with {len(self.all_functions)} functions")
        for i, func in enumerate(self.all_functions):
            logger.info(f"  Function {i}: '{func['name']}' (bvd_type: {func.get('bvd_type')})")
        
        # Try to auto-select based on existing assignment first, but validate it matches the name
        conductance_selected = False
        susceptance_selected = False
        
        for i, func in enumerate(self.all_functions):
            name_lower = func['name'].lower().strip()
            bvd_type = func.get('bvd_type')
            
            # Only use existing assignment if it makes sense (name matches type)
            if bvd_type == 'conductance':
                # Validate: name should contain "conductance" or "g"
                if ('conductance' in name_lower or 
                    name_lower == 'g' or 
                    name_lower.startswith('g ') or
                    name_lower.endswith(' g') or
                    ' g ' in name_lower or
                    name_lower.startswith('g(') or
                    name_lower.endswith('(g)') or
                    '(g)' in name_lower.lower()):
                    self.conductance_combo_tab4.setCurrentIndex(i)
                    conductance_selected = True
                    logger.info(f"Using existing assignment: '{func['name']}' as Conductance")
                else:
                    # Invalid assignment, clear it
                    func['bvd_type'] = None
                    logger.info(f"Clearing invalid assignment: '{func['name']}' was marked as Conductance but name doesn't match")
            elif bvd_type == 'susceptance':
                # Validate: name should contain "susceptance" or "b"
                if ('susceptance' in name_lower or 
                    name_lower == 'b' or 
                    name_lower.startswith('b ') or
                    name_lower.endswith(' b') or
                    ' b ' in name_lower or
                    name_lower.startswith('b(') or
                    name_lower.endswith('(b)') or
                    '(b)' in name_lower.lower()):
                    self.susceptance_combo_tab4.setCurrentIndex(i)
                    susceptance_selected = True
                    logger.info(f"Using existing assignment: '{func['name']}' as Susceptance")
                else:
                    # Invalid assignment, clear it
                    func['bvd_type'] = None
                    logger.info(f"Clearing invalid assignment: '{func['name']}' was marked as Susceptance but name doesn't match")
        
        # Fallback to name-based selection if no assignment exists
        # Temporarily block signals to avoid conflicts during auto-assignment
        self.conductance_combo_tab4.blockSignals(True)
        self.susceptance_combo_tab4.blockSignals(True)
        
        # Check for Conductance first
        if not conductance_selected:
            for i, func in enumerate(self.all_functions):
                name = func['name']
                name_lower = name.lower().strip()
                # Check if name contains "conductance" (case-insensitive)
                # Also check for "g" as a word (not just a letter)
                if ('conductance' in name_lower or 
                    name_lower == 'g' or 
                    name_lower.startswith('g ') or
                    name_lower.endswith(' g') or
                    ' g ' in name_lower or
                    name_lower.startswith('g(') or
                    name_lower.endswith('(g)') or
                    '(g)' in name_lower.lower()):
                    self.conductance_combo_tab4.setCurrentIndex(i)
                    conductance_selected = True
                    logger.info(f"Auto-assigned function '{name}' as Conductance (G)")
                    break
        
        # Check for Susceptance (independent check)
        if not susceptance_selected:
            for i, func in enumerate(self.all_functions):
                name = func['name']
                name_lower = name.lower().strip()
                # Check if name contains "susceptance" (case-insensitive)
                # Also check for "b" as a word (not just a letter)
                if ('susceptance' in name_lower or 
                    name_lower == 'b' or 
                    name_lower.startswith('b ') or
                    name_lower.endswith(' b') or
                    ' b ' in name_lower or
                    name_lower.startswith('b(') or
                    name_lower.endswith('(b)') or
                    '(b)' in name_lower.lower()):
                    # Make sure we don't select the same function for both
                    conductance_idx = self.conductance_combo_tab4.currentIndex()
                    if i != conductance_idx:
                        self.susceptance_combo_tab4.setCurrentIndex(i)
                        susceptance_selected = True
                        logger.info(f"Auto-assigned function '{name}' as Susceptance (B)")
                        break
        
        # Re-enable signals
        self.conductance_combo_tab4.blockSignals(False)
        self.susceptance_combo_tab4.blockSignals(False)
        
        # Trigger assignment update to save the selection
        self.onBVDAssignmentChanged()
    
    def onBVDAssignmentChanged(self):
        """Handle BVD function assignment change in TAB 4"""
        # Store assignment in function data
        if hasattr(self, 'conductance_combo_tab4') and hasattr(self, 'susceptance_combo_tab4'):
            conductance_idx = self.conductance_combo_tab4.currentIndex()
            susceptance_idx = self.susceptance_combo_tab4.currentIndex()
            
            # Clear previous assignments
            for func in self.all_functions:
                func['bvd_type'] = None
            
            # Set new assignments
            if conductance_idx >= 0 and conductance_idx < len(self.all_functions):
                self.all_functions[conductance_idx]['bvd_type'] = 'conductance'
            
            if susceptance_idx >= 0 and susceptance_idx < len(self.all_functions):
                self.all_functions[susceptance_idx]['bvd_type'] = 'susceptance'
            
            # Enable TAB 5 if both functions are assigned
            if (conductance_idx >= 0 and susceptance_idx >= 0 and 
                conductance_idx < len(self.all_functions) and 
                susceptance_idx < len(self.all_functions) and
                conductance_idx != susceptance_idx):
                self.tabs.setTabEnabled(4, True)  # Enable TAB 5
            else:
                self.tabs.setTabEnabled(4, False)  # Disable TAB 5 if assignments are invalid conductance_idx >= 0 and conductance_idx < len(self.all_functions):
                self.all_functions[conductance_idx]['bvd_type'] = 'conductance'
            if susceptance_idx >= 0 and susceptance_idx < len(self.all_functions):
                self.all_functions[susceptance_idx]['bvd_type'] = 'susceptance'
    
    def toggleFunction(self):
        """Show/hide selected function"""
        current_row = self.functions_list.currentRow()
        if current_row < 0 or current_row >= len(self.all_functions):
            QMessageBox.warning(self, "Error", "Select function")
            return
        
        func = self.all_functions[current_row]
        func['visible'] = not func.get('visible', True)
        self.showResults()
    
    def updateSmoothing(self, value):
        """Update spline when slider changes"""
        # Convert slider value (0-100) to smoothing value (0.0-1.0)
        smoothing_value = value / 100.0
        self.smoothing_value_label.setText(f"{smoothing_value:.2f}")
        self.smoothing_fine.setText(f"{smoothing_value:.2f}")
        self.recalculateFitting()
    
    def updateSmoothingFromInput(self):
        """Update spline from text field"""
        try:
            value = float(self.smoothing_fine.text())
            value = max(0.0, min(1.0, value))  # Limit 0.0-1.0
            # Convert to slider value (0-100)
            slider_value = int(value * 100)
            self.smoothing_slider.setValue(slider_value)
            self.smoothing_value_label.setText(f"{value:.2f}")
            self.recalculateFitting()
        except ValueError:
            QMessageBox.warning(self, "Error", "Enter valid number (0.0 to 1.0)")
    
    def onFittingTypeChanged(self, fitting_type):
        """Handle fitting type change"""
        # Enable/disable controls based on fitting type
        if fitting_type == "Polynomial":
            self.smoothing_slider.setEnabled(False)
            self.smoothing_fine.setEnabled(False)
            self.poly_degree_spin.setEnabled(True)
        else:  # Spline
            self.smoothing_slider.setEnabled(True)
            self.smoothing_fine.setEnabled(True)
            self.poly_degree_spin.setEnabled(False)
        
        # Update all functions with new fitting type
        fitting_type_lower = fitting_type.lower()
        for func in self.all_functions:
            func['fitting_type'] = fitting_type_lower
            # Initialize default parameters if needed
            if fitting_type_lower == 'polynomial' and 'poly_degree' not in func:
                func['poly_degree'] = self.poly_degree_spin.value()
            elif fitting_type_lower == 'spline' and 'smoothing' not in func:
                func['smoothing'] = 0.0
        
        self.recalculateFitting()
    
    def onFittingParamsChanged(self):
        """Handle fitting parameter changes (polynomial degree)"""
        poly_degree = self.poly_degree_spin.value()
        for func in self.all_functions:
            if func.get('fitting_type', 'spline') == 'polynomial':
                func['poly_degree'] = poly_degree
        self.recalculateFitting()
    
    def recalculateFitting(self):
        """Recalculate fitting for all functions with current parameters"""
        if not self.all_functions:
            return
        
        # Get current fitting parameters from UI
        fitting_type = self.fitting_type_combo.currentText().lower()
        smoothing_value = float(self.smoothing_fine.text()) if self.smoothing_fine.isEnabled() else 0.0
        poly_degree = self.poly_degree_spin.value()
        
        for func in self.all_functions:
            # Get original points
            orig_points = func['original_points']
            x_data = np.array([float(p[0]) for p in orig_points], dtype=np.float64)
            y_data = np.array([float(p[1]) for p in orig_points], dtype=np.float64)
            
            # Update function parameters
            func['fitting_type'] = fitting_type
            if fitting_type == 'spline':
                func['smoothing'] = smoothing_value
            else:
                func['poly_degree'] = poly_degree
            
            # Recalculate fitting
            try:
                if fitting_type == 'spline':
                    # Spline fitting with smoothing (0.0 = exact, 1.0 = strong smoothing)
                    # Convert smoothing to appropriate scale for UnivariateSpline
                    # UnivariateSpline uses s parameter where larger = more smoothing
                    # We'll use a scale: s = smoothing * (max(y) - min(y))^2 * len(data)
                    y_range = y_data.max() - y_data.min()
                    s_param = smoothing_value * (y_range ** 2) * len(x_data) if y_range > 0 else 0
                    
                    spline = UnivariateSpline(x_data, y_data, s=s_param, k=min(3, len(x_data)-1))
                    x_dense = np.linspace(x_data.min(), x_data.max(), 200)
                    y_dense = spline(x_dense)
                else:  # polynomial
                    # Polynomial fitting
                    # Limit degree to number of points - 1
                    degree = min(poly_degree, len(x_data) - 1)
                    if degree < 1:
                        degree = 1
                    
                    # Fit polynomial
                    poly = Polynomial.fit(x_data, y_data, degree)
                    x_dense = np.linspace(x_data.min(), x_data.max(), 200)
                    y_dense = poly(x_dense)
                
                # Update function data
                func['data'] = list(zip(x_dense, y_dense))
            except Exception as e:
                print(f"Error recalculating fitting: {e}")
        
        # Update display
        self.showResults()
    
    
    def showResults(self):
        """Display results on plot with separate Y axes for each function"""
        # Update RX/TX graphs if TAB 6 exists
        if hasattr(self, 'rx_plot') and hasattr(self, 'tx_plot'):
            self.updateRXTXGraphs()
        
        plot_item = self.result_plot.getPlotItem()
        
        # Clear old additional axes
        for axis in self.extra_axes:
            plot_item.layout.removeItem(axis)
            if axis.scene():
                axis.scene().removeItem(axis)
        
        self.extra_axes = []
        
        # Clear old additional ViewBoxes
        for vb in self.extra_viewboxes:
            if vb.scene():
                vb.scene().removeItem(vb)
        
        self.extra_viewboxes = []
        
        # Clear main plot
        self.result_plot.clear()
        
        # Update function list
        self.functions_list.clear()
        for i, func in enumerate(self.all_functions):
            visible_mark = "✓" if func.get('visible', True) else "✗"
            self.functions_list.addItem(f"{visible_mark} {func['name']}")
        
        # Collect only visible functions
        visible_functions = [func for func in self.all_functions if func.get('visible', True)]
        
        if not visible_functions:
            return
        
        # Colors for functions
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
        
        # Check display mode (with fallback if checkbox not yet created)
        if hasattr(self, 'use_real_y_ranges'):
            use_real_ranges = self.use_real_y_ranges.isChecked()
        else:
            use_real_ranges = True  # Default: show real ranges
        
        # For each function calculate range
        y_ranges = []
        for func in visible_functions:
            data = func['data']
            y = [p[1] for p in data]
            y_min, y_max = min(y), max(y)
            y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
            y_range = (y_min - y_margin, y_max + y_margin)
            y_ranges.append(y_range)
        
        if use_real_ranges:
            # Mode 1: Scale functions using Y ranges from TAB 2
            # Y ranges are taken from TAB 2 (saved in y_range_tab2 for each function)
            
            # Use single ViewBox - all functions scaled to their ranges from TAB 2
            # But displayed in first function's range for visual alignment
            main_vb = plot_item.getViewBox()
            first_func = visible_functions[0]
            
            # Get first function's display range from TAB 2
            first_axis_y_min = 0.0
            first_axis_y_max = 1.0
            if first_func.get('y_range_tab2') is not None:
                first_axis_y_min, first_axis_y_max = first_func['y_range_tab2']
            else:
                # Fallback to data range
                first_data = first_func['data']
                if first_data:
                    first_y_values = [p[1] for p in first_data]
                    first_axis_y_min = min(first_y_values)
                    first_axis_y_max = max(first_y_values)
            
            for i, func in enumerate(visible_functions):
                original_idx = self.all_functions.index(func)
                data = func['data']
                x = [p[0] for p in data]
                y_real = [p[1] for p in data]  # Real Y values - use these directly
                
                color = colors[original_idx % len(colors)]
                
                # Fitting for function
                fitting_type = func.get('fitting_type', 'spline')
                fitting_label = 'polynomial' if fitting_type == 'polynomial' else 'spline'
                
                # Original points
                orig = func['original_points']
                ox = [p[0] for p in orig]
                oy_real = [p[1] for p in orig]  # Real Y values - use these directly
                
                # Get function's real data range
                y_func_min, y_func_max = min(y_real), max(y_real)
                if y_func_max == y_func_min:
                    y_func_max = y_func_min + 0.1
                
                # Get Y range from TAB 2 for this function (for axis display)
                func_name = func['name']
                axis_y_min = y_func_min
                axis_y_max = y_func_max
                if func.get('y_range_tab2') is not None:
                    axis_y_min, axis_y_max = func['y_range_tab2']
                
                # For 1:1 display, map values to first function's axis range
                # Get first function's axis range for mapping
                first_func = visible_functions[0]
                first_axis_y_min = 0.0
                first_axis_y_max = 1.0
                if first_func.get('y_range_tab2') is not None:
                    first_axis_y_min, first_axis_y_max = first_func['y_range_tab2']
                else:
                    # Fallback to data range
                    first_data = first_func['data']
                    if first_data:
                        first_y_values = [p[1] for p in first_data]
                        first_axis_y_min = min(first_y_values)
                        first_axis_y_max = max(first_y_values)
                
                if i == 0:
                    # First function - use real values directly (1:1 mapping)
                    y_scaled = y_real.copy()
                    oy_scaled = oy_real.copy()
                else:
                    # Map to first function's range: scaled = (real / axis_y_max) * first_axis_y_max
                    y_scaled = []
                    for y_val in y_real:
                        if axis_y_max > 0:
                            # Normalize to [0, 1] based on this function's axis
                            normalized = y_val / axis_y_max
                            # Map to first function's range
                            scaled = normalized * first_axis_y_max
                        else:
                            scaled = 0.0
                        y_scaled.append(scaled)
                    
                    oy_scaled = []
                    for y_val in oy_real:
                        if axis_y_max > 0:
                            normalized = y_val / axis_y_max
                            scaled = normalized * first_axis_y_max
                        else:
                            scaled = 0.0
                        oy_scaled.append(scaled)
                
                # Plot scaled data
                plot_item.plot(x, y_scaled, pen=pg.mkPen(color, width=2), name=f"{func['name']} ({fitting_label})")
                
                if len(orig) > 100:
                    color_with_alpha = color + (100,)
                    plot_item.plot(ox, oy_scaled, pen=None, symbol='o', 
                                 symbolBrush=color_with_alpha, symbolSize=3,
                                 name=f"{func['name']} (points, {len(orig)})")
                else:
                    plot_item.plot(ox, oy_scaled, pen=None, symbol='o', 
                                 symbolBrush=color, symbolSize=8,
                                 name=f"{func['name']} (points, {len(orig)})")
                
                # Create Y axis for this function
                # Get Y range from TAB 2 that was saved when this function was extracted
                # This will be shown on the axis (real values)
                axis_y_min = y_func_min
                axis_y_max = y_func_max
                if func.get('y_range_tab2') is not None:
                    axis_y_min, axis_y_max = func['y_range_tab2']
                
                if i == 0:
                    # First function uses left axis - show values from TAB 2 (1:1 mapping, no custom ticks needed)
                    left_axis = plot_item.getAxis('left')
                    left_axis.setLabel(f"{func['name']} [{axis_y_min:.3f}, {axis_y_max:.3f}]", color=color)
                    left_axis.setPen(color)
                else:
                    # Additional functions use right axes - show values from TAB 2
                    # Need custom ticks to map display positions (in first function's range) to real axis values
                    axis = pg.AxisItem('right')
                    plot_item.layout.addItem(axis, 2, 3 + i - 1)
                    self.extra_axes.append(axis)
                    
                    # Get first function's range for mapping
                    first_func = visible_functions[0]
                    first_axis_y_min = 0.0
                    first_axis_y_max = 1.0
                    if first_func.get('y_range_tab2') is not None:
                        first_axis_y_min, first_axis_y_max = first_func['y_range_tab2']
                    else:
                        first_data = first_func['data']
                        if first_data:
                            first_y_values = [p[1] for p in first_data]
                            first_axis_y_min = min(first_y_values)
                            first_axis_y_max = max(first_y_values)
                    
                    # Create custom ticks: map display positions to real axis values
                    num_ticks = 6
                    tick_positions = []
                    for j in range(num_ticks):
                        norm_pos = j / (num_ticks - 1) if num_ticks > 1 else 0
                        # Display position in first function's range
                        display_pos = first_axis_y_min + norm_pos * (first_axis_y_max - first_axis_y_min)
                        # Real axis value for this function
                        # scaled = (real / axis_y_max) * first_axis_y_max
                        # So: real = (scaled / first_axis_y_max) * axis_y_max
                        if first_axis_y_max > 0:
                            normalized = display_pos / first_axis_y_max
                            axis_value = normalized * axis_y_max
                        else:
                            axis_value = axis_y_min + norm_pos * (axis_y_max - axis_y_min)
                        tick_positions.append((display_pos, f'{axis_value:.3f}'))
                    
                    axis.setTicks([tick_positions])
                    axis.setLabel(f"{func['name']} [{axis_y_min:.3f}, {axis_y_max:.3f}]", color=color)
                    axis.setPen(color)
                    axis.linkToView(main_vb)
            
            # Set Y range to first function's axis range (1:1 mapping)
            # Get first function's axis range from TAB 2
            first_func = visible_functions[0]
            first_axis_y_min = 0.0
            first_axis_y_max = 1.0
            if first_func.get('y_range_tab2') is not None:
                first_axis_y_min, first_axis_y_max = first_func['y_range_tab2']
            else:
                # Fallback to data range
                first_data = first_func['data']
                if first_data:
                    first_y_values = [p[1] for p in first_data]
                    first_axis_y_min = min(first_y_values)
                    first_axis_y_max = max(first_y_values)
            
            # Set Y range to first function's axis range
            main_vb.setYRange(first_axis_y_min, first_axis_y_max, padding=0)
            main_vb.enableAutoRange(axis='x', enable=True)
            main_vb.enableAutoRange(axis='y', enable=False)  # Fixed Y range for 1:1 display
            # Trigger autozoom update for X axis only
            main_vb.autoRange(padding=0.05)
            
        else:
            # Mode 2: Scale all functions to first function's range (original behavior)
            # Draw first function on main Y axis (left) without normalization
            first_func = visible_functions[0]
            original_idx = self.all_functions.index(first_func)
        
            data = first_func['data']
            x = [p[0] for p in data]
            y = [p[1] for p in data]
        
            color = colors[original_idx % len(colors)]
        
            # Fitting for first function
            fitting_type = first_func.get('fitting_type', 'spline')
            fitting_label = 'polynomial' if fitting_type == 'polynomial' else 'spline'
            self.result_plot.plot(x, y, pen=pg.mkPen(color, width=2), name=f"{first_func['name']} ({fitting_label})")
        
            # Original points for first function
            orig = first_func['original_points']
            ox = [p[0] for p in orig]
            oy = [p[1] for p in orig]
        
            if len(orig) > 100:
                color_with_alpha = color + (100,)
                self.result_plot.plot(ox, oy, pen=None, symbol='o', 
                                      symbolBrush=color_with_alpha, symbolSize=3,
                                      name=f"{first_func['name']} (points, {len(orig)})")
            else:
                self.result_plot.plot(ox, oy, pen=None, symbol='o', 
                                      symbolBrush=color, symbolSize=8,
                                      name=f"{first_func['name']} (points, {len(orig)})")
        
            # Set Y range for first function
        plot_item.setYRange(y_ranges[0][0], y_ranges[0][1], padding=0)
        
            # Configure main Y axis on left
        left_axis = plot_item.getAxis('left')
        left_axis.setLabel(first_func['name'], color=color)
        left_axis.setPen(color)
        
            # Create custom ticks for left axis (first function)
        y_base_min, y_base_max = y_ranges[0]
        num_ticks = 6
        left_tick_positions = []
        for j in range(num_ticks):
            norm_pos = j / (num_ticks - 1)
            tick_value = y_base_min + norm_pos * (y_base_max - y_base_min)
            left_tick_positions.append((tick_value, f'{tick_value:.3f}'))
        left_axis.setTicks([left_tick_positions])
        
            # For remaining functions create additional right axes with scaling
        for i in range(1, len(visible_functions)):
            func = visible_functions[i]
            original_idx = self.all_functions.index(func)
            
            data = func['data']
            x = [p[0] for p in data]
            y_real = [p[1] for p in data]
            
                # Scale Y to first function range for display
            y_func_min, y_func_max = y_ranges[i]
            y_base_min, y_base_max = y_ranges[0]
            
                # Normalize Y of this function to first function range
            y_scaled = []
            for y_val in y_real:
                    # Normalize to [0, 1]
                normalized = (y_val - y_func_min) / (y_func_max - y_func_min) if y_func_max > y_func_min else 0.5
                    # Scale to first function range
                scaled = y_base_min + normalized * (y_base_max - y_base_min)
                y_scaled.append(scaled)
            
            color = colors[original_idx % len(colors)]
            
            # Draw scaled data
            fitting_type = func.get('fitting_type', 'spline')
            fitting_label = 'polynomial' if fitting_type == 'polynomial' else 'spline'
            self.result_plot.plot(x, y_scaled, pen=pg.mkPen(color, width=2), name=f"{func['name']} ({fitting_label})")
            
            # Original points
            orig = func['original_points']
            ox = [p[0] for p in orig]
            oy_real = [p[1] for p in orig]
            
            # Scale original points
            oy_scaled = []
            for y_val in oy_real:
                normalized = (y_val - y_func_min) / (y_func_max - y_func_min) if y_func_max > y_func_min else 0.5
                scaled = y_base_min + normalized * (y_base_max - y_base_min)
                oy_scaled.append(scaled)
            
            if len(orig) > 100:
                color_with_alpha = color + (100,)
                self.result_plot.plot(ox, oy_scaled, pen=None, symbol='o', 
                                      symbolBrush=color_with_alpha, symbolSize=3,
                                      name=f"{func['name']} (points, {len(orig)})")
            else:
                self.result_plot.plot(ox, oy_scaled, pen=None, symbol='o', 
                                      symbolBrush=color, symbolSize=8,
                                      name=f"{func['name']} (points, {len(orig)})")
            
            # Create additional Y axis on right
            axis = pg.AxisItem('right')
            plot_item.layout.addItem(axis, 2, 3 + i - 1)
            self.extra_axes.append(axis)
            
            # Create custom ticks for this axis
            # Generate ticks in scaled coordinates but with real values
            num_ticks = 5
            tick_positions = []
            for j in range(num_ticks):
                # Position in normalized space
                norm_pos = j / (num_ticks - 1)
                # Position in scaled coordinates (first function range)
                scaled_pos = y_base_min + norm_pos * (y_base_max - y_base_min)
                # Real value of this function
                real_value = y_func_min + norm_pos * (y_func_max - y_func_min)
                tick_positions.append((scaled_pos, f'{real_value:.3f}'))
            
            axis.setTicks([tick_positions])
            axis.setLabel(func['name'], color=color)
            axis.setPen(color)
            
            # Link axis to main ViewBox
            axis.linkToView(plot_item.vb)
        
        plot_item.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        plot_item.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        plot_item.setYRange(y_ranges[0][0], y_ranges[0][1], padding=0)
        plot_item.vb.setLimits(yMin=y_ranges[0][0], yMax=y_ranges[0][1])
        
        # Add legend
        plot_item.addLegend()
        
        
        # Fill table with ALL functions - each function in its own columns
        if self.all_functions:
            # Calculate table structure: each function gets 3 columns (#, Frequency, Value)
            num_functions = len(self.all_functions)
            num_columns = num_functions * 3
            self.result_table.setColumnCount(num_columns)
            
            # Set column headers
            headers = []
            for func in self.all_functions:
                headers.extend([f"#{func['name']}", f"Freq {func['name']}", f"Value {func['name']}"])
            self.result_table.setHorizontalHeaderLabels(headers)
            
            # Find maximum number of data points
            max_points = max(len(func['data']) for func in self.all_functions) if self.all_functions else 0
            
            # Add rows: 1 header row + 1 range row + data rows
            total_rows = 2 + max_points
            self.result_table.setRowCount(total_rows)
            
            # Row 0: Function headers
            for col_idx, func in enumerate(self.all_functions):
                col_start = col_idx * 3
                header_item = QTableWidgetItem(f"=== {func['name']} ===")
                header_item.setBackground(QColor(200, 220, 255))
                font = header_item.font()
                font.setBold(True)
                header_item.setFont(font)
                # Merge cells for header
                self.result_table.setItem(0, col_start, header_item)
                self.result_table.setSpan(0, col_start, 1, 3)
            
            # Row 1: Range info for each function
            for col_idx, func in enumerate(self.all_functions):
                col_start = col_idx * 3
                data = func['data']
                if data:
                    x_values = [p[0] for p in data]
                    y_values = [p[1] for p in data]
                    x_min, x_max = min(x_values), max(x_values)
                    y_min, y_max = min(y_values), max(y_values)
                    
                    range_item = QTableWidgetItem(f"Y=[{y_min:.6f}, {y_max:.6f}]")
                    range_item.setBackground(QColor(240, 240, 255))
                    font = range_item.font()
                    font.setItalic(True)
                    range_item.setFont(font)
                    self.result_table.setItem(1, col_start, range_item)
                    self.result_table.setSpan(1, col_start, 1, 3)
            
            # Rows 2+: Data points
            for row_idx in range(max_points):
                table_row = row_idx + 2
                for col_idx, func in enumerate(self.all_functions):
                    col_start = col_idx * 3
                    data = func['data']
                    
                    if row_idx < len(data):
                        x, y = data[row_idx]
                        self.result_table.setItem(table_row, col_start, QTableWidgetItem(str(row_idx + 1)))
                        self.result_table.setItem(table_row, col_start + 1, QTableWidgetItem(f"{x:.4f}"))
                        self.result_table.setItem(table_row, col_start + 2, QTableWidgetItem(f"{y:.6f}"))
                    else:
                        # Empty cells if function has fewer points
                        self.result_table.setItem(table_row, col_start, QTableWidgetItem(""))
                        self.result_table.setItem(table_row, col_start + 1, QTableWidgetItem(""))
                        self.result_table.setItem(table_row, col_start + 2, QTableWidgetItem(""))
            
            # Resize columns to fit content
            self.result_table.resizeColumnsToContents()
    
    def exportCSV(self):
        if not self.all_functions:
            QMessageBox.warning(self, "Error", "No data to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "", "CSV Files (*.csv)"
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    for func in self.all_functions:
                        writer.writerow([f"# {func['name']}"])
                        writer.writerow(['Frequency (kHz)', 'Value'])
                        
                        for x, y in func['data']:
                            writer.writerow([f"{x:.6f}", f"{y:.6f}"])
                        
                        writer.writerow([])
                
                QMessageBox.information(self, "Success", "Data exported to CSV")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")
    
    def exportJSON(self):
        if not self.all_functions:
            QMessageBox.warning(self, "Error", "No data to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save JSON", "", "JSON Files (*.json)"
        )
        
        if filename:
            try:
                export_data = []
                for func in self.all_functions:
                    func_data = {
                        'name': func['name'],
                        'data': [[x, y] for x, y in func['data']],
                        'original_points': [[x, y] for x, y in func.get('original_points', [])],
                        'fitting_type': func.get('fitting_type', 'spline'),
                        'smoothing': func.get('smoothing', 0.0),
                        'poly_degree': func.get('poly_degree', 3),
                        'y_range_tab2': func.get('y_range_tab2'),
                        'bvd_type': func.get('bvd_type'),
                        'visible': func.get('visible', True)
                    }
                    export_data.append(func_data)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "Success", "Data exported to JSON")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")
    
    def loadDataFromCSV(self):
        """Load data from CSV file (wrapper for importCSV)"""
        self.importCSV()
    
    def loadDataFromJSON(self):
        """Load data from JSON file (wrapper for importJSON)"""
        self.importJSON()
    
    def importJSON(self):
        """Import functions from JSON file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load JSON", "", "JSON Files (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
                
                if not import_data:
                    QMessageBox.warning(self, "Error", "JSON file is empty")
                    return
                
                # Clear existing functions or append?
                existing_count = len(self.all_functions)
                if existing_count > 0:
                    reply = QMessageBox.question(
                        self, "Import Functions",
                        f"Found {existing_count} existing function(s).\n"
                        "Replace existing functions or add to them?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                else:
                    reply = QMessageBox.StandardButton.Yes  # No existing functions, so "replace" is fine
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.all_functions = []
                    logger.info("Replacing all existing functions with JSON data")
                else:
                    # When appending, only clear bvd_type for functions that don't match their names
                    for func in self.all_functions:
                        name_lower = func['name'].lower().strip()
                        bvd_type = func.get('bvd_type')
                        # Clear assignment only if it doesn't match the name
                        if bvd_type == 'conductance' and 'conductance' not in name_lower and 'g' not in name_lower:
                            func['bvd_type'] = None
                        elif bvd_type == 'susceptance' and 'susceptance' not in name_lower and 'b' not in name_lower:
                            func['bvd_type'] = None
                    logger.info(f"Adding JSON data to {len(self.all_functions)} existing function(s)")
                
                # Import functions
                for func_data in import_data:
                    # Reconstruct function data
                    func = {
                        'name': func_data.get('name', 'Imported Function'),
                        'data': [(p[0], p[1]) for p in func_data.get('data', [])],
                        'original_points': [(p[0], p[1]) for p in func_data.get('original_points', [])],
                        'fitting_type': func_data.get('fitting_type', 'spline'),
                        'smoothing': func_data.get('smoothing', 0.0),
                        'poly_degree': func_data.get('poly_degree', 3),
                        'y_range_tab2': func_data.get('y_range_tab2'),
                        'bvd_type': func_data.get('bvd_type'),
                        'visible': func_data.get('visible', True)
                    }
                    self.all_functions.append(func)
                
                # Update UI
                self.updateBVDAssignmentLists()
                self.showResults()
                self.tabs.setTabEnabled(3, True)
                # Enable TAB 6 (RX/TX) if there are functions
                if hasattr(self, 'tab_rxtx'):
                    self.tabs.setTabEnabled(5, True)
                # Enable other tabs
                self.tabs.setTabEnabled(1, True)  # Coordinates tab (may not be needed, but enable anyway)
                self.tabs.setTabEnabled(2, True)  # Data Points tab (may not be needed, but enable anyway)
                
                QMessageBox.information(self, "Success", f"Imported {len(import_data)} function(s)")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")
                import traceback
                traceback.print_exc()
    
    def loadDataFromCSV(self):
        """Load data from CSV file (wrapper for importCSV)"""
        self.importCSV()
    
    def loadDataFromJSON(self):
        """Load data from JSON file (wrapper for importJSON)"""
        self.importJSON()
    
    def importCSV(self):
        """Import functions from CSV file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load CSV", "", "CSV Files (*.csv)"
        )
        
        if filename:
            try:
                functions_data = []
                current_func = None
                
                with open(filename, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    
                    for row in reader:
                        if not row:
                            continue
                        
                        # Check if it's a function name header
                        if row[0].startswith('#'):
                            if current_func:
                                functions_data.append(current_func)
                            current_func = {
                                'name': row[0].strip('# ').strip(),
                                'data': []
                            }
                        elif len(row) >= 2 and row[0] != 'Frequency (kHz)' and row[1] != 'Value':
                            # Data row
                            try:
                                freq = float(row[0])
                                value = float(row[1])
                                if current_func:
                                    current_func['data'].append((freq, value))
                            except ValueError:
                                continue
                
                # Add last function
                if current_func and current_func['data']:
                    functions_data.append(current_func)
                
                if not functions_data:
                    QMessageBox.warning(self, "Error", "No valid data found in CSV file")
                    return
                
                # Clear existing functions or append?
                reply = QMessageBox.question(
                    self, "Import Functions",
                    "Replace existing functions or add to them?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.all_functions = []
                    logger.info("Replacing all existing functions with CSV data")
                else:
                    # When appending, only clear bvd_type for functions that don't match their names
                    # This preserves correct assignments while allowing auto-assignment for new functions
                    for func in self.all_functions:
                        name_lower = func['name'].lower().strip()
                        bvd_type = func.get('bvd_type')
                        # Clear assignment only if it doesn't match the name
                        if bvd_type == 'conductance' and 'conductance' not in name_lower and 'g' not in name_lower:
                            func['bvd_type'] = None
                        elif bvd_type == 'susceptance' and 'susceptance' not in name_lower and 'b' not in name_lower:
                            func['bvd_type'] = None
                    logger.info(f"Adding CSV data to {len(self.all_functions)} existing function(s)")
                
                # Import functions
                for func_data in functions_data:
                    func_name = func_data['name'].strip()
                    logger.info(f"Importing function from CSV: '{func_name}' with {len(func_data['data'])} data points")
                    func = {
                        'name': func_name,
                        'data': func_data['data'],
                        'original_points': func_data['data'].copy(),  # Use same data as original
                        'fitting_type': 'spline',  # Default
                        'smoothing': 0.0,
                        'poly_degree': 3,
                        'y_range_tab2': None,
                        'bvd_type': None,  # Always start with None for fresh assignment
                        'visible': True
                    }
                    self.all_functions.append(func)
                
                # Recalculate fitting for imported functions
                for func in self.all_functions:
                    if func['data']:
                        # Recalculate with default parameters
                        self.recalculateFittingForFunction(func)
                
                # Update UI
                self.updateBVDAssignmentLists()
                self.showResults()
                self.tabs.setTabEnabled(3, True)
                # Enable TAB 6 (RX/TX) if there are functions
                if hasattr(self, 'tab_rxtx'):
                    self.tabs.setTabEnabled(5, True)
                # Enable other tabs
                self.tabs.setTabEnabled(1, True)  # Coordinates tab (may not be needed, but enable anyway)
                self.tabs.setTabEnabled(2, True)  # Data Points tab (may not be needed, but enable anyway)
                
                QMessageBox.information(self, "Success", f"Imported {len(functions_data)} function(s)")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")
                import traceback
                traceback.print_exc()
    
    def recalculateFittingForFunction(self, func):
        """Recalculate fitting for a function"""
        if not func.get('original_points'):
            return
        
        # Get original points
        original_points = func['original_points']
        if len(original_points) < 2:
            return
        
        x_data = np.array([p[0] for p in original_points])
        y_data = np.array([p[1] for p in original_points])
        
        # Sort by x
        sort_idx = np.argsort(x_data)
        x_data = x_data[sort_idx]
        y_data = y_data[sort_idx]
        
        fitting_type = func.get('fitting_type', 'spline')
        
        if fitting_type == 'spline':
            # Spline fitting
            smoothing = func.get('smoothing', 0.0)
            y_range = y_data.max() - y_data.min()
            if y_range == 0:
                y_range = 1.0
            s_param = smoothing * (y_range ** 2) * len(x_data)
            
            try:
                spline = UnivariateSpline(x_data, y_data, s=s_param)
                x_fit = np.linspace(x_data.min(), x_data.max(), 200)
                y_fit = spline(x_fit)
                func['data'] = [(x, y) for x, y in zip(x_fit, y_fit)]
            except Exception:
                # Fallback to linear interpolation
                func['data'] = [(x, y) for x, y in zip(x_data, y_data)]
        else:
            # Polynomial fitting
            poly_degree = func.get('poly_degree', 3)
            max_degree = min(poly_degree, len(x_data) - 1)
            if max_degree < 1:
                max_degree = 1
            
            try:
                poly = Polynomial.fit(x_data, y_data, max_degree)
                x_fit = np.linspace(x_data.min(), x_data.max(), 200)
                y_fit = poly(x_fit)
                func['data'] = [(x, y) for x, y in zip(x_fit, y_fit)]
            except Exception:
                # Fallback to linear interpolation
                func['data'] = [(x, y) for x, y in zip(x_data, y_data)]
    
    def startNewFunction(self):
        """Start extracting new function - with calibration choice"""
        reply = QMessageBox.question(
            self, "New Function", 
            "Start extracting new function?\n\n"
            "Previously extracted functions will be saved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # If calibration exists, offer options
        if self.calibration and self.coord_points:
            msg = QMessageBox(self)
            msg.setWindowTitle("Coordinate System")
            msg.setText("Choose calibration option for new function:")
            msg.setInformativeText(
                "• Same calibration - use same points and axis values\n"
                "• New axis values - keep points but set different values\n"
                "• New calibration - place new 3 points and values"
            )
            
            btn_same = msg.addButton("Same calibration", QMessageBox.ButtonRole.YesRole)
            btn_values = msg.addButton("New axis values", QMessageBox.ButtonRole.NoRole)
            btn_new = msg.addButton("New calibration", QMessageBox.ButtonRole.RejectRole)
            msg.addButton(QMessageBox.StandardButton.Cancel)
            
            msg.exec()
            clicked = msg.clickedButton()
            
            if clicked == msg.button(QMessageBox.StandardButton.Cancel):
                return
            
            if clicked == btn_same:
                # Use current calibration completely
                self.data_points = []
                self.extract_image.clearPoints()
                self.label_point_count.setText("Points collected: 0")
                self.btn_extract_next.setEnabled(False)
                
                current_num = len(self.all_functions) + 1
                self.input_func_name.setText(f"Function {current_num}")
                
                self.tabs.setCurrentIndex(2)
                
                QMessageBox.information(
                    self, "New Function",
                    "Using current calibration.\nStart extracting points."
                )
                return
            
            elif clicked == btn_values:
                # Keep points but enter new axis values
                self.data_points = []
                self.extract_image.clearPoints()
                
                # Save old values as hints
                old_x_min = self.input_x_min.text()
                old_y_min = self.input_y_min.text()
                old_x_max = self.input_x_max.text()
                old_y_max = self.input_y_max.text()
                
                # Clear fields for new values
                self.input_x_min.clear()
                self.input_y_min.clear()
                self.input_x_max.clear()
                self.input_y_max.clear()
                
                # Set placeholders with old values
                self.input_x_min.setPlaceholderText(f"Was: {old_x_min}")
                self.input_y_min.setPlaceholderText(f"Was: {old_y_min}")
                self.input_x_max.setPlaceholderText(f"Was: {old_x_max}")
                self.input_y_max.setPlaceholderText(f"Was: {old_y_max}")
                
                current_num = len(self.all_functions) + 1
                self.input_func_name.setText(f"Function {current_num}")
                
                # Reset calibration but keep points
                self.calibration = None
                self.btn_calib_next.setEnabled(False)
                self.tabs.setTabEnabled(2, False)
                
                # Go to calibration tab to enter new values
                self.tabs.setCurrentIndex(1)
                
                QMessageBox.information(
                    self, "New Axis Values",
                    "Reference points saved.\n"
                    "Enter new values for coordinate axes (X₀, X_max, Y₀, Y_max)\n"
                    "and click 'Next' to recalculate calibration."
                )
                return
        
        # Create completely new calibration
        self.coord_points = []
        self.data_points = []
        self.calibration = None
        self.calib_image.clearPoints()
        self.extract_image.clearPoints()
        
        # Enable calibration mode
        self.calib_image.calibration_mode = True
        self.calib_image.setCursor(Qt.CursorShape.CrossCursor)
        
        current_num = len(self.all_functions) + 1
        self.input_func_name.setText(f"Function {current_num}")
        
        # Disable tabs and return to calibration
        self.btn_calib_next.setEnabled(False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setCurrentIndex(1)
        
        QMessageBox.information(
            self, "New Function Start",
            "Place 3 calibration points for new function."
        )
    
    def reset(self):
        """Full application reset"""
        reply = QMessageBox.question(
            self, "Confirmation", 
            "Are you sure? All data will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.coord_points = []
            self.data_points = []
            self.calibration = None
            self.extracted_data = []
            self.all_functions = []
            
            self.calib_image.clearPoints()
            self.extract_image.clearPoints()
            
            self.tabs.setTabEnabled(1, False)
            self.tabs.setTabEnabled(2, False)
            self.tabs.setTabEnabled(3, False)
            self.tabs.setCurrentIndex(0)
            
            self.preview_label.clear()
            self.preview_label.setText("Image not loaded")
            self.result_plot.clear()
            self.result_table.setRowCount(0)


def main():
    app = QApplication(sys.argv)
    
    # Application style
    app.setStyle('Fusion')
    
    window = GraphExtractorApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()