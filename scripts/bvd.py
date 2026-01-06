"""
Graph Extractor - Application for extracting data from graphs
Uses PyQt6, pyqtgraph for interactive data extraction
"""

import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QFileDialog,
                              QLineEdit, QTableWidget, QTableWidgetItem, 
                              QTabWidget, QGroupBox, QMessageBox, QSplitter)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import pyqtgraph as pg
from scipy.interpolate import UnivariateSpline, interp1d
import csv
import json

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
        
        self.initUI()
        
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Tabs for different stages
        self.tabs = QTabWidget()
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
        
        # Disable tabs until image is loaded
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)
        self.tabs.setTabEnabled(4, False)
        
    def createLoadTab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Instructions
        info = QLabel("""
        <b style='font-size: 18px;'>📁 STEP 1: Load Image</b><br><br>
        <b>Workflow:</b><br>
        1️⃣ Load graph image or PDF<br>
        2️⃣ Calibrate coordinate axes (3 points)<br>
        3️⃣ Extract data from graph<br>
        4️⃣ View results and export<br>
        5️⃣ (Optional) Calculate BVD parameters
        """)
        info.setStyleSheet("font-size: 14px; padding: 20px; background: #e3f2fd; border-radius: 8px;")
        layout.addWidget(info)
        
        # Load buttons
        btn_layout = QVBoxLayout()
        
        # Buttons
        buttons = QHBoxLayout()
        btn_image = QPushButton("Load Image")
        btn_image.clicked.connect(self.loadImage)
        buttons.addWidget(btn_image)
        
        if PDF_AVAILABLE:
            btn_pdf = QPushButton("Load PDF")
            btn_pdf.clicked.connect(lambda: self.loadPDFFromButton())
            buttons.addWidget(btn_pdf)
        btn_layout.addLayout(buttons)
        
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
        
        # Calculated parameters group
        params_group = QGroupBox("Calculated BVD Parameters")
        params_layout = QVBoxLayout()
        
        # Parameters table
        self.bvd_params_table = QTableWidget()
        self.bvd_params_table.setColumnCount(3)
        self.bvd_params_table.setHorizontalHeaderLabels(['Parameter', 'Value', 'Unit'])
        self.bvd_params_table.setRowCount(8)
        
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
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
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
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        btn_export_bvd = QPushButton("💾 Export BVD Parameters")
        btn_export_bvd.clicked.connect(self.exportBVDParams)
        btn_layout.addWidget(btn_export_bvd)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return widget
    
    def calculateBVD(self):
        """Calculate BVD parameters from Admittance data"""
        try:
            # Check for Conductance and Susceptance functions
            conductance_func = None
            susceptance_func = None
            
            for func in self.all_functions:
                name_lower = func['name'].lower()
                if 'conductance' in name_lower:
                    conductance_func = func
                elif 'susceptance' in name_lower:
                    susceptance_func = func
            
            if not conductance_func or not susceptance_func:
                QMessageBox.warning(
                    self, "Error", 
                    "Both functions must be extracted: Conductance and Susceptance"
                )
                return
            
            # Get C₀ and fs
            C0_nF = float(self.input_c0.text())
            fs_kHz = float(self.input_fs.text())
            
            # Convert to SI units
            C0 = C0_nF * 1e-9  # nF -> F
            fs = fs_kHz * 1e3  # kHz -> Hz
            
            # Conductance data
            g_data = conductance_func['data']
            freq_g = np.array([p[0] * 1e3 for p in g_data])  # kHz -> Hz
            g_values = np.array([p[1] * 1e-3 for p in g_data])  # mS -> S
            
            # Susceptance data
            b_data = susceptance_func['data']
            freq_b = np.array([p[0] * 1e3 for p in b_data])  # kHz -> Hz
            b_values = np.array([p[1] * 1e-3 for p in b_data])  # mS -> S
            
            # Find resonant frequency from Conductance maximum
            g_max_idx = np.argmax(g_values)
            fs_measured = freq_g[g_max_idx]
            g_max = g_values[g_max_idx]
            
            # Find antiresonant frequency
            # fp is the frequency of |Y| minimum after resonance
            # Approximately: where Susceptance crosses zero after resonance
            zero_crossings = np.where(np.diff(np.sign(b_values)))[0]
            fp_measured = fs_measured
            
            for idx in zero_crossings:
                if freq_b[idx] > fs_measured:
                    fp_measured = freq_b[idx]
                    break
            
            # If not found, use approximate formula
            if fp_measured == fs_measured:
                # Typical ratio fp/fs ≈ 1.05-1.15 for piezoceramics
                fp_measured = fs_measured * 1.1
            
            # Calculate BVD parameters
            # R1 from Conductance maximum
            R1 = 1.0 / g_max
            
            # C1 from frequency ratio
            freq_ratio = (fp_measured / fs_measured) ** 2
            C1 = C0 / (freq_ratio - 1)
            
            # L1 from resonant frequency
            L1 = 1.0 / (4 * np.pi**2 * fs_measured**2 * C1)
            
            # Mechanical Q-factor
            omega_s = 2 * np.pi * fs_measured
            Qm = omega_s * L1 / R1
            
            # Electromechanical coupling coefficient
            k = np.sqrt(1 - (fs_measured / fp_measured)**2)
            
            # Save parameters
            self.bvd_params = {
                'C0': C0,
                'fs': fs_measured,
                'fp': fp_measured,
                'R1': R1,
                'L1': L1,
                'C1': C1,
                'Qm': Qm,
                'k': k
            }
            
            # Update table
            values = [
                f"{C0*1e9:.2f}",
                f"{fs_measured*1e-3:.4f}",
                f"{fp_measured*1e-3:.4f}",
                f"{R1:.2f}",
                f"{L1*1e3:.4f}",
                f"{C1*1e9:.4f}",
                f"{Qm:.2f}",
                f"{k:.4f}"
            ]
            
            for i, val in enumerate(values):
                self.bvd_params_table.setItem(i, 1, QTableWidgetItem(val))
            
            # Build model curves
            self.plotBVDComparison(freq_g, g_values, freq_b, b_values)
            
            # Enable BVD tab
            self.tabs.setTabEnabled(4, True)
            self.tabs.setCurrentIndex(4)
            
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Parameter error: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"BVD calculation error: {e}")
    
    def bvd_admittance(self, freq, C0, R1, L1, C1):
        """Calculate Admittance from BVD model"""
        omega = 2 * np.pi * freq
        
        # Series branch impedance
        Z_series = R1 + 1j * (omega * L1 - 1 / (omega * C1))
        
        # Series branch admittance
        Y_series = 1 / Z_series
        
        # C0 admittance
        Y_C0 = 1j * omega * C0
        
        # Total admittance
        Y_total = Y_series + Y_C0
        
        return Y_total
    
    def plotBVDComparison(self, freq_g, g_exp, freq_b, b_exp):
        """Plot comparison of experimental and model data"""
        
        # Generate frequency grid
        freq_model = np.linspace(min(freq_g.min(), freq_b.min()), 
                                 max(freq_g.max(), freq_b.max()), 500)
        
        # Calculate model admittance
        Y_model = self.bvd_admittance(
            freq_model,
            self.bvd_params['C0'],
            self.bvd_params['R1'],
            self.bvd_params['L1'],
            self.bvd_params['C1']
        )
        
        g_model = np.real(Y_model) * 1e3  # S -> mS
        b_model = np.imag(Y_model) * 1e3  # S -> mS
        
        # Magnitude and phase
        y_mag_model = np.abs(Y_model) * 1e3  # mS
        y_phase_model = np.angle(Y_model, deg=True)  # degrees
        
        # Experimental magnitude and phase
        Y_exp_g = g_exp + 1j * np.interp(freq_g, freq_b, b_exp)
        y_mag_exp = np.abs(Y_exp_g) * 1e3
        y_phase_exp = np.angle(Y_exp_g, deg=True)
        
        # === Plot 1: Conductance ===
        self.bvd_plot_g.clear()
        self.bvd_plot_g.plot(
            freq_g * 1e-3, g_exp * 1e3, 
            pen=None, symbol='o', symbolBrush=(0, 100, 255, 150), symbolSize=6,
            name='PDF data'
        )
        self.bvd_plot_g.plot(
            freq_model * 1e-3, g_model,
            pen=pg.mkPen((255, 0, 0), width=2),
            name='BVD model'
        )
        
        # Mark resonance
        fs_kHz = self.bvd_params['fs'] * 1e-3
        self.bvd_plot_g.addLine(x=fs_kHz, pen=pg.mkPen('g', style=Qt.PenStyle.DashLine, width=2))
        
        # === Plot 2: Susceptance ===
        self.bvd_plot_b.clear()
        self.bvd_plot_b.plot(
            freq_b * 1e-3, b_exp * 1e3,
            pen=None, symbol='o', symbolBrush=(0, 100, 255, 150), symbolSize=6,
            name='PDF data'
        )
        self.bvd_plot_b.plot(
            freq_model * 1e-3, b_model,
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
            freq_model * 1e-3, y_mag_model,
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
            freq_model * 1e-3, y_phase_model,
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
        
        # RMSE
        rmse_g = np.sqrt(np.mean((g_exp * 1e3 - g_model_at_exp)**2))
        rmse_b = np.sqrt(np.mean((b_exp * 1e3 - b_model_at_exp)**2))
        
        # R² (coefficient of determination)
        ss_res_g = np.sum((g_exp * 1e3 - g_model_at_exp)**2)
        ss_tot_g = np.sum((g_exp * 1e3 - np.mean(g_exp * 1e3))**2)
        r2_g = 1 - (ss_res_g / ss_tot_g) if ss_tot_g > 0 else 0
        
        ss_res_b = np.sum((b_exp * 1e3 - b_model_at_exp)**2)
        ss_tot_b = np.sum((b_exp * 1e3 - np.mean(b_exp * 1e3))**2)
        r2_b = 1 - (ss_res_b / ss_tot_b) if ss_tot_b > 0 else 0
        
        # Maximum deviations
        max_error_g = np.max(np.abs(g_exp * 1e3 - g_model_at_exp))
        max_error_b = np.max(np.abs(b_exp * 1e3 - b_model_at_exp))
        
        # Mean relative errors
        mean_rel_error_g = np.mean(np.abs((g_exp * 1e3 - g_model_at_exp) / (g_exp * 1e3 + 1e-10))) * 100
        mean_rel_error_b = np.mean(np.abs((b_exp * 1e3 - b_model_at_exp) / (np.abs(b_exp * 1e3) + 1e-10))) * 100
        
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
        
        # Smoothing settings
        smoothing_group = QGroupBox("📊 Interactive Spline Smoothing")
        smoothing_layout = QHBoxLayout()
        
        smoothing_layout.addWidget(QLabel("Smoothing level:"))
        
        from PyQt6.QtWidgets import QSlider
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setMinimum(0)
        self.smoothing_slider.setMaximum(100)
        self.smoothing_slider.setValue(0)
        self.smoothing_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.smoothing_slider.setTickInterval(10)
        self.smoothing_slider.valueChanged.connect(self.updateSmoothing)
        smoothing_layout.addWidget(self.smoothing_slider)
        
        self.smoothing_value_label = QLabel("0")
        self.smoothing_value_label.setMinimumWidth(40)
        self.smoothing_value_label.setStyleSheet("font-weight: bold; padding: 5px;")
        smoothing_layout.addWidget(self.smoothing_value_label)
        
        smoothing_layout.addWidget(QLabel("  Fine tuning:"))
        self.smoothing_fine = QLineEdit("0")
        self.smoothing_fine.setMaximumWidth(80)
        self.smoothing_fine.returnPressed.connect(self.updateSmoothingFromInput)
        smoothing_layout.addWidget(self.smoothing_fine)
        
        btn_apply_smoothing = QPushButton("Apply")
        btn_apply_smoothing.clicked.connect(self.updateSmoothingFromInput)
        smoothing_layout.addWidget(btn_apply_smoothing)
        
        btn_reset_smoothing = QPushButton("Reset (0)")
        btn_reset_smoothing.clicked.connect(lambda: self.smoothing_slider.setValue(0))
        smoothing_layout.addWidget(btn_reset_smoothing)
        
        smoothing_layout.addStretch()
        smoothing_group.setLayout(smoothing_layout)
        plot_layout.addWidget(smoothing_group)
        
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
        
        # Results plot
        self.result_plot = pg.PlotWidget()
        self.result_plot.setBackground('w')
        self.result_plot.setLabel('left', 'Value')
        self.result_plot.setLabel('bottom', 'Frequency (kHz)')
        self.result_plot.showGrid(x=True, y=True, alpha=0.3)
        plot_layout.addWidget(self.result_plot)
        
        splitter.addWidget(plot_container)
        
        # Data table
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(['#', 'Frequency (kHz)', 'Value'])
        splitter.addWidget(self.result_table)
        
        layout.addWidget(splitter)
        
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
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        btn_export_csv = QPushButton("💾 Export CSV")
        btn_export_csv.clicked.connect(self.exportCSV)
        btn_layout.addWidget(btn_export_csv)
        
        btn_export_json = QPushButton("💾 Export JSON")
        btn_export_json.clicked.connect(self.exportJSON)
        btn_layout.addWidget(btn_export_json)
        
        btn_layout.addStretch()
        
        btn_bvd = QPushButton("🔬 Calculate BVD Model")
        btn_bvd.clicked.connect(self.calculateBVD)
        btn_layout.addWidget(btn_bvd)
        
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
    
    def loadImageFile(self, filename):
        """Load image"""
        pixmap = QPixmap(filename)
    def loadImageFile(self, filename):
        """Load image"""
        pixmap = QPixmap(filename)
        self.preview_label.setPixmap(pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        
        self.calib_image.setImage(pixmap)
        self.extract_image.setImage(pixmap)
        
        # Reset all data when loading new image
        self.coord_points = []
        self.data_points = []
        self.calibration = None
        self.extracted_data = []
        self.calib_image.clearPoints()
        self.extract_image.clearPoints()
        
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
            doc = fitz.open(filename)
            
            # If PDF has more than one page, let user choose
            page_num = 0
            if len(doc) > 1:
                from PyQt6.QtWidgets import QInputDialog
                page_num, ok = QInputDialog.getInt(
                    self, "Page Selection", 
                    f"Document contains {len(doc)} pages.\nSelect page to load:",
                    1, 1, len(doc), 1
                )
                if not ok:
                    doc.close()
                    return
                page_num -= 1  # Convert to 0-index
            
            page = doc[page_num]
            
            # Convert to high resolution image
            mat = fitz.Matrix(zoom_factor, zoom_factor)  # Adjustable scale for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to QPixmap
            img_data = pix.tobytes("png")
            qimg = QImage.fromData(img_data)
            pixmap = QPixmap.fromImage(qimg)
            
            self.preview_label.setPixmap(pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
            self.calib_image.setImage(pixmap)
            self.extract_image.setImage(pixmap)
            
            self.tabs.setTabEnabled(1, True)
            self.tabs.setCurrentIndex(1)
            
            doc.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load PDF: {e}")
    
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
        
        # Create spline
        try:
            # By default no smoothing, user will adjust in results
            smoothing = 0
            
            spline = UnivariateSpline(x_data, y_data, s=smoothing, k=min(3, len(x_data)-1))
            
            # Generate dense grid for visualization
            x_dense = np.linspace(x_data.min(), x_data.max(), 200)
            y_dense = spline(x_dense)
            
            # Save data
            self.extracted_data = list(zip(x_dense, y_dense))
            
            # Save function
            func_name = str(self.input_func_name.text())
            self.all_functions.append({
                'name': func_name,
                'data': self.extracted_data.copy(),
                'original_points': real_points,
                'visible': True  # Visibility flag
            })
            
            # Show results
            self.showResults()
            self.tabs.setTabEnabled(3, True)
            self.tabs.setCurrentIndex(3)
            
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
        pass  # Can add highlighting of selected function
    
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
        self.smoothing_value_label.setText(str(value))
        self.smoothing_fine.setText(str(value))
        self.recalculateSplines(value)
    
    def updateSmoothingFromInput(self):
        """Update spline from text field"""
        try:
            value = float(self.smoothing_fine.text())
            value = max(0, min(1000, value))  # Limit 0-1000
            self.smoothing_slider.setValue(int(min(100, value)))  # Slider up to 100
            self.smoothing_value_label.setText(f"{value:.1f}")
            self.recalculateSplines(value)
        except ValueError:
            QMessageBox.warning(self, "Error", "Enter valid number")
    
    def recalculateSplines(self, smoothing):
        """Recalculate splines for all functions with new smoothing parameter"""
        if not self.all_functions:
            return
        
        for func in self.all_functions:
            # Get original points
            orig_points = func['original_points']
            x_data = np.array([float(p[0]) for p in orig_points], dtype=np.float64)
            y_data = np.array([float(p[1]) for p in orig_points], dtype=np.float64)
            
            # Recalculate spline with new parameter
            try:
                spline = UnivariateSpline(x_data, y_data, s=smoothing, k=min(3, len(x_data)-1))
                x_dense = np.linspace(x_data.min(), x_data.max(), 200)
                y_dense = spline(x_dense)
                
                # Update function data
                func['data'] = list(zip(x_dense, y_dense))
            except Exception as e:
                print(f"Error recalculating spline: {e}")
        
        # Update display
        self.showResults()
    
    def showResults(self):
        """Display results on plot with separate Y axes for each function"""
        plot_item = self.result_plot.getPlotItem()
        
        # Clear old additional axes
        for axis in self.extra_axes:
            plot_item.layout.removeItem(axis)
            if axis.scene():
                axis.scene().removeItem(axis)
        
        self.extra_axes = []
        
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
        
        # For each function calculate range and scale to normalized range
        y_ranges = []
        normalized_data = []
        
        for func in visible_functions:
            data = func['data']
            y = [p[1] for p in data]
            y_min, y_max = min(y), max(y)
            y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
            y_range = (y_min - y_margin, y_max + y_margin)
            y_ranges.append(y_range)
            print(f"DEBUG: Function '{func['name']}': Y from {y_min:.4f} to {y_max:.4f}, range with margin: {y_range[0]:.4f} - {y_range[1]:.4f}")
        
        # Draw first function on main Y axis (left) without normalization
        first_func = visible_functions[0]
        original_idx = self.all_functions.index(first_func)
        
        data = first_func['data']
        x = [p[0] for p in data]
        y = [p[1] for p in data]
        
        print(f"DEBUG: First function Y values: min={min(y):.4f}, max={max(y):.4f}")
        print(f"DEBUG: Setting Y range: {y_ranges[0][0]:.4f} - {y_ranges[0][1]:.4f}")
        
        data = first_func['data']
        x = [p[0] for p in data]
        y = [p[1] for p in data]
        
        color = colors[original_idx % len(colors)]
        
        # Spline for first function
        self.result_plot.plot(x, y, pen=pg.mkPen(color, width=2), name=f"{first_func['name']} (spline)")
        
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
        print(f"DEBUG: setYRange called with {y_ranges[0][0]:.4f} - {y_ranges[0][1]:.4f}")
        
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
            self.result_plot.plot(x, y_scaled, pen=pg.mkPen(color, width=2), name=f"{func['name']} (spline)")
            
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
            plot_item.layout.addItem(axis, 2, 3 + i)
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
        
        # Add legend
        plot_item.addLegend()
        
        # CRITICALLY IMPORTANT: Disable auto range and explicitly set Y range
        # at the end AFTER adding legend and all elements
        print(f"DEBUG: Final Y range setting: {y_ranges[0][0]:.4f} - {y_ranges[0][1]:.4f}")
        plot_item.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        plot_item.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        plot_item.setYRange(y_ranges[0][0], y_ranges[0][1], padding=0)
        
        # Also set limits to prevent automatic scaling
        plot_item.vb.setLimits(yMin=y_ranges[0][0], yMax=y_ranges[0][1])
        
        # Check what we got
        actual_range = plot_item.vb.viewRange()
        print(f"DEBUG: Actual range after setting: Y from {actual_range[1][0]:.4f} to {actual_range[1][1]:.4f}")
        
        
        # Fill table with ALL functions
        if self.all_functions:
            # Count total number of rows (all points of all functions + headers)
            total_rows = 0
            for func in self.all_functions:
                total_rows += 1  # Function header
                total_rows += len(func['data'])  # Function data
            
            self.result_table.setRowCount(total_rows)
            
            row_idx = 0
            for func in self.all_functions:
                # Function header
                header_item = QTableWidgetItem(f"=== {func['name']} ===")
                header_item.setBackground(QColor(200, 220, 255))
                font = header_item.font()
                font.setBold(True)
                header_item.setFont(font)
                self.result_table.setItem(row_idx, 0, header_item)
                self.result_table.setItem(row_idx, 1, QTableWidgetItem(""))
                self.result_table.setItem(row_idx, 2, QTableWidgetItem(""))
                row_idx += 1
                
                # Function data
                data = func['data']
                for i, (x, y) in enumerate(data):
                    self.result_table.setItem(row_idx, 0, QTableWidgetItem(str(i+1)))
                    self.result_table.setItem(row_idx, 1, QTableWidgetItem(f"{x:.4f}"))
                    self.result_table.setItem(row_idx, 2, QTableWidgetItem(f"{y:.6f}"))
                    row_idx += 1
    
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
                    export_data.append({
                        'name': func['name'],
                        'data': [[x, y] for x, y in func['data']]
                    })
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "Success", "Data exported to JSON")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")
    
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