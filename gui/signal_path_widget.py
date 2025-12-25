"""
SignalPathWidget - widget for displaying signal path with interactive parameters.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox
from PyQt5.QtWidgets import QLabel, QDoubleSpinBox, QSpinBox, QSplitter
from PyQt5.QtCore import Qt, pyqtSignal, QRect
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPalette
from typing import Dict, Optional


class SignalPathDiagram(QWidget):
    """Widget for drawing signal path diagram."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.path_data = None
        self.setMinimumSize(600, 600)  # Increased height to accommodate summary below
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, Qt.white)
        self.setPalette(palette)
    
    def set_path_data(self, path_data: Dict):
        """Sets signal path data."""
        self.path_data = path_data
        self.update()  # Request repaint
        self.repaint()  # Force repaint
    
    def paintEvent(self, event):
        """Paints signal path diagram."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Background
        painter.fillRect(self.rect(), QColor(240, 240, 240))
        
        if not self.path_data:
            painter.setPen(QPen(QColor(200, 0, 0), 2))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data to display")
            return
        
        # Coordinates for element placement (compact version)
        # Left (x=20-140)
        # Center (x=180-280)
        # Right (x=320-420) - bottom
        # Further right (x=450+) - summary
        
        tx = self.path_data.get('tx', {})
        wf = self.path_data.get('water_forward', {})
        bottom = self.path_data.get('bottom', {})
        wb = self.path_data.get('water_backward', {})
        rx = self.path_data.get('rx', {})
        summary = self.path_data.get('summary', {})
        
        # === LEFT: TRANSMITTER (TX) ===
        tx_x, tx_y = 20, 30
        tx_width, tx_height = 120, 100
        tx_rect = QRect(tx_x, tx_y, tx_width, tx_height)
        
        # TX frame
        painter.setPen(QPen(QColor(0, 0, 200), 2))
        painter.setBrush(QBrush(QColor(200, 220, 255)))
        painter.drawRoundedRect(tx_rect, 5, 5)
        
        # TX header
        painter.setPen(QPen(QColor(0, 0, 150), 1))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(QRect(tx_x, tx_y, tx_width, 25), Qt.AlignCenter, "TRANSMITTER (TX)")
        
        # TX parameters (compact)
        painter.setFont(QFont("Courier", 7))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        y_offset = tx_y + 25
        params = [
            f"V={tx.get('voltage', 0):.1f}V",
            f"P={tx.get('power', 0)*1000:.0f}mW",
            f"SPL={tx.get('spl', 0):.1f}dB",
            f"S_TX={tx.get('sensitivity', 0):.1f}dB"
        ]
        for param in params:
            painter.drawText(tx_x + 5, y_offset, param)
            y_offset += 15
        
        # === CENTER: WATER (FORWARD) ===
        wf_x, wf_y = 160, 20
        wf_width, wf_height = 120, 120
        wf_rect = QRect(wf_x, wf_y, wf_width, wf_height)
        
        painter.setPen(QPen(QColor(0, 150, 200), 2))
        painter.setBrush(QBrush(QColor(200, 240, 255)))
        painter.drawRoundedRect(wf_rect, 5, 5)
        
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(QRect(wf_x, wf_y, wf_width, 20), Qt.AlignCenter, "WATER (FORWARD)")
        
        painter.setFont(QFont("Courier", 7))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        y_offset = wf_y + 25
        # All values come from core - no calculations in GUI
        spreading_fwd = wf.get('spreading_loss', 0)
        absorb_fwd = wf.get('absorption_loss', 0)
        total_fwd = wf.get('total_attenuation', 0)  # Use value from core, no fallback calculation
        params = [
            f"D={wf.get('distance', 0):.2f}m",
            f"T={wf.get('temperature', 0):.1f}°C",
            f"S={wf.get('salinity', 0):.1f}PSU",
            f"Spread={spreading_fwd:.2f}dB",
            f"Absorb={absorb_fwd:.2f}dB",
            f"Total={total_fwd:.2f}dB"
        ]
        for param in params:
            painter.drawText(wf_x + 5, y_offset, param)
            y_offset += 15
        
        # === RIGHT: BOTTOM ===
        bottom_x, bottom_y = 300, 120
        bottom_width, bottom_height = 100, 60
        bottom_rect = QRect(bottom_x, bottom_y, bottom_width, bottom_height)
        
        painter.setPen(QPen(QColor(139, 69, 19), 2))
        painter.setBrush(QBrush(QColor(244, 164, 96)))
        painter.drawRoundedRect(bottom_rect, 5, 5)
        
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(QRect(bottom_x, bottom_y, bottom_width, 20), Qt.AlignCenter, "BOTTOM")
        
        painter.setFont(QFont("Courier", 7))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawText(bottom_x + 5, bottom_y + 25, 
                        f"Refl={bottom.get('reflection_loss', 0):.1f}dB")
        
        # === CENTER: WATER (BACKWARD) ===
        wb_x, wb_y = 160, 200
        wb_width, wb_height = 120, 120
        wb_rect = QRect(wb_x, wb_y, wb_width, wb_height)
        
        painter.setPen(QPen(QColor(0, 150, 200), 2))
        painter.setBrush(QBrush(QColor(200, 240, 255)))
        painter.drawRoundedRect(wb_rect, 5, 5)
        
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(QRect(wb_x, wb_y, wb_width, 20), Qt.AlignCenter, "WATER (BACKWARD)")
        
        painter.setFont(QFont("Courier", 7))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        y_offset = wb_y + 25
        # All values come from core - no calculations in GUI
        spreading_bwd = wb.get('spreading_loss', 0)
        absorb_bwd = wb.get('absorption_loss', 0)
        total_bwd = wb.get('total_attenuation', 0)  # Use value from core, no fallback calculation
        params = [
            f"D={wb.get('distance', 0):.2f}m",
            f"T={wb.get('temperature', 0):.1f}°C",
            f"S={wb.get('salinity', 0):.1f}PSU",
            f"Spread={spreading_bwd:.2f}dB",
            f"Absorb={absorb_bwd:.2f}dB",
            f"Total={total_bwd:.2f}dB"
        ]
        for param in params:
            painter.drawText(wb_x + 5, y_offset, param)
            y_offset += 15
        
        # === LEFT: RECEIVER (RX) ===
        rx_x, rx_y = 20, 150
        rx_width, rx_height = 120, 130
        rx_rect = QRect(rx_x, rx_y, rx_width, rx_height)
        
        painter.setPen(QPen(QColor(0, 150, 0), 2))
        painter.setBrush(QBrush(QColor(200, 255, 200)))
        painter.drawRoundedRect(rx_rect, 5, 5)
        
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(QRect(rx_x, rx_y, rx_width, 20), Qt.AlignCenter, "RECEIVER (RX)")
        
        painter.setFont(QFont("Courier", 7))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        y_offset = rx_y + 25
        # S_RX: Receive sensitivity from transducer file (data/transducers/*.json)
        # Typical values: -191 to -193 dB re 1V/µPa (from transducer datasheet)
        s_rx_value = rx.get('transducer_sensitivity', 0)
        params = [
            f"S_RX={s_rx_value:.1f}dB",
            f"  (from transducer)",
            f"LNA: G={rx.get('lna_gain', 0):.1f}dB",
            f"VGA: G={rx.get('vga_gain', 0):.1f}dB",
            f"ADC: {rx.get('adc_bits', 0):.0f}bits",
            f"DR={rx.get('adc_dynamic_range', 0):.1f}dB"
        ]
        for param in params:
            painter.drawText(rx_x + 5, y_offset, param)
            y_offset += 15
        
        # === SIGNAL PATH ARROWS ===
        painter.setPen(QPen(QColor(0, 0, 200), 2))
        arrow_size = 6
        
        # TX -> Water (forward)
        self._draw_arrow(painter, tx_x + tx_width, tx_y + tx_height//2, 
                         wf_x, wf_y + wf_height//2, arrow_size)
        
        # Water (forward) -> Bottom
        self._draw_arrow(painter, wf_x + wf_width, wf_y + wf_height//2,
                         bottom_x, bottom_y + bottom_height//2, arrow_size)
        
        # Bottom -> Water (backward)
        self._draw_arrow(painter, bottom_x, bottom_y + bottom_height//2,
                         wb_x + wb_width, wb_y + wb_height//2, arrow_size)
        
        # Water (backward) -> RX
        self._draw_arrow(painter, wb_x, wb_y + wb_height//2,
                         rx_x + rx_width, rx_y + rx_height//2, arrow_size)
        
        # === SUMMARY PARAMETERS (BELOW DIAGRAM) ===
        # Position summary below the diagram (centered)
        summary_width, summary_height = 180, 270
        summary_x = (width - summary_width) // 2 - 100 # Center horizontally
        summary_y = 340  # Below all diagram elements
        summary_rect = QRect(summary_x, summary_y, summary_width, summary_height)
        
        painter.setPen(QPen(QColor(200, 100, 0), 2))
        painter.setBrush(QBrush(QColor(255, 230, 200)))
        painter.drawRoundedRect(summary_rect, 5, 5)
        
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(QRect(summary_x, summary_y, summary_width, 20), 
                        Qt.AlignCenter, "SUMMARY")
        
        painter.setFont(QFont("Courier", 7))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        y_offset = summary_y + 25
        
        # All values come from core - no calculations in GUI
        # All losses are already calculated as negative values in core for display
        water_fwd = summary.get('water_forward_loss', 0)  # One-way forward path loss (negative, from core)
        bottom = summary.get('bottom_loss', 0)  # Bottom reflection loss (negative, from core)
        water_bwd = summary.get('water_backward_loss', 0)  # One-way backward path loss (negative, from core)
        # Total path loss (round-trip: forward + bottom + backward) - already calculated in core
        total_loss = summary.get('total_path_loss', 0)  # Already calculated in core as sum of negative values
        
        # Display losses with clear labels
        # All values come from core - no calculations in GUI
        total_rx_gain = summary.get('total_rx_gain', 0)
        # RX Gain = S_RX + LNA_gain + VGA_gain
        # S_RX from transducer file, LNA/VGA from hardware files
        params = [
            f"D={summary.get('distance', 0):.2f}m",
            f"",
            f"Losses (one-way):",
            f"Water fwd: {water_fwd:.2f}dB",  # One-way: TX -> bottom
            f"Bottom: {bottom:.2f}dB",
            f"Water bwd: {water_bwd:.2f}dB",  # One-way: bottom -> RX
            f"",
            f"Total (round-trip):",
            f"{total_loss:.2f}dB",  # Complete: forward + bottom + backward
            f"",
            f"RX Gain:",
            f"{total_rx_gain:.2f}dB",
            f"  (S_RX+LNA+VGA)",
            f"",
            f"ADC Signal:",
            f"{summary.get('signal_at_adc', 0):.2f}dB"
        ]
        
        for param in params:
            if param:
                painter.drawText(summary_x + 5, y_offset, param)
            y_offset += 15
    
    def _draw_arrow(self, painter, x1, y1, x2, y2, arrow_size):
        """Draws arrow from (x1, y1) to (x2, y2)."""
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        painter.drawLine(x1, y1, x2, y2)
        
        # Draw arrowhead
        import math
        angle = math.atan2(y2 - y1, x2 - x1)
        
        # Arrowhead points
        x3 = int(x2 - arrow_size * math.cos(angle - math.pi / 6))
        y3 = int(y2 - arrow_size * math.sin(angle - math.pi / 6))
        x4 = int(x2 - arrow_size * math.cos(angle + math.pi / 6))
        y4 = int(y2 - arrow_size * math.sin(angle + math.pi / 6))
        
        painter.drawLine(x2, y2, x3, y3)
        painter.drawLine(x2, y2, x4, y4)


class SignalPathWidget(QWidget):
    """Widget with signal path diagram and parameters."""
    
    # Signals for parameter updates
    lna_gain_changed = pyqtSignal(float)
    lna_nf_changed = pyqtSignal(float)
    vga_gain_changed = pyqtSignal(float)
    # Bottom reflection is now in Environment Parameters, not here
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.path_data = None
        self._init_ui()
    
    def _init_ui(self):
        """Initializes UI."""
        layout = QHBoxLayout(self)
        
        # Left panel - parameters
        params_panel = QGroupBox("Parameters")
        params_layout = QFormLayout()
        
        # LNA parameters
        self.lna_gain_spin = QDoubleSpinBox()
        self.lna_gain_spin.setRange(0, 60)
        self.lna_gain_spin.setValue(20)
        self.lna_gain_spin.setSuffix(" dB")
        self.lna_gain_spin.valueChanged.connect(lambda v: self.lna_gain_changed.emit(v))
        params_layout.addRow("LNA - Gain:", self.lna_gain_spin)
        
        self.lna_nf_spin = QDoubleSpinBox()
        self.lna_nf_spin.setRange(0, 10)
        self.lna_nf_spin.setValue(2)
        self.lna_nf_spin.setSuffix(" dB")
        self.lna_nf_spin.valueChanged.connect(lambda v: self.lna_nf_changed.emit(v))
        params_layout.addRow("LNA - Noise Figure:", self.lna_nf_spin)
        
        # VGA parameters
        self.vga_gain_spin = QDoubleSpinBox()
        self.vga_gain_spin.setRange(0, 60)
        self.vga_gain_spin.setValue(30)
        self.vga_gain_spin.setSuffix(" dB")
        self.vga_gain_spin.valueChanged.connect(lambda v: self.vga_gain_changed.emit(v))
        params_layout.addRow("VGA - Gain:", self.vga_gain_spin)
        
        # Bottom reflection is now in Environment Parameters, not here
        
        params_panel.setLayout(params_layout)
        
        # Right panel - diagram
        self.diagram = SignalPathDiagram()
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(params_panel)
        splitter.addWidget(self.diagram)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
    
    def set_path_data(self, path_data: Dict):
        """Sets signal path data."""
        self.path_data = path_data
        self.diagram.set_path_data(path_data)
    
    def get_lna_gain(self) -> float:
        """Returns LNA gain."""
        return self.lna_gain_spin.value()
    
    def get_lna_nf(self) -> float:
        """Returns LNA noise figure."""
        return self.lna_nf_spin.value()
    
    def get_vga_gain(self) -> float:
        """Returns VGA gain."""
        return self.vga_gain_spin.value()
    
    # Bottom reflection is now in Environment Parameters, not here
    
    def set_lna_gain(self, value: float):
        """Sets LNA gain."""
        self.lna_gain_spin.setValue(value)
    
    def set_lna_nf(self, value: float):
        """Sets LNA noise figure."""
        self.lna_nf_spin.setValue(value)
    
    def set_vga_gain(self, value: float):
        """Sets VGA gain."""
        self.vga_gain_spin.setValue(value)
    
    # Bottom reflection is now in Environment Parameters, not here

