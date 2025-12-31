#!/usr/bin/env python3
"""
Main entry point for Sub-Bottom Profiler GUI.
"""

import sys
import logging
from pathlib import Path

# Setup logging
# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'simulator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Add root directory to path
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from PyQt5.QtWidgets import QApplication
from gui.sbp_window import SBPWindow


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = SBPWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

