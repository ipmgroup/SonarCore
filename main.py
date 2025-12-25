#!/usr/bin/env python3
"""
Application entry point.
"""

import sys
from pathlib import Path
 
# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent))

from gui.main_window import main

if __name__ == '__main__':
    main()

