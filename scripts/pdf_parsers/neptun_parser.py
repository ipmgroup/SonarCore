"""
NEPTUN Communications PDF Parser

Parser for NEPTUN Communications transducer PDFs.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

from pdf_parsers import BasePDFParser

logger = logging.getLogger(__name__)

try:
    from neptun_parsing_functions import (
        extract_text_from_pdf,
        parse_frequency_range,
        parse_sensitivity,
        parse_voltage,
        parse_capacitance,
        parse_resonant_frequency,
        parse_beam_angle,
        parse_beam_pattern,
        parse_impedance,
        test_parse_pdf
    )
    NEPTUN_AVAILABLE = True
except ImportError:
    NEPTUN_AVAILABLE = False
    logger.warning("neptun_parsing_functions module not available")


class NeptunParser(BasePDFParser):
    """Parser for NEPTUN Communications transducer PDFs"""
    
    def __init__(self):
        super().__init__()
        self.name = "NEPTUN_COMMUNICATIONS"
        self.description = "Parser for NEPTUN Communications transducer specifications"
    
    def parse(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Parse NEPTUN Communications PDF"""
        if not NEPTUN_AVAILABLE:
            logger.error("neptun_parsing_functions module not available")
            return None
        
        try:
            # Suppress console output
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                raw_results = test_parse_pdf(pdf_path, show_lines=0)
            
            if not raw_results:
                logger.warning("No parameters extracted from PDF")
                return None
            
            # Return results in expected format
            return raw_results
            
        except Exception as e:
            logger.error(f"Error parsing PDF with NeptunParser: {e}", exc_info=True)
            return None


# Parser will be auto-registered when module is imported
# You can also manually register it:
# from pdf_parsers import register_parser
# register_parser(NeptunParser, "NEPTUN_COMMUNICATIONS")
