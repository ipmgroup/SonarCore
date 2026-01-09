"""
250 SERIES PDF Parser

Parser for 250 SERIES side-scan transducer PDFs.
Extracts parameters from TECHNICAL SPECIFICATION table.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

from pdf_parsers import BasePDFParser

logger = logging.getLogger(__name__)

try:
    from series250_parsing_functions import (
        extract_text_from_pdf,
        parse_frequency_options,
        parse_beam_angle,
        parse_sensitivity,
        parse_bandwidth,
        parse_voltage,
        test_parse_pdf
    )
    SERIES250_AVAILABLE = True
except ImportError:
    SERIES250_AVAILABLE = False
    logger.warning("series250_parsing_functions module not available")


class Series250Parser(BasePDFParser):
    """Parser for 250 SERIES transducer PDFs"""
    
    def __init__(self):
        super().__init__()
        self.name = "250_SERIES"
        self.description = "Parser for 250 SERIES side-scan transducer specifications"
    
    def parse(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Parse 250 SERIES PDF"""
        if not SERIES250_AVAILABLE:
            logger.error("series250_parsing_functions module not available")
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
            
            # Ensure all expected keys are present
            # Return data for first model (D1) as primary, but include both models info
            results = {
                'f_0': raw_results.get('f_0'),
                'f_min': raw_results.get('f_min'),
                'f_max': raw_results.get('f_max'),
                'bandwidth_min': raw_results.get('bandwidth_min'),  # Bandwidth from PDF
                'bandwidth_max': raw_results.get('bandwidth_max'),  # Bandwidth from PDF
                'tx_sensitivity': raw_results.get('tx_sensitivity'),
                'rx_sensitivity': raw_results.get('rx_sensitivity'),
                'capacitance': raw_results.get('capacitance'),
                'v_max': raw_results.get('v_max'),
                'beam_angle': raw_results.get('beam_angle'),
                'beam_pattern_horizontal': raw_results.get('beam_pattern_horizontal'),
                'beam_pattern_vertical': raw_results.get('beam_pattern_vertical'),
                'impedance': raw_results.get('impedance'),
                # Additional info about multiple models
                'models': raw_results.get('models', {}),  # Will contain D1 and D2 data if available
                'model_count': raw_results.get('model_count', 1)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error parsing PDF with Series250Parser: {e}", exc_info=True)
            return None


# Parser will be auto-registered when module is imported
