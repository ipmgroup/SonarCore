"""
Example PDF Parser

This is an example parser showing how to create a new PDF parser.
Copy this file and modify it to create your own parser.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import re

from pdf_parsers import BasePDFParser, register_parser

logger = logging.getLogger(__name__)


class ExampleParser(BasePDFParser):
    """
    Example parser for demonstration purposes.
    
    This parser shows the basic structure of a PDF parser.
    Modify the parse() method to implement your specific parsing logic.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "ExampleParser"
        self.description = "Example parser for demonstration (modify for your needs)"
    
    def parse(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse transducer parameters from PDF.
        
        This is a template - modify it to match your PDF format.
        """
        try:
            # Extract text from PDF
            text = self.extract_text(pdf_path)
            if not text:
                logger.warning(f"Could not extract text from {pdf_path}")
                return None
            
            # Initialize results dictionary
            results = {
                'f_0': None,
                'f_min': None,
                'f_max': None,
                'tx_sensitivity': None,
                'rx_sensitivity': None,
                'capacitance': None,
                'v_max': None,
                'beam_angle': None,
                'beam_pattern_horizontal': None,
                'beam_pattern_vertical': None,
                'impedance': None
            }
            
            # Split text into lines for processing
            lines = [line.strip() for line in text.split('\n')]
            
            # Example: Search for resonant frequency
            # Modify this pattern to match your PDF format
            for i, line in enumerate(lines):
                line_lower = line.lower()
                
                # Example pattern matching
                if 'resonant frequency' in line_lower:
                    # Extract frequency value (modify regex to match your format)
                    match = re.search(r'(\d+\.?\d*)\s*khz', line, re.IGNORECASE)
                    if match:
                        freq_khz = float(match.group(1))
                        results['f_0'] = freq_khz * 1000  # Convert to Hz
                        logger.info(f"Found f_0: {results['f_0']} Hz")
                
                # Add more parsing logic here for other parameters
                # ...
            
            # Return None if no parameters were found
            if all(v is None for v in results.values()):
                logger.warning("No parameters extracted from PDF")
                return None
            
            return results
            
        except Exception as e:
            logger.error(f"Error parsing PDF with ExampleParser: {e}", exc_info=True)
            return None


# Uncomment the line below to register this parser
# register_parser(ExampleParser, "ExampleParser")
