"""
PDF Parser System for Transducer Parameters

This module provides an extensible system for parsing transducer parameters from PDF files.
To add a new parser:
1. Create a new parser class inheriting from BasePDFParser
2. Implement the parse() method
3. Register it using register_parser() or use the @register_parser decorator
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Parser registry
_parsers: Dict[str, type] = {}


class BasePDFParser(ABC):
    """Base class for PDF parsers"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.description = "PDF Parser"
    
    @abstractmethod
    def parse(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse transducer parameters from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted parameters or None if parsing failed.
            Expected keys (all optional):
            - f_0: Resonant frequency (Hz)
            - f_min: Minimum frequency (Hz)
            - f_max: Maximum frequency (Hz)
            - tx_sensitivity: Transmit sensitivity (dB)
            - rx_sensitivity: Receive sensitivity (dB)
            - capacitance: Capacitance (Farads)
            - v_max: Maximum voltage (Vrms)
            - beam_angle: Beam angle (degrees)
            - beam_pattern_horizontal: Dict with pattern info
            - beam_pattern_vertical: Dict with pattern info
            - impedance: Impedance (Ohms)
        """
        pass
    
    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file. Can be overridden for custom extraction.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            import subprocess
            result = subprocess.run(
                ['pdftotext', str(pdf_path), '-'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            return ""


def register_parser(parser_class: type, name: Optional[str] = None):
    """
    Register a parser class.
    
    Args:
        parser_class: Parser class (must inherit from BasePDFParser)
        name: Optional name for the parser (defaults to class name)
    """
    if not issubclass(parser_class, BasePDFParser):
        raise TypeError(f"Parser must inherit from BasePDFParser, got {parser_class}")
    
    parser_name = name or parser_class.__name__
    _parsers[parser_name] = parser_class
    logger.info(f"Registered parser: {parser_name}")


def get_parser(name: str) -> Optional[BasePDFParser]:
    """
    Get a parser instance by name.
    
    Args:
        name: Parser name
        
    Returns:
        Parser instance or None if not found
    """
    if name not in _parsers:
        return None
    
    try:
        return _parsers[name]()
    except Exception as e:
        logger.error(f"Failed to instantiate parser {name}: {e}")
        return None


def list_parsers() -> list:
    """
    Get list of available parser names.
    
    Returns:
        List of parser names
    """
    return list(_parsers.keys())


def get_parser_info(name: str) -> Optional[Dict[str, str]]:
    """
    Get parser information.
    
    Args:
        name: Parser name
        
    Returns:
        Dict with 'name' and 'description' or None if not found
    """
    if name not in _parsers:
        return None
    
    try:
        parser = _parsers[name]()
        return {
            'name': parser.name,
            'description': parser.description
        }
    except Exception:
        return None


# Auto-register parsers
def _auto_register_parsers():
    """Automatically discover and register parsers"""
    import importlib
    import pkgutil
    
    # Get the directory of this package
    package_path = Path(__file__).parent
    
    # Find all Python files in the package (except __init__.py)
    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.name == '__init__':
            continue
        
        try:
            # Import module
            full_module_name = f'{__name__}.{module_info.name}'
            module = importlib.import_module(full_module_name)
            
            # Look for classes that inherit from BasePDFParser
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePDFParser) and 
                    attr != BasePDFParser):
                    # Check if already registered
                    if attr_name not in _parsers:
                        register_parser(attr)
                        logger.info(f"Auto-registered parser: {attr_name}")
        except Exception as e:
            logger.warning(f"Failed to auto-register parser from {module_info.name}: {e}")


def load_all_parsers():
    """
    Explicitly load all available parsers.
    Call this function to ensure all parsers are registered.
    """
    _auto_register_parsers()


# Try to auto-register parsers (will be called when module is imported)
try:
    _auto_register_parsers()
except Exception as e:
    logger.warning(f"Failed to auto-register parsers: {e}")
