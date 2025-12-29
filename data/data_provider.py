"""
DataProvider - провайдер данных для CORE и GUI.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging


class DataProvider:
    """
    Провайдер данных для аппаратуры.
    
    Предоставляет доступ к данным о:
    - Трансдусерах
    - LNA
    - VGA
    - ADC
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize data provider.
        
        Args:
            data_dir: Path to data directory (default: data/)
        """
        if data_dir is None:
            # Determine path relative to this file
            current_dir = Path(__file__).parent.parent
            data_dir = current_dir / 'data'
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Cache for loaded data
        self._transducers = {}
        self._lna = {}
        self._vga = {}
        self._adc = {}
    
    def _load_metadata(self) -> Dict:
        """Loads metadata."""
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata: {e}")
        return {'version': '1.0', 'last_updated': ''}
    
    def _load_json(self, file_path: Path) -> Dict:
        """Loads JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def get_transducer(self, transducer_id: str) -> Dict:
        """
        Gets transducer parameters.
        
        Args:
            transducer_id: Transducer ID
        
        Returns:
            Dictionary with parameters
        """
        if transducer_id in self._transducers:
            return self._transducers[transducer_id]
        
        file_path = self.data_dir / 'transducers' / f'{transducer_id}.json'
        if not file_path.exists():
            raise ValueError(f"Transducer {transducer_id} not found")
        
        data = self._load_json(file_path)
        self._transducers[transducer_id] = data
        return data
    
    def get_lna(self, lna_id: str) -> Dict:
        """
        Gets LNA parameters.
        
        Args:
            lna_id: LNA ID
        
        Returns:
            Dictionary with parameters
        """
        if lna_id in self._lna:
            return self._lna[lna_id]
        
        file_path = self.data_dir / 'lna' / f'{lna_id}.json'
        if not file_path.exists():
            raise ValueError(f"LNA {lna_id} not found")
        
        data = self._load_json(file_path)
        self._lna[lna_id] = data
        return data
    
    def get_vga(self, vga_id: str) -> Dict:
        """
        Gets VGA parameters.
        
        Args:
            vga_id: VGA ID
        
        Returns:
            Dictionary with parameters
        """
        if vga_id in self._vga:
            return self._vga[vga_id]
        
        file_path = self.data_dir / 'vga' / f'{vga_id}.json'
        if not file_path.exists():
            raise ValueError(f"VGA {vga_id} not found")
        
        data = self._load_json(file_path)
        self._vga[vga_id] = data
        return data
    
    def get_adc(self, adc_id: str) -> Dict:
        """
        Gets ADC parameters.
        
        Args:
            adc_id: ADC ID
        
        Returns:
            Dictionary with parameters
        """
        if adc_id in self._adc:
            return self._adc[adc_id]
        
        file_path = self.data_dir / 'adc' / f'{adc_id}.json'
        if not file_path.exists():
            raise ValueError(f"ADC {adc_id} not found")
        
        data = self._load_json(file_path)
        self._adc[adc_id] = data
        return data
    
    def list_transducers(self) -> List[str]:
        """Returns list of available transducers."""
        transducers_dir = self.data_dir / 'transducers'
        if not transducers_dir.exists():
            return []
        
        # Exclude service files
        excluded_files = {'index', 'example_transducer'}
        return [f.stem for f in transducers_dir.glob('*.json') 
                if f.stem not in excluded_files]
    
    def list_lna(self) -> List[str]:
        """Returns list of available LNA."""
        lna_dir = self.data_dir / 'lna'
        if not lna_dir.exists():
            return []
        
        return [f.stem for f in lna_dir.glob('*.json')]
    
    def list_vga(self) -> List[str]:
        """Returns list of available VGA."""
        vga_dir = self.data_dir / 'vga'
        if not vga_dir.exists():
            return []
        
        return [f.stem for f in vga_dir.glob('*.json')]
    
    def list_adc(self) -> List[str]:
        """Returns list of available ADC."""
        adc_dir = self.data_dir / 'adc'
        if not adc_dir.exists():
            return []
        
        return [f.stem for f in adc_dir.glob('*.json')]
    
    def get_metadata(self) -> Dict:
        """Returns metadata."""
        return self.metadata.copy()

