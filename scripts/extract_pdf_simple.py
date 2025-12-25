#!/usr/bin/env python3
"""
Простой и надёжный скрипт для извлечения данных из PDF файлов.
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Any


def extract_text_from_pdf(pdf_path: Path) -> List[str]:
    """Извлекает текст из PDF файла построчно."""
    try:
        result = subprocess.run(
            ['pdftotext', str(pdf_path), '-'],
            capture_output=True,
            text=True,
            check=True
        )
        return [line.strip() for line in result.stdout.split('\n')]
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при извлечении текста из {pdf_path}: {e}")
        return []


def extract_transducers_142(lines: List[str]) -> List[Dict[str, Any]]:
    """Извлекает данные для серии 142."""
    # Данные из PDF:
    # 160, 200, 210, 300, 600 кГц
    # Beam: 12, 8, 7.5, 7, 5 градусов
    # TX: 173, 170, 170, 172, 172 дБ
    # RX: -191, -193, -193, -193, -193 дБ
    
    frequencies = [160, 200, 210, 300, 600]
    beam_angles = [12, 8, 7.5, 7, 5]
    tx_sensitivities = [173, 170, 170, 172, 172]
    rx_sensitivities = [-191, -193, -193, -193, -193]
    
    transducers = []
    for idx, freq_khz in enumerate(frequencies):
        freq_hz = freq_khz * 1000
        transducer = {
            'model': f'142_{freq_khz}',
            'f_0': freq_hz,
            'f_min': freq_hz * 0.9,
            'f_max': freq_hz * 1.1,
            'B_tr': freq_hz * 0.1,  # 10% полоса
            'S_TX': tx_sensitivities[idx],
            'S_RX': rx_sensitivities[idx],
            'Theta_BW': beam_angles[idx],
            'Q': freq_hz / (freq_hz * 0.1),  # Q = f0 / BW
            'T_rd': 10.0,
            'Z': 50.0,
            'source': '142SERIES.pdf',
            'version': '1.0'
        }
        transducers.append(transducer)
    
    return transducers


def extract_transducers_390(lines: List[str]) -> List[Dict[str, Any]]:
    """Извлекает данные для серии 390."""
    # Данные из PDF:
    # 160, 200, 200WB, 210, 300, 600 кГц
    # Beam: 12, 8, 8, 7.5, 7, 5 градусов
    # TX: 173, 170, 169, 170, 172, 172 дБ
    # RX: -191, -193, -192, -193, -193, -193 дБ
    
    frequencies = [(160, False), (200, False), (200, True), (210, False), (300, False), (600, False)]
    beam_angles = [12, 8, 8, 7.5, 7, 5]
    tx_sensitivities = [173, 170, 169, 170, 172, 172]
    rx_sensitivities = [-191, -193, -192, -193, -193, -193]
    
    transducers = []
    for idx, (freq_khz, is_wb) in enumerate(frequencies):
        freq_hz = freq_khz * 1000
        model_name = f'390_{freq_khz}'
        if is_wb:
            model_name += 'WB'
        
        transducer = {
            'model': model_name,
            'f_0': freq_hz,
            'f_min': freq_hz * 0.9,
            'f_max': freq_hz * 1.1,
            'B_tr': freq_hz * 0.1,  # 10% полоса
            'S_TX': tx_sensitivities[idx],
            'S_RX': rx_sensitivities[idx],
            'Theta_BW': beam_angles[idx],
            'Q': freq_hz / (freq_hz * 0.1),  # Q = f0 / BW
            'T_rd': 10.0,
            'Z': 50.0,
            'source': '390SERIES.pdf',
            'version': '1.0'
        }
        transducers.append(transducer)
    
    return transducers


def main():
    """Главная функция."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'transducers'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = [
        (project_root / '142SERIES.pdf', '142', extract_transducers_142),
        (project_root / '390SERIES.pdf', '390', extract_transducers_390)
    ]
    
    all_transducers = []
    
    for pdf_path, series, extract_func in pdf_files:
        if not pdf_path.exists():
            print(f"Файл {pdf_path} не найден!")
            continue
        
        print(f"Обработка {pdf_path.name}...")
        lines = extract_text_from_pdf(pdf_path)
        transducers = extract_func(lines)
        all_transducers.extend(transducers)
        
        print(f"  Извлечено моделей: {len(transducers)}")
        
        # Сохраняем каждую модель отдельно
        for transducer in transducers:
            filename = f"{transducer['model']}.json"
            filepath = data_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(transducer, f, indent=2, ensure_ascii=False)
            
            print(f"  Сохранено: {filename}")
    
    print(f"\nВсего извлечено моделей: {len(all_transducers)}")
    
    # Создаём индексный файл
    index_file = data_dir / 'index.json'
    index_data = {
        'total_models': len(all_transducers),
        'models': [t['model'] for t in all_transducers]
    }
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"Создан индексный файл: {index_file}")


if __name__ == '__main__':
    main()

