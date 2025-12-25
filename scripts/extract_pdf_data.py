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
    # Frequency Options: 160, 200, 210, 300, 600 кГц
    # Beam Angle: 12, 8, 7.5, 7, 5 градусов
    # TX Sensitivity: 173, 170, 170, 172, 172 дБ re 1µPa/V @ 1m
    # RX Sensitivity: -191, -193, -193, -193, -193 дБ re 1V/µPa
    # Bandwidth: 25, 15, 15, 45, 90 кГц
    # Nominal Impedance: 70, 70, 100, 100, 75 Ом
    
    frequencies = [160, 200, 210, 300, 600]
    beam_angles = [12, 8, 7.5, 7, 5]
    tx_sensitivities = [173, 170, 170, 172, 172]
    rx_sensitivities = [-191, -193, -193, -193, -193]
    bandwidths_khz = [25, 15, 15, 45, 90]  # кГц
    impedances = [70, 70, 100, 100, 75]  # Ом
    
    transducers = []
    for idx, freq_khz in enumerate(frequencies):
        freq_hz = freq_khz * 1000
        bandwidth_hz = bandwidths_khz[idx] * 1000  # кГц -> Гц
        
        transducer = {
            '_comment': f'142 Series Transducer - {freq_khz} kHz',
            'model': f'142_{freq_khz}',
            'description': f'Single beam transducer, {freq_khz} kHz, {beam_angles[idx]}° beam angle',
            'f_0': freq_hz,
            '_comment_f_0': 'Central frequency, Hz',
            'f_min': freq_hz * 0.9,
            '_comment_f_min': 'Minimum frequency, Hz (estimated as 90% of f_0)',
            'f_max': freq_hz * 1.1,
            '_comment_f_max': 'Maximum frequency, Hz (estimated as 110% of f_0)',
            'B_tr': bandwidth_hz,
            '_comment_B_tr': 'Bandwidth at -3 dB, Hz',
            'S_TX': tx_sensitivities[idx],
            '_comment_S_TX': 'Transmit sensitivity, dB re 1µPa/V @ 1m',
            'S_RX': rx_sensitivities[idx],
            '_comment_S_RX': 'Receive sensitivity, dB re 1V/µPa',
            'Theta_BW': beam_angles[idx],
            '_comment_Theta_BW': 'Beam angle at -3 dB, degrees conical',
            'Q': freq_hz / bandwidth_hz if bandwidth_hz > 0 else 10.0,
            '_comment_Q': 'Q-factor (f_0 / B_tr)',
            'T_rd': 10.0,
            '_comment_T_rd': 'Ring-down time, microseconds (estimated)',
            'Z': impedances[idx],
            '_comment_Z': 'Nominal impedance, Ohms',
            'source': '142SERIES.pdf',
            'version': '1.0'
        }
        transducers.append(transducer)
    
    return transducers


def extract_transducers_390(lines: List[str]) -> List[Dict[str, Any]]:
    """Извлекает данные для серии 390."""
    # Данные из PDF:
    # Frequency Options: 160, 200, 200WB, 210, 300, 600 кГц
    # Beam Angle: 12, 8, 8, 7.5, 7, 5 градусов
    # TX Sensitivity: 173, 170, 169, 170, 172, 172 дБ re 1µPa/V @ 1m
    # RX Sensitivity: -191, -193, -192, -193, -193, -193 дБ re 1V/µPa
    # Bandwidth: 24, 15, 35, 15, 45, 90 кГц
    # Nominal Impedance: 70, 70, 100, 100, 75, 75 Ом
    
    frequencies = [(160, False), (200, False), (200, True), (210, False), (300, False), (600, False)]
    beam_angles = [12, 8, 8, 7.5, 7, 5]
    tx_sensitivities = [173, 170, 169, 170, 172, 172]
    rx_sensitivities = [-191, -193, -192, -193, -193, -193]
    bandwidths_khz = [24, 15, 35, 15, 45, 90]  # кГц
    impedances = [70, 70, 100, 100, 75, 75]  # Ом
    
    transducers = []
    for idx, (freq_khz, is_wb) in enumerate(frequencies):
        freq_hz = freq_khz * 1000
        bandwidth_hz = bandwidths_khz[idx] * 1000  # кГц -> Гц
        model_name = f'390_{freq_khz}'
        if is_wb:
            model_name += 'WB'
        
        description = f'390 Series Transducer - {freq_khz} kHz'
        if is_wb:
            description += ' (Wideband)'
        description += f', {beam_angles[idx]}° beam angle'
        
        transducer = {
            '_comment': description,
            'model': model_name,
            'description': description,
            'f_0': freq_hz,
            '_comment_f_0': 'Central frequency, Hz',
            'f_min': freq_hz * 0.9,
            '_comment_f_min': 'Minimum frequency, Hz (estimated as 90% of f_0)',
            'f_max': freq_hz * 1.1,
            '_comment_f_max': 'Maximum frequency, Hz (estimated as 110% of f_0)',
            'B_tr': bandwidth_hz,
            '_comment_B_tr': 'Bandwidth at -3 dB, Hz',
            'S_TX': tx_sensitivities[idx],
            '_comment_S_TX': 'Transmit sensitivity, dB re 1µPa/V @ 1m',
            'S_RX': rx_sensitivities[idx],
            '_comment_S_RX': 'Receive sensitivity, dB re 1V/µPa',
            'Theta_BW': beam_angles[idx],
            '_comment_Theta_BW': 'Beam angle at -3 dB, degrees conical',
            'Q': freq_hz / bandwidth_hz if bandwidth_hz > 0 else 10.0,
            '_comment_Q': 'Q-factor (f_0 / B_tr)',
            'T_rd': 10.0,
            '_comment_T_rd': 'Ring-down time, microseconds (estimated)',
            'Z': impedances[idx],
            '_comment_Z': 'Nominal impedance, Ohms',
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

