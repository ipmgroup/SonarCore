#!/usr/bin/env python3
"""
Вспомогательные функции для парсинга PDF 250 SERIES.
Позволяет отлаживать и улучшать парсинг параметров трансдусеров 250 SERIES.
"""

import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import subprocess
import json


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Извлекает текст из PDF файла используя pdftotext."""
    try:
        result = subprocess.run(
            ['pdftotext', str(pdf_path), '-'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при извлечении текста из {pdf_path}: {e}")
        return ""


def parse_frequency_options(lines: List[str], start_idx: int) -> Tuple[Optional[float], Optional[float], Optional[float], List[float]]:
    """Parse frequency options from table like:
    Frequency Options
    115
    500
    kHz
    
    Returns: (f_0, f_min, f_max, [model1_freq, model2_freq]) in Hz, or (None, None, None, []) if not found
    """
    for i in range(start_idx, min(start_idx + 20, len(lines))):
        line_lower = lines[i].lower()
        if 'frequency options' in line_lower:
            # Look for numbers in next few lines
            values = []
            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    continue
                
                # Check if we hit the unit
                if 'khz' in next_line.lower():
                    break
                
                # Extract numbers
                numbers = re.findall(r'\d+\.?\d*', next_line)
                for num_str in numbers:
                    try:
                        val = float(num_str)
                        if 1 <= val <= 10000:  # Reasonable frequency range in kHz
                            values.append(val * 1000)  # Convert to Hz
                    except ValueError:
                        continue
            
            if values:
                if len(values) == 1:
                    freq_hz = values[0]
                    return (freq_hz, freq_hz, freq_hz, values)
                elif len(values) >= 2:
                    # First value as f_0, range as f_min-f_max, all values as models
                    return (values[0], min(values), max(values), values)
    
    return (None, None, None, [])


def parse_beam_angle(lines: List[str], start_idx: int, beam_type: str = 'horizontal') -> Tuple[Optional[float], List[float]]:
    """Parse beam angle from table like:
    Horizontal Beam (-3dB)
    1.5
    0.4
    Degrees Conical
    
    Returns: (first_angle, [all_angles]) in degrees, or (None, []) if not found
    """
    beam_keywords = {
        'horizontal': ['horizontal beam'],
        'vertical': ['vertical beam']
    }
    
    keywords = beam_keywords.get(beam_type.lower(), ['beam'])
    
    for i in range(start_idx, min(start_idx + 100, len(lines))):
        line_lower = lines[i].lower()
        if any(keyword in line_lower for keyword in keywords):
            # Look for numbers in next few lines (skip empty lines)
            values = []
            for j in range(i + 1, min(i + 15, len(lines))):
                next_line = lines[j].strip()
                
                # Check if we hit the unit (stop searching)
                if 'degrees' in next_line.lower() or 'deg' in next_line.lower():
                    break
                
                # Skip empty lines but continue searching
                if not next_line:
                    continue
                
                # Extract all valid numbers
                numbers = re.findall(r'\d+\.?\d*', next_line)
                for num_str in numbers:
                    try:
                        val = float(num_str)
                        if 0.1 <= val <= 180:  # Reasonable beam angle
                            values.append(val)
                    except ValueError:
                        continue
            
            if values:
                return (values[0], values)
    
    return (None, [])


def parse_sensitivity(lines: List[str], start_idx: int, is_rx: bool = False) -> Tuple[Optional[float], List[float]]:
    """Parse sensitivity from table like:
    Transmit Sensitivity
    171
    171
    dB re 1µPa/V @ 1m
    
    or
    
    Receive Sensitivity
    -183
    -203
    dB re 1V/µPa
    
    Returns: sensitivity value in dB, or None if not found
    """
    sensitivity_keywords = {
        False: ['transmit sensitivity'],
        True: ['receive sensitivity']
    }
    
    keywords = sensitivity_keywords.get(is_rx, ['sensitivity'])
    
    for i in range(start_idx, min(start_idx + 100, len(lines))):
        line_lower = lines[i].lower()
        if any(keyword in line_lower for keyword in keywords):
            # Look for numbers in next few lines (skip empty lines)
            values = []
            for j in range(i + 1, min(i + 15, len(lines))):
                next_line = lines[j].strip()
                
                # Check if we hit the unit (stop searching)
                if 'db' in next_line.lower():
                    break
                
                # Skip empty lines but continue searching
                if not next_line:
                    continue
                
                # Extract numbers (can be negative for RX)
                numbers = re.findall(r'-?\d+\.?\d*', next_line)
                for num_str in numbers:
                    try:
                        val = float(num_str)
                        if is_rx:
                            if -300 <= val <= -50:  # RX sensitivity range
                                values.append(val)
                        else:
                            if 50 <= val <= 250:  # TX sensitivity range
                                values.append(val)
                    except ValueError:
                        continue
            
            if values:
                # Return first value and all values
                return (values[0], values)
    
    return (None, [])


def parse_bandwidth(lines: List[str], start_idx: int) -> Tuple[Optional[float], List[float]]:
    """Parse bandwidth from table like:
    Bandwidth
    40
    100
    kHz
    
    Returns: (first_bandwidth, [all_bandwidths]) in Hz, or (None, []) if not found
    """
    for i in range(start_idx, min(start_idx + 100, len(lines))):
        line_lower = lines[i].lower()
        if 'bandwidth' in line_lower:
            # Look for numbers in next few lines (skip empty lines)
            values = []
            for j in range(i + 1, min(i + 15, len(lines))):
                next_line = lines[j].strip()
                
                # Check if we hit the unit (stop searching)
                if 'khz' in next_line.lower():
                    break
                
                # Skip empty lines but continue searching
                if not next_line:
                    continue
                
                # Extract numbers
                numbers = re.findall(r'\d+\.?\d*', next_line)
                for num_str in numbers:
                    try:
                        val = float(num_str)
                        if 1 <= val <= 10000:  # Reasonable bandwidth range in kHz
                            values.append(val * 1000)  # Convert to Hz
                    except ValueError:
                        continue
            
            if values:
                return (values[0], values)
    
    return (None, [])


def parse_voltage(lines: List[str], start_idx: int) -> Tuple[Optional[float], List[float]]:
    """Parse voltage from table like:
    Transmit Voltage / Duty Cycle (Abs. Max)
    500
    300
    Vrms at 10%
    
    Returns: voltage in V, or None if not found
    """
    for i in range(start_idx, min(start_idx + 100, len(lines))):
        line_lower = lines[i].lower()
        if 'voltage' in line_lower or ('transmit' in line_lower and 'voltage' in line_lower):
            # Look for numbers in next few lines (skip empty lines)
            values = []
            for j in range(i + 1, min(i + 15, len(lines))):
                next_line = lines[j].strip()
                
                # Check if we hit the unit (stop searching)
                if 'vrms' in next_line.lower() or ('v' in next_line.lower() and 'rms' in next_line.lower()):
                    break
                
                # Skip empty lines but continue searching
                if not next_line:
                    continue
                
                # Extract numbers
                numbers = re.findall(r'\d+\.?\d*', next_line)
                for num_str in numbers:
                    try:
                        val = float(num_str)
                        if 10 <= val <= 1000:  # Reasonable voltage range
                            values.append(val)
                    except ValueError:
                        continue
            
            if values:
                # Return maximum value and all values
                return (max(values), values)
    
    return (None, [])


def test_parse_pdf(pdf_path: Path, show_lines: int = 30):
    """Тестирует парсинг PDF файла и показывает результаты."""
    print(f"\n{'='*60}")
    print(f"Тестирование парсинга: {pdf_path.name}")
    print(f"{'='*60}")
    
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("Не удалось извлечь текст из PDF")
        return None
    
    lines = [line.strip() for line in text.split('\n')]
    
    print(f"\nПервые {show_lines} строк:")
    for i, line in enumerate(lines[:show_lines]):
        print(f"  {i:3d}: {line}")
    
    # Find TECHNICAL SPECIFICATION section
    spec_start = -1
    for i, line in enumerate(lines):
        if 'technical specification' in line.lower():
            spec_start = i
            break
    
    if spec_start == -1:
        print("\nНе найдена секция TECHNICAL SPECIFICATION")
        return None
    
    print(f"\nНайдена секция TECHNICAL SPECIFICATION на строке {spec_start}")
    
    # Ищем параметры
    results = {
        'f_0': None,
        'f_min': None,
        'f_max': None,
        'bandwidth_min': None,  # Bandwidth from separate field
        'bandwidth_max': None,  # Bandwidth from separate field
        'tx_sensitivity': None,
        'rx_sensitivity': None,
        'v_max': None,
        'beam_angle': None,
        'beam_pattern_horizontal': None,
        'beam_pattern_vertical': None,
        'models': {},  # Will store data for both models
        'model_count': 0
    }
    
    # Store values for both models
    model_data = {'D1': {}, 'D2': {}}
    
    # Parse frequency options (returns values for both models)
    # Frequency Options в PDF содержит только частоты для моделей (115 и 500 kHz)
    # Это f_0 для каждой модели, не f_min/f_max
    f_0, f_min_from_freq, f_max_from_freq, freq_values = parse_frequency_options(lines, spec_start)
    if f_0:
        results['f_0'] = f_0
        print(f"\n✓ Найдена частота f_0: {f_0/1000:.1f} kHz")
        if len(freq_values) >= 2:
            print(f"  Модели: {[f'{v/1000:.0f} kHz' for v in freq_values]}")
            model_data['D1']['f_0'] = freq_values[0]
            model_data['D2']['f_0'] = freq_values[1]
            results['model_count'] = 2
    # Не используем f_min/f_max из parse_frequency_options - они вычисляются неправильно
    # f_min/f_max будут вычислены из f_0 и bandwidth для каждой модели
    # Parse bandwidth (this is the -3dB bandwidth for each model)
    # Bandwidth values are model-specific: 40 kHz for D1, 100 kHz for D2 (or 60 for D2 as user mentioned)
    # f_min and f_max will be calculated for each model as: f_0 ± bandwidth/2
    bw_first, bw_values = parse_bandwidth(lines, spec_start)
    if bw_first and bw_values:
        print(f"\n✓ Найден bandwidth: {[f'{v/1000:.0f} kHz' for v in bw_values]}")
        # Store bandwidth for each model
        if len(bw_values) >= 2 and results.get('model_count', 0) >= 2:
            model_keys = sorted(model_data.keys())
            for i, model_key in enumerate(model_keys):
                if i < len(bw_values):
                    bandwidth_val = bw_values[i]
                    model_data[model_key]['bandwidth'] = bandwidth_val
                    print(f"  {model_key}: bandwidth={bandwidth_val/1000:.0f} kHz")
        elif len(bw_values) >= 1:
            # Single bandwidth value
            results['bandwidth_min'] = min(bw_values)
            results['bandwidth_max'] = max(bw_values)
    
    # Calculate f_min and f_max for each model: f_0 ± bandwidth/2
    if results.get('model_count', 0) >= 2:
        # Calculate f_min/f_max for each model based on their f_0 and bandwidth
        for model_key in model_data.keys():
            if 'f_0' in model_data[model_key]:
                f_0 = model_data[model_key]['f_0']
                # Get bandwidth for this model
                bandwidth = model_data[model_key].get('bandwidth')
                if bandwidth:
                    half_bandwidth = bandwidth / 2
                    model_data[model_key]['f_min'] = f_0 - half_bandwidth
                    model_data[model_key]['f_max'] = f_0 + half_bandwidth
                    print(f"  {model_key}: f_0={f_0/1000:.1f} kHz, bandwidth={bandwidth/1000:.0f} kHz, f_min={model_data[model_key]['f_min']/1000:.1f} kHz, f_max={model_data[model_key]['f_max']/1000:.1f} kHz")
        
        # Set overall f_min/f_max as range across all models
        all_f_mins = [m.get('f_min') for m in model_data.values() if 'f_min' in m]
        all_f_maxs = [m.get('f_max') for m in model_data.values() if 'f_max' in m]
        if all_f_mins and all_f_maxs:
            results['f_min'] = min(all_f_mins)
            results['f_max'] = max(all_f_maxs)
            print(f"  Общий диапазон: f_min={results['f_min']/1000:.1f} kHz, f_max={results['f_max']/1000:.1f} kHz")
    elif bw_first and results.get('f_0'):
        # Single model case
        f_0 = results['f_0']
        bandwidth = bw_values[0] if bw_values else None
        if bandwidth:
            half_bandwidth = bandwidth / 2
            results['f_min'] = f_0 - half_bandwidth
            results['f_max'] = f_0 + half_bandwidth
            print(f"  f_min={results['f_min']/1000:.1f} kHz, f_max={results['f_max']/1000:.1f} kHz")
    else:
        # Fallback: если bandwidth не найден, используем значения из frequency options
        # Но это не идеально, так как frequency options - это f_0, а не f_min/f_max
        if f_min_from_freq:
            results['f_min'] = f_min_from_freq
            print(f"✓ Найдена минимальная частота (из Frequency Options): {f_min_from_freq/1000:.1f} kHz")
        if f_max_from_freq:
            results['f_max'] = f_max_from_freq
            print(f"✓ Найдена максимальная частота (из Frequency Options): {f_max_from_freq/1000:.1f} kHz")
    
    # Parse horizontal beam (returns values for both models)
    h_beam, h_beam_values = parse_beam_angle(lines, spec_start, 'horizontal')
    if h_beam:
        results['beam_angle'] = h_beam
        results['beam_pattern_horizontal'] = {
            'pattern': 'directional',
            'angle': h_beam,
            'type': 'horizontal'
        }
        print(f"\n✓ Найден горизонтальный луч: {h_beam:.2f}°")
        if len(h_beam_values) >= 2:
            print(f"  Модели: {[f'{v:.2f}°' for v in h_beam_values]}")
            model_data['D1']['beam_horizontal'] = h_beam_values[0]
            model_data['D2']['beam_horizontal'] = h_beam_values[1]
    
    # Parse vertical beam (returns values for both models)
    v_beam, v_beam_values = parse_beam_angle(lines, spec_start, 'vertical')
    if v_beam:
        results['beam_pattern_vertical'] = {
            'pattern': 'directional',
            'angle': v_beam,
            'type': 'vertical'
        }
        print(f"\n✓ Найден вертикальный луч: {v_beam:.2f}°")
        if len(v_beam_values) >= 2:
            print(f"  Модели: {[f'{v:.2f}°' for v in v_beam_values]}")
            model_data['D1']['beam_vertical'] = v_beam_values[0]
            model_data['D2']['beam_vertical'] = v_beam_values[1]
    
    # Parse TX sensitivity (returns values for both models)
    tx_sens, tx_sens_values = parse_sensitivity(lines, spec_start, is_rx=False)
    if tx_sens:
        results['tx_sensitivity'] = tx_sens
        print(f"\n✓ Найдена TX чувствительность: {tx_sens:.1f} dB")
        if len(tx_sens_values) >= 2:
            print(f"  Модели: {[f'{v:.1f} dB' for v in tx_sens_values]}")
            model_data['D1']['tx_sensitivity'] = tx_sens_values[0]
            model_data['D2']['tx_sensitivity'] = tx_sens_values[1]
    
    # Parse RX sensitivity (returns values for both models)
    rx_sens, rx_sens_values = parse_sensitivity(lines, spec_start, is_rx=True)
    if rx_sens:
        results['rx_sensitivity'] = rx_sens
        print(f"\n✓ Найдена RX чувствительность: {rx_sens:.1f} dB")
        if len(rx_sens_values) >= 2:
            print(f"  Модели: {[f'{v:.1f} dB' for v in rx_sens_values]}")
            model_data['D1']['rx_sensitivity'] = rx_sens_values[0]
            model_data['D2']['rx_sensitivity'] = rx_sens_values[1]
    
    # Parse voltage (returns values for both models)
    voltage, voltage_values = parse_voltage(lines, spec_start)
    if voltage:
        results['v_max'] = voltage
        print(f"\n✓ Найдено напряжение: {voltage:.0f} Vrms")
        if len(voltage_values) >= 2:
            print(f"  Модели: {[f'{v:.0f} Vrms' for v in voltage_values]}")
            model_data['D1']['v_max'] = voltage_values[0]
            model_data['D2']['v_max'] = voltage_values[1]
    
    # Store models data if we have data for both models
    if results.get('model_count', 0) >= 2:
        results['models'] = model_data
    
    # Выводим итоговые результаты
    print(f"\n{'='*60}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"{'='*60}")
    for key, value in results.items():
        if value is not None:
            if key in ['f_0', 'f_min', 'f_max']:
                print(f"  {key:20s}: {value:,.0f} Hz ({value/1000:.1f} kHz)")
            elif key in ['tx_sensitivity', 'rx_sensitivity']:
                print(f"  {key:20s}: {value:.1f} dB")
            elif key == 'v_max':
                print(f"  {key:20s}: {value:.0f} Vrms")
            elif key == 'beam_angle':
                print(f"  {key:20s}: {value:.2f}°")
            elif key in ['beam_pattern_horizontal', 'beam_pattern_vertical']:
                pattern_str = value.get('pattern', 'unknown')
                if 'angle' in value:
                    pattern_str += f" {value['angle']}°"
                print(f"  {key:20s}: {pattern_str}")
            else:
                print(f"  {key:20s}: {value}")
        else:
            print(f"  {key:20s}: НЕ НАЙДЕНО")
    
    return results


def main():
    """Главная функция для тестирования."""
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python series250_parsing_functions.py <pdf_file>")
        print("\nПримеры:")
        print("  python series250_parsing_functions.py scripts/250SERIES.pdf")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Файл не найден: {pdf_path}")
        sys.exit(1)
    
    results = test_parse_pdf(pdf_path)
    
    # Сохраняем результаты в JSON
    if results:
        output_file = pdf_path.with_suffix('.parsed.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nРезультаты сохранены в: {output_file}")


if __name__ == '__main__':
    main()
