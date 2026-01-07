#!/usr/bin/env python3
"""
Вспомогательные функции для тестирования парсинга PDF без GUI.
Позволяет отлаживать и улучшать парсинг параметров трансдусеров.
"""

import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
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


def parse_frequency_range(line: str, next_line: str = None) -> Tuple[Optional[float], Optional[float]]:
    """Parse frequency range from line like '16 kHz to 30 kHz' or 'Band16 kHz to 30 kHz'
    Returns: (f_min, f_max) in Hz, or (None, None) if not found
    """
    # Pattern 1: "Band16 kHz to 30 kHz" (no space after "Band")
    match = re.search(r'band(\d+\.?\d*)\s*khz\s*(?:to|-)\s*(\d+\.?\d*)\s*khz', line, re.IGNORECASE)
    if match:
        return (float(match.group(1)) * 1000, float(match.group(2)) * 1000)
    
    # Pattern 2: "Band 16 kHz to 30 kHz" (with space)
    match = re.search(r'band\s+(\d+\.?\d*)\s*khz\s*(?:to|-)\s*(\d+\.?\d*)\s*khz', line, re.IGNORECASE)
    if match:
        return (float(match.group(1)) * 1000, float(match.group(2)) * 1000)
    
    # Pattern 3: "16 kHz to 30 kHz" (standalone)
    match = re.search(r'(\d+\.?\d*)\s*khz\s*(?:to|-)\s*(\d+\.?\d*)\s*khz', line, re.IGNORECASE)
    if match:
        return (float(match.group(1)) * 1000, float(match.group(2)) * 1000)
    
    # Pattern 4: "16 to 30 kHz" (first number without kHz)
    match = re.search(r'(\d+\.?\d*)\s*(?:to|-)\s*(\d+\.?\d*)\s*khz', line, re.IGNORECASE)
    if match:
        return (float(match.group(1)) * 1000, float(match.group(2)) * 1000)
    
    # Pattern 5: "16 kHz 30 kHz" (without "to")
    match = re.search(r'(\d+\.?\d*)\s*khz\s+(\d+\.?\d*)\s*khz', line, re.IGNORECASE)
    if match:
        return (float(match.group(1)) * 1000, float(match.group(2)) * 1000)
    
    # Check next line if provided
    if next_line:
        return parse_frequency_range(next_line)
    
    return (None, None)


def parse_sensitivity(line: str, next_line: str = None, is_rx: bool = False) -> Optional[float]:
    """Parse sensitivity from line like 'Sensitivity-190 dB' or 'Sensitivity 136 dB'
    Returns: sensitivity value in dB, or None if not found
    """
    # Pattern 1: "Sensitivity-190 dB" or "Sensitivity136 dB" (no space)
    match = re.search(r'sensitivity(-?\d+\.?\d*)\s*dB', line, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if is_rx and val < 0:
            return val
        elif not is_rx and val > 100:
            return val
    
    # Pattern 2: "Sensitivity -190 dB" or "Sensitivity 136 dB" (with space)
    match = re.search(r'sensitivity\s+(-?\d+\.?\d*)\s*dB', line, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if is_rx and val < 0:
            return val
        elif not is_rx and val > 100:
            return val
    
    # Pattern 3: "-190 dB" or "136 dB" anywhere in line
    match = re.search(r'(-?\d+\.?\d*)\s*dB', line, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if is_rx and val < 0:
            return val
        elif not is_rx and val > 100:
            return val
    
    # Check next line if provided
    if next_line:
        return parse_sensitivity(next_line, None, is_rx)
    
    return None


def parse_voltage(line: str, next_line: str = None) -> Optional[float]:
    """Parse voltage from line like 'Voltage (Max)600 Vrms' or '600 Vrms'
    Returns: voltage in V, or None if not found
    """
    # Pattern 1: "Voltage (Max)600 Vrms" (no space after Max)
    match = re.search(r'\(max\)\s*(\d+\.?\d*)\s*V(?:rms)?', line, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if val > 50:
            return val
    
    # Pattern 2: "600 Vrms" anywhere in line
    match = re.search(r'(\d+\.?\d*)\s*V(?:rms)?', line, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if val > 50:
            return val
    
    # Check next line if provided
    if next_line:
        return parse_voltage(next_line)
    
    return None


def parse_capacitance(line: str) -> Optional[float]:
    """Parse capacitance from line like 'cable)12,000 pF' or '12,000 pF'
    Returns: capacitance in Farads, or None if not found
    """
    # Pattern 1: "cable)12,000 pF" (no space after closing parenthesis)
    match = re.search(r'\)\s*([\d,]+\.?\d*)\s*(pF|nF|µF|uF|pf|nf)', line, re.IGNORECASE)
    if match:
        val = float(match.group(1).replace(',', ''))
        unit = match.group(2).upper()
        if 'PF' in unit:
            return val * 1e-12  # pF to F
        elif 'NF' in unit:
            return val * 1e-9   # nF to F
        elif 'UF' in unit or 'µF' in unit.upper():
            return val * 1e-6   # µF to F
    
    # Pattern 2: "12,000 pF" (with spaces)
    match = re.search(r'([\d,]+\.?\d*)\s*(pF|nF|µF|uF|pf|nf)', line, re.IGNORECASE)
    if match:
        val = float(match.group(1).replace(',', ''))
        unit = match.group(2).upper()
        if 'PF' in unit:
            return val * 1e-12
        elif 'NF' in unit:
            return val * 1e-9
        elif 'UF' in unit or 'µF' in unit.upper():
            return val * 1e-6
    
    return None


def parse_resonant_frequency(line: str) -> Optional[float]:
    """Parse resonant frequency from line like '24 kHz' or 'Resonant Frequency (Nominal)24 kHz'
    Returns: frequency in Hz, or None if not found
    """
    match = re.search(r'(\d+\.?\d*)\s*khz', line, re.IGNORECASE)
    if match:
        return float(match.group(1)) * 1000  # Convert to Hz
    return None


def parse_beam_angle(line: str) -> Optional[float]:
    """Parse beam angle from line like '8 degrees' or '8°'
    Returns: angle in degrees, or None if not found
    """
    match = re.search(r'(\d+\.?\d*)\s*(?:degrees?|deg|°)', line, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def parse_beam_pattern(line: str, next_line: str = None) -> Optional[Dict[str, Any]]:
    """Parse beam pattern from line like 'Beam Pattern (Horizontal)Omni ± 2 dB' or 'Toroidal (See Graph)'
    Returns: dict with 'type' (horizontal/vertical), 'pattern' (omni/toroidal/etc), and 'angle' if numeric, or None
    """
    result = {}
    line_lower = line.lower()
    
    # Определяем тип (horizontal/vertical)
    if 'horizontal' in line_lower:
        result['type'] = 'horizontal'
    elif 'vertical' in line_lower:
        result['type'] = 'vertical'
    else:
        return None
    
    # Ищем паттерн в текущей строке
    # Pattern 1: "Omni ± 2 dB" или "Omni"
    if 'omni' in line_lower:
        result['pattern'] = 'omni'
        # Ищем угол отклонения, если есть
        angle_match = re.search(r'[±]\s*(\d+\.?\d*)\s*dB', line, re.IGNORECASE)
        if angle_match:
            result['deviation'] = float(angle_match.group(1))
        return result
    
    # Pattern 2: "Toroidal" или "Toroidal (See Graph)"
    if 'toroidal' in line_lower:
        result['pattern'] = 'toroidal'
        return result
    
    # Pattern 3: Числовой угол (например, "8 degrees")
    angle = parse_beam_angle(line)
    if angle:
        result['pattern'] = 'directional'
        result['angle'] = angle
        return result
    
    # Проверяем следующую строку, если текущая только метка
    if next_line:
        next_lower = next_line.lower()
        if 'omni' in next_lower:
            result['pattern'] = 'omni'
            angle_match = re.search(r'[±]\s*(\d+\.?\d*)\s*dB', next_line, re.IGNORECASE)
            if angle_match:
                result['deviation'] = float(angle_match.group(1))
            return result
        if 'toroidal' in next_lower:
            result['pattern'] = 'toroidal'
            return result
        angle = parse_beam_angle(next_line)
        if angle:
            result['pattern'] = 'directional'
            result['angle'] = angle
            return result
    
    return None


def parse_impedance(line: str) -> Optional[float]:
    """Parse impedance from line like '50 Ohm' or '50Ω'
    Returns: impedance in Ohms, or None if not found
    """
    match = re.search(r'(\d+\.?\d*)\s*(?:Ohm|Ω|ohms?)', line, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def test_parse_line(line: str, parser_func, parser_name: str, **kwargs):
    """Тестирует функцию парсинга на одной строке."""
    print(f"\n{'='*60}")
    print(f"Тест: {parser_name}")
    print(f"Строка: '{line}'")
    result = parser_func(line, **kwargs)
    print(f"Результат: {result}")
    return result


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
    
    # Ищем параметры
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
    
    # Поиск параметров
    for i, line in enumerate(lines):
        line_lower = line.lower()
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        next_next_line = lines[i + 2] if i + 2 < len(lines) else None
        
        # Resonant Frequency - проверяем текущую, следующую и через одну строку
        if 'resonant frequency' in line_lower and 'nominal' in line_lower:
            f_0 = parse_resonant_frequency(line)
            if not f_0 and next_line:
                f_0 = parse_resonant_frequency(next_line)
            if not f_0 and next_next_line:
                f_0 = parse_resonant_frequency(next_next_line)
            if f_0:
                results['f_0'] = f_0
                print(f"\n✓ Найдена резонансная частота (строка {i}): {f_0} Hz")
        
        # Operating Band - проверяем текущую, следующую и через одну строку
        elif 'useful operating band' in line_lower or 'operating band' in line_lower:
            f_min, f_max = parse_frequency_range(line, next_line)
            if (not f_min or not f_max) and next_line:
                # Если не нашли в текущей строке, пробуем только следующую
                f_min, f_max = parse_frequency_range(next_line)
            if (not f_min or not f_max) and next_next_line:
                # Пробуем через одну строку
                f_min, f_max = parse_frequency_range(next_next_line)
            if f_min and f_max:
                results['f_min'] = f_min
                results['f_max'] = f_max
                print(f"\n✓ Найден диапазон частот (строка {i}): {f_min} - {f_max} Hz")
        
        # RX Sensitivity - проверяем текущую, следующую и через одну строку
        elif 'receive sensitivity' in line_lower or ('receive' in line_lower and 'sensitivity' in line_lower):
            rx = parse_sensitivity(line, next_line, is_rx=True)
            if not rx and next_line:
                rx = parse_sensitivity(next_line, None, is_rx=True)
            if not rx and next_next_line:
                rx = parse_sensitivity(next_next_line, None, is_rx=True)
            if rx:
                results['rx_sensitivity'] = rx
                print(f"\n✓ Найдена RX чувствительность (строка {i}): {rx} dB")
        
        # TX Sensitivity - проверяем текущую, следующую и через одну строку
        elif 'transmit sensitivity' in line_lower or ('transmit' in line_lower and 'sensitivity' in line_lower):
            tx = parse_sensitivity(line, next_line, is_rx=False)
            if not tx and next_line:
                tx = parse_sensitivity(next_line, None, is_rx=False)
            if not tx and next_next_line:
                tx = parse_sensitivity(next_next_line, None, is_rx=False)
            if tx:
                results['tx_sensitivity'] = tx
                print(f"\n✓ Найдена TX чувствительность (строка {i}): {tx} dB")
        
        # Capacitance - проверяем текущую, следующую и через одну строку
        elif 'capacitance' in line_lower:
            cap = parse_capacitance(line)
            if not cap and next_line:
                cap = parse_capacitance(next_line)
            if not cap and next_next_line:
                cap = parse_capacitance(next_next_line)
            if cap:
                results['capacitance'] = cap
                print(f"\n✓ Найдена емкость (строка {i}): {cap * 1e9:.2f} nF")
        
        # Voltage - проверяем текущую, следующую и через одну строку
        elif 'transmit voltage' in line_lower and 'max' in line_lower:
            v = parse_voltage(line, next_line)
            if not v and next_line:
                v = parse_voltage(next_line)
            if not v and next_next_line:
                v = parse_voltage(next_next_line)
            if v:
                results['v_max'] = v
                print(f"\n✓ Найдено напряжение (строка {i}): {v} V")
        
        # Beam Pattern - проверяем текущую, следующую и через одну строку
        elif 'beam' in line_lower and 'pattern' in line_lower:
            pattern = parse_beam_pattern(line, next_line)
            if not pattern and next_next_line:
                pattern = parse_beam_pattern(line, next_next_line)
            if pattern:
                pattern_type = pattern.get('type')
                pattern_name = pattern.get('pattern', 'unknown')
                if pattern_type == 'horizontal':
                    results['beam_pattern_horizontal'] = pattern
                    print(f"\n✓ Найден горизонтальный beam pattern (строка {i}): {pattern_name}")
                elif pattern_type == 'vertical':
                    results['beam_pattern_vertical'] = pattern
                    print(f"\n✓ Найден вертикальный beam pattern (строка {i}): {pattern_name}")
        
        # Beam Angle (старый формат) - проверяем текущую, следующую и через одну строку
        elif 'beam' in line_lower and 'angle' in line_lower and 'pattern' not in line_lower:
            angle = parse_beam_angle(line)
            if not angle and next_line:
                angle = parse_beam_angle(next_line)
            if not angle and next_next_line:
                angle = parse_beam_angle(next_next_line)
            if angle:
                results['beam_angle'] = angle
                print(f"\n✓ Найден угол луча (строка {i}): {angle}°")
        
        # Impedance - проверяем текущую и следующую строку
        elif 'impedance' in line_lower and 'nominal' in line_lower:
            z = parse_impedance(line)
            if not z and next_line:
                z = parse_impedance(next_line)
            if z:
                results['impedance'] = z
                print(f"\n✓ Найдено сопротивление (строка {i}): {z} Ohm")
    
    # Выводим итоговые результаты
    print(f"\n{'='*60}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"{'='*60}")
    for key, value in results.items():
        if value is not None:
            if key == 'capacitance':
                print(f"  {key:20s}: {value * 1e9:.2f} nF")
            elif key in ['f_0', 'f_min', 'f_max']:
                print(f"  {key:20s}: {value:,.0f} Hz")
            elif key in ['tx_sensitivity', 'rx_sensitivity']:
                print(f"  {key:20s}: {value:.1f} dB")
            elif key in ['beam_pattern_horizontal', 'beam_pattern_vertical']:
                pattern_str = value.get('pattern', 'unknown')
                if 'deviation' in value:
                    pattern_str += f" ±{value['deviation']} dB"
                elif 'angle' in value:
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
        print("Использование: python test_pdf_parsing.py <pdf_file>")
        print("\nПримеры:")
        print("  python test_pdf_parsing.py scripts/T257.pdf")
        print("  python test_pdf_parsing.py data/transducers/142_200.pdf")
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
