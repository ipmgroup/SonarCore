#!/usr/bin/env python3
"""
Улучшенный скрипт для извлечения данных из PDF файлов 142SERIES.pdf и 390SERIES.pdf.
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple


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


def extract_transducers_from_text(text: str, series: str) -> List[Dict[str, Any]]:
    """Извлекает данные о трансдусерах из текста."""
    transducers = []
    
    lines = [line.strip() for line in text.split('\n')]
    
    # Ищем секцию "TECHNICAL SPECIFICATION"
    spec_start = -1
    for i, line in enumerate(lines):
        if 'TECHNICAL SPECIFICATION' in line.upper():
            spec_start = i
            break
    
    if spec_start == -1:
        return transducers
    
    # Ищем строку "Frequency Options" или просто "Frequency"
    freq_options_idx = -1
    for i in range(spec_start, min(spec_start + 20, len(lines))):
        line = lines[i].strip()
        if 'Frequency Options' in line or ('Frequency' in line and 'Options' not in line):
            freq_options_idx = i
            break
    
    if freq_options_idx == -1:
        return transducers
    
    # Собираем частоты из следующих строк до "kHz" или до следующего параметра
    frequencies = []
    kHz_line_idx = -1
    next_param_idx = len(lines)  # Индекс следующего параметра после частот
    
    # Ищем строку с "kHz"
    for i in range(freq_options_idx + 1, min(freq_options_idx + 20, len(lines))):
        line = lines[i].strip()
        if 'kHz' in line.lower():
            kHz_line_idx = i
            break
    
    # Ищем следующий параметр (Beam Angle, Transmit Sensitivity и т.д.)
    for i in range(freq_options_idx + 1, min(spec_start + 100, len(lines))):
        line = lines[i].strip()
        if any(keyword in line for keyword in ['Beam Angle', 'Transmit Sensitivity', 'Receive Sensitivity', 'Bandwidth']):
            next_param_idx = i
            break
    
    # Определяем диапазон строк для поиска частот
    search_end = min(kHz_line_idx if kHz_line_idx > 0 else next_param_idx, freq_options_idx + 20)
    
    # Извлекаем все числа из строк между "Frequency" и "kHz" или следующим параметром
    for j in range(freq_options_idx + 1, search_end):
        prev_line = lines[j].strip()
        # Пропускаем пустые строки и строки с единицами измерения
        if not prev_line or prev_line.lower() in ['khz', 'hz', 'degrees', 'db', 'v', 'ohms']:
            continue
        # Ищем числа и WB (частоты могут быть от 10 до 200 кГц)
        for match in re.finditer(r'\b(\d{2,3})(WB)?\b', prev_line, re.IGNORECASE):
            freq_val = int(match.group(1))
            is_wb = match.group(2) is not None and match.group(2).upper() == 'WB'
            if 10 <= freq_val <= 200:  # Фильтруем реальные частоты (10-200 кГц)
                if (freq_val, is_wb) not in frequencies:  # Избегаем дубликатов
                    frequencies.append((freq_val, is_wb))
    
    if not frequencies:
        print(f"  Не найдено частот для серии {series}")
        return transducers
    
    print(f"  Найдено частот: {[f[0] for f in frequencies]}")
    
    # Теперь собираем параметры построчно
    beam_angles = []
    tx_sensitivities = []
    rx_sensitivities = []
    bandwidths = []
    
    # Ищем значения параметров
    for i in range(spec_start, min(spec_start + 100, len(lines))):
        line = lines[i]
        
        # Beam Angle
        if 'Beam Angle' in line or ('beam' in line.lower() and 'angle' in line.lower()):
            values = []
            # Собираем значения из следующих строк до "Degrees"
            for j in range(i + 1, min(i + 15, len(lines))):
                next_line = lines[j]
                if 'Degrees' in next_line or 'degrees' in next_line.lower():
                    # Извлекаем числа из строк между заголовком и "Degrees"
                    for k in range(i + 1, j + 1):  # Включаем строку с "Degrees"
                        line_to_parse = lines[k]
                        # Обрабатываем как простые числа, так и форматы типа "19x34"
                        # Берем первое число из пары (основной угол) для эллиптических
                        # Ищем паттерны типа "19x34" или просто "19"
                        # Паттерн: число, возможно за которым следует "x" и еще число
                        pattern = r'\b(\d+\.?\d*)(?:x\d+\.?\d*)?\b'
                        matches = re.findall(pattern, line_to_parse)
                        if matches:
                            # Берем только первое число из каждой пары (если есть "x")
                            for num_str in matches:
                                try:
                                    val = float(num_str)
                                    if 5 <= val <= 50:  # Разумный диапазон для beam angle
                                        values.append(val)
                                except ValueError:
                                    pass
                    break
            if values:
                beam_angles = values[:len(frequencies)]
                print(f"  Beam angles: {beam_angles}")
        
        # Transmit Sensitivity
        elif 'Transmit Sensitivity' in line or ('transmit' in line.lower() and 'sensitivity' in line.lower()):
            values = []
            for j in range(i + 1, min(i + 15, len(lines))):
                next_line = lines[j]
                if 'dB' in next_line or 'db' in next_line.lower():
                    for k in range(i + 1, j):
                        nums = re.findall(r'\b(\d+\.?\d*)\b', lines[k])
                        values.extend([float(n) for n in nums])
                    break
            if values:
                tx_sensitivities = values[:len(frequencies)]
                print(f"  TX sensitivities: {tx_sensitivities}")
        
        # Receive Sensitivity
        elif 'Receive Sensitivity' in line or ('receive' in line.lower() and 'sensitivity' in line.lower()):
            values = []
            for j in range(i + 1, min(i + 15, len(lines))):
                next_line = lines[j]
                if 'dB' in next_line or 'db' in next_line.lower():
                    for k in range(i + 1, j):
                        nums = re.findall(r'(-?\d+\.?\d*)\b', lines[k])
                        values.extend([float(n) for n in nums])
                    break
            if values:
                rx_sensitivities = values[:len(frequencies)]
                print(f"  RX sensitivities: {rx_sensitivities}")
        
        # Bandwidth
        elif 'Bandwidth' in line.lower():
            values = []
            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j]
                nums = re.findall(r'\b(\d+\.?\d*)\b', next_line)
                if nums:
                    values.extend([float(n) * 1000 for n in nums])  # кГц -> Гц
                    break
            if values:
                bandwidths = values[:len(frequencies)]
                print(f"  Bandwidths: {bandwidths}")
    
    # Создаём модели для каждой частоты
    for idx, (freq_khz, is_wb) in enumerate(frequencies):
        freq_hz = freq_khz * 1000
        
        model_name = f"{series}_{freq_khz}"
        if is_wb:
            model_name += "WB"
        
        # Вычисляем полосу пропускания
        if idx < len(bandwidths) and bandwidths[idx] > 0:
            bw = bandwidths[idx]
        else:
            # Используем типичную полосу 10% от центральной частоты
            bw = freq_hz * 0.1
        
        transducer = {
            'model': model_name,
            'f_0': freq_hz,
            'f_min': freq_hz * 0.9,
            'f_max': freq_hz * 1.1,
            'B_tr': bw,
            'S_TX': tx_sensitivities[idx] if idx < len(tx_sensitivities) else 170.0,
            'S_RX': rx_sensitivities[idx] if idx < len(rx_sensitivities) else -193.0,
            'Theta_BW': beam_angles[idx] if idx < len(beam_angles) else 8.0,
            'Q': freq_hz / bw if bw > 0 else 5.0,  # Q = f0 / BW
            'T_rd': 10.0,  # мкс
            'Z': 50.0,  # Ом
            'source': f'{series}SERIES.pdf',
            'version': '1.0'
        }
        
        transducers.append(transducer)
    
    return transducers


def process_pdf(pdf_path: Path, series: str) -> List[Dict[str, Any]]:
    """Обрабатывает PDF файл и извлекает данные о трансдусерах."""
    print(f"Обработка {pdf_path.name}...")
    
    transducers = []
    
    try:
        text = extract_text_from_pdf(pdf_path)
        parsed = extract_transducers_from_text(text, series)
        if parsed:
            transducers.extend(parsed)
            print(f"  Извлечено моделей: {len(parsed)}")
    except Exception as e:
        print(f"  Ошибка при извлечении данных: {e}")
        import traceback
        traceback.print_exc()
    
    return transducers


def main():
    """Главная функция."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'transducers'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = [
        (project_root / '65SERIES.pdf', '65'),
        (project_root / '142SERIES.pdf', '142'),
        (project_root / '172SERIES.pdf', '172'),
        (project_root / '320SERIES.pdf', '320'),
        (project_root / '390SERIES.pdf', '390'),
        (project_root / '395SERIES.pdf', '395')
    ]
    
    all_transducers = []
    
    for pdf_path, series in pdf_files:
        if not pdf_path.exists():
            print(f"Файл {pdf_path} не найден!")
            continue
        
        transducers = process_pdf(pdf_path, series)
        all_transducers.extend(transducers)
        
        # Сохраняем каждую модель отдельно
        for transducer in transducers:
            filename = f"{transducer['model']}.json"
            filepath = data_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(transducer, f, indent=2, ensure_ascii=False)
            
            print(f"  Сохранено: {filename}")
    
    print(f"\nВсего извлечено моделей: {len(all_transducers)}")
    
    # Создаём индексный файл, включая все существующие модели
    index_file = data_dir / 'index.json'
    all_models = set([t['model'] for t in all_transducers])
    
    # Добавляем существующие модели из файлов (кроме example и index)
    for json_file in data_dir.glob('*.json'):
        if json_file.name not in ['index.json', 'example_transducer.json']:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'model' in data:
                        all_models.add(data['model'])
            except Exception:
                pass  # Пропускаем файлы с ошибками
    
    index_data = {
        'total_models': len(all_models),
        'models': sorted(list(all_models))
    }
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"Создан индексный файл: {index_file} (всего моделей: {len(all_models)})")


if __name__ == '__main__':
    main()

