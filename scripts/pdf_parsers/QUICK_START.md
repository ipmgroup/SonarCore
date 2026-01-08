# Быстрый старт: Добавление нового парсера

## Минимальный пример

1. **Создайте файл** `pdf_parsers/my_new_parser.py`:

```python
from pathlib import Path
from typing import Dict, Any, Optional
from pdf_parsers import BasePDFParser

class MyNewParser(BasePDFParser):
    def __init__(self):
        super().__init__()
        self.name = "MyNewParser"
        self.description = "Парсер для моего формата PDF"
    
    def parse(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        text = self.extract_text(pdf_path)
        if not text:
            return None
        
        results = {}
        
        # Ваша логика парсинга
        # Например:
        if '24 kHz' in text:
            results['f_0'] = 24000  # Hz
        
        return results if results else None
```

2. **Готово!** Парсер автоматически появится в GUI в выпадающем списке "PDF Parser".

## Что нужно реализовать

Обязательно:
- Метод `parse()` - основная логика парсинга

Опционально:
- Переопределить `extract_text()` для кастомной экстракции текста
- Изменить `name` и `description` в `__init__()`

## Формат возвращаемых данных

Метод `parse()` должен возвращать словарь с любыми из этих ключей:

```python
{
    'f_0': float,                    # Резонансная частота (Hz)
    'f_min': float,                   # Минимальная частота (Hz)
    'f_max': float,                   # Максимальная частота (Hz)
    'tx_sensitivity': float,         # Чувствительность TX (dB)
    'rx_sensitivity': float,         # Чувствительность RX (dB)
    'capacitance': float,             # Емкость (Farads)
    'v_max': float,                   # Максимальное напряжение (Vrms)
    'beam_angle': float,              # Угол луча (degrees)
    'beam_pattern_horizontal': dict,  # Горизонтальный паттерн
    'beam_pattern_vertical': dict,     # Вертикальный паттерн
    'impedance': float                # Импеданс (Ohms)
}
```

Все ключи опциональны - верните только те, которые удалось извлечь.

## Примеры использования

### Простой парсер с регулярными выражениями

```python
import re

class RegexParser(BasePDFParser):
    def parse(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        text = self.extract_text(pdf_path)
        results = {}
        
        # Поиск частоты
        match = re.search(r'Frequency:\s*(\d+)\s*kHz', text)
        if match:
            results['f_0'] = float(match.group(1)) * 1000
        
        return results if results else None
```

### Парсер с построчным анализом

```python
class LineByLineParser(BasePDFParser):
    def parse(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        text = self.extract_text(pdf_path)
        lines = text.split('\n')
        results = {}
        
        for line in lines:
            if 'Resonant Frequency' in line:
                # Извлеките значение
                # ...
                pass
        
        return results if results else None
```

## Тестирование

Создайте тестовый скрипт `test_parser.py`:

```python
from pathlib import Path
from pdf_parsers import get_parser

parser = get_parser("MyNewParser")
if parser:
    results = parser.parse(Path("test.pdf"))
    print("Результаты:", results)
else:
    print("Парсер не найден")
```

## Отладка

Используйте логирование:

```python
import logging
logger = logging.getLogger(__name__)

class MyParser(BasePDFParser):
    def parse(self, pdf_path: Path):
        logger.info(f"Парсинг {pdf_path}")
        text = self.extract_text(pdf_path)
        logger.debug(f"Извлечено {len(text)} символов")
        # ...
```

Логи будут в файле `logs/bvd_metrics_*.log`.
