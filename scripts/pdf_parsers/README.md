# PDF Parser System

Расширяемая система парсеров для извлечения параметров трансдусеров из PDF файлов.

## Структура

```
pdf_parsers/
├── __init__.py          # Базовые классы и реестр парсеров
├── neptun_parser.py      # Парсер для NEPTUN Communications
├── example_parser.py     # Пример парсера для справки
└── README.md            # Эта документация
```

## Как добавить новый парсер

### Шаг 1: Создайте новый файл парсера

Создайте новый файл в директории `pdf_parsers/`, например `my_parser.py`:

```python
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from pdf_parsers import BasePDFParser, register_parser

logger = logging.getLogger(__name__)


class MyParser(BasePDFParser):
    """Парсер для вашего формата PDF"""
    
    def __init__(self):
        super().__init__()
        self.name = "MyParser"
        self.description = "Описание вашего парсера"
    
    def parse(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Парсинг PDF файла.
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Словарь с извлеченными параметрами или None при ошибке.
            
            Ожидаемые ключи (все опциональны):
            - f_0: Резонансная частота (Hz)
            - f_min: Минимальная частота (Hz)
            - f_max: Максимальная частота (Hz)
            - tx_sensitivity: Чувствительность передачи (dB)
            - rx_sensitivity: Чувствительность приема (dB)
            - capacitance: Емкость (Farads)
            - v_max: Максимальное напряжение (Vrms)
            - beam_angle: Угол луча (degrees)
            - beam_pattern_horizontal: Dict с информацией о горизонтальном паттерне
            - beam_pattern_vertical: Dict с информацией о вертикальном паттерне
            - impedance: Импеданс (Ohms)
        """
        # Извлеките текст из PDF
        text = self.extract_text(pdf_path)
        if not text:
            return None
        
        # Ваша логика парсинга здесь
        results = {
            'f_0': None,
            'f_min': None,
            # ... другие параметры
        }
        
        # Парсинг текста и заполнение results
        # ...
        
        return results


# Регистрация парсера
register_parser(MyParser, "MyParser")
```

### Шаг 2: Парсер автоматически зарегистрируется

Система автоматически обнаружит и зарегистрирует ваш парсер при импорте модуля `pdf_parsers`.

### Шаг 3: Использование в GUI

Парсер автоматически появится в выпадающем списке "PDF Parser" в TAB 1.

## Базовый класс BasePDFParser

### Методы

#### `parse(pdf_path: Path) -> Optional[Dict[str, Any]]`
Основной метод парсинга. Должен быть реализован в каждом парсере.

#### `extract_text(pdf_path: Path) -> str`
Извлекает текст из PDF используя `pdftotext`. Можно переопределить для кастомной экстракции.

### Атрибуты

- `name`: Имя парсера (используется в GUI)
- `description`: Описание парсера (показывается в tooltip)

## Формат возвращаемых данных

Парсер должен возвращать словарь со следующими ключами (все опциональны):

```python
{
    'f_0': float,                    # Hz
    'f_min': float,                  # Hz
    'f_max': float,                  # Hz
    'tx_sensitivity': float,        # dB
    'rx_sensitivity': float,         # dB
    'capacitance': float,            # Farads
    'v_max': float,                  # Vrms
    'beam_angle': float,              # degrees
    'beam_pattern_horizontal': dict, # {'pattern': str, 'deviation': float, ...}
    'beam_pattern_vertical': dict,    # {'pattern': str, 'angle': float, ...}
    'impedance': float                # Ohms
}
```

## Примеры

### Пример 1: Простой парсер с регулярными выражениями

```python
import re

class SimpleParser(BasePDFParser):
    def parse(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        text = self.extract_text(pdf_path)
        results = {}
        
        # Поиск резонансной частоты
        match = re.search(r'Frequency:\s*(\d+)\s*kHz', text, re.IGNORECASE)
        if match:
            results['f_0'] = float(match.group(1)) * 1000
        
        return results if results else None
```

### Пример 2: Парсер с использованием внешней библиотеки

```python
import some_pdf_library

class AdvancedParser(BasePDFParser):
    def extract_text(self, pdf_path: Path) -> str:
        # Переопределяем для использования другой библиотеки
        doc = some_pdf_library.open(pdf_path)
        return doc.get_text()
    
    def parse(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        # Используем кастомную экстракцию
        text = self.extract_text(pdf_path)
        # ... парсинг
```

## Отладка

Используйте логирование для отладки:

```python
logger.info("Начало парсинга")
logger.debug(f"Извлеченный текст: {text[:100]}...")
logger.warning("Параметр не найден")
logger.error(f"Ошибка: {e}", exc_info=True)
```

## Тестирование

Создайте тестовый скрипт:

```python
from pathlib import Path
from pdf_parsers import get_parser

parser = get_parser("MyParser")
if parser:
    results = parser.parse(Path("test.pdf"))
    print(results)
```

## Совместимость с SonarCore

Парсеры возвращают данные в промежуточном формате, который затем автоматически конвертируется в формат SonarCore (`/data/transducers/*.json`).

Конвертация выполняется автоматически в `bvd.py` в методе `parsePDFTransducerParams()`.
