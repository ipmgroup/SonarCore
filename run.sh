#!/bin/bash
# Скрипт для запуска приложения SonarCore

# Активируем виртуальное окружение
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    # Поддержка старого имени для обратной совместимости
    source venv/bin/activate
else
    echo "Виртуальное окружение не найдено. Создайте его командой:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Запускаем приложение
python3 main.py

