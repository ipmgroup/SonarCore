"""
Тесты для WaterModel.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.water_model import WaterModel


class TestWaterModel(unittest.TestCase):
    """Тесты для модели воды."""
    
    def setUp(self):
        """Инициализация тестов."""
        self.water_model = WaterModel()
    
    def test_sound_speed(self):
        """Тест расчёта скорости звука."""
        T, S, z = 15.0, 35.0, 10.0
        P = WaterModel.calculate_pressure(z)
        c = self.water_model.calculate_sound_speed(T, S, P)
        
        # Скорость звука в морской воде при 15°C должна быть около 1500 м/с
        self.assertGreater(c, 1400)
        self.assertLess(c, 1600)
    
    def test_attenuation(self):
        """Тест расчёта затухания."""
        f, T, S, z = 200000, 15.0, 35.0, 10.0
        P = WaterModel.calculate_pressure(z)
        alpha = self.water_model.calculate_attenuation(f, T, S, P)
        
        # Затухание должно быть положительным
        self.assertGreater(alpha, 0)
        # Затухание на 200 кГц должно быть разумным (не слишком большим)
        self.assertLess(alpha, 1.0)  # дБ/м
    
    def test_transmission_loss(self):
        """Тест расчёта потерь распространения."""
        D, f, T, S, z = 100.0, 200000, 15.0, 35.0, 10.0
        TL = self.water_model.calculate_transmission_loss(D, f, T, S, z)
        
        # Потери должны быть положительными
        self.assertGreater(TL, 0)
        # Для 100 м на 200 кГц потери должны быть разумными
        self.assertLess(TL, 100)  # дБ


if __name__ == '__main__':
    unittest.main()

