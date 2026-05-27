# -*- coding: utf-8 -*-
"""
lab2.py — точка входа лабораторной работы №1.

Запуск всего сразу:
  python lab2.py

Структура проекта
-----------------
  perceptron.py  — класс Perceptron (sigmoid, forward, loss, fit, predict)
  data.py        — подготовка данных (генерация, split, стандартизация)
  plots.py       — вспомогательные функции для графиков
  train.py       — базовое обучение (lr=0.1, epochs=100, batch=32)
  experiments.py — три эксперимента (lr / batch_size / init)
  lab2.py        — этот файл: запускает всё по порядку

Каждый модуль можно запустить отдельно:
  python data.py
  python train.py
  python experiments.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data        import prepare_data
from train       import run_training
from experiments import exp_learning_rate, exp_batch_size, exp_weight_init


# 1-4. Подготовка данных (генерация → split → стандартизация → визуализация)
X_train, X_test, y_train, y_test = prepare_data()

# 5-7. Базовое обучение, кривые потерь, разделяющая граница
model = run_training(X_train, X_test, y_train, y_test)

# 8. Эксперимент 1: влияние скорости обучения
exp_learning_rate(X_train, X_test, y_train, y_test)

# 9. Эксперимент 2: влияние размера батча
exp_batch_size(X_train, X_test, y_train, y_test)

# 10. Эксперимент 3: влияние инициализации весов
exp_weight_init(X_train, X_test, y_train, y_test)

print("\n" + "=" * 55)
print("ВСЕ ГРАФИКИ СОХРАНЕНЫ")
print("=" * 55)
