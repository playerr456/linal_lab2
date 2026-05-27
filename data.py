# -*- coding: utf-8 -*-
"""
data.py — подготовка данных (шаги 1-2 задания):

  1. Генерация датасета для бинарной классификации
  2. Разделение на train (70%) / test (30%) со стратификацией
  3. Стандартизация признаков методом Z-score по обучающей выборке
  4. Визуализация — сохраняет step2_data.png

Использование:
  from data import prepare_data
  X_train, X_test, y_train, y_test = prepare_data()

Или запуск напрямую:
  python data.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def prepare_data(verbose: bool = True) -> tuple:
    """
    Генерирует, делит и стандартизирует данные.

    Возвращает: (X_train, X_test, y_train, y_test)
    """

    # ------------------------------------------------------------------
    # 1) Генерация данных
    # ------------------------------------------------------------------
    # make_classification создаёт синтетический датасет для классификации.
    # Каждая точка — это 2 числовых признака (x1, x2), метка — 0 или 1.
    X, y = make_classification(
        n_samples=500,            # 500 точек
        n_features=2,             # 2 признака
        n_redundant=0,            # нет «лишних» признаков-копий
        n_informative=2,          # оба признака несут полезную информацию
        random_state=42,          # фиксируем seed — результат воспроизводим
        n_clusters_per_class=1,   # одно облако точек на класс
    )

    if verbose:
        print("=" * 55)
        print("ШАГ 1: Генерация данных")
        print("=" * 55)
        print("Размер X:", X.shape)
        print("Размер y:", y.shape)
        print("Классы  :", dict(zip(*np.unique(y, return_counts=True))))

    # ------------------------------------------------------------------
    # 2) Разделение train / test со стратификацией
    # ------------------------------------------------------------------
    # stratify=y гарантирует, что доля классов в train и test совпадает
    # с исходным датасетом (примерно 50/50 в данном случае).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,      # 30% — тест, 70% — обучение
        random_state=42,
        stratify=y,
    )

    if verbose:
        print("\n" + "=" * 55)
        print("ШАГ 2: Разделение train / test")
        print("=" * 55)
        print("Train:", X_train.shape, "| Test:", X_test.shape)

    # ------------------------------------------------------------------
    # 3) Стандартизация (Z-score) по обучающей выборке
    # ------------------------------------------------------------------
    # Z-score: x_new = (x - mean) / std
    # Каждый признак приводится к среднему 0 и std 1.
    #
    # ВАЖНО: mean и std считаем ТОЛЬКО на train-данных.
    # Затем применяем эти же числа к test — иначе «утечка» информации.
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)

    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    if verbose:
        print("\n" + "=" * 55)
        print("ШАГ 3: Стандартизация (Z-score)")
        print("=" * 55)
        print("mean после стандартизации:", X_train.mean(axis=0).round(4), "(должно быть ~0)")
        print("std  после стандартизации:", X_train.std(axis=0).round(4),  "(должно быть ~1)")

    # ------------------------------------------------------------------
    # 4) Визуализация
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 55)
        print("ШАГ 4: Визуализация данных")
        print("=" * 55)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, Xp, yp, title in [
        (axes[0], X_train, y_train, "Обучающая выборка (после стандартизации)"),
        (axes[1], X_test,  y_test,  "Тестовая выборка"),
    ]:
        ax.scatter(Xp[yp == 0, 0], Xp[yp == 0, 1], c="red",  label="класс 0", alpha=0.7)
        ax.scatter(Xp[yp == 1, 0], Xp[yp == 1, 1], c="blue", label="класс 1", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("step2_data.png", dpi=100)
    plt.close()

    if verbose:
        print("График сохранён: step2_data.png")

    return X_train, X_test, y_train, y_test


# Запуск напрямую: python data.py
if __name__ == "__main__":
    prepare_data(verbose=True)
