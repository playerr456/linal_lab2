# -*- coding: utf-8 -*-
"""
plots.py — вспомогательные функции для визуализации.

  plot_losses()   — кривые потерь (train / val) по эпохам
  plot_boundary() — разделяющая прямая перцептрона на фоне точек данных
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_losses(train_losses: list, val_losses: list,
                title: str = "", ax=None) -> None:
    """
    Строит график изменения функции потерь (BCE) по эпохам.

    Параметры
    ---------
    train_losses — список потерь на обучающей выборке (по эпохам)
    val_losses   — список потерь на валидационной выборке (по эпохам)
    title        — заголовок графика
    ax           — matplotlib Axes; если None — создаётся новый
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(train_losses, label="Train loss", linewidth=1.5)
    ax.plot(val_losses,   label="Val loss",   linewidth=1.5, linestyle="--")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("BCE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_boundary(model, X: np.ndarray, y: np.ndarray,
                  title: str = "", ax=None) -> None:
    """
    Рисует точки данных и разделяющую прямую перцептрона.

    Уравнение границы: w0·x1 + w1·x2 + b = 0
    Выражаем x2: x2 = -(w0·x1 + b) / w1

    Параметры
    ---------
    model — обученный Perceptron (должен иметь поля w и b)
    X     — матрица признаков (m, 2)
    y     — метки классов (m,)
    title — заголовок
    ax    — matplotlib Axes; если None — создаётся новый
    """
    if ax is None:
        _, ax = plt.subplots()

    # Точки двух классов
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="red",  label="класс 0", alpha=0.6)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="класс 1", alpha=0.6)

    # Разделяющая прямая (только если w[1] != 0, иначе вертикальная линия)
    x1_vals = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300)
    if abs(model.w[1]) > 1e-10:
        x2_vals = -(model.w[0] * x1_vals + model.b) / model.w[1]
        ax.plot(x1_vals, x2_vals, "k-", linewidth=2, label="граница решения")

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid(True, alpha=0.3)
