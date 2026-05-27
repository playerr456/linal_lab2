# -*- coding: utf-8 -*-
"""
train.py — базовое обучение перцептрона (шаги 3-4 задания):

  - Гиперпараметры: lr = 0.1, epochs = 100, batch_size = 32
  - Вычисляет accuracy на train и test
  - Сохраняет:
      step3_loss.png     — кривые потерь по эпохам
      step3_boundary.png — разделяющая прямая на тестовых данных

Использование:
  from train import run_training
  model = run_training(X_train, X_test, y_train, y_test)

Или запуск напрямую:
  python train.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
from perceptron import Perceptron
from plots import plot_losses, plot_boundary


def run_training(X_train, X_test, y_train, y_test) -> Perceptron:
    """
    Обучает перцептрон с базовыми гиперпараметрами задания.

    Возвращает обученную модель.
    """
    print("\n" + "=" * 55)
    print("ОБУЧЕНИЕ: lr=0.1 | epochs=100 | batch_size=32")
    print("=" * 55)

    # --- создаём и обучаем модель ---
    model = Perceptron(n_features=2, init="small")
    model.fit(X_train, y_train, X_test, y_test,
              epochs=100, lr=0.1, batch_size=32)

    # --- точность ---
    acc_train = model.accuracy(X_train, y_train)
    acc_test  = model.accuracy(X_test,  y_test)
    print(f"\nТочность на обучающей выборке: {acc_train:.4f}")
    print(f"Точность на тестовой  выборке: {acc_test:.4f}")

    # --- график кривых потерь ---
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_losses(
        model.train_losses, model.val_losses,
        title=f"Кривые потерь | train={acc_train:.3f}, test={acc_test:.3f}",
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig("step3_loss.png", dpi=100)
    plt.close()
    print("График сохранён: step3_loss.png")

    # --- график разделяющей границы ---
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_boundary(
        model, X_test, y_test,
        title=f"Разделяющая граница (тест, acc={acc_test:.3f})",
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig("step3_boundary.png", dpi=100)
    plt.close()
    print("График сохранён: step3_boundary.png")

    return model


# Запуск напрямую: python train.py
if __name__ == "__main__":
    from data import prepare_data
    X_train, X_test, y_train, y_test = prepare_data(verbose=False)
    run_training(X_train, X_test, y_train, y_test)
