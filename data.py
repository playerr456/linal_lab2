import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def prepare_data(verbose: bool = True) -> tuple:
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1,
    )

    if verbose:
        print("=" * 55)
        print("ШАГ 1: Генерация данных")
        print("=" * 55)
        print("Размер X:", X.shape)
        print("Размер y:", y.shape)
        print("Классы  :", dict(zip(*np.unique(y, return_counts=True))))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    if verbose:
        print("\n" + "=" * 55)
        print("ШАГ 2: Разделение train / test")
        print("=" * 55)
        print("Train:", X_train.shape, "| Test:", X_test.shape)

    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    if verbose:
        print("\n" + "=" * 55)
        print("ШАГ 3: Стандартизация (Z-score)")
        print("=" * 55)
        print("mean:", X_train.mean(axis=0).round(4))
        print("std :", X_train.std(axis=0).round(4))

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
        print("\n" + "=" * 55)
        print("ШАГ 4: Визуализация данных")
        print("=" * 55)
        print("График сохранён: step2_data.png")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_data(verbose=True)
