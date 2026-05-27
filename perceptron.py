# -*- coding: utf-8 -*-
"""
perceptron.py — реализация однослойного перцептрона.

Класс Perceptron содержит:
  sigmoid()       — функция активации
  forward()       — прямой проход (предсказание вероятности)
  compute_loss()  — бинарная кросс-энтропия
  fit()           — обучение с мини-батчами (SGD)
  predict()       — классификация (0 или 1)
  accuracy()      — доля правильных ответов
"""
import numpy as np


class Perceptron:
    """
    Однослойный перцептрон с сигмоидной активацией.

    Обучение: SGD с мини-батчами, функция потерь — бинарная кросс-энтропия.
    Только NumPy — никаких ML-библиотек.
    """

    # ------------------------------------------------------------------
    # Инициализация
    # ------------------------------------------------------------------

    def __init__(self, n_features: int, init: str = "small", seed: int = 42):
        """
        n_features — число входных признаков
        init       — способ инициализации весов:
                       'small' — N(0, 0.01)  рекомендуется
                       'zero'  — все нули    показывает медленный старт
                       'large' — N(0, 10)    показывает проблему насыщения
        seed       — фиксирует случайность для воспроизводимости
        """
        rng = np.random.default_rng(seed)
        if init == "small":
            self.w = rng.normal(0, 0.01, n_features)   # маленькие веса
        elif init == "zero":
            self.w = np.zeros(n_features)               # нули
        elif init == "large":
            self.w = rng.normal(0, 10, n_features)      # большие веса
        else:
            raise ValueError(f"init='{init}' неизвестен. Выберите: small, zero, large")

        self.b = 0.0                   # bias (смещение)
        self.train_losses: list = []   # история потерь на train
        self.val_losses:   list = []   # история потерь на val

    # ------------------------------------------------------------------
    # Основные функции перцептрона
    # ------------------------------------------------------------------

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Сигмоидная функция активации: σ(z) = 1 / (1 + e^{-z})

        np.clip(z, -500, 500) защищает от переполнения при очень больших |z|.
        Возвращает значения в диапазоне (0, 1).
        """
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Прямой проход (forward pass):
            z  = X · w + b      — линейная комбинация признаков
            ŷ  = σ(z)           — вероятность принадлежности к классу 1

        X : матрица (m, n_features)
        Возвращает вектор ŷ длиной m, каждый элемент ∈ (0, 1).
        """
        return self.sigmoid(X @ self.w + self.b)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Бинарная кросс-энтропия (Binary Cross-Entropy):

            L = -mean[ y · log(ŷ) + (1 - y) · log(1 - ŷ) ]

        Штрафует за уверенные неправильные предсказания сильнее,
        чем за неуверенные.

        eps = 1e-15 защищает от log(0).
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(y_pred)
                              + (1 - y_true) * np.log(1 - y_pred)))

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray,
            epochs:     int   = 100,
            lr:         float = 0.1,
            batch_size: int   = 32) -> "Perceptron":
        """
        Обучение методом мини-батчевого SGD.

        На каждой эпохе:
          1. Перемешиваем обучающую выборку (чтобы батчи не повторялись).
          2. Нарезаем на батчи размером batch_size.
          3. Для каждого батча считаем аналитические градиенты BCE:
                dw = (1/m) · X^T · (ŷ - y)
                db = mean(ŷ - y)
          4. Обновляем веса: w -= lr · dw,  b -= lr · db
          5. Записываем потери на train и val.

        Параметры
        ---------
        epochs     — число полных проходов по обучающим данным
        lr         — скорость обучения (learning rate, η)
        batch_size — размер одного мини-батча
        """
        rng = np.random.default_rng(0)    # фиксированный seed для воспроизводимости
        n = len(X_train)
        self.train_losses = []
        self.val_losses   = []

        for epoch in range(epochs):
            # --- перемешиваем выборку ---
            idx = rng.permutation(n)
            Xs, ys = X_train[idx], y_train[idx]

            # --- проходим по мини-батчам ---
            for start in range(0, n, batch_size):
                Xb = Xs[start:start + batch_size]   # (m, n_features)
                yb = ys[start:start + batch_size]   # (m,)
                m  = len(yb)

                yhat  = self.forward(Xb)             # прямой проход
                error = yhat - yb                    # ошибка (ŷ - y)

                # аналитические градиенты функции потерь по w и b
                dw = (Xb.T @ error) / m
                db = error.mean()

                # шаг градиентного спуска
                self.w -= lr * dw
                self.b -= lr * db

            # --- сохраняем потери после каждой эпохи ---
            self.train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            self.val_losses.append  (self.compute_loss(y_val,   self.forward(X_val)))

        return self

    # ------------------------------------------------------------------
    # Предсказание и метрика
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классификация: возвращает 0 или 1.
        Порог 0.5 — если ŷ >= 0.5, предсказываем класс 1.
        """
        return (self.forward(X) >= 0.5).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Доля правильно классифицированных примеров."""
        return float(np.mean(self.predict(X) == y))
