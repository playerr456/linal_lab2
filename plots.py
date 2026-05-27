import numpy as np
import matplotlib.pyplot as plt


def plot_losses(train_losses: list, val_losses: list, title: str = "", ax=None) -> None:
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(train_losses, label="Train loss", linewidth=1.5)
    ax.plot(val_losses,   label="Val loss",   linewidth=1.5, linestyle="--")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("BCE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_boundary(model, X: np.ndarray, y: np.ndarray, title: str = "", ax=None) -> None:
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="red",  label="класс 0", alpha=0.6)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="класс 1", alpha=0.6)
    x1_vals = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300)
    if abs(model.w[1]) > 1e-10:
        x2_vals = -(model.w[0] * x1_vals + model.b) / model.w[1]
        ax.plot(x1_vals, x2_vals, "k-", linewidth=2, label="граница решения")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid(True, alpha=0.3)
