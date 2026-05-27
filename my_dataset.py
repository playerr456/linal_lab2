import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from perceptron import Perceptron
from plots import plot_losses


def make_blobs(n_samples: int = 500,
               center0: tuple = (-2, -2),
               center1: tuple = (2,  2),
               cov: np.ndarray = None,
               noise: float = 0.0,
               seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    if cov is None:
        cov = np.eye(2)
    n = n_samples // 2
    X0 = rng.multivariate_normal(mean=center0, cov=cov, size=n)
    X1 = rng.multivariate_normal(mean=center1, cov=cov, size=n) # объяснить, почему именно это
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(int)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    if noise > 0:
        flip = rng.random(len(y)) < noise
        y[flip] = 1 - y[flip]
    return X, y


def make_xor(n_samples: int = 500,
             spread: float = 0.5,
             noise: float = 0.0,
             seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    n = n_samples // 4
    corners = [
        ((-2, -2), 0),
        (( 2, -2), 1),
        ((-2,  2), 1),
        (( 2,  2), 0),
    ]
    X_parts, y_parts = [], []
    for (cx, cy), label in corners:
        pts = rng.normal(loc=[cx, cy], scale=spread, size=(n, 2))
        X_parts.append(pts)
        y_parts.append(np.full(n, label, dtype=int))
    X = np.vstack(X_parts)
    y = np.hstack(y_parts)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    if noise > 0:
        flip = rng.random(len(y)) < noise
        y[flip] = 1 - y[flip]
    return X, y


def make_circles(n_samples: int = 500,
                 r_inner: float = 1.0,
                 r_outer: float = 2.5,
                 noise: float = 0.0,
                 seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    n = n_samples // 2

    def sample_annulus(n_pts, r_min, r_max):
        r = np.sqrt(rng.uniform(r_min**2, r_max**2, n_pts)) # объяснить, почему именно это
        theta = rng.uniform(0, 2 * np.pi, n_pts)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    X1 = sample_annulus(n, 0,       r_inner)
    X0 = sample_annulus(n, r_inner, r_outer)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(int)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    if noise > 0:
        flip = rng.random(len(y)) < noise
        y[flip] = 1 - y[flip]
    return X, y


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1
    return (X_train - mean) / std, (X_test - mean) / std


def train_and_report(X, y, label: str, epochs=200, lr=0.1, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train, X_test = standardize(X_train, X_test)
    model = Perceptron(n_features=2, init="small")
    model.fit(X_train, y_train, X_test, y_test,
              epochs=epochs, lr=lr, batch_size=batch_size)
    acc = model.accuracy(X_test, y_test)
    print(f"  {label:<30} test accuracy = {acc:.4f}")
    return model, X_train, X_test, y_train, y_test, acc


def plot_boundary_contour(model, X, y, title="", ax=None):
    if ax is None:
        _, ax = plt.subplots()
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                            np.linspace(x2_min, x2_max, 300))
    grid  = np.column_stack([xx1.ravel(), xx2.ravel()])
    proba = model.forward(grid).reshape(xx1.shape)
    ax.contourf(xx1, xx2, proba, levels=50, cmap="RdBu_r", alpha=0.55)
    ax.contour (xx1, xx2, proba, levels=[0.5], colors="k", linewidths=2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="red",  label="класс 0", alpha=0.7, edgecolors="w", s=30)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="класс 1", alpha=0.7, edgecolors="w", s=30)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)


if __name__ == "__main__":
    print("=" * 55)
    print("БОНУС 1: Собственный генератор данных")
    print("=" * 55)

    print("\n--- A. Гауссовые облака ---")
    datasets_blobs = [
        ("Лёгкие",     make_blobs(center0=(-3,-3), center1=(3,3),  cov=np.eye(2)*0.5, noise=0.0)),
        ("Средние",    make_blobs(center0=(-2,-2), center1=(2,2),  cov=np.eye(2),     noise=0.0)),
        ("Шум 15%",    make_blobs(center0=(-2,-2), center1=(2,2),  cov=np.eye(2),     noise=0.15)),
        ("Тяжёлые",    make_blobs(center0=(-1,-1), center1=(1,1),  cov=np.eye(2)*2,   noise=0.0)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, (label, (X, y)) in enumerate(datasets_blobs):
        model, Xtr, Xte, ytr, yte, acc = train_and_report(X, y, label)
        axes[0, col].scatter(X[y==0,0], X[y==0,1], c="red",  alpha=0.5, s=15, label="кл.0")
        axes[0, col].scatter(X[y==1,0], X[y==1,1], c="blue", alpha=0.5, s=15, label="кл.1")
        axes[0, col].set_title(label, fontsize=9)
        axes[0, col].legend(fontsize=7); axes[0, col].grid(True, alpha=0.3)
        plot_boundary_contour(model, Xte, yte, title=f"acc={acc:.3f}", ax=axes[1, col])
    axes[0, 0].set_ylabel("Исходные данные", fontsize=10)
    axes[1, 0].set_ylabel("Граница перцептрона", fontsize=10)
    plt.suptitle("Гауссовые облака: перцептрон СПРАВЛЯЕТСЯ", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("bonus1_blobs.png", dpi=100); plt.close()
    print("  График сохранён: bonus1_blobs.png")

    print("\n--- B. XOR ---")
    datasets_xor = [
        ("XOR spread=0.3",  make_xor(spread=0.3, noise=0.0)),
        ("XOR spread=0.6",  make_xor(spread=0.6, noise=0.0)),
        ("XOR шум 10%",     make_xor(spread=0.3, noise=0.10)),
        ("XOR шум 20%",     make_xor(spread=0.3, noise=0.20)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, (label, (X, y)) in enumerate(datasets_xor):
        model, Xtr, Xte, ytr, yte, acc = train_and_report(X, y, label)
        axes[0, col].scatter(X[y==0,0], X[y==0,1], c="red",  alpha=0.5, s=15, label="кл.0")
        axes[0, col].scatter(X[y==1,0], X[y==1,1], c="blue", alpha=0.5, s=15, label="кл.1")
        axes[0, col].set_title(label, fontsize=9)
        axes[0, col].legend(fontsize=7); axes[0, col].grid(True, alpha=0.3)
        plot_boundary_contour(model, Xte, yte, title=f"acc={acc:.3f}", ax=axes[1, col])
    axes[0, 0].set_ylabel("Исходные данные", fontsize=10)
    axes[1, 0].set_ylabel("Граница перцептрона", fontsize=10)
    plt.suptitle("XOR: перцептрон НЕ СПРАВЛЯЕТСЯ (~50%)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("bonus1_xor.png", dpi=100); plt.close()
    print("  График сохранён: bonus1_xor.png")

    print("\n--- C. Окружность ---")
    datasets_circles = [
        ("Чёткое разделение",    make_circles(r_inner=1.0, r_outer=2.5, noise=0.0)),
        ("Близкие радиусы",      make_circles(r_inner=1.5, r_outer=2.5, noise=0.0)),
        ("Окружность шум 10%",   make_circles(r_inner=1.0, r_outer=2.5, noise=0.10)),
        ("Окружность шум 25%",   make_circles(r_inner=1.0, r_outer=2.5, noise=0.25)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, (label, (X, y)) in enumerate(datasets_circles):
        model, Xtr, Xte, ytr, yte, acc = train_and_report(X, y, label)
        axes[0, col].scatter(X[y==0,0], X[y==0,1], c="red",  alpha=0.5, s=15, label="кл.0")
        axes[0, col].scatter(X[y==1,0], X[y==1,1], c="blue", alpha=0.5, s=15, label="кл.1")
        axes[0, col].set_title(label, fontsize=9)
        axes[0, col].legend(fontsize=7); axes[0, col].grid(True, alpha=0.3)
        plot_boundary_contour(model, Xte, yte, title=f"acc={acc:.3f}", ax=axes[1, col])
    axes[0, 0].set_ylabel("Исходные данные", fontsize=10)
    axes[1, 0].set_ylabel("Граница перцептрона", fontsize=10)
    plt.suptitle("Окружность: перцептрон НЕ СПРАВЛЯЕТСЯ", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("bonus1_circles.png", dpi=100); plt.close()
    print("  График сохранён: bonus1_circles.png")

    print("\n--- D. Итоговое сравнение ---")
    summary = [
        ("Гауссовые облака\n(линейно разд.)",  *make_blobs(center0=(-2,-2), center1=(2,2), cov=np.eye(2), noise=0.0)),
        ("XOR\n(нелинейно разд.)",              *make_xor(spread=0.4, noise=0.0)),
        ("Окружность\n(нелинейно разд.)",       *make_circles(r_inner=1.0, r_outer=2.5, noise=0.0)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, X, y) in zip(axes, summary):
        model, Xtr, Xte, ytr, yte, acc = train_and_report(X, y, label.replace("\n", " "))
        plot_boundary_contour(model, Xte, yte, title=f"{label}\nacc = {acc:.3f}", ax=ax)
    plt.suptitle("Границы применимости перцептрона", fontsize=13)
    plt.tight_layout()
    plt.savefig("bonus1_summary.png", dpi=100); plt.close()
    print("  График сохранён: bonus1_summary.png")
