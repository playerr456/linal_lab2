import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
from data import prepare_data
from plots import plot_losses


class PerceptronHinge:

    def __init__(self, n_features, seed=42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, n_features)
        self.b = 0.0
        self.train_losses = []
        self.val_losses   = []

    def decision(self, X):
        return X @ self.w + self.b

    def compute_loss(self, y, z):
        return float(np.mean(np.maximum(0, 1 - y * z)))

    def fit(self, X_train, y_train, X_val, y_val,
            epochs=100, lr=0.1, batch_size=32):
        yt = np.where(y_train == 0, -1, 1).astype(float)
        yv = np.where(y_val   == 0, -1, 1).astype(float)
        rng = np.random.default_rng(0)
        n = len(X_train)
        self.train_losses = []
        self.val_losses   = []

        for _ in range(epochs):
            idx = rng.permutation(n)
            Xs, ys = X_train[idx], yt[idx]

            for start in range(0, n, batch_size):
                Xb = Xs[start:start + batch_size]
                yb = ys[start:start + batch_size]
                m  = len(yb)

                z    = self.decision(Xb)
                mask = (yb * z < 1).astype(float)

                dw = -(Xb.T @ (yb * mask)) / m
                db = -(yb * mask).mean()

                self.w -= lr * dw
                self.b -= lr * db

            self.train_losses.append(self.compute_loss(yt, self.decision(X_train)))
            self.val_losses.append  (self.compute_loss(yv, self.decision(X_val)))

        return self

    def predict(self, X):
        return (self.decision(X) >= 0).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))


class PerceptronL2:

    def __init__(self, n_features, lam=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.w   = rng.normal(0, 0.01, n_features)
        self.b   = 0.0
        self.lam = lam
        self.train_losses = []
        self.val_losses   = []

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def compute_loss(self, y_true, y_pred):
        eps    = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        bce    = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        l2     = 0.5 * self.lam * np.sum(self.w ** 2)
        return float(bce + l2)

    def fit(self, X_train, y_train, X_val, y_val,
            epochs=100, lr=0.1, batch_size=32):
        rng = np.random.default_rng(0)
        n = len(X_train)
        self.train_losses = []
        self.val_losses   = []

        for _ in range(epochs):
            idx = rng.permutation(n)
            Xs, ys = X_train[idx], y_train[idx]

            for start in range(0, n, batch_size):
                Xb = Xs[start:start + batch_size]
                yb = ys[start:start + batch_size]
                m  = len(yb)

                yhat  = self.forward(Xb)
                error = yhat - yb

                dw = (Xb.T @ error) / m + self.lam * self.w
                db = error.mean()

                self.w -= lr * dw
                self.b -= lr * db

            self.train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            self.val_losses.append  (self.compute_loss(y_val,   self.forward(X_val)))

        return self

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data(verbose=False)

    print("=" * 55)
    print("БОНУС 2: Функции потерь и регуляризация")
    print("=" * 55)

    from perceptron import Perceptron
    bce_model   = Perceptron(n_features=2)
    hinge_model = PerceptronHinge(n_features=2)

    bce_model.fit  (X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    hinge_model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

    print("\n--- Сравнение BCE vs Hinge ---")
    print(f"{'Модель':>12} | {'Acc train':>10} | {'Acc test':>10}")
    print("-" * 38)
    print(f"{'BCE':>12} | {bce_model.accuracy(X_train,y_train):>10.4f} | {bce_model.accuracy(X_test,y_test):>10.4f}")
    print(f"{'Hinge':>12} | {hinge_model.accuracy(X_train,y_train):>10.4f} | {hinge_model.accuracy(X_test,y_test):>10.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_losses(bce_model.train_losses,   bce_model.val_losses,
                title=f"BCE  | test={bce_model.accuracy(X_test,y_test):.3f}",   ax=axes[0])
    plot_losses(hinge_model.train_losses, hinge_model.val_losses,
                title=f"Hinge | test={hinge_model.accuracy(X_test,y_test):.3f}", ax=axes[1])
    plt.suptitle("Бонус 2: BCE vs Hinge loss")
    plt.tight_layout()
    plt.savefig("losses_bce_vs_hinge.png", dpi=100)
    plt.close()
    print("График сохранён: losses_bce_vs_hinge.png")

    print("\n--- L2-регуляризация (λ влияние) ---")
    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    print(f"{'lambda':>8} | {'Acc test':>10} | {'||w||':>8}")
    print("-" * 32)
    for ax, lam in zip(axes, lambdas):
        m = PerceptronL2(n_features=2, lam=lam)
        m.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
        ate   = m.accuracy(X_test, y_test)
        w_norm = float(np.linalg.norm(m.w))
        plot_losses(m.train_losses, m.val_losses,
                    title=f"λ={lam}  acc={ate:.3f}\n||w||={w_norm:.3f}", ax=ax)
        print(f"{lam:>8} | {ate:>10.4f} | {w_norm:>8.4f}")
    plt.suptitle("Бонус 2: Влияние L2-регуляризации")
    plt.tight_layout()
    plt.savefig("losses_l2.png", dpi=100)
    plt.close()
    print("График сохранён: losses_l2.png")
