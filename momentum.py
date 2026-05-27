import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
from data import prepare_data
from plots import plot_losses


class PerceptronMomentum:

    def __init__(self, n_features, seed=42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, n_features)
        self.b = 0.0
        self.train_losses = []
        self.val_losses   = []

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def compute_loss(self, y_true, y_pred):
        eps    = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def fit(self, X_train, y_train, X_val, y_val,
            epochs=100, lr=0.1, batch_size=32, beta=0.9):
        rng = np.random.default_rng(0)
        n   = len(X_train)
        self.train_losses = []
        self.val_losses   = []

        vw = np.zeros_like(self.w)
        vb = 0.0

        for _ in range(epochs):
            idx = rng.permutation(n)
            Xs, ys = X_train[idx], y_train[idx]

            for start in range(0, n, batch_size):
                Xb = Xs[start:start + batch_size]
                yb = ys[start:start + batch_size]
                m  = len(yb)

                yhat  = self.forward(Xb)
                error = yhat - yb
                dw    = (Xb.T @ error) / m
                db    = error.mean()

                vw = beta * vw + dw
                vb = beta * vb + db

                self.w -= lr * vw
                self.b -= lr * vb

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
    print("БОНУС 4: Градиентный спуск с моментом")
    print("=" * 55)

    betas = [0.0, 0.5, 0.9, 0.99]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    print(f"\n{'beta':>6} | {'Acc train':>10} | {'Acc test':>10} | {'Финал loss':>12}")
    print("-" * 46)

    for ax, beta in zip(axes, betas):
        m = PerceptronMomentum(n_features=2)
        m.fit(X_train, y_train, X_test, y_test,
              epochs=100, lr=0.1, batch_size=32, beta=beta)
        atr  = m.accuracy(X_train, y_train)
        ate  = m.accuracy(X_test,  y_test)
        label = "SGD (β=0)" if beta == 0.0 else f"Momentum β={beta}"
        plot_losses(m.train_losses, m.val_losses,
                    title=f"{label}\ntrain={atr:.3f}, test={ate:.3f}", ax=ax)
        print(f"{beta:>6} | {atr:>10.4f} | {ate:>10.4f} | {m.train_losses[-1]:>12.4f}")

    plt.suptitle("Бонус 4: SGD vs Momentum (lr=0.1, batch=32)")
    plt.tight_layout()
    plt.savefig("momentum.png", dpi=100)
    plt.close()
    print("\nГрафик сохранён: momentum.png")
