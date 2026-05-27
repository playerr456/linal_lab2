import numpy as np


class Perceptron:

    def __init__(self, n_features: int, init: str = "small", seed: int = 42):
        rng = np.random.default_rng(seed)
        if init == "small":
            self.w = rng.normal(0, 0.01, n_features)
        elif init == "zero":
            self.w = np.zeros(n_features)
        elif init == "large":
            self.w = rng.normal(0, 10, n_features)
        else:
            raise ValueError(f"init='{init}' неизвестен. Выберите: small, zero, large")
        self.b = 0.0
        self.train_losses: list = []
        self.val_losses:   list = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(X @ self.w + self.b)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(y_pred)
                              + (1 - y_true) * np.log(1 - y_pred)))

    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray,
            epochs:     int   = 100,
            lr:         float = 0.1,
            batch_size: int   = 32) -> "Perceptron":
        rng = np.random.default_rng(0)
        n = len(X_train)
        self.train_losses = []
        self.val_losses   = []

        for epoch in range(epochs):
            idx = rng.permutation(n)
            Xs, ys = X_train[idx], y_train[idx]

            for start in range(0, n, batch_size):
                Xb = Xs[start:start + batch_size]
                yb = ys[start:start + batch_size]
                m  = len(yb)

                yhat  = self.forward(Xb)
                error = yhat - yb
                dw = (Xb.T @ error) / m
                db = error.mean()

                self.w -= lr * dw
                self.b -= lr * db

            self.train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            self.val_losses.append  (self.compute_loss(y_val,   self.forward(X_val)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.forward(X) >= 0.5).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))
