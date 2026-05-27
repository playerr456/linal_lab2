import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
from data import prepare_data
from perceptron import Perceptron


def precision(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def roc_curve(y_true, y_proba):
    thresholds = np.linspace(0, 1, 300)
    tprs, fprs = [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return np.array(fprs), np.array(tprs)


def auc(fprs, tprs):
    idx    = np.argsort(fprs)
    fprs_s = fprs[idx]
    tprs_s = tprs[idx]
    return float(np.trapz(tprs_s, fprs_s))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data(verbose=False)

    model = Perceptron(n_features=2)
    model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

    y_pred  = model.predict(X_test)
    y_proba = model.forward(X_test)

    p   = precision(y_test, y_pred)
    r   = recall   (y_test, y_pred)
    f1  = f1_score (y_test, y_pred)
    fprs, tprs = roc_curve(y_test, y_proba)
    roc_auc    = auc(fprs, tprs)

    print("=" * 55)
    print("БОНУС 3: Метрики качества и анализ ошибок")
    print("=" * 55)
    print(f"\nAccuracy  : {model.accuracy(X_test, y_test):.4f}")
    print(f"Precision : {p:.4f}")
    print(f"Recall    : {r:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    errors = y_pred != y_test

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
               c="red",  alpha=0.5, s=30, label="класс 0")
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
               c="blue", alpha=0.5, s=30, label="класс 1")
    ax.scatter(X_test[errors, 0], X_test[errors, 1],
               facecolors="none", edgecolors="black", s=120, linewidths=1.5,
               label=f"ошибки ({errors.sum()})")
    x1_vals = np.linspace(X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5, 300)
    if abs(model.w[1]) > 1e-10:
        x2_vals = -(model.w[0] * x1_vals + model.b) / model.w[1]
        ax.plot(x1_vals, x2_vals, "k-", linewidth=2)
    ax.set_title(f"Анализ ошибок | ошибок: {errors.sum()} / {len(y_test)}")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(fprs, tprs, color="darkorange", linewidth=2,
            label=f"ROC-кривая (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="случайный классификатор")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривая")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("Бонус 3: Метрики качества")
    plt.tight_layout()
    plt.savefig("metrics.png", dpi=100)
    plt.close()
    print("График сохранён: metrics.png")
