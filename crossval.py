import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
from data import prepare_data
from perceptron import Perceptron


def kfold_split(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    splits = []
    for i in range(k):
        val_idx   = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def cross_val_score(X, y, lr, batch_size, epochs=100, k=5):
    splits = kfold_split(len(X), k=k)
    accs = []
    for train_idx, val_idx in splits:
        Xtr, ytr = X[train_idx], y[train_idx]
        Xvl, yvl = X[val_idx],   y[val_idx]

        mean = Xtr.mean(axis=0)
        std  = Xtr.std(axis=0)
        std[std == 0] = 1
        Xtr = (Xtr - mean) / std
        Xvl = (Xvl - mean) / std

        model = Perceptron(n_features=X.shape[1], init="small")
        model.fit(Xtr, ytr, Xvl, yvl, epochs=epochs, lr=lr, batch_size=batch_size)
        accs.append(model.accuracy(Xvl, yvl))
    return float(np.mean(accs)), float(np.std(accs))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data(verbose=False)

    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])

    print("=" * 55)
    print("БОНУС 5: Кросс-валидация и подбор гиперпараметров")
    print("=" * 55)

    lrs         = [0.001, 0.01, 0.1, 0.5]
    batch_sizes = [16, 32, 64, 128]

    results = []
    print(f"\n{'lr':>8} | {'batch':>6} | {'mean acc':>10} | {'std':>8}")
    print("-" * 40)

    for lr in lrs:
        for bs in batch_sizes:
            mean_acc, std_acc = cross_val_score(X_all, y_all, lr=lr, batch_size=bs)
            results.append((lr, bs, mean_acc, std_acc))
            print(f"{lr:>8} | {bs:>6} | {mean_acc:>10.4f} | {std_acc:>8.4f}")

    best = max(results, key=lambda r: r[2])
    best_lr, best_bs, best_mean, best_std = best
    print(f"\nЛучшие параметры: lr={best_lr}, batch_size={best_bs}")
    print(f"CV accuracy: {best_mean:.4f} +/- {best_std:.4f}")

    mean_final = X_train.mean(axis=0)
    std_final  = X_train.std(axis=0)
    std_final[std_final == 0] = 1
    Xtr_s = (X_train - mean_final) / std_final
    Xte_s = (X_test  - mean_final) / std_final

    final_model = Perceptron(n_features=2, init="small")
    final_model.fit(Xtr_s, y_train, Xte_s, y_test,
                    epochs=100, lr=best_lr, batch_size=best_bs)
    final_acc = final_model.accuracy(Xte_s, y_test)
    print(f"Финальная точность на тесте: {final_acc:.4f}")

    means  = np.array([r[2] for r in results]).reshape(len(lrs), len(batch_sizes))
    stds   = np.array([r[3] for r in results]).reshape(len(lrs), len(batch_sizes))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    im = ax.imshow(means, aspect="auto", cmap="YlGn",
                   vmin=means.min() - 0.01, vmax=means.max() + 0.01)
    ax.set_xticks(range(len(batch_sizes))); ax.set_xticklabels(batch_sizes)
    ax.set_yticks(range(len(lrs)));        ax.set_yticklabels(lrs)
    ax.set_xlabel("batch_size"); ax.set_ylabel("lr")
    ax.set_title("CV mean accuracy (5-fold)")
    for i in range(len(lrs)):
        for j in range(len(batch_sizes)):
            ax.text(j, i, f"{means[i,j]:.3f}\n+/-{stds[i,j]:.3f}",
                    ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    lr_labels = [str(lr) for lr in lrs]
    for j, bs in enumerate(batch_sizes):
        col_means = means[:, j]
        col_stds  = stds[:, j]
        ax.errorbar(lr_labels, col_means, yerr=col_stds,
                    marker="o", capsize=4, label=f"batch={bs}")
    ax.set_xlabel("learning rate"); ax.set_ylabel("CV accuracy")
    ax.set_title("CV accuracy vs lr (by batch_size)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Бонус 5: Grid Search + 5-fold CV\nЛучшее: lr={best_lr}, batch={best_bs}, acc={best_mean:.4f}")
    plt.tight_layout()
    plt.savefig("crossval.png", dpi=100)
    plt.close()
    print("График сохранён: crossval.png")
