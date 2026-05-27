import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
from perceptron import Perceptron
from plots import plot_losses


def exp_learning_rate(X_train, X_test, y_train, y_test) -> None:
    print("\n" + "=" * 55)
    print("ЭКСПЕРИМЕНТ 1: Влияние скорости обучения (lr)")
    print("=" * 55)
    print(f"{'lr':>8} | {'Acc train':>10} | {'Acc test':>10}")
    print("-" * 35)

    lrs = [0.001, 0.01, 0.5, 1.0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, lr in enumerate(lrs):
        model = Perceptron(n_features=2, init="small")
        model.fit(X_train, y_train, X_test, y_test,
                  epochs=100, lr=lr, batch_size=32)
        atr = model.accuracy(X_train, y_train)
        ate = model.accuracy(X_test,  y_test)
        plot_losses(model.train_losses, model.val_losses, ax=axes[i],
                    title=f"lr = {lr}  |  train={atr:.3f}, test={ate:.3f}")
        print(f"{lr:>8} | {atr:>10.4f} | {ate:>10.4f}")

    plt.suptitle("Эксперимент 1: Влияние скорости обучения (batch=32, init=small)")
    plt.tight_layout()
    plt.savefig("exp1_lr.png", dpi=100)
    plt.close()
    print("График сохранён: exp1_lr.png")


def exp_batch_size(X_train, X_test, y_train, y_test) -> None:
    print("\n" + "=" * 55)
    print("ЭКСПЕРИМЕНТ 2: Влияние размера мини-батча (batch_size)")
    print("=" * 55)
    print(f"{'batch':>8} | {'Acc train':>10} | {'Acc test':>10}")
    print("-" * 35)

    batch_sizes = [1, 16, 64, 256]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, bs in enumerate(batch_sizes):
        model = Perceptron(n_features=2, init="small")
        model.fit(X_train, y_train, X_test, y_test,
                  epochs=100, lr=0.1, batch_size=bs)
        atr = model.accuracy(X_train, y_train)
        ate = model.accuracy(X_test,  y_test)
        plot_losses(model.train_losses, model.val_losses, ax=axes[i],
                    title=f"batch={bs}  |  train={atr:.3f}, test={ate:.3f}")
        print(f"{bs:>8} | {atr:>10.4f} | {ate:>10.4f}")

    plt.suptitle("Эксперимент 2: Влияние размера батча (lr=0.1, init=small)")
    plt.tight_layout()
    plt.savefig("exp2_batch.png", dpi=100)
    plt.close()
    print("График сохранён: exp2_batch.png")


def exp_weight_init(X_train, X_test, y_train, y_test) -> None:
    print("\n" + "=" * 55)
    print("ЭКСПЕРИМЕНТ 3: Влияние инициализации весов (init)")
    print("=" * 55)
    print(f"{'init':>8} | {'Acc train':>10} | {'Acc test':>10}")
    print("-" * 35)

    inits = ["zero", "small", "large"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, init in zip(axes, inits):
        model = Perceptron(n_features=2, init=init)
        model.fit(X_train, y_train, X_test, y_test,
                  epochs=100, lr=0.1, batch_size=32)
        atr = model.accuracy(X_train, y_train)
        ate = model.accuracy(X_test,  y_test)
        plot_losses(model.train_losses, model.val_losses, ax=ax,
                    title=f"init = {init}  |  train={atr:.3f}, test={ate:.3f}")
        print(f"{init:>8} | {atr:>10.4f} | {ate:>10.4f}")

    plt.suptitle("Эксперимент 3: Влияние инициализации весов (lr=0.1, batch=32)")
    plt.tight_layout()
    plt.savefig("exp3_init.png", dpi=100)
    plt.close()
    print("График сохранён: exp3_init.png")


if __name__ == "__main__":
    from data import prepare_data
    X_train, X_test, y_train, y_test = prepare_data(verbose=False)
    exp_learning_rate(X_train, X_test, y_train, y_test)
    exp_batch_size   (X_train, X_test, y_train, y_test)
    exp_weight_init  (X_train, X_test, y_train, y_test)
