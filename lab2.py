import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data        import prepare_data
from train       import run_training
from experiments import exp_learning_rate, exp_batch_size, exp_weight_init

X_train, X_test, y_train, y_test = prepare_data()

model = run_training(X_train, X_test, y_train, y_test)

exp_learning_rate(X_train, X_test, y_train, y_test)
exp_batch_size   (X_train, X_test, y_train, y_test)
exp_weight_init  (X_train, X_test, y_train, y_test)

print("\n" + "=" * 55)
print("ВСЕ ГРАФИКИ СОХРАНЕНЫ")
print("=" * 55)
