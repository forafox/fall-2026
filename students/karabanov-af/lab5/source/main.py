import numpy as np

import logistic
from data import add_bias, load_data, standardize, train_test_split


def main():
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"объектов: {len(y)}, признаков: {X.shape[1]}")
    print(f"классы: злокачественная {int((y == 0).sum())}, доброкачественная {int((y == 1).sum())}")
    print(f"обучение: {X_train.shape}, тест: {X_test.shape}, "
          f"доля класса 1: {y_train.mean():.3f} и {y_test.mean():.3f}")

    print("\nразмах признаков до стандартизации:")
    for j in [0, 3, 23]:
        print(f"  {feature_names[j]:24s} от {X[:, j].min():8.3f} до {X[:, j].max():8.3f}")

    F = add_bias(standardize(X_train))
    F_test = add_bias(standardize(X_test, X_train))
    labels, labels_test = logistic.signed(y_train), logistic.signed(y_test)
    print(f"\nматрица обучения со свободным членом: {F.shape}, "
          f"число обусловленности {np.linalg.cond(F):.1f}")

    newton(F, labels, F_test, labels_test)
    separability(F, labels, F_test, labels_test)


def newton(F, y, F_test, y_test, tau=1.0):
    """Newton-Raphson on the log-loss: the Hessian is rebuilt and inverted at every step."""
    trajectory = logistic.newton_raphson(F, y, tau=tau)
    weights = trajectory[-1]

    print(f"\nметод Ньютона-Рафсона, tau = {tau}:")
    print(f"{'шаг':>4} {'Q(w)':>12} {'изменение':>12} {'||w||':>9} {'cond гессиана':>15}")
    previous = None
    for step, w in enumerate(trajectory):
        value = logistic.log_loss(F, y, w, tau)
        change = "-" if previous is None else f"{previous - value:12.3e}"
        print(f"{step:>4} {value:>12.6f} {change:>12} {np.linalg.norm(w):>9.3f} "
              f"{np.linalg.cond(logistic.hessian(F, y, w, tau)):>15.1f}")
        previous = value

    print(f"  сошёлся за {len(trajectory) - 1} итераций, ||градиента|| = "
          f"{np.linalg.norm(logistic.gradient(F, y, weights, tau)):.2e}")
    print(f"  точность: обучение {logistic.accuracy(y, logistic.predict(F, weights)):.4f}, "
          f"тест {logistic.accuracy(y_test, logistic.predict(F_test, weights)):.4f}")


def separability(F, y, F_test, y_test):
    """Without regularization the maximum of the likelihood on separable data is at infinity."""
    print("\nчто делает регуляризация (та же задача при разных tau):")
    print(f"{'tau':>8} {'Q(w)':>10} {'||w||':>10} {'cond гессиана':>15} {'точность теста':>16}")
    for tau in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
        weights = logistic.newton_raphson(F, y, tau=tau)[-1]
        print(f"{tau:>8.2f} {logistic.log_loss(F, y, weights):>10.4f} {np.linalg.norm(weights):>10.3f} "
              f"{np.linalg.cond(logistic.hessian(F, y, weights, tau)):>15.3e} "
              f"{logistic.accuracy(y_test, logistic.predict(F_test, weights)):>16.4f}")


if __name__ == "__main__":
    main()
