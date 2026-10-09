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
    equivalence(F, labels, y_train)
    reweighting(F, labels)


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


def equivalence(F, y, y01, tau=1.0):
    """Newton-Raphson, IRLS and IRLS in GLM notation are the same step written three ways."""
    newton_path = logistic.newton_raphson(F, y, tau=tau)
    irls_path = logistic.irls(F, y, tau=tau)
    glm_path = logistic.irls_glm(F, y01, tau=tau)
    shared = min(len(newton_path), len(irls_path), len(glm_path))

    print("\nтри записи одного и того же шага:")
    print(f"{'шаг':>4} {'||w|| Ньютон':>14} {'|Ньютон - IRLS|':>17} {'|Ньютон - GLM|':>16}")
    for step in range(shared):
        print(f"{step:>4} {np.linalg.norm(newton_path[step]):>14.6f} "
              f"{np.abs(newton_path[step] - irls_path[step]).max():>17.2e} "
              f"{np.abs(newton_path[step] - glm_path[step]).max():>16.2e}")
    print(f"  итераций: Ньютон {len(newton_path) - 1}, IRLS {len(irls_path) - 1}, "
          f"GLM {len(glm_path) - 1} (критерии остановки разные: по Q и по sigma)")

    from_least_squares = logistic.irls(F, y, tau=tau, start=logistic.least_squares(F, y))
    print(f"  IRLS из МНК-приближения, как в лекции: {len(from_least_squares) - 1} итераций, "
          f"ответ отличается на {np.abs(from_least_squares[-1] - newton_path[-1]).max():.2e}")


def reweighting(F, y, tau=1.0):
    """What IRLS actually does: objects near the border get the biggest weight."""
    weights = logistic.newton_raphson(F, y, tau=tau)[-1]
    sigma = logistic.sigmoid(logistic.margins(F, y, weights))
    importance = (1 - sigma) * sigma
    order = np.argsort(-importance)

    print("\nвеса объектов на последней итерации IRLS:")
    print(f"{'объект':>8} {'маржа M':>10} {'sigma':>9} {'вес (1-s)s':>12} {'1/sigma':>10}")
    for i in np.concatenate([order[:3], order[-3:]]):
        print(f"{i:>8} {logistic.margins(F, y, weights)[i]:>10.3f} {sigma[i]:>9.6f} "
              f"{importance[i]:>12.3e} {1 / sigma[i]:>10.3f}")
    print(f"  суммарный вес {importance.sum():.2f} на {len(y)} объектов: "
          f"{int((importance > 0.01 * importance.max()).sum())} объектов несут почти всю информацию")


if __name__ == "__main__":
    main()
