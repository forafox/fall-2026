import numpy as np
from sklearn.decomposition import PCA

import os

import pca
import plots
from data import load_data, standardize, train_test_split


def main():
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"объектов: {len(y)}, признаков: {X.shape[1]}, признаки: {', '.join(feature_names)}")
    print(f"обучение: {X_train.shape}, тест: {X_test.shape}")
    print(f"целевая переменная: от {y.min():.0f} до {y.max():.0f}, среднее {y.mean():.1f}")

    Z = standardize(X_train)
    correlation = np.corrcoef(Z, rowvar=False)
    print("\nсамые скоррелированные пары признаков:")
    pairs = [(abs(correlation[i, j]), feature_names[i], feature_names[j], correlation[i, j])
             for i in range(len(feature_names)) for j in range(i + 1, len(feature_names))]
    for _, first, second, value in sorted(pairs, reverse=True)[:4]:
        print(f"  {first:4s} и {second:4s} r = {value:+.3f}")
    print(f"\nчисло обусловленности матрицы признаков: {np.linalg.cond(Z):.1f}")

    decomposition(Z)
    dimension(Z)
    compare_with_reference(Z, standardize(X_test, X_train))


IMAGES = os.path.join(os.path.dirname(__file__), "..", "images")


def decomposition(Z):
    """Checks that the SVD decomposition really has the properties PCA is built on."""
    mean, components, singular = pca.fit(Z)
    variance = pca.explained_variance(singular, len(Z))
    print("\nсингулярное разложение:")
    print("  сингулярные числа   :", singular.round(2))
    print("  дисперсии компонент :", variance.round(3))
    print("  доли дисперсии      :", pca.explained_variance_ratio(singular).round(3))

    P = pca.transform(Z, mean, components)
    covariance = np.cov(P, rowvar=False)
    print("\nпроверки:")
    print(f"  ортонормированность осей, max|V Vt - I| = "
          f"{np.abs(components @ components.T - np.eye(len(components))).max():.2e}")
    print(f"  некоррелированность проекций, max вне диагонали = "
          f"{np.abs(covariance - np.diag(np.diag(covariance))).max():.2e}")
    print(f"  дисперсия проекций равна s^2/(n-1): {np.allclose(P.var(axis=0, ddof=1), variance)}")
    print(f"  восстановление по всем осям, max|Z - Z'| = "
          f"{np.abs(pca.inverse_transform(P, mean, components) - Z).max():.2e}")

    print("\nпотеря информации при отбрасывании осей:")
    for k in [2, 5, 8, 9]:
        approximation = pca.inverse_transform(pca.transform(Z, mean, components, k), mean, components)
        error = ((approximation - Z) ** 2).sum() / (len(Z) - 1)
        print(f"  k = {k}: ошибка восстановления {error:.4f}, сумма отброшенных дисперсий {variance[k:].sum():.4f}")


def dimension(Z):
    """Effective dimension: how many axes are worth keeping, by four different criteria."""
    _, _, singular = pca.fit(Z)
    variance = pca.explained_variance(singular, len(Z))
    ratio = pca.explained_variance_ratio(singular)
    stick = pca.broken_stick(len(ratio))

    print("\nэффективная размерность:")
    print(f"{'номер':>6} {'дисперсия':>11} {'доля':>8} {'накопленно':>12} {'трость':>8}")
    for j, (v, r, c, b) in enumerate(zip(variance, ratio, np.cumsum(ratio), stick), start=1):
        print(f"{j:>6} {v:>11.3f} {r:>8.3f} {c:>12.3f} {b:>8.3f}")

    print("\nкритерии:")
    for level in [0.85, 0.95, 0.99]:
        print(f"  доля дисперсии {level:.0%}: {pca.effective_dimension(ratio, level)} компонент")
    print(f"  критерий Кайзера (дисперсия > 1): {int((variance > 1).sum())} компонент")
    print(f"  сломанная трость (доля > случайной): {int((ratio > stick).sum())} компонент")
    print(f"  обусловленность: s_max / s_min = {singular[0] / singular[-1]:.1f}, "
          f"при 8 компонентах {singular[0] / singular[7]:.1f}")

    plots.plot_scree(variance, ratio, stick, "Эффективная размерность выборки diabetes",
                     os.path.join(IMAGES, "scree.png"))


def compare_with_reference(Z, Z_test):
    """Same decomposition by sklearn: everything must match up to the sign of the axes."""
    mean, components, singular = pca.fit(Z)
    reference = PCA(n_components=None, svd_solver="full").fit(Z)

    signs = np.sign(np.sum(components * reference.components_, axis=1))
    print("\nсравнение с sklearn.decomposition.PCA:")
    print(f"  знаки осей совпали у {int((signs > 0).sum())} из {len(signs)} компонент")
    print(f"  max|среднее - mean_|            = {np.abs(mean - reference.mean_).max():.2e}")
    print(f"  max|сингулярные - singular_values_| = {np.abs(singular - reference.singular_values_).max():.2e}")
    print(f"  max|дисперсии - explained_variance_| = "
          f"{np.abs(pca.explained_variance(singular, len(Z)) - reference.explained_variance_).max():.2e}")
    print(f"  max|доли - explained_variance_ratio_| = "
          f"{np.abs(pca.explained_variance_ratio(singular) - reference.explained_variance_ratio_).max():.2e}")
    print(f"  max|оси - components_| с учётом знака = "
          f"{np.abs(signs[:, None] * components - reference.components_).max():.2e}")

    own = pca.transform(Z_test, mean, components)
    print(f"  max|проекции теста - transform()| с учётом знака = "
          f"{np.abs(own * signs - reference.transform(Z_test)).max():.2e}")
    for k in [2, 5, 8]:
        restored = pca.inverse_transform(pca.transform(Z_test, mean, components, k), mean, components)
        model = PCA(n_components=k, svd_solver="full").fit(Z)
        difference = np.abs(restored - model.inverse_transform(model.transform(Z_test))).max()
        print(f"  k = {k}: max|восстановление - inverse_transform()| = {difference:.2e}")


if __name__ == "__main__":
    main()
