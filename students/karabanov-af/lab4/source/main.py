import numpy as np

import pca
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


if __name__ == "__main__":
    main()
