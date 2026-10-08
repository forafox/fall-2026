import numpy as np

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


if __name__ == "__main__":
    main()
