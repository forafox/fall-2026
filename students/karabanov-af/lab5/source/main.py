import numpy as np

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

    Z = add_bias(standardize(X_train))
    print(f"\nматрица обучения со свободным членом: {Z.shape}, "
          f"число обусловленности {np.linalg.cond(Z):.1f}")


if __name__ == "__main__":
    main()
