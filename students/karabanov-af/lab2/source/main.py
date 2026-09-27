import numpy as np

from data import load_data, standardize, train_test_split


def main():
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"объектов: {len(y)}, признаков: {X.shape[1]}, классов: {len(np.unique(y))}")
    print(f"обучение: {X_train.shape}, по классам {np.bincount(y_train)}")
    print(f"тест: {X_test.shape}, по классам {np.bincount(y_test)}")

    print("\nразброс признаков до стандартизации:")
    for name, std in zip(feature_names, X_train.std(axis=0)):
        print(f"  {name:30s} std = {std:8.3f}")
    print("после стандартизации std всех признаков =", np.unique(standardize(X_train).std(axis=0).round(3)))


if __name__ == "__main__":
    main()
