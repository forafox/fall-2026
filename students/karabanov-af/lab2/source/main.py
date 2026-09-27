import numpy as np

import knn
from data import load_data, standardize, train_test_split
from metrics import accuracy


def main():
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"объектов: {len(y)}, признаков: {X.shape[1]}, классов: {len(np.unique(y))}")
    print(f"обучение: {X_train.shape}, по классам {np.bincount(y_train)}")
    print(f"тест: {X_test.shape}, по классам {np.bincount(y_test)}")

    Z_train = standardize(X_train)
    Z_test = standardize(X_test, X_train)

    print("\naccuracy на тесте:")
    print(f"{'k':>3} {'сырые, голосование':>20} {'станд., голосование':>21} {'станд., Парзен':>16}")
    for k in [1, 3, 5, 10, 20]:
        raw = accuracy(y_test, knn.predict(X_train, y_train, X_test, k))
        vote = accuracy(y_test, knn.predict(Z_train, y_train, Z_test, k))
        parzen = accuracy(y_test, knn.predict_parzen(Z_train, y_train, Z_test, k))
        print(f"{k:>3} {raw:>20.3f} {vote:>21.3f} {parzen:>16.3f}")


if __name__ == "__main__":
    main()
