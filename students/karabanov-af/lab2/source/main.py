import os

import numpy as np

import knn
import plots
from data import load_data, standardize, train_test_split
from metrics import accuracy

IMAGES = os.path.join(os.path.dirname(__file__), "..", "images")
KS = np.arange(1, 41)


def main():
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"объектов: {len(y)}, признаков: {X.shape[1]}, классов: {len(np.unique(y))}")
    print(f"обучение: {X_train.shape}, по классам {np.bincount(y_train)}")
    print(f"тест: {X_test.shape}, по классам {np.bincount(y_test)}")

    Z_train = standardize(X_train)
    Z_test = standardize(X_test, X_train)

    risks = {
        "голосование": knn.loo_risk(Z_train, y_train, KS, knn.predict),
        "окно Парзена": knn.loo_risk(Z_train, y_train, KS, knn.predict_parzen),
    }
    plots.plot_loo(KS, risks, "Эмпирический риск LOO в зависимости от k", os.path.join(IMAGES, "loo.png"))

    print("\nподбор k по LOO на обучающей выборке:")
    print(f"{'k':>3} {'голосование':>13} {'окно Парзена':>14}")
    for i, k in enumerate(KS):
        if k <= 12 or k % 5 == 0:
            print(f"{k:>3} {risks['голосование'][i]:>13.3f} {risks['окно Парзена'][i]:>14.3f}")

    D_test = knn.distances(Z_train, Z_test)
    print("\nлучшее k по LOO и качество на тесте:")
    for name, method in [("голосование", knn.predict), ("окно Парзена", knn.predict_parzen)]:
        k = KS[risks[name].argmin()]
        print(f"  {name:13s} k = {k:2d}, LOO = {risks[name].min():.3f}, "
              f"accuracy на тесте = {accuracy(y_test, method(D_test, y_train, k)):.3f}")


if __name__ == "__main__":
    main()
