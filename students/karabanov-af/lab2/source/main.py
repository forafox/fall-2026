import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

import knn
import plots
import prototypes
from data import load_data, standardize, train_test_split
from metrics import accuracy

IMAGES = os.path.join(os.path.dirname(__file__), "..", "images")
KS = np.arange(1, 41)


def main():
    X, y, feature_names, class_names = load_data()
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

    best_k = KS[risks["окно Парзена"].argmin()]
    compare_with_reference(Z_train, y_train, Z_test, y_test, KS[risks["голосование"].argmin()])
    prototype_experiment(Z_train, y_train, Z_test, y_test, class_names, best_k)


def prototype_experiment(Z_train, y_train, Z_test, y_test, class_names, k):
    idx, margins = prototypes.select(Z_train, y_train, k)
    noise = np.flatnonzero(margins <= 0)
    print(f"\nотбор эталонов при k = {k}:")
    print(f"  отсеяно как шум: {len(noise)}, эталонов: {len(idx)} из {len(y_train)} "
          f"({100 * len(idx) / len(y_train):.0f}%), по классам {np.bincount(y_train[idx])}")

    projection = PCA(n_components=2).fit_transform(Z_train)
    plots.plot_prototypes(projection, y_train, idx, noise, class_names,
                          f"Эталоны в проекции на две главные компоненты, k = {k}",
                          os.path.join(IMAGES, "prototypes.png"))
    plots.plot_margins(margins, f"Отступы обучающих объектов, k = {k}", os.path.join(IMAGES, "margins.png"))
    return idx


def compare_with_reference(Z_train, y_train, Z_test, y_test, k):
    print(f"\nсравнение с эталоном sklearn, k = {k}:")
    D_test = knn.distances(Z_train, Z_test)
    own = {"своя реализация, голосование": knn.predict(D_test, y_train, k),
           "своя реализация, Парзен": knn.predict_parzen(D_test, y_train, k)}
    reference = {}
    for weights in ["uniform", "distance"]:
        model = KNeighborsClassifier(n_neighbors=k, weights=weights).fit(Z_train, y_train)
        reference[f"KNeighborsClassifier, {weights}"] = model.predict(Z_test)

    for name, y_pred in {**own, **reference}.items():
        print(f"  {name:32s} accuracy = {accuracy(y_test, y_pred):.3f}")
    votes, uniform = own["своя реализация, голосование"], reference["KNeighborsClassifier, uniform"]
    print(f"  расхождений с эталоном (голосование против uniform): {int((votes != uniform).sum())} из {len(y_test)}")

    start = time.perf_counter()
    knn.predict(knn.distances(Z_train, Z_test), y_train, k)
    own_time = time.perf_counter() - start
    start = time.perf_counter()
    KNeighborsClassifier(n_neighbors=k).fit(Z_train, y_train).predict(Z_test)
    reference_time = time.perf_counter() - start
    print(f"  время на 53 объекта: своя {own_time * 1000:.1f} мс, sklearn {reference_time * 1000:.1f} мс")


if __name__ == "__main__":
    main()
