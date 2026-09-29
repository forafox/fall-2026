import os

import numpy as np

import plots
from data import load_moons, standardize, train_test_split
from svm import SVM, linear, polynomial, rbf

IMAGES = os.path.join(os.path.dirname(__file__), "..", "images")


def main():
    X, y = load_moons()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    Z_train, Z_test = standardize(X_train), standardize(X_test, X_train)

    kernels = {"линейное": linear, "полиномиальное, d = 3": polynomial(3), "RBF, γ = 1": rbf(1.0)}
    models = {}
    for name, kernel in kernels.items():
        model = SVM(C=1.0, kernel=kernel).fit(Z_train, y_train)
        accuracy = np.mean(model.predict(Z_test) == y_test)
        print(f"{name:22s} accuracy = {accuracy:.3f}, опорных = {len(model.lam)}")
        models[f"{name}, accuracy = {accuracy:.3f}"] = model

    plots.plot_decision(models, Z_train, y_train, os.path.join(IMAGES, "moons_kernels.png"))


if __name__ == "__main__":
    main()
