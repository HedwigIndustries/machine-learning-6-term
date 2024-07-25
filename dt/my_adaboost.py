import typing as tp

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def __init__(
            self,
            n_estimators: int = 50,
            max_depth: int = 1,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1
    ):
        self.n_estimators: int = n_estimators
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.alpha_lst: list[float] = []
        self.trees: list[DecisionTreeClassifier] = []

    def fit(self, X, y) -> None:
        n_samples, n_features = X.shape
        weights = np.ones(n_samples) / n_samples
        for _ in range(self.n_estimators):
            dt = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            dt.fit(X, y, weights)
            y_pred = dt.predict(X)
            miss = self._calc_miss(weights, y, y_pred)
            alpha_t = self._calc_optimal_alpha(miss)
            margin = y * y_pred
            weights = self._update_weights(alpha_t, margin, weights)
            weights = self._normalize_weights(weights)
            self.alpha_lst.append(alpha_t)
            self.trees.append(dt)

    @staticmethod
    def _normalize_weights(weights: np.ndarray[float]) -> np.ndarray[float]:
        return weights / np.sum(weights)

    @staticmethod
    def _update_weights(alpha_t, margin: np.ndarray[int], weights: np.ndarray[float]) -> np.ndarray[float]:
        return weights * np.exp(-alpha_t * margin)

    @staticmethod
    def _calc_optimal_alpha(miss: float) -> float:
        # maybe lr?
        # maybe eps
        eps = 1e6
        return 0.5 * np.log((1 - miss) / (miss + eps))

    @staticmethod
    def _calc_miss(weights: np.ndarray[float], y: np.ndarray[int], y_pred: np.ndarray[int]) -> float:
        miss_mask = y_pred != y
        np_sum = np.sum(weights)
        eps = 0 if np_sum == 0 else 1e-8
        return np.sum(weights * miss_mask) / (np_sum + eps)

    def predict(self, X):
        y_pred_lst = np.array([tree.predict(X) for tree in self.trees])
        return np.sign(np.dot(self.alpha_lst, y_pred_lst))


def print_accuracy(y_pred, y_test):
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy}")


def _get_split_data() -> tp.Any:
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


def test_adaboost() -> None:
    X_test, X_train, y_test, y_train = _get_split_data()
    boost = AdaBoost(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=2)
    boost.fit(X_train, y_train)
    y_pred = boost.predict(X_test)
    print_accuracy(y_pred, y_test)


def main() -> None:
    test_adaboost()


if __name__ == '__main__':
    main()
