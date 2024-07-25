import typing as tp

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class RandomForest:
    def __init__(
            self,
            n_estimators: int = 50,
            max_depth: int = 1,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
    ):
        self.n_estimators: int = n_estimators
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.trees: list[DecisionTreeClassifier] = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_part = X[indices]
            y_part = y[indices]
            dt = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            dt.fit(X_part, y_part)
            self.trees.append(dt)

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        n_trees = len(self.trees)
        predictions = np.zeros((n_samples, n_trees))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        return np.array([np.bincount(predictions[i].astype(int)).argmax() for i in range(n_samples)])


def print_accuracy(y_pred, y_test):
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy}")


def _get_split_data() -> tp.Any:
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


def test_random_forest() -> None:
    X_test, X_train, y_test, y_train = _get_split_data()
    random_forest = RandomForest(n_estimators=1000, max_depth=3)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    print_accuracy(y_pred, y_test)


def main() -> None:
    test_random_forest()


if __name__ == '__main__':
    main()
