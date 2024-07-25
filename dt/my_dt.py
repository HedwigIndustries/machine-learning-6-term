import typing as tp

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature: tp.Optional[int] = feature
        self.threshold: tp.Optional[float] = threshold
        self.left: tp.Optional[Node] = left
        self.right: tp.Optional[Node] = right
        self.value: tp.Optional[np.signedinteger] = value

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth: tp.Optional[int] = None, min_samples_split: int = 2, min_samples_leaf: int = 1):
        self.max_depth: tp.Optional[int] = max_depth
        self.min_samples_to_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.tree: tp.Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))

        if self.max_depth and depth >= self.max_depth \
                or self.n_classes == 1 \
                or n_samples < self.min_samples_to_split:
            leaf_value: np.signedinteger = self._most_common_label(y)
            return Node(value=leaf_value)

        rand_features = np.random.choice(n_features, n_features, replace=False)
        best_feature, best_threshold = self._best_criteria(X, y, rand_features)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)

        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            leaf_value: np.signedinteger = self._most_common_label(y)
            return Node(value=leaf_value)

        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_criteria(self, X: np.ndarray, y: np.ndarray, features: np.ndarray) -> tuple[int, float]:
        best_gain = -1
        best_feature, best_threshold = None, None
        for feature in features:
            X_column_by_feature = X[:, feature]
            thresholds = np.unique(X_column_by_feature)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column_by_feature=X_column_by_feature, split_thresh=threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _information_gain(self, y, X_column_by_feature: np.ndarray, split_thresh: float) -> float:
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column_by_feature, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n_total, n_left, n_right = len(y), len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        k_left, k_right = n_left / n_total, n_right / n_total
        child_entropy = k_left * entropy_left + k_right * entropy_right
        ig = parent_entropy - child_entropy
        return ig

    @staticmethod
    def _split(X_column_by_feature: np.ndarray, split_thresh: float) -> tuple[np.ndarray, np.ndarray]:
        left_idxs = np.argwhere(X_column_by_feature <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column_by_feature > split_thresh).flatten()
        return left_idxs, right_idxs

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        p_lst = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in p_lst if p > 0])

    @staticmethod
    def _most_common_label(y: np.ndarray) -> np.signedinteger:
        return np.bincount(y).argmax()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node) -> np.signedinteger:
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


def print_accuracy(y_pred, y_test):
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy}")


def _get_split_data() -> tp.Any:
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


def test_decision_tree() -> None:
    X_test, X_train, y_test, y_train = _get_split_data()
    dt = DecisionTree()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print_accuracy(y_pred, y_test)


def main() -> None:
    test_decision_tree()


if __name__ == '__main__':
    main()
