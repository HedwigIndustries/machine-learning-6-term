import typing as tp
from enum import Enum

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

MetricFunction = tp.Callable[[np.ndarray, np.ndarray], np.ndarray]
KernelFunction = tp.Callable[[np.ndarray], np.ndarray]


def my_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, alpha: int = 1) -> float:
    correct_predictions = np.sum(y_true == y_pred) * alpha
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy


def uniform_kernel(distances: np.ndarray) -> np.ndarray:
    return np.where(distances < 1, 1, 0)


def gaussian_kernel(distances: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * distances ** 2)


def epanechnikov_kernel(distances: np.ndarray) -> np.ndarray:
    return np.where(distances < 1, 0.75 * (1 - distances ** 2), 0)


def custom_kernel(distances: np.ndarray) -> np.ndarray:
    return np.ones_like(distances) * 0.5


def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    cosine_similarity = dot_product / (norm_x * norm_y)
    return 1 - cosine_similarity


def minkowski_distance(x: np.ndarray, y: np.ndarray, p: int = 2) -> np.ndarray:
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


def custom_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(x - y) ** 2)


def create_minkowski_distance(p: int) -> MetricFunction:
    def distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return minkowski_distance(x, y, p)

    return distance


class Metric(Enum):
    COSINE: MetricFunction = cosine_distance
    MINKOWSKI: tp.Callable[[int], MetricFunction] = create_minkowski_distance
    CUSTOM_METRIC: MetricFunction = custom_distance


class Kernel(Enum):
    UNIFORM: KernelFunction = uniform_kernel
    GAUSSIAN: KernelFunction = gaussian_kernel
    EPANECHNIKOV: KernelFunction = epanechnikov_kernel
    CUSTOM_KERNEL: KernelFunction = custom_kernel


class Window(Enum):
    FIXED = 'fixed'
    NON_FIXED = 'not_fixed'


class Knn(BaseEstimator, ClassifierMixin):
    def __init__(
            self, k: int = 3,
            window: Window = Window.FIXED,
            fixed_radius: float = 1.0,
            kernel: KernelFunction = Kernel.UNIFORM,
            metric: MetricFunction = Metric.MINKOWSKI,
            p: int = 2,
            sample_weights: tp.Optional[np.ndarray] = None
    ) -> None:
        self.k = k
        self.window = window
        self.fixed_radius = fixed_radius
        self.kernel: KernelFunction = kernel
        self.metric: MetricFunction = metric
        self.p = p
        self.sample_weights: tp.Optional[np.ndarray] = sample_weights
        self.X_train = None
        self.y_train = None
        self.nn = None
        self.classes_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('Number of training samples does not match')
        self.X_train: np.ndarray = X_train
        self.y_train: np.ndarray = y_train
        if self.metric == Metric.MINKOWSKI:
            self.nn: NearestNeighbors = NearestNeighbors(n_neighbors=self.k, metric=self.metric(self.p))
        else:
            self.nn: NearestNeighbors = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self.nn.fit(X_train)
        self.classes_ = np.unique(y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances, divisors, indexes = self._check_window_strategy(X)
        predictions = []
        items_count = X.shape[0]
        for item in range(items_count):
            neighbour_distances, neighbour_indexes = self._normalize_distances(distances, divisors, indexes, item)
            neighbor_labels = self.y_train[neighbour_indexes]
            kernel_weights = self.kernel(neighbour_distances)
            if self.sample_weights is not None:
                prior_weights = self.sample_weights[neighbour_indexes]
                combined_weights = kernel_weights * prior_weights
            else:
                combined_weights = kernel_weights

            prediction = self._get_most_relevant_class(combined_weights, neighbor_labels)
            predictions.append(prediction)
        return np.array(predictions)

    def _normalize_distances(self, distances, divisors, indexes, item):
        neighbour_distances = distances[item] / divisors[item]
        neighbour_indexes = indexes[item]
        if len(distances[item]) > self.k:
            neighbour_distances = neighbour_distances[:self.k]
            neighbour_indexes = neighbour_indexes[:self.k]
        return neighbour_distances, neighbour_indexes

    def _check_window_strategy(self, X):
        items_count = X.shape[0]
        if self.window == Window.FIXED:
            divisors = np.full(items_count, self.fixed_radius)
            distances, indexes = self.nn.kneighbors(X=X, n_neighbors=self.k)
        elif self.window == Window.NON_FIXED:
            distances, indexes = self.nn.kneighbors(X=X, n_neighbors=self.k + 1)
            divisors = distances[:, -1]
        else:
            raise ValueError('Window must be FIXED or NON_FIXED')
        return distances, divisors, indexes

    def _get_most_relevant_class(self, combined_weights: np.ndarray[float], neighbor_labels: np.ndarray[int]) -> int:
        classes_votes = np.zeros(len(self.classes_), dtype=float)
        for i in range(self.k):
            _class: int = neighbor_labels[i]
            classes_votes[_class] += combined_weights[i]
        prediction: int = np.argmax(classes_votes)
        return prediction

    def get_params(self, deep=True):
        return {
            'k': self.k,
            'window': self.window,
            'fixed_radius': self.fixed_radius,
            'kernel': self.kernel,
            'metric': self.metric,
            'p': 2,
            'sample_weights': self.sample_weights
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def test_model() -> None:
    X_test, X_train, y_test, y_train = _get_split_data()
    knn = Knn(k=1, window=Window.FIXED, metric=Metric.MINKOWSKI, p=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print_accuracy(y_pred, y_test)


def _get_split_data() -> tp.Any:
    data = load_iris()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


def print_accuracy(y_pred, y_test) -> None:
    test_accuracy = my_accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy}")


def main() -> None:
    test_model()


if __name__ == "__main__":
    main()
