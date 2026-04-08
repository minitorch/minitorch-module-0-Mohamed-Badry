import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate N random 2D points.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        List[Tuple[float, float]]: List of 2D points.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """A dataset graph containing points and their corresponding classes.

    Attributes
    ----------
        N (int): Number of points in the dataset.
        X (List[Tuple[float, float]]): List of 2D points.
        y (List[int]): List of class labels (0 or 1) for each point.

    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """A simple linearly separable dataset with 2 classes (left and right) and N points.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the dataset.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """A linearly separable dataset with 2 classes separated by a diagonal line.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the dataset.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """A dataset with 2 classes split into 3 vertical regions where one class
    is bounded by the other class from both sides.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the dataset.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """A dataset with 2 classes split into 4 quadrants.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the dataset.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """A circular dataset where one class falls inside the circle and the other outside it.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the dataset.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """A spiral dataset with both classes perfectly separated by a spiral.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the dataset.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
