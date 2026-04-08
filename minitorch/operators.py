from typing import Iterable, Any, Callable
import math

# Collection of the core mathematical operators used throughout the code base.


# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    """Multiplies two numbers"""
    return a * b


def id(a: float) -> float:
    """Returns the input unchanged"""
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers"""
    return a + b


def neg(a: float) -> float:
    """Negates a number"""
    return -a


def lt(a: float, b: float) -> float:
    """Checks if a is less than b"""
    return float(a < b)


def eq(a: float, b: float) -> float:
    """Checks if two numbers are equal"""
    return float(a == b)


def max(a: float, b: float) -> float:
    """Returns the larger of two numbers"""
    if a > b:
        return a
    else:
        return b


def is_close(a: float, b: float) -> float:
    """Checks if two numbers are close in value"""
    return abs(a - b) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    if x >= 0:
        return 1.0 / (1.0 + math.e ** (-x))
    else:
        return (math.e**x) / (1.0 + math.e**x)


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return max(0.0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm"""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal"""
    if x == 0:
        return x

    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg"""
    return inv(x) * y


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of inv times a second arg"""
    return inv(x**2) * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    if x < 0.0:
        return 0.0
    return 1.0 * y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(ls: Iterable[Any], fn: Callable[[Any], Any]) -> Iterable[Any]:
    """Higher-order function that applies a given function to each element of an iterable"""
    result = []
    for i, el in enumerate(ls):
        result.append(fn(el))

    return result


def zipWith(
    ls1: Iterable[Any], ls2: Iterable[Any], fn: Callable[[Any, Any], Any]
) -> Iterable[Any]:
    """Higher-order function that combines elements from two iterables using a given function"""
    result = []
    for x, y in zip(ls1, ls2):
        result.append(fn(x, y))

    return result


def reduce(ls: Iterable[Any], fn: Callable[[Any, Any], Any]) -> Any:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    iterator = iter(ls)

    try:
        result = next(iterator)
    except StopIteration:
        return 0.0

    for el in iterator:
        result = fn(result, el)

    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list"""
    return map(ls, neg)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists"""
    return zipWith(ls1, ls2, add)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list"""
    return reduce(ls, add)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list"""
    return reduce(ls, mul)
