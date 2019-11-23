#!/usr/bin/env python
import scipy.integrate as integrate
import typing as tp
import numpy as np


# section 1 subsection 1 problem 1
def solve_system(a: np.matrix, b: np.array) -> np.array:
    """
    :param a: m times n real matrix
    :param b: m-dimensional real vector
    :return: n-dimensional real vector x,
        least squares solution to A x = b.
    """
    return np.linalg.pinv(a) * b


# section 1 subsection 2 problem 1
def solve_summed_system(as_list: tp.List[np.matrix],
                        b: np.array) -> tp.List[np.array]:
    """
    :param as_list: list of length N,
        consisting of m times n real matrixes A_i
    :param b: m-dimensional real vector
    :return: list x of length N,
        consisting of n-dimensional real vectors x_i,
        least squares solution to sum_{i = 1}^N A_i x_i = b.
    """
    p_1 = sum(a_i * a_i.T for a_i in as_list)
    return [a_i.T * np.linalg.pinv(p_1) * b for a_i in as_list]


# section 1 subsection 2 problem 2
def solve_time_summed_system(a: tp.Callable[[float], np.matrix], b: np.array,
                             t: tp.List[float]) -> tp.List[np.array]:
    """
    :param a: matrix-valued function of time.
        Maps t_i to m times n real matrix A(t_i)
    :param b: m-dimensional real vector
    :param t: list of length N,
        consisting of time moments, t_i
    :return: list x of length N,
        consisting of n-dimensional real vectors x(t_i),
        least squares solution to sum_{i = 1}^N A(t_i) x(t_i) = b.
    """
    as_list = [a(t_i) for t_i in t]
    return solve_summed_system(as_list, b)


# section 1 subsection 3 problem 1
def solve_distributed_system(as_list: tp.List[np.matrix],
                             bs_list: tp.List[np.array]) -> np.array:
    """
    :param as_list: list of length N,
        consisting of m times n real matrixes A_i
    :param bs_list: list of length N,
        consisting of m-dimensional real vectors b_i
    :return: n-dimensional real vector x,
        least squares solution to A_i x = b_i, i = 1..N.
    """
    a_b = sum(a_i.T * b_i for a_i, b_i in zip(as_list, bs_list))
    p_2 = sum(a_i.T * a_i for a_i in as_list)
    return np.linalg.pinv(p_2) * a_b


# section 1 subsection 3 problem 2
def solve_time_distributed_system(a: tp.Callable[[float], np.matrix],
                                  b: tp.Callable[[float], np.array],
                                  t: tp.List[float]) -> np.array:
    """
    :param a: matrix-valued function of time.
        Maps t_i to m times n real matrix A(t_i)
    :param b: vector-valued function of time.
        Maps t_i to m-dimensional real vector b(t_i)
    :param t: list of length N,
        consisting of time moments, t_i
    :return: n-dimensional real vector x,
        least squares solution to A(t_i) x = b(t_i), i = 1..N.
    """
    as_list = [a(t_i) for t_i in t]
    bs_list = [b(t_i) for t_i in t]
    return solve_distributed_system(as_list, bs_list)


# section 1 subsection 4 problem 1
def solve_integral_system(a: tp.Callable[[float], np.matrix], b: np.array,
                          T: float) -> tp.Callable[[float], np.array]:
    """
    :param a: matrix-valued function of time.
        Maps t to m times n real matrix A(t)
    :param b: m-dimensional real vector
    :param T: end time
    :return: vector-valued function of time.
        Maps t to n-dimensional real vector x(t),
        least squares solution to int_0^T A(t) x(t) dt = b.
    """
    m, _ = (a(0) * a(0).T).shape

    p_1 = np.matrix([[integrate.quad(
        lambda t: (a(t) * a(t).T)[i, j], 0, T
    )[0] for j in range(m)] for i in range(m)])

    def x(t: float) -> np.array:
        return a(t).T * np.linalg.pinv(p_1) * b

    return x


# section 1 subsection 4 problem 2
def solve_functional_system(a: tp.Callable[[float], np.matrix],
                            b: tp.Callable[[float], np.array],
                            T: float) -> np.array:
    """
    :param a: matrix-valued function of time.
        Maps t to m times n real matrix A(t)
    :param b: vector-valued function of time.
        Maps t to m-dimensional real vector b(t)
    :param T: end time
    :return: n-dimensional real vector x,
        least squares solution to A(t) x = b(t), t in [0, T].
    """
    n, _ = (a(0).T * b(0)).shape

    a_b = np.array([[integrate.quad(
        lambda t: (a(t).T * b(t))[i], 0, T
    )[0]] for i in range(n)])

    p_2 = np.matrix([[integrate.quad(
        lambda t: (a(t).T * a(t))[i, j], 0, T
    )[0] for j in range(n)] for i in range(n)])

    return np.linalg.pinv(p_2) * a_b


# section 1 subsection 5 problem 1 dimensionality 1
def solve_1d_space_distributed_integral_system(
        g: tp.Callable[[float, float], float], us_list: tp.List[float],
        xts_list: tp.List[tp.Tuple[float, float]], a: float, b: float,
        T: float) -> tp.Callable[[float, float], float]:
    """
    :param g: real-valued function of space and time.
        Maps x, t to G(x, t)
    :param us_list: list of N real values, u(x_i, t_i)
    :param xts_list: list of length N,
        consisting of space-time points (x_i, t_i).
        The equation is optimized at these points
    :param a: lower bound of the x-domains of g and u
    :param b: upper bound of the x-domains of g and u
    :param T: end time
    :return: real-valued function f of space and time,
        least squares solutions to
        int_a^b int_0^T G(x - x_i, t - t_i) f(x, t) dt dx
            = u(x_i, t_i), i = 1..N.
    """
    class gComputer:
        def __init__(self, x: float, t: float) -> None:
            self._x, self._t = x, t

        def __call__(self, x: float, t: float) -> float:
            return g(self._x - x, self._t - t)

    g_1 = [gComputer(x_i, t_i) for x_i, t_i in xts_list]

    vec_u = np.array([[u_i] for u_i in us_list])

    p_1 = np.matrix([[integrate.dblquad(
        lambda t, x: g_i(x, t) * g_j(x, t), a, b, 0, T
    )[0] for g_j in g_1] for g_i in g_1])

    def f(x: float, t: float) -> float:
        g_1_local = np.array([g_i(x, t) for g_i in g_1])
        return (g_1_local * np.linalg.pinv(p_1) * vec_u)[0, 0]

    return f


def solve_1d_space_distributed_integral_system_ufunc(
        g: tp.Callable[[float, float], float],
        u: tp.Callable[[float, float], float],
        xts_list: tp.List[tp.Tuple[float, float]],
        a: float, b: float, T: float) -> tp.Callable[[float, float], float]:
    """
    :param g: real-valued function of space and time.
        Maps x, t to G(x, t)
    :param u: real-valued function of space and time.
        Maps x, t to u(x, t)
    :param xts_list: list of length N,
        consisting of space-time points (x_i, t_i).
        The equation is optimized at these points
    :param a: lower bound of the x-domains of g and u
    :param b: upper bound of the x-domains of g and u
    :param T: end time
    :return: real-valued function f of space and time,
        least squares solutions to
        int_a^b int_0^T G(x - x_i, t - t_i) f(x, t) dt dx
            = u(x_i, t_i), i = 1..N.
    """
    us_list = [u(x_i, t_i) for x_i, t_i in xts_list]

    return solve_1d_space_distributed_integral_system(
        g, us_list, xts_list, a, b, T)


# section 1 subsection 5 problem 1 dimensionality 2
def solve_2d_space_distributed_integral_system(
        g: tp.Callable[[float, float, float], float], us_list: tp.List[float],
        xyts_list: tp.List[tp.Tuple[float, float, float]],
        a: float, b: float, c: float, d: float,
        T: float) -> tp.Callable[[float, float, float], float]:
    """
    :param g: real-valued function of space and time.
        Maps x, y, t to G(x, y, t)
    :param us_list: list of length N,
        consisting of real values u(x_i, y_i, t_i)
    :param xyts_list: list of length N,
        consisting of space-time points (x_i, y_i, t_i).
        The equation is optimized at these points
    :param a: lower bound of the x-domains of g and u
    :param b: upper bound of the x-domains of g and u
    :param c: lower bound of the y-domains of g and u
    :param d: upper bound of the y-domains of g and u
    :param T: end time
    :return: real-valued function f of space and time,
        least squares solutions to
        int_c^d int_a^b int_0^T G(x - x_i, y - y_i, t - t_i) f(x, y, t) dt dx dy
            = u(x_i, y_i, t_i), i = 1..N.
    """
    class gComputer:
        def __init__(self, x: float, y: float, t: float) -> None:
            self._x, self._y, self._t = x, y, t

        def __call__(self, x: float, y: float, t: float) -> float:
            return g(self._x - x, self._y - y, self._t - t)

    g_1 = [gComputer(x_i, y_i, t_i) for x_i, y_i, t_i in xyts_list]

    vec_u = np.array([[u_i] for u_i in us_list])

    p_1 = np.matrix([[integrate.tplquad(
        lambda t, x, y: g_i(x, y, t) * g_j(x, y, t), c, d, a, b, 0, T
    )[0] for g_j in g_1] for g_i in g_1])

    def f(x: float, y: float, t: float) -> float:
        g_1_local = np.array([g_i(x, y, t) for g_i in g_1])
        return (g_1_local * np.linalg.pinv(p_1) * vec_u)[0, 0]

    return f


def solve_2d_space_distributed_integral_system_ufunc(
        g: tp.Callable[[float, float, float], float],
        u: tp.Callable[[float, float, float], float],
        xyts_list: tp.List[tp.Tuple[float, float, float]],
        a: float, b: float, c: float, d: float,
        T: float) -> tp.Callable[[float, float, float], float]:
    """
    :param g: real-valued function of space and time.
        Maps x, y, t to G(x, y, t)
    :param u: real-valued function of space and time.
        Maps x, y, t to u(x, y, t)
    :param xyts_list: list of length N,
        consisting of space-time points (x_i, y_i, t_i).
        The equation is optimized at these points
    :param a: lower bound of the x-domains of g and u
    :param b: upper bound of the x-domains of g and u
    :param c: lower bound of the y-domains of g and u
    :param d: upper bound of the y-domains of g and u
    :param T: end time
    :return: real-valued function f of space and time,
        least squares solutions to
        int_c^d int_a^b int_0^T G(x - x_i, y - y_i, t - t_i) f(x, y, t) dt dx dy
            = u(x_i, y_i, t_i), i = 1..N.
    """
    us_list = [u(x_i, y_i, t_i) for x_i, y_i, t_i in xyts_list]

    return solve_2d_space_distributed_integral_system(
        g, us_list, xyts_list, a, b, c, d, T)


# section 1 subsection 5 problem 2 dimensionality 1
def solve_1d_space_distributed_functional_system(
        g: tp.Callable[[float, float], float],
        u: tp.Callable[[float, float], float],
        xts_list: tp.List[tp.Tuple[float, float]],
        a: float, b: float, T: float) -> np.array:
    """
    :param g: real-valued function of space and time.
        Maps x, t to G(x, t)
    :param u: real-valued function of space and time.
        Maps x, t to u(x, t)
    :param xts_list: list of length N,
        consisting of space-time points (x_i, t_i).
        The equation is optimized at these points
    :param a: lower bound of the x-domains of g and u
    :param b: upper bound of the x-domains of g and u
    :param T: end time
    :return: N-dimensional real vector, f(x_i, t_i), i = 1..N,
        least squares solutions to
        G(x - x_i, t - t_i) f(x_i, t_i) = u(x, t), x in [a, b], t in [0, T].
    """
    class gComputer:
        def __init__(self, x: float, t: float) -> None:
            self._x, self._t = x, t

        def __call__(self, x: float, t: float) -> float:
            return g(self._x - x, self._t - t)

    g_2 = [gComputer(x_i, t_i) for x_i, t_i in xts_list]

    p_2 = np.matrix([[integrate.dblquad(
        lambda t, x: g_i(x, t) * g_j(x, t), a, b, 0, T
    )[0] for g_j in g_2] for g_i in g_2])

    g_u = np.array([[integrate.dblquad(
        lambda t, x: g_i(x, t) * u(x, t), a, b, 0, T
    )[0]] for g_i in g_2])

    return np.linalg.pinv(p_2) * g_u


# section 1 subsection 5 problem 2 dimensionality 2
def solve_2d_space_distributed_functional_system(
        g: tp.Callable[[float, float, float], float],
        u: tp.Callable[[float, float, float], float],
        xyts_list: tp.List[tp.Tuple[float, float, float]],
        a: float, b: float, c: float, d: float, T: float) -> np.array:
    """
    :param g: real-valued function of space and time.
        Maps x, y, t to G(x, y, t)
    :param u: real-valued function of space and time.
        Maps x, y, t to u(x, y, t)
    :param xyts_list: list of length N,
        consisting of space-time points (x_i, y_i, t_i).
        The equation is optimized at these points
    :param a: lower bound of the x-domains of g and u
    :param b: upper bound of the x-domains of g and u
    :param c: lower bound of the y-domains of g and u
    :param d: upper bound of the y-domains of g and u
    :param T: end time
    :return: N-dimensional real vector, f(x_i, y_i, t_i), i = 1..N,
        least squares solutions to
        G(x - x_i, y - y_i, t - t_i) f(x_i, y_i, t_i) = u(x, y, t),
        x in [a, b], y in [c, d], t in [0, T].
    """
    class gComputer:
        def __init__(self, x: float, y: float, t: float) -> None:
            self._x, self._y, self._t = x, y, t

        def __call__(self, x: float, y: float, t: float) -> float:
            return g(self._x - x, self._y - y, self._t - t)

    g_2 = [gComputer(x_i, y_i, t_i) for x_i, y_i, t_i in xyts_list]

    p_2 = np.matrix([[integrate.tplquad(
        lambda t, x, y: g_i(x, y, t) * g_j(x, y, t), c, d, a, b, 0, T
    )[0] for g_j in g_2] for g_i in g_2])

    g_u = np.array([[integrate.tplquad(
        lambda t, x, y: g_i(x, y, t) * u(x, y, t), c, d, a, b, 0, T
    )[0]] for g_i in g_2])

    return np.linalg.pinv(p_2) * g_u
