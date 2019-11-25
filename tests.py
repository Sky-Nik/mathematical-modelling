#!/usr/bin/env python
import numpy as np
import functools
import datetime
import sys
import lib


class AbstractTester:
    @classmethod
    def __init__(cls):
        for meth in dir(cls):
            if 'test' in meth:
                getattr(cls, meth)()


class SolveLinearSystemTester(AbstractTester):
    @staticmethod
    def test_diagonal():
        a = np.matrix([[1, 0], [0, 1]])
        b = np.array([[1], [2]])

        desired = np.array([[1], [2]])
        actual = lib.solve_system(a, b)

        np.testing.assert_almost_equal(actual, desired)

    @staticmethod
    def test_anti_diagonal():
        a = np.matrix([[0, 1], [1, 0]])
        b = np.array([[1], [2]])

        desired = np.array([[2], [1]])
        actual = lib.solve_system(a, b)

        np.testing.assert_almost_equal(actual, desired)

    @staticmethod
    def test_pseudo():
        a = np.matrix([[1, 2], [1, 3], [1, 5]])
        b = np.array([[1], [2], [3]])

        desired = np.array([[-1/7], [9/14]])
        actual = lib.solve_system(a, b)

        np.testing.assert_almost_equal(actual, desired)


class SolveSummedLinearSystemTester(AbstractTester):
    @staticmethod
    def test_two_easy():
        as_list = [
            np.matrix([[1, 0], [0, 1], [0, 0], [0, 0]]),
            np.matrix([[0, 0], [0, 0], [1, 0], [0, 1]]),
        ]

        b = np.array([[1], [2], [3], [4]])

        desired = [np.array([[1], [2]]), np.array([[3], [4]])]
        actual = lib.solve_summed_system(as_list, b)

        for actual_i, desired_i in zip(actual, desired):
            np.testing.assert_almost_equal(actual_i, desired_i)

    @staticmethod
    def test_two_hard():
        as_list = [np.matrix([[1, 2], [3, 4]]), np.matrix([[1, 2], [1, 3]])]
        b = np.array([[5], [7]])

        desired = [
            np.array([[-0.423077], [0.692308]]),
            np.array([[1.115385], [1.461538]]),
        ]

        actual = lib.solve_summed_system(as_list, b)

        for actual_i, desired_i in zip(actual, desired):
            np.testing.assert_almost_equal(actual_i, desired_i, 6)

    @staticmethod
    def test_two_pseudo():
        as_list = [
            np.matrix([[1, 0], [1, 1], [0, 1], [0, 0], [0, 0]]),
            np.matrix([[0, 0], [0, 0], [1, 0], [1, 1], [0, 1]]),
        ]

        b = np.array([[5], [8], [13], [21], [34]])

        desired = [np.array([[0.4], [12.2]]), np.array([[-3.8], [29.4]])]
        actual = lib.solve_summed_system(as_list, b)

        for actual_i, desired_i in zip(actual, desired):
            np.testing.assert_almost_equal(actual_i, desired_i)


class SolveTimeSummedLinearSystemTester(AbstractTester):
    @staticmethod
    def test_two_easy():
        t = [0.0, 1.0]

        def a(t_i: float) -> np.matrix:
            return {
                0.0: np.matrix([[1, 0], [0, 1], [0, 0], [0, 0]]),
                1.0: np.matrix([[0, 0], [0, 0], [1, 0], [0, 1]]),
            }[t_i]

        b = np.array([[1], [2], [3], [4]])

        desired = [np.array([[1], [2]]), np.array([[3], [4]])]
        actual = lib.solve_time_summed_system(a, b, t)

        for actual_i, desired_i in zip(actual, desired):
            np.testing.assert_almost_equal(actual_i, desired_i)

    @staticmethod
    def test_two_hard():
        t = [0.0, 1.0]

        def a(t_i: float) -> np.matrix:
            return {
                0.0: np.matrix([[1, 2], [3, 4]]),
                1.0: np.matrix([[1, 2], [1, 3]]),
            }[t_i]

        b = np.array([[5], [7]])

        desired = [
            np.array([[-0.423077], [0.692308]]),
            np.array([[1.115385], [1.461538]]),
        ]

        actual = lib.solve_time_summed_system(a, b, t)

        for actual_i, desired_i in zip(actual, desired):
            np.testing.assert_almost_equal(actual_i, desired_i, 6)

    @staticmethod
    def test_two_pseudo():
        t = [0.0, 1.0]

        def a(t_i: float) -> np.matrix:
            return {
                0.0: np.matrix([[1, 0], [1, 1], [0, 1], [0, 0], [0, 0]]),
                1.0: np.matrix([[0, 0], [0, 0], [1, 0], [1, 1], [0, 1]]),
            }[t_i]

        b = np.array([[5], [8], [13], [21], [34]])

        desired = [np.array([[0.4], [12.2]]), np.array([[-3.8], [29.4]])]
        actual = lib.solve_time_summed_system(a, b, t)

        for actual_i, desired_i in zip(actual, desired):
            np.testing.assert_almost_equal(actual_i, desired_i)


class SolveDistributedLinearSystemTester(AbstractTester):
    @staticmethod
    def test_two_easy():
        as_list = [
            np.matrix([[1, 0], [0, 1]]),
            np.matrix([[1, 0], [0, 1]]),
        ]

        bs_list = [
            np.array([[2], [3]]),
            np.array([[2], [3]]),
        ]

        desired = np.array([[2], [3]])
        actual = lib.solve_distributed_system(as_list, bs_list)

        np.testing.assert_almost_equal(actual, desired)

    @staticmethod
    def test_two_hard():
        as_list = [
            np.matrix([[1, 0], [0, 1]]),
            np.matrix([[1, 2], [3, 4]]),
        ]

        bs_list = [
            np.array([[2], [3]]),
            np.array([[8], [18]]),
        ]

        desired = np.array([[2], [3]])
        actual = lib.solve_distributed_system(as_list, bs_list)

        np.testing.assert_almost_equal(actual, desired)

    @staticmethod
    def test_two_pseudo():
        as_list = [
            np.matrix([[1, 0], [0, 1]]),
            np.matrix([[0, 1], [1, 0]]),
        ]

        bs_list = [
            np.array([[2], [3]]),
            np.array([[5], [8]]),
        ]

        desired = np.array([[5], [4]])
        actual = lib.solve_distributed_system(as_list, bs_list)

        np.testing.assert_almost_equal(actual, desired)


class SolveTimeDistributedLinearSystemTester(AbstractTester):
    @staticmethod
    def test_two_easy():
        t = [0.0, 1.0]

        def a(t_i: float) -> np.matrix:
            return {
                0.0: np.matrix([[1, 0], [0, 1]]),
                1.0: np.matrix([[1, 0], [0, 1]]),
            }[t_i]

        def b(t_i: float) -> np.array:
            return {
                0.0: np.array([[2], [3]]),
                1.0: np.array([[2], [3]]),
            }[t_i]

        desired = np.array([[2], [3]])
        actual = lib.solve_time_distributed_system(a, b, t)

        np.testing.assert_almost_equal(actual, desired)

    @staticmethod
    def test_two_hard():
        t = [0.0, 1.0]

        def a(t_i: float) -> np.matrix:
            return {
                0.0: np.matrix([[1, 0], [0, 1]]),
                1.0: np.matrix([[1, 2], [3, 4]]),
            }[t_i]

        def b(t_i: float) -> np.array:
            return {
                0.0: np.array([[2], [3]]),
                1.0: np.array([[8], [18]]),
            }[t_i]

        desired = np.array([[2], [3]])
        actual = lib.solve_time_distributed_system(a, b, t)

        np.testing.assert_almost_equal(actual, desired)

    @staticmethod
    def test_two_pseudo():
        t = [0.0, 1.0]

        def a(t_i: float) -> np.matrix:
            return {
                0.0: np.matrix([[1, 0], [0, 1]]),
                1.0: np.matrix([[0, 1], [1, 0]]),
            }[t_i]

        def b(t_i: float) -> np.array:
            return {
                0.0: np.array([[2], [3]]),
                1.0: np.array([[5], [8]]),
            }[t_i]

        desired = np.array([[5], [4]])
        actual = lib.solve_time_distributed_system(a, b, t)

        np.testing.assert_almost_equal(actual, desired)


class SolveIntegralLinearSystemTester(AbstractTester):
    @staticmethod
    def test_simple():
        def a(time: float) -> np.matrix:
            return np.matrix([[time, 2 * time], [3 * time, 4 * time]])

        b = np.array([[1], [2]])
        T = 1.0

        def desired(time: float) -> np.matrix:
            return np.matrix([[0], [1.5 * time]])

        actual = lib.solve_integral_system(a, b, T)
        for t in np.linspace(0.0, 1.0, 101):
            np.testing.assert_almost_equal(actual(t), desired(t))


class SolveFunctionalLinearSystemTester(AbstractTester):
    @staticmethod
    def test_simple():
        def a(t: float) -> np.matrix:
            return np.matrix([[t, 2 * t], [3 * t, 4 * t]])

        def b(t: float) -> np.matrix:
            return np.matrix([[t], [2 * t]])

        T = 1.0

        desired = np.array([[0], [0.5]])
        actual = lib.solve_functional_system(a, b, T)

        np.testing.assert_almost_equal(actual, desired)


class SolveSpaceDistributedIntegralSystemTester(AbstractTester):
    @staticmethod
    def test_1d_ufunc_simple():
        def g(x: float, t: float) -> float:
            return 1.0

        def u(x: float, t: float) -> float:
            return 1.0

        xts_list = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        a, b, T = 0.0, 1.0, 1.0

        def desired(x: float, t: float) -> float:
            return 1.0

        actual = lib.solve_1d_space_distributed_integral_system_ufunc(g, u, xts_list, a, b, T)

        for x_i, t_i in xts_list:
            np.testing.assert_almost_equal(desired(x_i, t_i), actual(x_i, t_i))

    @staticmethod
    def test_1d_simple():
        def g(x: float, t: float) -> float:
            return 1.0

        us_list = [1.0 for _ in range(4)]

        xts_list = [
            (0.0, 0.0), (0.0, 1.0),
            (1.0, 0.0), (1.0, 1.0),
        ]

        a, b, T = 0.0, 1.0, 1.0

        def desired(x: float, t: float) -> float:
            return 1.0

        actual = lib.solve_1d_space_distributed_integral_system(g, us_list, xts_list, a, b, T)

        for x_i, t_i in xts_list:
            np.testing.assert_almost_equal(desired(x_i, t_i), actual(x_i, t_i))

    @staticmethod
    def test_2d_ufunc_simple():
        def g(x: float, y: float, t: float) -> float:
            return 1.0

        def u(x: float, y: float, t: float) -> float:
            return 1.0

        xyts_list = [
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0), (0.0, 1.0, 1.0),

            (1.0, 0.0, 0.0), (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0), (1.0, 1.0, 1.0),
        ]

        a, b, c, d, T = 0.0, 1.0, 0.0, 1.0, 1.0

        def desired(x: float, y: float, t: float) -> float:
            return 1.0

        actual = lib.solve_2d_space_distributed_integral_system_ufunc(g, u, xyts_list, a, b, c, d, T)

        for x_i, y_i, t_i in xyts_list:
            np.testing.assert_almost_equal(desired(x_i, y_i, t_i), actual(x_i, y_i, t_i))

    @staticmethod
    def test_2d_simple():
        def g(x: float, y: float, t: float) -> float:
            return 1.0

        us_list = [1.0 for _ in range(8)]

        xyts_list = [
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0), (0.0, 1.0, 1.0),

            (1.0, 0.0, 0.0), (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0), (1.0, 1.0, 1.0),
        ]

        a, b, c, d, T = 0.0, 1.0, 0.0, 1.0, 1.0

        def desired(x: float, y: float, t: float) -> float:
            return 1.0

        actual = lib.solve_2d_space_distributed_integral_system(g, us_list, xyts_list, a, b, c, d, T)

        for x_i, y_i, t_i in xyts_list:
            np.testing.assert_almost_equal(desired(x_i, y_i, t_i), actual(x_i, y_i, t_i))


class SolveSpaceDistributedFunctionalSystemTester(AbstractTester):
    @staticmethod
    def test_1d_simple():
        def g(x: float, t: float) -> float:
            return 1.0

        def u(x: float, t: float) -> float:
            return 1.0

        xts_list = [
            (0.0, 0.0), (0.0, 1.0),
            (1.0, 0.0), (1.0, 1.0),
        ]

        a, b, T = 0.0, 1.0, 1.0

        desired = np.array([[0.25] for _ in range(4)])

        actual = lib.solve_1d_space_distributed_functional_system(g, u, xts_list, a, b, T)

        np.testing.assert_almost_equal(desired, actual)

    @staticmethod
    def test_2d_simple():
        def g(x: float, y: float, t: float) -> float:
            return 1.0

        def u(x: float, y: float, t: float) -> float:
            return 1.0

        xyts_list = [
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0), (0.0, 1.0, 1.0),

            (1.0, 0.0, 0.0), (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0), (1.0, 1.0, 1.0),
        ]

        a, b, c, d, T = 0.0, 1.0, 0.0, 1.0, 1.0

        desired = np.array([[0.125] for _ in range(8)])

        actual = lib.solve_2d_space_distributed_functional_system(g, u, xyts_list, a, b, c, d, T)

        np.testing.assert_almost_equal(desired, actual)


class SolveDiscreteObservationsDiscreteModellingTester(AbstractTester):
    @staticmethod
    def test_1d_simple():
        cond_x0s_list = [0.0, 1.0]

        cond_xtGammas_list = [
            (0.0, 0.5), (0.0, 1.0),
            (1.0, 0.5), (1.0, 1.0),
        ]

        cond_f0s_list = [1.0, 1.0]

        cond_fGammas_list = [
            1.0, 1.0,
            1.0, 1.0,
        ]

        model_xtInftys_list = [
            (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
            (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
            (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),
        ]

        model_x0s_list = [0.0, 0.5, 1.0]

        model_xtGammas_list = [
            (0.0, 0.0), (1.0, 0.0),
            (0.0, 0.5), (1.0, 0.5),
            (0.0, 1.0), (1.0, 1.0),
        ]

        def f(x: float, t: float) -> float:
            return 1.0

        def g(x: float, t: float) -> float:
            return 1.0

        def desired(x: float, t: float) -> float:
            return 1.0

        actual = lib.solve_1d_discrete_observations_discrete_modelling(
            cond_x0s_list, cond_xtGammas_list, cond_f0s_list, cond_fGammas_list,
            model_xtInftys_list, model_x0s_list, model_xtGammas_list, f, g)

        xts_list = [
            (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
            (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
            (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),
        ]

        for x_i, t_i in xts_list:
            np.testing.assert_almost_equal(desired(x_i, t_i), actual(x_i, t_i))

    @staticmethod
    def test_2d_simple():
        cond_xy0s_list = [
            (0.0, 0.0), (0.0, 1.0),
            (1.0, 0.0), (1.0, 1.0),
        ]

        cond_xytGammas_list = [
            (0.0, 0.0, 0.5), (0.0, 1.0, 0.5),
            (1.0, 0.0, 0.5), (1.0, 1.0, 0.5),

            (0.0, 0.0, 1.0), (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0), (1.0, 1.0, 1.0),
        ]

        cond_f0s_list = [
            1.0, 1.0,
            1.0, 1.0,
        ]

        cond_fGammas_list = [
            1.0, 1.0,
            1.0, 1.0,

            1.0, 1.0,
            1.0, 1.0,
        ]

        model_xytInftys_list = [
            (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (1.0, 0.0, 0.5),
            (0.0, 0.0, 1.0), (0.5, 0.0, 1.0), (1.0, 0.0, 1.0),

            (0.0, 0.5, 0.0), (0.5, 0.5, 0.0), (1.0, 0.5, 0.0),
            (0.0, 0.5, 0.5), (0.5, 0.5, 0.5), (1.0, 0.5, 0.5),
            (0.0, 0.5, 1.0), (0.5, 0.5, 1.0), (1.0, 0.5, 1.0),

            (0.0, 1.0, 0.0), (0.5, 1.0, 0.0), (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.5), (0.5, 1.0, 0.5), (1.0, 1.0, 0.5),
            (0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0),
        ]

        model_xy0s_list = [
            (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
            (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
            (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),
        ]

        model_xytGammas_list = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.5),
            (1.0, 0.0, 0.5), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),

            (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.5),
            (1.0, 1.0, 0.5), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0),
        ]

        def f(x: float, y: float, t: float) -> float:
            return 1.0

        def g(x: float, y: float, t: float) -> float:
            return 1.0

        def desired(x: float, y: float, t: float) -> float:
            return 1.0

        actual = lib.solve_2d_discrete_observations_discrete_modelling(
            cond_xy0s_list, cond_xytGammas_list, cond_f0s_list, cond_fGammas_list,
            model_xytInftys_list, model_xy0s_list, model_xytGammas_list, f, g)

        xyts_list = [
            (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (1.0, 0.0, 0.5),
            (0.0, 0.0, 1.0), (0.5, 0.0, 1.0), (1.0, 0.0, 1.0),

            (0.0, 0.5, 0.0), (0.5, 0.5, 0.0), (1.0, 0.5, 0.0),
            (0.0, 0.5, 0.5), (0.5, 0.5, 0.5), (1.0, 0.5, 0.5),
            (0.0, 0.5, 1.0), (0.5, 0.5, 1.0), (1.0, 0.5, 1.0),

            (0.0, 1.0, 0.0), (0.5, 1.0, 0.0), (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.5), (0.5, 1.0, 0.5), (1.0, 1.0, 0.5),
            (0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0),
        ]

        for x_i, y_i, t_i in xyts_list:
            np.testing.assert_almost_equal(desired(x_i, y_i, t_i), actual(x_i, y_i, t_i))


class SolveDiscreteObservationsContinuousModellingTester(AbstractTester):
    @staticmethod
    def test_1d_simple():
        cond_x0s_list = [0.0, 0.5, 1.0]

        cond_xtGammas_list = [
            (0.0, 0.5), (1.0, 0.5),
            (0.0, 1.0), (1.0, 1.0),
        ]

        cond_f0s_list = [1.0, 1.0, 1.0]

        cond_fGammas_list = [
            1.0, 1.0,
            1.0, 1.0,
        ]

        a, b, T = 0.0, 1.0, 1.0

        def f(x: float, t: float) -> float:
            return 1.0

        def g(x: float, t: float) -> float:
            return 1.0

        def desired(x: float, t: float) -> float: 
            return 1.0

        actual = lib.solve_1d_discrete_observations_continuous_modelling(
            cond_x0s_list, cond_xtGammas_list, cond_f0s_list, cond_fGammas_list, a, b, T, f, g)

        xts_list = [
            (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
            (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
            (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),
        ]

        for x_i, t_i in xts_list:
            np.testing.assert_almost_equal(desired(x_i, t_i), actual(x_i, t_i))

    # this test is slow because it involves 52 triple integrations and 135 double integrations
    @staticmethod
    def test_2d_simple():
        cond_xy0s_list = [
            (0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
            (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
            (1.0, 0.0), (1.0, 0.5), (1.0, 1.0),
        ]

        cond_xytGammas_list = [
            (0.0, 0.0, 0.5), (0.0, 0.5, 0.5), (0.0, 1.0, 0.5),
            (0.5, 0.0, 0.5), (0.5, 1.0, 0.5),
            (1.0, 0.0, 0.5), (1.0, 0.5, 0.5), (1.0, 1.0, 0.5),

            (0.0, 0.0, 1.0), (0.0, 0.5, 1.0), (0.0, 1.0, 1.0),
            (0.5, 0.0, 1.0), (0.5, 1.0, 1.0),
            (1.0, 0.0, 1.0), (1.0, 0.5, 1.0), (1.0, 1.0, 1.0),
        ]

        cond_f0s_list = [1.0 for _ in range(9)]

        cond_fGammas_list = [1.0 for _ in range(16)]

        a, b, c, d, T = 0.0, 1.0, 0.0, 1.0, 1.0

        def f(x: float, y: float, t: float) -> float:
            return 1.0

        def g(x: float, y: float, t: float) -> float:
            return 1.0

        def desired(x: float, y: float, t: float) -> float: 
            return 1.0

        actual = lib.solve_2d_discrete_observations_continuous_modelling(
            cond_xy0s_list, cond_xytGammas_list, cond_f0s_list, cond_fGammas_list, a, b, c, d, T, f, g)

        xyts_list = [
            (0.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.5), (0.0, 0.5, 0.5), (0.0, 1.0, 0.5),
            (0.0, 0.0, 1.0), (0.0, 0.5, 1.0), (0.0, 1.0, 1.0),

            (0.5, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 1.0, 0.0),
            (0.5, 0.0, 0.5), (0.5, 0.5, 0.5), (0.5, 1.0, 0.5),
            (0.5, 0.0, 1.0), (0.5, 0.5, 1.0), (0.5, 1.0, 1.0),
 
            (1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 1.0, 0.0),
            (1.0, 0.0, 0.5), (1.0, 0.5, 0.5), (1.0, 1.0, 0.5),
            (1.0, 0.0, 1.0), (1.0, 0.5, 1.0), (1.0, 1.0, 1.0),
        ]

        for x_i, y_i, t_i in xyts_list:
            np.testing.assert_almost_equal(desired(x_i, y_i, t_i), actual(x_i, y_i, t_i))


def timed_wrapper(to_wrap):
    @functools.wraps(to_wrap)
    def wrapped(*args, **kwargs):
        start_time = datetime.datetime.now()
        to_wrap(*args, **kwargs)
        time_taken = datetime.datetime.now() - start_time
        print(f"Tests took {time_taken.total_seconds():0.3f} seconds.")

    return wrapped


@timed_wrapper
def main_test():
    for name in dir(sys.modules[__name__]):
        if "Tester" in name and name != "AbstractTester":
            getattr(sys.modules[__name__], name)()


if __name__ == "__main__":
    main_test()
