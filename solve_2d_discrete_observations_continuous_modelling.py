# discrete observations continuous modelling functions dimensionality 2
def solve_2d_discrete_observations_continuous_modelling(
        cond_xy0s_list: tp.List[Tp.Tuple[float, float]],
        cond_xytGammas_list: tp.List[tp.Tuple[float, float, float]],
        cond_f0s_list: tp.List[float],
        cond_fGammas_list: tp.List[float],
        a: float,
        b: float,
        c: float,
        d: float,
        T: float,
        f: tp.Callable[[float, float], float],
        g: tp.Callable[[float, float], float],
) -> tp.Callable[[float, float], float]:
    """
    :param cond_xy0s_list: list of space points for initial conditions:
        u(cond_x0_i, cond_y0_i, 0) = cond_f0_i
    :param cond_xytGammas_list: list of space-time for boundary conditions:
        u(cond_xGamma_i, cond_yGamma_i, cond_tGamma_i) = cond_fGamma_i
    :param cond_f0s_list: list of real values for initial conditions:
        cond_f0_i = u(cond_x0_i, cond_y0_i, 0)
    :param cond_fGammas_list: list of real values for boundary conditions:
        cond_fGamma_i = u(cond_xGamma_i, cond_yGamma_i, cond_tGamma_i)
    :param a: lower bound of the x-domains of g and u
    :param b: upper bound of the x-domains of g and u
    :param c: lower bound of the y-domains of g and u
    :param d: upper bound of the y-domains of g and u
    :param T: end time
    :param f: real-valued function of space and time,
        represents external perturbations in the system.
    :param g: Green's function of the linear differential operator L
    :return: real-valued function u of space and time,
        least squares solution to L u(x, y, t) = f(x, y, t)
        under initial conditions u(cond_x0_i, cond_y0_i, 0) = cond_f0_i,
        and boundary conditions u(cond_xGamma_i, cond_yGamma_i, cond_tGamma_i) = cond_fGamma_i.
    """

    def u_infty(x: float, y: float, t: float) -> float:
        return integrate.tplquad(lambda t_other, x_other, y_other:
            g(x - x_other, y - y_other, t - t_other) * f(x_other, y_other, t_other),
        c, d, a, b, 0, T)[0]

    vec_u0 = np.array([[
        cond_f0_i - u_infty(cond_xy0_i[0], cond_xy0_i[1], 0.0)
    ] for cond_f0_i, cond_xy0_i in zip(cond_f0s_list, cond_xy0s_list)])

    vec_uGamma = np.array([[
        cond_fGamma_i - u_infty(cond_xytGamma_i[0], cond_xytGamma_i[1], cond_xytGamma_i[2])
    ] for cond_fGamma_i, cond_xytGamma_i in zip(cond_fGammas_list, cond_xytGammas_list)])

    vec_u = np.vstack((vec_u0, vec_uGamma))

    def A11(x: float, y: float) -> np.array:
        return np.array([[g(
            cond_x0_i - x,
            cond_y0_i - y,
            0.0 - 0.0,
        )] for cond_x0_i, cond_y0_i in cond_xy0s_list])

    def A12(x: float, y: float, t: float) -> np.array:
        return np.array([[g(
            cond_x0_i - x,
            cond_y0_i - y,
            0.0 - t,
        )] for cond_x0_i, cond_y0_i in cond_xy0s_list])

    def A21(x: float, y: float) -> np.array:
        return np.array([[g(
            cond_xGamma_i - x,
            cond_yGamma_i - y,
            cond_tGamma_i - 0.0,
        )] for cond_xGamma_i, cond_yGamma_i, cond_tGamma_i in cond_xytGammas_list])

    def A22(x: float, y: float, t: float) -> np.array:
        return np.array([[g(
            cond_xGamma_i - x,
            cond_yGamma_i - y,
            cond_tGamma_i - t,
        )] for cond_xGamma_i, cond_yGamma_i, cond_tGamma_i in cond_xytGammas_list])

    def A(x: float, y: float, t: float) -> np.matrix:
        return np.vstack((
            np.hstack((A11(x, y), A12(x, y, t))),
            np.hstack((A21(x, y), A22(x, y, t))),
        ))

    len0, lenGamma = len(cond_x0s_list), len(cond_xtGammas_list)

    P11 = np.matrix([[
        integrate.dblquad(lambda x, y: A11(x, y)[i] * A11(x, y)[j], c, d, a, b)[0] +
        integrate.dblquad(lambda t, y: A12(a, y, t)[i] * A12(a, y, t)[j], c, d, 0, T)[0] +
        integrate.dblquad(lambda t, y: A12(b, y, t)[i] * A12(b, y, t)[j], c, d, 0, T)[0] +
        integrate.dblquad(lambda t, x: A12(x, c, t)[i] * A12(x, c, t)[j], a, b, 0, T)[0] +
        integrate.dblquad(lambda t, x: A12(x, d, t)[i] * A12(x, d, t)[j], a, b, 0, T)[0]
    for j in range(len0)] for i in range(len0)])

    P12 = np.matrix([[
        integrate.dblquad(lambda x, y: A11(x, y)[i] * A21(x, y)[j], c, d, a, b)[0] +
        integrate.dblquad(lambda t, y: A12(a, y, t)[i] * A22(a, y, t)[j], c, d, 0, T)[0] +
        integrate.dblquad(lambda t, y: A12(b, y, t)[i] * A22(b, y, t)[j], c, d, 0, T)[0] +
        integrate.dblquad(lambda t, x: A12(x, c, t)[i] * A22(x, c, t)[j], a, b, 0, T)[0] +
        integrate.dblquad(lambda t, x: A12(x, d, t)[i] * A22(x, d, t)[j], a, b, 0, T)[0]
    for j in range(lenGamma)] for i in range(len0)])

    P21 = np.matrix([[
        integrate.dblquad(lambda x, y: A21(x, y)[i] * A11(x, y)[j], c, d, a, b)[0] +
        integrate.dblquad(lambda t, y: A22(a, y, t)[i] * A12(a, y, t)[j], c, d, 0, T)[0] +
        integrate.dblquad(lambda t, y: A22(b, y, t)[i] * A12(b, y, t)[j], c, d, 0, T)[0] +
        integrate.dblquad(lambda t, x: A22(x, c, t)[i] * A12(x, c, t)[j], a, b, 0, T)[0] +
        integrate.dblquad(lambda t, x: A22(x, d, t)[i] * A12(x, d, t)[j], a, b, 0, T)[0] +
    for j in range(len0)] for i in range(lenGamma)])

    P22 = np.matrix([[
        integrate.dblquad(lambda x, y: A21(x, y)[i] * A21(x, y)[j], c, d, a, b)[0] +
        integrate.dblquad(lambda t, y: A22(a, y, t)[i] * A22(a, y, t)[j], c, d, 0, T)[0] +
        integrate.dblquad(lambda t, y: A22(b, y, t)[i] * A22(b, y, t)[j], c, d, 0, T)[0] +
        integrate.dblquad(lambda t, x: A22(x, c, t)[i] * A22(x, c, t)[j], a, b, 0, T)[0] +
        integrate.dblquad(lambda t, x: A22(x, d, t)[i] * A22(x, d, t)[j], a, b, 0, T)[0] +
    for j in range(lenGamma)] for i in range(lenGamma)])

    P = np.vstack((
        np.hstack((P11, P12)),
        np.hstack((P21, P22)),
    ))

    def vec_f(x: float, y: float, t: float) -> np.array:
        return A(x, y, t).T * np.linalg.pinv(P) * vec_u

    def vec_f0(x: float, y: float, t: float) -> float:
        return vec_f(x, y, t)[0]

    def vec_fGamma(x: float, y: float, t: float) -> float:
        return vec_f(x, y, t)[1]

    def u_0(x: float, y: float, t: float) -> float:
        return integrate.dblquad(lambda x_other, y_other:
            g(x - x_other, y - y_other, t - 0.0) * vec_f0(x_other, y_other, 0.0),
        c, d, a, b)[0]

    def u_Gamma(x: float, y: float, t: float) -> float:
        return integrate.dblquad(lambda t_other, y_other:
            g(x - a, y - y_other, t - t_other) * vec_fGamma(a, y_other, t_other),
        c, d, 0, T)[0] + integrate.dblquad(lambda t_other, y_other:
            g(x - b, y - y_other, t - t_other) * vec_fGamma(b, y_other, t_other),
        c, d, 0, T)[0] + integrate.dblquad(lambda t_other, x_other:
            g(x - x_other, y - c, t - t_other) * vec_fGamma(x_other, c, t_other),
        a, b, 0, T)[0] + integrate.dblquad(lambda t_other, y_other:
            g(x - x_other, y - d, t - t_other) * vec_fGamma(x_other, d, t_other),
        a, b, 0, T)[0]

    def u(x: float, y: float, t: float) -> float:
        return u_infty(x, y, t) + u_0(x, y, t) + u_Gamma(x, y, t)

    return u
