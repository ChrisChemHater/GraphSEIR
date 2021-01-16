# -*- encoding:utf-8 -*-
"""
国家疫情演化的数值算法。扩展的SEIR模型核心为4变量一阶微分方程组，求解微分方程的常用算法包括Euler方法和RK方法。
此处给出了Euler方法和RK4方法的实现，Euler方法可用于非连续情况下的模拟(以天为单位进行演化)，RK4方法则提供了连续情况下
精确且计算开销较小的解决方案。
"""
import numpy as np
from kernel import Country


def _derivative(country: "Country", status: "np.ndarray, shape=(H, W, 4)", t: "float"):
    der = np.zeros_like(status)
    """
    扩展SEIR方程，计算某时刻下SEIR四个变量对时间的导数。
    status: 前两个维度为城市矩阵的形状，即 country.shape == status.shape[:2]，第三个维度为4，
        即 status[a][b] = [S, E, I, R] 为 City_ab 的SEIR状态
    t: 时间

    Return:
        np.ndarray, shape=(4,), 分别为SEIR在t时刻下对时间的导数。
    """
    # 外部输入输出准备
    for tfc in country.traffic:
        i, j = tfc.start.pos
        I, J = tfc.end.pos
        tfc.flux = tfc.transfer(status[i, j, :], status[I, J, :], t, tfc.distance)

    for i in range(status.shape[0]):
        for j in range(status.shape[1]):
            c = country.cities[i][j]

            # 1. SEIR方程本项
            cstatus = status[i, j, :]
            r = c.r(cstatus, t)
            beta = c.beta(cstatus, t)
            h = c.h(cstatus, t)
            theta = c.theta(cstatus, t)
            gamma = c.gamma(cstatus, t)
            N = status[i, j, :].sum()
            se = r * (beta * cstatus[2] + h * cstatus[1]) * cstatus[0] / N
            ei = theta * cstatus[1]
            ir = gamma * cstatus[2]
            der[i, j, :] = [-se, se - ei, ei - ir, ir]

            # 2. 外部输入输出项
            der[i, j, :] += np.array([tfc.flux for tfc in c.inPaths]).sum(axis=0)
            der[i, j, :] -= np.array([tfc.flux for tfc in c.outPaths]).sum(axis=0)

            # 3. 扰动项
            der[i, j, :] += c.nu(t)

    return der


def Euler(country: "Country", initials: "np.ndarray, shape=(H, W, 4)", time_span: "list[float, float]",
          step: "float" = 0.1, sampling: "int" = 1) -> "(time, track)":
    """
    Euler方法。
    initials: 初始状态。前两个维度为城市矩阵的形状，即 country.shape == initials.shape[:2]，第三个维度为4，
        即 initials[a][b] = [S, E, I, R] 为 City_ab 的初始SEIR状态
    time_span: 演化时间范围
    step: 演化时间步长
    sampling: 采样间隔，即每sampling个step记录一次系统状态。

    Return:
        time: np.ndarray, shape=(N,), 时间序列
        track: np.ndarray, shape=(N, H, W, 4), 演化轨迹
    """
    steps = int((time_span[1] - time_span[0]) / step) + 1
    n_sample = (steps - 1) // sampling + 1
    ts = time_span[0] + step * np.arange(steps)  # 计算时间序列
    time = time_span[0] + step * np.arange(n_sample)  # 采样时间序列
    track = np.zeros((n_sample, *initials.shape))

    x = initials.copy()
    for idx in range(steps):
        if idx % sampling == 0:
            track[idx // sampling] = x

        x += _derivative(country, x, ts[idx]) * step

    return time, track


def RK4(country: "Country", initials: "np.ndarray, shape=(H, W, 4)", time_span: "list[float, float]",
        step: "float" = 0.1, sampling: "int" = 1) -> "(time, track)":
    """
    RK4方法。
    initials: 初始状态。前两个维度为城市矩阵的形状，即 country.shape == initials.shape[:2]，第三个维度为4，
        即 initials[a][b] = [S, E, I, R] 为 City_ab 的初始SEIR状态
    time_span: 演化时间范围
    step: 演化时间步长
    sampling: 采样间隔，即每sampling个step记录一次系统状态。

    Return:
        time: np.ndarray, shape=(N,), 时间序列
        track: np.ndarray, shape=(N, H, W, 4), 演化轨迹
    """
    steps = int((time_span[1] - time_span[0]) / step) + 1
    n_sample = (steps - 1) // sampling + 1
    ts = time_span[0] + step * np.arange(steps)  # 计算时间序列
    time = time_span[0] + step * np.arange(n_sample)  # 采样时间序列
    track = np.zeros((n_sample, *initials.shape))

    x = initials.copy()
    for idx in range(steps):
        if idx % sampling == 0:
            track[idx // sampling] = x
            # print(f"city22 = {x[2,2,:]}, day = {idx * step:.2f}")

        t = ts[idx]
        k1 = _derivative(country, x, t)
        k = k1
        k1 = _derivative(country, x + k1 * step / 2, t + step / 2)
        k += k1 * 2
        k1 = _derivative(country, x + k1 * step / 2, t + step / 2)
        k += k1 * 2
        k1 = _derivative(country, x + k1 * step, t + step)
        k += k1
        x += k * step / 6.

    return time, track
