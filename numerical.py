# -*- encoding:utf-8 -*-
import numpy as np
from kernel import Country


def _derivative(country: "Country", status: "np.ndarray, shape=(H, W, 4)", t: "float"):
    der = np.zeros_like(status)
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
