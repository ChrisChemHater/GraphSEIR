# -*- encoding:utf-8 -*-
"""
默认参数和参数函数设定，包含：
SEIR参数：r, beta, h, theta, gamma
用于初始化kernel.City对象的常函数：const_parameter(), zero_disturbance
用于初始化kernel.Traffic对象的常函数和zipf函数：const_transfer(), zipf_transfer()

用户可依据这些模板自定义参数函数
"""
import numpy as np

MIN_DISTANCE = 100  # 最小城市间距，单位:km


class SEIRDefaultParameter(object):
    r = 20.
    beta = 0.048
    h = 0.048
    theta = 0.1
    gamma = 0.1


def const_transfer(S=0., E=0., I=0., R=0.):
    """
    用于初始化 Traffic().transfer
    常函数，返回从start城市到end城市SEIR迁移人口
    """
    def transfer(start_status: "np.ndarray, shape=(4,)", end_status: "np.ndarray, shape=(4,)",
                 t: "float", distance: "float") -> "np.ndarray, shape=(4,)":
        return np.array([S, E, I, R])

    return transfer


def zipf_transfer(K=0.04, alpha=2.):
    """
    用于初始化 Traffic().transfer
    zipf人口迁移函数，返回从start城市到end城市SEIR迁移人口
    """
    def transfer(start_status: "np.ndarray, shape=(4,)", end_status: "np.ndarray, shape=(4,)",
                 t: "float", distance: "float") -> "np.ndarray, shape=(4,)":
        Ns = start_status.sum()
        Ne = end_status.sum()
        Nt = K * Ns * Ne / distance ** alpha
        return Nt / Ns * start_status

    return transfer


def const_parameter(p):
    """
    用于初始化 City().r/beta/h/theta/gamma
    返回某时刻某SEIR状态下的参数值，此处为常数
    """
    def parameter(status: "np.ndarray, shape=(4,)", t: "float") -> "float":
        return p

    return parameter


def zero_disturbance(t: "float") -> "np.ndarray, shape=(4,)":
    """
    用于初始化 City().nu
    返回某时刻下对SEIR的外部扰动
    """
    return np.zeros(4, dtype=float)
