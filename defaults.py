# -*- encoding:utf-8 -*-
import numpy as np

MIN_DISTANCE = 100  # 最小城市间距，单位:km


class SEIRDefaultParameter(object):
    r = 100.
    beta = 0.048
    h = 0.048
    theta = 1. / 6.
    gamma = 1. / 21.


def const_transfer(S=0., E=0., I=0., R=0.):
    def transfer(start_status: "np.ndarray, shape=(4,)", end_status: "np.ndarray, shape=(4,)",
                 t: "float", distance: "float") -> "np.ndarray, shape=(4,)":
        """
        交通函数，接口示例函数，返回从start城市到end城市SEIR迁移人口
        """
        return np.array([S, E, I, R])

    return transfer


def zipf_transfer(K=4e-4, alpha=1.):
    def transfer(start_status: "np.ndarray, shape=(4,)", end_status: "np.ndarray, shape=(4,)",
                 t: "float", distance: "float") -> "np.ndarray, shape=(4,)":
        Ns = start_status.sum()
        Ne = end_status.sum()
        Nt = K * Ns * Ne / distance ** alpha
        return Nt / Ns * start_status

    return transfer


def const_parameter(p):
    def parameter(status: "np.ndarray, shape=(4,)", t: "float") -> "float":
        return p

    return parameter


def zero_disturbance(t: "float") -> "np.ndarray, shape=(4,)":
    return np.zeros(4, dtype=float)
