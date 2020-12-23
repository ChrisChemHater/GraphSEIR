# -*- encoding:utf-8 -*-
import numpy as np
from defaults import SEIRDefaultParameter, const_transfer, zipf_transfer, const_parameter, zero_disturbance


class City(object):
    def __init__(self, r, beta, h, theta, gamma, nu, pos):
        self.r = r
        self.beta = beta
        self.h = h
        self.theta = theta
        self.gamma = gamma
        self.nu = nu
        self.pos = pos
        self.inPaths = []
        self.outPaths = []


class Traffic(object):
    def __init__(self, start: "City", end: "City", distance, transfer):
        self.start = start
        self.end = end
        self.distance = distance
        self.transfer = transfer
        self.flux = np.zeros(4, dtype=float)


class Country(object):
    def __init__(self, shape: "tuple(int, int)", min_distance: "float"):
        self.minDistance = min_distance
        self.shape = shape

        # 节点初始化
        r = const_parameter(SEIRDefaultParameter.r)
        beta = const_parameter(SEIRDefaultParameter.beta)
        h = const_parameter(SEIRDefaultParameter.h)
        theta = const_parameter(SEIRDefaultParameter.theta)
        gamma = const_parameter(SEIRDefaultParameter.gamma)
        nu = zero_disturbance
        # transfer = const_transfer()
        transfer = zipf_transfer()
        self.cities = [[City(r, beta, h, theta, gamma, nu, (i, j)) for j in range(shape[1])] for i in range(shape[0])]

        # 构造边
        self.traffic = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                city = self.cities[i][j]
                for I in range(shape[0]):
                    for J in range(shape[1]):
                        if i == I and j == J:
                            continue
                        d = min_distance * ((i - I) ** 2 + (j - J) ** 2) ** 0.5
                        t = Traffic(city, self.cities[I][J], d, transfer)
                        self.traffic.append(t)
                        city.outPaths.append(t)
                        self.cities[I][J].inPaths.append(t)
