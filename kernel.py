# -*- encoding:utf-8 -*-
import numpy as np
import json, zipfile
from defaults import SEIRDefaultParameter, const_transfer, zipf_transfer, const_parameter


class City(object):
    def __init__(self, n_sample: "int", r, beta, h, theta, gamma, nu):
        self.track = np.zeros(n_sample)
        self.r = r
        self.beta = beta
        self.h = h
        self.theta = theta
        self.gamma = gamma
        self.nu = nu
        self.inPaths = []
        self.outPaths = []


class Traffic(object):
    def __init__(self, start: "City", end: "City", distance, transfer):
        self.start = start
        self.end = end
        self.distance = distance
        self.transfer = transfer


class Country(object):
    def __init__(self, initials: "np.ndarray, shape=(H, W, 4)", min_distance: "float", time_span: "list[float, float]",
                 step: "float" = 0.1, sampling: "int" = 1):
        self.minDistance = min_distance
        self.time_span = time_span
        self.steps = int((time_span[1] - time_span[0]) / step) + 1
        self.sampling = sampling
        n_sample = (self.steps - 1) // sampling + 1  # 采样点数量
        self.time = time_span[0] + step * np.arange(n_sample)

        # 节点初始化
        r = const_parameter(SEIRDefaultParameter.r)
        beta = const_parameter(SEIRDefaultParameter.beta)
        h = const_parameter(SEIRDefaultParameter.h)
        theta = const_parameter(SEIRDefaultParameter.theta)
        gamma = const_parameter(SEIRDefaultParameter.gamma)
        nu = lambda t: 0.
        # transfer = const_transfer(0., 0., 0., 0.)
        transfer = zipf_transfer(K=150., alpha=1.)
        self.cities = [[City(n_sample, r, beta, h, theta, gamma, nu) for _ in range(initials.shape[1])] for _ in
                       range(initials.shape[0])]
        for i in range(initials.shape[0]):
            for j in range(initials.shape[1]):
                self.cities[i][j].track[0] = initials[i, j, :]

        # 构造边
        for i in range(initials.shape[0]):
            for j in range(initials.shape[1]):
                city = self.cities[i][j]
                for I in range(initials.shape[0]):
                    for J in range(initials.shape[1]):
                        if i == I and j == J:
                            continue
                        d = min_distance * ((i - J) ** 2 + (j - J) ** 2) ** 0.5
                        t = Traffic(city, self.cities[I][J], d, transfer)
                        city.outPaths.append(t)
                        self.cities[I][J].inPaths.append(t)

    def evolute(self, method:"'RK4' or 'Euler'"="RK4"):
        pass
    
    def _euler(self):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass