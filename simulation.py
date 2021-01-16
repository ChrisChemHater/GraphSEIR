# -*- encoding:utf-8 -*-
"""
定义SimCountry, kernel.Country仅用于描述图结构，SimCountry增加了演化方法和演化轨迹。
"""
import numpy as np
from kernel import Country
from numerical import Euler, RK4


class SimCountry(Country):
    """
    初始化方法同kernel.Country

    methods:
        self.evolute(self, initials: "np.ndarray, shape=(H, W, 4)", time_span: "list[float, float]",
                     step: "float" = 0.1, sampling: "int" = 1, method: "'RK4' or 'Euler'" = "RK4")
            initials: 初始状态。前两个维度为城市矩阵的形状，即 country.shape == initials.shape[:2]，第三个维度为4，
                即 initials[a][b] = [S, E, I, R] 为 City_ab 的初始SEIR状态
            time_span: 演化时间范围
            step: 演化时间步长
            sampling: 采样间隔，即每sampling个step记录一次系统状态。
            method: RK4 or Euler
        self.save(self, filename)
            仅保存模型的演化轨迹，不保存图信息
            filename: path, 不需要后缀，默认以numpy提供的.npz文件格式保存模型的演化轨迹
        self.load(self, filename)
            加载已保存的演化轨迹
    ----------------------------------------------------------------------------
    Example:
        请见test.py
    """
    def __init__(self, shape: "tuple(int, int)", min_distance: "float"):
        super().__init__(shape, min_distance)

    def evolute(self, initials: "np.ndarray, shape=(H, W, 4)", time_span: "list[float, float]",
                step: "float" = 0.1, sampling: "int" = 1, method: "'RK4' or 'Euler'" = "RK4"):
        if method == "RK4":
            self.time, self.track = RK4(self, initials, time_span, step, sampling)
        elif method == "Euler":
            self.time, self.track = Euler(self, initials, time_span, step, sampling)
        else:
            raise KeyError("method must be one of 'RK4' and 'Euler'")

    def save(self, filename):
        try:
            np.savez(filename, time=self.time, track=self.track)
        except AttributeError:
            raise RuntimeError("Simulation must run before saving the results")

    def load(self, filename):
        npzfile = np.load(filename)
        self.time, self.track = npzfile['time'], npzfile['track']
