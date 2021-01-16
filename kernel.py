# -*- encoding:utf-8 -*-
"""
图模型的核心组件 City, Traffic 和 Country
City: Node, 包含人口、SEIR函数参数、入城和出城交通信息
Traffic: directed Edge, 连接两个City对象，包含距离和转移函数信息
Country: directed Graph, City 和 Traffic 对象的集合，以格点地图的方式将 City 对象组织起来
"""
import numpy as np
from defaults import SEIRDefaultParameter, const_transfer, zipf_transfer, const_parameter, zero_disturbance


class City(object):
    """
    Node
    r -> functional, 日均人际接触概率，一个人在一天中接触别人的数量；
    beta -> functional, 显性接触感染概率，每次和感染者接触而转化为潜伏者的概率；
    h -> functional, 隐性接触感染概率，每次和潜伏者接触而转化为潜伏者的概率；
    theta -> functional, 日均潜伏者发病概率，一个潜伏者在一天中发病而转化为感染者的概率；
    gamma -> functional, 日均感染者痊愈概率，一个感染者在一天中痊愈而转化为康复者的概率；
    nu -> functional, 扰动
    pos -> tuple(int, int), 城市在格点地图上的坐标
    inPaths -> list(Traffic), 进城交通
    outPaths -> list(Traffic), 出城交通
    -----------------------------------------------------
    Example:

        r = const_parameter(SEIRDefaultParameter.r)
        beta = const_parameter(SEIRDefaultParameter.beta)
        h = const_parameter(SEIRDefaultParameter.h)
        theta = const_parameter(SEIRDefaultParameter.theta)
        gamma = const_parameter(SEIRDefaultParameter.gamma)
        nu = zero_disturbance
        city = City(r, beta, h, theta, gamma, nu)
    """
    def __init__(self, r, beta, h, theta, gamma, nu, pos=(0, 0)):
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
    """
    Edge
    start -> City, 出发城市
    end -> City, 到达城市
    distance -> float, 城市间距(km)
    tranfer -> functional, 转移函数，用于计算flux
    flux -> numpy.ndarray, shape=(4,), 当前SEIR流量
    -----------------------------------------------
    Example:

        transfer = zipf_transfer()
        traffic = Traffic(city1, city2, 1000., transfer)
    """
    def __init__(self, start: "City", end: "City", distance, transfer):
        self.start = start
        self.end = end
        self.distance = distance
        self.transfer = transfer
        self.flux = np.zeros(4, dtype=float)


class Country(object):
    """
    Graph
    minDistance -> float, 格点间最小距离，晶胞单位(km)
    shape -> tuple(int, int), 城市矩阵的维度
    -------------------------------------------------
    Example:

        country = Country((3, 4), 1000.)
        country.cities[2][2].r = const_parameter(1.)  # 修改特定城市SEIR参数
        for traffic in country.citie[2][2].outPaths:
            traffic.transfer = const_transfer(0., 0., 0., 0.)  # 封禁出城道路
    """
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
