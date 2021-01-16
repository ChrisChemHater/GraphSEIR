# -*- encoding:utf-8 -*-
"""
模拟结果的可视化
plot_country: 绘制国家总SEIR-时间演化曲线
plot_all: 绘制各城市SEIR-时间演化曲线
animate: 制作国家演化视频
report: 生成csv格式的报告表，包含国家和各城市演化过程中的感染最大值及时刻、健康人数降至50%时刻信息
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulation import SimCountry


def _plot(ax, time: "np.ndarray, shape=(N,)", itrack: "np.ndarray, shape=(N, 4)"):
    ax.plot(time, itrack[:, 0], lw=0.6, label='S')
    ax.plot(time, itrack[:, 1], lw=0.6, label='E')
    ax.plot(time, itrack[:, 2], lw=0.6, label='I')
    ax.plot(time, itrack[:, 3], lw=0.6, label='R')
    ax.legend()
    ax.set_xlabel("time / day")
    ax.set_ylabel("population / $10^{4}$")


def plot_country(country: "SimCountry") -> "fig, ax":
    fig, ax = plt.subplots(dpi=170)
    _plot(ax, country.time, country.track.sum(axis=(1, 2)))
    return fig, ax


def plot_all(country: "SimCountry", directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig, ax = plt.subplots(dpi=170)
    _plot(ax, country.time, country.track[:, 0, 0, :])
    for i in range(country.shape[0]):
        for j in range(country.shape[1]):
            ax.clear()
            _plot(ax, country.time, country.track[:, i, j, :])
            fig.savefig(f"{directory}/city{i}{j}.svg")


def animate(country: "SimCountry", interval=20, gap=10):
    r, c = country.shape
    I, J = np.mgrid[:r, :c]
    sizes = country.track[:, :, :, 2].reshape(country.track.shape[0], -1) / 490 * 5120

    fig, ax = plt.subplots(figsize=(5, 5))
    pc = ax.scatter(I.flatten(), J.flatten(), c='red', s=sizes[0], alpha=0.5)
    title = ax.set_title(rf"t = ${country.time[0]:.2f}$ days")

    def Animate(i):
        idx = int(i * gap)
        pc.set_sizes(sizes[idx])
        title.set_text(rf"t = ${country.time[idx]:.2f}$ days")
        return pc, title

    ani = animation.FuncAnimation(
        fig, Animate, frames=country.time.shape[0] // gap, interval=interval, blit=False, save_count=50)

    return ani


def report(country: "SimCountry", filename):
    index = ['country'] + [f'city_{i}{j}' for j in range(country.shape[1]) for i in range(country.shape[0])]
    df = pd.DataFrame(index=index, columns=['I_max_time', 'I_max', 'S_50_time'])

    # 处理全国的情况
    I = country.track[:, :, :, 2].sum(axis=(1, 2))
    Im_arg = I.argmax()
    Im_t = country.time[Im_arg]
    Im = I[Im_arg]
    S = country.track[:, :, :, 0].sum(axis=(1, 2))
    S_frac = S / S[0]
    S50_t = country.time[(S_frac > 0.5).sum() - 1]
    df.loc['country', :] = [Im_t, Im, S50_t]

    # 处理各城市
    I = country.track[:, :, :, 2].reshape(country.time.size, -1)
    Im_arg = I.argmax(axis=0)
    Im_t = country.time[Im_arg]
    Im = I[Im_arg, np.arange(I.shape[1])]
    S = country.track[:, :, :, 0].reshape(country.time.size, -1)
    S_frac = S / S[0, :].reshape(1, S.shape[1])
    S50_t = country.time[(S_frac > 0.5).sum(axis=0) - 1]
    df.iloc[1:, 0] = Im_t
    df.iloc[1:, 1] = Im
    df.iloc[1:, 2] = S50_t

    df.to_csv(filename)
