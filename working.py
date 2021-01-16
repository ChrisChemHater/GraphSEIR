# -*- encoding:utf-8 -*-

# 模拟参数请见simulation.md 或 研究报告 supplement/复杂SEIR模型及其在分析新冠疫情封城政策中的应用.pdf
import numpy as np
import os
from multiprocessing import Pool
from simulation import SimCountry
from defaults import SEIRDefaultParameter, zipf_transfer, MIN_DISTANCE
from vision import plot_all, plot_country, animate, report
import matplotlib.pyplot as plt

lock_times = [5., 10., 15., 10000.]  # 最后一次不封锁
R_LOCK = 3.


def lockdown_parameter(p1, p2, time):
    """
    分段函数，封锁时间之前，参数为p1, 封锁之后为p2
    """
    def parameter(status: "np.ndarray, shape=(4,)", t: "float") -> "float":
        return p2 if t >= time else p1

    return parameter


def lockdown_transfer(time):
    """
    分段函数，封锁之前为zipf函数，封锁之后SEIR传输为0
    """
    def transfer(start_status: "np.ndarray, shape=(4,)", end_status: "np.ndarray, shape=(4,)",
                 t: "float", distance: "float") -> "np.ndarray, shape=(4,)":
        zipf = zipf_transfer()
        return zipf(start_status, end_status, t, distance) if t < time else np.zeros_like(start_status)

    return transfer


def step(idx, lock_time):
    # 单次模拟函数，用于多进程模拟。
    china = SimCountry((5, 5), MIN_DISTANCE)
    initials = np.zeros((5, 5, 4))
    initials[:, :, 0] = 1000.
    initials[2, 2, 1] = 1e-4
    time_span = [0., 360.]

    print(f"----------------- lock_time = {lock_time} --------------------")
    if not os.path.exists(f"reports/sim_{idx}"):
        os.makedirs(f"reports/sim_{idx}")
    if not os.path.exists("results"):
        os.mkdir("results")
    print("model initializing...")
    for i in range(china.shape[0]):
        for j in range(china.shape[1]):
            china.cities[i][j].r = lockdown_parameter(SEIRDefaultParameter.r, R_LOCK, lock_time)
    for traffic in china.traffic:
        traffic.transfer = lockdown_transfer(lock_time)
    print("done")
    print("Simulating...")
    china.evolute(initials, time_span)
    print("Simulation done, plotting...")
    china.save(f'results/sim_{idx}')
    # china.load(f'results/sim_{idx}.npz')

    fig, ax = plot_country(china)
    fig.savefig(f"reports/sim_{idx}/country.svg")
    plot_all(china, f"reports/sim_{idx}")
    print("plotting done, making animation...")
    ani = animate(china)
    ani.save(f"reports/sim_{idx}/animation.mp4")
    print("done")

    report(china, f"reports/sim_{idx}/report.csv")
    plt.close(fig)


def single_thread():
    # 未使用多进程技术的模拟。并未采用
    china = SimCountry((5, 5), MIN_DISTANCE)
    initials = np.zeros((5, 5, 4))
    initials[:, :, 0] = 1000.
    initials[2, 2, 1] = 1e-4
    time_span = [0., 360.]

    for idx, lock_time in zip(range(len(lock_times)), lock_times):single_thread
        print(f"----------------- lock_time = {lock_time} --------------------")
        if not os.path.exists(f"reports/sim_{idx}"):
            os.makedirs(f"reports/sim_{idx}")
        if not os.path.exists("results"):
            os.mkdir("results")
        print("model initializing...")
        for i in range(china.shape[0]):
            for j in range(china.shape[1]):
                china.cities[i][j].r = lockdown_parameter(SEIRDefaultParameter.r, R_LOCK, lock_time)
        for traffic in china.traffic:
            traffic.transfer = lockdown_transfer(lock_time)
        print("done")
        print("Simulating...")
        china.evolute(initials, time_span)
        print("Simulation done, plotting...")
        china.save(f'results/sim_{idx}')
        # china.load(f'results/sim_{idx}.npz')

        fig, ax = plot_country(china)
        fig.savefig(f"reports/sim_{idx}/country.svg")
        plot_all(china, f"reports/sim_{idx}")
        print("plotting done, making animation...")
        ani = animate(china)
        ani.save(f"reports/sim_{idx}/animation.mp4")
        print("done")

        report(china, f"reports/sim_{idx}/report.csv")
        plt.close(fig)


if __name__ == '__main__':
    pool = Pool()
    for idx, lock_time in zip(range(len(lock_times)), lock_times):
        pool.apply_async(step, args=(idx, lock_time))

    pool.close()
    pool.join()
