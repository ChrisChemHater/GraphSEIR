# -*- encoding:utf-8 -*-
import os


def test1():
    import numpy as np
    from simulation import SimCountry
    from vision import plot_all, plot_country, animate, report
    if not os.path.exists("test"):
        os.mkdir("test")
    china = SimCountry((5, 5), 100)  # 创建 5×5 的格点国家，城市间距 100 km
    initials = np.zeros((5, 5, 4))  # 生成初始状态
    initials[:, :, 0] = 1000.  # 各城市初始人口为健康的1000万人
    initials[2, 2, 1] += 1e-4  # 向中心 City_22 投放 1 位潜伏者
    time_span = [0., 120.]  # 模拟时间范围为 0 ~ 120 天

    print("Simulating...")
    china.evolute(initials, time_span)
    print("Simulation done, plotting...")
    china.save('test/test')
    # china.load('test/test.npz')

    fig, ax = plot_country(china)  # 绘制国家演化曲线
    fig.savefig("test/country.svg")
    plot_all(china, "test")  # 各城市演化曲线
    print("plotting done, making animation...")
    ani = animate(china)  # 国家演化视频
    ani.save("test/test.mp4")
    print("done")

    report(china, 'test/report.csv')  # 演化报告表


test1()
