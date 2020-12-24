# -*- encoding:utf-8 -*-


def test1():
    import numpy as np
    from simulation import SimCountry
    from vision import plot_all, plot_country, animate, report
    china = SimCountry((5, 5), 100)
    initials = np.zeros((5, 5, 4))
    initials[:, :, 0] = 1000.
    initials[2, 2, 1] = 1e-4
    time_span = [0., 120.]

    print("Simulating...")
    china.evolute(initials, time_span)
    print("Simulation done, plotting...")
    china.save('results/test')
    # china.load('results/test.npz')

    fig, ax = plot_country(china)
    fig.savefig("test/country.svg")
    plot_all(china, "test")
    print("plotting done, making animation...")
    ani = animate(china)
    ani.save("test/test.mp4")
    print("done")

    report(china, 'test/report.csv')


test1()
