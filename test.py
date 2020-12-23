# -*- encoding:utf-8 -*-


def test1():
    import numpy as np
    from simulation import SimCountry
    china = SimCountry((5, 5), 100)
    initials = np.zeros((5, 5, 4))
    initials[:, :, 0] = 1000.
    initials[2, 2, 1] = 1e-4
    time_span = [0., 360.]

    china.evolute(initials, time_span)


test1()
