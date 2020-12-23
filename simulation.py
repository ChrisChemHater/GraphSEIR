# -*- encoding:utf-8 -*-
import numpy as np
from kernel import Country
from numerical import Euler, RK4


class SimCountry(Country):
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
