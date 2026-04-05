from dataclasses import dataclass

import numpy as np


class UniformPrior:
    def __init__(self, a, b):
        if a >= b:
            raise ValueError("uniform prior requires a < b")
        self.a = a
        self.b = b

    def low(self):
        return self.a

    def high(self):
        return self.b


@dataclass
class VarConfig:
    name: str
    prior: UniformPrior

    def low(self):
        return self.prior.low()

    def high(self):
        return self.prior.high()


def build_problem(var_configs):
    return {
        "num_vars": len(var_configs),
        "names": [v.name for v in var_configs],
        "bounds": [[v.low(), v.high()] for v in var_configs],
    }
