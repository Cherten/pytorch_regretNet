import numpy as np
from base.base_generator import BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, config, mode, X=None, ADV=None):
        super(Generator, self).__init__(config, mode)
        self.build_generator(X=X, ADV=ADV)

    def generate_random_X(self, shape):
        """Generate random values uniformly from [0, 1]"""
        return np.random.rand(*shape)

    def generate_random_ADV(self, shape):
        """Generate random initial misreports uniformly from [0, 1]"""
        return np.random.rand(*shape) 