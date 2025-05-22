import numpy as np
from base.base_generator_ca import BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, config, mode, X=None, ADV=None):
        super().__init__(config, mode)
        self.build_generator(X=X, ADV=ADV)

    def generate_random_X(self, shape):
        """Generate random valuations for CA auction
        Shape: [num_instances, num_agents, 3] where 3 is [item1, item2, bundle]
        """
        return np.random.uniform(1.0, 2.0, size=shape)

    def generate_random_ADV(self, shape):
        """Generate random adversarial valuations
        Shape: [num_misreports, num_instances, num_agents, 3]
        """
        return np.random.uniform(1.0, 2.0, size=shape)
    
    def generate_random_C(self, shape):
        return np.random.uniform(-1.0, 1.0, size=shape) 