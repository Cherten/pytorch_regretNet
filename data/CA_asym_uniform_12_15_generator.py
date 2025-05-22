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
        X = np.zeros(shape)
        
        # Agent 1: uniform(1,2) for all three valuations
        X[:, 0, 0] = np.random.uniform(1.0, 2.0, size=shape[0])  # item1
        X[:, 0, 1] = np.random.uniform(1.0, 2.0, size=shape[0])  # item2
        X[:, 0, 2] = np.random.uniform(1.0, 2.0, size=shape[0])  # bundle
        
        # Agent 2: uniform(1,5) for all three valuations
        X[:, 1, 0] = np.random.uniform(1.0, 5.0, size=shape[0])  # item1
        X[:, 1, 1] = np.random.uniform(1.0, 5.0, size=shape[0])  # item2
        X[:, 1, 2] = np.random.uniform(1.0, 5.0, size=shape[0])  # bundle
        
        return X

    def generate_random_ADV(self, shape):
        """Generate random adversarial valuations
        Shape: [num_misreports, num_instances, num_agents, 3]
        """
        X = np.zeros(shape)
        
        # Agent 1: uniform(1,2) for all three valuations
        X[:, :, 0, 0] = np.random.uniform(1.0, 2.0, size=(shape[0], shape[1]))  # item1
        X[:, :, 0, 1] = np.random.uniform(1.0, 2.0, size=(shape[0], shape[1]))  # item2
        X[:, :, 0, 2] = np.random.uniform(1.0, 2.0, size=(shape[0], shape[1]))  # bundle
        
        # Agent 2: uniform(1,5) for all three valuations
        X[:, :, 1, 0] = np.random.uniform(1.0, 5.0, size=(shape[0], shape[1]))  # item1
        X[:, :, 1, 1] = np.random.uniform(1.0, 5.0, size=(shape[0], shape[1]))  # item2
        X[:, :, 1, 2] = np.random.uniform(1.0, 5.0, size=(shape[0], shape[1]))  # bundle
        
        return X
    
    def generate_random_C(self, shape):
        return np.random.uniform(-1.0, 1.0, size=shape) 