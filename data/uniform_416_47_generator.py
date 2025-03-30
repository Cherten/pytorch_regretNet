import numpy as np
from base.base_generator import BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, config, mode, X=None, ADV=None):
        super().__init__(config, mode)
        self.build_generator(X=X, ADV=ADV)

    def generate_random_X(self, shape):
        X = np.zeros(shape)
        size = (shape[0], shape[1])
        X[:, :, 0] = np.random.uniform(4.0, 16.0, size=size)
        X[:, :, 1] = np.random.uniform(4.0, 7.0, size=size)
        return X

    def generate_random_ADV(self, shape):
        X = np.zeros(shape)
        
        # Проверяем размерность входа
        if len(shape) == 3:  # [batch_size, num_agents, num_items]
            size = (shape[0], shape[1])
            X[:, :, 0] = np.random.uniform(4.0, 16.0, size=size)
            X[:, :, 1] = np.random.uniform(4.0, 7.0, size=size)
        elif len(shape) == 4:  # [num_misreports, batch_size, num_agents, num_items]
            size = (shape[0], shape[1], shape[2])
            X[:, :, :, 0] = np.random.uniform(4.0, 16.0, size=size)
            X[:, :, :, 1] = np.random.uniform(4.0, 7.0, size=size)
        else:
            raise ValueError(f"Unexpected shape for ADV generation: {shape}, expected 3D or 4D array")
            
        return X