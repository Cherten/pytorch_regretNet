import numpy as np
from base.base_generator import BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, config, mode, X=None, ADV=None):
        super().__init__(config, mode)
        self.build_generator(X=X, ADV=ADV)
        assert self.config.num_items == 2

    def generate_random_X(self, shape):
        r = np.random.rand(*shape)
        x = np.zeros(shape)
        x[:, :, 0] = np.sqrt(r[:, :, 0]) * (1 - r[:, :, 1])
        x[:, :, 1] = np.sqrt(r[:, :, 0]) * (r[:, :, 1])
        return x

    def generate_random_ADV(self, shape):
        # Проверяем размерность входного массива и адаптируем генерацию соответственно
        if len(shape) == 3:  # [batch_size, num_agents, num_items]
            r = np.random.rand(*shape)
            x = np.zeros(shape)
            x[:, :, 0] = np.sqrt(r[:, :, 0]) * (1 - r[:, :, 1])
            x[:, :, 1] = np.sqrt(r[:, :, 0]) * (r[:, :, 1])
            return x
        elif len(shape) == 4:  # [num_misreports, batch_size, num_agents, num_items]
            r = np.random.rand(*shape)
            x = np.zeros(shape)
            x[:, :, :, 0] = np.sqrt(r[:, :, :, 0]) * (1 - r[:, :, :, 1])
            x[:, :, :, 1] = np.sqrt(r[:, :, :, 0]) * (r[:, :, :, 1])
            return x
        else:
            raise ValueError(f"Unexpected shape for ADV generation: {shape}, expected 3D or 4D array")