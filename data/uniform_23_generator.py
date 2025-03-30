import numpy as np
from base.base_generator import BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, config, mode, X=None, ADV=None):
        super().__init__(config, mode)
        self.build_generator(X=X, ADV=ADV)

    def generate_random_X(self, shape):
        """Генерирует случайные значения в диапазоне [2, 3]"""
        return np.random.rand(*shape) + 2.0

    def generate_random_ADV(self, shape):
        """Генерирует случайные значения в диапазоне [2, 3]"""
        return np.random.rand(*shape) + 2.0
        
    def generate_next_batch(self):
        """Генерирует следующий батч данных"""
        # Определяем размеры тензоров
        batch_size = self.config[self.mode].batch_size
        x_shape = (batch_size, self.config.num_agents, self.config.num_items)
        adv_shape = (batch_size, self.config.num_agents, self.config.num_items)
        
        # Генерируем случайные данные
        x = self.generate_random_X(x_shape)
        adv = self.generate_random_ADV(adv_shape)
        
        return x, adv, None 