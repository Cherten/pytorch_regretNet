import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BaseGenerator:
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode
        self.num_agents = config.num_agents
        self.num_items = config.num_items
        self.num_instances = config[self.mode].num_batches * config[self.mode].batch_size
        self.num_misreports = config[self.mode].num_misreports
        self.batch_size = config[self.mode].batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def build_generator(self, X=None, ADV=None):
        if self.mode == "train":
            if self.config.train.data == "fixed":
                if self.config.train.restore_iter == 0:
                    self.get_data(X, ADV)
                else:
                    self.load_data_from_file(self.config.train.restore_iter)
                self.dataset = FixedDataset(self.X, self.ADV)
                self.dataloader = DataLoader(
                    self.dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True, 
                    num_workers=0,
                    drop_last=False
                )
            else:
                self.dataloader = self.gen_online()
        else:
            if self.config[self.mode].data == "fixed" or X is not None:
                self.get_data(X, ADV)
                self.dataset = FixedDataset(self.X, self.ADV)
                self.dataloader = DataLoader(
                    self.dataset, 
                    batch_size=self.batch_size, 
                    shuffle=False, 
                    num_workers=0,
                    drop_last=False
                )
            else:
                self.dataloader = self.gen_online()
        
        # Создаем gen_func для совместимости с TensorFlow версией
        self._dataloader_iter = iter(self.dataloader)
        self.gen_func = self._gen_func_wrapper()
        
    def get_data(self, X=None, ADV=None):
        """Generates data"""
        x_shape = [self.num_instances, self.num_agents, self.num_items]
        adv_shape = [self.num_misreports, self.num_instances, self.num_agents, self.num_items]
        
        if X is None:
            X = self.generate_random_X(x_shape)
        if ADV is None:
            ADV = self.generate_random_ADV(adv_shape)
        
        self.X = X
        self.ADV = ADV
    
    def load_data_from_file(self, iter):
        """Loads data from disk"""
        self.X = np.load(os.path.join(self.config.dir_name, 'X.npy'))
        self.ADV = np.load(os.path.join(self.config.dir_name, f'ADV_{iter}.npy'))
        
    def save_data(self, iter):
        """Saves data to disk"""
        if self.config.save_data is False:
            return
        
        if iter == 0:
            np.save(os.path.join(self.config.dir_name, 'X'), self.X)
        else:
            np.save(os.path.join(self.config.dir_name, f'ADV_{iter}'), self.ADV)
    
    def gen_online(self):
        """Generate data online (on-the-fly)"""
        class OnlineDataset(Dataset):
            def __init__(self, generator, batch_size, num_batches):
                self.generator = generator
                self.batch_size = batch_size
                self.num_batches = num_batches
                
            def __len__(self):
                return self.num_batches
                
            def __getitem__(self, idx):
                x_batch_shape = [self.batch_size, self.generator.num_agents, self.generator.num_items]
                adv_batch_shape = [self.generator.num_misreports, self.batch_size, 
                                  self.generator.num_agents, self.generator.num_items]
                
                X = self.generator.generate_random_X(x_batch_shape)
                ADV = self.generator.generate_random_ADV(adv_batch_shape)
                
                # Convert to PyTorch tensors
                X_tensor = torch.tensor(X, dtype=torch.float32)
                ADV_tensor = torch.tensor(ADV, dtype=torch.float32)
                
                # ИСПРАВЛЕНИЕ: Возвращаем фиктивный индекс вместо None для совместимости с DataLoader
                dummy_idx = torch.tensor(-1)  # Фиктивный индекс для онлайн режима
                
                return X_tensor, ADV_tensor, dummy_idx
        
        dataset = OnlineDataset(self, self.batch_size, self.config[self.mode].num_batches)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    def update_adv(self, idx, adv_new):
        """Updates ADV for caching"""
        # ADV имеет форму [num_misreports, num_instances, num_agents, num_items]
        # adv_new имеет форму [batch_size, num_agents, num_items]
        
        # Проверяем, что idx это тензор или список
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif not isinstance(idx, (list, np.ndarray)):
            idx = [idx]
        
        # Конвертируем в numpy и проверяем размерности
        adv_numpy = adv_new.cpu().numpy()
        
        # Если num_misreports = 1, специальная обработка
        if self.num_misreports == 1:
            # Для каждого элемента в батче
            for i, index in enumerate(idx):
                # Проверяем границы
                if i < len(adv_numpy) and index < self.X.shape[0]:
                    self.ADV[0, index, :, :] = adv_numpy[i, :, :]
        else:
            # Если у нас есть несколько misreports, размерность adv_new должна быть [num_misreports, batch_size, num_agents, num_items]
            if len(adv_numpy.shape) == 4:
                for i, index in enumerate(idx):
                    if i < adv_numpy.shape[1] and index < self.X.shape[0]:
                        for m in range(min(self.num_misreports, adv_numpy.shape[0])):
                            self.ADV[m, index, :, :] = adv_numpy[m, i, :, :]
    
    def generate_random_X(self, shape):
        """Rewrite this for new distributions"""
        raise NotImplementedError
    
    def generate_random_ADV(self, shape):
        """Rewrite this for new distributions"""
        raise NotImplementedError
    
    def _gen_func_wrapper(self):
        """Обертка для совместимости с TensorFlow интерфейсом"""
        def gen_func():
            while True:
                try:
                    X_batch, ADV_batch, idx = next(self._dataloader_iter)
                    # Убираем лишний batch dimension от DataLoader
                    if X_batch.dim() == 4:  # [1, batch_size, agents, items] -> [batch_size, agents, items]
                        X_batch = X_batch.squeeze(0)
                    if ADV_batch.dim() == 5:  # [1, misreports, batch_size, agents, items] -> [misreports, batch_size, agents, items]
                        ADV_batch = ADV_batch.squeeze(0)
                    
                    # ИСПРАВЛЕНИЕ: Правильная обработка индексов для батчей
                    if idx is not None:
                        # Убираем лишние размерности
                        if idx.dim() > 1:
                            idx = idx.squeeze()
                        # Проверяем, является ли это фиктивными индексами (-1)
                        # Если все элементы равны -1, то это онлайн режим
                        if idx.numel() > 0 and torch.all(idx == -1):
                            idx = None
                    
                    return X_batch, ADV_batch, idx
                except StopIteration:
                    # Перезапускаем итератор
                    self._dataloader_iter = iter(self.dataloader)
                    X_batch, ADV_batch, idx = next(self._dataloader_iter)
                    if X_batch.dim() == 4:
                        X_batch = X_batch.squeeze(0)
                    if ADV_batch.dim() == 5:
                        ADV_batch = ADV_batch.squeeze(0)
                    
                    # ИСПРАВЛЕНИЕ: Правильная обработка индексов (дублируем логику)
                    if idx is not None:
                        if idx.dim() > 1:
                            idx = idx.squeeze()
                        if idx.numel() > 0 and torch.all(idx == -1):
                            idx = None
                    
                    return X_batch, ADV_batch, idx
        return gen_func


class FixedDataset(Dataset):
    def __init__(self, X, ADV):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.ADV = torch.tensor(ADV, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.ADV[:, idx, :, :], idx 