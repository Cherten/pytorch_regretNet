import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def create_var(name, shape, initializer=None, wd=None, requires_grad=True):
    """
    Вспомогательная функция для создания переменной (тензора) с заданными параметрами
    
    Аргументы:
        name: имя переменной (для удобства отладки)
        shape: размерность тензора
        initializer: инициализатор (функция из torch.nn.init)
        wd: параметр регуляризации весов (weight decay)
        requires_grad: требуется ли градиент для этой переменной
    
    Возвращает:
        torch.Tensor: созданная переменная
    """
    var = torch.zeros(shape, requires_grad=requires_grad)
    
    # Применяем инициализатор, если он указан
    if initializer:
        initializer(var)
    
    return var


class BaseNet(nn.Module):
    """
    Базовый класс для нейронных сетей в RegretNet.
    
    Предоставляет общий функционал для всех типов сетей, включая:
    - Настройку инициализатора весов
    - Настройку функции активации
    - Расчет регуляризации
    
    Все конкретные реализации сетей должны наследоваться от этого класса
    и реализовывать методы build_net() и forward().
    """
    
    def __init__(self, config):
        """
        Инициализация базового класса сети
        
        Аргументы:
            config: объект конфигурации с настройками сети
        """
        super(BaseNet, self).__init__()
        self.config = config
        
        # Настройка инициализатора весов в соответствии с конфигурацией
        if self.config.net.init == 'None':
            self.init = None  # Без инициализации (стандартная инициализация PyTorch)
        elif self.config.net.init == 'gu':
            self.init = lambda x: init.xavier_uniform_(x)  # Равномерная инициализация Xavier/Glorot
        elif self.config.net.init == 'gn':
            self.init = lambda x: init.xavier_normal_(x)   # Нормальная инициализация Xavier/Glorot
        elif self.config.net.init == 'hu':
            self.init = lambda x: init.kaiming_uniform_(x) # Равномерная инициализация He/Kaiming
        elif self.config.net.init == 'hn':
            self.init = lambda x: init.kaiming_normal_(x)  # Нормальная инициализация He/Kaiming
        
        # Настройка функции активации в соответствии с конфигурацией
        if self.config.net.activation == 'tanh':
            self.activation = torch.tanh  # Гиперболический тангенс
        elif self.config.net.activation == 'relu':
            self.activation = F.relu      # ReLU (Rectified Linear Unit)
    
    def build_net(self):
        """
        Метод для инициализации параметров и архитектуры сети.
        
        Должен быть реализован в дочерних классах.
        """
        raise NotImplementedError
        
    def forward(self, x):
        """
        Прямой проход через сеть (эквивалент inference в TensorFlow).
        
        Должен быть реализован в дочерних классах.
        
        Аргументы:
            x: входные данные
            
        Возвращает:
            Выходные данные после прохода через сеть
        """
        raise NotImplementedError
    
    def get_reg_loss(self, model_parameters, wd):
        """
        Расчет потери регуляризации L2 (weight decay)
        
        Аргументы:
            model_parameters: параметры модели, к которым применяется регуляризация
            wd: коэффициент регуляризации (weight decay)
            
        Возвращает:
            float: значение потери регуляризации
        """
        if wd is None:
            return 0.0
            
        reg_loss = 0.0
        for param in model_parameters:
            reg_loss += torch.sum(param ** 2)
            
        return reg_loss * wd / 2.0 