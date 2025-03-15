import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

class Net(BaseNet):
    def __init__(self, config):
        super(Net, self).__init__(config)
        self.build_net()
        
    def build_net(self):
        """
        Инициализирует параметры нейронной сети RegretNet.
        
        RegretNet состоит из двух основных компонентов:
        1. Сеть распределения (allocation network) - определяет вероятность выделения предметов агентам
        2. Сеть платежей (payment network) - определяет, сколько каждый агент должен заплатить
        """
        num_agents = self.config.num_agents  # Количество участников аукциона
        num_items = self.config.num_items    # Количество предметов для продажи
        
        # Параметры архитектуры из конфигурации
        num_a_layers = self.config.net.num_a_layers            # Количество слоев в сети распределения
        num_p_layers = self.config.net.num_p_layers            # Количество слоев в сети платежей
        
        num_a_hidden_units = self.config.net.num_a_hidden_units  # Количество нейронов в скрытых слоях сети распределения
        num_p_hidden_units = self.config.net.num_p_hidden_units  # Количество нейронов в скрытых слоях сети платежей
        
        # Инициализация слоев
        # СЕТЬ РАСПРЕДЕЛЕНИЯ (allocation network)
        self.alloc_layers = nn.ModuleList()
        
        # Входной слой - принимает ставки всех агентов за все предметы
        num_in = num_agents * num_items
        self.alloc_layers.append(nn.Linear(num_in, num_a_hidden_units))
        
        # Скрытые слои
        for i in range(1, num_a_layers - 1):
            self.alloc_layers.append(nn.Linear(num_a_hidden_units, num_a_hidden_units))
            
        # Выходной слой - размерность включает фиктивного агента и фиктивный предмет
        # Фиктивный агент представляет случай, когда предмет не распределяется никому
        # Фиктивный предмет представляет случай, когда агент не получает ни одного предмета
        self.alloc_layers.append(nn.Linear(num_a_hidden_units, (num_agents + 1) * (num_items + 1)))
        
        # СЕТЬ ПЛАТЕЖЕЙ (payment network)
        self.pay_layers = nn.ModuleList()
        
        # Входной слой - те же входные данные, что и для сети распределения
        self.pay_layers.append(nn.Linear(num_in, num_p_hidden_units))
        
        # Скрытые слои
        for i in range(1, num_p_layers - 1):
            self.pay_layers.append(nn.Linear(num_p_hidden_units, num_p_hidden_units))
            
        # Выходной слой - платеж для каждого из num_agents агентов
        self.pay_layers.append(nn.Linear(num_p_hidden_units, num_agents))
        
        # Инициализация весов выбранным в конфигурации способом
        for layer in self.alloc_layers:
            if self.init is not None:
                self.init(layer.weight)
                nn.init.zeros_(layer.bias)  # Смещения инициализируются нулями
                
        for layer in self.pay_layers:
            if self.init is not None:
                self.init(layer.weight)
                nn.init.zeros_(layer.bias)  # Смещения инициализируются нулями
    
    def forward(self, x):
        """
        Прямой проход через сеть (предсказание)
        
        Параметры:
            x: Входные данные [batch_size, num_agents, num_items] - ставки агентов
        
        Возвращает:
            a: Распределение предметов [batch_size, num_agents, num_items] - 
               вероятности получения предметов агентами
            p: Платежи [batch_size, num_agents] - 
               сколько каждый агент должен заплатить
        """
        batch_size = x.size(0)
        num_agents = self.config.num_agents
        num_items = self.config.num_items
        
        # Преобразуем входные данные в плоский вектор для обработки нейронной сетью
        x_in = x.reshape(batch_size, num_agents * num_items)
        
        # ---- СЕТЬ РАСПРЕДЕЛЕНИЯ ----
        a = x_in
        # Проходим через все слои, кроме последнего, с активацией
        for i in range(len(self.alloc_layers) - 1):
            a = self.alloc_layers[i](a)
            a = self.activation(a)
            
        # Выходной слой (без активации на данном этапе)
        a = self.alloc_layers[-1](a)
        
        # Преобразуем выход в формат [batch_size, num_agents+1, num_items+1]
        # +1 для фиктивного агента и фиктивного предмета
        a = a.reshape(batch_size, num_agents + 1, num_items + 1)
        
        # Применяем softmax по измерению агентов (dim=1)
        # Это гарантирует, что вероятности распределения каждого предмета в сумме дают 1
        a = F.softmax(a, dim=1)
        
        # Отбрасываем фиктивного агента и фиктивный предмет, оставляя только реальные распределения
        a = a[:, :num_agents, :num_items]
        
        # ---- СЕТЬ ПЛАТЕЖЕЙ ----
        p = x_in
        # Проходим через все слои, кроме последнего, с активацией
        for i in range(len(self.pay_layers) - 1):
            p = self.pay_layers[i](p)
            p = self.activation(p)
        
        # Выходной слой с сигмоидной активацией
        # Сигмоида гарантирует, что выход находится в диапазоне [0,1]
        p = self.pay_layers[-1](p)
        p = torch.sigmoid(p)
        
        # Расчет полезности для каждого агента: сумма ценностей полученных предметов
        # u[i,j] = сумма(a[i,j,k] * x[i,j,k]) для всех предметов k
        u = torch.sum(a * x, dim=2)
        
        # Масштабирование платежей согласно полезности
        # Это гарантирует, что агент никогда не платит больше, чем его ценность (IR constraint)
        p = p * u
        
        return a, p 