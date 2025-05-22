import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.build_net()
        
    def build_net(self):
        """
        Инициализирует слои нейронной сети для комбинаторного аукциона 2x2
        """
        num_agents = self.config.num_agents
        num_items = self.config.num_items
        
        assert num_agents == 2, "Эта сеть поддерживает только 2 агентов"
        assert num_items == 3, "Эта сеть поддерживает CA с 3 значениями (item1, item2, bundle)"
        
        num_a_layers = self.config.net.num_a_layers
        num_p_layers = self.config.net.num_p_layers
        
        num_a_hidden_units = self.config.net.num_a_hidden_units
        num_p_hidden_units = self.config.net.num_p_hidden_units
        
        # Выбор функции активации
        if self.config.net.activation == "tanh":
            self.activation = torch.tanh
        elif self.config.net.activation == "sigmoid":
            self.activation = torch.sigmoid
        elif self.config.net.activation == "relu":
            self.activation = torch.relu
        else:
            raise ValueError(f"Неизвестная функция активации: {self.config.net.activation}")
            
        # Выбор инициализации весов
        if self.config.net.init == "gu":
            initializer = nn.init.xavier_uniform_
        elif self.config.net.init == "gn":
            initializer = nn.init.xavier_normal_
        elif self.config.net.init == "hu":
            initializer = nn.init.kaiming_uniform_
        elif self.config.net.init == "hn":
            initializer = nn.init.kaiming_normal_
        else:
            raise ValueError(f"Неизвестная инициализация: {self.config.net.init}")
            
        # Слои сети распределения (allocation)
        self.alloc_layers = nn.ModuleList()
        num_in = 6  # Для CA 2x2 (3 значения для каждого из 2 агентов)
        
        # Входной слой
        self.alloc_layers.append(nn.Linear(num_in, num_a_hidden_units))
        initializer(self.alloc_layers[-1].weight)
        nn.init.zeros_(self.alloc_layers[-1].bias)
        
        # Скрытые слои
        for i in range(1, num_a_layers-1):
            self.alloc_layers.append(nn.Linear(num_a_hidden_units, num_a_hidden_units))
            initializer(self.alloc_layers[-1].weight)
            nn.init.zeros_(self.alloc_layers[-1].bias)
            
        # Выходные слои для распределения
        # Выходы для item1 (5 выходов: agent1, agent2, bundle1, bundle2, nobody)
        self.wi1_a = nn.Linear(num_a_hidden_units, 5)
        initializer(self.wi1_a.weight)
        nn.init.zeros_(self.wi1_a.bias)
        
        # Выходы для item2 (5 выходов: agent1, agent2, bundle1, bundle2, nobody)
        self.wi2_a = nn.Linear(num_a_hidden_units, 5)
        initializer(self.wi2_a.weight)
        nn.init.zeros_(self.wi2_a.bias)
        
        # Выходы для agent1 bundles (3 выхода: item1, item2, bundle)
        self.wa1_a = nn.Linear(num_a_hidden_units, 3)
        initializer(self.wa1_a.weight)
        nn.init.zeros_(self.wa1_a.bias)
        
        # Выходы для agent2 bundles (3 выхода: item1, item2, bundle)
        self.wa2_a = nn.Linear(num_a_hidden_units, 3)
        initializer(self.wa2_a.weight)
        nn.init.zeros_(self.wa2_a.bias)
        
        # Слои сети платежей (payment)
        self.payment_layers = nn.ModuleList()
        
        # Входной слой
        self.payment_layers.append(nn.Linear(num_in, num_p_hidden_units))
        initializer(self.payment_layers[-1].weight)
        nn.init.zeros_(self.payment_layers[-1].bias)
        
        # Скрытые слои
        for i in range(1, num_p_layers-1):
            self.payment_layers.append(nn.Linear(num_p_hidden_units, num_p_hidden_units))
            initializer(self.payment_layers[-1].weight)
            nn.init.zeros_(self.payment_layers[-1].bias)
            
        # Выходной слой для платежей
        self.payment_out = nn.Linear(num_p_hidden_units, num_agents)
        initializer(self.payment_out.weight)
        nn.init.zeros_(self.payment_out.bias)
    
    def forward(self, x):
        """
        Прямой проход через сеть
        """
        batch_size = x.shape[0]
        num_agents = self.config.num_agents
        
        # Формируем вход для сети - CA 2x2 имеет 6 входов (2 агента, по 3 значения)
        x_in = x.view(batch_size, 6)
        
        # Прямой проход через скрытые слои распределения (allocation)
        a = x_in
        for layer in self.alloc_layers:
            a = layer(a)
            a = self.activation(a)
        
        # Выходы для итемов (точно как в TensorFlow версии)
        a_item1_ = F.softmax(self.wi1_a(a), dim=1)
        a_item2_ = F.softmax(self.wi2_a(a), dim=1)
        
        # Выходы для бандлов агентов
        a_agent1_bundle = F.softmax(self.wa1_a(a), dim=1)
        a_agent2_bundle = F.softmax(self.wa2_a(a), dim=1)
        
        # Собираем распределения для агентов (точно как в оригинальном TensorFlow коде)
        # a_agent1_ = [item1_agent1, item2_agent1, bundle_min]
        a_agent1_ = torch.cat([
            a_item1_[:, 0:1],  # agent1 для item1
            a_item2_[:, 0:1],  # agent1 для item2  
            torch.min(a_item1_[:, 2:3], a_item2_[:, 2:3])  # минимум между bundle1 для обоих items
        ], dim=1)
        
        # a_agent2_ = [item1_agent2, item2_agent2, bundle_min]
        a_agent2_ = torch.cat([
            a_item1_[:, 1:2],  # agent2 для item1
            a_item2_[:, 1:2],  # agent2 для item2
            torch.min(a_item1_[:, 3:4], a_item2_[:, 3:4])  # минимум между bundle2 для обоих items
        ], dim=1)
        
        # Финальное распределение - минимум агентских распределений и бандлов
        a_agent1 = torch.min(a_agent1_, a_agent1_bundle)
        a_agent2 = torch.min(a_agent2_, a_agent2_bundle)
        
        # Объединяем распределения обоих агентов
        alloc = torch.stack([a_agent1, a_agent2], dim=1)
        
        # Прямой проход через сеть платежей (payment)
        p = x_in
        for layer in self.payment_layers:
            p = layer(p)
            p = self.activation(p)
        
        # Выходной слой платежей с сигмоидом
        pay = self.payment_out(p)
        pay = torch.sigmoid(pay)
        
        # Умножаем платежи на полезность
        # CA 2x2 имеет специфическую структуру полезности (item1, item2, bundle)
        u = torch.sum(alloc * x.view(batch_size, 2, 3), dim=2)
        pay = pay * u
        
        return alloc, pay 