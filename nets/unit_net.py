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
        Инициализирует слои нейронной сети
        """
        num_agents = self.config.num_agents
        num_items = self.config.num_items
        
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
        num_in = num_agents * num_items
        
        # Входной слой
        self.alloc_layers.append(nn.Linear(num_in, num_a_hidden_units))
        initializer(self.alloc_layers[-1].weight)
        nn.init.zeros_(self.alloc_layers[-1].bias)
        
        # Скрытые слои
        for i in range(1, num_a_layers-1):
            self.alloc_layers.append(nn.Linear(num_a_hidden_units, num_a_hidden_units))
            initializer(self.alloc_layers[-1].weight)
            nn.init.zeros_(self.alloc_layers[-1].bias)
            
        # Выходной слой для распределения
        self.alloc_out = nn.Linear(num_a_hidden_units, (num_agents + 1) * (num_items + 1))
        initializer(self.alloc_out.weight)
        nn.init.zeros_(self.alloc_out.bias)
        
        self.alloc_out_i = nn.Linear(num_a_hidden_units, (num_agents + 1) * (num_items + 1))
        initializer(self.alloc_out_i.weight)
        nn.init.zeros_(self.alloc_out_i.bias)
        
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
        num_items = self.config.num_items
        
        # Формируем вход для сети
        x_in = x.view(batch_size, num_agents * num_items)
        
        # Прямой проход через сеть распределения (allocation)
        a = x_in
        for layer in self.alloc_layers:
            a = layer(a)
            a = self.activation(a)
        
        # Выходной слой распределения с софтмакс
        agent = self.alloc_out(a)
        agent = agent.view(batch_size, num_agents + 1, num_items + 1)
        agent = F.softmax(agent, dim=1)
        
        item = self.alloc_out_i(a)
        item = item.view(batch_size, num_agents + 1, num_items + 1)
        item = F.softmax(item, dim=2)
        
        # Определяем распределение как минимум вероятностей
        alloc = torch.min(agent, item)
        alloc = alloc[:, :num_agents, :num_items]
        
        # Прямой проход через сеть платежей (payment)
        p = x_in
        for layer in self.payment_layers:
            p = layer(p)
            p = self.activation(p)
        
        # Выходной слой платежей с сигмоидом
        pay = self.payment_out(p)
        pay = torch.sigmoid(pay)
        
        # Умножаем платежи на полезность
        u = torch.sum(alloc * x.view(batch_size, num_agents, num_items), dim=2)
        pay = pay * u
        
        return alloc, pay 