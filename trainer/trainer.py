import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import traceback


class Trainer:
    """
    Класс для обучения и тестирования модели RegretNet.
    
    RegretNet использует метод множителей Лагранжа и градиентный подъем для решения задачи 
    максимизации выручки аукциона при соблюдении ограничения индивидуальной рациональности (IR) 
    и стимулирующей совместимости (IC).
    
    Ключевые компоненты:
    1. Оптимизация основной сети для максимизации выручки
    2. Поиск наилучших "misreports" (неправдивых заявок) для оценки сожаления (regret)
    3. Обновление множителей Лагранжа для балансировки между выручкой и ограничениями
    """
    def __init__(self, config, mode, net, clip_op_lambda, callback=None):
        """
        Инициализация тренера RegretNet
        
        Аргументы:
            config: объект конфигурации
            mode: режим работы ('train' или 'test')
            net: экземпляр нейронной сети
            clip_op_lambda: функция для ограничения значений в допустимом диапазоне
            callback: функция обратного вызова для отслеживания прогресса
        """
        self.config = config
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Создание директории для вывода
        if not os.path.exists(self.config.dir_name):
            os.makedirs(self.config.dir_name)
            
        # Настройка логирования
        if self.mode == "train":
            log_suffix = f'_{self.config.train.restore_iter}' if self.config.train.restore_iter > 0 else ''
            self.log_fname = os.path.join(self.config.dir_name, f'train{log_suffix}.txt')
        else:
            log_suffix = f"_iter_{self.config.test.restore_iter}_m_{self.config.test.num_misreports}_gd_{self.config.test.gd_iter}"
            self.log_fname = os.path.join(self.config.dir_name, f"test{log_suffix}.txt")
            
        # Установка seed для воспроизводимости результатов
        np.random.seed(self.config[self.mode].seed)
        torch.manual_seed(self.config[self.mode].seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config[self.mode].seed)
            
        # Логирование информации об устройстве
        print(f"Используется устройство: {self.device}")
        if self.device.type == 'cuda':
            print(f"CUDA устройство: {torch.cuda.get_device_name(0)}")
            print(f"Выделено памяти: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
        # Инициализация логгера
        self.init_logger()
        
        # Инициализация сети
        self.net = net.to(self.device)
        
        # Функция ограничения значений (clip operation)
        self.clip_op_lambda = clip_op_lambda
        
        # Инициализация масок для генерации misreports
        self.init_misreport_masks()
        
        # Инициализация графа обучения или тестирования
        if self.mode == "train":
            self.init_train()
        elif self.mode == "test":
            self.init_test()
        
        self.callback = callback  # Функция обратного вызова
    
    def init_logger(self):
        """Инициализация системы логирования"""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Очистка существующих обработчиков
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Обработчик для вывода в консоль
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Обработчик для вывода в файл
        handler = logging.FileHandler(self.log_fname, 'w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        self.logger = logger
    
    def init_misreport_masks(self):
        """
        Инициализация масок для генерации misreports
        
        Создает маску, которая используется для замены правдивых ценностей одного агента
        на misreports, сохраняя ценности других агентов неизменными.
        """
        # Создание маски размерностью [num_agents, batch_size, num_agents, num_items]
        batch_size = self.config.train.batch_size
        num_agents = self.config.num_agents
        num_items = self.config.num_items
        
        # Инициализируем маску нулями
        adv_mask = torch.zeros(num_agents, batch_size, num_agents, num_items)
        
        # Для каждого агента i, установим маску = 1 только для его ценностей
        # Это позволит заменять только значения i-го агента при генерации misreports
        for i in range(num_agents):
            adv_mask[i, :, i, :] = 1
            
        self.adv_mask = adv_mask.to(self.device)
    
    def compute_rev(self, pay):
        """
        Расчет выручки аукциона как суммы платежей всех агентов
        
        Аргументы:
            pay: платежи [batch_size, num_agents]
        
        Возвращает:
            Среднюю выручку по батчу (скаляр)
        """
        return torch.mean(torch.sum(pay, dim=1))
    
    def compute_utility(self, x, alloc, pay):
        """
        Расчет полезности для каждого агента
        
        Полезность = ценность полученных предметов - платеж
        
        Аргументы:
            x: ценности [batch_size, num_agents, num_items]
            alloc: распределение [batch_size, num_agents, num_items]
            pay: платежи [batch_size, num_agents]
        
        Возвращает:
            utility: полезность [batch_size, num_agents]
        """
        return torch.sum(alloc * x, dim=2) - pay
    
    def get_misreports(self, x, adv_var):
        """
        Генерация misreports для каждого агента
        
        Создает варианты входных данных, где для каждого агента i его истинные ценности
        заменяются на оптимизированные неправдивые заявки (adv_var), в то время как
        ценности остальных агентов остаются неизменными.
        
        Аргументы:
            x: истинные ценности [batch_size, num_agents, num_items]
            adv_var: оптимизированные неправдивые заявки
                     [batch_size, num_agents, num_items] или
                     [num_misreports, batch_size, num_agents, num_items]
        
        Возвращает:
            x_mis: размноженные исходные данные для вычисления misreports
            misreports: набор входных данных с замененными ценностями агентов
        """
        batch_size = x.size(0)
        num_misreports = adv_var.size(1) if len(adv_var.shape) > 3 else 1
        
        # Если adv_var имеет форму [batch_size, num_agents, num_items], преобразуем в
        # [num_misreports, batch_size, num_agents, num_items] для согласованности
        if len(adv_var.shape) == 3:
            adv_var = adv_var.unsqueeze(0)
        
        # Размножаем x для вычисления misreports
        x_mis = x.repeat(self.config.num_agents * num_misreports, 1, 1)
        
        # Изменяем форму для удобства манипуляции
        x_r = x_mis.reshape(self.config.num_agents, num_misreports, batch_size, 
                           self.config.num_agents, self.config.num_items)
        
        # Создаем misreports путем замены истинных ценностей на adversarial значения
        # используя маску для выбора только релевантных элементов
        adv = adv_var.unsqueeze(0).expand(self.config.num_agents, -1, -1, -1, -1)
        y = x_r * (1 - self.adv_mask) + adv * self.adv_mask
        
        # Возвращаем форму к батч-формату
        misreports = y.reshape(-1, self.config.num_agents, self.config.num_items)
        
        return x_mis, misreports
    
    def init_train(self):
        """
        Инициализация компонентов для обучения
        
        Настраивает оптимизаторы, множители Лагранжа и другие
        компоненты, необходимые для обучения модели.
        """
        # Скорость обучения
        self.learning_rate = self.config.train.learning_rate
        
        # Оптимизатор для параметров основной сети
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        
        # Множители Лагранжа для ограничений на regret
        # Инициализируем значением w_rgt_init_val или 0.0 если не указано
        w_rgt_init_val = 0.0 if "w_rgt_init_val" not in self.config.train else self.config.train.w_rgt_init_val
        # Каждому агенту соответствует свой множитель Лагранжа
        self.w_rgt = torch.ones(self.config.num_agents, device=self.device) * w_rgt_init_val
        
        # Коэффициент обновления для метода штрафов
        self.update_rate = torch.tensor(self.config.train.update_rate, device=self.device)
        
        # Инициализация TensorBoard для визуализации процесса обучения
        self.writer = SummaryWriter(log_dir=self.config.dir_name)
        
        # Метрики для отслеживания
        self.metric_names = ["Revenue", "Regret", "Reg_Loss", "Lag_Loss", "Net_Loss", "w_rgt_mean", "update_rate"]
        
        # Проверяем наличие параметров валидации, если нет - устанавливаем значения по умолчанию
        if not hasattr(self.config, 'val'):
            self.config.val = type('', (), {})()
            
        if not hasattr(self.config.val, 'gd_iter'):
            # По умолчанию 100 итераций для оптимизации misreports при валидации
            self.config.val.gd_iter = 100  
            
        if not hasattr(self.config.val, 'gd_lr'):
            # По умолчанию learning rate 0.1 для оптимизации misreports при валидации
            self.config.val.gd_lr = 0.1  
    
    def init_test(self):
        """
        Инициализация компонентов для тестирования
        """
        # Метрики для отслеживания при тестировании
        self.metric_names = ["Revenue", "Regret", "IRP"]
    
    def train(self, generators):
        """
        Запуск цикла обучения
        
        Аргументы:
            generators: кортеж (train_generator, val_generator) для генерации 
                        обучающих и валидационных данных
        """
        self.train_gen, self.val_gen = generators
        
        # Текущая итерация
        iter = self.config.train.restore_iter
        
        # Создание директории для контрольных точек модели
        if not os.path.exists(self.config.dir_name):
            os.makedirs(self.config.dir_name)
        
        # Загрузка модели, если восстанавливаемся из контрольной точки
        if iter > 0:
            model_path = os.path.join(self.config.dir_name, f'model-{iter}.pt')
            checkpoint = torch.load(model_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.w_rgt = checkpoint['w_rgt']
            self.update_rate = checkpoint['update_rate']
            
        # Сохраняем начальный датасет
        if iter == 0:
            self.train_gen.save_data(0)
            
        # Инициализация множителей Лагранжа на первой итерации (как в TF версии)
        initial_lagrange_update = (iter == 0)
            
        # Цикл обучения - точно как в TensorFlow версии (один batch за итерацию)
        while iter < self.config.train.max_iter:
            start_time = time.time()
            
            # Получаем ОДИН пакет (как в TF версии) 
            X_batch, ADV_batch, idx = self.train_gen.gen_func()
            
            # Убедитесь, что типы тензоров и устройства правильные
            X_batch = X_batch.to(self.device)
            ADV_batch = ADV_batch.to(self.device)
            
            if idx is not None:
                idx = idx.cpu().numpy()
            
            # Обновление множителей Лагранжа на первой итерации (как в TF версии)
            if iter == 0:
                # Временно обучаем сеть для получения начального regret
                alloc, pay = self.net(X_batch)
                utility = self.compute_utility(X_batch, alloc, pay)
                initial_regret = self.compute_regret_unified(X_batch, utility, ADV_batch)
                # Создаем вектор regret для каждого агента
                regret_per_agent = torch.full((self.config.num_agents,), initial_regret.item(), device=self.device)
                self.update_lagrange_multipliers(regret_per_agent.detach())
            
            # Инициализируем misreports
            adv_var = ADV_batch.clone().detach().requires_grad_(True)
            
            # Оптимизируем misreports (найдем лучшие неправдивые заявки)
            adv_var = self.optimize_misreports(X_batch, adv_var, "train")
            
            # Кешируем оптимизированные misreports, если используем фиксированные данные
            if self.config.train.adv_reuse and idx is not None:
                self.train_gen.update_adv(idx, adv_var.detach())
            
            # Обучаем основную сеть
            metrics = self.train_step(X_batch, adv_var)
            
            # Увеличиваем итерацию (как в TF версии - ПОСЛЕ train_step)
            iter += 1
            
            # Обновление множителей Лагранжа (как в TF версии - ПОСЛЕ увеличения iter)
            if iter % self.config.train.update_frequency == 0:
                # Получаем текущий regret для обновления множителей
                alloc, pay = self.net(X_batch)
                utility = self.compute_utility(X_batch, alloc, pay)
                current_regret = self.compute_regret_unified(X_batch, utility, adv_var)
                # Создаем вектор regret для каждого агента
                regret_per_agent = torch.full((self.config.num_agents,), current_regret.item(), device=self.device)
                self.update_lagrange_multipliers(regret_per_agent.detach())
            
            # Увеличиваем коэффициент обновления для метода штрафов
            if iter % self.config.train.up_op_frequency == 0:
                self.update_rate += self.config.train.up_op_add
                self.logger.info(f"Увеличиваем update_rate до {self.update_rate.item()}")
            
            # Логирование
            if iter % self.config.train.print_iter == 0:
                self.log_metrics(iter, metrics, start_time)
                
            # Сохраняем контрольную точку модели
            if iter % self.config.train.save_iter == 0 or iter == self.config.train.max_iter:
                self.save_model(iter)
                
            # Валидация
            if iter % self.config.val.print_iter == 0:
                self.validate(iter)
                    
        # Финальное сохранение
        self.save_model(iter)
        self.writer.close()
        
    def train_step(self, X, adv_var):
        """
        Шаг обучения
        """
        import time
        time_profile = {}
        
        t_start = time.time()
        
        self.net.train()
        self.optimizer.zero_grad()
        
        batch_size = X.size(0)
        num_agents = self.config.num_agents
        num_items = self.config.num_items
        
        # Получаем распределения и платежи для истинных ценностей
        t1 = time.time()
        alloc, pay = self.net(X)
        time_profile['forward_pass'] = time.time() - t1
        
        # Выручка - средний платеж по всем агентам и батчам
        revenue = self.compute_rev(pay)
        
        # Полезность при истинных заявках
        utility = self.compute_utility(X, alloc, pay)
        
        # Вычисляем сожаление (regret) для каждого агента
        t1 = time.time()
        regrets = []
        
        # Создаем X_mis и misreports используя adv_var
        num_misreports = adv_var.size(0) if len(adv_var.shape) == 4 else 1
        
        # Расширяем X для создания misreports
        X_mis = X.repeat(num_agents * num_misreports, 1, 1)
        
        # Создаем маску для замены значений
        adv_mask = torch.zeros(num_agents, num_misreports, batch_size, num_agents, num_items, device=self.device)
        for agent_idx in range(num_agents):
            adv_mask[agent_idx, :, :, agent_idx, :] = 1.0
        
        # Преобразуем X_mis в нужную форму для применения маски
        X_r = X_mis.reshape(num_agents, num_misreports, batch_size, num_agents, num_items)
        
        # Создаем misreports, заменяя значения согласно маске
        adv = adv_var.unsqueeze(0).expand(num_agents, -1, -1, -1, -1)
        misreports_reshaped = X_r * (1 - adv_mask) + adv * adv_mask
        
        # Преобразуем обратно в пакетный формат
        misreports = misreports_reshaped.reshape(-1, num_agents, num_items)
        
        # Получаем распределения и платежи для неправдивых заявок
        t2 = time.time()
        alloc_mis, pay_mis = self.net(misreports)
        time_profile['misreport_forward'] = time.time() - t2
        
        # Вычисляем полезность для неправдивых заявок
        utility_mis = self.compute_utility(X_mis, alloc_mis, pay_mis)
        
        # Создаем маску для выбора только соответствующих значений полезности
        u_mask = torch.zeros(num_agents, num_misreports, batch_size, num_agents, device=self.device)
        for agent_idx in range(num_agents):
            u_mask[agent_idx, :, :, agent_idx] = 1.0
        
        # Преобразуем utility_mis в нужную форму для вычисления regret
        utility_mis_reshaped = utility_mis.reshape(num_agents, num_misreports, batch_size, num_agents)
        
        # Расширяем истинную полезность для сравнения
        utility_true_expanded = utility.repeat(num_agents * num_misreports, 1)
        utility_true_reshaped = utility_true_expanded.reshape(num_agents, num_misreports, batch_size, num_agents)
        
        # Вычисляем excess_utility - разницу между utility_mis и utility_true, применяя маску
        # и беря только положительные значения - точно как в TF версии
        excess_utility = torch.relu((utility_mis_reshaped - utility_true_reshaped) * u_mask)
        
        # Вычисляем regret для каждого агента как максимум по всем неправдивым заявкам и по всем батчам
        for agent_idx in range(num_agents):
            # Максимум по misreports для каждого агента и батча
            max_over_mis = torch.max(excess_utility[agent_idx], dim=0)[0]  # [batch_size, num_agents]
            # Затем берем максимум по batch, а потом среднее - как в TF
            agent_regret = torch.mean(torch.max(max_over_mis, dim=1)[0])  # скаляр
            regrets.append(agent_regret)
        
        regrets = torch.stack(regrets)
        time_profile['regret_computation'] = time.time() - t1
        
        # Compute Individual Rationality Penalty (IRP)
        irp = torch.mean(torch.relu(-utility))
        
        # Lagrangian loss terms - точно как в TF версии
        rgt_penalty = self.update_rate * torch.sum(torch.square(regrets)) / 2.0
        lag_loss = torch.sum(self.w_rgt * regrets)
        
        # Total loss - точно как в TF версии
        loss = -revenue + rgt_penalty + lag_loss
        
        # Add regularization loss if specified
        wd = None if "wd" not in self.config.train else self.config.train.wd
        if wd is not None:
            reg_loss = self.net.get_reg_loss(self.net.parameters(), wd)
            loss = loss + reg_loss
        
        # Backpropagation and optimization
        t1 = time.time()
        loss.backward()
        self.optimizer.step()
        time_profile['backward_pass'] = time.time() - t1
        
        # НЕ обновляем множители Лагранжа здесь - это делается в основном цикле обучения
        # точно как в TensorFlow версии
        
        # Collect metrics
        rgt_mean = torch.mean(regrets)
        metrics = [revenue.item(), rgt_mean.item(), rgt_penalty.item(), 
                  lag_loss.item(), loss.item(), torch.mean(self.w_rgt).item(), 
                  self.update_rate.item()]
        
        # Write to TensorBoard
        self.writer.add_scalar('train/revenue', revenue.item())
        self.writer.add_scalar('train/regret', rgt_mean.item())
        self.writer.add_scalar('train/rgt_penalty', rgt_penalty.item())
        self.writer.add_scalar('train/lag_loss', lag_loss.item())
        self.writer.add_scalar('train/net_loss', loss.item())
        self.writer.add_scalar('train/w_rgt_mean', torch.mean(self.w_rgt).item())
        
        time_profile['total'] = time.time() - t_start
        
        # Логирование профилирования времени только каждый 50-й шаг
        if hasattr(self, 'step_counter'):
            self.step_counter += 1
        else:
            self.step_counter = 0
            
        if self.step_counter % 50 == 0:
            # Создаем более компактное представление профилирования
            profiling_summary = " | ".join([f"{k}: {v:.4f}s" for k, v in time_profile.items()])
            self.logger.info(f"Time profiling: {profiling_summary}")
        
        return metrics
    
    def optimize_misreports(self, x, adv_var, mode="train"):
        """
        Find optimal misreports via gradient ascent
        Точно соответствует оригинальной TensorFlow реализации
        """
        import time
        t_start = time.time()
        time_profile = {}
        
        # Set network to evaluation mode during misreport optimization
        self.net.eval()
        
        batch_size = x.size(0)
        num_agents = self.config.num_agents
        num_items = self.config.num_items
        num_misreports = adv_var.size(0) if len(adv_var.shape) == 4 else 1
        
        # Number of gradient steps
        if mode == "train":
            gd_iter = self.config.train.gd_iter
            gd_lr = self.config.train.gd_lr
        elif mode == "val":
            gd_iter = self.config.val.gd_iter
            gd_lr = self.config.val.gd_lr
        else:  # test
            gd_iter = self.config.test.gd_iter
            gd_lr = self.config.test.gd_lr
        
        # Создаем оптимизатор для неправдивых заявок
        mis_opt = optim.Adam([adv_var], lr=gd_lr)
        
        # Создаем X_mis для использования в вычислении полезности
        t1 = time.time()
        X_mis = x.repeat(num_agents * num_misreports, 1, 1)
        
        # Создаем маски для замены значений и выбора полезности
        adv_mask = torch.zeros(num_agents, num_misreports, batch_size, num_agents, num_items, device=self.device)
        u_mask = torch.zeros(num_agents, num_misreports, batch_size, num_agents, device=self.device)
        
        for agent_idx in range(num_agents):
            adv_mask[agent_idx, :, :, agent_idx, :] = 1.0
            u_mask[agent_idx, :, :, agent_idx] = 1.0
        time_profile['preparation'] = time.time() - t1
        
        # Оптимизируем неправдивые заявки для максимизации полезности
        t_gd_total = 0
        for i in range(gd_iter):
            t_gd_start = time.time()
            mis_opt.zero_grad()
            
            # Преобразуем X_mis в нужную форму для применения маски
            X_r = X_mis.reshape(num_agents, num_misreports, batch_size, num_agents, num_items)
            
            # Создаем misreports, заменяя значения согласно маске
            adv = adv_var.unsqueeze(0).expand(num_agents, -1, -1, -1, -1)
            misreports_reshaped = X_r * (1 - adv_mask) + adv * adv_mask
            
            # Преобразуем обратно в пакетный формат
            misreports = misreports_reshaped.reshape(-1, num_agents, num_items)
            
            # Получаем распределения и платежи для неправдивых заявок
            alloc_mis, pay_mis = self.net(misreports)
            
            # Вычисляем полезность для неправдивых заявок
            utility_mis = self.compute_utility(X_mis, alloc_mis, pay_mis)
            
            # Преобразуем в нужную форму и применяем маску
            utility_mis_reshaped = utility_mis.reshape(num_agents, num_misreports, batch_size, num_agents)
            utility_mis_masked = utility_mis_reshaped * u_mask
            
            # Максимизируем полезность - минимизируем -utility_mis_masked, как в TF оригинале
            loss = -torch.sum(utility_mis_masked)
            
            # Обратное распространение
            loss.backward()
            
            # Обновляем неправдивые заявки
            mis_opt.step()
            
            # Применяем клиппинг к допустимому диапазону
            with torch.no_grad():
                adv_var.data = self.clip_op_lambda(adv_var.data)
            
            t_gd_total += time.time() - t_gd_start
        
        time_profile['gd_steps'] = t_gd_total
        time_profile['total'] = time.time() - t_start
        
        # Профилирование выводим только каждый 100-й вызов
        if hasattr(self, 'misreport_counter'):
            self.misreport_counter += 1
        else:
            self.misreport_counter = 0
            
        if self.misreport_counter % 100 == 0:
            # Создаем более компактное представление профилирования
            profiling_summary = f"prep: {time_profile['preparation']:.4f}s | gd: {time_profile['gd_steps']:.4f}s | total: {time_profile['total']:.4f}s"
            self.logger.info(f"Misreport opt ({mode}): {profiling_summary}")
        
        # Set network back to training mode
        if mode == "train":
            self.net.train()
            
        return adv_var
    
    def validate(self, iter):
        """
        Run validation
        """
        self.logger.info(f"Running validation at iter {iter}")
        self.net.eval()
        
        try:
            all_metrics = []
            
            # Как в TensorFlow версии - используем gen_func
            for batch_idx in range(self.config.val.num_batches):
                # Выводим информацию только в начале и каждые 5 батчей
                if batch_idx == 0 or (batch_idx + 1) % 5 == 0 or batch_idx == self.config.val.num_batches - 1:
                    self.logger.info(f"Validating: {batch_idx+1}/{self.config.val.num_batches} batches processed")
                
                # Получаем batch как в TensorFlow версии
                X_batch, ADV_batch, idx = self.val_gen.gen_func()
                
                # Move to device
                X_batch = X_batch.to(self.device)
                ADV_batch = ADV_batch.to(self.device)
                
                # Get allocations and payments
                alloc, pay = self.net(X_batch)
                revenue = self.compute_rev(pay)
                
                # Вычисляем полезность
                utility = self.compute_utility(X_batch, alloc, pay)
                
                # На первой итерации (iter=0) не делаем сложных вычислений
                if iter == 0:
                    # Считаем только IRP без Regret
                    irp = torch.mean(torch.relu(-utility))
                    metrics = [revenue.item(), 0.0, irp.item()]
                else:
                    # Инициализируем misreports
                    adv_var = ADV_batch.clone().detach().requires_grad_(True)
                    
                    # Оптимизируем misreports
                    val_gd_iter = min(self.config.val.gd_iter, 100)  # Ограничиваем для скорости
                    val_gd_lr = self.config.val.gd_lr
                    
                    # Создаем оптимизатор для misreports
                    mis_opt = optim.Adam([adv_var], lr=val_gd_lr)
                    
                    # Проходим итерации оптимизации
                    for i in range(val_gd_iter):
                        mis_opt.zero_grad()
                        
                        # Получаем распределения и платежи для misreports
                        mis_alloc, mis_pay = self.net(adv_var)
                        mis_utility = self.compute_utility(X_batch, mis_alloc, mis_pay)
                        
                        # Максимизируем полезность (минимизируем отрицательную полезность)
                        loss = -torch.sum(mis_utility)
                        
                        # Обратное распространение
                        loss.backward()
                        
                        # Шаг оптимизации
                        mis_opt.step()
                        
                        # Применяем клиппинг
                        with torch.no_grad():
                            adv_var.data = self.clip_op_lambda(adv_var.data)
                    
                    # Теперь вычисляем regret используя унифицированную функцию
                    regret_mean = self.compute_regret_unified(X_batch, utility, adv_var)
                    
                    metrics = [revenue.item(), regret_mean.item(), torch.mean(torch.relu(-utility)).item()]
                
                all_metrics.append(metrics)
            
            # Compute average metrics
            avg_metrics = np.mean(all_metrics, axis=0)
            
            # Структурированный вывод финальных метрик
            self.logger.info("=" * 50)
            self.logger.info(f"VALIDATION RESULTS (Iteration {iter})")
            self.logger.info("-" * 50)
            for name, value in zip(["Revenue", "Regret", "IRP"], avg_metrics):
                self.logger.info(f"{name}: {value:.6f}")
                self.writer.add_scalar(f'val/{name.lower()}', value, iter)
            self.logger.info("=" * 50)
        
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            self.logger.error(traceback.format_exc())
            
        finally:
            # Set model back to training mode
            self.net.train()
    
    def test(self, generator):
        """
        Run testing
        """
        self.test_gen = generator
        
        # Load model
        model_path = os.path.join(self.config.dir_name, f'model-{self.config.test.restore_iter}.pt')
        self.logger.info(f"Loading model from {model_path} for testing")
        
        try:
            # Проверка конфигурации
            self.logger.info(f"Model configuration:" + 
                            f"\n - Distribution: {self.config.distribution_type if hasattr(self.config, 'distribution_type') else 'N/A'}" +
                            f"\n - Num agents: {self.config.num_agents}" +
                            f"\n - Num items: {self.config.num_items}")
            
            # Используем map_location для совместимости между разными устройствами
            checkpoint = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            
            # Устанавливаем сеть в режим оценки
            self.net.eval()
            
            # Структурированный вывод информации о тестировании
            self.logger.info("=" * 60)
            self.logger.info(f"TESTING MODEL FROM ITERATION {self.config.test.restore_iter}")
            self.logger.info(f"Test configuration: {self.config.test.num_batches} batches, batch size {self.config.test.batch_size}")
            self.logger.info(f"GD iterations: {self.config.test.gd_iter}, GD learning rate: {self.config.test.gd_lr}")
            self.logger.info("-" * 60)
            
            metrics_sum = {name: 0.0 for name in self.metric_names}
            batch_metrics = []  # Для сбора статистики по всем батчам
            num_batches = 0
            
            # Напрямую генерируем батчи данных вместо использования dataloader
            for batch_idx in range(self.config.test.num_batches):
                # Выводим прогресс только для каждого 10-го батча или в начале/конце
                if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or batch_idx == self.config.test.num_batches - 1:
                    self.logger.info(f"Testing progress: {batch_idx+1}/{self.config.test.num_batches} batches")
                
                try:
                    # Генерируем данные онлайн
                    x_shape = (self.config.test.batch_size, self.config.num_agents, self.config.num_items)
                    # Для ADV нужна 4-мерная форма: [num_misreports, batch_size, num_agents, num_items]
                    adv_shape = (self.config.test.num_misreports, self.config.test.batch_size, self.config.num_agents, self.config.num_items)
                    
                    # Генерируем X и ADV с проверкой формы
                    try:
                        X_batch_np = self.test_gen.generate_random_X(x_shape)
                        self.logger.debug(f"X shape: {X_batch_np.shape}")
                        X_batch = torch.tensor(X_batch_np, dtype=torch.float32, device=self.device)
                    except Exception as x_e:
                        self.logger.error(f"Error generating X values: {x_e}")
                        self.logger.error(f"X shape: {x_shape}")
                        raise x_e
                        
                    try:
                        ADV_batch_np = self.test_gen.generate_random_ADV(adv_shape)
                        self.logger.debug(f"ADV shape: {ADV_batch_np.shape}")
                        ADV_batch = torch.tensor(ADV_batch_np, dtype=torch.float32, device=self.device)
                        
                        # Для CA аукционов: берем первый misreport для инициализации
                        # Преобразуем из [num_misreports, batch_size, num_agents, num_items] в [batch_size, num_agents, num_items]
                        if self.config.test.num_misreports > 1:
                            initial_adv = ADV_batch[0]  # Берем первый misreport
                        else:
                            initial_adv = ADV_batch.squeeze(0)  # Убираем первую размерность если она равна 1
                            
                    except Exception as adv_e:
                        self.logger.error(f"Error generating ADV values: {adv_e}")
                        self.logger.error(f"ADV shape: {adv_shape}")
                        raise adv_e
                    
                    # Инициализируем misreports с правильной размерностью
                    adv_var = initial_adv.clone().detach().requires_grad_(True)
                    
                    # Ограничиваем количество итераций для предотвращения зависания
                    test_gd_iter = min(50, self.config.test.gd_iter)
                    test_gd_lr = min(0.05, self.config.test.gd_lr)
                    
                    # Создаем оптимизатор для misreports с уменьшенным learning rate
                    mis_opt = optim.Adam([adv_var], lr=test_gd_lr)
                    
                    # Проходим несколько итераций оптимизации (без логирования каждой итерации)
                    for i in range(test_gd_iter):
                        mis_opt.zero_grad()
                        
                        # Создаем misreports напрямую
                        mis_alloc, mis_pay = self.net(adv_var)
                        mis_utility = self.compute_utility(X_batch, mis_alloc, mis_pay)
                        
                        # Максимизируем полезность
                        loss = -torch.sum(mis_utility)
                        
                        # Обратное распространение
                        loss.backward()
                        
                        # Шаг оптимизации
                        mis_opt.step()
                        
                        # Применяем клиппинг
                        with torch.no_grad():
                            adv_var.data = self.clip_op_lambda(adv_var.data)
                    
                    # Вычисляем метрики
                    alloc, pay = self.net(X_batch)
                    revenue = self.compute_rev(pay)
                    
                    # Вычисляем полезность
                    utility = self.compute_utility(X_batch, alloc, pay)
                    
                    # Вычисляем IRP
                    irp = torch.mean(torch.relu(-utility))
                    
                    # Получаем финальные распределения и платежи для misreports
                    mis_alloc, mis_pay = self.net(adv_var)
                    mis_utility = self.compute_utility(X_batch, mis_alloc, mis_pay)
                    
                    # Вычисляем regret используя унифицированную функцию
                    # Для regret нужно передать оригинальный 4-мерный ADV
                    regret_mean = self.compute_regret_unified(X_batch, utility, ADV_batch)
                    
                    # Собираем метрики
                    metrics = {
                        "Revenue": revenue.item(),
                        "Regret": regret_mean.item(),
                        "IRP": irp.item()
                    }
                    
                    # Сохраняем метрики для статистики
                    batch_metrics.append([metrics["Revenue"], metrics["Regret"], metrics["IRP"]])
                    
                    # Аккумулируем метрики
                    for name in self.metric_names:
                        metrics_sum[name] += metrics[name]
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx+1}: {str(e)}")
                    continue
            
            # Если хотя бы один батч был успешно обработан
            if num_batches > 0:
                # Усредняем метрики
                for name in self.metric_names:
                    metrics_sum[name] /= num_batches
                
                # Вычисляем статистику по батчам если их было несколько
                batch_metrics_array = np.array(batch_metrics)
                metrics_std = np.std(batch_metrics_array, axis=0) if len(batch_metrics) > 1 else np.zeros(3)
                
                # Форматируем вывод результатов
                self.logger.info("=" * 60)
                self.logger.info(f"TEST RESULTS (Model iteration: {self.config.test.restore_iter})")
                self.logger.info("-" * 60)
                self.logger.info(f"Successful batches: {num_batches}/{self.config.test.num_batches}")
                self.logger.info("-" * 60)
                self.logger.info("Metric          | Mean      | Std Dev   ")
                self.logger.info("-" * 60)
                
                for i, name in enumerate(self.metric_names):
                    self.logger.info(f"{name:<15} | {metrics_sum[name]:<9.6f} | {metrics_std[i]:<9.6f}")
                    
                self.logger.info("=" * 60)
            else:
                self.logger.error("No batches were successfully processed during testing")
            
        except Exception as e:
            self.logger.error(f"Error testing model: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def log_metrics(self, iter, metrics, start_time):
        """Log training metrics"""
        time_elapsed = time.time() - start_time
        
        # Более компактное и структурированное представление метрик
        revenue, regret, reg_loss, lag_loss, net_loss, w_rgt_mean, update_rate = metrics
        
        # Каждые 10 итераций - детальный лог
        if iter % 10 == 0:
            self.logger.info(f"TRAIN ({iter}) | Rev: {revenue:.4f} | Rgt: {regret:.4f} | Loss: {net_loss:.4f} | Time: {time_elapsed:.2f}s")
            
        # Каждые 100 итераций - полный лог с дополнительной информацией
        if iter % 100 == 0:
            self.logger.info("-" * 60)
            self.logger.info(f"TRAIN ITER {iter} - DETAILED METRICS")
            self.logger.info(f"Revenue: {revenue:.6f} | Regret: {regret:.6f}")
            self.logger.info(f"Reg_Loss: {reg_loss:.6f} | Lag_Loss: {lag_loss:.6f} | Net_Loss: {net_loss:.6f}")
            self.logger.info(f"w_rgt_mean: {w_rgt_mean:.6f} | update_rate: {update_rate:.6f}")
            self.logger.info(f"Elapsed time: {time_elapsed:.2f}s")
            self.logger.info("-" * 60)
            
        # Вызываем функцию обратного вызова, если она определена
        if self.callback is not None:
            try:
                self.callback(iter, metrics, time_elapsed)
            except Exception as e:
                self.logger.warning(f"Ошибка при вызове функции обратного вызова: {str(e)}")
    
    def save_model(self, iter):
        """Save model checkpoint"""
        model_path = os.path.join(self.config.dir_name, f'model-{iter}.pt')
        
        checkpoint = {
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'w_rgt': self.w_rgt,
            'update_rate': self.update_rate,
            'iter': iter
        }
        
        torch.save(checkpoint, model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def update_lagrange_multipliers(self, rgt):
        """
        Update Lagrangian multipliers for regret constraints
        Использует градиентный спуск как в оригинальной TensorFlow реализации
        """
        # Отсоединяем от вычислительного графа
        with torch.no_grad():
            # В TF версии: loss_3 = -lag_loss = -tf.reduce_sum(self.w_rgt * rgt)
            # Градиент loss_3 по w_rgt = -rgt
            # GradientDescentOptimizer минимизирует loss_3, поэтому w_rgt = w_rgt - lr * grad
            # То есть: w_rgt = w_rgt - update_rate * (-rgt) = w_rgt + update_rate * rgt
            self.w_rgt = torch.clamp(self.w_rgt + self.update_rate * rgt, min=0.0) 
    
    def compute_regret_unified(self, X_batch, utility, adv_var):
        """
        Унифицированное вычисление regret, соответствующее оригинальной TensorFlow версии
        Точно копирует логику из train_step
        """
        batch_size = X_batch.size(0)
        num_agents = self.config.num_agents
        num_items = self.config.num_items
        
        # Определяем количество misreports из размера adv_var
        if len(adv_var.shape) == 4:
            num_misreports = adv_var.size(0)
        else:
            num_misreports = 1
        
        # Расширяем X для создания misreports - точно как в train_step
        X_mis = X_batch.repeat(num_agents * num_misreports, 1, 1)
        
        # Создаем маску для замены значений - точно как в train_step
        adv_mask = torch.zeros(num_agents, num_misreports, batch_size, num_agents, num_items, device=self.device)
        for agent_idx in range(num_agents):
            adv_mask[agent_idx, :, :, agent_idx, :] = 1.0
        
        # Преобразуем X_mis в нужную форму для применения маски - точно как в train_step
        X_r = X_mis.reshape(num_agents, num_misreports, batch_size, num_agents, num_items)
        
        # Создаем misreports, заменяя значения согласно маске - точно как в train_step
        adv = adv_var.unsqueeze(0).expand(num_agents, -1, -1, -1, -1)
        misreports_reshaped = X_r * (1 - adv_mask) + adv * adv_mask
        
        # Преобразуем обратно в пакетный формат - точно как в train_step
        misreports = misreports_reshaped.reshape(-1, num_agents, num_items)
        
        # Получаем распределения и платежи для неправдивых заявок - точно как в train_step
        alloc_mis, pay_mis = self.net(misreports)
        
        # Вычисляем полезность для неправдивых заявок - точно как в train_step
        utility_mis = self.compute_utility(X_mis, alloc_mis, pay_mis)
        
        # Создаем маску для выбора только соответствующих значений полезности - точно как в train_step
        u_mask = torch.zeros(num_agents, num_misreports, batch_size, num_agents, device=self.device)
        for agent_idx in range(num_agents):
            u_mask[agent_idx, :, :, agent_idx] = 1.0
        
        # Преобразуем utility_mis в нужную форму для вычисления regret - точно как в train_step
        utility_mis_reshaped = utility_mis.reshape(num_agents, num_misreports, batch_size, num_agents)
        
        # Расширяем истинную полезность для сравнения - точно как в train_step
        utility_true_expanded = utility.repeat(num_agents * num_misreports, 1)
        utility_true_reshaped = utility_true_expanded.reshape(num_agents, num_misreports, batch_size, num_agents)
        
        # Вычисляем excess_utility - точно как в train_step
        excess_utility = torch.relu((utility_mis_reshaped - utility_true_reshaped) * u_mask)
        
        # Вычисляем regret для каждого агента - точно как в train_step
        regrets = []
        for agent_idx in range(num_agents):
            # Максимум по misreports для каждого агента и батча
            max_over_mis = torch.max(excess_utility[agent_idx], dim=0)[0]  # [batch_size, num_agents]
            # Затем берем максимум по batch, а потом среднее - как в TF
            agent_regret = torch.mean(torch.max(max_over_mis, dim=1)[0])  # скаляр
            regrets.append(agent_regret)
        
        regrets = torch.stack(regrets)
        return torch.mean(regrets) 