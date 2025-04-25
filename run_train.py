import os
import sys
import torch
import numpy as np

# Словарь настроек для разных сценариев аукционов
# Каждый сценарий определяет:
# - cfg: файл конфигурации
# - net: тип нейронной сети
# - generator: генератор данных для обучения
# - clip_op: операция ограничения значений
settings = {
    "additive_1x2_uniform": {
        "cfg": "additive_1x2_uniform_config",
        "net": "additive_net",
        "generator": "uniform_01_generator",
        "clip_op": "clip_op_01"
    },
    "additive_1x2_uniform_04_03": {
        "cfg": "additive_1x2_uniform_04_03_config",
        "net": "additive_net",
        "generator": "uniform_04_03_generator",
        "clip_op": "clip_op_04_03"
    },
    "additive_1x2_uniform_416_47": {
        "cfg": "additive_1x2_uniform_416_47_config",
        "net": "additive_net",
        "generator": "uniform_416_47_generator",
        "clip_op": "clip_op_416_47"
    },
    "additive_1x2_uniform_triangle": {
        "cfg": "additive_1x2_uniform_triangle_config",
        "net": "additive_net",
        "generator": "uniform_triangle_01_generator",
        "clip_op": "clip_op_triangle_01"
    },
    "additive_1x10_uniform": {
        "cfg": "additive_1x10_uniform_config",
        "net": "additive_net",
        "generator": "uniform_01_generator",
        "clip_op": "clip_op_01"
    },
    "additive_2x2_uniform": {
        "cfg": "additive_2x2_uniform_config",
        "net": "additive_net",
        "generator": "uniform_01_generator",
        "clip_op": "clip_op_01"
    },
    "additive_2x3_uniform": {
        "cfg": "additive_2x3_uniform_config",
        "net": "additive_net",
        "generator": "uniform_01_generator",
        "clip_op": "clip_op_01"
    },
    "additive_3x10_uniform": {
        "cfg": "additive_3x10_uniform_config",
        "net": "additive_net",
        "generator": "uniform_01_generator",
        "clip_op": "clip_op_01"
    },
    "additive_5x10_uniform": {
        "cfg": "additive_5x10_uniform_config",
        "net": "additive_net",
        "generator": "uniform_01_generator",
        "clip_op": "clip_op_01"
    },
    "unit_1x2_uniform": {
        "cfg": "unit_1x2_uniform_config",
        "net": "unit_net",
        "generator": "uniform_01_generator",
        "clip_op": "clip_op_01"
    },
    "unit_1x2_uniform_23": {
        "cfg": "unit_1x2_uniform_23_config",
        "net": "unit_net",
        "generator": "uniform_23_generator",
        "clip_op": "clip_op_23"
    },
    "unit_2x2_uniform": {
        "cfg": "unit_2x2_uniform_config",
        "net": "unit_net",
        "generator": "uniform_01_generator",
        "clip_op": "clip_op_01"
    },
    "CA_asym_uniform_12_15": {
        "cfg": "CA_asym_uniform_12_15_config",
        "net": "ca2x2_net",
        "generator": "CA_asym_uniform_12_15_generator",
        "clip_op": "clip_op_12_15"
    },
    "CA_sym_uniform_12": {
        "cfg": "CA_sym_uniform_12_config",
        "net": "ca2x2_net",
        "generator": "CA_sym_uniform_12_generator",
        "clip_op": "clip_op_12"
    }
}

def run_training(setting_name=None):
    """
    Функция запуска обучения модели для заданной конфигурации
    
    Аргументы:
        setting_name: имя конфигурации из словаря settings.
                     Если None, используется значение из аргументов командной строки
                     или "additive_1x2_uniform" по умолчанию
    """
    # Получаем настройку из аргументов командной строки или используем значение по умолчанию
    if setting_name is None:
        setting_name = sys.argv[1] if len(sys.argv) > 1 else "additive_1x2_uniform"
    
    if setting_name not in settings:
        raise ValueError(f"Unknown setting: {setting_name}")
    
    setting = setting_name  # Для совместимости с существующим кодом
    
    # Динамический импорт модулей в зависимости от выбранной настройки
    # Импортируем файл конфигурации
    cfg_module = __import__(f"cfgs.{settings[setting]['cfg']}", fromlist=["cfg"])
    cfg = cfg_module.cfg
    
    # Импортируем класс нейронной сети
    net_module = __import__(f"nets.{settings[setting]['net']}", fromlist=["Net"])
    net = net_module.Net(cfg)
    
    # Импортируем генератор данных
    gen_module = __import__(f"data.{settings[setting]['generator']}", fromlist=["Generator"])
    train_gen = gen_module.Generator(cfg, 'train')  # Генератор обучающих данных
    val_gen = gen_module.Generator(cfg, 'val')      # Генератор валидационных данных
    
    # Импортируем функцию ограничения значений (clip operation)
    clip_ops = __import__("clip_ops.clip_ops", fromlist=[settings[setting]['clip_op']])
    clip_op = getattr(clip_ops, settings[setting]['clip_op'])
    
    # Импортируем класс тренера
    from trainer import Trainer
    
    # Выводим информацию о текущей конфигурации
    print(f"Конфигурация: {setting}")
    print(f"Размер сети: {cfg.net.num_a_hidden_units} юнитов, {cfg.net.num_a_layers} слоев")
    print(f"Максимум итераций: {cfg.train.max_iter}")
    print(f"Batch size: {cfg.train.batch_size}")
    print(f"Learning rate: {cfg.train.learning_rate}")
    print(f"Стартовый множитель Лагранжа: {cfg.train.w_rgt_init_val}")
    print(f"Начальный update_rate: {cfg.train.update_rate}")
    
    # Создаем экземпляр тренера
    trainer = Trainer(cfg, 'train', net, clip_op)
    print("Starting full training...")
    # Запускаем обучение, передавая генераторы данных
    trainer.train((train_gen, val_gen))
    print("Training completed!")

# Запускаем обучение только при прямом запуске файла
if __name__ == "__main__":
    run_training()