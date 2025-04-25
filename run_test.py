import os
import sys
import torch
import numpy as np

# Словарь настроек для разных сценариев аукционов
# Структура такая же, как в run_train.py
# Содержит конфигурации для различных механизмов аукционов:
# - additive: аддитивные ценности (ценность пакета = сумма ценностей отдельных предметов)
# - unit: единичные ценности (агент получает полезность только от первого предмета)
# - CA: комбинаторные аукционы (учитывают зависимости между предметами)
# 
# Формат имени конфигурации: [тип_ценности]_[кол_агентов]x[кол_предметов]_[распределение]
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

def get_best_model_info(config_dir):
    """
    Получает информацию о лучшей модели из best_model.pt
    
    Возвращает:
        best_iter: итерация лучшей модели или None если файл не найден
        best_regret: лучший regret или None
    """
    best_model_path = os.path.join(config_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            best_iter = checkpoint.get('iter', None)
            best_regret = checkpoint.get('best_val_regret', None)
            return best_iter, best_regret
        except Exception as e:
            print(f"Ошибка при загрузке best_model.pt: {e}")
            return None, None
    return None, None

def print_usage():
    """Выводит информацию об использовании скрипта"""
    print("\nИспользование:")
    print("python run_test.py [конфигурация] [модель_опция]")
    print("\nПараметры:")
    print("  конфигурация    - имя конфигурации (по умолчанию: additive_1x2_uniform)")
    print("  модель_опция    - 'best' для лучшей модели или номер итерации (по умолчанию: best)")
    print("\nПримеры:")
    print("  python run_test.py                              # тестирование лучшей модели additive_1x2_uniform")
    print("  python run_test.py additive_1x2_uniform         # тестирование лучшей модели указанной конфигурации")
    print("  python run_test.py additive_1x2_uniform best    # тестирование лучшей модели")
    print("  python run_test.py additive_1x2_uniform 1500    # тестирование модели с итерации 1500")
    print(f"\nДоступные конфигурации: {', '.join(settings.keys())}")

if __name__ == "__main__":
    # Определяем конфигурацию и модель для тестирования
    setting = "additive_1x2_uniform"  # Конфигурация по умолчанию
    use_best_model = True             # По умолчанию используем лучшую модель
    restore_iter = None               # Будет определена позже
    
    # УЛУЧШЕННЫЙ разбор аргументов командной строки
    if len(sys.argv) > 1:
        # Проверяем первый аргумент
        if sys.argv[1] in settings:
            # Первый аргумент - имя конфигурации
            setting = sys.argv[1]
        elif sys.argv[1] in ['help', '-h', '--help']:
            print_usage()
            sys.exit(0)
        else:
            print(f"Неизвестная конфигурация: {sys.argv[1]}")
            print_usage()
            sys.exit(1)
    
    # Проверяем второй аргумент (модель)
    if len(sys.argv) > 2:
        if sys.argv[2].lower() == 'best':
            use_best_model = True
        else:
            try:
                restore_iter = int(sys.argv[2])
                use_best_model = False
            except ValueError:
                print(f"Ошибка: второй аргумент должен быть 'best' или числом (итерация модели)")
                print_usage()
                sys.exit(1)
    
    # Загружаем выбранную конфигурацию
    cfg_module = __import__(f"cfgs.{settings[setting]['cfg']}", fromlist=["cfg"])
    cfg = cfg_module.cfg
    
    # УЛУЧШЕНИЕ: Определяем какую модель использовать
    if use_best_model:
        # Пытаемся найти лучшую модель
        best_iter, best_regret = get_best_model_info(cfg.dir_name)
        if best_iter is not None:
            restore_iter = best_iter
            print(f"Найдена лучшая модель: итерация {best_iter}, regret {best_regret:.6f}")
        else:
            # Если лучшая модель не найдена, используем последнюю доступную
            print("Лучшая модель не найдена, ищем последнюю сохраненную модель...")
            
            # Ищем все сохраненные модели
            available_models = []
            if os.path.exists(cfg.dir_name):
                for file in os.listdir(cfg.dir_name):
                    if file.startswith('model-') and file.endswith('.pt'):
                        try:
                            iter_num = int(file.replace('model-', '').replace('.pt', ''))
                            available_models.append(iter_num)
                        except ValueError:
                            continue
            
            if available_models:
                restore_iter = max(available_models)
                print(f"Используем последнюю сохраненную модель: итерация {restore_iter}")
            else:
                print(f"Модели не найдены в директории {cfg.dir_name}")
                print("Убедитесь, что модель была обучена.")
                sys.exit(1)
    
    # Устанавливаем параметры тестирования
    cfg.test.restore_iter = restore_iter
    cfg.test.data = "online"  # Режим генерации данных (online - генерация на лету)
    
    # УЛУЧШЕНИЕ: Используем те же параметры оптимизации misreports, что и при валидации
    cfg.test.gd_iter = cfg.train.gd_iter  # Используем те же итерации как при обучении/валидации
    cfg.test.gd_lr = cfg.train.gd_lr      # Используем тот же learning rate
    
    # Выводим информацию о текущей конфигурации тестирования
    print("=" * 60)
    print("НАСТРОЙКИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print(f"Конфигурация: {setting}")
    print(f"Модель: {'лучшая' if use_best_model else 'пользовательская'} (итерация {cfg.test.restore_iter})")
    print(f"Режим генерации данных: {cfg.test.data}")
    print(f"Размер сети: {cfg.net.num_a_hidden_units} юнитов, {cfg.net.num_a_layers} слоев")
    print(f"Размер тестового батча: {cfg.test.batch_size}")
    print(f"Количество тестовых батчей: {cfg.test.num_batches}")
    print(f"Итерации оптимизации misreports: {cfg.test.gd_iter}")
    print(f"Learning rate для misreports: {cfg.test.gd_lr}")
    print("=" * 60)
    
    # Инициализация нейронной сети в соответствии с выбранной конфигурацией
    net_module = __import__(f"nets.{settings[setting]['net']}", fromlist=["Net"])
    net = net_module.Net(cfg)
    
    # Создаем генератор тестовых данных (распределений ценностей)
    gen_module = __import__(f"data.{settings[setting]['generator']}", fromlist=["Generator"])
    test_gen = gen_module.Generator(cfg, 'test')
    
    # Получаем функцию ограничения значений (clip operation)
    # Используется для ограничения значений misreports в допустимом диапазоне
    clip_ops = __import__("clip_ops.clip_ops", fromlist=[settings[setting]['clip_op']])
    clip_op = getattr(clip_ops, settings[setting]['clip_op'])
    
    # Инициализируем тренер в режиме тестирования
    # Тренер в режиме тестирования вычисляет метрики производительности модели:
    # - Выручка (revenue): средний платеж от всех агентов
    # - Regret: насколько агенты могут улучшить свою полезность, сообщая ложные ценности
    # - IR (индивидуальная рациональность): неотрицательна ли полезность агентов
    from trainer import Trainer
    trainer = Trainer(cfg, 'test', net, clip_op)
    
    # Запускаем тестирование
    print("Запуск тестирования...")
    trainer.test(test_gen)
    print("Тестирование завершено!") 