import os
import sys
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Кеш для загруженных моделей
@st.cache_resource  
def load_cached_model(selected_config, selected_iter, model_path):
    """Кешированная загрузка модели для визуализации"""
    
    # Получаем настройки для выбранной конфигурации
    from run_train import settings
    config_settings = settings[selected_config]
    
    # Импортируем файл конфигурации
    cfg_module = __import__(f"cfgs.{config_settings['cfg']}", fromlist=["cfg"])
    cfg = cfg_module.cfg
    cfg.test.restore_iter = selected_iter
    
    # Импортируем класс нейронной сети
    net_module = __import__(f"nets.{config_settings['net']}", fromlist=["Net"])
    
    # Загружаем checkpoint для проверки архитектуры
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict']
    
    # Функция для определения архитектуры из state_dict
    def infer_architecture_from_state_dict(state_dict):
        """Извлекаем параметры архитектуры из state_dict модели"""
        
        # Находим все ключи, относящиеся к слоям распределения
        alloc_layers = [k for k in state_dict.keys() if k.startswith('a_net.') and 'weight' in k]
        # Находим все ключи, относящиеся к слоям платежей  
        payment_layers = [k for k in state_dict.keys() if k.startswith('p_net.') and 'weight' in k]
        
        # Определяем количество слоев (исключаем выходной слой)
        num_a_layers = len(alloc_layers) - 1 if len(alloc_layers) > 0 else 1
        num_p_layers = len(payment_layers) - 1 if len(payment_layers) > 0 else 1
        
        # Определяем количество скрытых единиц из первого скрытого слоя
        if len(alloc_layers) > 1:
            # Ключ первого скрытого слоя
            first_hidden_key = min([k for k in alloc_layers if not k.endswith('output.weight')])
            num_a_hidden = state_dict[first_hidden_key].shape[0]
        else:
            num_a_hidden = cfg.net.num_a_hidden_units  # fallback
            
        if len(payment_layers) > 1:
            # Ключ первого скрытого слоя платежей
            first_hidden_key = min([k for k in payment_layers if not k.endswith('output.weight')])
            num_p_hidden = state_dict[first_hidden_key].shape[0]
        else:
            num_p_hidden = cfg.net.num_p_hidden_units  # fallback
        
        return num_a_layers, num_p_layers, num_a_hidden, num_p_hidden
    
    # Определяем архитектуру из checkpoint'а
    try:
        num_a_layers, num_p_layers, num_a_hidden, num_p_hidden = infer_architecture_from_state_dict(state_dict)
        
        # Обновляем конфигурацию с обнаруженными параметрами
        cfg.net.num_a_layers = num_a_layers
        cfg.net.num_p_layers = num_p_layers
        cfg.net.num_a_hidden_units = num_a_hidden
        cfg.net.num_p_hidden_units = num_p_hidden
        
        # Создаем сеть с обновленной конфигурацией
        net = net_module.Net(cfg)
        
        # Загружаем состояние модели
        net.load_state_dict(state_dict)
        net.eval()
        
        return net, cfg, config_settings
        
    except Exception:
        # Если автоматическое определение не сработало, пробуем стандартные конфигурации
        common_configs = [
            {'num_a_layers': 2, 'num_p_layers': 2, 'num_a_hidden': 50, 'num_p_hidden': 50},
            {'num_a_layers': 2, 'num_p_layers': 2, 'num_a_hidden': 100, 'num_p_hidden': 100},
            {'num_a_layers': 3, 'num_p_layers': 3, 'num_a_hidden': 50, 'num_p_hidden': 50},
            {'num_a_layers': 3, 'num_p_layers': 3, 'num_a_hidden': 100, 'num_p_hidden': 100},
            {'num_a_layers': 1, 'num_p_layers': 1, 'num_a_hidden': 50, 'num_p_hidden': 50},
            {'num_a_layers': 4, 'num_p_layers': 4, 'num_a_hidden': 50, 'num_p_hidden': 50},
            {'num_a_layers': 2, 'num_p_layers': 2, 'num_a_hidden': 200, 'num_p_hidden': 200},
            {'num_a_layers': 2, 'num_p_layers': 2, 'num_a_hidden': 128, 'num_p_hidden': 128},
            # Добавляем конфигурации с большим количеством слоев
            {'num_a_layers': 5, 'num_p_layers': 5, 'num_a_hidden': 50, 'num_p_hidden': 50},
            {'num_a_layers': 6, 'num_p_layers': 6, 'num_a_hidden': 50, 'num_p_hidden': 50},
            {'num_a_layers': 6, 'num_p_layers': 6, 'num_a_hidden': 100, 'num_p_hidden': 100},
            {'num_a_layers': 7, 'num_p_layers': 7, 'num_a_hidden': 50, 'num_p_hidden': 50},
            {'num_a_layers': 8, 'num_p_layers': 8, 'num_a_hidden': 50, 'num_p_hidden': 50},
            # Добавляем варианты с разными размерами скрытых слоев
            {'num_a_layers': 4, 'num_p_layers': 4, 'num_a_hidden': 100, 'num_p_hidden': 100},
            {'num_a_layers': 5, 'num_p_layers': 5, 'num_a_hidden': 100, 'num_p_hidden': 100},
            {'num_a_layers': 3, 'num_p_layers': 3, 'num_a_hidden': 200, 'num_p_hidden': 200},
            {'num_a_layers': 4, 'num_p_layers': 4, 'num_a_hidden': 200, 'num_p_hidden': 200},
        ]
        
        for config_attempt in common_configs:
            try:
                # Обновляем конфигурацию
                cfg.net.num_a_layers = config_attempt['num_a_layers']
                cfg.net.num_p_layers = config_attempt['num_p_layers']
                cfg.net.num_a_hidden_units = config_attempt['num_a_hidden']
                cfg.net.num_p_hidden_units = config_attempt['num_p_hidden']
                
                # Создаем новую сеть
                net = net_module.Net(cfg)
                
                # Пытаемся загрузить
                net.load_state_dict(state_dict)
                net.eval()
                
                return net, cfg, config_settings
                
            except Exception:
                continue
        
        # Если ни одна конфигурация не сработала
        raise Exception("Не удалось определить архитектуру модели и загрузить её состояние")

# Функция для поиска доступных моделей для конфигурации
@st.cache_data
def find_model_iterations(config_name):
    # Импортируем settings в функции для избежания циклических импортов
    from run_train import settings
    
    if config_name in settings:
        config_settings = settings[config_name]
        cfg_module = __import__(f"cfgs.{config_settings['cfg']}", fromlist=["cfg"])
        cfg = cfg_module.cfg
        
        # Проверяем директорию эксперимента
        experiment_dir = Path(cfg.dir_name)
        if not experiment_dir.exists():
            return []
        
        # Ищем модели
        model_files = list(experiment_dir.glob("model-*.pt"))
        iterations = [int(f.name.split("-")[1].split(".")[0]) for f in model_files]
        iterations.sort()
        return iterations
    return []

# Установка заголовка приложения
st.set_page_config(
    page_title="RegretNet - Оптимальные аукционы с глубоким обучением",
    page_icon="📊",
    layout="wide"
)

# Стили для улучшения внешнего вида
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2563EB;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #4B5563;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #4B5563;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.markdown('<div class="main-header">RegretNet - Проектирование оптимальных аукционов</div>', unsafe_allow_html=True)
st.markdown('<div class="info-text">Обучение и тестирование моделей для проектирования оптимальных аукционов с использованием глубокого обучения</div>', unsafe_allow_html=True)

# Загружаем все доступные конфигурации
# Важно: импортируем только settings, не запуская при этом обучение
@st.cache_data
def load_auction_settings():
    """Загрузка всех доступных настроек аукционов"""
    # Импортируем только настройки, без запуска обучения
    # В run_train.py обучение теперь обернуто в функцию и не запускается при импорте
    from run_train import settings
    return settings

    # Используем функцию find_model_iterations, объявленную выше

try:
    settings = load_auction_settings()
    auction_configs = list(settings.keys())
except Exception as e:
    st.error(f"Ошибка при загрузке конфигураций аукционов: {str(e)}")
    st.exception(e)
    auction_configs = []

# Создаем боковую панель для выбора режима и настроек
st.sidebar.markdown('<div class="sub-header">Настройки</div>', unsafe_allow_html=True)

# Выбор режима: обучение, тестирование или визуализация
mode = st.sidebar.radio("Режим", ["Обучение", "Тестирование", "Визуализация результатов", "Визуализация аукционов"])

if mode == "Обучение":
    st.markdown('<div class="sub-header">Обучение модели</div>', unsafe_allow_html=True)
    
    # Информация о расширенных настройках
    with st.expander("ℹ️ Справка по настройкам обучения", expanded=False):
        st.markdown("""
        **Основные параметры:**
        - **Максимальное количество итераций**: Общее количество шагов обучения
        - **Размер батча**: Количество образцов, обрабатываемых за один шаг
        - **Скорость обучения**: Шаг градиентного спуска
        - **Частота сохранения**: Интервал сохранения контрольных точек модели
        
        **Множители Лагранжа** управляют ограничениями в задаче оптимизации:
        - **Начальный множитель**: Стартовое значение для штрафа за нарушение ограничений
        - **Мин/Макс значения**: Границы для множителей Лагранжа
        - **Частота обновления**: Как часто обновлять множители
        
        **Оптимизатор** определяет алгоритм обновления весов:
        - **Adam**: Адаптивный алгоритм с моментумом (рекомендуется)
        - **SGD**: Стохастический градиентный спуск
        - **Weight decay**: L2 регуляризация для предотвращения переобучения
        
        **Планировщик скорости обучения** изменяет скорость обучения во время тренировки:
        - **StepLR**: Уменьшает на фиксированный коэффициент каждые N шагов
        - **ExponentialLR**: Экспоненциальное уменьшение
        - **CosineAnnealingLR**: Косинусное расписание
        
        **Архитектура сети**:
        - **Количество слоев**: Глубина нейронной сети
        - **Скрытые единицы**: Ширина каждого слоя
        - **Dropout**: Вероятность отключения нейронов (предотвращает переобучение)
        """)
    
    # Выбор конфигурации аукциона
    selected_config = st.selectbox("Выберите конфигурацию аукциона", auction_configs)
    
    # Предустановленные конфигурации
    st.markdown("### Быстрые настройки")
    preset_config = st.selectbox("Выберите предустановленную конфигурацию", 
                                ["Пользовательские", "Быстрое обучение", "Качественное обучение", "Экспериментальное"], 
                                help="Выберите готовый набор настроек или настройте вручную")
    
    # Словарь предустановленных значений
    presets = {
        "Быстрое обучение": {
            "max_iter": 500, "batch_size": 64, "lr": 0.003, "gd_iter": 15,
            "w_rgt_init": 3.0, "num_a_layers": 2, "num_p_layers": 2,
            "use_lr_scheduler": False, "use_early_stopping": True
        },
        "Качественное обучение": {
            "max_iter": 5000, "batch_size": 32, "lr": 0.001, "gd_iter": 50,
            "w_rgt_init": 5.0, "num_a_layers": 3, "num_p_layers": 3,
            "use_lr_scheduler": True, "use_early_stopping": True
        },
        "Экспериментальное": {
            "max_iter": 10000, "batch_size": 16, "lr": 0.0005, "gd_iter": 100,
            "w_rgt_init": 10.0, "num_a_layers": 4, "num_p_layers": 4,
            "use_lr_scheduler": True, "use_early_stopping": False
        }
    }
    
    # Применяем предустановленные значения или используем значения по умолчанию
    def get_preset_value(key, default_value):
        if preset_config != "Пользовательские" and key in presets[preset_config]:
            return presets[preset_config][key]
        return default_value
    
    # Организуем настройки по категориям с использованием expander'ов
    st.markdown("### Основные параметры обучения")
    col1, col2, col3 = st.columns(3)
    with col1:
        max_iter = st.number_input("Максимальное количество итераций", min_value=100, max_value=20000, 
                                  value=get_preset_value("max_iter", 2000), step=100)
        batch_size = st.number_input("Размер батча", min_value=16, max_value=512, 
                                    value=get_preset_value("batch_size", 64), step=16)
    with col2:
        lr = st.number_input("Скорость обучения", min_value=0.0001, max_value=0.01, 
                            value=get_preset_value("lr", 0.001), format="%.4f", step=0.0001)
        gd_iter = st.number_input("Итерации для вычисления misreports", min_value=10, max_value=200, 
                                 value=get_preset_value("gd_iter", 25), step=5)
    with col3:
        save_frequency = st.number_input("Частота сохранения модели", min_value=10, max_value=1000, value=100, step=10, help="Каждые N итераций сохранять модель")
        validation_frequency = st.number_input("Частота валидации", min_value=1, max_value=100, value=10, step=1, help="Каждые N итераций проводить валидацию")
    
    # Параметры множителей Лагранжа
    with st.expander("🎯 Настройки множителей Лагранжа", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            w_rgt_init = st.number_input("Начальный множитель Лагранжа", min_value=0.1, max_value=50.0, 
                                        value=get_preset_value("w_rgt_init", 5.0), step=0.5)
            w_rgt_max = st.number_input("Максимальный множитель Лагранжа", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
        with col2:
            w_rgt_min = st.number_input("Минимальный множитель Лагранжа", min_value=0.01, max_value=5.0, value=0.1, step=0.01)
            update_rate = st.number_input("Начальный коэффициент обновления", min_value=0.01, max_value=20.0, value=1.0, step=0.1)
        with col3:
            lagrange_update_freq = st.number_input("Частота обновления Лагранжа", min_value=1, max_value=50, value=5, step=1, help="Каждые N итераций обновлять множители")
            rgt_start_iter = st.number_input("Начальная итерация для Regret", min_value=0, max_value=1000, value=0, step=50, help="С какой итерации начинать учитывать regret")
    
    # Параметры оптимизатора
    with st.expander("⚙️ Настройки оптимизатора", expanded=False):
        optimizer_type = st.selectbox("Тип оптимизатора", ["Adam", "SGD", "RMSprop"], index=0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            weight_decay = st.number_input("Регуляризация L2 (weight_decay)", min_value=0.0, max_value=0.01, value=0.0, format="%.6f", step=0.000001)
            momentum = st.number_input("Momentum (только для SGD)", min_value=0.0, max_value=0.999, value=0.9, step=0.01) if optimizer_type == "SGD" else 0.9
        with col2:
            if optimizer_type == "Adam":
                beta1 = st.number_input("Beta1 (Adam)", min_value=0.5, max_value=0.999, value=0.9, step=0.01)
                beta2 = st.number_input("Beta2 (Adam)", min_value=0.9, max_value=0.9999, value=0.999, step=0.001)
            else:
                beta1, beta2 = 0.9, 0.999
        with col3:
            if optimizer_type in ["Adam", "RMSprop"]:
                eps = st.number_input("Epsilon", min_value=1e-10, max_value=1e-6, value=1e-8, format="%.2e")
            else:
                eps = 1e-8
    
    # Параметры расписания скорости обучения
    with st.expander("📈 Расписание скорости обучения", expanded=False):
        use_lr_scheduler = st.checkbox("Использовать планировщик скорости обучения", 
                                       value=get_preset_value("use_lr_scheduler", False))
        
        if use_lr_scheduler:
            col1, col2, col3 = st.columns(3)
            with col1:
                lr_scheduler_type = st.selectbox("Тип планировщика", ["StepLR", "ExponentialLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
            with col2:
                if lr_scheduler_type == "StepLR":
                    lr_step_size = st.number_input("Размер шага (StepLR)", min_value=50, max_value=2000, value=500, step=50)
                    lr_gamma = st.number_input("Коэффициент уменьшения", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
                elif lr_scheduler_type == "ExponentialLR":
                    lr_gamma = st.number_input("Коэффициент экспоненты", min_value=0.9, max_value=0.999, value=0.95, step=0.01)
                    lr_step_size = None
                elif lr_scheduler_type == "CosineAnnealingLR":
                    T_max = st.number_input("T_max (период)", min_value=100, max_value=5000, value=1000, step=100)
                    lr_step_size, lr_gamma = None, None
                else:  # ReduceLROnPlateau
                    lr_patience = st.number_input("Терпение (эпохи без улучшения)", min_value=5, max_value=100, value=20, step=5)
                    lr_gamma = st.number_input("Коэффициент уменьшения", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
                    lr_step_size = None
            with col3:
                lr_min = st.number_input("Минимальная скорость обучения", min_value=1e-8, max_value=1e-4, value=1e-6, format="%.2e")
        else:
            lr_scheduler_type = None
            lr_step_size, lr_gamma, T_max, lr_patience, lr_min = None, None, None, None, None
    
    # Параметры регуляризации и архитектуры
    with st.expander("🏗️ Архитектура сети и регуляризация", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            dropout_rate = st.number_input("Dropout rate", min_value=0.0, max_value=0.8, value=0.0, step=0.05)
            batch_norm = st.checkbox("Использовать Batch Normalization", value=False)
        with col2:
            num_a_layers = st.number_input("Количество слоев распределения", min_value=1, max_value=10, 
                                          value=get_preset_value("num_a_layers", 2), step=1)
            num_p_layers = st.number_input("Количество слоев платежей", min_value=1, max_value=10, 
                                          value=get_preset_value("num_p_layers", 2), step=1)
        with col3:
            num_a_hidden = st.number_input("Скрытые единицы (распределение)", min_value=16, max_value=512, value=100, step=16)
            num_p_hidden = st.number_input("Скрытые единицы (платежи)", min_value=16, max_value=512, value=100, step=16)
            activation = st.selectbox("Функция активации", ["tanh", "relu", "elu", "leaky_relu"], index=0)
    
    # Параметры мониторинга и раннего останова
    with st.expander("📊 Мониторинг и ранний останов", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            use_early_stopping = st.checkbox("Ранний останов", value=get_preset_value("use_early_stopping", False))
            if use_early_stopping:
                early_stopping_patience = st.number_input("Терпение раннего останова", min_value=10, max_value=500, value=100, step=10)
                early_stopping_min_delta = st.number_input("Минимальное улучшение", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
            else:
                early_stopping_patience, early_stopping_min_delta = None, None
        with col2:
            log_frequency = st.number_input("Частота логирования", min_value=1, max_value=100, value=10, step=1)
            verbose_training = st.checkbox("Подробное логирование", value=True)
    
    # Дополнительные параметры
    with st.expander("🔧 Дополнительные параметры", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            grad_clip = st.number_input("Обрезка градиентов (0 = отключено)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            warmup_steps = st.number_input("Шагов разогрева", min_value=0, max_value=1000, value=0, step=50)
        with col2:
            seed = st.number_input("Случайное зерно", min_value=0, max_value=999999, value=42, step=1)
            num_workers = st.number_input("Количество worker'ов", min_value=0, max_value=8, value=0, step=1, help="Для загрузки данных")
    
    # Экспорт/импорт настроек
    st.markdown("### Управление настройками")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Экспорт настроек"):
            # Собираем все настройки в словарь
            current_settings = {
                "max_iter": max_iter,
                "batch_size": batch_size,
                "lr": lr,
                "gd_iter": gd_iter,
                "w_rgt_init": w_rgt_init,
                "w_rgt_max": w_rgt_max,
                "w_rgt_min": w_rgt_min,
                "update_rate": update_rate,
                "lagrange_update_freq": lagrange_update_freq,
                "rgt_start_iter": rgt_start_iter,
                "optimizer_type": optimizer_type,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "beta1": beta1,
                "beta2": beta2,
                "eps": eps,
                "use_lr_scheduler": use_lr_scheduler,
                "lr_scheduler_type": lr_scheduler_type if use_lr_scheduler else None,
                "num_a_layers": num_a_layers,
                "num_p_layers": num_p_layers,
                "num_a_hidden": num_a_hidden,
                "num_p_hidden": num_p_hidden,
                "activation": activation,
                "dropout_rate": dropout_rate,
                "batch_norm": batch_norm,
                "use_early_stopping": use_early_stopping,
                "early_stopping_patience": early_stopping_patience if use_early_stopping else None,
                "grad_clip": grad_clip,
                "seed": seed
            }
            
            # Конвертируем в JSON
            settings_json = json.dumps(current_settings, indent=2, ensure_ascii=False)
            
            # Создаем кнопку скачивания
            st.download_button(
                label="💾 Скачать настройки",
                data=settings_json,
                file_name=f"regretnet_settings_{selected_config}_{max_iter}iter.json",
                mime="application/json",
                help="Сохранить текущие настройки в JSON файл"
            )
    
    with col2:
        uploaded_file = st.file_uploader("📤 Импорт настроек", type=['json'], 
                                        help="Загрузить ранее сохраненные настройки")
        if uploaded_file is not None:
            try:
                settings_data = json.load(uploaded_file)
                st.success("✅ Настройки успешно загружены! Обновите страницу для применения.")
                
                # Показываем предпросмотр загруженных настроек
                with st.expander("Предпросмотр загруженных настроек"):
                    st.json(settings_data)
                    
            except Exception as e:
                st.error(f"❌ Ошибка при загрузке настроек: {str(e)}")
    
    # Сводка настроек
    if selected_config:
        with st.expander("📋 Сводка настроек обучения", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Основные параметры:**")
                st.write(f"• Конфигурация: {selected_config}")
                st.write(f"• Итерации: {max_iter}")
                st.write(f"• Размер батча: {batch_size}")
                st.write(f"• Скорость обучения: {lr:.4f}")
                st.write(f"• GD итерации: {gd_iter}")
                
            with col2:
                st.markdown("**Архитектура:**")
                st.write(f"• Слои распределения: {num_a_layers}")
                st.write(f"• Слои платежей: {num_p_layers}")
                st.write(f"• Скрытые единицы: {num_a_hidden}")
                st.write(f"• Активация: {activation}")
                if dropout_rate > 0:
                    st.write(f"• Dropout: {dropout_rate}")
                
            with col3:
                st.markdown("**Дополнительно:**")
                st.write(f"• Множитель Лагранжа: {w_rgt_init}")
                st.write(f"• Оптимизатор: {optimizer_type}")
                if use_lr_scheduler:
                    st.write(f"• LR планировщик: {lr_scheduler_type}")
                if use_early_stopping:
                    st.write("• Ранний останов: Включен")
                if grad_clip > 0:
                    st.write(f"• Обрезка градиентов: {grad_clip}")
    
    # Валидация настроек
    def validate_settings():
        errors = []
        warnings = []
        
        # Проверка основных параметров
        if max_iter < 100:
            errors.append("Количество итераций должно быть не менее 100")
        if batch_size < 8:
            errors.append("Размер батча должен быть не менее 8")
        if lr <= 0 or lr > 0.1:
            warnings.append("Скорость обучения выходит за типичные границы (0.0001-0.01)")
        
        # Проверка множителей Лагранжа
        if w_rgt_min >= w_rgt_max:
            errors.append("Минимальный множитель Лагранжа должен быть меньше максимального")
        if w_rgt_init < w_rgt_min or w_rgt_init > w_rgt_max:
            warnings.append("Начальный множитель Лагранжа выходит за установленные границы")
        
        # Проверка архитектуры
        if num_a_layers < 1 or num_p_layers < 1:
            errors.append("Количество слоев должно быть не менее 1")
        if num_a_hidden < 8 or num_p_hidden < 8:
            warnings.append("Малое количество скрытых единиц может ухудшить качество модели")
        
        # Проверка дополнительных параметров
        if dropout_rate >= 0.8:
            warnings.append("Высокое значение dropout может замедлить обучение")
        if use_early_stopping and early_stopping_patience and early_stopping_patience > max_iter // 10:
            warnings.append("Терпение раннего останова слишком велико относительно количества итераций")
        
        return errors, warnings
    
    # Кнопка для запуска обучения
    if st.button("🚀 Запустить обучение", type="primary"):
        # Проверяем настройки
        errors, warnings = validate_settings()
        
        if errors:
            st.error("❌ Обнаружены ошибки в настройках:")
            for error in errors:
                st.error(f"• {error}")
            st.stop()
        
        if warnings:
            st.warning("⚠️ Предупреждения о настройках:")
            for warning in warnings:
                st.warning(f"• {warning}")
            
            if not st.checkbox("Я понимаю предупреждения и хочу продолжить"):
                st.stop()
        
        st.info("Подготовка к обучению модели...")
        
        # Создаем прогресс-бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.container()
        
        # Контейнеры для графиков
        chart_col1, chart_col2 = st.columns(2)
        revenue_chart = chart_col1.empty()
        regret_chart = chart_col2.empty()
        
        try:
            # Проверяем наличие выбранной конфигурации
            if selected_config in settings:
                # Получаем настройки для выбранной конфигурации
                config_settings = settings[selected_config]
                
                # Импортируем файл конфигурации
                cfg_module = __import__(f"cfgs.{config_settings['cfg']}", fromlist=["cfg"])
                cfg = cfg_module.cfg
                
                # Обновляем параметры конфигурации
                cfg.train.max_iter = max_iter
                cfg.train.batch_size = batch_size
                cfg.train.learning_rate = lr
                cfg.train.w_rgt_init_val = w_rgt_init
                cfg.train.update_rate = update_rate
                cfg.train.gd_iter = gd_iter
                
                # Применяем дополнительные настройки множителей Лагранжа
                if hasattr(cfg.train, 'w_rgt_max_val'):
                    cfg.train.w_rgt_max_val = w_rgt_max
                if hasattr(cfg.train, 'w_rgt_min_val'):
                    cfg.train.w_rgt_min_val = w_rgt_min
                if hasattr(cfg.train, 'lagrange_update_freq'):
                    cfg.train.lagrange_update_freq = lagrange_update_freq
                if hasattr(cfg.train, 'rgt_start_iter'):
                    cfg.train.rgt_start_iter = rgt_start_iter
                
                # Применяем настройки оптимизатора
                if hasattr(cfg.train, 'optimizer_type'):
                    cfg.train.optimizer_type = optimizer_type
                if hasattr(cfg.train, 'weight_decay'):
                    cfg.train.weight_decay = weight_decay
                if hasattr(cfg.train, 'momentum'):
                    cfg.train.momentum = momentum
                if hasattr(cfg.train, 'beta1'):
                    cfg.train.beta1 = beta1
                if hasattr(cfg.train, 'beta2'):
                    cfg.train.beta2 = beta2
                if hasattr(cfg.train, 'eps'):
                    cfg.train.eps = eps
                
                # Применяем настройки планировщика скорости обучения
                if hasattr(cfg.train, 'use_lr_scheduler'):
                    cfg.train.use_lr_scheduler = use_lr_scheduler
                if use_lr_scheduler and hasattr(cfg.train, 'lr_scheduler_type'):
                    cfg.train.lr_scheduler_type = lr_scheduler_type
                    if lr_step_size is not None and hasattr(cfg.train, 'lr_step_size'):
                        cfg.train.lr_step_size = lr_step_size
                    if lr_gamma is not None and hasattr(cfg.train, 'lr_gamma'):
                        cfg.train.lr_gamma = lr_gamma
                    if hasattr(cfg.train, 'T_max') and lr_scheduler_type == "CosineAnnealingLR":
                        cfg.train.T_max = T_max
                    if hasattr(cfg.train, 'lr_patience') and lr_scheduler_type == "ReduceLROnPlateau":
                        cfg.train.lr_patience = lr_patience
                    if lr_min is not None and hasattr(cfg.train, 'lr_min'):
                        cfg.train.lr_min = lr_min
                
                # Применяем настройки архитектуры сети
                if hasattr(cfg.net, 'dropout_rate'):
                    cfg.net.dropout_rate = dropout_rate
                if hasattr(cfg.net, 'batch_norm'):
                    cfg.net.batch_norm = batch_norm
                if hasattr(cfg.net, 'num_a_layers'):
                    cfg.net.num_a_layers = num_a_layers
                if hasattr(cfg.net, 'num_p_layers'):
                    cfg.net.num_p_layers = num_p_layers
                if hasattr(cfg.net, 'num_a_hidden_units'):
                    cfg.net.num_a_hidden_units = num_a_hidden
                if hasattr(cfg.net, 'num_p_hidden_units'):
                    cfg.net.num_p_hidden_units = num_p_hidden
                if hasattr(cfg.net, 'activation'):
                    cfg.net.activation = activation
                
                # Применяем настройки мониторинга
                if hasattr(cfg.train, 'save_frequency'):
                    cfg.train.save_frequency = save_frequency
                if hasattr(cfg.train, 'validation_frequency'):
                    cfg.train.validation_frequency = validation_frequency
                if hasattr(cfg.train, 'log_frequency'):
                    cfg.train.log_frequency = log_frequency
                if hasattr(cfg.train, 'verbose_training'):
                    cfg.train.verbose_training = verbose_training
                
                # Применяем настройки раннего останова
                if hasattr(cfg.train, 'use_early_stopping'):
                    cfg.train.use_early_stopping = use_early_stopping
                if use_early_stopping:
                    if hasattr(cfg.train, 'early_stopping_patience'):
                        cfg.train.early_stopping_patience = early_stopping_patience
                    if hasattr(cfg.train, 'early_stopping_min_delta'):
                        cfg.train.early_stopping_min_delta = early_stopping_min_delta
                
                # Применяем дополнительные настройки
                if hasattr(cfg.train, 'grad_clip'):
                    cfg.train.grad_clip = grad_clip
                if hasattr(cfg.train, 'warmup_steps'):
                    cfg.train.warmup_steps = warmup_steps
                if hasattr(cfg.train, 'seed'):
                    cfg.train.seed = seed
                if hasattr(cfg.train, 'num_workers'):
                    cfg.train.num_workers = num_workers
                
                # Импортируем класс нейронной сети
                net_module = __import__(f"nets.{config_settings['net']}", fromlist=["Net"])
                net = net_module.Net(cfg)
                
                # Импортируем генератор данных
                gen_module = __import__(f"data.{config_settings['generator']}", fromlist=["Generator"])
                train_gen = gen_module.Generator(cfg, 'train')
                val_gen = gen_module.Generator(cfg, 'val')
                
                # Импортируем функцию ограничения значений
                clip_ops = __import__("clip_ops.clip_ops", fromlist=[config_settings['clip_op']])
                clip_op = getattr(clip_ops, config_settings['clip_op'])
                
                # Импортируем класс Trainer
                from trainer import Trainer
                
                # Списки для хранения метрик обучения
                revenue_history = []
                regret_history = []
                
                # Функция обратного вызова для отслеживания прогресса
                def training_callback(iter, metrics, time_elapsed):
                    # Обновляем прогресс-бар
                    progress = min(iter / max_iter, 1.0)
                    progress_bar.progress(progress)
                    
                    # Распаковываем метрики
                    revenue, regret, reg_loss, lag_loss, net_loss, w_rgt_mean, update_rate_val = metrics
                    
                    # Вычисляем оценку времени завершения
                    if iter > 0:
                        avg_time_per_iter = time_elapsed / iter
                        remaining_iters = max_iter - iter
                        estimated_remaining_time = avg_time_per_iter * remaining_iters
                        
                        # Форматируем время
                        def format_time(seconds):
                            if seconds < 60:
                                return f"{seconds:.0f}с"
                            elif seconds < 3600:
                                return f"{seconds/60:.1f}м"
                            else:
                                return f"{seconds/3600:.1f}ч"
                        
                        time_info = f"Время: {format_time(time_elapsed)} | Осталось: {format_time(estimated_remaining_time)}"
                    else:
                        time_info = f"Время: {time_elapsed:.2f}с"
                    
                    # Обновляем статус
                    status_text.text(f"Итерация {iter}/{max_iter} ({progress*100:.1f}%) | {time_info}")
                    
                    # Добавляем текущие метрики в историю
                    revenue_history.append(revenue)
                    regret_history.append(regret)
                    
                    # Обновляем графики каждые 10 итераций
                    if iter % 10 == 0:
                        with metrics_container:
                            cols = st.columns(4)
                            cols[0].metric("Выручка", f"{revenue:.4f}")
                            cols[1].metric("Regret", f"{regret:.4f}")
                            cols[2].metric("Множитель Лагранжа", f"{w_rgt_mean:.4f}")
                            cols[3].metric("Потери", f"{net_loss:.4f}")
                        
                        # Обновляем графики
                        x_axis = list(range(1, len(revenue_history) + 1))
                        
                        fig1, ax1 = plt.subplots()
                        ax1.plot(x_axis, revenue_history)
                        ax1.set_xlabel('Итерация')
                        ax1.set_ylabel('Выручка')
                        ax1.set_title('Динамика выручки')
                        revenue_chart.pyplot(fig1)
                        
                        fig2, ax2 = plt.subplots()
                        ax2.plot(x_axis, regret_history)
                        ax2.set_xlabel('Итерация')
                        ax2.set_ylabel('Regret')
                        ax2.set_title('Динамика Regret')
                        regret_chart.pyplot(fig2)
                
                # Создаем экземпляр тренера с функцией обратного вызова
                trainer = Trainer(cfg, 'train', net, clip_op, callback=training_callback)
                
                # Запускаем обучение
                st.info(f"Начало обучения модели для конфигурации {selected_config}...")
                trainer.train((train_gen, val_gen))
                
                # После завершения обучения
                st.success(f"Обучение завершено! Модель сохранена в {cfg.dir_name}")
                
            else:
                st.error(f"Конфигурация {selected_config} не найдена в настройках")
                
        except Exception as e:
            st.error(f"Ошибка при обучении модели: {str(e)}")
            st.exception(e)

elif mode == "Тестирование":
    st.markdown('<div class="sub-header">Тестирование модели</div>', unsafe_allow_html=True)
    
    # Выбор конфигурации аукциона
    selected_config = st.selectbox("Выберите конфигурацию аукциона", auction_configs)
    
    if selected_config:
        model_iterations = find_model_iterations(selected_config)
        
        if model_iterations:
            selected_iter = st.selectbox("Выберите итерацию модели", model_iterations, format_func=lambda x: f"Итерация {x}")
            
            # Настройки тестирования
            col1, col2 = st.columns(2)
            with col1:
                num_batches = st.number_input("Количество батчей для тестирования", min_value=1, max_value=100, value=10)
                batch_size = st.number_input("Размер батча", min_value=16, max_value=256, value=100, step=16)
            
            with col2:
                gd_iter = st.number_input("Итерации для вычисления misreports", min_value=10, max_value=2000, value=200, step=10)
                gd_lr = st.number_input("Скорость обучения для misreports", min_value=0.001, max_value=0.5, value=0.1, step=0.01)
            
            # Кнопка для запуска тестирования
            if st.button("Запустить тестирование"):
                st.info("Подготовка к тестированию модели...")
                
                try:
                    # Динамически импортируем необходимые модули на основе выбранной конфигурации
                    if selected_config in settings:
                        # Получаем настройки для выбранной конфигурации
                        config_settings = settings[selected_config]
                        
                        # Импортируем файл конфигурации
                        cfg_module = __import__(f"cfgs.{config_settings['cfg']}", fromlist=["cfg"])
                        cfg = cfg_module.cfg
                        
                        # Обновляем параметры конфигурации
                        cfg.test.restore_iter = selected_iter
                        cfg.test.batch_size = batch_size
                        cfg.test.num_batches = num_batches
                        cfg.test.gd_iter = gd_iter
                        cfg.test.gd_lr = gd_lr
                        
                        # Создаем путь к модели
                        model_path = os.path.join(cfg.dir_name, f'model-{cfg.test.restore_iter}.pt')
                        
                        # Загружаем модель с правильной архитектурой
                        try:
                            net, cfg_updated, _ = load_cached_model(selected_config, selected_iter, model_path)
                            # Обновляем конфигурацию с правильными параметрами
                            cfg = cfg_updated
                            cfg.test.restore_iter = selected_iter
                            cfg.test.batch_size = batch_size
                            cfg.test.num_batches = num_batches
                            cfg.test.gd_iter = gd_iter
                            cfg.test.gd_lr = gd_lr
                        except Exception as load_error:
                            st.error(f"Ошибка при загрузке модели: {str(load_error)}")
                            st.exception(load_error)
                            st.stop()
                        
                        # Импортируем генератор данных
                        gen_module = __import__(f"data.{config_settings['generator']}", fromlist=["Generator"])
                        test_gen = gen_module.Generator(cfg, 'test')
                        
                        # Импортируем функцию ограничения значений
                        clip_ops = __import__("clip_ops.clip_ops", fromlist=[config_settings['clip_op']])
                        clip_op = getattr(clip_ops, config_settings['clip_op'])
                        
                        # Импортируем класс Trainer
                        from trainer import Trainer
                        
                        # Создаем контейнеры для вывода результатов
                        progress_text = st.empty()
                        results_container = st.container()
                        
                        # Создаем экземпляр тренера в режиме тестирования
                        trainer = Trainer(cfg, 'test', net, clip_op)
                        trainer.net.eval()
                        
                        # Запускаем тестирование
                        progress_text.text("Тестирование модели...")
                        trainer.test(test_gen)
                        progress_text.text("Тестирование завершено!")
                        
                        # Загружаем результаты тестирования из лог-файла
                        log_suffix = f"_iter_{cfg.test.restore_iter}_m_{cfg.test.num_misreports}_gd_{cfg.test.gd_iter}"
                        log_file = os.path.join(cfg.dir_name, f"test{log_suffix}.txt")
                        
                        if os.path.exists(log_file):
                            with open(log_file, 'r') as f:
                                log_content = f.read()
                            
                            # Ищем строки с результатами
                            import re
                            results = {}
                            
                            # Ищем основные метрики
                            metrics_pattern = r"(\w+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)"
                            matches = re.findall(metrics_pattern, log_content)
                            
                            for match in matches:
                                metric_name, mean, std = match
                                results[metric_name] = {"mean": float(mean), "std": float(std)}
                            
                            # Отображаем результаты
                            with results_container:
                                st.markdown('<div class="sub-header">Результаты тестирования</div>', unsafe_allow_html=True)
                                
                                metric_cols = st.columns(len(results))
                                for i, (metric, values) in enumerate(results.items()):
                                    with metric_cols[i]:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-value">{values['mean']:.6f}</div>
                                            <div class="metric-label">{metric}</div>
                                            <div class="metric-label">Std: {values['std']:.6f}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                # Отображаем содержимое лог-файла
                                with st.expander("Подробные результаты тестирования"):
                                    st.text(log_content)
                        else:
                            st.warning(f"Файл с результатами тестирования не найден: {log_file}")
                    
                except Exception as e:
                    st.error(f"Ошибка при тестировании модели: {str(e)}")
                    st.exception(e)
        else:
            st.warning(f"Не найдено обученных моделей для конфигурации {selected_config}")

elif mode == "Визуализация результатов":
    st.markdown('<div class="sub-header">Визуализация результатов</div>', unsafe_allow_html=True)
    
    # Значения оптимального механизма для различных конфигураций
    optimal_mechanism_values = {
        'additive_1x2_uniform': 0.550,
        'unit_1x2_uniform_23': 2.137,
        'additive_3x10_uniform': 5.541,
        'additive_5x10_uniform': 6.778,
        'additive_1x2_uniform_416_47': 9.781,
        'additive_1x2_uniform_triangle': 0.388,
        'unit_1x2_uniform': 0.384,
        'additive_2x2_uniform': 0.878
    }
    
    # Выбор конфигурации аукциона
    available_configs_with_optimal = [c for c in auction_configs if c in optimal_mechanism_values]
    
    if not available_configs_with_optimal:
        st.warning("Нет доступных конфигураций с известными значениями оптимального механизма.")
        st.info("Для отображения результатов необходимо сначала обучить модели для следующих конфигураций:")
        for config in optimal_mechanism_values.keys():
            st.write(f"• {config}")
        st.stop()
    
    selected_config = st.selectbox("Выберите конфигурацию аукциона", available_configs_with_optimal)
    
    if selected_config:
        # Находим директорию с экспериментами для выбранной конфигурации
        if selected_config in settings:
            config_settings = settings[selected_config]
            cfg_module = __import__(f"cfgs.{config_settings['cfg']}", fromlist=["cfg"])
            cfg = cfg_module.cfg
            
            experiment_dir = Path(cfg.dir_name)
            if experiment_dir.exists():
                # Ищем логи обучения
                train_logs = list(experiment_dir.glob("train*.txt"))
                
                if train_logs:
                    st.subheader("Динамика обучения vs Оптимальный механизм")
                    
                    # Считываем лог-файл
                    def parse_train_log(log_file):
                        revenues = []
                        regrets = []
                        iterations = []
                        
                        with open(log_file, 'r') as f:
                            for line in f:
                                # Ищем строки с метриками обучения
                                if "TRAIN (" in line and "Rev:" in line and "Rgt:" in line:
                                    parts = line.strip().split("|")
                                    
                                    try:
                                        # Извлекаем итерацию
                                        iter_part = parts[0].strip()
                                        iter_num = int(iter_part[iter_part.find("(")+1:iter_part.find(")")])
                                        
                                        # Извлекаем выручку
                                        rev_part = parts[1].strip()
                                        revenue = float(rev_part[rev_part.find(":")+1:].strip())
                                        
                                        # Извлекаем regret
                                        rgt_part = parts[2].strip()
                                        regret = float(rgt_part[rgt_part.find(":")+1:].strip())
                                        
                                        iterations.append(iter_num)
                                        revenues.append(revenue)
                                        regrets.append(regret)
                                    except (ValueError, IndexError):
                                        # Пропускаем строки с неправильным форматом
                                        continue
                        
                        return iterations, revenues, regrets
                    
                    try:
                        # Берем самый последний лог
                        latest_log = sorted(train_logs)[-1]
                        iterations, revenues, regrets = parse_train_log(latest_log)
                        
                        if iterations:
                            # Получаем значение оптимального механизма
                            optimal_value = optimal_mechanism_values[selected_config]
                            
                            # Создаем графики в стиле из изображения
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                                        
                                                        # График выручки (Test Revenue)
                            ax1.plot(iterations, revenues, 'b-', linewidth=2, label='RegretNet')
                            ax1.axhline(y=optimal_value, color='g', linestyle='--', linewidth=2, label='Optimal Mechanism')
                            ax1.set_xlabel('Iterations')
                            ax1.set_ylabel('Test Revenue')
                            ax1.set_title('Test Revenue')
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            # Устанавливаем разумные пределы для оси Y
                            y_min = min(min(revenues), optimal_value) * 0.95
                            y_max = max(max(revenues), optimal_value) * 1.05
                            ax1.set_ylim(y_min, y_max)
                            
                            # График regret (Test Regret)
                            ax2.plot(iterations, regrets, 'b-', linewidth=2, label='RegretNet')
                            ax2.axhline(y=0, color='g', linestyle='--', linewidth=2, label='Optimal Mechanism')
                            ax2.set_xlabel('Iterations')
                            ax2.set_ylabel('Test Regret')
                            ax2.set_title('Test Regret')
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)
                            
                            # Устанавливаем пределы для regret (обычно от 0 до некоторого максимума)
                            ax2.set_ylim(0, max(regrets) * 1.1 if regrets else 0.01)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                             
                             # Показываем итоговую статистику
                            col1, col2, col3 = st.columns(3)
                             
                            with col1:
                                final_revenue = revenues[-1] if revenues else 0
                                st.metric("Финальная выручка RegretNet", f"{final_revenue:.4f}")
                                st.metric("Оптимальный механизм", f"{optimal_value:.4f}")
                                 
                            with col2:
                                revenue_gap = abs(final_revenue - optimal_value) if revenues else 0
                                revenue_ratio = (final_revenue / optimal_value * 100) if revenues and optimal_value > 0 else 0
                                st.metric("Разрыв в выручке", f"{revenue_gap:.4f}")
                                st.metric("% от оптимума", f"{revenue_ratio:.2f}%")
                                
                            with col3:
                                final_regret = regrets[-1] if regrets else 0
                                st.metric("Финальный regret", f"{final_regret:.6f}")
                                if final_regret < 0.001:
                                    st.success("✅ Очень низкий regret!")
                                elif final_regret < 0.01:
                                    st.info("✓ Приемлемый regret")
                                else:
                                    st.warning("⚠️ Высокий regret")
                             
                             # Дополнительная информация
                            with st.expander("📊 Детали обучения"):
                                # Создаем DataFrame для удобства работы с данными
                                df = pd.DataFrame({
                                    'Итерация': iterations,
                                    'Выручка': revenues,
                                    'Regret': regrets,
                                    'Разрыв_с_оптимумом': [abs(r - optimal_value) for r in revenues]
                                })
                                 
                                st.dataframe(df)
                                 
                                # Добавляем возможность скачать данные
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="💾 Скачать данные в CSV",
                                    data=csv,
                                    file_name=f"{selected_config}_training_results.csv",
                                    mime="text/csv"
                                )
                                
                            # Информация о конфигурации
                            with st.expander("ℹ️ Информация о конфигурации"):
                                st.write(f"**Конфигурация:** {selected_config}")
                                st.write(f"**Оптимальное значение:** {optimal_value}")
                                st.write(f"**Всего итераций обучения:** {max(iterations) if iterations else 0}")
                                st.write(f"**Лог-файл:** {latest_log}")
                                
                                # Анализ сходимости
                                if len(revenues) > 10:
                                    # Проверяем, сходится ли модель к оптимуму
                                    last_10_revenues = revenues[-10:]
                                    avg_last_10 = np.mean(last_10_revenues)
                                    convergence_gap = abs(avg_last_10 - optimal_value) / optimal_value * 100
                                    
                                    st.write(f"**Средняя выручка последних 10 итераций:** {avg_last_10:.4f}")
                                    st.write(f"**Отклонение от оптимума:** {convergence_gap:.2f}%")
                                    
                                    if convergence_gap < 1:
                                        st.success("🎯 Отличная сходимость к оптимуму!")
                                    elif convergence_gap < 5:
                                        st.info("✓ Хорошая сходимость")
                                    else:
                                        st.warning("⚠️ Модель далека от оптимума")
                            
                        else:
                            st.warning("Не удалось извлечь данные о динамике обучения из лог-файла")
                    
                    except Exception as e:
                        st.error(f"Ошибка при анализе лог-файла: {str(e)}")
                        st.exception(e)
                else:
                    st.info("Логи обучения не найдены. Сначала обучите модель.")
            else:
                st.warning(f"Директория с экспериментами не найдена: {experiment_dir}")
    else:
        st.info("Выберите конфигурацию аукциона для просмотра результатов")

elif mode == "Визуализация аукционов":
    st.markdown('<div class="sub-header">Визуализация механизмов аукционов</div>', unsafe_allow_html=True)
    
    # Выбор конфигурации аукциона  
    available_configs = [c for c in auction_configs if ('1x' in c or '2x2' in c)]
    if not available_configs:
        st.warning("Нет доступных конфигураций для визуализации. Убедитесь, что модели обучены.")
        st.stop()
    
    selected_config = st.selectbox("Выберите конфигурацию аукциона", available_configs)
    
    if selected_config:
        # Поиск доступных моделей для выбранной конфигурации
        model_iterations = find_model_iterations(selected_config)
        
        if model_iterations:
            selected_iter = st.selectbox("Выберите итерацию модели", 
                                        model_iterations, 
                                        format_func=lambda x: f"Итерация {x}")
            
            # Настройки визуализации
            st.markdown("### Настройки визуализации")
            resolution = st.slider("Разрешение сетки", min_value=20, max_value=100, value=50, step=10)
            
            if st.button("Визуализировать аукцион"):
                st.info("Подготовка визуализации...")
                
                try:
                    # Получаем настройки для выбранной конфигурации
                    from run_train import settings
                    config_settings = settings[selected_config]
                    
                    # Импортируем файл конфигурации для получения пути
                    cfg_module = __import__(f"cfgs.{config_settings['cfg']}", fromlist=["cfg"])
                    temp_cfg = cfg_module.cfg
                    
                    # Путь к модели
                    model_path = os.path.join(temp_cfg.dir_name, f'model-{selected_iter}.pt')
                    
                    # Проверяем существование файла модели
                    if not os.path.exists(model_path):
                        st.error(f"Файл модели не найден: {model_path}")
                        st.stop()
                    
                    # Загружаем модель с кешированием
                    try:
                        with st.spinner("Загружаем модель..."):
                            net, cfg, config_settings = load_cached_model(selected_config, selected_iter, model_path)
                        st.success(f"✅ Модель успешно загружена: {cfg.net.num_a_layers} слоев распределения ({cfg.net.num_a_hidden_units} ед.), {cfg.net.num_p_layers} слоев платежей ({cfg.net.num_p_hidden_units} ед.)")
                    except Exception as e:
                        # Fallback к методу подбора конфигурации (без промежуточных сообщений)
                        with st.spinner("Автоматически подбираем архитектуру модели..."):
                            # Загружаем checkpoint напрямую
                            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                            state_dict = checkpoint['model_state_dict']
                            
                            # Импортируем класс нейронной сети
                            net_module = __import__(f"nets.{config_settings['net']}", fromlist=["Net"])
                            
                                        # Список часто используемых конфигураций для попыток
                            common_configs = [
                                {'num_a_layers': 2, 'num_p_layers': 2, 'num_a_hidden': 50, 'num_p_hidden': 50},
                                {'num_a_layers': 2, 'num_p_layers': 2, 'num_a_hidden': 100, 'num_p_hidden': 100},
                                {'num_a_layers': 3, 'num_p_layers': 3, 'num_a_hidden': 50, 'num_p_hidden': 50},
                                {'num_a_layers': 3, 'num_p_layers': 3, 'num_a_hidden': 100, 'num_p_hidden': 100},
                                {'num_a_layers': 1, 'num_p_layers': 1, 'num_a_hidden': 50, 'num_p_hidden': 50},
                                {'num_a_layers': 4, 'num_p_layers': 4, 'num_a_hidden': 50, 'num_p_hidden': 50},
                                {'num_a_layers': 2, 'num_p_layers': 2, 'num_a_hidden': 200, 'num_p_hidden': 200},
                                {'num_a_layers': 2, 'num_p_layers': 2, 'num_a_hidden': 128, 'num_p_hidden': 128},
                                # Добавляем конфигурации с большим количеством слоев
                                {'num_a_layers': 5, 'num_p_layers': 5, 'num_a_hidden': 50, 'num_p_hidden': 50},
                                {'num_a_layers': 6, 'num_p_layers': 6, 'num_a_hidden': 50, 'num_p_hidden': 50},
                                {'num_a_layers': 6, 'num_p_layers': 6, 'num_a_hidden': 100, 'num_p_hidden': 100},
                                {'num_a_layers': 7, 'num_p_layers': 7, 'num_a_hidden': 50, 'num_p_hidden': 50},
                                {'num_a_layers': 8, 'num_p_layers': 8, 'num_a_hidden': 50, 'num_p_hidden': 50},
                                # Добавляем варианты с разными размерами скрытых слоев
                                {'num_a_layers': 4, 'num_p_layers': 4, 'num_a_hidden': 100, 'num_p_hidden': 100},
                                {'num_a_layers': 5, 'num_p_layers': 5, 'num_a_hidden': 100, 'num_p_hidden': 100},
                                {'num_a_layers': 3, 'num_p_layers': 3, 'num_a_hidden': 200, 'num_p_hidden': 200},
                                {'num_a_layers': 4, 'num_p_layers': 4, 'num_a_hidden': 200, 'num_p_hidden': 200},
                            ]
                            
                            net = None
                            cfg = temp_cfg
                            cfg.test.restore_iter = selected_iter
                            successful_config = None
                            
                            # Пробуем все конфигурации молча
                            for i, config_attempt in enumerate(common_configs):
                                try:
                                    # Обновляем конфигурацию
                                    cfg.net.num_a_layers = config_attempt['num_a_layers']
                                    cfg.net.num_p_layers = config_attempt['num_p_layers']
                                    cfg.net.num_a_hidden_units = config_attempt['num_a_hidden']
                                    cfg.net.num_p_hidden_units = config_attempt['num_p_hidden']
                                    
                                    # Создаем новую сеть
                                    net = net_module.Net(cfg)
                                    
                                    # Пытаемся загрузить
                                    net.load_state_dict(state_dict)
                                    net.eval()
                                    
                                    successful_config = config_attempt
                                    break
                                    
                                except Exception:
                                    # Тихо продолжаем к следующей конфигурации
                                    continue
                            
                            # Показываем результат только в конце
                            if net is not None:
                                st.success(f"✅ Модель успешно загружена с автоматически подобранной архитектурой: {successful_config['num_a_layers']} слоев распределения ({successful_config['num_a_hidden']} ед.), {successful_config['num_p_layers']} слоев платежей ({successful_config['num_p_hidden']} ед.)")
                            else:
                                raise Exception("Не удалось загрузить модель ни с одной из стандартных конфигураций")
                    
                    # Импортируем функцию ограничения значений
                    clip_ops = __import__("clip_ops.clip_ops", fromlist=[config_settings['clip_op']])
                    clip_op = getattr(clip_ops, config_settings['clip_op'])
                    
                    # Импортируем класс Trainer
                    from trainer import Trainer
                    
                    # Создаем экземпляр тренера в режиме тестирования
                    trainer = Trainer(cfg, 'test', net, clip_op)
                    trainer.net.eval()
                    
                    # Создаем сетку точек для визуализации
                    D = resolution
                    x = np.linspace(0, 1.0, D)
                    
                    # Для разных типов аукционов могут быть разные входные данные
                    if '1x2' in selected_config:  # 1 покупатель, 2 предмета
                        D = resolution
                        x = np.linspace(0, 1.0, D)
                        X_grid = np.stack([v.flatten() for v in np.meshgrid(x, x)], axis=-1)
                        
                        # Специальная обработка для разных типов распределений
                        if '04_03' in selected_config:
                            # Масштабируем входные данные до [4,3]
                            X_grid[:, 0] = X_grid[:, 0] * 4.0
                            X_grid[:, 1] = X_grid[:, 1] * 3.0
                            x_extent = [0, 4, 0, 3]
                            aspect_ratio = 4/3
                        elif '416_47' in selected_config:
                            # Масштабируем входные данные до [4,16] × [4,7]
                            X_grid[:, 0] = X_grid[:, 0] * 12.0 + 4.0  # [0,1] -> [4,16]
                            X_grid[:, 1] = X_grid[:, 1] * 3.0 + 4.0   # [0,1] -> [4,7]
                            x_extent = [4, 16, 4, 7]
                            aspect_ratio = 12/3  # (16-4)/(7-4) = 12/3 = 4.0
                        elif 'triangle' in selected_config:
                            # Маскируем область где v1+v2 >= 1
                            mask = X_grid.sum(-1) >= 1.0
                            X_grid[mask] = 0.0
                            x_extent = [0, 1, 0, 1]
                            aspect_ratio = 1.0
                        elif '23' in selected_config:
                            # Масштабируем входные данные до [2,3] × [2,3]
                            X_grid[:, 0] = X_grid[:, 0] * 1.0 + 2.0   # [0,1] -> [2,3]
                            X_grid[:, 1] = X_grid[:, 1] * 1.0 + 2.0   # [0,1] -> [2,3]
                            x_extent = [2, 3, 2, 3]
                            aspect_ratio = 1.0  # (3-2)/(3-2) = 1/1 = 1.0
                        else:
                            # Стандартное распределение [0,1]x[0,1]
                            x_extent = [0, 1, 0, 1]
                            aspect_ratio = 1.0
                        
                        X_grid = np.expand_dims(X_grid, 1)  # (D*D, 1, 2)
                        
                        # Получаем выходные данные модели
                        device = next(trainer.net.parameters()).device  # Получаем device из параметров сети
                        X_tensor = torch.from_numpy(X_grid).float().to(device)
                        with torch.no_grad():
                            allocs, payments = trainer.net(X_tensor)
                        allocs = allocs.cpu().numpy()
                        payments = payments.cpu().numpy()
                        
                        # Преобразуем выходные данные для отображения на сетке
                        alloc1 = allocs[:, 0, 0].reshape(D, D)
                        alloc2 = allocs[:, 0, 1].reshape(D, D)
                        payment = payments[:, 0].reshape(D, D)
                        
                        # Создаем 3D графики для визуализации
                        fig = plt.figure(figsize=(18, 6))
                        
                        # График распределения первого предмета
                        ax1 = fig.add_subplot(131, projection='3d')
                        xx, yy = np.meshgrid(x, x)
                        surf1 = ax1.plot_surface(xx, yy, alloc1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
                        ax1.set_xlabel('Оценка предмета 1')
                        ax1.set_ylabel('Оценка предмета 2')
                        ax1.set_zlabel('Вероятность получения предмета 1')
                        ax1.set_title('Распределение предмета 1')
                        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
                        
                        # График распределения второго предмета
                        ax2 = fig.add_subplot(132, projection='3d')
                        surf2 = ax2.plot_surface(xx, yy, alloc2, cmap=cm.coolwarm, linewidth=0, antialiased=True)
                        ax2.set_xlabel('Оценка предмета 1')
                        ax2.set_ylabel('Оценка предмета 2')
                        ax2.set_zlabel('Вероятность получения предмета 2')
                        ax2.set_title('Распределение предмета 2')
                        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
                        
                        # График платежей
                        ax3 = fig.add_subplot(133, projection='3d')
                        surf3 = ax3.plot_surface(xx, yy, payment, cmap=cm.coolwarm, linewidth=0, antialiased=True)
                        ax3.set_xlabel('Оценка предмета 1')
                        ax3.set_ylabel('Оценка предмета 2')
                        ax3.set_zlabel('Платеж')
                        ax3.set_title('Функция платежа')
                        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Визуализация через тепловую карту для лучшего понимания границ
                        st.markdown("### Тепловые карты с теоретическими границами")
                        
                        # Определяем границы в зависимости от типа конфигурации
                        plt.rcParams.update({'font.size': 10, 'axes.labelsize': 'x-large'})
                        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                        
                        # Определяем теоретические границы для разных типов аукционов
                        boundary_points_1 = []
                        boundary_points_2 = []
                        
                        if 'unit' in selected_config:
                            # Для unit demand аукционов
                            if '23' in selected_config:
                                # Для unit_1x2_uniform_23 (область [2,3] × [2,3])
                                # Оригинальная сложная граница из visualize_unit_1x2_uniform_23.ipynb
                                x1 = 4.0/3.0 + np.sqrt(4.0 + 3.0/2.0)/3.0
                                points1 = [(3.0 - 1.0/3.0, 3.0), (2.0, 2.0 + 1.0/3.0)]                    
                                points2 = [(2.0, 2 * x1 - 2.0), (2 * x1 - 2.0, 2.0)]                 
                                points3 = [(2.0 + 1.0/3.0, 2.0), (3.0, 3.0 - 1.0/3.0)]
                                # Границы для предмета 1 и 2 одинаковые
                                boundary_points_1 = [points1, points2, points3]
                                boundary_points_2 = [points1, points2, points3]
                            else:
                                # Стандартный unit случай
                                x1_boundary = np.sqrt(3.0) / 3.0
                                boundary_points_1 = [(x1_boundary, 0), (x1_boundary, x1_boundary), (1.0, 1.0)]
                                boundary_points_2 = [(0.0, x1_boundary), (x1_boundary, x1_boundary), (1.0, 1.0)]
                        elif 'additive' in selected_config:
                            if 'triangle' in selected_config:
                                # Для треугольного распределения
                                x1_boundary = np.sqrt(1.0/3.0)
                                boundary_points_1 = [(x1_boundary, 0), (0, x1_boundary)]
                                boundary_points_2 = [(x1_boundary, 0), (0, x1_boundary)]
                            elif '04_03' in selected_config:
                                # Для additive_1x2_uniform_04_03 - масштабируем стандартные границы на [0,4] × [0,3]
                                x1 = (2.0 - np.sqrt(2.0)) / 3.0 * 4.0  # масштабируем на [0,4]
                                x2 = 2.0 / 3.0 * 3.0  # масштабируем на [0,3]
                                boundary_points_1 = [(x1, 3.0), (x1, x2), (x2*4.0/3.0, x1*3.0/4.0), (x2*4.0/3.0, 0)]
                                boundary_points_2 = [(0.0, x2), (x1, x2), (x2*4.0/3.0, x1*3.0/4.0), (4.0, x1*3.0/4.0)]
                            elif '416_47' in selected_config:
                                # Для additive_1x2_uniform_416_47 - границы из visualize_asymetric_daskalakis.ipynb
                                boundary_points_1 = [(4, 6), (8, 4), (8, 7)]
                                boundary_points_2 = [(4, 6), (8, 4)]
                            else:
                                # Стандартный additive случай
                                x1_boundary = (2.0 - np.sqrt(2.0)) / 3.0
                                x2_boundary = 2.0 / 3.0
                                boundary_points_1 = [(x1_boundary, 1.0), (x1_boundary, x2_boundary), (x2_boundary, x1_boundary), (x2_boundary, 0)]
                                boundary_points_2 = [(0.0, x2_boundary), (x1_boundary, x2_boundary), (x2_boundary, x1_boundary), (1.0, x1_boundary)]
                        
                        # Подготовка данных для отображения
                        alloc1_display = alloc1.copy()
                        alloc2_display = alloc2.copy()
                        payment_display = payment.copy()
                        
                        # Специальная обработка для треугольного распределения
                        if 'triangle' in selected_config:
                            # Создаем маску для области где v1+v2 >= 1
                            x_coords = np.linspace(0, 1.0, D)
                            XX, YY = np.meshgrid(x_coords, x_coords)
                            triangle_mask = XX + YY >= 1.0
                            
                            # Устанавливаем значения выше максимума для маскированных областей
                            from matplotlib.colors import ListedColormap
                            import matplotlib.pyplot as plt
                            from copy import copy
                            palette = copy(plt.cm.YlOrRd)
                            palette.set_over('w')  # Белый цвет для маскированных областей
                            
                            alloc1_display[triangle_mask] = 10.0
                            alloc2_display[triangle_mask] = 10.0 
                            payment_display[triangle_mask] = 10.0
                            cmap_to_use = palette
                        else:
                            cmap_to_use = 'YlOrRd'
                        
                        # График 1: Распределение предмета 1
                        im1 = axes[0].imshow(alloc1_display[::-1], extent=x_extent, vmin=0.0, vmax=1.0, 
                                           cmap=cmap_to_use, aspect=aspect_ratio)
                        axes[0].set_title('Вероятность получения предмета 1')
                        axes[0].set_xlabel('$v_1$')
                        axes[0].set_ylabel('$v_2$')
                        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
                        
                        # Добавляем теоретические границы
                        if boundary_points_1:
                            if '23' in selected_config and 'unit' in selected_config:
                                # Для unit_1x2_uniform_23 рисуем три отдельные линии
                                for line_points in boundary_points_1:
                                    x_coords = [p[0] for p in line_points]
                                    y_coords = [p[1] for p in line_points]
                                    axes[0].plot(x_coords, y_coords, linewidth=2, linestyle='--', c='black')
                            else:
                                # Для остальных конфигураций - одна линия
                                x_coords = [p[0] for p in boundary_points_1]
                                y_coords = [p[1] for p in boundary_points_1]
                                axes[0].plot(x_coords, y_coords, linewidth=2, linestyle='--', c='black')
                            
                            # Добавляем подписи регионов
                            if 'unit' in selected_config:
                                if '23' in selected_config:
                                    # Подписи из оригинального notebook
                                    axes[0].text(2.2, 2.8, s='0', color='black', fontsize='10', fontweight='bold')
                                    axes[0].text(2.05, 2.05, s='0', color='black', fontsize='10', fontweight='bold')
                                    axes[0].text(2.5, 2.5, s='0.5', color='black', fontsize='10', fontweight='bold')
                                    axes[0].text(2.8, 2.2, s='1', color='black', fontsize='10', fontweight='bold')
                                else:
                                    axes[0].text(0.2, 0.4, s='0', color='black', fontsize='10', fontweight='bold')
                                    axes[0].text(0.8, 0.4, s='1', color='black', fontsize='10', fontweight='bold')
                            elif 'additive' in selected_config:
                                if 'triangle' in selected_config:
                                    axes[0].text(0.15, 0.15, s='0', color='black', fontsize='10', fontweight='bold')
                                    axes[0].text(0.4, 0.4, s='1', color='black', fontsize='10', fontweight='bold')
                                elif '04_03' in selected_config:
                                    # Подписи для области [0,4] × [0,3]
                                    axes[0].text(1.0, 1.0, s='0', color='black', fontsize='10', fontweight='bold')
                                    axes[0].text(2.6, 2.6, s='1', color='black', fontsize='10', fontweight='bold')
                                elif '416_47' in selected_config:
                                    # Подписи из оригинального notebook для [4,16] × [4,7]
                                    axes[0].text(5, 4.5, s='0', color='black', fontsize='10', fontweight='bold')
                                    axes[0].text(5.25, 6, s='0.5', color='black', fontsize='10', fontweight='bold')
                                    axes[0].text(11.5, 5.5, s='1', color='black', fontsize='10', fontweight='bold')
                                else:
                                    axes[0].text(0.25, 0.25, s='0', color='black', fontsize='10', fontweight='bold')
                                    axes[0].text(0.65, 0.65, s='1', color='black', fontsize='10', fontweight='bold')
                        
                        # График 2: Распределение предмета 2
                        im2 = axes[1].imshow(alloc2_display[::-1], extent=x_extent, vmin=0.0, vmax=1.0, 
                                           cmap=cmap_to_use, aspect=aspect_ratio)
                        axes[1].set_title('Вероятность получения предмета 2')
                        axes[1].set_xlabel('$v_1$')
                        axes[1].set_ylabel('$v_2$')
                        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
                        
                        # Добавляем теоретические границы
                        if boundary_points_2:
                            if '23' in selected_config and 'unit' in selected_config:
                                # Для unit_1x2_uniform_23 рисуем три отдельные линии
                                for line_points in boundary_points_2:
                                    x_coords = [p[0] for p in line_points]
                                    y_coords = [p[1] for p in line_points]
                                    axes[1].plot(x_coords, y_coords, linewidth=2, linestyle='--', c='black')
                            else:
                                # Для остальных конфигураций - одна линия
                                x_coords = [p[0] for p in boundary_points_2]
                                y_coords = [p[1] for p in boundary_points_2]
                                axes[1].plot(x_coords, y_coords, linewidth=2, linestyle='--', c='black')
                            
                            # Добавляем подписи регионов
                            if 'unit' in selected_config:
                                if '23' in selected_config:
                                    # Подписи из оригинального notebook для предмета 2
                                    axes[1].text(2.2, 2.8, s='1', color='black', fontsize='10', fontweight='bold')
                                    axes[1].text(2.05, 2.05, s='0', color='black', fontsize='10', fontweight='bold')
                                    axes[1].text(2.5, 2.5, s='0.5', color='black', fontsize='10', fontweight='bold')
                                    axes[1].text(2.8, 2.2, s='0', color='black', fontsize='10', fontweight='bold')
                                else:
                                    axes[1].text(0.6, 0.4, s='0', color='black', fontsize='10', fontweight='bold')
                                    axes[1].text(0.4, 0.8, s='1', color='black', fontsize='10', fontweight='bold')
                            elif 'additive' in selected_config:
                                if 'triangle' in selected_config:
                                    axes[1].text(0.15, 0.15, s='0', color='black', fontsize='10', fontweight='bold')
                                    axes[1].text(0.4, 0.4, s='1', color='black', fontsize='10', fontweight='bold')
                                elif '04_03' in selected_config:
                                    # Подписи для области [0,4] × [0,3] для предмета 2
                                    axes[1].text(1.0, 1.0, s='0', color='black', fontsize='10', fontweight='bold')
                                    axes[1].text(2.6, 2.6, s='1', color='black', fontsize='10', fontweight='bold')
                                elif '416_47' in selected_config:
                                    # Подписи из оригинального notebook для предмета 2
                                    axes[1].text(5, 4.5, s='0', color='black', fontsize='10', fontweight='bold')
                                    axes[1].text(11.5, 5.5, s='1', color='black', fontsize='10', fontweight='bold')
                                else:
                                    axes[1].text(0.25, 0.25, s='0', color='black', fontsize='10', fontweight='bold')
                                    axes[1].text(0.65, 0.65, s='1', color='black', fontsize='10', fontweight='bold')
                        
                        # График 3: Платежи
                        im3 = axes[2].imshow(payment_display[::-1], extent=x_extent, cmap=cmap_to_use, 
                                           aspect=aspect_ratio)
                        axes[2].set_title('Платеж')
                        axes[2].set_xlabel('$v_1$')
                        axes[2].set_ylabel('$v_2$')
                        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Дополнительная информация о конфигурации
                        st.markdown("### Информация о конфигурации")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Тип агента:** {cfg.agent_type}")
                            st.write(f"**Количество агентов:** {cfg.num_agents}")
                            st.write(f"**Количество предметов:** {cfg.num_items}")
                            st.write(f"**Тип распределения:** {cfg.distribution_type}")
                            
                        with col2:
                            st.write(f"**Архитектура загруженной модели:**")
                            st.write(f"- Слои распределения: {cfg.net.num_a_layers}")
                            st.write(f"- Слои платежей: {cfg.net.num_p_layers}")
                            st.write(f"- Скрытые юниты распределения: {cfg.net.num_a_hidden_units}")
                            st.write(f"- Скрытые юниты платежей: {cfg.net.num_p_hidden_units}")
                            st.write(f"- Активация: {cfg.net.activation}")
                            
                            # Показываем итерацию модели
                            st.write(f"**Итерация модели:** {selected_iter}")
                        
                        # Показываем статистику распределений
                        st.markdown("### Статистика распределений")
                        col1, col2, col3 = st.columns(3)
                        
                        # Вычисляем статистику только для валидных областей
                        if 'triangle' in selected_config:
                            # Для треугольного распределения исключаем маскированные области
                            valid_mask = ~triangle_mask
                            valid_alloc1 = alloc1[valid_mask]
                            valid_alloc2 = alloc2[valid_mask]
                            valid_payment = payment[valid_mask]
                        else:
                            valid_alloc1 = alloc1
                            valid_alloc2 = alloc2 
                            valid_payment = payment
                        
                        with col1:
                            st.metric("Среднее распределение предмета 1", f"{np.mean(valid_alloc1):.4f}")
                            st.metric("Максимум предмета 1", f"{np.max(valid_alloc1):.4f}")
                            
                        with col2:
                            st.metric("Среднее распределение предмета 2", f"{np.mean(valid_alloc2):.4f}")
                            st.metric("Максимум предмета 2", f"{np.max(valid_alloc2):.4f}")
                            
                        with col3:
                            st.metric("Средний платеж", f"{np.mean(valid_payment):.4f}")
                            st.metric("Максимальный платеж", f"{np.max(valid_payment):.4f}")
                            
                        # Дополнительная информация о распределении
                        st.markdown("### Информация о распределении")
                        if 'triangle' in selected_config:
                            st.info("**Треугольное распределение**: Ограничено областью где v₁ + v₂ ≤ 1")
                        elif '04_03' in selected_config:
                            st.info("**Прямоугольное распределение**: v₁ ∈ [0,4], v₂ ∈ [0,3]")
                        elif '416_47' in selected_config:
                            st.info("**Прямоугольное распределение**: v₁ ∈ [4,16], v₂ ∈ [4,7]")
                        elif '23' in selected_config:
                            st.info("**Прямоугольное распределение**: v₁ ∈ [2,3], v₂ ∈ [2,3]")
                        else:
                            st.info("**Единичное распределение**: v₁, v₂ ∈ [0,1]")
                        
                    elif '1x10' in selected_config or any(x in selected_config for x in ['1x3', '1x4', '1x5', '1x6', '1x7', '1x8', '1x9', '1x10']):  # 1 покупатель, много предметов
                        st.warning("Визуализация для аукционов с большим количеством предметов ограничена")
                        st.info(f"Для конфигурации {selected_config} сложно создать полную визуализацию из-за высокой размерности данных.")
                        
                        # Можем показать базовую статистику
                        try:
                            # Определяем количество предметов
                            num_items = cfg.num_items
                            st.info(f"Количество предметов в аукционе: {num_items}")
                            
                            # Создаем случайную выборку для демонстрации
                            np.random.seed(42)
                            sample_size = 1000
                            sample_data = np.random.rand(sample_size, 1, num_items)
                            
                            # Получаем выходные данные модели
                            device = next(trainer.net.parameters()).device
                            X_tensor = torch.from_numpy(sample_data).float().to(device)
                            with torch.no_grad():
                                allocs, payments = trainer.net(X_tensor)
                            allocs = allocs.cpu().numpy()
                            payments = payments.cpu().numpy()
                            
                            # Показываем статистику по каждому предмету
                            st.markdown("### Статистика распределений по предметам")
                            cols = st.columns(min(5, num_items))  # Не более 5 колонок
                            
                            for i in range(num_items):
                                col_idx = i % len(cols)
                                with cols[col_idx]:
                                    item_allocs = allocs[:, 0, i]
                                    st.metric(f"Предмет {i+1}", 
                                             f"Средн: {np.mean(item_allocs):.4f}",
                                             f"Макс: {np.max(item_allocs):.4f}")
                            
                            # Показываем общую статистику платежей
                            st.markdown("### Статистика платежей")
                            agent_payments = payments[:, 0]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Средний платеж", f"{np.mean(agent_payments):.4f}")
                            with col2:
                                st.metric("Максимальный платеж", f"{np.max(agent_payments):.4f}")
                            with col3:
                                st.metric("Минимальный платеж", f"{np.min(agent_payments):.4f}")
                            
                            # График распределения платежей
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.hist(agent_payments, bins=50, alpha=0.7, edgecolor='black')
                            ax.set_xlabel('Платеж')
                            ax.set_ylabel('Частота')
                            ax.set_title('Распределение платежей')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"Ошибка при анализе аукциона с множественными предметами: {str(e)}")
                            st.exception(e)
                    
                    elif '2x2' in selected_config:  # 2 покупателя, 2 предмета
                        st.warning("Визуализация для 2x2 аукционов ограничена")
                        st.info("Для 2x2 аукционов визуализация более сложная, так как входные данные имеют размерность 4D (2 агента × 2 предмета)")
                        
                        # Базовая визуализация для 2x2 - фиксируем ставки одного агента
                        st.markdown("### Визуализация при фиксированных ставках агента 2")
                        
                        # Ползунки для фиксации ставок второго агента
                        col1, col2 = st.columns(2)
                        with col1:
                            v2_item1 = st.slider("Ставка агента 2 за предмет 1", 0.0, 1.0, 0.5, 0.1)
                        with col2:
                            v2_item2 = st.slider("Ставка агента 2 за предмет 2", 0.0, 1.0, 0.5, 0.1)
                        
                        if st.button("Визуализировать 2x2 аукцион"):
                            try:
                                D = resolution
                                x = np.linspace(0, 1.0, D)
                                
                                # Создаем сетку для первого агента, фиксируя ставки второго
                                X_grid = []
                                for v1_1 in x:
                                    for v1_2 in x:
                                        # [агент1_предмет1, агент1_предмет2, агент2_предмет1, агент2_предмет2]
                                        X_grid.append([v1_1, v1_2, v2_item1, v2_item2])
                                
                                X_grid = np.array(X_grid).reshape(D*D, 2, 2)  # (batch, agents, items)
                                
                                # Получаем выходные данные модели
                                device = next(trainer.net.parameters()).device
                                X_tensor = torch.from_numpy(X_grid).float().to(device)
                                with torch.no_grad():
                                    allocs, payments = trainer.net(X_tensor)
                                allocs = allocs.cpu().numpy()
                                payments = payments.cpu().numpy()
                                
                                # Преобразуем для визуализации
                                # allocs: (batch, agents, items), payments: (batch, agents)
                                agent1_item1 = allocs[:, 0, 0].reshape(D, D)  # Агент 1 получает предмет 1
                                agent1_item2 = allocs[:, 0, 1].reshape(D, D)  # Агент 1 получает предмет 2
                                agent1_payment = payments[:, 0].reshape(D, D)  # Платеж агента 1
                                
                                # Создаем тепловые карты
                                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                                
                                # График 1: Вероятность получения предмета 1 агентом 1
                                im1 = axes[0].imshow(agent1_item1[::-1], extent=[0, 1, 0, 1], vmin=0.0, vmax=1.0, 
                                                   cmap='YlOrRd')
                                axes[0].set_title('Агент 1: Вероятность получения предмета 1')
                                axes[0].set_xlabel('Ставка агента 1 за предмет 1')
                                axes[0].set_ylabel('Ставка агента 1 за предмет 2')
                                fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
                                
                                # График 2: Вероятность получения предмета 2 агентом 1
                                im2 = axes[1].imshow(agent1_item2[::-1], extent=[0, 1, 0, 1], vmin=0.0, vmax=1.0, 
                                                   cmap='YlOrRd')
                                axes[1].set_title('Агент 1: Вероятность получения предмета 2')
                                axes[1].set_xlabel('Ставка агента 1 за предмет 1')
                                axes[1].set_ylabel('Ставка агента 1 за предмет 2')
                                fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
                                
                                # График 3: Платеж агента 1
                                im3 = axes[2].imshow(agent1_payment[::-1], extent=[0, 1, 0, 1], 
                                                   cmap='YlOrRd')
                                axes[2].set_title('Платеж агента 1')
                                axes[2].set_xlabel('Ставка агента 1 за предмет 1')
                                axes[2].set_ylabel('Ставка агента 1 за предмет 2')
                                fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Показываем статистику
                                st.markdown("### Статистика для агента 1")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Среднее распределение предмета 1", f"{np.mean(agent1_item1):.4f}")
                                    st.metric("Максимум предмета 1", f"{np.max(agent1_item1):.4f}")
                                    
                                with col2:
                                    st.metric("Среднее распределение предмета 2", f"{np.mean(agent1_item2):.4f}")
                                    st.metric("Максимум предмета 2", f"{np.max(agent1_item2):.4f}")
                                    
                                with col3:
                                    st.metric("Средний платеж", f"{np.mean(agent1_payment):.4f}")
                                    st.metric("Максимальный платеж", f"{np.max(agent1_payment):.4f}")
                                
                                st.info(f"Фиксированные ставки агента 2: предмет 1 = {v2_item1:.1f}, предмет 2 = {v2_item2:.1f}")
                                
                            except Exception as e:
                                st.error(f"Ошибка при визуализации 2x2 аукциона: {str(e)}")
                                st.exception(e)
                    
                    else:
                        st.warning(f"Визуализация для конфигурации {selected_config} пока не поддерживается")
                        st.info("Поддерживаемые типы визуализации:")
                        st.write("• 1x2 - аукционы с 1 агентом и 2 предметами")
                        st.write("• 1x10 - аукционы с 1 агентом и множественными предметами") 
                        st.write("• 2x2 - аукционы с 2 агентами и 2 предметами (упрощенная визуализация)")
                        
                        # Показываем базовую информацию о конфигурации
                        st.markdown("### Информация о конфигурации")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Тип агента:** {cfg.agent_type}")
                            st.write(f"**Количество агентов:** {cfg.num_agents}")
                            st.write(f"**Количество предметов:** {cfg.num_items}")
                            st.write(f"**Тип распределения:** {cfg.distribution_type}")
                            
                        with col2:
                            st.write(f"**Архитектура модели:**")
                            st.write(f"- Слои распределения: {cfg.net.num_a_layers}")
                            st.write(f"- Слои платежей: {cfg.net.num_p_layers}")
                            st.write(f"- Скрытые юниты распределения: {cfg.net.num_a_hidden_units}")
                            st.write(f"- Скрытые юниты платежей: {cfg.net.num_p_hidden_units}")
                            st.write(f"- Активация: {cfg.net.activation}")
                    
                except Exception as e:
                    st.error(f"Ошибка при визуализации аукциона: {str(e)}")
                    st.exception(e)
        else:
            st.warning(f"Не найдено обученных моделей для конфигурации {selected_config}")
    else:
        st.info("Выберите конфигурацию аукциона для визуализации")

# Добавление Streamlit в requirements.txt
import subprocess
try:
    with open("requirements.txt", "r") as f:
        requirements = f.read()
    
    if "streamlit" not in requirements:
        with open("requirements.txt", "a") as f:
            f.write("\nstreamlit>=1.15.0\n")
except Exception as e:
    st.warning(f"Ошибка при обновлении requirements.txt: {str(e)}") 