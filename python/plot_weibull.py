#!/usr/bin/env python3
"""
Визуализация результатов MLE/MLS для распределения Вейбулла
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.special
import sys

def read_results(filename):
    """Чтение результатов из файла"""
    params = {}
    data = []
    censored = []
    
    with open(filename, 'r') as f:
        section = None
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if '# Данные' in line:
                    section = 'data'
                continue
            
            if section == 'data':
                parts = line.split()
                if len(parts) == 2:
                    data.append(float(parts[0]))
                    censored.append(int(parts[1]))
            else:
                parts = line.split()
                if len(parts) == 2:
                    params[parts[0]] = float(parts[1])
    
    return params, np.array(data), np.array(censored)

def plot_mle_weibull(params, data, censored, output_file):
    """Построение графика на вероятностной бумаге Вейбулла для MLE"""
    scale = params['parameter_1']  # λ (lambda)
    shape = params['parameter_2']  # k (форма)

    # Увеличиваем размер для лучшего качества при масштабировании
    fig, ax = plt.subplots(figsize=(14, 9), dpi=100)
    fig.suptitle('Weibull MLE', fontsize=14, fontweight='bold')

    complete_data = data[censored == 0]
    complete_data = complete_data[complete_data > 0]  # Убираем неположительные значения
    n = len(complete_data)

    # Сортируем данные для получения порядковых статистик
    sorted_data = np.sort(complete_data)

    # Вычисляем эмпирические вероятности
    i = np.arange(1, n + 1)
    empirical_probs = (i - 0.375) / (n + 0.25)

    # Для Вейбулла: F(x) = 1 - exp(-(x/λ)^k)
    # Линеаризация: ln(-ln(1-F)) = k·ln(x) - k·ln(λ)
    # y = ln(-ln(1-F)), x = ln(значение)

    # Эмпирические точки в преобразованных координатах
    y_empirical = np.log(-np.log(1 - empirical_probs))

    # 1. Точки данных (эмпирические)
    ax.scatter(sorted_data, y_empirical,
              color='darkblue', s=80, alpha=0.8,
              edgecolors='black', linewidths=1,
              label='Data points', zorder=3)

    # 2. Апроксимирующая прямая (теоретическое распределение)
    # y = k·ln(x) - k·ln(λ)
    # Ограничиваем диапазон только данными
    x_line = np.linspace(sorted_data.min(), sorted_data.max(), 200)
    x_line = np.maximum(x_line, 0.001)  # Избегаем логарифма от неположительных
    y_line = shape * np.log(x_line) - shape * np.log(scale)
    ax.plot(x_line, y_line, 'r-', linewidth=2.5,
            label='Fitted line', zorder=2)

    # 3. Доверительные границы (отклонения) - плавные дуги
    # Вычисляем доверительные границы вдоль теоретической линии
    # Ограничиваем область построения, чтобы избежать артефактов на краях

    # Теоретические вероятности для x_line
    probs_line = stats.weibull_min.cdf(x_line, shape, scale=scale)

    # Фильтруем только разумный диапазон вероятностей (избегаем краев)
    valid_mask = (probs_line > 0.01) & (probs_line < 0.99)
    x_conf = x_line[valid_mask]
    probs_conf = probs_line[valid_mask]

    # Для Вейбулла используем дельта-метод
    # y = ln(-ln(1-F))
    # dy/dF = 1/((1-F)*ln(1-F))
    y_conf = np.log(-np.log(1 - probs_conf))

    # Производная преобразования
    dy_df = 1.0 / ((1 - probs_conf) * np.abs(np.log(1 - probs_conf)))

    # SE для эмпирической функции распределения
    se_prob = np.sqrt(probs_conf * (1 - probs_conf) / n)

    # SE для преобразованных значений (дельта-метод)
    se_y = se_prob * dy_df

    confidence_level = 1.96  # 95% доверительный интервал

    # Доверительные интервалы
    y_upper = y_conf + confidence_level * se_y
    y_lower = y_conf - confidence_level * se_y

    # Верхняя граница (плавная дуга)
    ax.plot(x_conf, y_upper, 'b--', linewidth=2,
            alpha=0.7, label='Confidence bounds', zorder=1)

    # Нижняя граница (плавная дуга)
    ax.plot(x_conf, y_lower, 'b--', linewidth=2,
            alpha=0.7, zorder=1)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('ln(-ln(1-F))', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {output_file}")
    plt.close()

def plot_mls_weibull(params, data, censored, output_file):
    """Построение графика на вероятностной бумаге Вейбулла для MLS"""
    scale = params['parameter_1']  # λ (lambda)
    shape = params['parameter_2']  # k (форма)

    # Увеличиваем размер для лучшего качества при масштабировании
    fig, ax = plt.subplots(figsize=(14, 9), dpi=100)
    fig.suptitle('Weibull MLS', fontsize=14, fontweight='bold')

    complete_data = data[censored == 0]
    complete_data = complete_data[complete_data > 0]
    censored_data = data[censored == 1]
    censored_data = censored_data[censored_data > 0]
    n = len(complete_data)

    # Сортируем данные для получения порядковых статистик
    sorted_data = np.sort(complete_data)

    # Вычисляем эмпирические вероятности
    i = np.arange(1, n + 1)
    empirical_probs = (i - 0.375) / (n + 0.25)

    # Для Вейбулла: F(x) = 1 - exp(-(x/λ)^k)
    # Линеаризация: ln(-ln(1-F)) = k·ln(x) - k·ln(λ)

    # Эмпирические точки в преобразованных координатах
    y_empirical = np.log(-np.log(1 - empirical_probs))

    # 1. Точки данных (эмпирические - полные наблюдения)
    ax.scatter(sorted_data, y_empirical,
              color='darkblue', s=80, alpha=0.8,
              edgecolors='black', linewidths=1,
              label='Complete data', zorder=3)

    # Цензурированные данные - показываем крестиками
    if len(censored_data) > 0:
        # Для цензурированных данных используем теоретическую линию
        y_censored = shape * np.log(censored_data) - shape * np.log(scale)
        ax.scatter(censored_data, y_censored,
                  marker='x', color='red', s=100, linewidths=2,
                  label='Censored data', zorder=3)

    # 2. Апроксимирующая прямая (теоретическое распределение)
    # y = k·ln(x) - k·ln(λ)
    # Ограничиваем диапазон только полными данными
    x_line = np.linspace(sorted_data.min(), sorted_data.max(), 200)
    x_line = np.maximum(x_line, 0.001)
    y_line = shape * np.log(x_line) - shape * np.log(scale)
    ax.plot(x_line, y_line, 'r-', linewidth=2.5,
            label='Fitted line', zorder=2)

    # 3. Доверительные границы (отклонения) - плавные дуги
    # Вычисляем доверительные границы вдоль теоретической линии
    # Ограничиваем область построения, чтобы избежать артефактов на краях

    # Теоретические вероятности для x_line
    probs_line = stats.weibull_min.cdf(x_line, shape, scale=scale)

    # Фильтруем только разумный диапазон вероятностей (избегаем краев)
    valid_mask = (probs_line > 0.01) & (probs_line < 0.99)
    x_conf = x_line[valid_mask]
    probs_conf = probs_line[valid_mask]

    # Для Вейбулла используем дельта-метод
    # y = ln(-ln(1-F))
    # dy/dF = 1/((1-F)*ln(1-F))
    y_conf = np.log(-np.log(1 - probs_conf))

    # Производная преобразования
    dy_df = 1.0 / ((1 - probs_conf) * np.abs(np.log(1 - probs_conf)))

    # SE для эмпирической функции распределения
    se_prob = np.sqrt(probs_conf * (1 - probs_conf) / n)

    # SE для преобразованных значений (дельта-метод)
    se_y = se_prob * dy_df

    confidence_level = 1.96  # 95% доверительный интервал

    # Доверительные интервалы
    y_upper = y_conf + confidence_level * se_y
    y_lower = y_conf - confidence_level * se_y

    # Верхняя граница (плавная дуга)
    ax.plot(x_conf, y_upper, 'b--', linewidth=2,
            alpha=0.7, label='Confidence bounds', zorder=1)

    # Нижняя граница (плавная дуга)
    ax.plot(x_conf, y_lower, 'b--', linewidth=2,
            alpha=0.7, zorder=1)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('ln(-ln(1-F))', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {output_file}")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Использование: python plot_weibull.py [mle|mls]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()

    # Используем пути относительно текущей директории (которая должна быть /alg2)
    if mode == 'mle':
        results_file = 'output/mle_weibull_complete.txt'
        output_file = 'output/plot_mle_weibull.png'
        params, data, censored = read_results(results_file)
        plot_mle_weibull(params, data, censored, output_file)
    elif mode == 'mls':
        results_file = 'output/mls_weibull_censored.txt'
        output_file = 'output/plot_mls_weibull.png'
        params, data, censored = read_results(results_file)
        plot_mls_weibull(params, data, censored, output_file)
    else:
        print("Ошибка: укажите 'mle' или 'mls'")
        sys.exit(1)

if __name__ == '__main__':
    main()