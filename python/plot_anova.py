#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для визуализации результатов ANOVA
Показывает эмпирические функции распределения (CDF) для каждой группы
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os

# Переходим в корневую директорию проекта
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)

def read_anova_results(filename):
    """
    Читает результаты ANOVA из файла

    Returns:
        tuple: (результаты теста, данные групп)
    """
    result = {
        'group_means': [],
        'group_sizes': [],
        'group_count': 0
    }
    groups_data = []

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        in_group_info = False
        in_data_section = False
        current_group = []

        for line in lines:
            line = line.strip()

            # Извлекаем параметры
            if line.startswith('Количество групп:'):
                result['num_groups'] = int(line.split('= ')[1])
            elif line.startswith('Общее количество наблюдений:'):
                result['total_n'] = int(line.split('= ')[1])
            elif 'Уровень значимости:' in line and 'α = ' in line:
                result['alpha'] = float(line.split('α = ')[1])
            elif line.startswith('F-статистика = '):
                result['f_statistic'] = float(line.split('= ')[1])
            elif line.startswith('p-value = ') or line.startswith('P-значение = '):
                result['p_value'] = float(line.split('= ')[1])
            elif 'H0 ОТВЕРГАЕТСЯ' in line:
                result['reject_h0'] = True
            elif 'H0 НЕ ОТВЕРГАЕТСЯ' in line:
                result['reject_h0'] = False

            # Читаем информацию о группах
            elif line.startswith('Информация о группах:'):
                in_group_info = True
            elif in_group_info and line.startswith('Группа'):
                parts = line.split(',')
                n = int(parts[0].split('= ')[1])
                mean = float(parts[1].split('= ')[1])
                result['group_sizes'].append(n)
                result['group_means'].append(mean)
            elif in_group_info and 'Общее среднее:' in line:
                result['grand_mean'] = float(line.split('= ')[1])
                in_group_info = False

            # Читаем данные групп
            elif line.startswith('# Данные групп'):
                in_data_section = True
            elif in_data_section and line.startswith('# Группа'):
                if current_group:
                    groups_data.append(np.array(current_group))
                current_group = []
            elif in_data_section and line and not line.startswith('#'):
                try:
                    current_group.append(float(line))
                except:
                    pass

        # Добавляем последнюю группу
        if current_group:
            groups_data.append(np.array(current_group))

        return result, groups_data

    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return None, None

def plot_anova_cdf(result, groups_data, output_filename):
    """
    Создает график с эмпирическими функциями распределения (CDF) для каждой группы
    Показывает различия в дисперсиях между группами

    Args:
        result: словарь с результатами теста
        groups_data: список массивов с данными групп
        output_filename: имя выходного файла
    """
    if not groups_data or len(groups_data) == 0:
        print("Ошибка: нет данных групп для построения графика")
        return

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(14, 9), dpi=100)

    # Цвета для групп
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']

    # Строим эмпирические CDF для каждой группы
    for i, group in enumerate(groups_data):
        sorted_data = np.sort(group)
        n = len(sorted_data)
        # Эмпирическая CDF: F(x) = i/n
        cdf_values = np.arange(1, n + 1) / n

        color = colors[i % len(colors)]

        # Рисуем CDF как ступенчатую функцию
        ax.plot(sorted_data, cdf_values, '-', linewidth=2.5,
                color=color, label=f'Выборка {i+1} (n={n})', alpha=0.8)

        # Добавляем вертикальную пунктирную линию для среднего
        mean = np.mean(group)
        ax.axvline(mean, color=color, linestyle='--', alpha=0.4, linewidth=1.5)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('F(x)', fontsize=12)
    ax.set_title('ANOVA', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    # Добавляем информацию о тесте
    if result.get('reject_h0'):
        test_result = 'H₀ ОТВЕРГАЕТСЯ'
        box_color = 'lightcoral'
    else:
        test_result = 'H₀ НЕ ОТВЕРГАЕТСЯ'
        box_color = 'lightgreen'

    info_text = f'{test_result}\n\n'
    info_text += f'F = {result.get("f_statistic", 0.0):.4f}\n'
    info_text += f'p = {result.get("p_value", 0.0):.4f}\n'
    info_text += f'α = {result.get("alpha", 0.05):.2f}'

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"График ANOVA (CDF) сохранен: {output_filename}")
    plt.close()

def main():
    """
    Основная функция
    """
    input_file = 'output/anova_result.txt'

    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден")
        return

    print("Создание графиков для ANOVA...")

    result, groups_data = read_anova_results(input_file)

    if result and groups_data:
        plot_anova_cdf(result, groups_data, 'output/plot_anova_f_distribution.png')
        print("Визуализация ANOVA завершена!")
    else:
        print("Ошибка при чтении результатов ANOVA")

if __name__ == "__main__":
    main()
