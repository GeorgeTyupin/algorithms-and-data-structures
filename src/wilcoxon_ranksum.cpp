#include "wilcoxon_ranksum.h"
#include "boost_distributions.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>

/**
 * @brief Алгоритм AS 62: точное распределение Mann-Whitney U-статистики
 *
 * Реализация Applied Statistics Algorithm AS 62 (1973)
 * Генерирует частоты для распределения Mann-Whitney U-статистики
 */
int udist(int m, int n, std::vector<double>& frqncy, std::vector<double>& work) {
    int minmn, mn1, maxmn, n1, in, l, k, j;
    double sum;

    // Проверка корректности размера меньшей выборки
    minmn = std::min(m, n);
    if (minmn < 1) {
        return 1;  // IFAULT = 1
    }

    // Проверка размера массива результатов
    mn1 = m * n + 1;
    if (static_cast<int>(frqncy.size()) < mn1) {
        return 2;  // IFAULT = 2
    }

    // Инициализация для первого цикла
    maxmn = std::max(m, n);
    n1 = maxmn + 1;
    for (int i = 0; i < n1; i++) {
        frqncy[i] = 1.0;
    }

    // Если минимальный размер = 1, завершаем
    if (minmn == 1) {
        return 0;  // IFAULT = 0
    }

    // Проверка размера рабочего массива
    int required_work_size = (mn1 + 1) / 2 + minmn;
    if (static_cast<int>(work.size()) < required_work_size) {
        return 3;  // IFAULT = 3
    }

    // Очистка оставшейся части frqncy
    for (int i = n1; i < mn1; i++) {
        frqncy[i] = 0.0;
    }

    // Генерация распределений более высокого порядка
    work[0] = 0.0;
    in = maxmn;

    for (int i = 1; i < minmn; i++) {
        work[i] = 0.0;
        in = in + maxmn;
        n1 = in + 2;
        l = 1 + in / 2;
        k = i;

        // Генерация полного распределения снаружи внутрь
        for (j = 0; j < l; j++) {
            k = k + 1;
            n1 = n1 - 1;
            sum = frqncy[j] + work[j];
            frqncy[j] = sum;
            work[k] = sum - frqncy[n1];
            frqncy[n1] = sum;
        }
    }

    return 0;  // IFAULT = 0
}

/**
 * @brief Вычисление p-value для Mann-Whitney U-статистики по точному распределению
 *
 * Использует алгоритм AS 62 для получения точного распределения
 * и вычисления p-value для двустороннего теста
 */
double mann_whitney_pvalue_exact(double u_statistic, int m, int n) {
    int mn1 = m * n + 1;
    int required_work_size = (mn1 + 1) / 2 + std::min(m, n);

    std::vector<double> frqncy(mn1);
    std::vector<double> work(required_work_size);

    // Генерируем точное распределение
    int ifault = udist(m, n, frqncy, work);

    if (ifault != 0) {
        std::cerr << "Ошибка в udist: код " << ifault << std::endl;
        return -1.0;
    }

    // Преобразуем частоты в функцию распределения
    double total = 0.0;
    for (int i = 0; i < mn1; i++) {
        total += frqncy[i];
    }

    // Накопленная функция распределения
    std::vector<double> cdf(mn1);
    double cum_sum = 0.0;
    for (int i = 0; i < mn1; i++) {
        cum_sum += frqncy[i];
        cdf[i] = cum_sum / total;
    }

    // P-value для двустороннего теста
    // U-статистика округляется до целого для индексации
    int u_index = static_cast<int>(std::round(u_statistic));

    // Защита от выхода за границы
    if (u_index < 0) u_index = 0;
    if (u_index >= mn1) u_index = mn1 - 1;

    // P(U <= u_obs)
    double p_lower = cdf[u_index];

    // Для двустороннего теста: p-value = 2 * min(P(U <= u_obs), P(U >= u_obs))
    double p_upper = 1.0 - p_lower + frqncy[u_index] / total;
    double p_value = 2.0 * std::min(p_lower, p_upper);

    // Ограничиваем p-value значением 1.0
    if (p_value > 1.0) {
        p_value = 1.0;
    }

    return p_value;
}

/**
 * @brief Структура для хранения значения и его источника
 */
struct RankedValue {
    double value;
    size_t group;  // 1 или 2 - номер группы
    double rank;   // Ранг (может быть дробным при связях)

    bool operator<(const RankedValue& other) const {
        return value < other.value;
    }
};

/**
 * @brief Вычисление рангов с учетом связанных значений (ties)
 *
 * Связанным значениям присваивается средний ранг
 *
 * @param values Вектор значений для ранжирования
 * @return Количество групп связанных значений
 */
static size_t assign_ranks(std::vector<RankedValue>& values) {
    if (values.empty()) return 0;

    // Сортируем по значению
    std::sort(values.begin(), values.end());

    size_t num_ties = 0;
    size_t i = 0;
    while (i < values.size()) {
        // Находим группу одинаковых значений
        size_t j = i;
        while (j < values.size() && values[j].value == values[i].value) {
            ++j;
        }

        // Вычисляем средний ранг для группы
        // Ранги начинаются с 1, а не с 0
        double avg_rank = (i + 1 + j) / 2.0;

        // Присваиваем средний ранг всем элементам группы
        for (size_t k = i; k < j; ++k) {
            values[k].rank = avg_rank;
        }

        // Если в группе больше одного элемента, это связь
        if (j - i > 1) {
            ++num_ties;
        }

        i = j;
    }

    return num_ties;
}

/**
 * @brief Вычисление поправки на связи для дисперсии W
 *
 * При наличии связанных рангов дисперсия уменьшается
 * Поправка (формула 3.35): correction = Σ(tᵢ³ - tᵢ) / (12(N-1))
 * где tᵢ - размер i-й группы связанных значений
 *
 * @param values Ранжированные значения
 * @return Поправка на связи
 */
static double compute_tie_correction(const std::vector<RankedValue>& values) {
    if (values.empty()) return 0.0;

    std::map<double, size_t> tie_groups;

    // Подсчитываем размеры групп связанных рангов
    for (const auto& v : values) {
        tie_groups[v.rank]++;
    }

    // Вычисляем поправку
    double correction = 0.0;
    for (const auto& pair : tie_groups) {
        size_t t = pair.second;
        if (t > 1) {
            correction += (t * t * t - t);
        }
    }

    size_t N = values.size();
    if (N > 1) {
        correction /= (12.0 * (N - 1));
    }

    return correction;
}

/**
 * @brief Критерий ранга суммы Уилкоксона
 *
 * Реализация по формулам 3.32-3.35 из порядковых статистик
 */
WilcoxonRankSumResult wilcoxon_ranksum_test(const std::vector<double>& data1,
                                             const std::vector<double>& data2,
                                             double alpha) {
    WilcoxonRankSumResult result;
    result.alpha = alpha;
    result.n1 = data1.size();
    result.n2 = data2.size();
    result.total_n = result.n1 + result.n2;

    // Проверка корректности входных данных
    if (data1.empty() || data2.empty()) {
        std::cerr << "Ошибка: обе выборки должны быть непустыми" << std::endl;
        result.reject_h0 = false;
        return result;
    }

    // Объединяем выборки и помечаем источник
    std::vector<RankedValue> all_values;
    all_values.reserve(result.total_n);

    for (double val : data1) {
        all_values.push_back({val, 1, 0.0});
    }

    for (double val : data2) {
        all_values.push_back({val, 2, 0.0});
    }

    // Ранжируем все значения
    result.num_ties = assign_ranks(all_values);

    // Вычисляем сумму рангов первой выборки (формула 3.32)
    // W = Σ Rᵢ, где Rᵢ - ранг наблюдения из первой выборки
    result.w_statistic = 0.0;
    for (const auto& val : all_values) {
        if (val.group == 1) {
            result.w_statistic += val.rank;
        }
    }

    // Вычисляем статистику U Манна-Уитни
    // U₁ = W - n₁(n₁ + 1) / 2
    double u1 = result.w_statistic - result.n1 * (result.n1 + 1.0) / 2.0;
    double u2 = result.n1 * result.n2 - u1;
    result.u_statistic = std::min(u1, u2);

    // Вычисляем математическое ожидание W под H0 (формула 3.33)
    // E[W] = n₁(n₁ + n₂ + 1) / 2
    result.mean_w = result.n1 * (result.total_n + 1.0) / 2.0;

    // Вычисляем поправку на связи
    result.tie_correction = compute_tie_correction(all_values);

    // Вычисляем дисперсию W под H0 (формула 3.33 с поправкой 3.35)
    // Var[W] = n₁n₂(n₁ + n₂ + 1) / 12 - n₁n₂ * correction
    double var_w = result.n1 * result.n2 * (result.total_n + 1.0) / 12.0;
    var_w -= result.n1 * result.n2 * result.tie_correction;
    result.std_w = std::sqrt(var_w);

    // Определяем, использовать ли нормальное приближение или точное распределение AS 62
    // Для больших выборок (n₁ > 50 или n₂ > 50) используем нормальное приближение
    // Для малых и средних выборок используем точное распределение AS 62
    result.use_normal_approx = (result.n1 > 50 || result.n2 > 50);

    if (result.use_normal_approx) {
        // Вычисляем Z-статистику с поправкой на непрерывность (формула 3.34)
        // Z = (W - E[W] - 0.5) / SD[W] для двустороннего теста
        double continuity_correction = 0.5;
        if (result.w_statistic > result.mean_w) {
            result.z_statistic = (result.w_statistic - result.mean_w - continuity_correction) / result.std_w;
        } else {
            result.z_statistic = (result.w_statistic - result.mean_w + continuity_correction) / result.std_w;
        }

        // P-значение для двустороннего теста
        // P = 2 * P(|Z| > |z_obs|)
        double z_abs = std::abs(result.z_statistic);
        result.p_value = 2.0 * (1.0 - norm_cdf(z_abs));

        // Критическое значение для нормального распределения
        result.critical_value = norm_ppf(1.0 - alpha / 2.0);

        // Проверяем гипотезу H0
        result.reject_h0 = (z_abs > result.critical_value);

    } else {
        // Для малых и средних выборок используем точное распределение AS 62
        result.z_statistic = (result.w_statistic - result.mean_w) / result.std_w;

        // Вычисляем p-value по точному распределению AS 62
        result.p_value = mann_whitney_pvalue_exact(result.u_statistic, result.n1, result.n2);

        // Критическое значение для нормального распределения (для сравнения)
        result.critical_value = norm_ppf(1.0 - alpha / 2.0);

        // Проверяем гипотезу H0 на основе p-value
        result.reject_h0 = (result.p_value < alpha);
    }

    return result;
}

/**
 * @brief Вывод результатов критерия Уилкоксона
 */
void print_wilcoxon_ranksum_result(const WilcoxonRankSumResult& result,
                                   const std::string& filename) {
    std::ostream* out = &std::cout;
    std::ofstream file;

    if (!filename.empty()) {
        file.open(filename);
        if (file.is_open()) {
            out = &file;
        }
    }

    *out << "============================================================" << std::endl;
    *out << "  КРИТЕРИЙ РАНГА СУММЫ УИЛКОКСОНА" << std::endl;
    *out << "  (Wilcoxon rank-sum test / Mann-Whitney U test)" << std::endl;
    *out << "  Непараметрический критерий для двух независимых выборок" << std::endl;
    *out << "============================================================" << std::endl;
    *out << std::endl;

    *out << "Размеры выборок: n₁ = " << result.n1 << ", n₂ = " << result.n2 << std::endl;
    *out << "Общее количество наблюдений: N = " << result.total_n << std::endl;
    *out << "Уровень значимости: α = " << result.alpha << std::endl;
    *out << "Метод: " << (result.use_normal_approx ?
                          "нормальное приближение" :
                          "точное распределение AS 62") << std::endl;
    *out << std::endl;

    *out << std::fixed << std::setprecision(6);
    *out << "Статистики:" << std::endl;
    *out << "  W-статистика (сумма рангов) = " << result.w_statistic << std::endl;
    *out << "  U-статистика (Манна-Уитни) = " << result.u_statistic << std::endl;
    *out << "  E[W] под H0 = " << result.mean_w << std::endl;
    *out << "  SD[W] под H0 = " << result.std_w << std::endl;
    *out << "  Z-статистика = " << result.z_statistic << std::endl;
    *out << std::endl;

    if (result.num_ties > 0) {
        *out << "Обнаружено связанных групп: " << result.num_ties << std::endl;
        *out << "Поправка на связи: " << std::setprecision(8) << result.tie_correction << std::endl;
        *out << std::endl;
    }

    *out << std::setprecision(6);
    *out << "Критическое значение (Z) = " << result.critical_value << std::endl;
    *out << "P-значение (двусторонний тест) = " << std::setprecision(4) << result.p_value << std::endl;
    *out << std::endl;

    *out << "Гипотеза H0: F₁(x) = F₂(x) (распределения одинаковы)" << std::endl;
    if (result.reject_h0) {
        *out << "РЕЗУЛЬТАТ: H0 ОТВЕРГАЕТСЯ (распределения различаются)" << std::endl;
        *out << "|Z| (" << std::setprecision(6) << std::abs(result.z_statistic)
             << ") > Z_critical (" << result.critical_value << ")" << std::endl;
        *out << "p-value (" << std::setprecision(4) << result.p_value
             << ") < α (" << result.alpha << ")" << std::endl;
    } else {
        *out << "РЕЗУЛЬТАТ: H0 НЕ ОТВЕРГАЕТСЯ (нет оснований отвергнуть гипотезу о равенстве распределений)" << std::endl;
        *out << "|Z| (" << std::setprecision(6) << std::abs(result.z_statistic)
             << ") ≤ Z_critical (" << result.critical_value << ")" << std::endl;
        *out << "p-value (" << std::setprecision(4) << result.p_value
             << ") ≥ α (" << result.alpha << ")" << std::endl;
    }
    *out << std::endl;

    *out << "Примечание: Критерий Уилкоксона не требует нормальности распределения" << std::endl;
    *out << "и устойчив к выбросам. Проверяет различие распределений в целом," << std::endl;
    *out << "а не только различие средних." << std::endl;
    *out << std::endl;

    if (file.is_open()) {
        std::cout << "Результаты сохранены в файл: " << filename << std::endl;
        file.close();
    }
}
