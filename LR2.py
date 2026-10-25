import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==========================================
# 1. ФУНКЦИЯ ЗАДАНИЯ НАЧАЛЬНЫХ ПАРАМЕТРОВ
# ==========================================
def get_initial_parameters():
    params = {
        'N_k0': 1000.0,   # Начальная численность Красных
        'N_c0': 800.0,   # Начальная численность Синих
        
        # Вероятность поражения цели одним выстрелом (P)
        'P_k': 0.1,     
        'P_c': 0.15,     
        
        # Плотность потока выстрелов (выстрелов в час)
        'lambda_rate_k': 10.0, 
        'lambda_rate_c': 15.0,  
        
        't_max': 500.0, 
        
        # Критерии остановки
        'surrender_ratio': 3.0,  # Превышение в 3 раза
        'destruction_ratio': 0.05 # Остаток 5%
    }
    
    # Эффективная скорострельность (лямбда большая) = P * lambda_rate
    params['lambda_eff_k'] = params['P_k'] * params['lambda_rate_k']
    params['lambda_eff_c'] = params['P_c'] * params['lambda_rate_c']
    
    return params

# ==========================================
# 2. ФУНКЦИЯ ЗАПУСКА МОДЕЛИ
# ==========================================
def run_simulation(params):
    def lanchester_poorly_organized(t, y):
        N_k, N_c = y
        if N_k <= 0 or N_c <= 0:
            return [0, 0]
        dN_k_dt = -params['lambda_eff_c'] * N_c * (N_k / params['N_k0'])
        dN_c_dt = -params['lambda_eff_k'] * N_k * (N_c / params['N_c0'])
        return [dN_k_dt, dN_c_dt]

    def blue_surrenders_event(t, y):
        N_k, N_c = y
        return N_k - params['surrender_ratio'] * N_c
    blue_surrenders_event.terminal = True
    blue_surrenders_event.direction = 1 

    def red_surrenders_event(t, y):
        N_k, N_c = y
        return N_c - params['surrender_ratio'] * N_k
    red_surrenders_event.terminal = True
    red_surrenders_event.direction = 1

    def red_destroyed_event(t, y):
        N_k, N_c = y
        return N_k - params['destruction_ratio'] * params['N_k0']
    red_destroyed_event.terminal = True
    red_destroyed_event.direction = -1 

    def blue_destroyed_event(t, y):
        N_k, N_c = y
        return N_c - params['destruction_ratio'] * params['N_c0']
    blue_destroyed_event.terminal = True
    blue_destroyed_event.direction = -1

    events = [blue_surrenders_event, red_surrenders_event, red_destroyed_event, blue_destroyed_event]

    y0 = [params['N_k0'], params['N_c0']]
    t_span = (0, params['t_max'])
    t_eval = np.linspace(0, params['t_max'], 1000)

    sol = solve_ivp(
        fun=lanchester_poorly_organized,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        events=events,
        dense_output=True
    )
    return sol

# ==========================================
# 3. ФУНКЦИЯ АНАЛИЗА И ВИЗУАЛИЗАЦИИ
# ==========================================
def analyze_and_plot(sol, params):
    t_result = sol.t
    y_result = sol.y

    for event_times in sol.t_events:
        if len(event_times) > 0:
            t_event = event_times[0]
            y_event = sol.sol(t_event)
            if t_event > t_result[-1] + 1e-9:
                t_result = np.append(t_result, t_event)
                y_result = np.append(y_result, y_event.reshape(2, 1), axis=1)
            break 

    N_k_result = y_result[0]
    N_c_result = y_result[1]
    
    print(f"\n--- РЕЗУЛЬТАТЫ БОЯ ---")
    print(f"Время боя: {t_result[-1]:.2f} ч.")
    print(f"Остаток Красных: {N_k_result[-1]:.0f} ({N_k_result[-1]/params['N_k0']*100:.1f}%)")
    print(f"Остаток Синих:   {N_c_result[-1]:.0f} ({N_c_result[-1]/params['N_c0']*100:.1f}%)")
    
    if len(sol.t_events[0]) > 0:
        print("ИСХОД: Синие сдались в плен (Красные превысили численность в 3 раза).")
    elif len(sol.t_events[1]) > 0:
        print("ИСХОД: Красные сдались в плен (Синие превысили численность в 3 раза).")
    elif len(sol.t_events[2]) > 0:
        print("ИСХОД: Красные уничтожены (остаток < 5%).")
    elif len(sol.t_events[3]) > 0:
        print("ИСХОД: Синие уничтожены (остаток < 5%).")
    else:
        print("ИСХОД: Бой не завершился за заданное время (t_max).")

    plt.figure(figsize=(10, 6))
    plt.plot(t_result, N_k_result, 'r-', linewidth=2, label='Красные')
    plt.plot(t_result, N_c_result, 'b-', linewidth=2, label='Синие')
    
    plt.axhline(y=params['N_k0'] * params['destruction_ratio'], color='r', linestyle='--', alpha=0.5, label='5% от Красных')
    plt.axhline(y=params['N_c0'] * params['destruction_ratio'], color='b', linestyle='--', alpha=0.5, label='5% от Синих')
    
    plt.title('Динамика плохоорганизованного боя (Модель Ланчестера)')
    plt.xlabel('Время (часы)')
    plt.ylabel('Количество боевых единиц')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.plot(t_result[-1], N_k_result[-1], 'ro', markersize=8)
    plt.plot(t_result[-1], N_c_result[-1], 'bo', markersize=8)
    
    plt.show()

# ==========================================
# 4. ФУНКЦИИ КОНСОЛЬНОГО ИНТЕРФЕЙСА
# ==========================================
def print_current_params(params):
    print("\n" + "="*40)
    print(" ТЕКУЩИЕ ПАРАМЕТРЫ ")
    print("="*40)
    print(f"1. Численность Красных (N_k0): {params['N_k0']}")
    print(f"2. Численность Синих (N_c0):   {params['N_c0']}")
    print(f"3. Вероятность поражения (P_k): {params['P_k']}")
    print(f"4. Вероятность поражения (P_c): {params['P_c']}")
    print(f"5. Плотность выстрелов (Красные): {params['lambda_rate_k']}")
    print(f"6. Плотность выстрелов (Синие):   {params['lambda_rate_c']}")
    print(f"7. Макс. время боя (t_max): {params['t_max']} ч.")
    print(f"8. Коэфф. сдачи в плен: {params['surrender_ratio']}")
    print(f"9. Коэфф. уничтожения (доля): {params['destruction_ratio']}")
    print("-" * 40)
    print(f"-> Эффективная скорострельность Красных: {params['lambda_eff_k']:.3f}")
    print(f"-> Эффективная скорострельность Синих:   {params['lambda_eff_c']:.3f}")
    print("="*40)

def edit_params(params):
    print("\n--- ИЗМЕНЕНИЕ ПАРАМЕТРОВ ---")
    print("(Нажмите Enter, чтобы оставить текущее значение)")
    
    editable_params = [
        ('N_k0', 'Численность Красных'),
        ('N_c0', 'Численность Синих'),
        ('P_k', 'Вероятность поражения (Красные)'),
        ('P_c', 'Вероятность поражения (Синие)'),
        ('lambda_rate_k', 'Плотность выстрелов (Красные)'),
        ('lambda_rate_c', 'Плотность выстрелов (Синие)'),
        ('t_max', 'Макс. время боя (ч)'),
        ('surrender_ratio', 'Коэфф. сдачи в плен'),
        ('destruction_ratio', 'Коэфф. уничтожения (доля)')
    ]
    
    for key, name in editable_params:
        while True:
            user_input = input(f"{name} [{params[key]}]: ").strip()
            if user_input == "":
                break 
            try:
                params[key] = float(user_input)
                break
            except ValueError:
                print("Ошибка: пожалуйста, введите число.")
                
    params['lambda_eff_k'] = params['P_k'] * params['lambda_rate_k']
    params['lambda_eff_c'] = params['P_c'] * params['lambda_rate_c']
    print("\nПараметры успешно обновлены!")

def main_menu():
    params = get_initial_parameters()
    
    while True:
        print("\n" + "#"*40)
        print(" ГЛАВНОЕ МЕНЮ (Модель плохоорганизованного боя) ")
        print("#"*40)
        print("1. Текущие параметры")
        print("2. Задание параметров")
        print("3. Расчет (запуск и графики)")
        print("4. Выход")
        print("#"*40)
        
        choice = input("Выберите действие (1-4): ").strip()
        
        if choice == '1':
            print_current_params(params)
            # ПАУЗА
            input("\n[ Нажмите Enter, чтобы вернуться в меню... ]")
            
        elif choice == '2':
            edit_params(params)
            # ПАУЗА
            input("\n[ Нажмите Enter, чтобы вернуться в меню... ]")
            
        elif choice == '3':
            print("\nЗапуск расчета...")
            try:
                solution = run_simulation(params)
                analyze_and_plot(solution, params)
            except Exception as e:
                print(f"Произошла ошибка при расчете: {e}")
            # ПАУЗА (позволит прочитать результаты в консоли, даже если график уже закрыт)
            input("\n[ Нажмите Enter, чтобы вернуться в меню... ]")
            
        elif choice == '4':
            # Пауза перед выходом (чтобы пользователь случайно не закрыл консоль)
            input("\nЗавершение работы. [ Нажмите Enter для выхода... ]")
            break
            
        else:
            print("\nНеверный ввод. Пожалуйста, выберите пункт от 1 до 4.")
            # ПАУЗА при ошибке ввода
            input("\n[ Нажмите Enter, чтобы продолжить... ]")

# ==========================================
# 5. ТОЧКА ВХОДА
# ==========================================
if __name__ == "__main__":
    main_menu()