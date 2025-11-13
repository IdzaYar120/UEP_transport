import numpy as np
from flask import Flask, render_template, request
import pulp

# Створюємо екземпляр Flask
app = Flask(__name__)

@app.route('/')
def setup():
    """Сторінка 1: Вибір M та N"""
    return render_template('setup.html')


@app.route('/input')
def index():
    """Сторінка 2: Генерація форми вводу"""
    try:
        num_suppliers = int(request.args.get('num_suppliers', 1))
        num_consumers = int(request.args.get('num_consumers', 1))

        if not (1 <= num_suppliers <= 10 and 1 <= num_consumers <= 10):
            raise ValueError("Кількість має бути в межах від 1 до 10.")
        
        return render_template('index.html', 
                               num_suppliers=num_suppliers, 
                               num_consumers=num_consumers)
    
    except Exception as e:
        error_message = f"Помилка: {e}. Будь ласка, введіть числа від 1 до 10."
        return render_template('setup.html', error=error_message)


@app.route('/solve', methods=['POST'])
def solve():
    """Сторінка 3: Розрахунок (з фіктивним постачальником)"""
    
    try:
        # 1. Отримуємо базові M і N
        num_suppliers_orig = int(request.form['num_suppliers'])
        num_consumers = int(request.form['num_consumers'])
        
        # 2. Збираємо дані з форми
        supply = [float(request.form[f'supply_{i}']) for i in range(num_suppliers_orig)]
        demand = [float(request.form[f'demand_{i}']) for i in range(num_consumers)]
        
        costs = []
        for i in range(num_suppliers_orig):
            row = [float(request.form[f'cost_{i}_{j}']) for j in range(num_consumers)]
            costs.append(row)

        total_supply = sum(supply)
        total_demand = sum(demand)
        
        # Зберігаємо оригінальні дані для відображення
        # (бо ми можемо змінити `supply`, `costs` та `num_suppliers` для розв'язувача)
        original_data = {
            "supply": list(supply),
            "costs": [list(row) for row in costs],
            "num_suppliers": num_suppliers_orig,
            "total_supply": total_supply
        }
        
        is_fictitious_supplier_added = False
        num_suppliers_solver = num_suppliers_orig # К-ть постачальників для PuLP

        # 3. ⭐️⭐️⭐️ ОСНОВНА ЛОГІКА: ДОДАВАННЯ ФІКТИВНОГО ПОСТАЧАЛЬНИКА ⭐️⭐️⭐️
        if total_demand > total_supply:
            is_fictitious_supplier_added = True
            
            # Дефіцит, який "покриє" фіктивний постачальник
            shortage = total_demand - total_supply
            
            # Додаємо його запаси до списку
            supply.append(shortage)
            
            # Додаємо нульові витрати для нього
            fictitious_costs_row = [0.0] * num_consumers
            costs.append(fictitious_costs_row)
            
            # Збільшуємо к-ть постачальників ТІЛЬКИ для розв'язувача
            num_suppliers_solver += 1
            
            # Тепер задача "збалансована" (supply == demand)

        # 4. Створюємо задачу в PuLP
        prob = pulp.LpProblem("Transport_Problem", pulp.LpMinimize)

        # 5. Створюємо матрицю змінних M x N (використовуючи `num_suppliers_solver`)
        routes = pulp.LpVariable.dicts("Route", 
                                       ((i, j) for i in range(num_suppliers_solver) for j in range(num_consumers)), 
                                       lowBound=0, 
                                       cat='Continuous')

        # 6. Цільова функція (витрати)
        prob += pulp.lpSum(routes[i, j] * costs[i][j] 
                           for i in range(num_suppliers_solver) 
                           for j in range(num_consumers)), "Total Cost"

        # 7. Обмеження
        
        # 7.1. Обмеження по запасах (для M або M+1 постачальників)
        for i in range(num_suppliers_solver):
            prob += pulp.lpSum(routes[i, j] for j in range(num_consumers)) <= supply[i], f"Supply_Constraint_{i}"

        # 7.2. Обмеження по потребах
        # Оскільки ми збалансували задачу, ми можемо вимагати ТОЧНОГО задоволення попиту
        # (реальні постачальники + фіктивний)
        for j in range(num_consumers):
            prob += pulp.lpSum(routes[i, j] for i in range(num_suppliers_solver)) == demand[j], f"Demand_Constraint_{j}"

        # 8. Вирішуємо задачу
        prob.solve()
        
        if prob.status == pulp.LpStatusOptimal:
            total_cost = pulp.value(prob.objective)
            
            # 9. Готуємо результати
            
            # Створюємо план ТІЛЬКИ для реальних постачальників
            solution_plan = np.zeros((num_suppliers_orig, num_consumers))
            for i in range(num_suppliers_orig):
                for j in range(num_consumers):
                    solution_plan[i, j] = routes[i, j].varValue

            # Окремо отримуємо "поставки" від фіктивного (тобто дефіцит)
            fictitious_shipments = []
            if is_fictitious_supplier_added:
                fictitious_supplier_index = num_suppliers_solver - 1 # (це останній доданий)
                for j in range(num_consumers):
                    fictitious_shipments.append(routes[fictitious_supplier_index, j].varValue)

            return render_template(
                'result.html',
                total_cost=round(total_cost, 2),
                solution=solution_plan.round(2),
                
                # Передаємо оригінальні дані
                supply=original_data["supply"],
                demand=demand,
                costs=original_data["costs"],
                total_supply=original_data["total_supply"],
                total_demand=total_demand,
                num_suppliers=original_data["num_suppliers"],
                num_consumers=num_consumers,
                
                # Передаємо нові дані про дефіцит
                is_fictitious_supplier_added=is_fictitious_supplier_added,
                fictitious_shipments=np.array(fictitious_shipments).round(2)
            )
        
        else:
            # Якщо щось пішло не так (вже не через дефіцит, а з іншої причини)
            error_message = f"Не вдалося знайти оптимальний розв'язок. Статус: {pulp.LpStatus[prob.status]}"
            return render_template('index.html',
                                   error=error_message,
                                   num_suppliers=num_suppliers_orig,
                                   num_consumers=num_consumers)

    except ValueError:
        # ... (обробка помилок залишається такою ж) ...
        error_message = "Помилка: Будь ласка, введіть лише числа у всі поля."
        return render_template('index.html', 
                               error=error_message, 
                               num_suppliers=request.form.get('num_suppliers', 1), 
                               num_consumers=request.form.get('num_consumers', 1))
    except Exception as e:
        error_message = f"Виникла неочікувана помилка: {e}"
        return render_template('setup.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)