import numpy as np
from flask import Flask, render_template, request
import pulp

app = Flask(__name__)

@app.route('/')
def setup():
    return render_template('setup.html')


@app.route('/input')
def index():
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
    
    try:
        num_suppliers_orig = int(request.form['num_suppliers'])
        num_consumers = int(request.form['num_consumers'])
        
        supply = [float(request.form[f'supply_{i}']) for i in range(num_suppliers_orig)]
        demand = [float(request.form[f'demand_{i}']) for i in range(num_consumers)]
        
        costs = []
        for i in range(num_suppliers_orig):
            row = [float(request.form[f'cost_{i}_{j}']) for j in range(num_consumers)]
            costs.append(row)

        total_supply = sum(supply)
        total_demand = sum(demand)
        
       
        original_data = {
            "supply": list(supply),
            "costs": [list(row) for row in costs],
            "num_suppliers": num_suppliers_orig,
            "total_supply": total_supply
        }
        
        is_fictitious_supplier_added = False
        num_suppliers_solver = num_suppliers_orig # К-ть постачальників для PuLP

        if total_demand > total_supply:
            is_fictitious_supplier_added = True
            
            shortage = total_demand - total_supply
            
            supply.append(shortage)
            
            fictitious_costs_row = [0.0] * num_consumers
            costs.append(fictitious_costs_row)
            
            num_suppliers_solver += 1
            
           
        prob = pulp.LpProblem("Transport_Problem", pulp.LpMinimize)

        routes = pulp.LpVariable.dicts("Route", 
                                       ((i, j) for i in range(num_suppliers_solver) for j in range(num_consumers)), 
                                       lowBound=0, 
                                       cat='Continuous')

        prob += pulp.lpSum(routes[i, j] * costs[i][j] 
                           for i in range(num_suppliers_solver) 
                           for j in range(num_consumers)), "Total Cost"

       
        for i in range(num_suppliers_solver):
            prob += pulp.lpSum(routes[i, j] for j in range(num_consumers)) <= supply[i], f"Supply_Constraint_{i}"

       
        for j in range(num_consumers):
            prob += pulp.lpSum(routes[i, j] for i in range(num_suppliers_solver)) == demand[j], f"Demand_Constraint_{j}"

        prob.solve()
        
        if prob.status == pulp.LpStatusOptimal:
            total_cost = pulp.value(prob.objective)
            
           
            solution_plan = np.zeros((num_suppliers_orig, num_consumers))
            for i in range(num_suppliers_orig):
                for j in range(num_consumers):
                    solution_plan[i, j] = routes[i, j].varValue

            fictitious_shipments = []
            if is_fictitious_supplier_added:
                fictitious_supplier_index = num_suppliers_solver - 1 # (це останній доданий)
                for j in range(num_consumers):
                    fictitious_shipments.append(routes[fictitious_supplier_index, j].varValue)

            return render_template(
                'result.html',
                total_cost=round(total_cost, 2),
                solution=solution_plan.round(2),
                
                supply=original_data["supply"],
                demand=demand,
                costs=original_data["costs"],
                total_supply=original_data["total_supply"],
                total_demand=total_demand,
                num_suppliers=original_data["num_suppliers"],
                num_consumers=num_consumers,
                
                is_fictitious_supplier_added=is_fictitious_supplier_added,
                fictitious_shipments=np.array(fictitious_shipments).round(2)
            )
        
        else:
            error_message = f"Не вдалося знайти оптимальний розв'язок. Статус: {pulp.LpStatus[prob.status]}"
            return render_template('index.html',
                                   error=error_message,
                                   num_suppliers=num_suppliers_orig,
                                   num_consumers=num_consumers)

    except ValueError:
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
