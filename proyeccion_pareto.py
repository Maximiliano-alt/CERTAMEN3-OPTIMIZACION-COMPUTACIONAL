import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

class Problem:
    def __init__(self, lambda_param=0.5):
        self.dim = 10
        self.lambda_param = lambda_param
        self.min_values = [0, 0, 0, 0, 0, 65, 90, 40, 60, 20]
        self.max_values = [15, 10, 25, 4, 30, 85, 95, 60, 80, 30]
        self.variable_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'q1', 'q2', 'q3', 'q4', 'q5']
        self.media_names = ['TV-tarde', 'TV-noche', 'Diario', 'Revista', 'Radio']

    def get_costs(self, q):
        c1 = 2 * q[0] + 30
        c2 = 10 * q[1] - 600
        c3 = 2 * q[2] - 40
        c4 = q[3] + 40
        c5 = q[4] - 10
        return [c1, c2, c3, c4, c5]

    def check(self, x):
        x_vars = x[:5]
        q_vars = x[5:]
        costs = self.get_costs(q_vars)
        c1, c2, c3, c4, c5 = costs
        
        tv_budget = c1 * x_vars[0] + c2 * x_vars[1] <= 3800
        diario_revista = c3 * x_vars[2] + c4 * x_vars[3] <= 2800
        diario_radio = c3 * x_vars[2] + c5 * x_vars[4] <= 3500
        
        return tv_budget and diario_revista and diario_radio

    def get_q_and_c(self, x):
        x_vars = x[:5]
        q_vars = x[5:]
        Q = sum(q_vars[i] * x_vars[i] for i in range(5))
        costs = self.get_costs(q_vars)
        C = sum(costs[i] * x_vars[i] for i in range(5))
        return Q, C

    def get_budget_usage(self, x):
        x_vars = x[:5]
        q_vars = x[5:]
        costs = self.get_costs(q_vars)
        
        tv_used = costs[0] * x_vars[0] + costs[1] * x_vars[1]
        dr_used = costs[2] * x_vars[2] + costs[3] * x_vars[3]
        drad_used = costs[2] * x_vars[2] + costs[4] * x_vars[4]
        
        return {
            'TV': (tv_used, 3800),
            'Diario+Revista': (dr_used, 2800),
            'Diario+Radio': (drad_used, 3500)
        }

    def fit(self, x):
        if not self.check(x):
            return -1e9
        Q, C = self.get_q_and_c(x)
        return self.lambda_param * Q - (1 - self.lambda_param) * C

    def keep_domain(self, idx, val):
        lo, hi = self.min_values[idx], self.max_values[idx]
        if idx < 5:
            return int(max(lo, min(hi, round(val))))
        else:
            return max(lo, min(hi, val))

class Individual:
    def __init__(self, problem):
        self.p = problem
        self.dimension = self.p.dim
        self.x = []
        for i in range(self.dimension):
            if i < 5:
                self.x.append(random.randint(self.p.min_values[i], self.p.max_values[i]))
            else:
                self.x.append(random.uniform(self.p.min_values[i], self.p.max_values[i]))

    def is_feasible(self):
        return self.p.check(self.x)

    def fitness(self):
        return self.p.fit(self.x)

    def is_better_than(self, other):
        return self.fitness() > other.fitness()

    def move_towards(self, leader, sigma=1.0):
        for i in range(self.dimension):
            r1 = random.random()
            r2 = random.random()
            attraction = r1 * (leader.x[i] - self.x[i])
            noise = r2 * random.gauss(0, sigma)
            new_val = self.x[i] + attraction + noise
            self.x[i] = self.p.keep_domain(i, new_val)

    def copy(self, other):
        if isinstance(other, Individual):
            self.x = other.x.copy()

class Swarm:
    def __init__(self, problem, n_ind=40, iters=500):
        self.p = problem
        self.max_iter = iters
        self.n_individual = n_ind
        self.n_groups = 4
        self.group_size = self.n_individual // self.n_groups
        self.leader_rotation_freq = 20
        self.swarm = []
        self.best_global = None
        self.groups = []

    def initialize_population(self):
        for _ in range(self.n_individual):
            feasible = False
            attempts = 0
            while not feasible and attempts < 100:
                ind = Individual(self.p)
                feasible = ind.is_feasible()
                attempts += 1
            if feasible:
                self.swarm.append(ind)
        
        if len(self.swarm) == 0:
            raise ValueError("No se pudo generar población inicial factible")
        
        self.best_global = max(self.swarm, key=lambda ind: ind.fitness())
        self.organize_groups()

    def organize_groups(self):
        self.groups = []
        n_groups_actual = min(self.n_groups, len(self.swarm))
        group_size_actual = len(self.swarm) // n_groups_actual
        
        for g in range(n_groups_actual):
            start_idx = g * group_size_actual
            if g == n_groups_actual - 1:
                end_idx = len(self.swarm)
            else:
                end_idx = start_idx + group_size_actual
            
            group_members = self.swarm[start_idx:end_idx]
            leader = max(group_members, key=lambda ind: ind.fitness())
            self.groups.append({'leader': leader, 'members': group_members})

    def rotate_leaders(self):
        for group in self.groups:
            group['leader'] = random.choice(group['members'])

    def evolve(self):
        t = 1
        while t <= self.max_iter:
            if t % self.leader_rotation_freq == 0:
                self.rotate_leaders()
            
            for group in self.groups:
                leader = group['leader']
                for individual in group['members']:
                    if individual is not leader:
                        temp_ind = Individual(self.p)
                        temp_ind.copy(individual)
                        feasible = False
                        attempts = 0
                        while not feasible and attempts < 5:
                            temp_ind.copy(individual)
                            temp_ind.move_towards(leader)
                            feasible = temp_ind.is_feasible()
                            attempts += 1
                        if feasible:
                            individual.copy(temp_ind)
            
            current_best = max(self.swarm, key=lambda ind: ind.fitness())
            if current_best.is_better_than(self.best_global):
                self.best_global.copy(current_best)
            t += 1

    def run(self):
        self.initialize_population()
        self.evolve()
        return self.best_global

def is_dominated(sol1, sol2):
    q1, c1 = sol1[0], sol1[1]
    q2, c2 = sol2[0], sol2[1]
    q_better_or_equal = q2 >= q1
    c_better_or_equal = c2 <= c1
    strictly_better = (q2 > q1) or (c2 < c1)
    return q_better_or_equal and c_better_or_equal and strictly_better

def calculate_pareto_frontier(solutions):
    pareto_solutions = []
    for i, sol in enumerate(solutions):
        is_dominated_flag = False
        for j, other_sol in enumerate(solutions):
            if i != j and is_dominated(sol, other_sol):
                is_dominated_flag = True
                break
        if not is_dominated_flag:
            pareto_solutions.append(sol)
    
    pareto_solutions.sort(key=lambda x: x[1])
    return pareto_solutions

def collect_pareto_solutions():
    """Recopilar soluciones de Pareto ejecutando GGO con diferentes λ"""
    print("Generando soluciones no dominadas...")
    
    lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_solutions = []
    
    for lambda_val in lambda_values:
        print(f"  Procesando λ = {lambda_val}...")
        problem = Problem(lambda_param=lambda_val)
        swarm = Swarm(problem, n_ind=40, iters=500)
        best_individual = swarm.run()
        
        Q, C = problem.get_q_and_c(best_individual.x)
        Z = best_individual.fitness()
        x_vars = best_individual.x[:5]
        q_vars = best_individual.x[5:]
        
        solution = (Q, C, lambda_val, x_vars, q_vars, Z, problem)
        all_solutions.append(solution)
    
    return calculate_pareto_frontier(all_solutions)

def create_detailed_projections(pareto_solutions):
    """Crear proyecciones detalladas de las soluciones no dominadas"""
    
    # Configurar el estilo de matplotlib
    plt.style.use('default')
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Frontera de Pareto principal
    ax1 = plt.subplot(3, 3, 1)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    lambda_colors = {0.1: 'red', 0.3: 'blue', 0.5: 'green', 0.7: 'orange', 0.9: 'purple'}
    
    for i, sol in enumerate(pareto_solutions):
        Q, C, lambda_val = sol[0], sol[1], sol[2]
        color = lambda_colors[lambda_val]
        plt.scatter(C, Q, c=color, s=150, alpha=0.8, label=f'λ={lambda_val}')
    
    # Conectar puntos con línea
    if len(pareto_solutions) > 1:
        pareto_c = [sol[1] for sol in pareto_solutions]
        pareto_q = [sol[0] for sol in pareto_solutions]
        plt.plot(pareto_c, pareto_q, 'k--', alpha=0.5, linewidth=2)
    
    plt.xlabel('Costo C (u.m.)')
    plt.ylabel('Calidad Q (pts)')
    plt.title('Frontera de Pareto')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Variables de decisión x_i
    ax2 = plt.subplot(3, 3, 2)
    x_matrix = np.array([sol[3] for sol in pareto_solutions])
    lambda_vals = [sol[2] for sol in pareto_solutions]
    
    bar_width = 0.15
    x_pos = np.arange(len(lambda_vals))
    
    for i in range(5):
        plt.bar(x_pos + i * bar_width, x_matrix[:, i], bar_width, 
                label=f'x{i+1}', alpha=0.8)
    
    plt.xlabel('Soluciones (λ)')
    plt.ylabel('Cantidades')
    plt.title('Variables de Decisión (x_i)')
    plt.xticks(x_pos + bar_width * 2, [f'λ={lam}' for lam in lambda_vals])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Variables de calidad q_i
    ax3 = plt.subplot(3, 3, 3)
    q_matrix = np.array([sol[4] for sol in pareto_solutions])
    
    for i in range(5):
        plt.bar(x_pos + i * bar_width, q_matrix[:, i], bar_width, 
                label=f'q{i+1}', alpha=0.8)
    
    plt.xlabel('Soluciones (λ)')
    plt.ylabel('Calidades')
    plt.title('Variables de Calidad (q_i)')
    plt.xticks(x_pos + bar_width * 2, [f'λ={lam}' for lam in lambda_vals])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Uso de presupuesto
    ax4 = plt.subplot(3, 3, 4)
    budget_data = []
    for sol in pareto_solutions:
        problem = sol[6]
        budget_usage = problem.get_budget_usage(sol[3] + sol[4])
        budget_data.append([
            budget_usage['TV'][0] / budget_usage['TV'][1] * 100,
            budget_usage['Diario+Revista'][0] / budget_usage['Diario+Revista'][1] * 100,
            budget_usage['Diario+Radio'][0] / budget_usage['Diario+Radio'][1] * 100
        ])
    
    budget_matrix = np.array(budget_data)
    
    for i, label in enumerate(['TV', 'Diario+Revista', 'Diario+Radio']):
        plt.bar(x_pos + i * bar_width, budget_matrix[:, i], bar_width, 
                label=label, alpha=0.8)
    
    plt.xlabel('Soluciones (λ)')
    plt.ylabel('Uso de Presupuesto (%)')
    plt.title('Utilización de Presupuestos')
    plt.xticks(x_pos + bar_width, [f'λ={lam}' for lam in lambda_vals])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Límite')
    
    # 5. Eficiencia Q/C
    ax5 = plt.subplot(3, 3, 5)
    efficiency = [sol[0] / sol[1] for sol in pareto_solutions]
    colors_eff = [lambda_colors[sol[2]] for sol in pareto_solutions]
    
    bars = plt.bar(range(len(lambda_vals)), efficiency, color=colors_eff, alpha=0.8)
    plt.xlabel('Soluciones')
    plt.ylabel('Eficiencia (Q/C)')
    plt.title('Eficiencia por Solución')
    plt.xticks(range(len(lambda_vals)), [f'λ={lam}' for lam in lambda_vals])
    plt.grid(True, alpha=0.3)
    
    # 6. Contribución por medio
    ax6 = plt.subplot(3, 3, 6)
    contrib_data = []
    for sol in pareto_solutions:
        x_vars, q_vars = sol[3], sol[4]
        contributions = [q_vars[i] * x_vars[i] for i in range(5)]
        contrib_data.append(contributions)
    
    contrib_matrix = np.array(contrib_data)
    media_names = ['TV-tarde', 'TV-noche', 'Diario', 'Revista', 'Radio']
    
    for i, media in enumerate(media_names):
        plt.bar(x_pos + i * bar_width, contrib_matrix[:, i], bar_width, 
                label=media, alpha=0.8)
    
    plt.xlabel('Soluciones (λ)')
    plt.ylabel('Contribución a Calidad')
    plt.title('Contribución por Medio')
    plt.xticks(x_pos + bar_width * 2, [f'λ={lam}' for lam in lambda_vals])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Costos unitarios
    ax7 = plt.subplot(3, 3, 7)
    costs_data = []
    for sol in pareto_solutions:
        problem = sol[6]
        costs = problem.get_costs(sol[4])
        costs_data.append(costs)
    
    costs_matrix = np.array(costs_data)
    
    for i in range(5):
        plt.bar(x_pos + i * bar_width, costs_matrix[:, i], bar_width, 
                label=f'c{i+1}', alpha=0.8)
    
    plt.xlabel('Soluciones (λ)')
    plt.ylabel('Costo Unitario')
    plt.title('Costos Unitarios Dinámicos')
    plt.xticks(x_pos + bar_width * 2, [f'λ={lam}' for lam in lambda_vals])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Función objetivo Z
    ax8 = plt.subplot(3, 3, 8)
    z_values = [sol[5] for sol in pareto_solutions]
    colors_z = [lambda_colors[sol[2]] for sol in pareto_solutions]
    
    bars = plt.bar(range(len(lambda_vals)), z_values, color=colors_z, alpha=0.8)
    plt.xlabel('Soluciones')
    plt.ylabel('Función Objetivo Z')
    plt.title('Valores de Z por λ')
    plt.xticks(range(len(lambda_vals)), [f'λ={lam}' for lam in lambda_vals])
    plt.grid(True, alpha=0.3)
    
    # 9. Distribución de inversión
    ax9 = plt.subplot(3, 3, 9)
    investment_data = []
    for sol in pareto_solutions:
        x_vars, q_vars = sol[3], sol[4]
        problem = sol[6]
        costs = problem.get_costs(q_vars)
        investments = [costs[i] * x_vars[i] for i in range(5)]
        investment_data.append(investments)
    
    investment_matrix = np.array(investment_data)
    
    # Gráfico de barras apiladas
    bottom = np.zeros(len(lambda_vals))
    for i, media in enumerate(media_names):
        plt.bar(range(len(lambda_vals)), investment_matrix[:, i], 
                bottom=bottom, label=media, alpha=0.8)
        bottom += investment_matrix[:, i]
    
    plt.xlabel('Soluciones')
    plt.ylabel('Inversión por Medio')
    plt.title('Distribución de Inversión')
    plt.xticks(range(len(lambda_vals)), [f'λ={lam}' for lam in lambda_vals])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('proyecciones_pareto.png', dpi=300, bbox_inches='tight')
    print("Proyecciones guardadas como: proyecciones_pareto.png")
    
    return fig

def create_summary_table(pareto_solutions):
    """Crear tabla resumen de las soluciones no dominadas"""
    print("\n" + "="*100)
    print("TABLA RESUMEN DE SOLUCIONES NO DOMINADAS")
    print("="*100)
    
    headers = ['λ', 'Q', 'C', 'Z', 'Q/C', 'x1', 'x2', 'x3', 'x4', 'x5', 'q1', 'q2', 'q3', 'q4', 'q5']
    print(f"{'':>3} {'λ':>5} {'Q':>8} {'C':>8} {'Z':>8} {'Q/C':>6} {'x1':>3} {'x2':>3} {'x3':>3} {'x4':>3} {'x5':>3} {'q1':>5} {'q2':>5} {'q3':>5} {'q4':>5} {'q5':>5}")
    print("-" * 100)
    
    for i, sol in enumerate(pareto_solutions):
        Q, C, lambda_val, x_vars, q_vars, Z = sol[:6]
        efficiency = Q / C
        print(f"{i+1:>3} {lambda_val:>5.1f} {Q:>8.1f} {C:>8.1f} {Z:>8.1f} {efficiency:>6.3f} " +
              f"{x_vars[0]:>3} {x_vars[1]:>3} {x_vars[2]:>3} {x_vars[3]:>3} {x_vars[4]:>3} " +
              f"{q_vars[0]:>5.1f} {q_vars[1]:>5.1f} {q_vars[2]:>5.1f} {q_vars[3]:>5.1f} {q_vars[4]:>5.1f}")
    
    print("="*100)
    
    # Análisis de restricciones
    print("\nANÁLISIS DE RESTRICCIONES:")
    print("-" * 50)
    for i, sol in enumerate(pareto_solutions):
        problem = sol[6]
        budget_usage = problem.get_budget_usage(sol[3] + sol[4])
        print(f"Solución {i+1} (λ={sol[2]}):")
        for constraint, (used, limit) in budget_usage.items():
            usage_pct = (used / limit) * 100
            status = "✓" if used <= limit else "✗"
            print(f"  {constraint}: {used:.1f}/{limit} ({usage_pct:.1f}%) {status}")
        print()

def main():
    print("PROYECCIÓN DE SOLUCIONES NO DOMINADAS")
    print("="*50)
    
    # Recopilar soluciones de Pareto
    pareto_solutions = collect_pareto_solutions()
    
    # Crear proyecciones visuales
    fig = create_detailed_projections(pareto_solutions)
    
    # Crear tabla resumen
    create_summary_table(pareto_solutions)
    
    # Mostrar gráfico
    plt.show()
    
    print(f"\nSe encontraron {len(pareto_solutions)} soluciones no dominadas")
    print("Archivos generados:")
    print("- proyecciones_pareto.png: Visualizaciones detalladas")
    print("- pareto_ggo.png: Frontera de Pareto básica")

if __name__ == "__main__":
    main() 