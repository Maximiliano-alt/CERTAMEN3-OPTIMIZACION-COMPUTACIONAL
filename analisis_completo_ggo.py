import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class Problem:
    def __init__(self, lambda_param=0.5):
        self.dim = 10
        self.lambda_param = lambda_param
        self.min_values = [0, 0, 0, 0, 0, 65, 90, 40, 60, 20]
        self.max_values = [15, 10, 25, 4, 30, 85, 95, 60, 80, 30]

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
    def __init__(self, problem, n_ind=40, iters=1000):
        self.p = problem
        self.max_iter = iters
        self.n_individual = n_ind
        self.n_groups = 4
        self.group_size = self.n_individual // self.n_groups
        self.leader_rotation_freq = 20
        self.swarm = []
        self.best_global = None
        self.groups = []
        self.fitness_history = []
        self.all_feasible_individuals = []

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
                self.all_feasible_individuals.append(ind.x.copy())
        
        if len(self.swarm) == 0:
            raise ValueError("No se pudo generar población inicial factible")
        
        self.best_global = max(self.swarm, key=lambda ind: ind.fitness())
        self.fitness_history.append(self.best_global.fitness())
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
                            self.all_feasible_individuals.append(individual.x.copy())
            
            current_best = max(self.swarm, key=lambda ind: ind.fitness())
            if current_best.is_better_than(self.best_global):
                self.best_global.copy(current_best)
            
            self.fitness_history.append(self.best_global.fitness())
            t += 1

    def run(self):
        self.initialize_population()
        self.evolve()
        return self.best_global, self.fitness_history, self.all_feasible_individuals

def is_dominated(sol1, sol2):
    q1, c1 = sol1
    q2, c2 = sol2
    q_better_or_equal = q2 >= q1
    c_better_or_equal = c2 <= c1
    strictly_better = (q2 > q1) or (c2 < c1)
    return q_better_or_equal and c_better_or_equal and strictly_better

def calculate_pareto_frontier(qc_pairs):
    pareto_solutions = []
    for i, sol in enumerate(qc_pairs):
        is_dominated_flag = False
        for j, other_sol in enumerate(qc_pairs):
            if i != j and is_dominated(sol, other_sol):
                is_dominated_flag = True
                break
        if not is_dominated_flag:
            pareto_solutions.append(sol)
    
    pareto_solutions.sort(key=lambda x: x[1])
    return pareto_solutions

def run_experiments():
    lambda_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_replicas = 10
    
    results = defaultdict(list)
    convergence_data = defaultdict(list)
    all_feasible_solutions = []
    
    print("Ejecutando experimentos GGO...")
    
    for lambda_val in lambda_values:
        print(f"Procesando λ = {lambda_val}...")
        
        for replica in range(n_replicas):
            problem = Problem(lambda_param=lambda_val)
            swarm = Swarm(problem, n_ind=40, iters=1000)
            best_individual, fitness_history, feasible_individuals = swarm.run()
            
            Q, C = problem.get_q_and_c(best_individual.x)
            Z = best_individual.fitness()
            
            results[lambda_val].append({
                'Z': Z,
                'Q': Q,
                'C': C,
                'solution': best_individual.x
            })
            
            convergence_data[lambda_val].append(fitness_history)
            
            for ind_x in feasible_individuals[:500]:
                Q_ind, C_ind = problem.get_q_and_c(ind_x)
                all_feasible_solutions.append((Q_ind, C_ind, lambda_val))
    
    return results, convergence_data, all_feasible_solutions

def create_descriptive_table(results):
    print("\n" + "="*90)
    print("TABLA DESCRIPTIVA - ESTADÍSTICAS POR λ")
    print("="*90)
    print(f"{'λ':>5} {'Mejor Z':>10} {'Peor Z':>10} {'Promedio Z':>12} {'Mediana Z':>11} {'Desv.Est Z':>12} {'Media Q':>10} {'Media C':>10}")
    print("-" * 90)
    
    for lambda_val in sorted(results.keys()):
        z_values = [r['Z'] for r in results[lambda_val]]
        q_values = [r['Q'] for r in results[lambda_val]]
        c_values = [r['C'] for r in results[lambda_val]]
        
        mejor_z = max(z_values)
        peor_z = min(z_values)
        promedio_z = np.mean(z_values)
        mediana_z = np.median(z_values)
        desv_z = np.std(z_values)
        media_q = np.mean(q_values)
        media_c = np.mean(c_values)
        
        print(f"{lambda_val:>5.1f} {mejor_z:>10.2f} {peor_z:>10.2f} {promedio_z:>12.2f} {mediana_z:>11.2f} {desv_z:>12.2f} {media_q:>10.2f} {media_c:>10.2f}")
    
    print("="*90)

def create_pareto_plot(all_feasible_solutions):
    if len(all_feasible_solutions) > 5000:
        all_feasible_solutions = random.sample(all_feasible_solutions, 5000)
    
    qc_pairs = [(sol[0], sol[1]) for sol in all_feasible_solutions]
    pareto_frontier = calculate_pareto_frontier(qc_pairs)
    
    plt.figure(figsize=(10, 7))
    
    all_q = [sol[0] for sol in all_feasible_solutions]
    all_c = [sol[1] for sol in all_feasible_solutions]
    
    plt.scatter(all_c, all_q, c='lightcoral', alpha=0.3, s=10)
    
    if len(pareto_frontier) > 1:
        pareto_c = [sol[1] for sol in pareto_frontier]
        pareto_q = [sol[0] for sol in pareto_frontier]
        
        plt.plot(pareto_c, pareto_q, 'red', linewidth=3, label='Frente de Pareto')
        plt.fill_between(pareto_c, pareto_q, alpha=0.2, color='red')
    
    plt.xlabel('Costo Total')
    plt.ylabel('Valorización Total')
    plt.title('Frontera de Pareto - GGO')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pareto_ggo_filled.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return len(qc_pairs), len(pareto_frontier)

def create_convergence_plot(convergence_data):
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(convergence_data)))
    
    for i, (lambda_val, histories) in enumerate(sorted(convergence_data.items())):
        max_len = max(len(h) for h in histories)
        avg_fitness = []
        
        for iteration in range(max_len):
            iteration_values = []
            for history in histories:
                if iteration < len(history):
                    iteration_values.append(history[iteration])
                else:
                    iteration_values.append(history[-1])
            avg_fitness.append(np.mean(iteration_values))
        
        plt.plot(avg_fitness, color=colors[i], linewidth=2, label=f'λ={lambda_val}')
    
    plt.xlabel('Iteración')
    plt.ylabel('Fitness Promedio')
    plt.title('Convergencia GGO - Fitness Promedio por Iteración')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergencia_ggo.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dispersion_plot(all_feasible_solutions):
    plt.figure(figsize=(12, 8))
    
    lambda_values = sorted(set(sol[2] for sol in all_feasible_solutions))
    colors = plt.cm.tab10(np.linspace(0, 1, len(lambda_values)))
    color_map = {lam: colors[i] for i, lam in enumerate(lambda_values)}
    
    for lambda_val in lambda_values:
        lambda_solutions = [sol for sol in all_feasible_solutions if sol[2] == lambda_val]
        if lambda_solutions:
            q_vals = [sol[0] for sol in lambda_solutions]
            c_vals = [sol[1] for sol in lambda_solutions]
            plt.scatter(c_vals, q_vals, c=[color_map[lambda_val]], alpha=0.6, 
                       s=20, label=f'λ={lambda_val}')
    
    plt.xlabel('Costo Total')
    plt.ylabel('Valorización Total')
    plt.title('Dispersión Q-C por λ')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dispersion_ggo.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    results, convergence_data, all_feasible_solutions = run_experiments()
    
    create_descriptive_table(results)
    
    total_feasible, total_pareto = create_pareto_plot(all_feasible_solutions)
    
    create_convergence_plot(convergence_data)
    
    create_dispersion_plot(all_feasible_solutions)
    
    print(f"\nRESUMEN FINAL:")
    print(f"- Puntos factibles totales: {total_feasible}")
    print(f"- Puntos no dominados: {total_pareto}")
    print(f"- Archivos generados:")
    print(f"  • pareto_ggo_filled.png")
    print(f"  • convergencia_ggo.png")
    print(f"  • dispersion_ggo.png")

if __name__ == "__main__":
    main() 