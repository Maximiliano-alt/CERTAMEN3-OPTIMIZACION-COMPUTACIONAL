import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as stats

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

def generate_replicas_data():
    lambda_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_replicas = 10
    results = defaultdict(list)
    
    print("Generando datos de réplicas para análisis MSA...")
    
    for lambda_val in lambda_values:
        print(f"Procesando λ = {lambda_val}...")
        
        for replica in range(n_replicas):
            problem = Problem(lambda_param=lambda_val)
            swarm = Swarm(problem, n_ind=40, iters=1000)
            best_individual = swarm.run()
            
            Q, C = problem.get_q_and_c(best_individual.x)
            Z = best_individual.fitness()
            
            results[lambda_val].append((Q, C, Z))
    
    return results

def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - h, mean + h

def create_descriptive_table(results):
    print("\n## ANÁLISIS DE SENSIBILIDAD MULTI-OBJETIVO")
    print("\n### Tabla Descriptiva con Intervalos de Confianza (95%)")
    print("\n| λ | Q̄ | σ(Q) | C̄ | σ(C) | Z̄ | σ(Z) | IC95%(Z) |")
    print("|---|----|----|----|----|----|----|----------|")
    
    statistics = {}
    
    for lambda_val in sorted(results.keys()):
        q_values = [r[0] for r in results[lambda_val]]
        c_values = [r[1] for r in results[lambda_val]]
        z_values = [r[2] for r in results[lambda_val]]
        
        q_mean = np.mean(q_values)
        q_std = np.std(q_values, ddof=1)
        c_mean = np.mean(c_values)
        c_std = np.std(c_values, ddof=1)
        z_mean = np.mean(z_values)
        z_std = np.std(z_values, ddof=1)
        
        z_ci_lower, z_ci_upper = calculate_confidence_interval(z_values)
        
        statistics[lambda_val] = {
            'Q_mean': q_mean, 'Q_std': q_std,
            'C_mean': c_mean, 'C_std': c_std,
            'Z_mean': z_mean, 'Z_std': z_std,
            'Z_ci': (z_ci_lower, z_ci_upper)
        }
        
        print(f"| {lambda_val:.1f} | {q_mean:.1f} | {q_std:.1f} | {c_mean:.1f} | {c_std:.1f} | {z_mean:.1f} | {z_std:.1f} | [{z_ci_lower:.1f}, {z_ci_upper:.1f}] |")
    
    return statistics

def create_sensitivity_plot(statistics):
    lambda_values = sorted(statistics.keys())
    
    q_means = [statistics[lam]['Q_mean'] for lam in lambda_values]
    q_stds = [statistics[lam]['Q_std'] for lam in lambda_values]
    c_means = [statistics[lam]['C_mean'] for lam in lambda_values]
    c_stds = [statistics[lam]['C_std'] for lam in lambda_values]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('λ')
    ax1.set_ylabel('Calidad (Q)', color=color1)
    
    line1 = ax1.plot(lambda_values, q_means, color=color1, linewidth=2, label='Q̄')
    ax1.fill_between(lambda_values, 
                     [q_means[i] - q_stds[i] for i in range(len(lambda_values))],
                     [q_means[i] + q_stds[i] for i in range(len(lambda_values))],
                     color=color1, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Costo (C)', color=color2)
    
    line2 = ax2.plot(lambda_values, c_means, color=color2, linestyle='--', linewidth=2, label='C̄')
    ax2.fill_between(lambda_values,
                     [c_means[i] - c_stds[i] for i in range(len(lambda_values))],
                     [c_means[i] + c_stds[i] for i in range(len(lambda_values))],
                     color=color2, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('Sensibilidad multiobjetivo – prom. de 10 réplicas')
    plt.tight_layout()
    plt.savefig('sensibilidad_QC_lambda.png', dpi=300, bbox_inches='tight')
    plt.close()

def detect_inflection_point(statistics):
    lambda_values = sorted(statistics.keys())
    q_means = [statistics[lam]['Q_mean'] for lam in lambda_values]
    c_means = [statistics[lam]['C_mean'] for lam in lambda_values]
    
    slopes = []
    for i in range(len(lambda_values) - 1):
        delta_q = q_means[i+1] - q_means[i]
        delta_c = c_means[i+1] - c_means[i]
        if delta_c != 0:
            slope = delta_q / delta_c
            slopes.append((lambda_values[i], lambda_values[i+1], slope))
    
    print("\n### Análisis de Punto de Inflexión")
    
    inflection_found = False
    for i, (lam1, lam2, slope) in enumerate(slopes):
        print(f"Entre λ={lam1:.1f} y λ={lam2:.1f}: ΔQ/ΔC = {slope:.3f}")
        
        if slope < 0.5 and not inflection_found:
            print(f"\n**Punto de inflexión:** entre λ = {lam1:.1f} y λ = {lam2:.1f} el costo marginal por punto de calidad se duplica.")
            inflection_found = True
    
    if not inflection_found:
        print("\nNo se detectó punto de inflexión donde ΔQ/ΔC < 0.5 en el rango analizado.")

def main():
    print("=== ANÁLISIS DE SENSIBILIDAD MULTI-OBJETIVO (MSA) ===")
    
    results = generate_replicas_data()
    
    statistics = create_descriptive_table(results)
    
    create_sensitivity_plot(statistics)
    
    detect_inflection_point(statistics)
    
    print(f"\n### Archivos Generados")
    print("- `sensibilidad_QC_lambda.png`: Gráfico de sensibilidad con doble eje Y")
    
    print(f"\n### Resumen")
    print(f"- Total de experimentos: {len(results)} λ × 10 réplicas = {len(results) * 10}")
    print("- Intervalos de confianza calculados al 95%")
    print("- Análisis de punto de inflexión completado")

if __name__ == "__main__":
    main() 