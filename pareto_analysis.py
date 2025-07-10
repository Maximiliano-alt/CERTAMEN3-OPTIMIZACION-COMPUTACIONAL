import random
import math
import numpy as np
import matplotlib.pyplot as plt

class Problem:
    def __init__(self, lambda_param=0.5):
        # Problema extendido: [x1,x2,x3,x4,x5, q1,q2,q3,q4,q5]
        self.dim = 10
        self.lambda_param = lambda_param
        
        # Dominios para x_i: [0-15], [0-10], [0-25], [0-4], [0-30]
        # Dominios para q_i: [65-85], [90-95], [40-60], [60-80], [20-30]
        self.min_values = [0, 0, 0, 0, 0, 65, 90, 40, 60, 20]
        self.max_values = [15, 10, 25, 4, 30, 85, 95, 60, 80, 30]

    def get_costs(self, q):
        """Calcular costos c_i a partir de calidades q_i"""
        c1 = 2 * q[0] + 30      # c1 = 2*q1 + 30
        c2 = 10 * q[1] - 600    # c2 = 10*q2 - 600
        c3 = 2 * q[2] - 40      # c3 = 2*q3 - 40
        c4 = q[3] + 40          # c4 = q4 + 40
        c5 = q[4] - 10          # c5 = q5 - 10
        return [c1, c2, c3, c4, c5]

    def check(self, x):
        """Verificar restricciones usando costos calculados dinámicamente"""
        # Extraer x_i y q_i
        x_vars = x[:5]  # x1, x2, x3, x4, x5
        q_vars = x[5:]  # q1, q2, q3, q4, q5
        
        # Calcular costos c_i
        costs = self.get_costs(q_vars)
        c1, c2, c3, c4, c5 = costs
        
        # Verificar restricciones presupuestarias
        tv_budget = c1 * x_vars[0] + c2 * x_vars[1] <= 3800
        diario_revista = c3 * x_vars[2] + c4 * x_vars[3] <= 2800
        diario_radio = c3 * x_vars[2] + c5 * x_vars[4] <= 3500
        
        return tv_budget and diario_revista and diario_radio

    def get_q_and_c(self, x):
        """Obtener Q y C por separado"""
        x_vars = x[:5]  # x1, x2, x3, x4, x5
        q_vars = x[5:]  # q1, q2, q3, q4, q5
        
        # Calcular calidad total Q
        Q = sum(q_vars[i] * x_vars[i] for i in range(5))
        
        # Calcular costo total C
        costs = self.get_costs(q_vars)
        C = sum(costs[i] * x_vars[i] for i in range(5))
        
        return Q, C

    def fit(self, x):
        """Función objetivo Z = λ*Q - (1-λ)*C"""
        if not self.check(x):
            return -1e9  # Penalización por infactibilidad
            
        Q, C = self.get_q_and_c(x)
        return self.lambda_param * Q - (1 - self.lambda_param) * C

    def keep_domain(self, idx, val):
        """Redondear y recortar valor al rango válido para la variable idx"""
        lo, hi = self.min_values[idx], self.max_values[idx]
        if idx < 5:  # Variables x_i (enteras)
            return int(max(lo, min(hi, round(val))))
        else:  # Variables q_i (reales)
            return max(lo, min(hi, val))

class Individual:
    def __init__(self, problem):
        self.p = problem
        self.dimension = self.p.dim
        
        # Generar solución inicial aleatoria en los dominios válidos
        self.x = []
        for i in range(self.dimension):
            if i < 5:  # Variables x_i (enteras)
                self.x.append(random.randint(self.p.min_values[i], self.p.max_values[i]))
            else:  # Variables q_i (reales)
                self.x.append(random.uniform(self.p.min_values[i], self.p.max_values[i]))

    def is_feasible(self):
        """Verificar si la solución cumple las restricciones"""
        return self.p.check(self.x)

    def fitness(self):
        """Evaluar la función objetivo"""
        return self.p.fit(self.x)

    def is_better_than(self, other):
        """Comparar fitness con otro individuo (maximización)"""
        return self.fitness() > other.fitness()

    def move_towards(self, leader, sigma=1.0):
        """Operador de movimiento GGO hacia el líder"""
        for i in range(self.dimension):
            r1 = random.random()  # Factor de atracción hacia líder
            r2 = random.random()  # Factor de ruido gaussiano
            
            # Movimiento: x_i ← x_i + r1·(leader_i - x_i) + r2·N(0,σ)
            attraction = r1 * (leader.x[i] - self.x[i])
            noise = r2 * random.gauss(0, sigma)
            new_val = self.x[i] + attraction + noise
            
            # Aplicar keep_domain para mantener en rango válido
            self.x[i] = self.p.keep_domain(i, new_val)

    def copy(self, other):
        """Copiar otro individuo"""
        if isinstance(other, Individual):
            self.x = other.x.copy()

class Swarm:
    def __init__(self, problem, n_ind=40, iters=500):
        self.p = problem
        self.max_iter = iters
        self.n_individual = n_ind
        
        # Parámetros específicos de GGO
        self.n_groups = 4
        self.group_size = self.n_individual // self.n_groups
        self.leader_rotation_freq = 20  # Rotar líder cada 20 iteraciones
        
        # Inicializar enjambre
        self.swarm = []
        self.best_global = None
        
        # Grupos de gansos
        self.groups = []

    def initialize_population(self):
        """Generar población inicial factible"""
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
        
        # Encontrar mejor global inicial
        self.best_global = max(self.swarm, key=lambda ind: ind.fitness())
        
        # Organizar en grupos
        self.organize_groups()

    def organize_groups(self):
        """Organizar gansos en grupos con líderes"""
        self.groups = []
        n_groups_actual = min(self.n_groups, len(self.swarm))
        group_size_actual = len(self.swarm) // n_groups_actual
        
        for g in range(n_groups_actual):
            start_idx = g * group_size_actual
            if g == n_groups_actual - 1:  # Último grupo toma el resto
                end_idx = len(self.swarm)
            else:
                end_idx = start_idx + group_size_actual
            
            group_members = self.swarm[start_idx:end_idx]
            
            # El mejor del grupo es el líder inicial
            leader = max(group_members, key=lambda ind: ind.fitness())
            self.groups.append({
                'leader': leader,
                'members': group_members
            })

    def rotate_leaders(self):
        """Rotar líderes dentro de cada grupo"""
        for group in self.groups:
            # Seleccionar nuevo líder aleatoriamente del grupo
            group['leader'] = random.choice(group['members'])

    def evolve(self):
        """Proceso evolutivo principal"""
        t = 1
        
        while t <= self.max_iter:
            # Rotar líderes cada cierta frecuencia
            if t % self.leader_rotation_freq == 0:
                self.rotate_leaders()
            
            # Movimiento de gansos en cada grupo
            for group in self.groups:
                leader = group['leader']
                for individual in group['members']:
                    if individual is not leader:
                        # Crear individuo temporal para movimiento
                        temp_ind = Individual(self.p)
                        temp_ind.copy(individual)
                        
                        # Intentar movimiento hasta encontrar solución factible
                        feasible = False
                        attempts = 0
                        while not feasible and attempts < 5:
                            temp_ind.copy(individual)
                            temp_ind.move_towards(leader)
                            feasible = temp_ind.is_feasible()
                            attempts += 1
                        
                        if feasible:
                            individual.copy(temp_ind)
            
            # Actualizar mejor global
            current_best = max(self.swarm, key=lambda ind: ind.fitness())
            if current_best.is_better_than(self.best_global):
                self.best_global.copy(current_best)
            
            t += 1

    def get_best_solution(self):
        """Obtener la mejor solución encontrada"""
        return self.best_global

    def run(self):
        """Ejecutar el algoritmo completo"""
        self.initialize_population()
        self.evolve()
        return self.get_best_solution()

def is_dominated(sol1, sol2):
    """Verificar si sol1 es dominada por sol2 (retorna True si sol1 es dominada)"""
    q1, c1 = sol1[0], sol1[1]
    q2, c2 = sol2[0], sol2[1]
    
    # sol2 domina a sol1 si sol2.Q >= sol1.Q y sol2.C <= sol1.C
    # y al menos una desigualdad es estricta
    q_better_or_equal = q2 >= q1
    c_better_or_equal = c2 <= c1
    strictly_better = (q2 > q1) or (c2 < c1)
    
    return q_better_or_equal and c_better_or_equal and strictly_better

def calculate_pareto_frontier(solutions):
    """Calcular la frontera de Pareto"""
    pareto_solutions = []
    
    for i, sol in enumerate(solutions):
        is_dominated_flag = False
        for j, other_sol in enumerate(solutions):
            if i != j and is_dominated(sol, other_sol):
                is_dominated_flag = True
                break
        
        if not is_dominated_flag:
            pareto_solutions.append(sol)
    
    # Ordenar por costo ascendente
    pareto_solutions.sort(key=lambda x: x[1])
    return pareto_solutions

def main():
    print("Generando frontera de Pareto con GGO...")
    
    # Valores de lambda para el barrido
    lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_solutions = []
    
    # Colores para cada lambda
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    color_map = {lam: colors[i] for i, lam in enumerate(lambda_values)}
    
    for lambda_val in lambda_values:
        print(f"Ejecutando GGO con λ = {lambda_val}...")
        
        # Crear problema con lambda específico
        problem = Problem(lambda_param=lambda_val)
        
        # Ejecutar enjambre
        swarm = Swarm(problem, n_ind=40, iters=500)
        best_individual = swarm.run()
        
        # Obtener Q y C de la mejor solución
        Q, C = problem.get_q_and_c(best_individual.x)
        Z = best_individual.fitness()
        
        # Guardar solución
        x_vars = best_individual.x[:5]
        q_vars = best_individual.x[5:]
        solution = (Q, C, lambda_val, x_vars, q_vars, Z)
        all_solutions.append(solution)
        
        print(f"  λ={lambda_val}: Q={Q:.2f}, C={C:.2f}, Z={Z:.2f}")
    
    # Calcular frontera de Pareto
    pareto_solutions = calculate_pareto_frontier(all_solutions)
    
    # Crear gráfico
    plt.figure(figsize=(12, 8))
    
    # Scatter de todas las soluciones con leyenda única por lambda
    legend_added = set()
    for sol in all_solutions:
        Q, C, lambda_val = sol[0], sol[1], sol[2]
        color = color_map[lambda_val]
        label = f'λ={lambda_val}' if lambda_val not in legend_added else ""
        if lambda_val not in legend_added:
            legend_added.add(lambda_val)
        plt.scatter(C, Q, c=color, s=100, alpha=0.7, label=label)
    
    # Línea de Pareto
    if len(pareto_solutions) > 1:
        pareto_c = [sol[1] for sol in pareto_solutions]
        pareto_q = [sol[0] for sol in pareto_solutions]
        plt.plot(pareto_c, pareto_q, 'k-', linewidth=2, alpha=0.8, label='Frontera Pareto')
    
    # Puntos Pareto con marcadores más grandes
    for sol in pareto_solutions:
        Q, C, lambda_val, _, _, Z = sol
        plt.scatter(C, Q, c='black', s=200, marker='o', edgecolors='white', linewidth=2)
        # Anotar λ y Z
        plt.annotate(f'λ={lambda_val}\nZ={Z:.1f}', 
                    (C, Q), xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=9, ha='left')
    
    plt.xlabel('Costo C (u.m.)', fontsize=12)
    plt.ylabel('Calidad Q (pts)', fontsize=12)
    plt.title('Frontera de Pareto – Mezcla publicitaria (GGO)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Guardar imagen
    plt.tight_layout()
    plt.savefig('pareto_ggo.png', dpi=300, bbox_inches='tight')
    print("\nGráfico guardado como: pareto_ggo.png")
    
    # Imprimir puntos Pareto
    print("\nPuntos de la frontera de Pareto:")
    print("[ (Q, C, λ) ]")
    for sol in pareto_solutions:
        Q, C, lambda_val = sol[0], sol[1], sol[2]
        print(f"  ({Q:.2f}, {C:.2f}, {lambda_val})")
    
    plt.show()

if __name__ == "__main__":
    main() 