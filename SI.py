import random, math, numpy as np

class Problem:
    def __init__(self):
        # Problema de mezcla publicitaria con 5 variables
        self.dim = 5
        # Dominios: x1(0-15), x2(0-10), x3(0-25), x4(0-4), x5(0-30)
        self.min_values = [0, 0, 0, 0, 0]
        self.max_values = [15, 10, 25, 4, 30]
        
        # Costos unitarios a calidad máxima
        self.costs = [200, 350, 80, 120, 20]   # c1, c2, c3, c4, c5
        
        # Calidades a q_max (simplificado)
        self.qualities = [85, 95, 60, 80, 30]  # q1, q2, q3, q4, q5

    def check(self, x):
        """Verificar que la solución respete las tres restricciones presupuestarias"""
        c1, c2, c3, c4, c5 = self.costs
        
        # TV: 200·x1 + 350·x2 ≤ 3800
        tv_budget = c1 * x[0] + c2 * x[1] <= 3800
        
        # Diario+Revista: 80·x3 + 120·x4 ≤ 2800  
        diario_revista = c3 * x[2] + c4 * x[3] <= 2800
        
        # Diario+Radio: 80·x3 + 20·x5 ≤ 3500
        diario_radio = c3 * x[2] + c5 * x[4] <= 3500
        
        return tv_budget and diario_revista and diario_radio

    def fit(self, x):
        """Función objetivo Z = Q - C (calidad - costo)"""
        if not self.check(x):
            return -1e9  # Penalización por infactibilidad
            
        # Calcular calidad total Q
        Q = sum(self.qualities[i] * x[i] for i in range(self.dim))
        
        # Calcular costo total C
        C = sum(self.costs[i] * x[i] for i in range(self.dim))
        
        return Q - C

    def keep_domain(self, idx, val):
        """Redondear y recortar valor al rango válido para la variable idx"""
        lo, hi = self.min_values[idx], self.max_values[idx]
        return int(max(lo, min(hi, round(val))))

class Individual:
    def __init__(self, problem):
        self.p = problem
        self.dimension = self.p.dim
        
        # Generar solución inicial aleatoria en los dominios válidos
        self.x = []
        for i in range(self.dimension):
            self.x.append(random.randint(self.p.min_values[i], self.p.max_values[i]))

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

    def __str__(self):
        return f"x: {self.x}, fitness: {self.fitness()}"

class Swarm:
    def __init__(self, problem, n_ind=40, iters=1000):
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
        self.best_curve = []  # Para graficar convergencia
        
        # Grupos de gansos
        self.groups = []

    def initialize_population(self):
        """Generar población inicial factible"""
        for _ in range(self.n_individual):
            feasible = False
            while not feasible:
                ind = Individual(self.p)
                feasible = ind.is_feasible()
            self.swarm.append(ind)
        
        # Encontrar mejor global inicial
        self.best_global = max(self.swarm, key=lambda ind: ind.fitness())
        self.best_curve.append(self.best_global.fitness())
        
        # Organizar en grupos
        self.organize_groups()

    def organize_groups(self):
        """Organizar gansos en grupos con líderes"""
        self.groups = []
        for g in range(self.n_groups):
            start_idx = g * self.group_size
            end_idx = min(start_idx + self.group_size, self.n_individual)
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
                        while not feasible and attempts < 10:
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
            
            # Registrar progreso
            self.best_curve.append(self.best_global.fitness())
            
            # Mostrar progreso cada 200 iteraciones
            if t % 200 == 0:
                print(f"Iter {t}: Z={self.best_global.fitness():.2f}  x={self.best_global.x}")
            
            t += 1

    def show_final_results(self):
        """Mostrar resultados finales detallados"""
        print("\n" + "="*60)
        print("RESULTADOS FINALES - GGO (Greylag Goose Optimization)")
        print("="*60)
        
        best_x = self.best_global.x
        print(f"Mejor solución x: {best_x}")
        
        # Calcular componentes separadamente
        Q = sum(self.p.qualities[i] * best_x[i] for i in range(self.p.dim))
        C = sum(self.p.costs[i] * best_x[i] for i in range(self.p.dim))
        Z = Q - C
        
        print(f"Calidad total Q: {Q}")
        print(f"Costo total C: {C}")
        print(f"Función objetivo Z (Q-C): {Z}")
        
        # Verificar restricciones
        print(f"\nVerificación de restricciones:")
        tv_used = self.p.costs[0]*best_x[0] + self.p.costs[1]*best_x[1]
        dr_used = self.p.costs[2]*best_x[2] + self.p.costs[3]*best_x[3]
        drad_used = self.p.costs[2]*best_x[2] + self.p.costs[4]*best_x[4]
        
        print(f"TV: {tv_used}/3800 ({'✓' if tv_used <= 3800 else '✗'})")
        print(f"Diario+Revista: {dr_used}/2800 ({'✓' if dr_used <= 2800 else '✗'})")
        print(f"Diario+Radio: {drad_used}/3500 ({'✓' if drad_used <= 3500 else '✗'})")
        
        print(f"\nCurva de convergencia (primeros 10 valores): {self.best_curve[:10]}")
        print(f"Valor final: {self.best_curve[-1]}")

    def optimizer(self):
        """Método principal del algoritmo"""
        print("Iniciando GGO (Greylag Goose Optimization)...")
        print(f"Población: {self.n_individual}, Iteraciones: {self.max_iter}")
        print(f"Grupos: {self.n_groups}, Tamaño grupo: {self.group_size}")
        
        self.initialize_population()
        print(f"Mejor inicial: Z={self.best_global.fitness():.2f}")
        
        self.evolve()
        self.show_final_results()

# Ejecutar el algoritmo
if __name__ == "__main__":
    # Crear problema y ejecutar optimización
    problem = Problem()
    swarm = Swarm(problem, n_ind=40, iters=1000)
    swarm.optimizer()