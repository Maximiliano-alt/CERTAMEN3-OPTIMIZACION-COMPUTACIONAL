import random, math

class Problem:
    def __init__(self):
        self.n_objetcts = 5
        self.values = [3, 4, 5, 8, 10]
        self.weights = [2, 3, 4, 5, 9]
        self.W = 10

    def check(self, x):
        acc = 0
        for j in range(self.n_objetcts):
           acc = acc + x[j] * self.weights[j]
        
        return acc <= self.W

    def fit(self, x):
        acc = 0
        for j in range(self.n_objetcts):
           acc = acc + x[j] * self.values[j]
        
        return acc

    def keep_domain(self, v):
        sigmoide = 1 / (1 + math.exp(-v))
        if sigmoide > random.random():
            return 1
        else:
            return 0

class Particle:
    def __init__(self):
        self.p = Problem()
        self.dimension = self.p.n_objetcts

        self.position = [] # asignaciones
        self.velocity = []
        self.p_best = []
        
        # Crear una solución inicial sin validar
        for _ in range(self.dimension):
            self.position.append(random.randint(0, 1))
            self.velocity.append(0)
        
        self.update_p_best()

    def update_p_best(self):
        self.p_best = self.position.copy()
        
    def is_feasible(self):
        return self.p.check(self.position)

    def fitness(self):
        return self.p.fit(self.position)

    def fitness_p_best(self):
        return self.p.fit(self.p_best)

    def is_better_than_p_best(self):
        # > para problemas de maximización
        return self.fitness() > self.fitness_p_best()

    def is_better_than(self, g):
        # > para problemas de maximización
        return self.fitness_p_best() > g.fitness_p_best()

    def move(self, g, theta, alpha, beta):
        for j in range(self.dimension):
            self.velocity[j] = (self.velocity[j] * theta +
                         alpha * random.random() * (g.p_best[j] - self.position[j]) +
                         beta * random.random() * (self.p_best[j] - self.position[j]))
            self.position[j] = self.p.keep_domain(self.velocity[j])

    def copy(self, other):
        if isinstance(other, Particle):
            self.position = other.position.copy()
            self.velocity = other.velocity.copy()
            self.p_best = other.p_best.copy()

    def __str__(self):
        return f"p_best: {self.p_best}, fitness {self.fitness_p_best()}"

class PSO:
    def __init__(self):
        self.max_iter = 25
        self.n_particles = 10
        self.theta = 0.7
        self.alpha = 2
        self.beta = 2
        self.swarm = []
        self.g = None

    def random(self):
        for _ in range(self.n_particles):
            feasible = False
            while not feasible:
                p = Particle()
                feasible = p.is_feasible() 
            self.swarm.append(p)
        
        self.g = self.swarm[0]
        for i in range(1, self.n_particles):
            if self.swarm[i].is_better_than(self.g):
                self.g.copy(self.swarm[i])

        self.show_results(0)

    def evolve(self):
        t = 1
        p = Particle()
        while t <= self.max_iter:
            for i in range(1, self.n_particles):
                feasible = False
                while not feasible:
                    p.copy(self.swarm[i])                    
                    p.move(self.g, self.theta, self.alpha, self.beta)
                    feasible = p.is_feasible()

                self.swarm[i].copy(p)
                if self.swarm[i].is_better_than_p_best():
                    self.swarm[i].update_p_best()
                if self.swarm[i].is_better_than(self.g):
                    self.g.copy(self.swarm[i])

            self.show_results(t)
            t += 1
        
    def show_results(self, t):
        print(f"t: {t}, g: {self.g}")

    def solve(self):
        self.random()
        self.evolve()

# Ejecutar
PSO().solve()