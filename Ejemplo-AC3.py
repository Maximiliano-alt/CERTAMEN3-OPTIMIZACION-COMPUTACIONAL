domains = {
    'A': [1, 2, 3, 4],
    'B': [1, 2, 3, 4],
    'C': [1, 2, 3],
    'D': [1, 3, 4],
}

constraints = {
    ('A', 'B'): lambda a, b: a == b - 1,
    ('B', 'A'): lambda b, a: b - 1 == a,
    ('A', 'C'): lambda a, c: a != c,
    ('C', 'A'): lambda c, a: c != a,
    ('A', 'D'): lambda a, d: a != d,
    ('D', 'A'): lambda d, a: d != a,
    ('B', 'C'): lambda b, c: abs(b - c) > 1,
    ('C', 'B'): lambda c, b: 1 < abs(b - c),
    ('B', 'D'): lambda b, d: b > d,
    ('D', 'B'): lambda d, b: d < b,
    ('C', 'D'): lambda c, d: c != d,
    ('D', 'C'): lambda d, c: d != c
}

def revise(x, y):
    revised = False
    x_domain = domains[x]
    y_domain = domains[y]
    all_constraints = [
        constraint for constraint in constraints if constraint[0] == x and constraint[1] == y]
    for x_value in x_domain:
        satisfies = False
        for y_value in y_domain:
            for constraint in all_constraints:
                constraint_func = constraints[constraint]
                if constraint_func(x_value, y_value):
                    satisfies = True
        if not satisfies:
            x_domain.remove(x_value)
            revised = True
    return revised

def ac3(arcs):
    queue = arcs[:]
    while queue:
        (x, y) = queue.pop(0)
        revised = revise(x, y)
        if revised:
            neighbors = [neighbor for neighbor in arcs if neighbor[1] == x]
            queue = queue + neighbors

arcs = [
    ('A', 'B'), ('B', 'A'), 
    ('A', 'C'), ('C', 'A'), 
    ('A', 'D'), ('D', 'A'), 
    ('B', 'C'), ('C', 'B'), 
    ('B', 'D'), ('D', 'B'), 
    ('C', 'D'), ('D', 'C')
]

ac3(arcs)

print(domains)