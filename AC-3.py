# =========================
#  AC-3 for ad-mix problem
# =========================
#
# Variables  x1 … x5  = # of ads per medium
# Domains    according to inventory limits
# Budgets    use WORST-CASE costs (quality = q_max) so any surviving pair
#            will also be feasible for lower qualities.

# ---------- 1. Domains ----------
domains = {
    'x1': list(range(0, 16)),   # TV-tarde     (0 … 15)
    'x2': list(range(0, 11)),   # TV-noche     (0 … 10)
    'x3': list(range(0, 26)),   # Diario       (0 … 25)
    'x4': list(range(0, 5)),    # Revista      (0 …  4)
    'x5': list(range(0, 31)),   # Radio        (0 … 30)
}

# Worst-case unit costs at q_max
c1, c2, c3, c4, c5 = 200, 350, 80, 120, 20   # ← change here if tu tabla difiere

# ---------- 2. Binary constraints ----------
constraints = {
    # Presupuesto TV
    ('x1', 'x2'): lambda a, b: c1*a + c2*b <= 3800,
    ('x2', 'x1'): lambda b, a: c1*a + c2*b <= 3800,

    # Presupuesto Diario + Revista
    ('x3', 'x4'): lambda a, b: c3*a + c4*b <= 2800,
    ('x4', 'x3'): lambda b, a: c3*a + c4*b <= 2800,

    # Presupuesto Diario + Radio
    ('x3', 'x5'): lambda a, b: c3*a + c5*b <= 3500,
    ('x5', 'x3'): lambda b, a: c3*a + c5*b <= 3500,
}

# ---------- 3. AC-3 helpers ----------
from collections import deque

def revise(Xi, Xj):
    """Return True if we delete a value from domains[Xi]."""
    revised = False
    for x_val in domains[Xi][:]:               # iterate over a *copy*
        # does x_val have *any* supporting value in Xj?
        if not any(constraints[(Xi, Xj)](x_val, y_val) for y_val in domains[Xj]):
            domains[Xi].remove(x_val)
            revised = True
    return revised

def ac3():
    queue = deque(constraints.keys())          # all arcs
    while queue:
        Xi, Xj = queue.popleft()
        if revise(Xi, Xj):
            if not domains[Xi]:
                return False                   # inconsistencia total
            # re-enqueue all neighbours except Xj
            for Xk in [k for (k, _) in constraints if _ == Xi and k != Xj]:
                queue.append((Xk, Xi))
    return True

# ---------- 4. Run AC-3 ----------
if __name__ == "__main__":
    consistent = ac3()
    print("¿Quedó consistente? ->", consistent)
    print("Dominios finales:")
    for var, dom in domains.items():
        print(f"  {var}: {dom}")
