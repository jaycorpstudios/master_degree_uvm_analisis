from math import comb, sqrt

n = 5  # Número de ensayos
side_effects_p = 0.03  # Probabilidad de efectos secundarios por paciente según el estudio anterior

# Definición de la función para calcular la distribución binomial
def distribucion_binomial(n, k, p):
    return comb(n, k) * (p**k) * ((1 - p)**(n-k))

# a) Probabilidad de que ningún paciente tenga efectos secundarios (X=0)

p_x0 = distribucion_binomial(n = 5, k = 0, p = 0.03) # 0.86
print(f"a) Probabilidad de que ningún paciente tenga efectos secundarios (X=0): {p_x0:.2f}")

# b) Probabilidad de al menos 3 pacientes con efectos secundarios (X >= 3)

p_x3 = distribucion_binomial(n, 3, side_effects_p)
p_x4 = distribucion_binomial(n, 4, side_effects_p)
p_x5 = distribucion_binomial(n, 5, side_effects_p)
p_x_geq_3 = p_x3 + p_x4 + p_x5 # .0003

print(f"b) Probabilidad de al menos 3 pacientes con efectos secundarios (X >= 3): {p_x_geq_3:.4f}")

# c) Número medio de pacientes con efectos secundarios en 100 pacientes

muestra = 100  # Número de pacientes en el caso de media y desviación estándar
mu = muestra * side_effects_p # 3

# d) Desviación estándar en 100 pacientes

sigma = sqrt(muestra * side_effects_p * (1 - side_effects_p)) # 1.71



print(f"c) Número medio de pacientes con efectos secundarios en 100 pacientes: {mu:.2f}")
print(f"d) Desviación estándar en 100 pacientes: {sigma:.2f}")
