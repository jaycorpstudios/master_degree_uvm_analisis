import scipy.stats as stats

# Parámetros de la distribución
mean = 25  # Media
std_deviation = 4  # Desviación estándar

# Calculos de Z para los valores de interés
z1 = (20 - mean) / std_deviation
z2 = (30 - mean) / std_deviation

# Cálculo de probabilidades
p_menor_20 = stats.norm.cdf(z1)  # P(X < 20)
p_mayor_30 = 1 - stats.norm.cdf(z2)  # P(X > 30)
p_entre_20_30 = stats.norm.cdf(z2) - stats.norm.cdf(z1)  # P(20 < X < 30)

print("Probabilidad de que el precio de las acciones")
print(f"sea menor a 20: {p_menor_20:.4f}")
print(f"sea mayor a 30: {p_mayor_30:.4f}")
print(f"esté entre 20 y 30: {p_entre_20_30:.4f}")
