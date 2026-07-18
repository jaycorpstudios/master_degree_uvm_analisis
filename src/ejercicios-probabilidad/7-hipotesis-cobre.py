import math
import scipy.stats as stats

# Parámetros del problema
mean_sample = 23.5  # Media muestral
mu_0 = 23.2   # Media poblacional bajo H0
sigma = 0.24  # Desviación estándar muestral
n = 10        # Tamaño de la muestra

# Cálculo del valor z
z = (mean_sample - mu_0) / (sigma / math.sqrt(n))

# Cálculo de los valores críticos de z para los niveles de significación 0.01 y 0.05
alpha_01 = 0.01
alpha_05 = 0.05

z_crit_01 = stats.norm.ppf(1 - alpha_01 / 2)  # Z crítico para 0.01
z_crit_05 = stats.norm.ppf(1 - alpha_05 / 2)  # Z crítico para 0.05

print(f"Valor z: {z:.4f}")
print(f"Z crítico para 0.01: {z_crit_01:.4f}")
print(f"Z crítico para 0.05: {z_crit_05:.4f}")
