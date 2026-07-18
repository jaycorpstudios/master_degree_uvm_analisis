import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from src.theme import color

saludables = np.array([23, 39, 40, 41, 43, 47, 51, 58, 63, 66, 67, 69, 72])

# Histograma para analizar la distribución de los datos
plt.figure()
plt.hist(saludables, bins='auto', edgecolor=color['secondary_color_500'], color=color['secondary_color_200'])
plt.axvline(np.mean(saludables), color=color['uvm_brand_red_500'], linestyle='dashed', linewidth=1)
plt.title("Histograma - Individuos Saludables", color=color['gray_500'])
plt.xlabel("Volumen de distribución ajustado", color=color['gray_500'])
plt.ylabel("Frecuencia", color=color['gray_500'])

# Intervalo con 95% de confianza que cubra al menos el 95% de la población

n = len(saludables)
mean = np.mean(saludables)
s = np.std(saludables, ddof=1)
confidence_level = 0.95

z = norm.ppf((1 + confidence_level) / 2) # Z para 95%
k = z * np.sqrt((n + 1) / n) # Aproximación de k


lim_inf = mean - k * s
lim_sup = mean + k * s

print(f"Media: {mean:.2f}")
print(f"Desviación estándar: {s:.2f}")
print(f"Intervalo de tolerancia 95%/95%: ({lim_inf:.2f}, {lim_sup:.2f})")

# Intervalo de predicción del 95% para un solo individuo saludable

alpha = 0.05

t_crit = t.ppf(1 - alpha / 2, df=n - 1)
pred_margin = t_crit * s * np.sqrt(1 + 1/n)

pred_inf = mean - pred_margin
pred_sup = mean + pred_margin

print(f"Intervalo de predicción 95%: ({pred_inf:.2f}, {pred_sup:.2f})")
print(f"Ancho del intervalo de tolerancia: {lim_sup - lim_inf:.2f}")
print(f"Ancho del intervalo de predicción: {pred_sup - pred_inf:.2f}")

plt.show()