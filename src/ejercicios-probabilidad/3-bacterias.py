from scipy.stats import poisson

# Parámetros del problema
event_rate = 30  # Tasa esperada en 50 ml
k = 20           # Número de bacterias deseado

# Aplicación de la fórmula de Poisson
probabilidad = poisson.pmf(k, event_rate)

# Mostrar resultado
print(f"La probabilidad de encontrar exactamente {k} bacterias en 50 ml es: {probabilidad:.4f}")
