from scipy.stats import poisson

# Definimos las variables que define el problema
total_events = 6  # total de llamadas en 2 horas
minutes_in_hour = 60
total_time = 2 * minutes_in_hour
interval_time = 15  # Intervalo que nos interesa

# Tasa de eventos en el intervalo de tiempo que nos interesa
event_rate = total_events * (interval_time / total_time)

# Probabilidades de X = 0 y X = 1 en dicho intervalo
p_0 = poisson.pmf(0, event_rate)
p_1 = poisson.pmf(1, event_rate)

# Probabilidad de más de 1 llamada
p_more_than_1 = 1 - (p_0 + p_1)

# Mostrar resultados
print(f"Tasa de llamadas en 15 minutos: {event_rate:.2f}")
print(f"Probabilidad de 0 llamadas: {p_0:.4f}")
print(f"Probabilidad de 1 llamada: {p_1:.4f}")
print(f"Probabilidad de más de 1 llamada: {p_more_than_1:.4f}")
