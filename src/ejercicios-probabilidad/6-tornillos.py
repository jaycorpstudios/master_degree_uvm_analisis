from scipy import stats

# Parámetros del problema
n = 60  # tamaño de la muestra
p = 0.01  # probabilidad de que un tornillo sea defectuoso

# a) Probabilidad de que ningún tornillo salga defectuoso (k = 0)
prob_ninguno_defectuoso = stats.binom.pmf(k=0, n=n, p=p)
print(f"a) Probabilidad de que ningún tornillo sea defectuoso: {prob_ninguno_defectuoso:.6f}")

# b) Probabilidad de que exactamente 3 tornillos salgan defectuosos (k = 3)
prob_tres_defectuosos = stats.binom.pmf(k=3, n=n, p=p)
print(f"b) Probabilidad de que exactamente 3 tornillos sean defectuosos: {prob_tres_defectuosos:.6f}")

# c) Probabilidad de que más de 3 tornillos salgan defectuosos (k > 3)
# Usamos la función de distribución acumulada complementaria en lugar de
# la función de distribución acumulada debido a que es más eficiente
prob_mas_de_tres = 1 - stats.binom.cdf(k=3, n=n, p=p)
print(f"c) Probabilidad de que más de 3 tornillos sean defectuosos: {prob_mas_de_tres:.6f}")
