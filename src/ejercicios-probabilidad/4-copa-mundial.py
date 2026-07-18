# Definir parámetros
p = 0.6  # Probabilidad de ganar


def geom_prob(p, k):
    q = 1 - p
    return q * (p ** (k - 1))

# Calcular probabilidades para cada instancia de eliminación
p_octaves = geom_prob(p, 1)
p_quarters = geom_prob(p, 2)
p_semifinals = geom_prob(p, 3)
p_final = geom_prob(p, 4)
p_win = p ** 4  # Si gana todos los partidos

# Mostrar resultados
print(f"Probabilidad de perder en Octavos: {p_octaves:.4f}")
print(f"Probabilidad de perder en Cuartos: {p_quarters:.4f}")
print(f"Probabilidad de perder en Semifinal: {p_semifinals:.4f}")
print(f"Probabilidad de perder en Final: {p_final:.4f}")
print(f"Probabilidad de ganar el torneo: {p_win:.4f}")
