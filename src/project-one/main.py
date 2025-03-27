import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from theme import color
from utils import format_to_millions
from scipy.stats import norm
df = pd.read_csv("src/project-one/data/mx_population_cleaned.csv", 
                 usecols=['age', 'population'],
                 dtype={
                     'age': 'Int64',
                     'population': 'int64',
                 })

print(df.head(5))
df = df.iloc[:-1] # Eliminamos la última fila debido a que no es claro que edad representa

# group the data by age by 10 year intervals
intervals = [0, 12, 18, 25, 60, 100]
labels = ["Niñez (0-12)", "Adolescencia (13-18)", "Juventud (19-25)", "Adultez (26-60)", "Vejez (61-100)"]
simplified_labels = ["Niñez", "Adolescencia", "Juventud", "Adultez", "Vejez"]
df_age_population = df.groupby(pd.cut(df['age'], bins=intervals, right=False, labels=labels)).sum()
df_age_population = df_age_population.drop(columns=['age'])
print(df_age_population.head(20))

#  Realizar la descripción de los datos agrupados
# Calculamos la media ponderada usando population como pesos
mean_age = (df['age'] * df['population']).sum() / df['population'].sum()

# Para la mediana, necesitamos encontrar el valor que divide la población en dos partes iguales
cumsum = df['population'].cumsum()
total = df['population'].sum()
median_age = df['age'][cumsum >= total/2].iloc[0]

# Para la moda, buscamos la edad con mayor población
mode_age = df['age'][df['population'] == df['population'].max()].iloc[0]

# Obtener las medidas de dispersión de varianza y desviación estándar
# Calculamos la varianza ponderada
variance_age = ((df['population'] * (df['age'] - mean_age) ** 2).sum()) / df['population'].sum()

# Calculamos la desviación estándar ponderada
std_age = math.sqrt(variance_age)

# create a dataset with the mean, median, mode, variance and standard deviation and save it to a csv file
statistics_description = {
  'mean': [mean_age],
  'median': [median_age],
  'mode': [mode_age],
  'variance': [variance_age],
  'standard_deviation': [std_age]
}

statistics_description_df = pd.DataFrame(statistics_description)
print(statistics_description_df.head(10))
statistics_description_df.to_csv('src/project-one/data/statistics_description.csv', index=False)

# Histograma con poligono de frecuencias
plt.figure(figsize=(12, 6))
frequencies_by_group, bins_edges, _ = plt.hist(
  df['age'],
  bins=intervals,
  weights=df['population'],
  edgecolor=color['secondary_color_700'],
  color=color['secondary_color_500'],
  alpha=0.5,
  label="Distribución por Edad"
)

# Graficar el polígono de frecuencias
midpoints = [(intervals[i] + intervals[i+1]) / 2 for i in range(len(intervals) - 1)]
plt.plot(midpoints, frequencies_by_group, marker='o', linestyle='-', color=color['uvm_brand_red_500'], label="Tendencia de Población")

plt.title("Distribución y Tendencia de la Población por Grupos de Edad")
plt.xlabel("Edad")
plt.ylabel("Frecuencia (Población)")
plt.xticks(intervals)
ax = plt.gca()

# Formato y ajustes visuales
label_positions = [(intervals[i] + intervals[i+1])/2 for i in range(len(intervals)-1)]
ax.set_xticks(label_positions, minor=True)
ax.set_xticklabels(simplified_labels, minor=True, rotation=90, ha='center', color=color['secondary_color_400'])
plt.subplots_adjust(bottom=0.2)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_to_millions))
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.gca().set_facecolor('none')
plt.gcf().set_facecolor('none')
plt.savefig('src/project-one/plots/histogram_and_polygon_of_population.png', dpi=300)

# Ojiva de la población
cumulative_frequencies = np.cumsum(frequencies_by_group)
plt.figure(figsize=(12, 6))
plt.plot(midpoints, cumulative_frequencies, marker='o', linestyle='-', color=color['secondary_color_500'], label="Frecuencias Acumuladas")

# Agregar líneas verticales para los rangos de edad
for interval in intervals:
    plt.axvline(x=interval, color='gray', linestyle='--', alpha=0.3)

# Agregar etiquetas de población acumulada en cada punto
for i, (mid, freq) in enumerate(zip(midpoints, cumulative_frequencies)):
    plt.annotate(f'{freq/1_000_000:.1f}M', 
                xy=(mid, freq),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='bottom')

# Configurar etiquetas
plt.xlabel("Edad")
plt.ylabel("Población Acumulada")
plt.title("Distribución Acumulada de la Población por Edad")

# Configurar el eje X
plt.xticks(intervals)  # Mantener los números originales
ax = plt.gca()

# Agregar etiquetas de categorías
label_positions = [(intervals[i] + intervals[i+1])/2 for i in range(len(intervals)-1)]
ax.set_xticks(label_positions, minor=True)
ax.set_xticklabels(simplified_labels, minor=True, rotation=90, ha='center', color=color['secondary_color_400'])

# Formatear el eje Y en millones
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_to_millions))

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.subplots_adjust(bottom=0.2)
plt.gca().set_facecolor('none')
plt.gcf().set_facecolor('none')
plt.savefig('src/project-one/plots/cumulative_frequency_polygon.png', dpi=300)

plt.show()


# Random values for each group
target_ages = [8, 17, 22, 32, 61]


# Calcular la propabilidad de que un individuo de la población tenga una edad en el rango de los target_ages 
# usando la distribución normal y basado en el valor de z para cada grupo
means_per_group = []
standard_deviations_per_group = []
z_values_per_group = []
probabilities_per_group = []

for i in range(len(intervals)-1):
    range_values = (df['age'] >= intervals[i]) & (df['age'] < intervals[i+1])
    group_mean = (df[range_values]['age'] * df[range_values]['population']).sum() / df[range_values]['population'].sum()
    means_per_group.append(group_mean)
    group_standard_deviation = math.sqrt(((df[range_values]['age'] - group_mean) ** 2 * df[range_values]['population']).sum() / df[range_values]['population'].sum())
    standard_deviations_per_group.append(group_standard_deviation)
    z_values_per_group.append((target_ages[i] - group_mean) / group_standard_deviation)
    probabilities_per_group.append(norm.cdf(z_values_per_group[i]) * 100)

mean_per_group = pd.Series(means_per_group, index=simplified_labels)
standard_deviation_per_group = pd.Series(standard_deviations_per_group, index=simplified_labels)
z_values_per_group = pd.Series(z_values_per_group, index=simplified_labels)
probabilities_per_group = pd.Series(probabilities_per_group, index=simplified_labels)

df_age_population_with_statistics = df_age_population.copy()
df_age_population_with_statistics['mean'] = mean_per_group.values
df_age_population_with_statistics['standard_deviation'] = standard_deviation_per_group.values
df_age_population_with_statistics['z_value'] = z_values_per_group.values
df_age_population_with_statistics['probability'] = probabilities_per_group.values

print("DataFrame con estadísticas de cada grupo:")
print(df_age_population_with_statistics)

# Guardar el dataframe en un csv
df_age_population_with_statistics.to_csv('src/project-one/data/df_age_population_with_statistics.csv', index=False)

