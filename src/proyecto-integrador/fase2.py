import pandas as pd
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from theme import color
from utils import format_to_millions, get_weighted_statistics
from scipy.stats import norm
df = pd.read_csv("src/proyecto-integrador/data/mx_population_cleaned.csv", 
                 usecols=['age', 'population', 'men', 'women'],
                 dtype={
                     'age': 'Int64',
                     'population': 'int64',
                     'men': 'int64',
                     'women': 'int64'
                 })

print(df.head(5))
df = df.iloc[:-1] # Eliminamos la última fila debido a que no es claro que edad representa

# group the data by age intervals
intervals = [0, 12, 18, 25, 60, 100]
labels = ["Niñez (0-12)", "Adolescencia (13-18)", "Juventud (19-25)", "Adultez (26-60)", "Vejez (61-100)"]
simplified_labels = ["Niñez", "Adolescencia", "Juventud", "Adultez", "Vejez"]
df_age_population = df.groupby(pd.cut(df['age'], bins=intervals, right=False, labels=labels)).sum()
df_age_population = df_age_population.rename(columns={'age': 'age_sum'})
df_age_population['group'] = df_age_population.index

group_info = pd.DataFrame({
    'group': labels,
    'min_age': [0 if age == 0 else age + 1 for age in intervals[:-1]], 
    'max_age': [i for i in intervals[1:]],  # porque right=False
})
group_info['mid_age'] = (group_info['min_age'] + group_info['max_age']) / 2

df_age_population = df_age_population.merge(group_info, on='group')
df_age_population = df_age_population.set_index('group')

print(df_age_population.head(20))

print('group_info')
print(group_info.head(20))

print("La población total es: ", df_age_population['population'].sum())

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
print('statistics based on the original data and the population column')
statistics_description = {
  'mean': [mean_age],
  'median': [median_age],
  'mode': [mode_age],
  'variance': [variance_age],
  'standard_deviation': [std_age]
}
print(statistics_description)

statistics_per_group = get_weighted_statistics(
  df_age_population,
  value_columns=['population', 'men', 'women'],
  mid_wight_column='mid_age'
)

print('statistics per group')
# pretty print the statistics per group
print(json.dumps(statistics_per_group, indent=4))


# Estimar cota superior en inferior

lower_bound = df_age_population['mid_age'].min()
upper_bound = df_age_population['mid_age'].max()

#  Calcular θ (el valor que maximiza la verosimilitud)

# Calcular la densidad para cada grupo

# Encontrar el estimador de máxima verosimilitud
