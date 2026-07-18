import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from scipy.stats import norm
df = pd.read_csv("src/proyecto-integrador/data/mx_population_cleaned.csv", 
                 usecols=['age', 'men', 'women'],
                 dtype={
                     'age': 'Int64',
                     'men': 'int64',
                     'women': 'int64'
                 })

df = df.iloc[:-1] # Eliminamos la última fila debido a que no es claro que edad representa

# Agrupamos por intervalos
intervals = [0, 12, 18, 25, 60, 101]
labels = ["Niñez (0-12)", "Adolescencia (13-18)", "Juventud (19-25)", "Adultez (26-60)", "Vejez (61-100)"]
group_dtype = CategoricalDtype(categories=labels, ordered=True)
simplified_labels = ["Niñez", "Adolescencia", "Juventud", "Adultez", "Vejez"]
df['group'] = pd.cut(df['age'], bins=intervals, right=False, labels=labels).astype(str)
df['group'] = df['group'].astype(group_dtype)


df_age_population = df.groupby('group')[['men', 'women']].sum()
df_age_population['age_sum'] = df.groupby('group')['age'].sum()
df_age_population = df_age_population.reset_index()

print('df_age_population')
print(df_age_population.head(50))


# Calcular la media ponderada
def weighted_mean(data, value_column, weight_column):
    return np.average(data[value_column], weights=data[weight_column])

# Calcular la desviación estándar ponderada
def weighted_std(data, value_column, weight_column):
    average = np.average(data[value_column], weights=data[weight_column])
    variance = np.average((data[value_column] - average) ** 2, weights=data[weight_column])
    return np.sqrt(variance)


# Agrupar y calcular edad promedio para hombres y mujeres por grupo
group_stats = df.groupby('group').apply(lambda g: pd.Series({
    'avg_age_men': weighted_mean(g, 'age', 'men'),
    'std_age_men': weighted_std(g, 'age', 'men'),
    'avg_age_women': weighted_mean(g, 'age', 'women'),
    'std_age_women': weighted_std(g, 'age', 'women')
})).reset_index()

group_stats['sample_age'] = [8, 17, 22 ,32 ,61]
group_stats['avg_age_men'] = group_stats['avg_age_men'].round(2)
group_stats['avg_age_women'] = group_stats['avg_age_women'].round(2)

# Calcular el error absoluto por grupo
group_stats['abs_error_men'] = (group_stats['sample_age'] - group_stats["avg_age_men"]).abs()
group_stats['abs_error_women'] = (group_stats['sample_age'] - group_stats["avg_age_women"]).abs()


# merge con df_age_population
df_age_population = df_age_population.merge(group_stats, on='group')
df_age_population = df_age_population.set_index('group')

# Print table for men and then for women
print('\n\n\n-------------------')
print('Estadísticas por grupo para hombres')
print(df_age_population[['men', 'avg_age_men', 'std_age_men', 'sample_age', 'abs_error_men']].head())
df_age_population[['men', 'avg_age_men', 'std_age_men', 'sample_age', 'abs_error_men']].to_markdown('src/proyecto-integrador/data/men_stats.md', floatfmt=".2f")
print('-------------------\n\n\n')
print('Estadísticas por grupo para mujeres')
print(df_age_population[['women', 'avg_age_women', 'std_age_women', 'sample_age', 'abs_error_women']].head())
df_age_population[['women', 'avg_age_women', 'std_age_women', 'sample_age', 'abs_error_women']].to_markdown('src/proyecto-integrador/data/women_stats.md', floatfmt=".2f")
print('-------------------\n\n\n')


# Estimar una cota superior e inferior para el estimador de la media
df_age_population['m_lower_men'] = df_age_population['avg_age_men'] - df_age_population['std_age_men']
df_age_population['m_upper_men'] = df_age_population['avg_age_men'] + df_age_population['std_age_men']
df_age_population['m_lower_women'] = df_age_population['avg_age_women'] - df_age_population['std_age_women']
df_age_population['m_upper_women'] = df_age_population['avg_age_women'] + df_age_population['std_age_women']

# Calcular la densidad de probabilidad para hombres y mujeres
df_age_population['density_men'] = norm.pdf(
    df_age_population['sample_age'],
    loc=df_age_population['avg_age_men'],
    scale=df_age_population['std_age_men']
)

df_age_population['density_women'] = norm.pdf(
    df_age_population['sample_age'],
    loc=df_age_population['avg_age_women'],
    scale=df_age_population['std_age_women']
)

log_likelihood_men = np.sum(np.log(df_age_population['density_men']))
log_likelihood_women = np.sum(np.log(df_age_population['density_women']))

print(f"log-verosimilitud hombres: {log_likelihood_men:.4f}")
print(f"log-verosimilitud mujeres: {log_likelihood_women:.4f}")



print("Diferentes estimaciones de log-verosimilitud mujeres:")
for delta in [-3, -2, -1, 0, 1, 2, 3]:
    theta_alt = df_age_population['avg_age_women'] + delta
    density = norm.pdf(
        df_age_population['sample_age'],
        loc=theta_alt,
        scale=df_age_population['std_age_women']
    )
    log_likelihood = np.sum(np.log(density))
    print(f"Δ={delta}, -> {log_likelihood:.4f}")

