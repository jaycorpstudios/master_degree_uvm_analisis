import pandas as pd
import matplotlib.pyplot as plt
from utils import show_pulation_by_range, format_to_millions

df = pd.read_csv("src/project-one/data/mx_population_cleaned.csv", 
                 usecols=['age', 'population'],
                 dtype={
                     'age': 'Int64',
                     'population': 'int64',
                 })

print(df.head(5))
# drop the last row
df = df.iloc[:-1]
total_population = df['population'].sum()
print(f"Total population before grouping: {total_population}")

show_pulation_by_range(df, 0, 9)

# group the data by age by 10 year intervals
intervals = range(0, 120, 10)
df_age_population = df.groupby(pd.cut(df['age'], bins=intervals, right=False)).sum()
df_age_population = df_age_population.drop(columns=['age'])
df_age_population.index = df_age_population.index.map(lambda x: f"{x.left}-{x.right - 1} años" if x.left < 100 else "100 años y más")
print(df_age_population.head(20))

total_population_after_grouping = df_age_population['population'].sum()
print(f"Total population after grouping: {total_population_after_grouping}")

# save the grouped data to a csv file
df_age_population.to_csv('src/project-one/data/mx_population_grouped.csv', index=True)

# flat the grouped data but preserve the index as a column
df_age_population = df_age_population.reset_index()
print(df_age_population.head(20))

# pd.options.display.float_format = '{:,.0f}'.format

# get statistics of the grouped data
print(df_age_population.describe())

# Plot the data on a histogram
plt.figure(figsize=(15, 6))
plt.bar(df_age_population['age'], df_age_population['population'])
plt.xlabel('Edad')
plt.ylabel('Población en Miles')
plt.title('Población por Grupo de Edad')
# Adjust values of the y axis to be more readable
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_to_millions))
plt.ylim(0, df_age_population['population'].max() * 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

media = df_age_population['population'].mean()
q1 = df_age_population['population'].quantile(0.25)
q3 = df_age_population['population'].quantile(0.75)
std = df_age_population['population'].std()

print('Media: ', media)
print('Q1: ', q1)
print('Q3: ', q3)
print('Desviación estándar: ', std)

# Agregar líneas verticales para las estadísticas
# plt.axvline(x=media, color='r', linestyle='--', label='Media')
# plt.axvline(x=q1, color='g', linestyle=':', label='Q1')
# plt.axvline(x=q3, color='g', linestyle=':', label='Q3')
# plt.axvline(x=media + std, color='y', linestyle='-.', label='+1 Desv. Est.')
# plt.axvline(x=media - std, color='y', linestyle='-.', label='-1 Desv. Est.')

# Agregar leyenda
# plt.legend()


plt.show()
