import pandas as pd
from src.theme import color
from utils import format_to_millions
import matplotlib.pyplot as plt
from scipy.stats import linregress, f

df = pd.read_csv("src/proyecto-integrador/data/mx_population_cleaned.csv", 
                 usecols=['age', 'men', 'women'],
                 dtype={
                     'age': 'Int64',
                     'men': 'int64',
                     'women': 'int64'
                 })

df = df.iloc[:-1] # Eliminamos la última fila debido a que no es claro que edad representa

# variable independiente
x = df['age']

# Variables dependientes
y_men = df['men']
y_women = df['women']

# Regresión lineal para hombres
slope_men, intercept_men, r_value_men, p_value_men, std_err_men = linregress(x, y_men)

# Regresión lineal para mujeres
slope_women, intercept_women, r_value_women, p_value_women, std_err_women = linregress(x, y_women)


print((slope_men, intercept_men), (slope_women, intercept_women))

# Graph regression line

plt.figure(figsize=(10, 6))
plt.scatter(x, y_men, label='Hombres', alpha=0.6, color=color['secondary_color_300'])
plt.plot(x, slope_men * x + intercept_men, color=color['uvm_brand_red_600'], label='Regresión hombres', linewidth=2)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_to_millions))
plt.title('Regresión lineal - Población de hombres vs Edad')
plt.xlabel('Edad')
plt.ylabel('Población de hombres')
plt.gca().set_facecolor('none')
plt.gcf().set_facecolor('none')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(x, y_women, label='Mujeres', alpha=0.6, color=color['uvm_brand_red_300'])
plt.plot(x, slope_women * x + intercept_women, color=color['uvm_brand_red_600'], label='Regresión mujeres', linewidth=2)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_to_millions))
plt.title('Regresión lineal - Población de mujeres vs Edad')
plt.xlabel('Edad')
plt.ylabel('Población de mujeres')
plt.gca().set_facecolor('none')
plt.gcf().set_facecolor('none')
plt.legend()
plt.grid(True)
plt.show()

# Calcular F-calculada

n = len(df['age'])
k = 1

# F para hombres
r2_men = r_value_men**2
F_men = (r2_men / k) / ((1 - r2_men) / (n - k - 1))

# F para mujeres
r2_women = r_value_women**2
F_women = (r2_women / k) / ((1 - r2_women) / (n - k - 1))

# f-crítica para α = 0.05
alpha = 0.05
f_critical = f.ppf(1 - alpha, dfn=k, dfd=n - k - 1)

print("Calculos de F para hombres y mujeres")
print(" F Men, F Women, F Crítica")
print(F_men, F_women, f_critical)
