import math

def show_pulation_by_range(df, start_index, end_index):
  temp_df = df.iloc[start_index:end_index + 1]
  total = temp_df['population'].sum()
  print(f"Range: {start_index} - {end_index} | Total: {total}")

def format_to_millions(x, _p):
    return f'{x/1_000_000:.1f}M'

import math

def get_weighted_statistics(df, value_columns, mid_wight_column='mid_age'):
    """
    Calcula estadísticas ponderadas (media, mediana, moda, varianza, desviación estándar)
    para cada columna en 'value_columns', usando los valores en 'mid_wight_column' como eje de ponderación.
    
    Retorna un diccionario de estadísticas por columna.
    """
    weights = df[mid_wight_column].to_numpy()
    results = {}

    for column in value_columns:
        values = df[column].to_numpy()
        total = values.sum()

        mean = (weights * values).sum() / total

        cumulative = values.cumsum()
        median_idx = (cumulative >= total / 2).argmax()
        median = weights[median_idx]

        mode_idx = values.argmax()
        mode = weights[mode_idx]

        variance = ((values * (weights - mean) ** 2).sum()) / total
        std_dev = math.sqrt(variance)

        results[column] = {
            'mean': mean,
            'median': median,
            'mode': mode,
            'variance': variance,
            'standard_deviation': std_dev
        }

    return results