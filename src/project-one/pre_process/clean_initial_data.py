import pandas as pd

# Load and clean data
def load_and_clean_population_data(path: str) -> pd.DataFrame:
    """
    Carga y limpia los datos de población de México.
    Lee el archivo CSV, remueve espacios y convierte columnas numéricas.

    Returns:
        pd.DataFrame: DataFrame limpio con datos de población
    """
    df = pd.read_csv(path)
    cols_to_convert = ['population', 'men', 'women', 'm_w_ratio']

    # Remover espacios y convertir a numérico
    df[cols_to_convert] = df[cols_to_convert].apply(
        lambda x: x.str.replace(' ', '')
        if pd.api.types.is_string_dtype(x) else x
    )
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric)
    return df


df = load_and_clean_population_data("src/project-one/data/mx_population.csv")
df = df.rename(columns={'age': 'category'})
df['age'] = pd.to_numeric(df['category'].str.extract(r'(\d+) años')[0], errors='coerce').astype('Int64')
df['age'] = df['age'].fillna(pd.NA)
df = df.drop(columns=['m_w_ratio'])
df['ratio_men_women_percentage'] = (df['men'] / df['women']) * 100
df = df[['category', 'age', 'population', 'men', 'women', 'ratio_men_women_percentage']]



print("First few rows of the dataframe:")
print(df.head())

# Display the shape of the dataframe
print("Shape of the dataframe:")
print(df.shape)

# Display the column names
print("Column names:")
print(df.columns)

# Display the data types of the columns
print("Data types of the columns:")
print(df.dtypes)


# save as csv
df.to_csv("src/project-one/data/mx_population_cleaned.csv", index=False)
