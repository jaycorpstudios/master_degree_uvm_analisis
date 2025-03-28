import pandas as pd

# Read the mx_population_cleaned.csv file

df = pd.read_csv("src/project-one/data/mx_population_cleaned.csv", usecols=['category', 'population', 'men', 'women'])

# format population, men and women columns as 000,000
# save to csv on external folder: 
df.to_csv("/Users/jaycorp/sandbox/tables-to-design-parser/src/data/initial-data.csv", index=False)