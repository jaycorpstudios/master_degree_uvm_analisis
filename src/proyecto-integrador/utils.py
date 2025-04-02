def show_pulation_by_range(df, start_index, end_index):
  temp_df = df.iloc[start_index:end_index + 1]
  total = temp_df['population'].sum()
  print(f"Range: {start_index} - {end_index} | Total: {total}")

def format_to_millions(x, _p):
    return f'{x/1_000_000:.1f}M'
