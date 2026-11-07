import pandas as pd
df = pd.read_csv("processed_behavior_data.csv", comment="#", skip_blank_lines=True)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Dtypes:")
print(df.dtypes)
print("\nFirst row:")
print(df.iloc[0])
print("\nUnique values in likely coding columns:")
for col in df.columns:
    if df[col].nunique() < 20:
        print(f"  {col}: {sorted(df[col].dropna().unique().tolist())}")