import pandas as pd

df = pd.read_excel('Gen_AI Dataset.xlsx', sheet_name=None)
print('SHEETS:', list(df.keys()))
for k, v in df.items():
    print(f'\nSHEET: {k}')
    print(f'SHAPE: {v.shape}')
    print(f'COLUMNS: {list(v.columns)}')
    print('FIRST 3 ROWS:')
    print(v.head(3).to_string())
