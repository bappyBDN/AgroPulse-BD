import pandas as pd

FILE_NAME = 'F:\\Agro_PalseBD\\AgroPulse_Test_Dataset_Final.csv'


df = pd.read_csv(FILE_NAME)


rename_map = {
    'Temperature_C': 'Temperature',
    'Mean_NDVI': 'NDVI_Clean'
}
df.rename(columns=rename_map, inplace=True)


df.to_csv(FILE_NAME, index=False)

print(f"✅ Columns successfully renamed in {FILE_NAME}")
print("New Columns:", df.columns.tolist())
