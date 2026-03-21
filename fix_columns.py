import pandas as pd

FILE_NAME = 'F:\\Agro_PalseBD\\AgroPulse_Test_Dataset_Final.csv'

# ১. ফাইলটি রিড করা
df = pd.read_csv(FILE_NAME)

# ২. কলামের নাম পরিবর্তন করা
rename_map = {
    'Temperature_C': 'Temperature',
    'Mean_NDVI': 'NDVI_Clean'
}
df.rename(columns=rename_map, inplace=True)

# ৩. সেভ করা
df.to_csv(FILE_NAME, index=False)

print(f"✅ Columns successfully renamed in {FILE_NAME}")
print("New Columns:", df.columns.tolist())