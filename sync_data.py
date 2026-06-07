import pandas as pd
import os


VERIFIED_CSV = 'Agro_pulsedataset(verified).csv'
UPDATED_CSV = 'AgroPulse_Test_Dataset_Final.csv'

def sync_datasets():
    print(f"🔄 Syncing data from '{UPDATED_CSV}' to '{VERIFIED_CSV}'...")

   
    if not os.path.exists(VERIFIED_CSV):
        print(f"❌ Error: {VERIFIED_CSV} ফাইলটি পাওয়া যায়নি!")
        return
    if not os.path.exists(UPDATED_CSV):
        print(f"❌ Error: {UPDATED_CSV} ফাইলটি পাওয়া যায়নি!")
        return

 
    df_verified = pd.read_csv(VERIFIED_CSV)
    df_updated = pd.read_csv(UPDATED_CSV)

    
    df_verified['Date'] = pd.to_datetime(df_verified['Date'])
    df_updated['Date'] = pd.to_datetime(df_updated['Date'])

    
    existing_dates = df_verified['Date'].dt.strftime('%Y-%m-%d').tolist()
    
   
    new_data = df_updated[~df_updated['Date'].dt.strftime('%Y-%m-%d').isin(existing_dates)]

  
    if new_data.empty:
        print("✅ Verified dataset is already up-to-date! No missing dates found.")
    else:
        num_new_rows = len(new_data)
        print(f"📥 Found {num_new_rows} missing day(s). Merging into verified dataset...")
        
      
        df_verified = pd.concat([df_verified, new_data], ignore_index=True)
        
        
        df_verified = df_verified.sort_values('Date').reset_index(drop=True)
        
        df_verified['Date'] = df_verified['Date'].dt.strftime('%Y-%m-%d')
        
        df_verified.to_csv(VERIFIED_CSV, index=False)
        
        print(f"🎉 SUCCESS! {VERIFIED_CSV} has been updated with {num_new_rows} new records.")
        print("\n--- Newly Added Dates ---")
        print(new_data['Date'].dt.strftime('%Y-%m-%d').tolist())

if __name__ == "__main__":
    sync_datasets()
