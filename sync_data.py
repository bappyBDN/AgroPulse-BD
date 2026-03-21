import pandas as pd
import os

# ফাইলের নামগুলো সেট করা
VERIFIED_CSV = 'Agro_pulsedataset(verified).csv'
UPDATED_CSV = 'AgroPulse_Test_Dataset_Final.csv'

def sync_datasets():
    print(f"🔄 Syncing data from '{UPDATED_CSV}' to '{VERIFIED_CSV}'...")

    # ১. ফাইলগুলো ফোল্ডারে আছে কি না চেক করা
    if not os.path.exists(VERIFIED_CSV):
        print(f"❌ Error: {VERIFIED_CSV} ফাইলটি পাওয়া যায়নি!")
        return
    if not os.path.exists(UPDATED_CSV):
        print(f"❌ Error: {UPDATED_CSV} ফাইলটি পাওয়া যায়নি!")
        return

    # ২. দুটি ডেটাসেট রিড করা
    df_verified = pd.read_csv(VERIFIED_CSV)
    df_updated = pd.read_csv(UPDATED_CSV)

    # Date কলামগুলোকে স্ট্যান্ডার্ড Datetime ফরম্যাটে নেওয়া (যাতে মেলাতে সুবিধা হয়)
    df_verified['Date'] = pd.to_datetime(df_verified['Date'])
    df_updated['Date'] = pd.to_datetime(df_updated['Date'])

    # ৩. Verified ডেটাসেটে থাকা তারিখগুলোর লিস্ট তৈরি করা
    existing_dates = df_verified['Date'].dt.strftime('%Y-%m-%d').tolist()
    
    # ৪. Updated ডেটাসেট থেকে শুধু সেই সারিগুলো বের করা, যেগুলো Verified-এ নেই
    new_data = df_updated[~df_updated['Date'].dt.strftime('%Y-%m-%d').isin(existing_dates)]

    # ৫. নতুন ডেটা যুক্ত করা (যদি থাকে)
    if new_data.empty:
        print("✅ Verified dataset is already up-to-date! No missing dates found.")
    else:
        num_new_rows = len(new_data)
        print(f"📥 Found {num_new_rows} missing day(s). Merging into verified dataset...")
        
        # নতুন ডেটা Verified ডেটাসেটের নিচে যুক্ত করা
        df_verified = pd.concat([df_verified, new_data], ignore_index=True)
        
        # তারিখ অনুযায়ী সাজিয়ে নেওয়া
        df_verified = df_verified.sort_values('Date').reset_index(drop=True)
        
        # ডেট ফরম্যাট আবার আগের মতো (YYYY-MM-DD) করে দেওয়া
        df_verified['Date'] = df_verified['Date'].dt.strftime('%Y-%m-%d')
        
        # ফাইলটি সেভ করা
        df_verified.to_csv(VERIFIED_CSV, index=False)
        
        print(f"🎉 SUCCESS! {VERIFIED_CSV} has been updated with {num_new_rows} new records.")
        print("\n--- Newly Added Dates ---")
        print(new_data['Date'].dt.strftime('%Y-%m-%d').tolist())

if __name__ == "__main__":
    sync_datasets()