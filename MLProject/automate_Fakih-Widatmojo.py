import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
import os

def load_data(path):
    """Memuat dataset dari path yang diberikan."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def preprocess_data(df):
    """
    Melakukan cleaning, RFM aggregation, dan scaling sesuai logika eksperimen.
    """
    print("--- Memulai Preprocessing ---")
    
    # 1. Konversi Tipe Data
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    # 2. Cleaning Data
    # Hapus baris tanpa CustomerID
    df_clean = df.dropna(subset=["CustomerID"]).copy()
    
    # Ubah CustomerID jadi integer
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
    
    # Filter: Hanya transaksi sukses (Not Returned)
    if 'ReturnStatus' in df_clean.columns:
        df_clean = df_clean[df_clean['ReturnStatus'] == 'Not Returned']
    
    # Filter: Hapus nilai negatif/nol
    df_clean = df_clean[df_clean['Quantity'] > 0]
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    
    # 3. Feature Engineering (TotalPrice)
    # Clip diskon agar logis (0-1)
    df_clean['DiscountClipped'] = df_clean['Discount'].clip(lower=0, upper=1)
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice'] * (1 - df_clean['DiscountClipped'])
    
    print(f"Data bersih: {df_clean.shape[0]} baris.")
    
    # 4. RFM Aggregation
    # Tentukan snapshot date (max date + 1 hari)
    snapshot_date = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)
    
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
        'InvoiceNo': 'count',                                    # Frequency
        'TotalPrice': 'sum'                                      # Monetary
    }).reset_index()
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)
    
    print(f"Terbentuk tabel RFM dengan {len(rfm)} customer.")
    
    # 5. Log Transformation & Scaling
    # Log transform untuk mengurangi skewness
    rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)
    
    # Standard Scaling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    
    # Kembalikan ke DataFrame
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])
    rfm_scaled_df['CustomerID'] = rfm['CustomerID']
    
    return rfm_scaled_df

def main():
    # Path dataset input 
    input_path = 'online-sales-dataset_raw.csv'
    output_path = 'online-sales-dataset_preprocessing.csv'
    
    # Load
    print(f"Loading data dari {input_path}...")
    try:
        df = load_data(input_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Process
    df_processed = preprocess_data(df)
    
    # Save
    print(f"Menyimpan processed data ke {output_path}...")
    df_processed.to_csv(output_path, index=False)
    print("Preprocessing Selesai!")

if __name__ == "__main__":

    main()

