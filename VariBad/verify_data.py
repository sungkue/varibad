import pandas as pd
import os
import glob

def verify_data():
    base_dir = 'data'
    intervals = ['1h', '15m']
    
    for interval in intervals:
        dir_path = os.path.join(base_dir, interval)
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue
            
        print(f"\nVerifying {interval} data in {dir_path}:")
        files = glob.glob(os.path.join(dir_path, "*.parquet"))
        
        for file in files:
            try:
                df = pd.read_parquet(file)
                start_date = df['datetime'].min()
                end_date = df['datetime'].max()
                rows = len(df)
                filename = os.path.basename(file)
                print(f"  {filename}: {rows} rows, {start_date} to {end_date}")
            except Exception as e:
                print(f"  Error reading {file}: {e}")

if __name__ == "__main__":
    verify_data()
