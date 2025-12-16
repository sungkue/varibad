import ccxt
import pandas as pd
import os
import time
import glob
from datetime import datetime, timedelta

def download_dot():
    print("Downloading DOT/USDT...")
    symbol = 'DOT/USDT'
    intervals = ['1h', '15m']
    years = 3
    exchange = ccxt.binance({'enableRateLimit': True})
    end_time = datetime.now()
    start_time_limit = end_time - timedelta(days=365 * years)
    
    for interval in intervals:
        print(f"  Interval: {interval}")
        output_dir = os.path.join('data', interval)
        os.makedirs(output_dir, exist_ok=True)
        
        all_ohlcv = []
        since = int(start_time_limit.timestamp() * 1000)
        
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, since=since, limit=1000)
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                last_timestamp = ohlcv[-1][0]
                since = last_timestamp + 1
                if last_timestamp >= int(end_time.timestamp() * 1000): break
            except Exception as e:
                print(f"    Error: {e}")
                time.sleep(5)
                continue
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        safe_symbol = symbol.replace('/', '')
        output_file = os.path.join(output_dir, f"{safe_symbol}_3y.parquet")
        df.to_parquet(output_file, index=False)
        print(f"    Saved {len(df)} rows to {output_file}")

def interpolate_all():
    print("\nInterpolating missing data for ALL files...")
    base_dir = 'data'
    intervals = ['1h', '15m']
    
    for interval in intervals:
        dir_path = os.path.join(base_dir, interval)
        if not os.path.exists(dir_path): continue
        
        files = glob.glob(os.path.join(dir_path, "*.parquet"))
        for file_path in files:
            try:
                # Load
                df = pd.read_parquet(file_path)
                original_rows = len(df)
                
                # Set index
                df = df.set_index('datetime')
                df = df.sort_index()
                
                # Freq
                freq = '1h' if interval == '1h' else '15min'
                
                # Reindex
                full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
                df_reindexed = df.reindex(full_idx)
                
                missing_count = df_reindexed['close'].isnull().sum()
                
                if missing_count > 0:
                    print(f"  Fixing {os.path.basename(file_path)}: Found {missing_count} missing rows.")
                    
                    if missing_count > 100:
                        print(f"    [WARNING] Large gap detected ({missing_count} rows). Interpolating anyway.")

                    # Interpolate
                    df_interpolated = df_reindexed.interpolate(method='linear')
                    
                    # Fix timestamp
                    df_interpolated['timestamp'] = df_interpolated.index.astype('int64') // 10**6
                    
                    # Save
                    df_final = df_interpolated.reset_index(names='datetime')
                    df_final.to_parquet(file_path, index=False)
                    print(f"    Saved. Rows: {original_rows} -> {len(df_final)}")
                # else:
                #     print(f"  {os.path.basename(file_path)}: OK")
                    
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")

if __name__ == "__main__":
    # download_dot() # Already done
    interpolate_all()
