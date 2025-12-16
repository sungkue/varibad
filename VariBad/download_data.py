import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def download_data():
    # Configuration
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
        'DOGE/USDT', 'ADA/USDT', 'TRX/USDT', 'AVAX/USDT', 'LINK/USDT'
    ]
    intervals = ['1h', '15m']
    years = 3
    
    # Initialize exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    # Calculate start time (3 years ago)
    end_time = datetime.now()
    start_time_limit = end_time - timedelta(days=365 * years)
    
    for interval in intervals:
        print(f"\nProcessing interval: {interval}")
        
        # Create directory for interval
        output_dir = os.path.join('data', interval)
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol in symbols:
            print(f"  Downloading {symbol}...")
            
            all_ohlcv = []
            since = int(start_time_limit.timestamp() * 1000)
            
            while True:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, since=since, limit=1000)
                    
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    
                    # Update 'since' to the timestamp of the last candle + 1ms to avoid duplicates
                    # ohlcv is list of [timestamp, open, high, low, close, volume]
                    last_timestamp = ohlcv[-1][0]
                    since = last_timestamp + 1
                    
                    # If we reached the end (or near current time), stop
                    # A safety check if the last candle is very recent
                    if last_timestamp >= int(end_time.timestamp() * 1000):
                        break

                    # Optional: sleep slightly more if needed, though enableRateLimit handles it mostly
                    # time.sleep(0.1) 
                    
                except Exception as e:
                    print(f"    Error fetching {symbol}: {e}")
                    time.sleep(5) # Retry delay
                    continue
            
            if not all_ohlcv:
                print(f"    No data found for {symbol}")
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Filter to strictly 3 years (sometimes fetch_ohlcv with 'since' behaves well, but let's be sure)
            # Actually, since we started fetching from start_time_limit, we just need to clip the end if it goes too far into future (unlikely)
            # or if we want to be precise about the window.
            # Let's just keep what we fetched starting from 'since'.
            
            # Save to Parquet
            safe_symbol = symbol.replace('/', '')
            output_file = os.path.join(output_dir, f"{safe_symbol}_3y.parquet")
            df.to_parquet(output_file, index=False)
            
            print(f"    Saved {len(df)} rows to {output_file}")

if __name__ == "__main__":
    download_data()
