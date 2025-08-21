import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.set_index('date')
    return df

def compute_summary(df):
    return {
        "rows": len(df),
        "date_min": df.index.min().date().isoformat(),
        "date_max": df.index.max().date().isoformat(),
        "close_min": float(df['close'].min()),
        "close_max": float(df['close'].max()),
        "adj_close_min": float(df['adj_close'].min()),
        "adj_close_max": float(df['adj_close'].max())
    }
