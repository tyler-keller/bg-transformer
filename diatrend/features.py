import numpy as np
import pandas as pd

def insulin_activity(delta_min: float) -> float:
    """calculates insulin activity based on time since bolus."""
    if delta_min < 15:
        return 0.0
    elif delta_min < 60:
        return (delta_min - 15) / 45.0
    elif delta_min < 300:
        return 1.0 - (delta_min - 60) / 240.0
    else:
        return 0.0

def calculate_iob_slow(df: pd.DataFrame, bolus_events: list) -> pd.Series:
    """accurate but slow iterative iob calculation."""
    iob_values = np.zeros(len(df))
    # converting to numpy arrays for speed boost over .loc
    times = df['date'].values
    
    for i, t in enumerate(times):
        t = pd.Timestamp(t)
        total_iob = 0.0
        for dose_time, dose in bolus_events:
            delta_min = (t - dose_time).total_seconds() / 60.0
            if delta_min > 300:
                continue
            if delta_min < 0:
                # assuming chronological order, we can break if future dose
                break 
            total_iob += dose * insulin_activity(delta_min)
        iob_values[i] = total_iob
    return pd.Series(iob_values, index=df.index)

def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """adds sin/cos encoding for hour and minute."""
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    return df