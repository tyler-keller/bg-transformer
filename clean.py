import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from diatrend.features import calculate_iob_slow, add_cyclic_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=int, default=5)
    parser.add_argument('--input_dir', default='data/original')
    parser.add_argument('--output_dir', default='data/cleaned')
    args = parser.parse_args()

    print(f"processing subject {args.subject_id}...")
    
    # 1. read data
    file_path = os.path.join(args.input_dir, f'Subject{args.subject_id}.xlsx')
    xls = pd.read_excel(file_path, None)
    cgm_df = xls['CGM']
    bolus_df = xls['Bolus']

    # 2. standardize time
    cgm_df['date'] = pd.to_datetime(cgm_df['date']).dt.floor('5min')
    bolus_df['date'] = pd.to_datetime(bolus_df['date']).dt.floor('5min')
    
    # 3. interpolate gaps <= 60 mins
    print(cgm_df.head())
    print(cgm_df.duplicated('date').sum())
    print(cgm_df[cgm_df.duplicated('date', keep=False)].sort_values('date').head())
    print(cgm_df.groupby('date').size().head())
    # keeping the mean of duplicates dates
    cgm_df = cgm_df.set_index('date').resample('5min').mean()
    # linear interpolate valid gaps (limit=12 means 1 hour of 5 min chunks)
    cgm_df['mg/dl'] = cgm_df['mg/dl'].interpolate(method='linear', limit=12)
    cgm_df = cgm_df.reset_index()

    # 4. merge bolus
    bolus_agg = bolus_df[['date', 'normal']].groupby('date', as_index=False).sum()
    merged = pd.merge(cgm_df, bolus_agg, on='date', how='outer').sort_values('date')
    
    # drop where we still don't have cgm data after interpolation
    merged = merged.dropna(subset=['mg/dl']).reset_index(drop=True)
    merged['normal'] = merged['normal'].fillna(0)

    # 5. calculate iob
    print("calculating iob...")
    # get list of (time, dose) tuples for efficiency
    bolus_events = merged[merged['normal'] > 0][['date', 'normal']].values.tolist()
    merged['iob'] = calculate_iob_slow(merged, bolus_events)

    # 6. time features
    merged['hour'] = merged['date'].dt.hour
    merged['minute'] = merged['date'].dt.minute
    merged['dayofweek'] = merged['date'].dt.dayofweek
    
    # 7. segmentation (break on gaps > 5 mins)
    merged['time_diff'] = merged['date'].diff()
    # gap > 5min + epsilon to catch floating point oddities
    breaks = merged['time_diff'] > pd.Timedelta('5min')
    merged['segment_id'] = breaks.cumsum()

    # 8. feature engineering & save
    save_path = os.path.join(args.output_dir, str(args.subject_id))
    os.makedirs(save_path, exist_ok=True)
    
    segment_counts = merged['segment_id'].value_counts()
    valid_segments = segment_counts[segment_counts >= 12].index # keep > 1 hour
    
    count = 0
    for seg_id in tqdm(valid_segments, desc="saving segments"):
        segment = merged[merged['segment_id'] == seg_id].copy().sort_values('date')
        
        # cyclic encoding
        segment = add_cyclic_features(segment)
        
        # one-hot day of week
        for d in range(7):
            segment[f'dow_{d}'] = (segment['dayofweek'] == d).astype(float)

        # select final columns
        cols = ['mg/dl', 'iob', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos'] + [f'dow_{d}' for d in range(7)]
        
        # float32 for model
        final_df = segment[cols].astype(np.float32)
        
        final_df.to_csv(os.path.join(save_path, f'{count}.csv'), index=False)
        count += 1

if __name__ == '__main__':
    main()