import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(prog='clean data')
parser.add_argument('--to_csv', action='store_true', help='write cleaned data to csv?')
parser.add_argument('--verbose', action='store_true', help='print stuff?')
parser.add_argument('--subject_id', default=5, required=True, type=int, help='which subject are we cleaning? default: 5')
parser.add_argument('--fast_iob', action='store_true', help='use less accurate fast convolve? warning: doesn\'t work')
parser.add_argument('--sanity_check', action='store_true', help='only run for a single segment for model training sanity checking?')
args = parser.parse_args()

def print_counts(df_col: pd.Series):
    print(df_col.value_counts().reset_index(drop=False).sort_values('count', ascending=False))

def print_info(cgm_df: pd.DataFrame, bolus_df: pd.DataFrame, merged: pd.DataFrame):
    print(bolus_df.head().to_markdown())
    print(cgm_df.head().to_markdown())
    print(merged.head().to_markdown())

    print_counts(cgm_df['date'])
    print_counts(bolus_df['date'])
    print_counts(merged['date'])

    print((~bolus_df['normal'].isna()).sum())
    print((~merged['normal'].isna()).sum())

    print(bolus_df.head(25).to_markdown())
    print(merged[~merged['normal'].isna()].head(25).to_markdown())

    print(merged['time_diff'].value_counts())
    print(f"NUMBER OF TRAINABLE SEGMENTS: {(merged['time_diff'] > pd.Timedelta('1 hour')).sum()}")

    breaks = merged['time_diff'] != pd.Timedelta('5min')
    breaks.iloc[0] = True
    merged['segment_id'] = breaks.cumsum()

    segment_lengths = merged.groupby('segment_id').size().reset_index(name='n_points')
    segment_lengths['duration_minutes'] = segment_lengths['n_points'] * 5
    print(f"LENGTH OF TRAINABLE SEGMENTS:")
    print_counts(segment_lengths['duration_minutes'])

# read original data
print(f'READING CSV FOR SUBJECT {args.subject_id}')
map = pd.read_excel(f'../data/original/Subject{args.subject_id}.xlsx', None)
cgm_df = map['CGM']
bolus_df = map['Bolus']

print(f'INTERPOLATING GAPS < 60 MINS')
# clean and fill gaps in CGM
cgm_df['date'] = pd.to_datetime(cgm_df['date']).dt.floor('5min')
cgm_df.drop_duplicates(subset=['date'], keep='first', inplace=True)
cgm_df['time_diff'] = cgm_df['date'].diff()
cgm_df = cgm_df.reset_index(drop=True)

new_rows = []
for i in range(1, len(cgm_df)):
    gap = cgm_df.loc[i, 'time_diff']
    if pd.Timedelta(minutes=5) < gap <= pd.Timedelta(minutes=60):
        start = cgm_df.loc[i-1, 'date']
        end = cgm_df.loc[i, 'date']
        new_dates = pd.date_range(start + pd.Timedelta(minutes=5), end - pd.Timedelta(minutes=5), freq='5min')
        for d in new_dates:
            new_rows.append({'date': d, 'mg/dl': np.nan})

cgm_df = pd.concat([cgm_df, pd.DataFrame(new_rows)], ignore_index=True)
cgm_df = cgm_df.sort_values('date').reset_index(drop=True)
cgm_df['mg/dl'] = cgm_df['mg/dl'].interpolate(limit_direction='both')

# clean bolus
bolus_df['date'] = pd.to_datetime(bolus_df['date']).dt.floor('5min')
bolus_df = bolus_df[['date', 'normal']]
bolus_df = bolus_df.groupby('date', as_index=False).sum()

# restrict cgm range
start = bolus_df['date'].min()
end = bolus_df['date'].max()
cgm_df = cgm_df[(cgm_df['date'] >= start) & (cgm_df['date'] <= end)]

# merge cgm + bolus
print(f'MERGING CGM AND BOLUS DATAFRAMES')
merged = pd.merge(cgm_df, bolus_df, on='date', how='outer')
merged = merged.sort_values('date')
merged = merged.dropna(subset=['mg/dl'])

# add time-based features
merged['hour'] = merged['date'].dt.hour
merged['minute'] = merged['date'].dt.minute
merged['dayofweek'] = merged['date'].dt.dayofweek
merged['month'] = merged['date'].dt.month
merged['day'] = merged['date'].dt.day

# drop remaining NaNs in 'normal'
merged['normal'] = merged['normal'].fillna(0)

# estimating insulin-on-board
def insulin_activity(delta_min):
    # no effect before onset
    if delta_min < 15:
        return 0.0
    # linear rise to peak
    elif delta_min < 60:
        return (delta_min - 15) / (60 - 15)
    # linear fall to zero from peak to 5h
    elif delta_min < 300:
        return 1 - (delta_min - 60) / (300 - 60)
    else:
        return 0.0

merged = merged.sort_values('date').reset_index(drop=True)
merged['iob'] = 0.0

bolus_events = merged[merged['normal'] > 0][['date', 'normal']].values.tolist()

print(f'COMPUTING IOB (USING {"FAST INACCURATE IOB ESTIMATION" if args.fast_iob else "SLOW ACCURATE IOB ESTIMATION"}):')
def fast_iob_convolution(normal_series):
    # build kernel: 5min steps, total of 5 hours = 60 steps
    minutes = np.arange(0, 301, 5)
    activity = np.zeros_like(minutes, dtype=float)
    for i, m in enumerate(minutes):
        if m < 15:
            activity[i] = 0.0
        elif m < 60:
            activity[i] = (m - 15) / 45
        elif m < 300:
            activity[i] = 1 - (m - 60) / 240
        else:
            activity[i] = 0.0

    kernel = activity

    # # pad to preserve length
    padded = np.pad(normal_series.values, (len(kernel) - 1, 0), mode='constant')
    iob = np.convolve(padded, kernel, mode='valid')
    return iob

if args.fast_iob:
    merged = merged.sort_values('date').reset_index(drop=True)
    merged['iob'] = fast_iob_convolution(merged['normal'])
else:
    # TODO: doing this on merge is *wrong* -- we should be computing on continuous segments, otherwise our 5 min time jump assumption breaks
    # TODO: also important for the later fast conv code
    if args.sanity_check:
        data_len = 5000
    else:
        data_len = len(merged)
    for i in tqdm(range(data_len)):
        t = merged.loc[i, 'date']
        total_iob = 0.0
        for dose_time, dose in bolus_events:
            delta_min = (t - dose_time).total_seconds() / 60
            if delta_min > 300:
                continue
            activity = insulin_activity(delta_min)
            total_iob += dose * activity
        merged.at[i, 'iob'] = total_iob

# filter only segments with >= 1 hour (12 points)
merged = merged.sort_values('date').reset_index(drop=True)
merged['time_diff'] = merged['date'] - merged['date'].shift(1)
breaks = merged['time_diff'] != pd.Timedelta('5min')
breaks.iloc[0] = True
merged['segment_id'] = breaks.cumsum()

segment_counts = merged['segment_id'].value_counts()
if args.verbose:
    print(segment_counts)
valid_segments = segment_counts[segment_counts >= 12].index
filtered = merged[merged['segment_id'].isin(valid_segments)]

# write each segment to csv
data_dir = f'../data/cleaned/{"fast_iob" if args.fast_iob else "slow_iob"}/{args.subject_id}'
if args.to_csv:
    os.makedirs(data_dir, exist_ok=True)
    for i, seg_id in enumerate(valid_segments):
        segment = filtered[filtered['segment_id'] == seg_id].reset_index(drop=True)
        segment = segment.sort_values(by='date')
        if args.verbose and i==0:
            print(f'PREPROCESS: \n{segment.head(5).to_markdown()}')
        segment = segment[["mg/dl", "iob", "hour", "minute", "dayofweek"]]

        # add cyclic encodings
        segment["hour_sin"] = np.sin(2 * np.pi * segment["hour"] / 24)
        segment["hour_cos"] = np.cos(2 * np.pi * segment["hour"] / 24)
        segment["minute_sin"] = np.sin(2 * np.pi * segment["minute"] / 60)
        segment["minute_cos"] = np.cos(2 * np.pi * segment["minute"] / 60)

        # one-hot for weekday
        segment = pd.get_dummies(segment, columns=["dayofweek"], prefix="dow")
        for dow in range(7):
            col = f"dow_{dow}"
            if col not in segment.columns:
                segment[col] = 0.0
        segment = segment[
            ['mg/dl', 'iob', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos'] + [f'dow_{dow}' for dow in range(7)]
        ]

        segment = segment.astype(np.float32)

        if args.verbose and i==0:
            print(f'POSTPROCESS: \n{segment.head(5).to_markdown()}')

        segment.to_csv(f'{data_dir}/{i}.csv', index=False)

if args.verbose:
    print_info(cgm_df, bolus_df, merged)