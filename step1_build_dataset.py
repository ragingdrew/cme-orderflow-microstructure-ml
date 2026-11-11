import pandas as pd
import numpy as np
import glob
import os
import zipfile

# =====================================================================
# STEP 1: AUTO-LOAD FROM THE SAME FOLDER AS THIS SCRIPT (OneDrive\Documents)
# =====================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BATCH_NAME  = "GLBX-20261110-8CVQRGG5PT"
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "6e_1yr_footprint_bars.csv")

TICK_SIZE = 0.00005  # 0.5 pips minimum tick on CME 6E
PIP_SIZE  = 0.00010  # 1.0 standard forex pip

def locate_and_prepare_files(batch_name):
    """Finds GLBX-20261110-8CVQRGG5PT right next to step1_build_dataset.py."""
    print(f"Looking for '{batch_name}' inside: {SCRIPT_DIR}")

    folder_path = os.path.join(SCRIPT_DIR, batch_name)
    zip_path    = os.path.join(SCRIPT_DIR, f"{batch_name}.zip")

    # Also check any matching file starting with the batch name
    matches = glob.glob(os.path.join(SCRIPT_DIR, f"{batch_name}*"))

    if os.path.isdir(folder_path):
        target_dir = folder_path
        print(f"Found extracted folder: {target_dir}")
    elif os.path.isfile(zip_path):
        target_dir = folder_path
        print(f"Found ZIP archive '{zip_path}'. Extracting all 316 files now...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
    elif matches:
        first_match = matches[0]
        if zipfile.is_zipfile(first_match):
            target_dir = folder_path
            print(f"Found ZIP archive '{first_match}'. Extracting all 316 files now...")
            with zipfile.ZipFile(first_match, 'r') as zf:
                zf.extractall(target_dir)
        else:
            target_dir = first_match
    else:
        raise FileNotFoundError(f"Could not find {batch_name} in {SCRIPT_DIR}")

    # Grab all daily .zst trade files inside the extracted folder
    zst_files = sorted(glob.glob(os.path.join(target_dir, "**", "*trades*.zst"), recursive=True))
    if not zst_files:
        zst_files = sorted(glob.glob(os.path.join(target_dir, "**", "*.zst"), recursive=True))

    return zst_files

def load_single_daily_file(filepath):
    """Reads a single daily Databento .zst file and isolates the active front-month 6E contract."""
    try:
        # Try reading as CSV first; if it's binary DBN format, use databento library
        try:
            df = pd.read_csv(filepath)
        except Exception:
            import databento as db
            df = db.DBNStore.from_file(filepath).to_df().reset_index()
    except Exception as e:
        print(f"  Skipping {os.path.basename(filepath)}: {e}")
        return pd.DataFrame()

    if df.empty or 'price' not in df.columns:
        return pd.DataFrame()

    # Convert fixed-point integer prices (1e9) to decimal if needed
    if df['price'].median() > 1000:
        df['price'] = df['price'] / 1e9

    # 1. Remove Calendar Spreads (outright 6E trades around ~1.05 to 1.15; spreads trade near 0.00)
    df = df[df['price'] > 0.50].copy()
    if 'symbol' in df.columns:
        df = df[~df['symbol'].astype(str).str.contains('-', na=False)]

    # 2. Keep only Aggressive Buy ('A') and Aggressive Sell ('B') executions
    df = df[df['side'].isin(['A', 'B'])].copy()
    if df.empty:
        return pd.DataFrame()

    # 3. Lock onto the single highest-volume contract of that day (Front Month)
    id_col = 'symbol' if 'symbol' in df.columns else ('instrument_id' if 'instrument_id' in df.columns else None)
    if id_col and df[id_col].nunique() > 1:
        front_month = df.groupby(id_col)['size'].sum().idxmax()
        df = df[df[id_col] == front_month]

    return df[['ts_event', 'price', 'size', 'side']]

def process_day_to_5min(df):
    """Aggregates one day of ticks into 5-minute Footprint bars."""
    if pd.api.types.is_numeric_dtype(df['ts_event']):
        df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True)
    else:
        df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)

    df = df.set_index('ts_event').sort_index()
    rows = []

    for bar_time, group in df.groupby(pd.Grouper(freq="5min")):
        if len(group) < 15:
            continue

        open_p  = group['price'].iloc[0]
        high_p  = group['price'].max()
        low_p   = group['price'].min()
        close_p = group['price'].iloc[-1]
        bar_range = max(high_p - low_p, TICK_SIZE)

        fp = group.pivot_table(index='price', columns='side', values='size', aggfunc='sum', fill_value=0)
        if 'A' not in fp.columns: fp['A'] = 0
        if 'B' not in fp.columns: fp['B'] = 0
        fp = fp.sort_index()
        fp['total_at_price'] = fp['A'] + fp['B']

        ask_vol = fp['A'].sum()
        bid_vol = fp['B'].sum()
        total_vol = ask_vol + bid_vol
        delta = ask_vol - bid_vol
        delta_pct = delta / total_vol if total_vol > 0 else 0

        poc_price = fp['total_at_price'].idxmax()
        poc_position = (poc_price - low_p) / bar_range
        close_position = (close_p - low_p) / bar_range

        bullish_absorption = int((delta < 0) and (poc_position <= 0.45) and (close_position >= 0.55))
        bearish_absorption = int((delta > 0) and (poc_position >= 0.55) and (close_position <= 0.45))

        imbalance_buy_flags = (fp['A'] >= (fp['B'] * 3)) & (fp['A'] >= 15)
        stacked_buy_imbalance = int(imbalance_buy_flags.rolling(3).sum().max() >= 3)

        rows.append({
            'bar_time': bar_time,
            'open': round(open_p, 5), 'high': round(high_p, 5),
            'low': round(low_p, 5), 'close': round(close_p, 5),
            'bar_range_pips': round(bar_range / PIP_SIZE, 2),
            'ask_vol': ask_vol, 'bid_vol': bid_vol, 'total_vol': total_vol,
            'delta': delta, 'delta_pct': round(delta_pct, 3),
            'poc_position': round(poc_position, 2),
            'close_position': round(close_position, 2),
            'bullish_absorption': bullish_absorption,
            'bearish_absorption': bearish_absorption,
            'stacked_buy_imbalance': stacked_buy_imbalance
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    daily_files = locate_and_prepare_files(BATCH_NAME)
    print(f"Found {len(daily_files)} daily trade files! Building 5-min Footprint dataset...")

    all_bars = []
    for idx, fpath in enumerate(daily_files, 1):
        if idx % 25 == 0 or idx == 1 or idx == len(daily_files):
            print(f"  -> [{idx}/{len(daily_files)}] Reading {os.path.basename(fpath)}...")
        day_df = load_single_daily_file(fpath)
        if not day_df.empty:
            bars_df = process_day_to_5min(day_df)
            if not bars_df.empty:
                all_bars.append(bars_df)

    full_df = pd.concat(all_bars, ignore_index=True)
    full_df = full_df.sort_values('bar_time').reset_index(drop=True)

    # Add Data Science Session & Normalized Rolling Features
    full_df['hour_utc'] = pd.to_datetime(full_df['bar_time']).dt.hour
    full_df['session'] = np.select(
        [
            full_df['hour_utc'].between(7, 11),
            full_df['hour_utc'].between(12, 16),
            full_df['hour_utc'].between(17, 20)
        ],
        ['London', 'NY_Overlap', 'NY_Afternoon'],
        default='Asian'
    )

    full_df['vol_ratio_20'] = (full_df['total_vol'] / full_df['total_vol'].rolling(20).mean()).round(2)
    full_df['delta_zscore_20'] = (
        (full_df['delta'] - full_df['delta'].rolling(20).mean()) /
        (full_df['delta'].rolling(20).std() + 1e-5)
    ).round(2)

    full_df['rolling_1h_low'] = full_df['low'].rolling(12).min().shift(1)
    full_df['liquidity_sweep_low'] = (full_df['low'] <= full_df['rolling_1h_low']).astype(int)

    full_df = full_df.dropna().reset_index(drop=True)
    full_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUCCESS! Saved '{OUTPUT_FILE}' with {len(full_df):,} clean 5-minute Footprint bars!")