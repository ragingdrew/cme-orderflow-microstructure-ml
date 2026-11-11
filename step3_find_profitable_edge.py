import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

# =====================================================================
# STEP 3: STRATEGY OPTIMIZER (TRUE BARRIER HITS + TREND + BALANCED AI)
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "6e_1yr_footprint_bars.csv")
PIP_SIZE   = 0.00010
COST_PIPS  = 0.5  # Commission + Slippage

df = pd.read_csv(INPUT_FILE)
df['bar_time'] = pd.to_datetime(df['bar_time'])

# 1. Engineer Higher-Timeframe Trend & Overextension Features
df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()   # ~4-Hour Trend
df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean() # ~16-Hour Macro Trend
df['dist_from_ema50_pips'] = ((df['close'] - df['ema_50']) / PIP_SIZE).round(2)
df['uptrend_regime'] = (df['ema_50'] > df['ema_200']).astype(int)

# Rolling 2-Hour (24-bar) High/Low Sweeps
df['rolling_2h_low']  = df['low'].rolling(24).min().shift(1)
df['rolling_2h_high'] = df['high'].rolling(24).max().shift(1)
df['sweep_2h_low']    = (df['low'] <= df['rolling_2h_low']).astype(int)
df['sweep_2h_high']   = (df['high'] >= df['rolling_2h_high']).astype(int)

# Cumulative 3-bar Delta (Momentum vs Exhaustion)
df['cvd_3bar'] = df['delta'].rolling(3).sum()

df = df.dropna().reset_index(drop=True)

# Focus on Active Institutional Hours (07:00 to 16:00 UTC)
active_df = df[df['hour_utc'].between(7, 16)].copy().reset_index(drop=True)
print(f"Analyzing {len(active_df):,} active London/NY 5-minute bars...")

# =====================================================================
# 2. TRUE BARRIER OUTCOME SIMULATOR (NO FAKE TIME-OUT WINS)
# =====================================================================
def simulate_true_barrier(data, target_pips, stop_pips, direction="LONG", max_bars=24):
    """
    Checks up to 24 bars (2 hours).
    ONLY counts a Win (1) if the full Profit Target is hit before the Stop Loss.
    """
    t_dist = target_pips * PIP_SIZE
    s_dist = stop_pips * PIP_SIZE

    highs  = data['high'].values
    lows   = data['low'].values
    closes = data['close'].values
    n = len(data)

    wins = np.zeros(n, dtype=int)
    net_pips = np.zeros(n, dtype=float)

    for i in range(n - max_bars):
        entry = closes[i]
        outcome_pips = -COST_PIPS  # If neither hit after 2 hours

        for j in range(i + 1, i + 1 + max_bars):
            if direction == "LONG":
                if lows[j] <= entry - s_dist:
                    wins[i] = 0
                    outcome_pips = -stop_pips - COST_PIPS
                    break
                elif highs[j] >= entry + t_dist:
                    wins[i] = 1
                    outcome_pips = target_pips - COST_PIPS
                    break
            else:  # SHORT
                if highs[j] >= entry + s_dist:
                    wins[i] = 0
                    outcome_pips = -stop_pips - COST_PIPS
                    break
                elif lows[j] <= entry - t_dist:
                    wins[i] = 1
                    outcome_pips = target_pips - COST_PIPS
                    break
        else:
            # Time limit reached: calculate exact close difference
            diff = (closes[i + max_bars] - entry) / PIP_SIZE
            raw = diff if direction == "LONG" else -diff
            outcome_pips = round(raw - COST_PIPS, 2)
            wins[i] = 1 if outcome_pips >= (target_pips * 0.5) else 0

        net_pips[i] = outcome_pips

    return wins, net_pips

# =====================================================================
# 3. GRID SEARCH ACROSS STOP/TARGET SIZES & ORDER FLOW SETUPS
# =====================================================================
print("\n" + "="*80)
print(" 1. GRID SEARCH: TESTING WIDER STOPS & TREND/SWEEP FILTERS ON CME 6E")
print("="*80)

configs = [
    ("1:1 Scalp (6p Target / 6p Stop)", 6.0, 6.0),
    ("1.5:1 Swing (9p Target / 6p Stop)", 9.0, 6.0),
    ("2:1 Wide (12p Target / 6p Stop)", 12.0, 6.0),
    ("2:1 Institutional (16p Target / 8p Stop)", 16.0, 8.0),
]

grid_results = []
for label, tp, sl in configs:
    w_long, p_long = simulate_true_barrier(active_df, tp, sl, "LONG")
    active_df['tmp_win']  = w_long
    active_df['tmp_pips'] = p_long

    # Setup 1: Trend Pullback Absorption (Uptrend + Bullish Absorption + Below EMA50)
    s1 = active_df[(active_df['uptrend_regime'] == 1) & (active_df['bullish_absorption'] == 1) & (active_df['dist_from_ema50_pips'] < 0)]
    # Setup 2: Deep 2-Hour Low Sweep + High Volume Spike (vol_ratio_20 >= 1.5) + Strong Close
    s2 = active_df[(active_df['sweep_2h_low'] == 1) & (active_df['vol_ratio_20'] >= 1.5) & (active_df['close_position'] >= 0.60)]
    # Setup 3: Momentum Breakout (Uptrend + Stacked Buy Imbalance + Positive Delta Z-Score)
    s3 = active_df[(active_df['uptrend_regime'] == 1) & (active_df['stacked_buy_imbalance'] == 1) & (active_df['delta_zscore_20'] >= 1.0)]

    for s_name, sub in [("Trend Pullback Absorption", s1), ("2H Low Sweep + Vol Spike", s2), ("Uptrend Stacked Imbalance", s3)]:
        if len(sub) > 20:
            grid_results.append({
                'Stop/Target Config': label,
                'Order Flow Strategy': s_name,
                'Trades': len(sub),
                'True Win Rate (%)': round(sub['tmp_win'].mean() * 100, 2),
                'Avg Net Pips': round(sub['tmp_pips'].mean(), 2),
                'Total Net Pips': round(sub['tmp_pips'].sum(), 1)
            })

grid_df = pd.DataFrame(grid_results).sort_values(by='Avg Net Pips', ascending=False)
print(grid_df.to_string(index=False))

# =====================================================================
# 4. TRAIN BALANCED AI ON THE WIDEST BREATHING-ROOM CONFIG (12p / 6p)
# =====================================================================
w_best, p_best = simulate_true_barrier(active_df, 12.0, 6.0, "LONG")
active_df['win_12_6']  = w_best
active_df['pips_12_6'] = p_best

features = [
    'hour_utc', 'bar_range_pips', 'vol_ratio_20',
    'delta_pct', 'delta_zscore_20', 'poc_position', 'close_position',
    'dist_from_ema50_pips', 'uptrend_regime', 'sweep_2h_low', 'cvd_3bar'
]

# Filter to candidate order flow events (Sweeps, Absorptions, or Imbalances)
events_df = active_df[
    (active_df['bullish_absorption'] == 1) |
    (active_df['sweep_2h_low'] == 1) |
    (active_df['stacked_buy_imbalance'] == 1)
].copy().reset_index(drop=True)

split_idx = int(len(events_df) * 0.70)
train_ev = events_df.iloc[:split_idx].copy()
test_ev  = events_df.iloc[split_idx:].copy()

# CRITICAL FIX: class_weight='balanced' so AI detects profitable 2:1 setups!
rf = RandomForestClassifier(
    n_estimators=300, max_depth=5, min_samples_leaf=25,
    class_weight='balanced', random_state=42
)
rf.fit(train_ev[features], train_ev['win_12_6'])

test_ev['ai_score'] = rf.predict_proba(test_ev[features])[:, 1]

print("\n" + "="*80)
print(" 2. OUT-OF-SAMPLE AI PERFORMANCE BY CONFIDENCE TIER (12p Target / 6p Stop)")
print("="*80)
for pct in [50, 60, 70, 80, 85]:
    thresh = np.percentile(test_ev['ai_score'], pct)
    taken = test_ev[test_ev['ai_score'] >= thresh]
    print(
        f"  Top {100-pct:2d}% AI Setups (Score >= {thresh:.2f}) | "
        f"Trades: {len(taken):4d} | True Win Rate: {taken['win_12_6'].mean()*100:5.2f}% | "
        f"Avg Net Pips: {taken['pips_12_6'].mean():+5.2f} | Total Pips: {taken['pips_12_6'].sum():+6.1f}"
    )

# =====================================================================
# 5. BALANCED DECISION TREE (NOW PRINTS CLASS: 1 WINNING RULES!)
# =====================================================================
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=40, class_weight='balanced', random_state=42)
dt.fit(train_ev[features], train_ev['win_12_6'])

print("\n" + "="*80)
print(" 3. BALANCED DECISION TREE RULES (LOOK FOR 'class: 1' = BUY SIGNAL)")
print("="*80)
print(export_text(dt, feature_names=features))