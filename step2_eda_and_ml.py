import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score

# =====================================================================
# STEP 2 (FIXED): REPAIR INT64 DELTA + RUN TRUE EDA & AI BACKTEST
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "6e_1yr_footprint_bars.csv")

PIP_SIZE = 0.00010
PROFIT_TARGET_PIPS = 8.0   # Calibrated to Table 1 volatility (2:1 R:R)
STOP_LOSS_PIPS = 4.0       # 4.0 pips stop loss
COST_PIPS = 0.5            # 0.5 pips commission + slippage

# 1. Load the CSV and instantly repair the uint32 overflow in 'delta'
df = pd.read_csv(INPUT_FILE)
df['bar_time'] = pd.to_datetime(df['bar_time'])

print("Repairing uint32 Delta overflow using signed 64-bit integers (int64)...")
df['ask_vol']   = df['ask_vol'].astype('int64')
df['bid_vol']   = df['bid_vol'].astype('int64')
df['total_vol'] = df['ask_vol'] + df['bid_vol']
df['delta']     = df['ask_vol'] - df['bid_vol']
df['delta_pct'] = (df['delta'] / df['total_vol'].replace(0, np.nan)).fillna(0).round(3)

# Recalculate Rolling 20-bar Delta Z-Score & True Bullish/Bearish Absorption
df['delta_zscore_20'] = (
    (df['delta'] - df['delta'].rolling(20).mean()) /
    (df['delta'].rolling(20).std() + 1e-5)
).round(2)

# Bullish Absorption: Sellers aggressive (delta < 0), POC in lower half (<= 0.50), close in upper half (>= 0.55)
df['bullish_absorption'] = (
    (df['delta'] < 0) &
    (df['poc_position'] <= 0.50) &
    (df['close_position'] >= 0.55)
).astype(int)

df['bearish_absorption'] = (
    (df['delta'] > 0) &
    (df['poc_position'] >= 0.50) &
    (df['close_position'] <= 0.45)
).astype(int)

# Save the repaired dataset back to CSV so it is permanently clean
df.to_csv(INPUT_FILE, index=False)
print(f"Repaired! Found {df['bullish_absorption'].sum():,} true Bullish Absorption bars across 66,586 candles.")

# =====================================================================
# 2. EVALUATE TRIPLE-BARRIER OUTCOMES (+8 PIPS TARGET / -4 PIPS STOP)
# =====================================================================
target_dist = PROFIT_TARGET_PIPS * PIP_SIZE
stop_dist   = STOP_LOSS_PIPS * PIP_SIZE
lookahead   = 8  # Next 8 bars (40 minutes)

outcomes, net_pips_list = [], []
highs  = df['high'].values
lows   = df['low'].values
closes = df['close'].values

for i in range(len(df)):
    if i + lookahead >= len(df):
        outcomes.append(np.nan)
        net_pips_list.append(np.nan)
        continue

    entry = closes[i]
    win = 0
    net_pips = -STOP_LOSS_PIPS - COST_PIPS

    for j in range(i + 1, i + 1 + lookahead):
        if lows[j] <= entry - stop_dist:
            win = 0
            net_pips = -STOP_LOSS_PIPS - COST_PIPS
            break
        elif highs[j] >= entry + target_dist:
            win = 1
            net_pips = PROFIT_TARGET_PIPS - COST_PIPS
            break
    else:
        raw_pips = (closes[i + lookahead] - entry) / PIP_SIZE
        net_pips = round(raw_pips - COST_PIPS, 2)
        win = 1 if net_pips > 0 else 0

    outcomes.append(win)
    net_pips_list.append(net_pips)

df['win_flag'] = outcomes
df['net_pips'] = net_pips_list
df = df.dropna().reset_index(drop=True)

# Focus analysis on active London & NY_Overlap sessions (where volume is 570 - 1,139 lots)
active_df = df[df['session'].isin(['London', 'NY_Overlap'])].copy().reset_index(drop=True)

# =====================================================================
# 3. PRINT REPAIRED HYPOTHESIS TEST (LONDON & NY OVERLAP SESSIONS)
# =====================================================================
print("\n" + "="*70)
print(" 2. REPAIRED HYPOTHESIS TEST (ACTIVE LONDON & NY SESSIONS ONLY)")
print("="*70)

s_base  = active_df
s_imb   = active_df[active_df['stacked_buy_imbalance'] == 1]
s_abs   = active_df[active_df['bullish_absorption'] == 1]
s_aplus = active_df[(active_df['bullish_absorption'] == 1) & (active_df['liquidity_sweep_low'] == 1)]
s_zabs  = active_df[(active_df['bullish_absorption'] == 1) & (active_df['delta_zscore_20'] <= -1.0) & (active_df['liquidity_sweep_low'] == 1)]

def summarize_setup(name, sub):
    if len(sub) == 0:
        return {'Setup Type': name, 'Trades': 0, 'Win Rate (%)': 0.0, 'Avg Net Pips': 0.0, 'Total Net Pips': 0.0}
    return {
        'Setup Type': name,
        'Trades': len(sub),
        'Win Rate (%)': round(sub['win_flag'].mean() * 100, 2),
        'Avg Net Pips': round(sub['net_pips'].mean(), 2),
        'Total Net Pips': round(sub['net_pips'].sum(), 1)
    }

report_df = pd.DataFrame([
    summarize_setup("A. Baseline (All London/NY Candles)", s_base),
    summarize_setup("B. Stacked Buy Imbalance Alone", s_imb),
    summarize_setup("C. Bullish Absorption Alone", s_abs),
    summarize_setup("D. Bullish Absorption + 1H Low Sweep", s_aplus),
    summarize_setup("E. Heavy Delta (Z <= -1.0) + Absorption + 1H Sweep", s_zabs)
])
print(report_df.to_string(index=False))

# =====================================================================
# 4. WALK-FORWARD MACHINE LEARNING & EXTRACT EXACT GOCHARTING RULES
# =====================================================================
features = [
    'hour_utc', 'bar_range_pips', 'total_vol', 'vol_ratio_20',
    'delta', 'delta_pct', 'delta_zscore_20',
    'poc_position', 'close_position',
    'bullish_absorption', 'stacked_buy_imbalance', 'liquidity_sweep_low'
]

split_idx = int(len(active_df) * 0.70)
train_df = active_df.iloc[:split_idx].copy()
test_df  = active_df.iloc[split_idx:].copy()

model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=42)
model.fit(train_df[features], train_df['win_flag'])

test_df['ai_prob'] = model.predict_proba(test_df[features])[:, 1]
roc_auc = roc_auc_score(test_df['win_flag'], test_df['ai_prob'])

print("\n" + "="*70)
print(f" 3. REPAIRED ML FEATURE IMPORTANCE (Out-of-Sample ROC-AUC: {roc_auc:.3f})")
print("="*70)
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
for feat, imp in importances.items():
    print(f"  {feat:24s}: {imp*100:5.1f}% importance")

# Compare Out-of-Sample AI-Filtered Trades vs Unfiltered Signals
candidate_test = test_df[(test_df['bullish_absorption'] == 1) | (test_df['liquidity_sweep_low'] == 1)].copy()
top_ai_trades  = candidate_test[candidate_test['ai_prob'] >= np.percentile(candidate_test['ai_prob'], 75)]

print("\n" + "="*70)
print(" 4. OUT-OF-SAMPLE BACKTEST RESULT (TOP 25% AI CONFIDENCE SETUPS)")
print("="*70)
print(f"  Out-of-Sample Trades Taken : {len(top_ai_trades)}")
print(f"  Out-of-Sample Win Rate (%) : {top_ai_trades['win_flag'].mean()*100:.2f}% (Break-even required: 36.7%)")
print(f"  Avg Net Pips per Trade     : {top_ai_trades['net_pips'].mean():+.2f} pips")
print(f"  Total Out-of-Sample Pips   : {top_ai_trades['net_pips'].sum():+.1f} pips (${top_ai_trades['net_pips'].sum()*12.50:+,.2f} on 1 lot)")

# Train a shallow Decision Tree on Absorption + Sweep setups to print human-readable rules for GoCharting
rule_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=42)
rule_tree.fit(train_df[features], train_df['win_flag'])
print("\n" + "="*70)
print(" 5. AI DECISION TREE RULES (FOR GOCHARTING LIVE TRADING)")
print("="*70)
print(export_text(rule_tree, feature_names=features))