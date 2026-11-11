import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# =====================================================================
# STEP 4: APPLY THE AI DECISION TREE RULES (LONG & SHORT + BREAKEVEN)
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "6e_1yr_footprint_bars.csv")
PIP_SIZE   = 0.00010
COST_PIPS  = 0.4  # Realistic CME 6E round-turn fee (~0.4 pips = $5.00)

df = pd.read_csv(INPUT_FILE)
df['bar_time'] = pd.to_datetime(df['bar_time'])

# 1. Calculate the Exact Features Discovered by the Step 3 Decision Tree
df['ema_50']  = df['close'].ewm(span=50, adjust=False).mean()
df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
df['dist_ema50_pips'] = ((df['close'] - df['ema_50']) / PIP_SIZE).round(2)

# Rolling 2-Hour (24-bar) High & Low Liquidity Sweeps
df['rolling_2h_low']  = df['low'].rolling(24).min().shift(1)
df['rolling_2h_high'] = df['high'].rolling(24).max().shift(1)
df['sweep_2h_low']    = (df['low'] <= df['rolling_2h_low']).astype(int)
df['sweep_2h_high']   = (df['high'] >= df['rolling_2h_high']).astype(int)

# 3-Bar Cumulative Volume Delta
df['cvd_3bar'] = df['delta'].rolling(3).sum()

df = df.dropna().reset_index(drop=True)

# Filter to Active Institutional Hours found by the Tree (07:00 to 15:00 UTC)
active_df = df[df['hour_utc'].between(7, 15)].copy().reset_index(drop=True)

# =====================================================================
# 2. TRADE SIMULATOR (WITH OPTIONAL BREAKEVEN STOP MANAGEMENT)
# =====================================================================
def simulate_strategy(data, signal_mask, direction, target_pips, stop_pips, be_trigger_pips=None, max_bars=24):
    """
    Simulates trades only when signal_mask is True.
    If be_trigger_pips is set (e.g. +6 pips), moves stop loss to +0.5p (breakeven + fee) once reached.
    """
    indices = np.where(signal_mask)[0]
    highs   = data['high'].values
    lows    = data['low'].values
    closes  = data['close'].values
    times   = data['bar_time'].values
    n       = len(data)

    t_dist  = target_pips * PIP_SIZE
    s_dist  = stop_pips * PIP_SIZE
    be_dist = (be_trigger_pips * PIP_SIZE) if be_trigger_pips else None

    results = []
    last_exit_idx = -1  # Prevent overlapping trades at the same time

    for i in indices:
        if i <= last_exit_idx or i + max_bars >= n:
            continue

        entry = closes[i]
        current_stop_pips = -stop_pips
        moved_to_be = False
        net_pips = 0.0
        exit_idx = i + max_bars

        for j in range(i + 1, i + 1 + max_bars):
            if direction == "LONG":
                # Check if stop hit first
                stop_price = entry + (current_stop_pips * PIP_SIZE)
                if lows[j] <= stop_price:
                    net_pips = current_stop_pips - COST_PIPS
                    exit_idx = j
                    break
                # Check if full target hit
                if highs[j] >= entry + t_dist:
                    net_pips = target_pips - COST_PIPS
                    exit_idx = j
                    break
                # Check if breakeven trigger reached
                if be_dist and not moved_to_be and highs[j] >= entry + be_dist:
                    current_stop_pips = 0.5  # Lock in +0.5 pips to cover fee
                    moved_to_be = True
            else:  # SHORT
                stop_price = entry - (current_stop_pips * PIP_SIZE)
                if highs[j] >= stop_price:
                    net_pips = current_stop_pips - COST_PIPS
                    exit_idx = j
                    break
                if lows[j] <= entry - t_dist:
                    net_pips = target_pips - COST_PIPS
                    exit_idx = j
                    break
                if be_dist and not moved_to_be and lows[j] <= entry - be_dist:
                    current_stop_pips = 0.5
                    moved_to_be = True
        else:
            diff = (closes[i + max_bars] - entry) / PIP_SIZE
            raw = diff if direction == "LONG" else -diff
            net_pips = round(raw - COST_PIPS, 2)

        last_exit_idx = exit_idx
        results.append({
            'entry_time': times[i],
            'direction': direction,
            'net_pips': round(net_pips, 2),
            'win': 1 if net_pips > 0 else 0
        })

    return pd.DataFrame(results)

# =====================================================================
# 3. TEST THE EXACT AI DECISION TREE DISCOVERIES (NO CHOP ZONE!)
# =====================================================================
# Strategy A: AI Extreme Rubber-Band Reversal (Outside +/- 12.4 pips from EMA50 + Sweep/Rejection)
long_rev = (
    (active_df['dist_ema50_pips'] <= -12.4) &
    (active_df['close_position'] >= 0.55) &
    ((active_df['sweep_2h_low'] == 1) | (active_df['bullish_absorption'] == 1))
)
short_rev = (
    (active_df['dist_ema50_pips'] >= 12.4) &
    (active_df['close_position'] <= 0.45) &
    ((active_df['sweep_2h_high'] == 1) | (active_df['bearish_absorption'] == 1))
)

# Strategy B: AI Breakout Continuation (> 6.8 pips from EMA50 + Stacked Imbalance / Strong Delta)
long_brk = (
    (active_df['dist_ema50_pips'] > 6.8) &
    (active_df['stacked_buy_imbalance'] == 1) &
    (active_df['close_position'] >= 0.60)
)
short_brk = (
    (active_df['dist_ema50_pips'] < -6.8) &
    (active_df['delta_zscore_20'] <= -1.5) &
    (active_df['close_position'] <= 0.40)
)

configs_to_test = [
    ("1. AI Rubber-Band Reversal (10p TP / 6p SL, Static)", long_rev, short_rev, 10.0, 6.0, None),
    ("2. AI Rubber-Band Reversal (12p TP / 6p SL + Breakeven @ 5p)", long_rev, short_rev, 12.0, 6.0, 5.0),
    ("3. AI Rubber-Band Reversal (15p TP / 8p SL + Breakeven @ 6p)", long_rev, short_rev, 15.0, 8.0, 6.0),
    ("4. AI Breakout Imbalance (8p TP / 6p SL + Breakeven @ 4p)", long_brk, short_brk, 8.0, 6.0, 4.0),
    ("5. AI Breakout Imbalance (12p TP / 6p SL + Breakeven @ 5p)", long_brk, short_brk, 12.0, 6.0, 5.0),
]

print("\n" + "="*90)
print(" FINAL STRATEGY RESULTS (APPLYING AI TREE FILTER: BLOCKING THE -12.4 TO +6.8 CHOP ZONE)")
print("="*90)

best_trades = None
best_pips = -99999

summary_rows = []
for name, l_mask, s_mask, tp, sl, be in configs_to_test:
    df_l = simulate_strategy(active_df, l_mask, "LONG", tp, sl, be)
    df_s = simulate_strategy(active_df, s_mask, "SHORT", tp, sl, be)
    combined = pd.concat([df_l, df_s], ignore_index=True).sort_values('entry_time').reset_index(drop=True)

    if len(combined) == 0:
        continue

    wins = combined[combined['net_pips'] > 0]
    losses = combined[combined['net_pips'] <= 0]
    win_rate = (len(wins) / len(combined)) * 100
    gross_win = wins['net_pips'].sum()
    gross_loss = abs(losses['net_pips'].sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else np.inf
    tot_pips = combined['net_pips'].sum()

    summary_rows.append({
        'Strategy Configuration': name,
        'Trades': len(combined),
        'Win Rate (%)': round(win_rate, 2),
        'Profit Factor': round(pf, 2),
        'Avg Net Pips': round(combined['net_pips'].mean(), 2),
        'Total Net Pips': round(tot_pips, 1),
        'Net Profit ($ 1 Lot)': f"${tot_pips * 12.50:+,.0f}"
    })

    if tot_pips > best_pips:
        best_pips = tot_pips
        best_trades = combined.copy()
        best_name = name

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

if best_trades is not None and len(best_trades) > 0:
    best_trades['cum_pips'] = best_trades['net_pips'].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(best_trades['entry_time'], best_trades['cum_pips'], color='#1f77b4', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f"Best Strategy Equity Curve: {best_name}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Net Pips (After Fees)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    chart_file = os.path.join(SCRIPT_DIR, "6e_winning_strategy_equity.png")
    plt.savefig(chart_file, dpi=150)
    print(f"\nSaved equity curve chart to '{chart_file}'!")