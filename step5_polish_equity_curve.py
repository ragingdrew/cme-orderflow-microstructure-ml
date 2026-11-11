import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# =====================================================================
# STEP 5: ELIMINATE COUNTER-TREND DRAWDOWNS & BUILD MONTHLY TEAR SHEET
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "6e_1yr_footprint_bars.csv")
PIP_SIZE   = 0.00010
COST_PIPS  = 0.4  # CME 6E round-turn fee (~0.4 pips)

df = pd.read_csv(INPUT_FILE)
df['bar_time'] = pd.to_datetime(df['bar_time'])

# 1. Calculate Structural & Microstructure Features
df['ema_50']  = df['close'].ewm(span=50, adjust=False).mean()
df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
df['dist_ema50_pips'] = ((df['close'] - df['ema_50']) / PIP_SIZE).round(2)
df['ema200_slope_pips'] = ((df['ema_200'] - df['ema_200'].shift(12)) / PIP_SIZE).round(2)

df['rolling_2h_low']  = df['low'].rolling(24).min().shift(1)
df['rolling_2h_high'] = df['high'].rolling(24).max().shift(1)
df['sweep_2h_low']    = (df['low'] <= df['rolling_2h_low']).astype(int)
df['sweep_2h_high']   = (df['high'] >= df['rolling_2h_high']).astype(int)

df = df.dropna().reset_index(drop=True)

# Active London & NY Overlap Hours (07:00 to 15:00 UTC)
active_df = df[df['hour_utc'].between(7, 15)].copy().reset_index(drop=True)

# =====================================================================
# 2. SIMULATOR WITH PROFIT-LOCK TRAILING STOP
# =====================================================================
def simulate_strategy(data, signal_mask, direction, target_pips, stop_pips, trigger_pips=None, lock_pips=0.5, max_bars=24):
    indices = np.where(signal_mask)[0]
    highs   = data['high'].values
    lows    = data['low'].values
    closes  = data['close'].values
    times   = data['bar_time'].values
    n       = len(data)

    t_dist    = target_pips * PIP_SIZE
    trig_dist = (trigger_pips * PIP_SIZE) if trigger_pips else None

    results = []
    last_exit_idx = -1

    for i in indices:
        if i <= last_exit_idx or i + max_bars >= n:
            continue

        entry = closes[i]
        current_stop_pips = -stop_pips
        locked = False
        net_pips = 0.0
        exit_idx = i + max_bars

        for j in range(i + 1, i + 1 + max_bars):
            if direction == "LONG":
                stop_price = entry + (current_stop_pips * PIP_SIZE)
                if lows[j] <= stop_price:
                    net_pips = current_stop_pips - COST_PIPS
                    exit_idx = j
                    break
                if highs[j] >= entry + t_dist:
                    net_pips = target_pips - COST_PIPS
                    exit_idx = j
                    break
                if trig_dist and not locked and highs[j] >= entry + trig_dist:
                    current_stop_pips = lock_pips
                    locked = True
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
                if trig_dist and not locked and lows[j] <= entry - trig_dist:
                    current_stop_pips = lock_pips
                    locked = True
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
# 3. COMPARE STEP 4 BASELINE VS REFINED HIGH-CONVICTION FILTERS
# =====================================================================
# 1. Step 4 Baseline (Loose OR condition)
l_step4 = (active_df['dist_ema50_pips'] <= -12.4) & (active_df['close_position'] >= 0.55) & ((active_df['sweep_2h_low'] == 1) | (active_df['bullish_absorption'] == 1))
s_step4 = (active_df['dist_ema50_pips'] >= 12.4)  & (active_df['close_position'] <= 0.45) & ((active_df['sweep_2h_high'] == 1) | (active_df['bearish_absorption'] == 1))

# 2. Refined Filter A: Require Volume Climax (vol_ratio_20 >= 1.2) + Do Not Fight Runaway Macro Crash/Rally
l_climax = l_step4 & (active_df['vol_ratio_20'] >= 1.2) & (active_df['ema200_slope_pips'] > -8.0)
s_climax = s_step4 & (active_df['vol_ratio_20'] >= 1.2) & (active_df['ema200_slope_pips'] < 8.0)

# 3. Refined Filter B: Strict True Footprint Absorption (Negative Delta on Longs / Positive Delta on Shorts + Sweep)
l_strict = (active_df['dist_ema50_pips'] <= -10.0) & (active_df['bullish_absorption'] == 1) & (active_df['sweep_2h_low'] == 1)
s_strict = (active_df['dist_ema50_pips'] >= 10.0)  & (active_df['bearish_absorption'] == 1) & (active_df['sweep_2h_high'] == 1)

# 4. Refined Filter C: Extreme Stretch (>= 15 pips from EMA50) + High Volume Rejection
l_extreme = (active_df['dist_ema50_pips'] <= -15.0) & (active_df['close_position'] >= 0.55) & (active_df['vol_ratio_20'] >= 1.1)
s_extreme = (active_df['dist_ema50_pips'] >= 15.0)  & (active_df['close_position'] <= 0.45) & (active_df['vol_ratio_20'] >= 1.1)

models = [
    ("1. Step 4 Baseline (15p TP / 8p SL, Lock +0.5p @ 6p)", l_step4, s_step4, 15.0, 8.0, 6.0, 0.5),
    ("2. Step 4 + Profit Lock +2.5p @ 7p (Higher Payout)",   l_step4, s_step4, 15.0, 8.0, 7.0, 2.5),
    ("3. Volume Climax + Macro Trend Filter (15p/8p, Lock +2p @ 6p)", l_climax, s_climax, 15.0, 8.0, 6.0, 2.0),
    ("4. Strict Absorption AND 2H Sweep (15p/8p, Lock +2p @ 6p)",     l_strict, s_strict, 15.0, 8.0, 6.0, 2.0),
    ("5. Extreme 15p Rubber-Band + Vol Spike (16p/8p, Lock +2p @ 6p)", l_extreme, s_extreme, 16.0, 8.0, 6.0, 2.0),
]

print("\n" + "="*95)
print(" STEP 5: SMOOTHING THE EQUITY CURVE & BOOSTING PROFIT FACTOR")
print("="*95)

best_df = None
best_pf = -1.0
best_title = ""
rows = []

for name, lm, sm, tp, sl, trig, lock in models:
    dfl = simulate_strategy(active_df, lm, "LONG", tp, sl, trig, lock)
    dfs = simulate_strategy(active_df, sm, "SHORT", tp, sl, trig, lock)
    comb = pd.concat([dfl, dfs], ignore_index=True).sort_values('entry_time').reset_index(drop=True)
    if len(comb) == 0:
        continue

    wins = comb[comb['net_pips'] > 0]
    losses = comb[comb['net_pips'] <= 0]
    wr = (len(wins) / len(comb)) * 100
    g_win = wins['net_pips'].sum()
    g_loss = abs(losses['net_pips'].sum())
    pf = (g_win / g_loss) if g_loss > 0 else np.inf
    tot_pips = comb['net_pips'].sum()

    # Max Drawdown calculation
    eq = comb['net_pips'].cumsum()
    max_dd = (eq - eq.cummax()).min()

    rows.append({
        'Strategy Name': name,
        'Trades': len(comb),
        'Win Rate (%)': round(wr, 2),
        'Profit Factor': round(pf, 2),
        'Max DD (Pips)': round(max_dd, 1),
        'Total Net Pips': round(tot_pips, 1),
        'Net Profit ($ 1 Lot)': f"${tot_pips * 12.50:+,.0f}"
    })

    if pf > best_pf and len(comb) >= 80:
        best_pf = pf
        best_df = comb.copy()
        best_title = name

print(pd.DataFrame(rows).to_string(index=False))

# Print Monthly Tear Sheet for the Highest Profit-Factor Strategy
if best_df is not None:
    best_df['month'] = pd.to_datetime(best_df['entry_time']).dt.to_period('M')
    monthly = best_df.groupby('month').agg(
        trades=('net_pips', 'count'),
        win_rate=('win', lambda x: round(x.mean()*100, 1)),
        net_pips=('net_pips', lambda x: round(x.sum(), 1)),
        profit_usd=('net_pips', lambda x: round(x.sum() * 12.50, 0))
    )
    print("\n" + "="*95)
    print(f" MONTHLY QUANTITATIVE TEAR SHEET FOR: {best_title}")
    print("="*95)
    print(monthly.to_string())

    # Plot Comparison: Step 4 vs Best Step 5 Equity Curve
    best_df['cum_pips'] = best_df['net_pips'].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(best_df['entry_time'], best_df['cum_pips'], color='#2e8b57', linewidth=2.2, label=best_title)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f"Polished CME 6E Order Flow Equity Curve (PF: {best_pf:.2f})")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Net Pips (After Fees)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_img = os.path.join(SCRIPT_DIR, "6e_polished_equity_curve.png")
    plt.savefig(out_img, dpi=150)
    print(f"\nSaved polished chart to '{out_img}'!")