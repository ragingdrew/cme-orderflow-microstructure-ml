import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# =====================================================================
# STEP 6: PROP-DESK DAILY CIRCUIT BREAKER & NEWS SPIKE FILTER
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "6e_1yr_footprint_bars.csv")
PIP_SIZE   = 0.00010
COST_PIPS  = 0.4

df = pd.read_csv(INPUT_FILE)
df['bar_time'] = pd.to_datetime(df['bar_time'])

# 1. Core Features
df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
df['dist_ema50_pips'] = ((df['close'] - df['ema_50']) / PIP_SIZE).round(2)

df['rolling_2h_low']  = df['low'].rolling(24).min().shift(1)
df['rolling_2h_high'] = df['high'].rolling(24).max().shift(1)
df['sweep_2h_low']    = (df['low'] <= df['rolling_2h_low']).astype(int)
df['sweep_2h_high']   = (df['high'] >= df['rolling_2h_high']).astype(int)

df = df.dropna().reset_index(drop=True)
active_df = df[df['hour_utc'].between(7, 15)].copy().reset_index(drop=True)

# =====================================================================
# 2. SIMULATOR WITH DAILY PROP-DESK CIRCUIT BREAKER
# =====================================================================
def simulate_with_daily_cap(
    data, long_mask, short_mask,
    target_pips=15.0, stop_pips=8.0, trig_pips=7.0, lock_pips=2.5,
    max_trades_per_day=99, stop_day_on_full_loss=False, max_bars=24
):
    highs  = data['high'].values
    lows   = data['low'].values
    closes = data['close'].values
    times  = pd.to_datetime(data['bar_time']).values
    dates  = pd.to_datetime(data['bar_time']).dt.date.values
    n      = len(data)

    t_dist    = target_pips * PIP_SIZE
    trig_dist = trig_pips * PIP_SIZE

    results = []
    last_exit_idx = -1
    daily_trade_count = {}
    daily_stopped_out = set()

    for i in range(n - max_bars):
        if i <= last_exit_idx:
            continue

        is_long  = long_mask.iloc[i]
        is_short = short_mask.iloc[i]
        if not (is_long or is_short):
            continue

        cur_date = dates[i]
        if stop_day_on_full_loss and cur_date in daily_stopped_out:
            continue
        if daily_trade_count.get(cur_date, 0) >= max_trades_per_day:
            continue

        direction = "LONG" if is_long else "SHORT"
        entry = closes[i]
        current_stop_pips = -stop_pips
        locked = False
        net_pips = 0.0
        hit_full_sl = False
        exit_idx = i + max_bars

        for j in range(i + 1, i + 1 + max_bars):
            if direction == "LONG":
                stop_price = entry + (current_stop_pips * PIP_SIZE)
                if lows[j] <= stop_price:
                    net_pips = current_stop_pips - COST_PIPS
                    hit_full_sl = not locked
                    exit_idx = j
                    break
                if highs[j] >= entry + t_dist:
                    net_pips = target_pips - COST_PIPS
                    exit_idx = j
                    break
                if not locked and highs[j] >= entry + trig_dist:
                    current_stop_pips = lock_pips
                    locked = True
            else:
                stop_price = entry - (current_stop_pips * PIP_SIZE)
                if highs[j] >= stop_price:
                    net_pips = current_stop_pips - COST_PIPS
                    hit_full_sl = not locked
                    exit_idx = j
                    break
                if lows[j] <= entry - t_dist:
                    net_pips = target_pips - COST_PIPS
                    exit_idx = j
                    break
                if not locked and lows[j] <= entry - trig_dist:
                    current_stop_pips = lock_pips
                    locked = True
        else:
            diff = (closes[i + max_bars] - entry) / PIP_SIZE
            raw = diff if direction == "LONG" else -diff
            net_pips = round(raw - COST_PIPS, 2)

        last_exit_idx = exit_idx
        daily_trade_count[cur_date] = daily_trade_count.get(cur_date, 0) + 1
        if hit_full_sl:
            daily_stopped_out.add(cur_date)

        results.append({
            'entry_time': times[i],
            'direction': direction,
            'net_pips': round(net_pips, 2),
            'win': 1 if net_pips > 0 else 0
        })

    return pd.DataFrame(results)

# =====================================================================
# 3. COMPARE UNLIMITED TRADES VS PROP-DESK DAILY RULES
# =====================================================================
# Base Signal from Step 5 Winner
l_base = (active_df['dist_ema50_pips'] <= -12.4) & (active_df['close_position'] >= 0.55) & ((active_df['sweep_2h_low'] == 1) | (active_df['bullish_absorption'] == 1))
s_base = (active_df['dist_ema50_pips'] >= 12.4)  & (active_df['close_position'] <= 0.45) & ((active_df['sweep_2h_high'] == 1) | (active_df['bearish_absorption'] == 1))

# News Spike Filter: Avoid giant > 7.5 pip news shock bars
no_news_spike = active_df['bar_range_pips'] <= 7.5
l_clean = l_base & no_news_spike
s_clean = s_base & no_news_spike

tests = [
    ("1. Step 5 Winner (Unlimited Daily Trades)", l_base, s_base, 99, False),
    ("2. Stop for Day After 1 Full Stop-Loss (Circuit Breaker)", l_base, s_base, 99, True),
    ("3. Max 2 Trades/Day + Stop After 1 Full Loss", l_base, s_base, 2, True),
    ("4. Max 1 Trade/Day (First A+ Setup of Day Only)", l_base, s_base, 1, True),
    ("5. No News Spikes (<=7.5p) + Stop After 1 Loss", l_clean, s_clean, 2, True),
]

print("\n" + "="*95)
print(" STEP 6: PROP-DESK DAILY CIRCUIT BREAKER COMPARISON")
print("="*95)

best_df = None
best_pips = -9999
best_name = ""
summary = []

for name, lm, sm, max_td, stop_on_sl in tests:
    res = simulate_with_daily_cap(
        active_df, lm, sm,
        target_pips=15.0, stop_pips=8.0, trig_pips=7.0, lock_pips=2.5,
        max_trades_per_day=max_td, stop_day_on_full_loss=stop_on_sl
    )
    if len(res) == 0:
        continue

    wins = res[res['net_pips'] > 0]
    losses = res[res['net_pips'] <= 0]
    wr = (len(wins) / len(res)) * 100
    pf = (wins['net_pips'].sum() / abs(losses['net_pips'].sum())) if len(losses) > 0 else np.inf
    tot_pips = res['net_pips'].sum()
    eq = res['net_pips'].cumsum()
    max_dd = (eq - eq.cummax()).min()

    summary.append({
        'Risk Management Rule': name,
        'Trades': len(res),
        'Win Rate (%)': round(wr, 2),
        'Profit Factor': round(pf, 2),
        'Max DD (Pips)': round(max_dd, 1),
        'Total Net Pips': round(tot_pips, 1),
        'Net Profit ($ 1 Lot)': f"${tot_pips * 12.50:+,.0f}"
    })

    if tot_pips > best_pips:
        best_pips = tot_pips
        best_df = res.copy()
        best_name = name

print(pd.DataFrame(summary).to_string(index=False))

if best_df is not None:
    best_df['month'] = pd.to_datetime(best_df['entry_time']).dt.to_period('M')
    monthly = best_df.groupby('month').agg(
        trades=('net_pips', 'count'),
        win_rate=('win', lambda x: round(x.mean()*100, 1)),
        net_pips=('net_pips', lambda x: round(x.sum(), 1)),
        profit_usd=('net_pips', lambda x: round(x.sum() * 12.50, 0))
    )
    print("\n" + "="*95)
    print(f" UPDATED MONTHLY TEAR SHEET FOR: {best_name}")
    print("="*95)
    print(monthly.to_string())

    best_df['cum_pips'] = best_df['net_pips'].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(best_df['entry_time'], best_df['cum_pips'], color='#1b7837', linewidth=2.2, label=best_name)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f"CME 6E Equity Curve with Prop-Desk Circuit Breaker ({best_name})")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Net Pips (After Fees)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_img = os.path.join(SCRIPT_DIR, "6e_final_prop_equity_curve.png")
    plt.savefig(out_img, dpi=150)
    print(f"\nSaved final prop equity curve to '{out_img}'!")