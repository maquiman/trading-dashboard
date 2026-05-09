# Trading Dashboard

Streamlit dashboard for stocks and crypto with a fully rule-based, explainable signal engine. Version 4.5 adds richer signal interpretation, volatility percentile logic, and a more actionable trade setup panel without using machine learning.

## What Changed

- Added a clearly defined market regime detector:
  - `Trending Up`
  - `Trending Down`
  - `Ranging`
  - `High Volatility` as an overlay or standalone regime when no directional structure is present
- Rebuilt signal scoring around explicit rule weights instead of opaque heuristics
- Renamed `Confidence` to `Signal Strength`
- Added the note: `This is rule-based signal strength, not a calibrated probability.`
- Added signal style classification:
  - `Trend-Following`
  - `Mean-Reversion`
  - `Breakout`
  - `Neutral`
- Updated the backtest to generate signals historically with only data available up to each date
- Added buy-and-hold comparison and signal-based equity drawdown
- Added actionable-vs-historical level separation for parabolic names and price-discovery mode
- Added volatility percentile-based high-volatility detection and watch-state bias labels
- Added a trade setup interpretation panel with confirmation, invalidation, and simple risk/reward

## Signal Engine Rules

The dashboard uses fixed indicator settings for the signal engine, even if the chart overlays use different display periods:

- Returns:
  - `daily_return = close.pct_change()`
  - `5_period_return = close / close.shift(5) - 1`
  - `20_period_return = close / close.shift(20) - 1`
- EMA:
  - `EMA20 = close.ewm(span=20, adjust=False).mean()`
  - `EMA20 slope = EMA20 / EMA20.shift(5) - 1`
- SMA:
  - `SMA50 = close.rolling(50).mean()`
  - `SMA50 slope = SMA50 / SMA50.shift(5) - 1`
- Volatility:
  - `daily_vol = daily_return.rolling(20).std()`
  - `annual_vol = daily_vol * sqrt(252)` for stocks
  - `annual_vol = daily_vol * sqrt(365)` for tickers containing `-USD`
- ATR:
  - `true_range = max(high-low, abs(high-prev_close), abs(low-prev_close))`
  - `ATR14 = true_range.rolling(14).mean()`
- Support and resistance distance:
  - `distance_to_support_atr = (close - nearest_support) / ATR`
  - `distance_to_resistance_atr = (nearest_resistance - close) / ATR`
  - `distance_to_support_pct = close / nearest_support - 1`
  - `distance_to_resistance_pct = nearest_resistance / close - 1`

## Regime Detection

`Trending Up`

- `close > EMA20`
- `EMA20 slope over 5 periods > +1%`
- `close > SMA50` when SMA50 is available
- `20_period_return > +5%`

`Trending Down`

- `close < EMA20`
- `EMA20 slope over 5 periods < -1%`
- `close < SMA50` when SMA50 is available
- `20_period_return < -5%`

`High Volatility`

- `annual_vol` is elevated relative to the ticker's own selected-period history
- current 20-day annualized volatility percentile is shown in the dashboard
- the warning banner appears only when volatility is in the high percentile regime

`Ranging`

- not `Trending Up`
- not `Trending Down`
- `20_period_return` between `-5%` and `+5%`
- price is between the nearest support and nearest resistance

## Scoring Model

Base score starts at `0`.

`Trending Up`

- `+2` if `close > EMA20`
- `+1` if `EMA20 slope > 0`
- `+1` if `5_period_return > 0`
- `+2` if `close` makes a 20-bar high
- `-1` if annual volatility is above its 75th percentile
- `-2` if price is within `0.75 ATR` of strong resistance

`Trending Down`

- `-2` if `close < EMA20`
- `-1` if `EMA20 slope < 0`
- `-1` if `5_period_return < 0`
- `-2` if `close` makes a 20-bar low
- `+1` if annual volatility is above its 75th percentile
- `+2` if price is within `0.75 ATR` of strong support

`Ranging`

- `+2` if price is within `0.75 ATR` above support
- `-2` if price is within `0.75 ATR` below resistance
- `+1` if `RSI < 35`
- `-1` if `RSI > 65`
- breakout rewards are suppressed unless price closes outside the zone by more than `1 ATR`

`High Volatility`

- current 20-day annualized volatility must be above the ticker's own high percentile threshold
- the current implementation uses the `80th percentile`
- final score is reduced by `30%`
- signal strength is capped at `70%`
- explanation includes: `High volatility reduces signal reliability`

Breakout and breakdown logic:

- Resistance breakout:
  - if `close > nearest_resistance + 0.5 ATR`
  - in a ranging regime, breakout reward requires more than `1 ATR`
  - `+2` if `volume > 1.2 * average_volume_20`
  - `+1` if `EMA20 slope > 0`
  - `+1` if `close` is a 20-bar high
- Support breakdown:
  - if `close < nearest_support - 0.5 ATR`
  - in a ranging regime, breakdown reward requires more than `1 ATR`
  - `-2` if `volume > 1.2 * average_volume_20`
  - `-1` if `EMA20 slope < 0`
  - `-1` if `close` is a 20-bar low

Strong support and resistance are currently defined as levels with at least `2` clustered swing hits.

## Final Signal Mapping

- `score >= +3`: `BUY`
- `score <= -3`: `SELL`
- `+1 <= score < +3`: `WATCH BULLISH`
- `-3 < score <= -1`: `WATCH BEARISH`
- otherwise: `NEUTRAL`

Signal strength:

- `signal_strength = min(95, 50 + abs(score) * 10)`
- `High Volatility` caps the result at `70`
- `Signal Strength is a rule-based score, not a calibrated probability.`

## Backtest

The dashboard backtest is historical and no-look-ahead:

1. Generate a signal on each date using only information available up to that date.
2. For each `BUY`, compute forward returns over `5`, `10`, and `20` bars.
3. For each `SELL`, compute inverse forward returns over `5`, `10`, and `20` bars.
4. Report:
   - total signals
   - BUY signals
   - SELL signals
   - BUY win rate
   - SELL win rate
   - average forward return over `5`, `10`, and `20` bars
   - median forward return over `5`, `10`, and `20` bars
   - max drawdown of the signal-based equity curve
   - buy-and-hold return

## UI Updates

The dashboard now shows:

- detected regime near the top
- current bias and trade trigger separately
- signal strength
- score
- annualized volatility and volatility percentile
- explanation list
- signal style
- high-volatility warning when active
- actionable levels separate from historical levels
- price discovery and extended-move warnings
- trade setup interpretation with confirmation and invalidation levels

## Level Relevance

Support and resistance are now classified relative to the current price:

- only levels below the current price are treated as support
- only levels above the current price are treated as resistance
- distant levels are excluded from the actionable view unless they are within the configured distance filters
- default actionable filters are `5 ATR` or `25%`
- a sidebar toggle can reveal all historical levels

Additional candidates are included for parabolic and trend-driven charts:

- recent support from the last `20`, `50`, and `100` bars
- `EMA20`
- `SMA50`

If no resistance exists above the current price, the dashboard shows:

- `No overhead resistance detected - price discovery mode.`

If the price is more than `20%` above `EMA20`, the dashboard warns that the stock may be extended.

## Trade Setup Interpretation

The trade setup panel summarizes the current rule-based view:

- market regime
- current bias
- confirmation level
- invalidation level
- nearest support
- nearest resistance
- potential upside to resistance
- downside risk to support
- simple risk/reward ratio when both support and resistance exist

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

For Windows double-click launching, use [Open Trading Dashboard.bat](/C:/Users/kiyam/Trading%20Dashboard/Open%20Trading%20Dashboard.bat). You can place a shortcut to that file on your desktop and open the dashboard by double-clicking it.

## Smoke Test

Run the requested ticker validation:

```bash
python test_decision_engine.py
```

This checks the current engine against:

- `AMD`
- `MSFT`
- `SPY`
- `BTC-USD`
- `SNDK`

## Notes

- The signal engine is fully rule-based and explainable.
- No machine learning is used.
- Daily bars are the intended calibration for the regime rules. Weekly and monthly views remain available for charting, but the signal model is less reliable there.
- The backtest is exploratory and does not include trading costs, slippage, taxes, or execution risk.
