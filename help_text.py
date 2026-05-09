"""
Centralized dashboard help text and tooltip copy.
"""

TOOLTIPS = {
    "ticker_symbol": (
        "Stock or ETF ticker to analyze, such as MSFT, AAPL, SPY, or QQQ. "
        "Use valid market symbols supported by the dashboard's Yahoo Finance data source."
    ),
    "start_date": (
        "First date used for historical analysis. A longer history can improve volatility and support/resistance context, "
        "but it may also include older market regimes that are less relevant to the current setup."
    ),
    "end_date": (
        "Last date used for analysis. This is usually the most recent trading date available from the data source."
    ),
    "interval": (
        "Timeframe for each candle or data point. For example, 1d means daily bars. "
        "Shorter intervals react faster but are noisier; longer intervals are smoother but slower."
    ),
    "show_sma": (
        "Toggle the simple moving average overlay on the price chart. "
        "Moving averages help visualize trend direction but should not be used alone."
    ),
    "sma_period": (
        "Number of periods used to calculate the simple moving average. "
        "Common daily-chart values are 20 for short-term context, 50 for intermediate trend, and 200 for long-term trend."
    ),
    "show_ema": (
        "Toggle the exponential moving average overlay on the price chart. "
        "EMAs respond faster to recent price changes than SMAs."
    ),
    "ema_period": (
        "Number of periods used to calculate the exponential moving average. "
        "Common daily-chart values are 10 or 20 for short-term momentum and 50 or 200 for longer trend context."
    ),
    "show_actionable_zones": (
        "Show nearby support and resistance zones that the dashboard currently considers actionable. "
        "These are the levels most relevant to the current price."
    ),
    "show_historical_levels": (
        "Show older or more distant support and resistance levels that are not currently actionable. "
        "Useful for context, but these zones may be too far away to affect the next move."
    ),
    "max_actionable_levels": (
        "Maximum number of nearby support and resistance zones shown as actionable. "
        "Higher values show more levels but can make the chart busier."
    ),
    "swing_sensitivity": (
        "Controls how strict the dashboard is when detecting swing highs and swing lows. "
        "Higher values produce fewer but more significant swing points; lower values react faster but create more noise."
    ),
    "include_dynamic_levels": (
        "Include EMA20 and SMA50 as dynamic support or resistance candidates. "
        "These can help in trending markets where static swing levels alone are not enough."
    ),
    "use_regime_engine": (
        "Show the dashboard's rule-based market regime and signal interpretation module. "
        "This is a decision-support view, not an automated trading command."
    ),
    "show_backtest": (
        "Show the historical no-look-ahead backtest summary for the current rules. "
        "This is exploratory analysis only and does not include trading costs, slippage, taxes, or execution risk."
    ),
    "enable_vcp_detection": (
        "Enable the Volatility Contraction Pattern detector. "
        "VCP is treated as a setup or watchlist condition, not a guaranteed breakout or trade recommendation."
    ),
    "show_vcp_annotations": (
        "Show swing markers and pullback annotations on the VCP evidence chart when a VCP candidate is displayed."
    ),
    "show_vcp_diagnostics": (
        "Show the VCP evidence chart even when no strong VCP candidate is detected. "
        "Useful for learning, but it can add visual clutter."
    ),
    "vcp_base_min_length": (
        "Minimum number of daily bars used when scanning for a VCP base. "
        "Shorter bases are faster but noisier; longer bases are more selective."
    ),
    "vcp_base_max_length": (
        "Maximum number of daily bars used when scanning for a VCP base. "
        "Larger values allow deeper or longer consolidations to be considered."
    ),
    "regime": (
        "Current market environment detected by the dashboard, such as trending, ranging, or high volatility. "
        "Regime helps decide whether breakout, pullback, mean-reversion, or wait-and-see behavior is more appropriate."
    ),
    "current_bias": (
        "Directional context based on the dashboard's technical rules, including trend, moving averages, momentum, "
        "support/resistance, and confirmation indicators. Higher-confidence bullish or bearish bias is better only when risk is controlled."
    ),
    "signal_type": (
        "Classification of the current technical signal. It summarizes whether the dashboard sees a trend-following, "
        "breakout, mean-reversion, neutral, or similar setup depending on the current rules."
    ),
    "trade_trigger": (
        "Suggested action from the rule-based system. BUY means conditions may support a bullish trade, SELL may support "
        "exit or bearish action, and HOLD means the setup is not strong or clear enough for action."
    ),
    "score": (
        "Numerical score behind the Trade Trigger. Positive values favor bullish conditions, negative values favor bearish conditions, "
        "and values near zero indicate mixed or neutral evidence. The score should be interpreted together with regime and risk/reward."
    ),
    "setup_type": (
        "Plain-English label describing the detected setup, such as breakout, pullback, trend continuation, reversal, exhaustion, "
        "high-volatility neutral, or standard standby behavior. Higher is not inherently better; context matters."
    ),
    "signal_strength": (
        "Rule-based score from 0% to 100% summarizing how many bullish or bearish conditions are currently aligned. "
        "Higher values mean a stronger technical setup, but this is not a calibrated probability of success. "
        "Rough guide: below 40% = weak/noisy, 40-60% = mixed, 60-75% = moderate, above 75% = strong but still requires risk control."
    ),
    "annual_volatility": (
        "Estimated annualized volatility based on recent price movements. Higher volatility means the ticker has been moving more aggressively. "
        "High volatility can create opportunity, but also increases risk, stop distance, and position sizing concerns."
    ),
    "volatility_percentile": (
        "Shows where current volatility ranks compared with the ticker's own recent volatility history. "
        "For example, 91% means volatility is higher than about 91% of the selected lookback period. High values are context-dependent and can signal unusual risk."
    ),
    "high_vol_threshold": (
        "Percentile threshold used to classify the market as high volatility. "
        "If the volatility percentile rises above this level, the dashboard treats the ticker as being in a high-volatility regime."
    ),
    "nearest_support": (
        "Closest price level below or near the current price where buyers have historically stepped in. "
        "Support may come from swing lows, clustered levels, moving averages, or other dashboard rules. Support is not guaranteed to hold."
    ),
    "nearest_resistance": (
        "Closest price level above or near the current price where sellers have historically appeared. "
        "Resistance can act as a profit target or an area where price could stall."
    ),
    "ema20": (
        "20-period exponential moving average. It gives more weight to recent prices and is commonly used to estimate short-term trend. "
        "Price above EMA20 can suggest short-term strength; price below EMA20 can suggest weakness."
    ),
    "actionable_threshold": (
        "Distance filter used to decide whether a support or resistance zone is close enough to matter right now. "
        "Zones beyond this threshold are treated as historical context rather than actionable levels."
    ),
    "market_regime": (
        "Same regime concept as the headline regime card, shown here inside the trade setup panel so the support, confirmation, "
        "and invalidation levels can be interpreted in the right context."
    ),
    "confirmation_level": (
        "Price or technical level that would provide stronger evidence that the setup is working. "
        "For bullish setups this may be a breakout above resistance; for bearish setups it may be a break below support."
    ),
    "invalidation_level": (
        "Price or technical level where the setup would no longer be valid. "
        "This can help define a stop-loss or risk-control area, but it is not a guarantee of execution price."
    ),
    "potential_upside": (
        "Estimated upside from the current price to the nearest resistance or target level. "
        "Higher upside is helpful only when downside risk is controlled."
    ),
    "downside_risk": (
        "Estimated downside from the current price to the nearest support or invalidation level. "
        "Lower downside risk is generally better, but extremely tight levels can also cause premature stop-outs."
    ),
    "risk_reward": (
        "Ratio comparing potential upside to downside risk. Values above 1.0 mean potential upside is larger than downside risk. "
        "Many traders prefer setups above 2.0, but the acceptable range depends on strategy and win rate."
    ),
    "latest_close": (
        "Most recent closing price in the selected dataset."
    ),
    "twenty_bar_return": (
        "Percentage return over the last 20 bars. Higher positive values indicate stronger recent momentum; negative values indicate recent weakness."
    ),
    "period_return": (
        "Percentage return from the selected start date to the selected end date using the same close-price series used elsewhere in the dashboard."
    ),
    "average_volume_20": (
        "Average daily volume over the last 20 bars. Higher volume often means stronger participation, but it is context-dependent."
    ),
    "cluster_tolerance": (
        "ATR-adjusted tolerance used when grouping nearby swing highs or lows into support/resistance zones. "
        "More volatile tickers get a wider clustering tolerance."
    ),
    "max_drawdown": (
        "Largest peak-to-trough decline over the selected period. Larger negative drawdowns indicate deeper historical pullbacks."
    ),
    "start_price": (
        "Closing price at the start of the selected analysis window."
    ),
    "end_price": (
        "Closing price at the end of the selected analysis window."
    ),
    "bars_used": (
        "Number of candles or bars used in the current analysis window."
    ),
    "last_data_refresh": (
        "Timestamp for when the current dataset was most recently fetched and normalized by the dashboard. "
        "If the same inputs are requested again before the cache expires, Streamlit may reuse the cached result instead of downloading fresh data."
    ),
    "rows_downloaded": (
        "Number of cleaned rows available for the current analysis after the dashboard removes incomplete OHLC rows. "
        "More rows usually improve long-lookback context, but very large datasets can make advanced scans slower."
    ),
    "data_completeness": (
        "Percentage of downloaded rows that were complete enough to keep after OHLC cleaning. "
        "Values near 100% are best. Lower values mean more missing or unusable data in the selected range."
    ),
    "cache_ttl": (
        "How long Streamlit keeps cached data or calculation results before refreshing them automatically. "
        "Daily data uses a longer cache window than intraday data so the app stays responsive without holding stale data forever."
    ),
    "performance_diagnostics": (
        "Optional timing breakdown for major dashboard steps such as data download, indicator calculations, support/resistance detection, VCP analysis, backtest, and chart rendering. "
        "Use this to see which parts of the app are doing the most work."
    ),
    "total_signals": (
        "Total number of historical BUY and SELL signals generated by the current rule set over the selected period."
    ),
    "buy_signals": (
        "Number of historical BUY signals generated by the backtest."
    ),
    "sell_signals": (
        "Number of historical SELL signals generated by the backtest."
    ),
    "buy_win_rate": (
        "Percentage of historical BUY signals that produced a positive forward return over the dashboard's primary evaluation horizon. "
        "Higher is generally better, but it does not reflect trade sizing or drawdown."
    ),
    "sell_win_rate": (
        "Percentage of historical SELL signals that produced a positive inverse forward return over the primary evaluation horizon. "
        "Higher is generally better, but it is still context-dependent."
    ),
    "average_forward_return": (
        "Average forward return after historical signals over the specified horizon. Higher positive values are generally better, but averages can be skewed by outliers."
    ),
    "median_forward_return": (
        "Median forward return after historical signals over the specified horizon. Median values are often more stable than averages because they are less affected by outliers."
    ),
    "signal_equity_drawdown": (
        "Maximum drawdown of the simplified signal-based equity curve used by the backtest. Larger negative values indicate rougher historical performance."
    ),
    "buy_and_hold_return": (
        "Return from simply holding the ticker from the backtest start date to the end date, shown as a comparison with the dashboard's signal logic."
    ),
    "wills_signal_score": (
        "Daily-chart bullish setup score from 0 to 10. Higher scores mean more bullish continuation conditions are aligned. "
        "This score is separate from the dashboard's other signal engines and does not guarantee a successful trade."
    ),
    "wills_signal_interpretation": (
        "Plain-English summary of the current Will's Signal score band. "
        "Use it as a quick bias guide, then confirm with the indicator breakdown and risk controls."
    ),
    "wills_exhaustion_signal_score": (
        "Daily-chart exhaustion warning score from 0 to 10. Higher scores mean more evidence that a bullish move may be aging, weakening, or becoming extended. "
        "This is a warning score, not an automatic sell signal."
    ),
    "wills_exhaustion_interpretation": (
        "Plain-English summary of the current Will's Exhaustion Signal score band. "
        "Higher scores suggest more caution, tighter risk control, or the need to review whether the move is becoming tired."
    ),
    "daily_bars_used": (
        "Number of daily candles used to calculate Will's Signal and Will's Exhaustion Signal. "
        "These sections always use daily data only, even when the chart interval is changed elsewhere."
    ),
    "non_scored_warnings": (
        "Supplemental warnings that do not change the Will's Signal score. "
        "These are caution flags for earnings risk or price extension, not score components."
    ),
    "vcp_status": (
        "Current status from the Volatility Contraction Pattern detector. "
        "VCP is a watchlist or setup condition, not an automatic BUY signal."
    ),
    "vcp_score": (
        "Rule-based 0 to 100 VCP quality score built from prior uptrend, pullback contraction, volatility contraction, volume contraction, pivot proximity, and base cleanliness."
    ),
    "vcp_base_length": (
        "Number of daily bars in the consolidation base currently selected by the VCP detector. "
        "Longer bases can be more mature and selective; shorter bases react faster but can be noisier."
    ),
    "vcp_base_high": (
        "Highest price reached inside the detected VCP base."
    ),
    "vcp_base_low": (
        "Lowest price reached inside the detected VCP base."
    ),
    "vcp_base_depth": (
        "Percentage drawdown from the top of the detected base to the lowest point in the base. "
        "Shallower, tighter bases are often preferred, while deeper bases can be weaker or more volatile."
    ),
    "vcp_pullback_sequence": (
        "Sequence of pullbacks identified inside the detected base. "
        "Smaller pullbacks over time can suggest constructive contraction, while erratic or deep pullbacks can weaken the pattern."
    ),
    "vcp_atr_contraction_ratio": (
        "Ratio comparing current ATR-based volatility with ATR near the start of the base. "
        "Lower values suggest better volatility contraction."
    ),
    "vcp_volume_contraction_ratio": (
        "Ratio comparing recent average volume with earlier volume inside the base. "
        "Lower values suggest volume is drying up, which can be constructive for a VCP setup."
    ),
    "pivot_price": (
        "Reference breakout price used by the VCP detector. A close above the pivot is more meaningful when accompanied by stronger volume."
    ),
    "distance_to_pivot": (
        "Distance between the current close and the VCP pivot. Smaller values mean price is closer to a possible breakout point."
    ),
    "vcp_breakout_trigger": (
        "Price level that must be exceeded for the dashboard to treat the move as a potential VCP breakout trigger."
    ),
    "vcp_breakout_volume_requirement": (
        "Minimum volume threshold the dashboard expects for stronger breakout confirmation. Higher required volume means the move needs broader participation."
    ),
    "vcp_volume_confirmed": (
        "Shows whether the current breakout attempt is meeting the dashboard's volume-confirmation rule. "
        "Confirmed volume is usually better than a breakout that occurs on weak participation."
    ),
}


DASHBOARD_GUIDE = [
    (
        "Trade Trigger",
        "BUY, HOLD, or SELL from the dashboard's rule-based system. Treat this as decision support, not an automatic instruction."
    ),
    (
        "Score",
        "Shows the direction and strength of the current setup. Positive scores favor bullish conditions, negative scores favor bearish conditions, and values near zero are mixed."
    ),
    (
        "Current Bias",
        "Bullish, bearish, or neutral context based on the dashboard's existing technical rules."
    ),
    (
        "Signal Strength",
        "Rule-based quality score summarizing how aligned the current conditions are. This is not a calibrated probability and should not be treated as guaranteed odds."
    ),
    (
        "Reminder",
        "This dashboard is a rule-based decision-support tool. It helps organize technical evidence, but it does not predict the future or replace position sizing, stops, or risk management."
    ),
]
