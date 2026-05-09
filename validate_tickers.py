from datetime import datetime, timedelta

from app import calculate_annualized_volatility, calculate_max_drawdown, get_series, load_data
from levels import build_level_snapshot, calculate_atr, get_nearest_levels


symbols = ["AAPL", "AMD", "SPY", "BTC-USD"]
end = datetime.now()
start = end - timedelta(days=365)

for symbol in symbols:
    df = load_data(symbol, start, end, "1d")
    if df.empty:
        raise SystemExit(f"No data for {symbol}")

    close_series = get_series(df, "Close")
    close = float(close_series.iloc[-1])
    atr = calculate_atr(df, period=14)
    level_snapshot = build_level_snapshot(
        df,
        current_price=close,
        atr=atr,
        max_levels=4,
        show_all_historical=False,
    )
    nearest = get_nearest_levels(
        current_price=close,
        support_levels=level_snapshot["actionable_supports"],
        resistance_levels=level_snapshot["actionable_resistances"],
        atr=atr,
    )
    vol = calculate_annualized_volatility(close_series, symbol)
    mdd = calculate_max_drawdown(close_series)
    print(
        symbol,
        "rows",
        len(df),
        "close",
        f"{close:.2f}",
        "vol%",
        f"{vol:.2f}",
        "mdd%",
        f"{mdd:.2f}",
        "sup",
        nearest["support"],
        "res",
        nearest["resistance"],
    )
