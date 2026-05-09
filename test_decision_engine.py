from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

from app import load_data
from backtest import backtest_signals_no_lookahead
from levels import build_level_snapshot, calculate_atr
from signals import generate_signal
from wills_signal import (
    calculate_wills_exhaustion_signal,
    calculate_wills_signal,
    interpret_wills_exhaustion_signal,
    interpret_wills_signal,
)


SYMBOLS = ["AMD", "MSFT", "SPY", "BTC-USD", "SNDK"]


def _make_daily_df(length: int = 80, final_drop: float = 0.0, final_spike: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=length, freq="B")
    base = np.linspace(100.0, 132.0, length)
    wave = np.sin(np.linspace(0, 8, length)) * 0.6
    close = base + wave
    close[-1] += final_spike
    close[-1] -= final_drop
    open_ = close - 0.5
    high = close + 1.0
    low = close - 1.0
    volume = np.linspace(1_000_000, 1_250_000, length)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def _series_frame(values: list[float], index: pd.Index, column_name: str) -> pd.DataFrame:
    return pd.DataFrame({column_name: values}, index=index)


def _patched_exhaustion_case(case_name: str):
    healthy_df = _make_daily_df()
    early_warning_df = _make_daily_df(final_spike=6.0)
    aging_df = _make_daily_df(final_spike=7.0)
    strong_warning_df = _make_daily_df(final_drop=9.0)

    if case_name == "healthy":
        df = healthy_df
        dmi = pd.DataFrame(
            {
                "adx": np.linspace(24.0, 31.0, len(df)),
                "plus_di": np.linspace(24.0, 34.0, len(df)),
                "minus_di": np.linspace(17.0, 15.0, len(df)),
                "dx": np.linspace(18.0, 28.0, len(df)),
            },
            index=df.index,
        )
        rsi = pd.Series(np.linspace(53.0, 58.0, len(df)), index=df.index)
        macd = pd.DataFrame(
            {
                "macd_line": np.linspace(0.6, 1.3, len(df)),
                "signal_line": np.linspace(0.4, 1.0, len(df)),
                "histogram": np.linspace(0.05, 0.30, len(df)),
            },
            index=df.index,
        )
        atr = pd.Series([3.0] * len(df), index=df.index)
        expected_band = "Trend Still Healthy"
    elif case_name == "early":
        df = early_warning_df
        dmi = pd.DataFrame(
            {
                "adx": np.concatenate([np.linspace(26.0, 39.8, len(df) - 1), [40.0]]),
                "plus_di": np.concatenate([np.linspace(26.0, 32.0, len(df) - 1), [31.0]]),
                "minus_di": np.concatenate([np.linspace(12.0, 16.0, len(df) - 1), [16.5]]),
                "dx": np.linspace(20.0, 25.0, len(df)),
            },
            index=df.index,
        )
        rsi = pd.Series(np.concatenate([np.linspace(57.0, 72.5, len(df) - 1), [73.0]]), index=df.index)
        macd = pd.DataFrame(
            {
                "macd_line": np.linspace(0.6, 1.15, len(df)),
                "signal_line": np.linspace(0.45, 1.0, len(df)),
                "histogram": np.concatenate([np.linspace(0.30, 0.16, len(df) - 2), [0.12, 0.09]]),
            },
            index=df.index,
        )
        atr = pd.Series([2.4] * len(df), index=df.index)
        expected_band = "Early Warning"
    elif case_name == "aging":
        df = aging_df
        adx_values = np.concatenate([np.linspace(24.0, 37.5, len(df) - 1), [35.5]])
        plus_values = np.concatenate([np.linspace(25.0, 31.0, len(df) - 1), [30.0]])
        minus_values = np.concatenate([np.linspace(14.0, 18.0, len(df) - 1), [19.5]])
        dmi = pd.DataFrame(
            {
                "adx": adx_values,
                "plus_di": plus_values,
                "minus_di": minus_values,
                "dx": np.linspace(20.0, 27.0, len(df)),
            },
            index=df.index,
        )
        rsi_values = np.concatenate([np.linspace(58.0, 75.0, len(df) - 3), [76.0, 74.5, 72.0]])
        rsi = pd.Series(rsi_values, index=df.index)
        macd = pd.DataFrame(
            {
                "macd_line": np.linspace(0.7, 1.2, len(df)),
                "signal_line": np.linspace(0.45, 1.0, len(df)),
                "histogram": np.concatenate([np.linspace(0.32, 0.15, len(df) - 2), [0.11, 0.08]]),
            },
            index=df.index,
        )
        atr = pd.Series([2.8] * len(df), index=df.index)
        expected_band = "Move May Be Aging \u2014 Tighten Stops"
    else:
        df = strong_warning_df
        adx_values = np.concatenate([np.linspace(27.0, 38.0, len(df) - 1), [34.0]])
        plus_values = np.concatenate([np.linspace(26.0, 28.0, len(df) - 1), [17.0]])
        minus_values = np.concatenate([np.linspace(14.0, 20.0, len(df) - 1), [22.0]])
        dmi = pd.DataFrame(
            {
                "adx": adx_values,
                "plus_di": plus_values,
                "minus_di": minus_values,
                "dx": np.linspace(18.0, 26.0, len(df)),
            },
            index=df.index,
        )
        rsi_values = np.concatenate([np.linspace(60.0, 76.0, len(df) - 3), [77.0, 73.0, 68.0]])
        rsi = pd.Series(rsi_values, index=df.index)
        macd = pd.DataFrame(
            {
                "macd_line": np.concatenate([np.linspace(0.8, 1.1, len(df) - 1), [0.7]]),
                "signal_line": np.concatenate([np.linspace(0.5, 1.0, len(df) - 1), [0.82]]),
                "histogram": np.concatenate([np.linspace(0.25, 0.06, len(df) - 2), [0.02, -0.08]]),
            },
            index=df.index,
        )
        atr = pd.Series([2.6] * len(df), index=df.index)
        expected_band = "Strong Sell / Trim Warning"

    return df, dmi, rsi, macd, atr, expected_band


def verify_wills_signal_interpretation_bands():
    cases = [
        (9, "Strong Long Candidate"),
        (6, "Tradable / Watchlist Candidate"),
        (3, "Weak / Usually Skip"),
    ]
    for score, expected_label in cases:
        interpretation = interpret_wills_signal(score)
        assert interpretation["label"] == expected_label


def verify_wills_exhaustion_interpretation_bands():
    cases = [
        (2, "Trend Still Healthy"),
        (4, "Early Warning"),
        (6, "Move May Be Aging \u2014 Tighten Stops"),
        (9, "Strong Sell / Trim Warning"),
    ]
    for score, expected_label in cases:
        interpretation = interpret_wills_exhaustion_signal(score)
        assert interpretation["label"] == expected_label


def verify_wills_signal_insufficient_data():
    short_df = pd.DataFrame(
        {
            "Open": [10.0, 10.5, 10.2, 10.4, 10.6],
            "High": [10.5, 10.7, 10.4, 10.8, 10.9],
            "Low": [9.8, 10.0, 10.1, 10.2, 10.4],
            "Close": [10.2, 10.3, 10.25, 10.7, 10.8],
            "Volume": [1000, 1200, 1100, 1300, 1250],
        },
        index=pd.date_range("2026-01-01", periods=5, freq="D"),
    )
    wills_signal = calculate_wills_signal(short_df, ticker="TEST", include_earnings_warning=False)
    exhaustion_signal = calculate_wills_exhaustion_signal(short_df, ticker="TEST")
    assert wills_signal["score"] == 0
    assert len(wills_signal["rows"]) == 6
    assert all(row["points"] == 0 for row in wills_signal["rows"])
    assert all(row["rule_triggered"] == "Insufficient data" for row in wills_signal["rows"])
    assert exhaustion_signal["score"] == 0
    assert len(exhaustion_signal["rows"]) == 5
    assert all(row["points"] == 0 for row in exhaustion_signal["rows"])
    assert all(row["rule_triggered"] == "Insufficient data" for row in exhaustion_signal["rows"])


def verify_wills_exhaustion_score_bands():
    for case_name, min_score, max_score in [
        ("healthy", 0, 3),
        ("early", 4, 5),
        ("aging", 6, 7),
        ("strong", 8, 10),
    ]:
        df, dmi, rsi, macd, atr, expected_label = _patched_exhaustion_case(case_name)
        with (
            patch("wills_signal.calculate_dmi_adx", return_value=dmi),
            patch("wills_signal.calculate_rsi", return_value=rsi),
            patch("wills_signal.calculate_macd", return_value=macd),
            patch("wills_signal.calculate_atr_series", return_value=atr),
        ):
            exhaustion_signal = calculate_wills_exhaustion_signal(df, ticker="TEST")

        assert min_score <= exhaustion_signal["score"] <= max_score
        assert exhaustion_signal["score"] == sum(row["points"] for row in exhaustion_signal["rows"])
        assert len(exhaustion_signal["rows"]) == 5
        assert exhaustion_signal["interpretation"]["label"] == expected_label


def verify_wills_signal_independence():
    df, dmi, rsi, macd, atr, _ = _patched_exhaustion_case("strong")
    with (
        patch("wills_signal.calculate_dmi_adx", return_value=dmi),
        patch("wills_signal.calculate_rsi", return_value=rsi),
        patch("wills_signal.calculate_macd", return_value=macd),
        patch("wills_signal.calculate_atr_series", return_value=atr),
    ):
        bullish_signal_before = calculate_wills_signal(df, ticker="TEST", include_earnings_warning=False)
        exhaustion_signal = calculate_wills_exhaustion_signal(df, ticker="TEST")
        bullish_signal_after = calculate_wills_signal(df, ticker="TEST", include_earnings_warning=False)

    assert bullish_signal_before["score"] == bullish_signal_after["score"]
    assert bullish_signal_before["rows"] == bullish_signal_after["rows"]
    assert exhaustion_signal["score"] >= 0


def main():
    end = datetime(2025, 5, 1)
    start = end - timedelta(days=365)

    print("=" * 80)
    print("SIGNAL INTERPRETATION ENGINE SMOKE TEST")
    print("=" * 80)

    verify_wills_signal_interpretation_bands()
    verify_wills_exhaustion_interpretation_bands()
    verify_wills_signal_insufficient_data()
    verify_wills_exhaustion_score_bands()
    verify_wills_signal_independence()
    print("Will's Signal interpretation bands: PASS")
    print("Will's Exhaustion Signal interpretation bands: PASS")
    print("Will's Signal and Will's Exhaustion Signal insufficient data handling: PASS")
    print("Will's Exhaustion Signal synthetic band tests: PASS")
    print("Will's Signal independence from exhaustion score: PASS")

    skipped_symbols = []

    for symbol in SYMBOLS:
        print(f"\nTesting {symbol}")
        print("-" * 80)

        df = load_data(symbol, start, end, "1d")
        if df.empty:
            skipped_symbols.append(symbol)
            print(f"Skipped {symbol}: no data returned from Yahoo Finance for the selected range.")
            continue

        atr = calculate_atr(df, period=14)
        level_snapshot = build_level_snapshot(
            df,
            current_price=float(df["Close"].iloc[-1]),
            atr=atr,
            max_levels=4,
            relevance_atr=5.0,
            relevance_pct=0.25,
            show_all_historical=False,
        )

        signal = generate_signal(
            df=df,
            ticker=symbol,
            support_levels=level_snapshot["actionable_supports"],
            resistance_levels=level_snapshot["actionable_resistances"],
            max_levels=4,
        )

        assert signal["trade_signal"] in {"BUY", "SELL", "HOLD"}
        assert signal["current_bias"] in {"BUY", "SELL", "WATCH BULLISH", "WATCH BEARISH", "NEUTRAL"}
        assert signal["regime"] in {"Trending Up", "Trending Down", "Ranging", "High Volatility", "Neutral"}
        assert 0 <= signal["signal_strength"] <= 95
        assert signal["trade_setup"]["market_regime"]
        if signal["high_volatility"]:
            assert signal["signal_strength"] <= 70

        backtest = backtest_signals_no_lookahead(
            df=df,
            ticker=symbol,
            max_levels=4,
            lookback_periods=(5, 10, 20),
            relevance_atr=5.0,
            relevance_pct=0.25,
        )
        wills_signal = calculate_wills_signal(df, ticker=symbol)
        exhaustion_signal = calculate_wills_exhaustion_signal(df, ticker=symbol)

        print(f"Latest close: ${df['Close'].iloc[-1]:.2f}")
        print(f"Regime: {signal['regime_label']}")
        print(f"Current bias: {signal['current_bias']} | Trade trigger: {signal['trade_signal']} | Score: {signal['score']:+.2f}")
        if signal["features"]["annual_vol_percentile"] is not None:
            print(
                f"Volatility: {signal['features']['annual_vol'] * 100:.2f}% annualized "
                f"({signal['features']['annual_vol_percentile'] * 100:.0f}th percentile)"
            )
        print(f"Signal type: {signal['signal_style']}")
        print(f"Nearest support: {signal['nearest_levels']['support']}")
        print(f"Nearest resistance: {signal['nearest_levels']['resistance']}")
        print(
            "Will's Signal: "
            f"{wills_signal['score']}/{wills_signal['max_score']} | "
            f"{wills_signal['interpretation']['label']}"
        )
        print(
            "Will's Exhaustion Signal: "
            f"{exhaustion_signal['score']}/{exhaustion_signal['max_score']} | "
            f"{exhaustion_signal['interpretation']['label']}"
        )
        assert wills_signal["score"] == sum(row["points"] for row in wills_signal["rows"])
        assert exhaustion_signal["score"] == sum(row["points"] for row in exhaustion_signal["rows"])
        assert 0 <= wills_signal["score"] <= 10
        assert 0 <= exhaustion_signal["score"] <= 10
        assert len(wills_signal["rows"]) == 6
        assert len(exhaustion_signal["rows"]) == 5
        assert all(key in wills_signal["warnings"] for key in ["earnings", "atr_extension"])
        assert wills_signal["warnings"]["earnings"]["message"]
        assert wills_signal["warnings"]["atr_extension"]["message"]
        print(
            "Backtest: "
            f"{backtest['total_signals']} signals, "
            f"BUY win rate {backtest['buy_win_rate']:.1f}%, "
            f"SELL win rate {backtest['sell_win_rate']:.1f}%, "
            f"buy-and-hold {backtest['buy_and_hold_return']:.2f}%"
        )

    if skipped_symbols:
        print("\nSkipped symbols due to unavailable data:")
        for symbol in skipped_symbols:
            print(f"- {symbol}")

    print("\nSmoke test completed.")


if __name__ == "__main__":
    main()
