from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import app as dashboard_app


def _synthetic_daily_frame(length: int = 260) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=length, freq="B")
    trend = np.linspace(100.0, 156.0, length)
    pullback = np.zeros(length)
    pullback[-45:] = np.linspace(0.0, -4.5, 45)
    rebound = np.zeros(length)
    rebound[-12:] = np.linspace(0.0, 6.5, 12)
    base_close = trend + pullback + rebound
    oscillation = np.sin(np.linspace(0, 18, length)) * 1.8
    close = base_close + oscillation
    open_ = close - 0.9
    high = close + 1.8
    low = close - 1.7
    volume = np.linspace(1_000_000, 1_450_000, length) + (np.cos(np.linspace(0, 16, length)) * 45_000)

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


def _stub_backtest(**kwargs) -> dict:
    return {
        "total_signals": 10,
        "buy_signals": 6,
        "sell_signals": 2,
        "buy_win_rate": 66.7,
        "sell_win_rate": 50.0,
        "average_forward_return_5": 2.1,
        "average_forward_return_10": 3.4,
        "average_forward_return_20": 5.2,
        "median_forward_return_5": 1.9,
        "median_forward_return_10": 3.1,
        "median_forward_return_20": 4.8,
        "max_drawdown": -6.5,
        "buy_and_hold_return": 21.7,
        "start_price": 100.0,
        "end_price": 121.7,
        "bars_used": 260,
        "signal_history": pd.DataFrame(),
    }


def _stub_detect_vcp(**kwargs) -> dict:
    return {
        "status": "No Clear VCP",
        "score": 0.0,
        "pivot": None,
        "current_close": float(kwargs["df"]["Close"].iloc[-1]),
        "distance_to_pivot_pct": None,
        "base_length": "N/A",
        "base_high": None,
        "base_low": None,
        "base_depth_pct": None,
        "pullbacks": [],
        "atr_contraction_ratio": None,
        "volume_contraction_ratio": None,
        "breakout_trigger_price": None,
        "breakout_volume_requirement": None,
        "breakout_volume_confirmed": False,
        "explanations": [],
        "failed_reasons": ["Synthetic render harness disables VCP pattern detection."],
        "current_visual_analysis": True,
        "volume_expanding_downward": False,
        "base_too_loose": False,
    }


dashboard_app.load_data = lambda ticker, start_date, end_date, interval: _synthetic_daily_frame(260)
dashboard_app.get_latest_available_market_date = lambda probe_ticker="SPY": date(2025, 5, 1)
dashboard_app.backtest_signals_no_lookahead = _stub_backtest
dashboard_app.detect_vcp = _stub_detect_vcp
dashboard_app.apply_vcp_to_signal = lambda signal_result, vcp_result: signal_result


dashboard_app.main()
