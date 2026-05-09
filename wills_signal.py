"""
Standalone daily-chart scoring for Will's Signal and Will's Exhaustion Signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

from levels import calculate_atr_series
from signals import calculate_rsi


WILLS_SIGNAL_TITLE = "Will\u2019s Signal"
WILLS_EXHAUSTION_TITLE = "Will\u2019s Exhaustion Signal"


@dataclass(frozen=True)
class WillsSignalConfig:
    """Centralized thresholds so both scores are easy to adjust later."""

    adx_period: int = 14
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_period: int = 20
    atr_period: int = 14
    min_daily_bars: int = 60
    moderate_extension_atr: float = 0.75
    high_extension_atr: float = 1.25
    fallback_extension_pct: float = 0.05
    adx_entry_low_threshold: float = 18.0
    adx_entry_strong_threshold: float = 25.0
    adx_exhaustion_strong_threshold: float = 25.0
    adx_exhaustion_very_high_threshold: float = 35.0
    adx_exhaustion_extreme_threshold: float = 40.0
    adx_flat_delta: float = 0.50
    rsi_overbought_threshold: float = 70.0
    rsi_entry_extended_threshold: float = 72.0
    rsi_healthy_low: float = 50.0
    rsi_healthy_high: float = 65.0
    rsi_divergence_lookback: int = 30
    rsi_divergence_swing_window: int = 2
    macd_histogram_shrink_bars: int = 3


DEFAULT_WILLS_SIGNAL_CONFIG = WillsSignalConfig()


def _is_valid_number(value) -> bool:
    return value is not None and pd.notna(value) and np.isfinite(value)


def _format_value(value: float | None, digits: int = 2) -> str:
    if not _is_valid_number(value):
        return "Insufficient data"
    return f"{float(value):.{digits}f}"


def _normalize_calendar_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return pd.Timestamp(value).date()
        except Exception:
            return None
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def calculate_dmi_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate daily +DI, -DI, DX, and ADX using Wilder-style smoothing."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
        dtype=float,
    )

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1 / period
    smoothed_tr = true_range.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return pd.DataFrame(
        {
            "plus_di": plus_di,
            "minus_di": minus_di,
            "adx": adx,
            "dx": dx,
        },
        index=df.index,
    )


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """Calculate daily MACD components from the close series."""
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
        },
        index=close.index,
    )


def _build_indicator_result(category: str, current_reading: str, rule_triggered: str, points: int, diagnostic: str) -> dict:
    return {
        "category": category,
        "current_reading": current_reading,
        "rule_triggered": rule_triggered,
        "points": int(points),
        "diagnostic": diagnostic,
    }


def _insufficient_indicator(category: str, message: str) -> dict:
    return _build_indicator_result(category, "Insufficient data", "Insufficient data", 0, message)


def _empty_rows_for_signal() -> list[dict]:
    return [
        _insufficient_indicator("ADX(14)", "Insufficient data to calculate ADX(14)."),
        _insufficient_indicator("+DI vs -DI", "Insufficient data to calculate +DI and -DI."),
        _insufficient_indicator("RSI(14)", "Insufficient data to calculate RSI(14)."),
        _insufficient_indicator("MACD Daily", "Insufficient data to calculate MACD."),
        _insufficient_indicator("20 EMA Slope", "Insufficient data to calculate the 20 EMA slope."),
        _insufficient_indicator("Price vs 20 EMA", "Insufficient data to compare price with the 20 EMA."),
    ]


def _empty_rows_for_exhaustion() -> list[dict]:
    return [
        _insufficient_indicator("ADX Trend Aging", "Insufficient data to calculate ADX trend aging."),
        _insufficient_indicator("+DI / -DI Weakening", "Insufficient data to calculate +DI and -DI weakening."),
        _insufficient_indicator("RSI Exhaustion / Divergence", "Insufficient data to calculate RSI exhaustion."),
        _insufficient_indicator("MACD Weakening", "Insufficient data to calculate MACD weakening."),
        _insufficient_indicator("Price vs 20 EMA / Extension", "Insufficient data to evaluate price extension."),
    ]


def _build_common_context(daily_df: pd.DataFrame, config: WillsSignalConfig) -> dict:
    """Build daily-only indicators once so both scores use the same context."""
    close = daily_df["Close"].astype(float)
    dmi_adx = calculate_dmi_adx(daily_df, period=config.adx_period)
    rsi_series = calculate_rsi(close, period=config.rsi_period)
    macd_frame = calculate_macd(
        close,
        fast=config.macd_fast,
        slow=config.macd_slow,
        signal_period=config.macd_signal,
    )
    ema20 = close.ewm(span=config.ema_period, adjust=False).mean()
    atr14 = calculate_atr_series(daily_df, period=config.atr_period)

    return {
        "close": close,
        "dmi_adx": dmi_adx,
        "rsi": rsi_series,
        "macd": macd_frame,
        "ema20": ema20,
        "atr14": atr14,
    }


def _score_entry_adx(indicators: pd.DataFrame, config: WillsSignalConfig) -> dict:
    current_adx = indicators["adx"].iloc[-1] if len(indicators) >= 1 else np.nan
    prior_adx = indicators["adx"].iloc[-2] if len(indicators) >= 2 else np.nan

    if not (_is_valid_number(current_adx) and _is_valid_number(prior_adx)):
        return _insufficient_indicator("ADX(14)", "Insufficient data to calculate ADX(14).")

    current_adx = float(current_adx)
    prior_adx = float(prior_adx)
    rising = current_adx > prior_adx
    reading = f"ADX {_format_value(current_adx)}, prior {_format_value(prior_adx)}, {'rising' if rising else 'not rising'}"

    if current_adx < config.adx_entry_low_threshold:
        return _build_indicator_result(
            "ADX(14)",
            reading,
            "ADX below 18",
            0,
            "ADX is below 18, suggesting the stock may be in a weak or choppy trend environment.",
        )
    if current_adx <= config.adx_entry_strong_threshold:
        return _build_indicator_result(
            "ADX(14)",
            reading,
            "ADX between 18 and 25",
            1,
            "ADX suggests a trend may be developing, but trend strength is not yet clearly strong.",
        )
    if rising:
        return _build_indicator_result(
            "ADX(14)",
            reading,
            "ADX above 25 and rising",
            2,
            "ADX is above 25 and rising, suggesting a strong and strengthening trend.",
        )
    return _build_indicator_result(
        "ADX(14)",
        reading,
        "ADX above 25 but not rising",
        1,
        "ADX suggests a trend may be developing, but trend strength is not yet clearly strong.",
    )


def _score_entry_di(indicators: pd.DataFrame) -> dict:
    current_plus = indicators["plus_di"].iloc[-1] if len(indicators) >= 1 else np.nan
    current_minus = indicators["minus_di"].iloc[-1] if len(indicators) >= 1 else np.nan
    prior_plus = indicators["plus_di"].iloc[-2] if len(indicators) >= 2 else np.nan
    prior_minus = indicators["minus_di"].iloc[-2] if len(indicators) >= 2 else np.nan

    if not all(_is_valid_number(value) for value in [current_plus, current_minus, prior_plus, prior_minus]):
        return _insufficient_indicator("+DI vs -DI", "Insufficient data to calculate +DI and -DI.")

    current_plus = float(current_plus)
    current_minus = float(current_minus)
    prior_plus = float(prior_plus)
    prior_minus = float(prior_minus)
    current_spread = current_plus - current_minus
    prior_spread = prior_plus - prior_minus
    widening = current_spread > prior_spread
    reading = (
        f"+DI {_format_value(current_plus)}, -DI {_format_value(current_minus)}, "
        f"spread {_format_value(current_spread)}, prior spread {_format_value(prior_spread)}"
    )

    if current_plus <= current_minus:
        return _build_indicator_result(
            "+DI vs -DI",
            reading,
            "-DI above +DI",
            0,
            "-DI is above +DI, suggesting sellers currently have stronger directional pressure.",
        )
    if widening:
        return _build_indicator_result(
            "+DI vs -DI",
            reading,
            "+DI above -DI and widening",
            2,
            "+DI is above -DI and the gap is widening, suggesting bullish directional pressure is increasing.",
        )
    return _build_indicator_result(
        "+DI vs -DI",
        reading,
        "+DI above -DI",
        1,
        "+DI is above -DI, suggesting buyers have stronger directional pressure.",
    )


def _score_entry_rsi(rsi_series: pd.Series, config: WillsSignalConfig) -> dict:
    current_rsi = rsi_series.iloc[-1] if len(rsi_series) >= 1 else np.nan
    prior_rsi = rsi_series.iloc[-2] if len(rsi_series) >= 2 else np.nan

    if not (_is_valid_number(current_rsi) and _is_valid_number(prior_rsi)):
        return _insufficient_indicator("RSI(14)", "Insufficient data to calculate RSI(14).")

    current_rsi = float(current_rsi)
    prior_rsi = float(prior_rsi)
    turning_up = current_rsi > prior_rsi
    reading = f"RSI {_format_value(current_rsi)}, prior {_format_value(prior_rsi)}"

    if current_rsi < 40:
        return _build_indicator_result(
            "RSI(14)",
            reading,
            "RSI below 40",
            0,
            "RSI is below 40, suggesting bullish momentum may be weakening.",
        )
    if current_rsi < 50:
        if turning_up:
            return _build_indicator_result(
                "RSI(14)",
                reading,
                "RSI 40-50 and turning up",
                2,
                "RSI is turning up from the 40-50 pullback zone, which can be constructive for bullish continuation.",
            )
        return _build_indicator_result(
            "RSI(14)",
            reading,
            "RSI 40-50 but not turning up",
            1,
            "RSI is in a potential pullback zone, but it has not yet clearly turned higher.",
        )
    if current_rsi < config.rsi_healthy_high:
        return _build_indicator_result(
            "RSI(14)",
            reading,
            "RSI 50-65",
            2,
            "RSI is in a healthy bullish momentum range.",
        )
    if current_rsi <= config.rsi_entry_extended_threshold:
        return _build_indicator_result(
            "RSI(14)",
            reading,
            "RSI 65-72",
            1,
            "RSI shows strong momentum, but the stock may be somewhat extended.",
        )
    return _build_indicator_result(
        "RSI(14)",
        reading,
        "RSI above 72",
        0,
        "RSI is above 72, suggesting the stock may be stretched and a new bullish entry may be late.",
    )


def _score_entry_macd(macd_frame: pd.DataFrame) -> dict:
    current_macd = macd_frame["macd_line"].iloc[-1] if len(macd_frame) >= 1 else np.nan
    current_signal = macd_frame["signal_line"].iloc[-1] if len(macd_frame) >= 1 else np.nan
    current_hist = macd_frame["histogram"].iloc[-1] if len(macd_frame) >= 1 else np.nan
    prior_macd = macd_frame["macd_line"].iloc[-2] if len(macd_frame) >= 2 else np.nan
    prior_signal = macd_frame["signal_line"].iloc[-2] if len(macd_frame) >= 2 else np.nan
    prior_hist = macd_frame["histogram"].iloc[-2] if len(macd_frame) >= 2 else np.nan

    if not all(_is_valid_number(value) for value in [current_macd, current_signal, current_hist, prior_hist]):
        return _insufficient_indicator("MACD Daily", "Insufficient data to calculate MACD.")

    current_macd = float(current_macd)
    current_signal = float(current_signal)
    current_hist = float(current_hist)
    prior_macd = float(prior_macd) if _is_valid_number(prior_macd) else np.nan
    prior_signal = float(prior_signal) if _is_valid_number(prior_signal) else np.nan
    prior_hist = float(prior_hist)

    histogram_improving = current_hist > prior_hist
    bullish_cross_starting = (
        _is_valid_number(prior_macd)
        and _is_valid_number(prior_signal)
        and current_macd > current_signal
        and prior_macd <= prior_signal
    )
    clearly_strengthening = current_macd > current_signal and histogram_improving
    bullish_confirmed = (current_macd > current_signal and current_macd > 0) or clearly_strengthening
    reading = (
        f"MACD {_format_value(current_macd)}, signal {_format_value(current_signal)}, "
        f"hist {_format_value(current_hist)}, prior hist {_format_value(prior_hist)}"
    )

    if bullish_confirmed:
        return _build_indicator_result(
            "MACD Daily",
            reading,
            "MACD bullish/strengthening",
            2,
            "MACD is bullish or clearly strengthening, suggesting improving upward momentum.",
        )
    if histogram_improving or bullish_cross_starting:
        return _build_indicator_result(
            "MACD Daily",
            reading,
            "Histogram improving or bullish cross starting",
            1,
            "MACD momentum is improving, but confirmation is not yet fully strong.",
        )
    return _build_indicator_result(
        "MACD Daily",
        reading,
        "Bearish / histogram falling",
        0,
        "MACD is bearish or the histogram is falling, suggesting momentum is weakening.",
    )


def _score_entry_ema_slope(ema_series: pd.Series) -> dict:
    current_ema = ema_series.iloc[-1] if len(ema_series) >= 1 else np.nan
    prior_ema = ema_series.iloc[-2] if len(ema_series) >= 2 else np.nan

    if not (_is_valid_number(current_ema) and _is_valid_number(prior_ema)):
        return _insufficient_indicator("20 EMA Slope", "Insufficient data to calculate the 20 EMA slope.")

    current_ema = float(current_ema)
    prior_ema = float(prior_ema)
    reading = f"EMA20 {_format_value(current_ema)}, prior {_format_value(prior_ema)}"

    if current_ema > prior_ema:
        return _build_indicator_result(
            "20 EMA Slope",
            reading,
            "20 EMA rising",
            1,
            "The 20 EMA is rising, suggesting the short-term daily trend is improving.",
        )
    return _build_indicator_result(
        "20 EMA Slope",
        reading,
        "20 EMA flat/down",
        0,
        "The 20 EMA is flat or declining, suggesting the short-term daily trend is not clearly rising.",
    )


def _score_entry_price_vs_ema(close_series: pd.Series, ema_series: pd.Series) -> dict:
    current_close = close_series.iloc[-1] if len(close_series) >= 1 else np.nan
    current_ema = ema_series.iloc[-1] if len(ema_series) >= 1 else np.nan

    if not (_is_valid_number(current_close) and _is_valid_number(current_ema)):
        return _insufficient_indicator("Price vs 20 EMA", "Insufficient data to compare price with the 20 EMA.")

    current_close = float(current_close)
    current_ema = float(current_ema)
    reading = f"Close {_format_value(current_close)}, EMA20 {_format_value(current_ema)}"

    if current_close > current_ema:
        return _build_indicator_result(
            "Price vs 20 EMA",
            reading,
            "Price above 20 EMA",
            1,
            "Price is above the 20 EMA, suggesting the stock is holding above short-term daily trend support.",
        )
    return _build_indicator_result(
        "Price vs 20 EMA",
        reading,
        "Price below 20 EMA",
        0,
        "Price is below the 20 EMA, suggesting the stock is not currently above short-term daily trend support.",
    )


def interpret_wills_signal(score: int) -> dict:
    """Map the bullish setup score to the requested interpretation bands."""
    if score >= 8:
        return {
            "label": "Strong Long Candidate",
            "message": (
                "Will\u2019s Signal is strong. The daily chart shows favorable bullish trend strength, "
                "directional pressure, momentum, and price structure. This may be a strong bullish "
                "continuation candidate, but entry timing and risk management should still be evaluated separately."
            ),
            "accent": "#16a34a",
        }
    if score >= 6:
        return {
            "label": "Tradable / Watchlist Candidate",
            "message": (
                "Will\u2019s Signal is moderate. The setup may be tradable, but the daily chart is not fully aligned. "
                "Consider waiting for cleaner confirmation, a better pullback, or stronger momentum before acting."
            ),
            "accent": "#ca8a04",
        }
    return {
        "label": "Weak / Usually Skip",
        "message": (
            "Will\u2019s Signal is weak. The daily chart does not currently show enough bullish technical alignment "
            "for a high-quality bullish continuation setup."
        ),
        "accent": "#dc2626",
    }


def interpret_wills_exhaustion_signal(score: int) -> dict:
    """Map the exhaustion score to the requested warning bands."""
    if score >= 8:
        return {
            "label": "Strong Sell / Trim Warning",
            "message": (
                "Will\u2019s Exhaustion Signal is high. Multiple daily indicators suggest exhaustion, weakening momentum, "
                "or trend deterioration. Consider whether trimming, selling, or protecting gains is appropriate based on the trading plan."
            ),
            "accent": "#dc2626",
        }
    if score >= 6:
        return {
            "label": "Move May Be Aging \u2014 Tighten Stops",
            "message": (
                "Will\u2019s Exhaustion Signal is elevated. The move may be aging or weakening. Consider tightening stops, "
                "reducing risk, or waiting for renewed confirmation before adding exposure."
            ),
            "accent": "#ea580c",
        }
    if score >= 4:
        return {
            "label": "Early Warning",
            "message": (
                "Will\u2019s Exhaustion Signal is moderate. Early signs of trend aging or momentum weakening may be appearing. "
                "Consider reviewing position size, stops, and whether new entries still offer favorable risk/reward."
            ),
            "accent": "#ca8a04",
        }
    return {
        "label": "Trend Still Healthy",
        "message": (
            "Will\u2019s Exhaustion Signal is low. The daily trend still appears generally healthy, with limited evidence of exhaustion. "
            "Continue monitoring momentum and risk levels."
        ),
        "accent": "#16a34a",
    }


def get_earnings_warning(ticker: str, as_of_date: date | None = None) -> dict:
    """Fetch an upcoming earnings date when Yahoo Finance exposes one."""
    try:
        calendar = yf.Ticker(ticker).calendar
    except Exception:
        return {
            "status": "unavailable",
            "message": "Earnings date not available from current data source.",
            "earnings_date": None,
        }

    earnings_date = None

    try:
        if isinstance(calendar, pd.DataFrame) and not calendar.empty:
            for candidate in ["Earnings Date", "Earnings Date Start", "Earnings Date End"]:
                if candidate in calendar.index:
                    raw_value = calendar.loc[candidate].iloc[0]
                    earnings_date = _normalize_calendar_date(raw_value)
                    if earnings_date is not None:
                        break
                if candidate in calendar.columns:
                    raw_value = calendar[candidate].iloc[0]
                    earnings_date = _normalize_calendar_date(raw_value)
                    if earnings_date is not None:
                        break
        elif isinstance(calendar, dict):
            for candidate in ["Earnings Date", "Earnings Date Start", "Earnings Date End"]:
                if candidate in calendar:
                    raw_value = calendar[candidate]
                    if isinstance(raw_value, (list, tuple)) and raw_value:
                        raw_value = raw_value[0]
                    earnings_date = _normalize_calendar_date(raw_value)
                    if earnings_date is not None:
                        break
    except Exception:
        earnings_date = None

    if earnings_date is None:
        return {
            "status": "unavailable",
            "message": "Earnings date not available from current data source.",
            "earnings_date": None,
        }

    if as_of_date is not None and earnings_date < as_of_date:
        return {
            "status": "not_upcoming",
            "message": "No upcoming earnings date found in available data.",
            "earnings_date": earnings_date.isoformat(),
        }

    return {
        "status": "upcoming",
        "message": (
            f"Upcoming earnings date: {earnings_date.isoformat()}. Consider avoiding new bullish entries or "
            "cash-secured puts immediately before earnings unless the strategy intentionally accepts earnings risk."
        ),
        "earnings_date": earnings_date.isoformat(),
    }


def get_atr_extension_warning(close_series: pd.Series, ema_series: pd.Series, atr_series: pd.Series, config: WillsSignalConfig) -> dict:
    """Build the requested non-scored ATR extension warning."""
    current_close = close_series.iloc[-1] if len(close_series) >= 1 else np.nan
    current_ema = ema_series.iloc[-1] if len(ema_series) >= 1 else np.nan
    current_atr = atr_series.iloc[-1] if len(atr_series) >= 1 else np.nan

    if not (_is_valid_number(current_close) and _is_valid_number(current_ema) and _is_valid_number(current_atr)):
        return {
            "status": "unavailable",
            "message": "ATR extension warning not available from current data.",
            "extension_atr": None,
        }

    current_close = float(current_close)
    current_ema = float(current_ema)
    current_atr = float(current_atr)

    if current_atr <= 0:
        return {
            "status": "unavailable",
            "message": "ATR extension warning not available from current data.",
            "extension_atr": None,
        }

    if current_close <= current_ema:
        return {
            "status": "below_ema",
            "message": "Price is below the 20 EMA, so bullish upside extension is not applicable.",
            "extension_atr": float((current_close - current_ema) / current_atr),
        }

    extension_atr = float((current_close - current_ema) / current_atr)

    if extension_atr < config.moderate_extension_atr:
        message = "Price is not significantly extended above the 20 EMA based on ATR."
        status = "not_extended"
    elif extension_atr <= config.high_extension_atr:
        message = "Price is moderately extended above the 20 EMA. Consider waiting for a pullback before entry."
        status = "moderately_extended"
    else:
        message = "Price is significantly extended above the 20 EMA. A new bullish entry may be chasing."
        status = "highly_extended"

    return {
        "status": status,
        "message": message,
        "extension_atr": extension_atr,
    }


def _find_local_maxima(values: pd.Series, window: int) -> list[int]:
    """Simple local-maximum detection used for the RSI divergence heuristic."""
    maxima = []
    if len(values) < (2 * window + 1):
        return maxima

    for idx in range(window, len(values) - window):
        slice_ = values.iloc[idx - window: idx + window + 1]
        center = values.iloc[idx]
        if not _is_valid_number(center):
            continue
        if center == slice_.max() and (slice_ == center).sum() == 1:
            maxima.append(idx)
    return maxima


def detect_bearish_rsi_divergence(close_series: pd.Series, rsi_series: pd.Series, config: WillsSignalConfig) -> dict:
    """
    Flag a simple bearish RSI divergence.

    Assumption: compare the last two local price highs within the recent lookback
    window and inspect RSI at those same dates.
    """
    lookback = min(config.rsi_divergence_lookback, len(close_series))
    if lookback < 8:
        return {
            "flag": False,
            "message": "Not enough data to evaluate RSI divergence.",
        }

    close_window = close_series.tail(lookback)
    rsi_window = rsi_series.reindex(close_window.index).tail(lookback)
    maxima = _find_local_maxima(close_window, window=config.rsi_divergence_swing_window)
    if len(maxima) < 2:
        return {
            "flag": False,
            "message": "Not enough price swing highs to evaluate RSI divergence.",
        }

    first_idx, second_idx = maxima[-2], maxima[-1]
    first_price = close_window.iloc[first_idx]
    second_price = close_window.iloc[second_idx]
    first_rsi = rsi_window.iloc[first_idx]
    second_rsi = rsi_window.iloc[second_idx]

    divergence = bool(
        _is_valid_number(first_price)
        and _is_valid_number(second_price)
        and _is_valid_number(first_rsi)
        and _is_valid_number(second_rsi)
        and second_price > first_price
        and second_rsi < first_rsi
    )

    return {
        "flag": divergence,
        "message": (
            f"Price swing highs {_format_value(first_price)} -> {_format_value(second_price)}, "
            f"RSI swing highs {_format_value(first_rsi)} -> {_format_value(second_rsi)}"
        ),
    }


def _score_exhaustion_adx(indicators: pd.DataFrame, config: WillsSignalConfig) -> dict:
    current_adx = indicators["adx"].iloc[-1] if len(indicators) >= 1 else np.nan
    prior_adx = indicators["adx"].iloc[-2] if len(indicators) >= 2 else np.nan

    if not (_is_valid_number(current_adx) and _is_valid_number(prior_adx)):
        return _insufficient_indicator("ADX Trend Aging", "Insufficient data to calculate ADX trend aging.")

    current_adx = float(current_adx)
    prior_adx = float(prior_adx)
    delta = current_adx - prior_adx
    reading = f"ADX {_format_value(current_adx)}, prior {_format_value(prior_adx)}, delta {_format_value(delta)}"

    strong_run = max(current_adx, prior_adx) >= config.adx_exhaustion_strong_threshold
    flattening = current_adx >= config.adx_exhaustion_very_high_threshold and abs(delta) <= config.adx_flat_delta
    falling_after_strong_run = strong_run and current_adx < prior_adx

    if falling_after_strong_run:
        return _build_indicator_result(
            "ADX Trend Aging",
            reading,
            "ADX falling after a strong run",
            2,
            "ADX is falling after a strong run, suggesting the trend may be losing strength.",
        )
    if flattening or (current_adx >= config.adx_exhaustion_extreme_threshold and delta <= config.adx_flat_delta):
        return _build_indicator_result(
            "ADX Trend Aging",
            reading,
            "ADX very high but flattening",
            1,
            "ADX is very high but flattening, suggesting the trend may be mature.",
        )
    return _build_indicator_result(
        "ADX Trend Aging",
        reading,
        "ADX rising and trend still strong",
        0,
        "ADX is still rising or trend strength remains healthy, suggesting the trend has not clearly aged yet.",
    )


def _score_exhaustion_di(indicators: pd.DataFrame) -> dict:
    current_plus = indicators["plus_di"].iloc[-1] if len(indicators) >= 1 else np.nan
    current_minus = indicators["minus_di"].iloc[-1] if len(indicators) >= 1 else np.nan
    prior_plus = indicators["plus_di"].iloc[-2] if len(indicators) >= 2 else np.nan
    prior_minus = indicators["minus_di"].iloc[-2] if len(indicators) >= 2 else np.nan

    if not all(_is_valid_number(value) for value in [current_plus, current_minus, prior_plus, prior_minus]):
        return _insufficient_indicator("+DI / -DI Weakening", "Insufficient data to calculate +DI and -DI weakening.")

    current_plus = float(current_plus)
    current_minus = float(current_minus)
    prior_plus = float(prior_plus)
    prior_minus = float(prior_minus)
    current_spread = current_plus - current_minus
    prior_spread = prior_plus - prior_minus
    reading = (
        f"+DI {_format_value(current_plus)}, -DI {_format_value(current_minus)}, "
        f"spread {_format_value(current_spread)}, prior spread {_format_value(prior_spread)}"
    )

    if current_plus < current_minus:
        return _build_indicator_result(
            "+DI / -DI Weakening",
            reading,
            "+DI crossed below -DI",
            2,
            "+DI has crossed below -DI, suggesting bearish directional pressure has overtaken bullish pressure.",
        )
    if current_spread < prior_spread:
        return _build_indicator_result(
            "+DI / -DI Weakening",
            reading,
            "+DI narrowing toward -DI",
            1,
            "The +DI / -DI spread is narrowing, suggesting bullish directional pressure may be weakening.",
        )
    return _build_indicator_result(
        "+DI / -DI Weakening",
        reading,
        "+DI well above -DI",
        0,
        "+DI remains comfortably above -DI, suggesting bullish directional pressure is still intact.",
    )


def _score_exhaustion_rsi(close_series: pd.Series, rsi_series: pd.Series, config: WillsSignalConfig) -> dict:
    current_rsi = rsi_series.iloc[-1] if len(rsi_series) >= 1 else np.nan
    prior_rsi = rsi_series.iloc[-2] if len(rsi_series) >= 2 else np.nan

    if not (_is_valid_number(current_rsi) and _is_valid_number(prior_rsi)):
        return _insufficient_indicator("RSI Exhaustion / Divergence", "Insufficient data to calculate RSI exhaustion.")

    divergence = detect_bearish_rsi_divergence(close_series, rsi_series, config)
    current_rsi = float(current_rsi)
    prior_rsi = float(prior_rsi)
    recently_overbought = bool(rsi_series.tail(5).max() >= config.rsi_overbought_threshold) if len(rsi_series) >= 1 else False
    rolled_over = recently_overbought and current_rsi < prior_rsi
    reading = (
        f"RSI {_format_value(current_rsi)}, prior {_format_value(prior_rsi)}, "
        f"bearish divergence {'Yes' if divergence['flag'] else 'No'}"
    )

    if divergence["flag"] or rolled_over:
        return _build_indicator_result(
            "RSI Exhaustion / Divergence",
            reading,
            "RSI rolled over from elevated level or bearish divergence",
            2,
            "RSI has rolled over from an elevated level or shows bearish divergence, suggesting possible momentum exhaustion.",
        )
    if current_rsi > config.rsi_overbought_threshold and current_rsi >= prior_rsi:
        return _build_indicator_result(
            "RSI Exhaustion / Divergence",
            reading,
            "RSI above 70 and still rising",
            1,
            "RSI is above 70 and still rising, suggesting strong momentum but some extension risk.",
        )
    return _build_indicator_result(
        "RSI Exhaustion / Divergence",
        reading,
        "RSI 50-65 and stable",
        0,
        "RSI remains in a healthy range, suggesting momentum is not showing clear exhaustion.",
    )


def _score_exhaustion_macd(macd_frame: pd.DataFrame, config: WillsSignalConfig) -> dict:
    current_macd = macd_frame["macd_line"].iloc[-1] if len(macd_frame) >= 1 else np.nan
    current_signal = macd_frame["signal_line"].iloc[-1] if len(macd_frame) >= 1 else np.nan
    current_hist = macd_frame["histogram"].iloc[-1] if len(macd_frame) >= 1 else np.nan
    prior_macd = macd_frame["macd_line"].iloc[-2] if len(macd_frame) >= 2 else np.nan
    prior_signal = macd_frame["signal_line"].iloc[-2] if len(macd_frame) >= 2 else np.nan
    prior_hist = macd_frame["histogram"].iloc[-2] if len(macd_frame) >= 2 else np.nan

    if not all(_is_valid_number(value) for value in [current_macd, current_signal, current_hist, prior_hist]):
        return _insufficient_indicator("MACD Weakening", "Insufficient data to calculate MACD weakening.")

    current_macd = float(current_macd)
    current_signal = float(current_signal)
    current_hist = float(current_hist)
    prior_macd = float(prior_macd) if _is_valid_number(prior_macd) else np.nan
    prior_signal = float(prior_signal) if _is_valid_number(prior_signal) else np.nan
    prior_hist = float(prior_hist)

    histogram_negative = current_hist < 0
    bearish_cross = (
        _is_valid_number(prior_macd)
        and _is_valid_number(prior_signal)
        and current_macd < current_signal
        and prior_macd >= prior_signal
    )
    hist_tail = macd_frame["histogram"].tail(config.macd_histogram_shrink_bars).dropna()
    histogram_shrinking = bool(
        len(hist_tail) >= config.macd_histogram_shrink_bars
        and (hist_tail > 0).all()
        and all(hist_tail.iloc[idx] < hist_tail.iloc[idx - 1] for idx in range(1, len(hist_tail)))
    )
    reading = (
        f"MACD {_format_value(current_macd)}, signal {_format_value(current_signal)}, "
        f"hist {_format_value(current_hist)}, prior hist {_format_value(prior_hist)}"
    )

    if bearish_cross or histogram_negative:
        return _build_indicator_result(
            "MACD Weakening",
            reading,
            "MACD bearish cross or histogram negative",
            2,
            "MACD has turned bearish or the histogram has flipped negative, suggesting momentum has weakened materially.",
        )
    if histogram_shrinking:
        return _build_indicator_result(
            "MACD Weakening",
            reading,
            "Histogram shrinking for 2-3 bars",
            1,
            "MACD histogram is shrinking, suggesting upward momentum may be slowing.",
        )
    return _build_indicator_result(
        "MACD Weakening",
        reading,
        "MACD above signal and histogram rising",
        0,
        "MACD remains constructive, suggesting momentum is still supporting the trend.",
    )


def _score_exhaustion_price(close_series: pd.Series, ema_series: pd.Series, atr_series: pd.Series, config: WillsSignalConfig) -> dict:
    current_close = close_series.iloc[-1] if len(close_series) >= 1 else np.nan
    current_ema = ema_series.iloc[-1] if len(ema_series) >= 1 else np.nan
    prior_ema = ema_series.iloc[-2] if len(ema_series) >= 2 else np.nan
    current_atr = atr_series.iloc[-1] if len(atr_series) >= 1 else np.nan

    if not (_is_valid_number(current_close) and _is_valid_number(current_ema) and _is_valid_number(prior_ema)):
        return _insufficient_indicator("Price vs 20 EMA / Extension", "Insufficient data to evaluate price extension.")

    current_close = float(current_close)
    current_ema = float(current_ema)
    prior_ema = float(prior_ema)
    extension_atr = None
    extension_pct = None
    if _is_valid_number(current_atr) and float(current_atr) > 0:
        extension_atr = float((current_close - current_ema) / float(current_atr))
    elif current_ema != 0:
        extension_pct = float(current_close / current_ema - 1)

    reading_parts = [
        f"Close {_format_value(current_close)}",
        f"EMA20 {_format_value(current_ema)}",
        f"prior EMA20 {_format_value(prior_ema)}",
        f"ATR14 {_format_value(float(current_atr) if _is_valid_number(current_atr) else None)}",
    ]
    if extension_atr is not None:
        reading_parts.append(f"extension {_format_value(extension_atr)} ATR")
    elif extension_pct is not None:
        reading_parts.append(f"extension {_format_value(extension_pct * 100)}%")
    reading = ", ".join(reading_parts)

    if current_close < current_ema:
        return _build_indicator_result(
            "Price vs 20 EMA / Extension",
            reading,
            "Price closed below 20 EMA",
            2,
            "Price has closed below the 20 EMA, suggesting the short-term daily trend structure may be breaking down.",
        )

    if extension_atr is not None and extension_atr > config.high_extension_atr:
        return _build_indicator_result(
            "Price vs 20 EMA / Extension",
            reading,
            "Price extended far above 20 EMA",
            1,
            "Price is extended far above the 20 EMA, suggesting a pullback or consolidation risk may be elevated.",
        )

    if extension_atr is None and extension_pct is not None and extension_pct > config.fallback_extension_pct:
        return _build_indicator_result(
            "Price vs 20 EMA / Extension",
            reading,
            "Price extended far above 20 EMA using percentage fallback",
            1,
            "Price is extended far above the 20 EMA, suggesting a pullback or consolidation risk may be elevated.",
        )

    return _build_indicator_result(
        "Price vs 20 EMA / Extension",
        reading,
        "Price above rising 20 EMA",
        0,
        "Price remains above a rising 20 EMA without a major extension warning, suggesting trend structure remains healthy.",
    )


def _build_signal_payload(
    title: str,
    rows: list[dict],
    interpretation: dict,
    daily_bars_used: int,
    as_of_date: str | None,
    is_daily_only: bool = True,
    warnings: dict | None = None,
) -> dict:
    return {
        "title": title,
        "score": int(sum(row["points"] for row in rows)),
        "max_score": 10,
        "interpretation": interpretation,
        "rows": rows,
        "component_scores": {row["category"]: row["points"] for row in rows},
        "warnings": warnings or {},
        "daily_bars_used": daily_bars_used,
        "as_of_date": as_of_date,
        "is_daily_only": is_daily_only,
    }


def calculate_wills_signal(
    daily_df: pd.DataFrame,
    ticker: str,
    config: WillsSignalConfig = DEFAULT_WILLS_SIGNAL_CONFIG,
    include_earnings_warning: bool = True,
) -> dict:
    """Calculate the original bullish daily setup score."""
    required_columns = {"High", "Low", "Close"}
    if daily_df.empty or not required_columns.issubset(daily_df.columns):
        rows = _empty_rows_for_signal()
        interpretation = interpret_wills_signal(0)
        return _build_signal_payload(
            title=WILLS_SIGNAL_TITLE,
            rows=rows,
            interpretation=interpretation,
            daily_bars_used=int(len(daily_df)),
            as_of_date=None,
            warnings={
                "earnings": {
                    "status": "unavailable",
                    "message": "Earnings date not available from current data source.",
                    "earnings_date": None,
                },
                "atr_extension": {
                    "status": "unavailable",
                    "message": "ATR extension warning not available from current data.",
                    "extension_atr": None,
                },
            },
        )

    context = _build_common_context(daily_df, config)
    close = context["close"]
    dmi_adx = context["dmi_adx"]
    rsi_series = context["rsi"]
    macd_frame = context["macd"]
    ema20 = context["ema20"]
    atr14 = context["atr14"]

    # Each category is scored independently so one missing indicator cannot crash the section.
    rows = [
        _score_entry_adx(dmi_adx, config)
        if len(daily_df) >= config.adx_period * 2
        else _insufficient_indicator("ADX(14)", "Insufficient data to calculate ADX(14)."),
        _score_entry_di(dmi_adx)
        if len(daily_df) >= config.adx_period * 2
        else _insufficient_indicator("+DI vs -DI", "Insufficient data to calculate +DI and -DI."),
        _score_entry_rsi(rsi_series, config)
        if len(daily_df) >= config.rsi_period + 1
        else _insufficient_indicator("RSI(14)", "Insufficient data to calculate RSI(14)."),
        _score_entry_macd(macd_frame)
        if len(daily_df) >= config.macd_slow + config.macd_signal
        else _insufficient_indicator("MACD Daily", "Insufficient data to calculate MACD."),
        _score_entry_ema_slope(ema20)
        if len(daily_df) >= config.ema_period + 1
        else _insufficient_indicator("20 EMA Slope", "Insufficient data to calculate the 20 EMA slope."),
        _score_entry_price_vs_ema(close, ema20)
        if len(daily_df) >= config.ema_period
        else _insufficient_indicator("Price vs 20 EMA", "Insufficient data to compare price with the 20 EMA."),
    ]

    score = int(sum(row["points"] for row in rows))
    interpretation = interpret_wills_signal(score)
    as_of_date = daily_df.index[-1].date() if len(daily_df) else None
    earnings_warning = {
        "status": "unavailable",
        "message": "Earnings date not available from current data source.",
        "earnings_date": None,
    }
    if include_earnings_warning:
        earnings_warning = get_earnings_warning(ticker, as_of_date=as_of_date)
    atr_extension_warning = get_atr_extension_warning(close, ema20, atr14, config)

    payload = _build_signal_payload(
        title=WILLS_SIGNAL_TITLE,
        rows=rows,
        interpretation=interpretation,
        daily_bars_used=int(len(daily_df)),
        as_of_date=as_of_date.isoformat() if as_of_date else None,
        warnings={
            "earnings": earnings_warning,
            "atr_extension": atr_extension_warning,
        },
    )
    payload["has_sufficient_history"] = len(daily_df) >= config.min_daily_bars
    return payload


def calculate_wills_exhaustion_signal(
    daily_df: pd.DataFrame,
    ticker: str,
    config: WillsSignalConfig = DEFAULT_WILLS_SIGNAL_CONFIG,
) -> dict:
    """Calculate the separate daily exhaustion / trim warning score."""
    required_columns = {"High", "Low", "Close"}
    if daily_df.empty or not required_columns.issubset(daily_df.columns):
        rows = _empty_rows_for_exhaustion()
        interpretation = interpret_wills_exhaustion_signal(0)
        payload = _build_signal_payload(
            title=WILLS_EXHAUSTION_TITLE,
            rows=rows,
            interpretation=interpretation,
            daily_bars_used=int(len(daily_df)),
            as_of_date=None,
        )
        payload["ticker"] = ticker
        return payload

    context = _build_common_context(daily_df, config)
    close = context["close"]
    dmi_adx = context["dmi_adx"]
    rsi_series = context["rsi"]
    macd_frame = context["macd"]
    ema20 = context["ema20"]
    atr14 = context["atr14"]

    rows = [
        _score_exhaustion_adx(dmi_adx, config)
        if len(daily_df) >= config.adx_period * 2
        else _insufficient_indicator("ADX Trend Aging", "Insufficient data to calculate ADX trend aging."),
        _score_exhaustion_di(dmi_adx)
        if len(daily_df) >= config.adx_period * 2
        else _insufficient_indicator("+DI / -DI Weakening", "Insufficient data to calculate +DI and -DI weakening."),
        _score_exhaustion_rsi(close, rsi_series, config)
        if len(daily_df) >= max(config.rsi_period + 1, config.rsi_divergence_lookback)
        else _insufficient_indicator("RSI Exhaustion / Divergence", "Insufficient data to calculate RSI exhaustion."),
        _score_exhaustion_macd(macd_frame, config)
        if len(daily_df) >= config.macd_slow + config.macd_signal
        else _insufficient_indicator("MACD Weakening", "Insufficient data to calculate MACD weakening."),
        _score_exhaustion_price(close, ema20, atr14, config)
        if len(daily_df) >= config.ema_period + 1
        else _insufficient_indicator("Price vs 20 EMA / Extension", "Insufficient data to evaluate price extension."),
    ]

    score = int(sum(row["points"] for row in rows))
    interpretation = interpret_wills_exhaustion_signal(score)
    as_of_date = daily_df.index[-1].date() if len(daily_df) else None
    payload = _build_signal_payload(
        title=WILLS_EXHAUSTION_TITLE,
        rows=rows,
        interpretation=interpretation,
        daily_bars_used=int(len(daily_df)),
        as_of_date=as_of_date.isoformat() if as_of_date else None,
    )
    payload["ticker"] = ticker
    payload["has_sufficient_history"] = len(daily_df) >= config.min_daily_bars
    return payload
