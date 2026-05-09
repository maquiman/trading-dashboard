"""
Explainable, regime-based signal engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from levels import build_level_snapshot, calculate_atr_series, get_nearest_levels


STRONG_LEVEL_MIN_HITS = 2
VOLATILITY_ALERT_PERCENTILE = 0.80
TRENDING_UP = "Trending Up"
TRENDING_DOWN = "Trending Down"
RANGING = "Ranging"
HIGH_VOLATILITY = "High Volatility"
NEUTRAL = "Neutral"


def annualization_factor_for_ticker(ticker: str) -> int:
    """Use stock or crypto annualization conventions."""
    return 365 if "-USD" in ticker.upper() else 252


def calculate_slope_series(series: pd.Series, lookback: int = 5) -> pd.Series:
    """Calculate slope as a lookback return."""
    return series / series.shift(lookback) - 1


def calculate_sma_slope(sma_series: pd.Series, lookback: int = 5) -> float:
    """Return the latest SMA slope."""
    slope_series = calculate_slope_series(sma_series, lookback=lookback)
    latest_value = slope_series.iloc[-1] if not slope_series.empty else np.nan
    return 0.0 if pd.isna(latest_value) else float(latest_value)


def calculate_ema_slope(ema_series: pd.Series, lookback: int = 5) -> float:
    """Return the latest EMA slope."""
    slope_series = calculate_slope_series(ema_series, lookback=lookback)
    latest_value = slope_series.iloc[-1] if not slope_series.empty else np.nan
    return 0.0 if pd.isna(latest_value) else float(latest_value)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate a simple rolling RSI."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    both_flat = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask(both_flat, 50)
    return rsi


def _expanding_percentile(series: pd.Series, percentile: float, min_periods: int = 20) -> pd.Series:
    """Calculate an expanding percentile threshold using only history available so far."""
    return series.expanding(min_periods=min_periods).apply(
        lambda values: float(np.nanquantile(values, percentile)) if np.isfinite(values).any() else np.nan,
        raw=True,
    )


def _percentile_rank_last(values: np.ndarray) -> float:
    """Return the percentile rank of the latest finite value in the array."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan

    latest_value = finite_values[-1]
    return float(np.mean(finite_values <= latest_value))


def _expanding_percentile_rank(series: pd.Series, min_periods: int = 20) -> pd.Series:
    """Calculate the expanding percentile rank of the latest value."""
    return series.expanding(min_periods=min_periods).apply(_percentile_rank_last, raw=True)


def build_feature_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Build the full rule-based feature frame."""
    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(0.0, index=df.index)
    ema20 = close.ewm(span=20, adjust=False).mean()
    sma50 = close.rolling(window=50, min_periods=50).mean()
    daily_return = close.pct_change()
    daily_vol = daily_return.rolling(window=20, min_periods=20).std()
    annual_factor = annualization_factor_for_ticker(ticker)
    annual_vol = daily_vol * np.sqrt(annual_factor)
    atr14 = calculate_atr_series(df, period=14)

    features = pd.DataFrame(index=df.index)
    features["close"] = close
    features["high"] = df["High"]
    features["low"] = df["Low"]
    features["volume"] = volume
    features["daily_return"] = daily_return
    features["5_period_return"] = close / close.shift(5) - 1
    features["20_period_return"] = close / close.shift(20) - 1
    features["ema20"] = ema20
    features["ema20_slope"] = calculate_slope_series(ema20, lookback=5)
    features["sma50"] = sma50
    features["sma50_slope"] = calculate_slope_series(sma50, lookback=5)
    features["daily_vol"] = daily_vol
    features["annual_vol"] = annual_vol
    features["annual_vol_75th"] = _expanding_percentile(annual_vol, percentile=0.75, min_periods=20)
    features["annual_vol_80th"] = _expanding_percentile(annual_vol, percentile=VOLATILITY_ALERT_PERCENTILE, min_periods=20)
    features["annual_vol_percentile"] = _expanding_percentile_rank(annual_vol, min_periods=20)
    features["volatility_threshold_percentile"] = VOLATILITY_ALERT_PERCENTILE
    features["atr14"] = atr14
    features["average_volume_20"] = volume.rolling(window=20, min_periods=20).mean()
    features["rsi14"] = calculate_rsi(close, period=14)
    features["rolling_high_20"] = close.rolling(window=20, min_periods=20).max()
    features["rolling_low_20"] = close.rolling(window=20, min_periods=20).min()
    features["close_is_20_high"] = close >= features["rolling_high_20"]
    features["close_is_20_low"] = close <= features["rolling_low_20"]
    features["high_volatility"] = features["annual_vol_percentile"] >= VOLATILITY_ALERT_PERCENTILE
    return features


def detect_regime(row: pd.Series) -> dict:
    """Detect the current market regime from a fully prepared row."""
    sma50_available = pd.notna(row.get("sma50"))

    is_trending_up = (
        pd.notna(row.get("close"))
        and pd.notna(row.get("ema20"))
        and row["close"] > row["ema20"]
        and pd.notna(row.get("ema20_slope"))
        and row["ema20_slope"] > 0.01
        and (not sma50_available or row["close"] > row["sma50"])
        and pd.notna(row.get("20_period_return"))
        and row["20_period_return"] > 0.05
    )

    is_trending_down = (
        pd.notna(row.get("close"))
        and pd.notna(row.get("ema20"))
        and row["close"] < row["ema20"]
        and pd.notna(row.get("ema20_slope"))
        and row["ema20_slope"] < -0.01
        and (not sma50_available or row["close"] < row["sma50"])
        and pd.notna(row.get("20_period_return"))
        and row["20_period_return"] < -0.05
    )

    high_volatility = bool(pd.notna(row.get("annual_vol_percentile")) and row.get("high_volatility", False))

    is_ranging = (
        not is_trending_up
        and not is_trending_down
        and pd.notna(row.get("20_period_return"))
        and -0.05 <= row["20_period_return"] <= 0.05
        and bool(row.get("price_between_levels", False))
    )

    if is_trending_up:
        regime = TRENDING_UP
    elif is_trending_down:
        regime = TRENDING_DOWN
    elif is_ranging:
        regime = RANGING
    elif high_volatility:
        regime = HIGH_VOLATILITY
    else:
        regime = NEUTRAL

    if high_volatility and regime not in {HIGH_VOLATILITY, NEUTRAL}:
        label = f"{regime} + {HIGH_VOLATILITY}"
    else:
        label = regime

    return {
        "regime": regime,
        "label": label,
        "high_volatility": high_volatility,
        "is_trending_up": is_trending_up,
        "is_trending_down": is_trending_down,
        "is_ranging": is_ranging,
    }


def _normalize_nearest_levels(nearest_levels: dict | None) -> dict:
    """Provide a stable nearest-level structure."""
    nearest_levels = nearest_levels or {}
    return {
        "support": nearest_levels.get("support"),
        "support_strength": nearest_levels.get("support_strength", 0),
        "resistance": nearest_levels.get("resistance"),
        "resistance_strength": nearest_levels.get("resistance_strength", 0),
        "distance_to_support_atr": nearest_levels.get("distance_to_support_atr"),
        "distance_to_resistance_atr": nearest_levels.get("distance_to_resistance_atr"),
        "distance_to_support_pct": nearest_levels.get("distance_to_support_pct"),
        "distance_to_resistance_pct": nearest_levels.get("distance_to_resistance_pct"),
        "price_between_levels": nearest_levels.get("price_between_levels", False),
    }


def _append_level_context(row: pd.Series, nearest_levels: dict) -> pd.Series:
    """Attach nearest level metadata to the working row."""
    context = row.copy()
    for key, value in _normalize_nearest_levels(nearest_levels).items():
        context[key] = value
    return context


def _classify_signal(score: float) -> dict:
    """Map score to both a trade trigger and a more descriptive bias label."""
    if score >= 3:
        return {"trade_signal": "BUY", "bias": "BUY"}
    if score <= -3:
        return {"trade_signal": "SELL", "bias": "SELL"}
    if score >= 1:
        return {"trade_signal": "HOLD", "bias": "WATCH BULLISH"}
    if score <= -1:
        return {"trade_signal": "HOLD", "bias": "WATCH BEARISH"}
    return {"trade_signal": "HOLD", "bias": "NEUTRAL"}


def _pick_closer_level(current_price: float, candidates: list[tuple[str, float | None]], prefer_below: bool | None = None):
    """Pick the level closest to current price, optionally restricting side."""
    filtered = []
    for label, level in candidates:
        if level is None or pd.isna(level):
            continue
        level_value = float(level)
        if prefer_below is True and level_value >= current_price:
            continue
        if prefer_below is False and level_value <= current_price:
            continue
        filtered.append((label, level_value))

    if not filtered:
        fallback = []
        for label, level in candidates:
            if level is None or pd.isna(level):
                continue
            fallback.append((label, float(level)))
        filtered = fallback

    if not filtered:
        return None, None

    label, level_value = min(filtered, key=lambda item: abs(current_price - item[1]))
    return label, level_value


def _format_level(level: float | None) -> str:
    """Format a price level for display."""
    if level is None or pd.isna(level):
        return "N/A"
    return f"${level:,.2f}"


def _build_trade_setup(row: pd.Series, regime_label: str, current_bias: str, nearest_levels: dict) -> dict:
    """Create a rule-based trade setup interpretation payload."""
    close = float(row["close"])
    ema20 = None if pd.isna(row.get("ema20")) else float(row["ema20"])
    atr = None if pd.isna(row.get("atr14")) else float(row["atr14"])
    support = nearest_levels.get("support")
    resistance = nearest_levels.get("resistance")
    recent_high = None if pd.isna(row.get("rolling_high_20")) else float(row["rolling_high_20"])
    recent_low = None if pd.isna(row.get("rolling_low_20")) else float(row["rolling_low_20"])

    setup_direction = "neutral"
    if current_bias in {"BUY", "WATCH BULLISH"}:
        setup_direction = "bullish"
    elif current_bias in {"SELL", "WATCH BEARISH"}:
        setup_direction = "bearish"

    confirmation_label = "None"
    confirmation_level = None
    invalidation_label = "None"
    invalidation_level = None

    if setup_direction == "bullish":
        confirmation_label, confirmation_level = _pick_closer_level(
            close,
            [("Nearest Resistance", resistance), ("Recent 20-Bar High", recent_high)],
            prefer_below=False,
        )
        invalidation_label, invalidation_level = _pick_closer_level(
            close,
            [("Nearest Support", support), ("EMA20", ema20)],
            prefer_below=True,
        )
    elif setup_direction == "bearish":
        breakdown_level = None
        if support is not None:
            breakdown_level = float(support - 0.5 * atr) if atr and atr > 0 else float(support)

        confirmation_label, confirmation_level = _pick_closer_level(
            close,
            [("Nearest Support Breakdown", breakdown_level), ("Recent 20-Bar Low", recent_low)],
            prefer_below=True,
        )
        invalidation_label, invalidation_level = _pick_closer_level(
            close,
            [("Nearest Resistance", resistance), ("EMA20", ema20)],
            prefer_below=False,
        )

    potential_upside_pct = None
    if resistance is not None and close != 0:
        potential_upside_pct = float(resistance / close - 1)

    downside_risk_pct = None
    if support is not None and support != 0:
        downside_risk_pct = float(close / support - 1)

    risk_reward_ratio = None
    if potential_upside_pct is not None and downside_risk_pct is not None and downside_risk_pct > 0:
        risk_reward_ratio = float(potential_upside_pct / downside_risk_pct)

    if setup_direction == "bullish":
        narrative = (
            f"Price bias is bullish within a {regime_label.lower()} regime. "
            f"Nearest support is {_format_level(support)} and confirmation sits at {_format_level(confirmation_level)}. "
        )
        if resistance is not None:
            narrative += f"A move toward resistance around {_format_level(resistance)} remains the upside objective. "
        else:
            narrative += "No overhead resistance is visible, so the setup remains in price discovery mode. "
        narrative += "Avoid aggressive BUY unless price confirms strength."
    elif setup_direction == "bearish":
        narrative = (
            f"Price bias is bearish within a {regime_label.lower()} regime. "
            f"Nearest resistance is {_format_level(resistance)} and confirmation sits at {_format_level(confirmation_level)}. "
        )
        if support is not None:
            narrative += f"A break toward support around {_format_level(support)} keeps downside pressure active. "
        else:
            narrative += "There is no clear nearby support, so downside targets are less defined. "
        narrative += "Avoid aggressive SELL unless price confirms weakness."
    else:
        narrative = (
            f"Price is in a {regime_label.lower()} regime with a neutral bias. "
            f"Support is {_format_level(support)} and resistance is {_format_level(resistance)}. "
            "Wait for clearer confirmation before taking an aggressive directional view."
        )

    return {
        "market_regime": regime_label,
        "current_bias": current_bias,
        "confirmation_level": confirmation_level,
        "confirmation_label": confirmation_label,
        "invalidation_level": invalidation_level,
        "invalidation_label": invalidation_label,
        "nearest_support": support,
        "nearest_resistance": resistance,
        "potential_upside_pct": potential_upside_pct,
        "downside_risk_pct": downside_risk_pct,
        "risk_reward_ratio": risk_reward_ratio,
        "narrative": narrative,
    }


def evaluate_signal_row(row: pd.Series, nearest_levels: dict | None = None) -> dict:
    """Evaluate one row of features with explicit nearest support and resistance."""
    row = _append_level_context(row, nearest_levels or {})
    regime_info = detect_regime(row)
    regime = regime_info["regime"]
    high_volatility = regime_info["high_volatility"]

    atr = row.get("atr14")
    atr_is_valid = pd.notna(atr) and atr and atr > 0
    avg_volume = row.get("average_volume_20")
    avg_volume_is_valid = pd.notna(avg_volume) and avg_volume > 0

    support_distance_atr = row.get("distance_to_support_atr")
    resistance_distance_atr = row.get("distance_to_resistance_atr")
    strong_support_near = bool(
        atr_is_valid
        and row.get("support") is not None
        and row.get("support_strength", 0) >= STRONG_LEVEL_MIN_HITS
        and support_distance_atr is not None
        and 0 <= support_distance_atr <= 0.75
    )
    strong_resistance_near = bool(
        atr_is_valid
        and row.get("resistance") is not None
        and row.get("resistance_strength", 0) >= STRONG_LEVEL_MIN_HITS
        and resistance_distance_atr is not None
        and 0 <= resistance_distance_atr <= 0.75
    )
    support_near = bool(
        atr_is_valid
        and row.get("support") is not None
        and support_distance_atr is not None
        and 0 <= support_distance_atr <= 0.75
    )
    resistance_near = bool(
        atr_is_valid
        and row.get("resistance") is not None
        and resistance_distance_atr is not None
        and 0 <= resistance_distance_atr <= 0.75
    )

    breakout_threshold_atr = 1.0 if regime == RANGING else 0.5
    breakdown_threshold_atr = 1.0 if regime == RANGING else 0.5

    breakout_active = bool(
        atr_is_valid
        and row.get("resistance") is not None
        and row["close"] > row["resistance"] + breakout_threshold_atr * atr
    )
    breakdown_active = bool(
        atr_is_valid
        and row.get("support") is not None
        and row["close"] < row["support"] - breakdown_threshold_atr * atr
    )
    volume_confirmed = bool(
        avg_volume_is_valid and pd.notna(row.get("volume")) and row["volume"] > 1.2 * avg_volume
    )

    score_components = {
        "trend_position": 0,
        "ema_slope": 0,
        "recent_return": 0,
        "range_location": 0,
        "rsi": 0,
        "high_20_low_20": 0,
        "volatility": 0,
        "support_resistance": 0,
        "breakout": 0,
        "breakdown": 0,
        "vcp": 0,
    }
    explanations = []

    if regime == TRENDING_UP:
        if pd.notna(row.get("ema20")) and row["close"] > row["ema20"]:
            score_components["trend_position"] += 2
        if pd.notna(row.get("ema20_slope")) and row["ema20_slope"] > 0:
            score_components["ema_slope"] += 1
        if pd.notna(row.get("5_period_return")) and row["5_period_return"] > 0:
            score_components["recent_return"] += 1
        if bool(row.get("close_is_20_high", False)):
            score_components["high_20_low_20"] += 2
        if high_volatility:
            score_components["volatility"] -= 1
        if strong_resistance_near:
            score_components["support_resistance"] -= 2

    elif regime == TRENDING_DOWN:
        if pd.notna(row.get("ema20")) and row["close"] < row["ema20"]:
            score_components["trend_position"] -= 2
        if pd.notna(row.get("ema20_slope")) and row["ema20_slope"] < 0:
            score_components["ema_slope"] -= 1
        if pd.notna(row.get("5_period_return")) and row["5_period_return"] < 0:
            score_components["recent_return"] -= 1
        if bool(row.get("close_is_20_low", False)):
            score_components["high_20_low_20"] -= 2
        if high_volatility:
            score_components["volatility"] += 1
        if strong_support_near:
            score_components["support_resistance"] += 2

    elif regime == RANGING:
        if support_near:
            score_components["range_location"] += 2
        if resistance_near:
            score_components["range_location"] -= 2
        if pd.notna(row.get("rsi14")) and row["rsi14"] < 35:
            score_components["rsi"] += 1
        if pd.notna(row.get("rsi14")) and row["rsi14"] > 65:
            score_components["rsi"] -= 1

    if breakout_active:
        if volume_confirmed:
            score_components["breakout"] += 2
        if pd.notna(row.get("ema20_slope")) and row["ema20_slope"] > 0:
            score_components["breakout"] += 1
        if bool(row.get("close_is_20_high", False)):
            score_components["breakout"] += 1

    if breakdown_active:
        if volume_confirmed:
            score_components["breakdown"] -= 2
        if pd.notna(row.get("ema20_slope")) and row["ema20_slope"] < 0:
            score_components["breakdown"] -= 1
        if bool(row.get("close_is_20_low", False)):
            score_components["breakdown"] -= 1

    raw_score = float(sum(score_components.values()))
    final_score = raw_score
    if high_volatility:
        final_score *= 0.7
        explanations.append("High volatility reduces signal reliability")

    rounded_score = round(final_score, 2)
    classified_signal = _classify_signal(rounded_score)
    trade_signal = classified_signal["trade_signal"]
    current_bias = classified_signal["bias"]

    signal_strength = min(95, 50 + abs(rounded_score) * 10)
    if high_volatility:
        signal_strength = min(signal_strength, 70)

    if regime == TRENDING_UP:
        explanations.insert(0, "Trending Up regime favors long trend-following setups")
    elif regime == TRENDING_DOWN:
        explanations.insert(0, "Trending Down regime favors defensive or short-biased setups")
    elif regime == RANGING:
        explanations.insert(0, "Ranging regime favors mean-reversion near support and resistance")
    elif regime == HIGH_VOLATILITY:
        explanations.insert(0, "High volatility regime is active without a strong directional trend")
    else:
        explanations.insert(0, "Neutral regime with mixed evidence")

    if score_components["breakout"] > 0:
        explanations.append("Resistance breakout conditions are active")
    if score_components["breakdown"] < 0:
        explanations.append("Support breakdown conditions are active")
    if score_components["support_resistance"] < 0:
        explanations.append("Price is close to strong resistance")
    if score_components["support_resistance"] > 0:
        explanations.append("Price is close to strong support")
    if score_components["range_location"] > 0:
        explanations.append("Price is trading close to support inside the range")
    if score_components["range_location"] < 0:
        explanations.append("Price is trading close to resistance inside the range")
    if score_components["rsi"] > 0:
        explanations.append("RSI is oversold for a range environment")
    if score_components["rsi"] < 0:
        explanations.append("RSI is overbought for a range environment")
    if score_components["high_20_low_20"] > 0:
        explanations.append("Price is printing a fresh 20-bar high")
    if score_components["high_20_low_20"] < 0:
        explanations.append("Price is printing a fresh 20-bar low")

    if score_components["breakout"] != 0 or score_components["breakdown"] != 0:
        signal_style = "Breakout"
    elif regime in {TRENDING_UP, TRENDING_DOWN}:
        signal_style = "Trend-Following"
    elif regime == RANGING and (score_components["range_location"] != 0 or score_components["rsi"] != 0):
        signal_style = "Mean-Reversion"
    else:
        signal_style = "Neutral"

    trade_setup = _build_trade_setup(row, regime_info["label"], current_bias, _normalize_nearest_levels(nearest_levels or {}))

    annual_vol = None if pd.isna(row.get("annual_vol")) else float(row["annual_vol"])
    annual_vol_75th = None if pd.isna(row.get("annual_vol_75th")) else float(row["annual_vol_75th"])
    annual_vol_80th = None if pd.isna(row.get("annual_vol_80th")) else float(row["annual_vol_80th"])
    annual_vol_percentile = None if pd.isna(row.get("annual_vol_percentile")) else float(row["annual_vol_percentile"])

    return {
        "signal": current_bias,
        "trade_signal": trade_signal,
        "current_bias": current_bias,
        "score": rounded_score,
        "raw_score": raw_score,
        "signal_strength": round(signal_strength, 1),
        "signal_strength_note": "Signal Strength is a rule-based score, not a calibrated probability.",
        "regime": regime,
        "regime_label": regime_info["label"],
        "high_volatility": high_volatility,
        "signal_style": signal_style,
        "score_components": score_components,
        "features": {
            "ema20": None if pd.isna(row.get("ema20")) else float(row["ema20"]),
            "ema20_slope": None if pd.isna(row.get("ema20_slope")) else float(row["ema20_slope"]),
            "sma50": None if pd.isna(row.get("sma50")) else float(row["sma50"]),
            "sma50_slope": None if pd.isna(row.get("sma50_slope")) else float(row["sma50_slope"]),
            "rsi14": None if pd.isna(row.get("rsi14")) else float(row["rsi14"]),
            "5_period_return": None if pd.isna(row.get("5_period_return")) else float(row["5_period_return"]),
            "20_period_return": None if pd.isna(row.get("20_period_return")) else float(row["20_period_return"]),
            "annual_vol": annual_vol,
            "annual_vol_75th": annual_vol_75th,
            "annual_vol_80th": annual_vol_80th,
            "annual_vol_percentile": annual_vol_percentile,
            "volatility_threshold_percentile": VOLATILITY_ALERT_PERCENTILE,
            "atr14": None if pd.isna(row.get("atr14")) else float(row["atr14"]),
            "average_volume_20": None if pd.isna(row.get("average_volume_20")) else float(row["average_volume_20"]),
            "rolling_high_20": None if pd.isna(row.get("rolling_high_20")) else float(row["rolling_high_20"]),
            "rolling_low_20": None if pd.isna(row.get("rolling_low_20")) else float(row["rolling_low_20"]),
        },
        "nearest_levels": _normalize_nearest_levels(nearest_levels or {}),
        "trade_setup": trade_setup,
        "explanations": explanations,
    }


def generate_signal(
    df: pd.DataFrame,
    ticker: str,
    support_levels=None,
    resistance_levels=None,
    max_levels: int = 4,
    swing_sensitivity: int = 5,
) -> dict:
    """Generate the latest signal for a ticker using the full explainable ruleset."""
    features = build_feature_frame(df, ticker)
    latest_row = features.iloc[-1]

    if support_levels is None or resistance_levels is None:
        snapshot = build_level_snapshot(
            df,
            current_price=float(df["Close"].iloc[-1]),
            atr=latest_row.get("atr14"),
            max_levels=max_levels,
            sensitivity=swing_sensitivity,
            include_dynamic_levels=True,
        )
        support_levels = snapshot["actionable_supports"]
        resistance_levels = snapshot["actionable_resistances"]

    nearest_levels = get_nearest_levels(
        current_price=float(df["Close"].iloc[-1]),
        support_levels=support_levels,
        resistance_levels=resistance_levels,
        atr=latest_row.get("atr14"),
    )

    result = evaluate_signal_row(latest_row, nearest_levels=nearest_levels)
    result["support_levels"] = support_levels
    result["resistance_levels"] = resistance_levels
    return result
