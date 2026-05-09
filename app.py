import streamlit as st
import yfinance as yf
import yfinance.cache as yf_cache
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta, timezone
from html import escape
from pathlib import Path
from time import perf_counter

from backtest import backtest_signals_no_lookahead
from help_text import DASHBOARD_GUIDE, TOOLTIPS
from levels import (
    build_level_snapshot,
    calculate_actionable_threshold_pct,
    calculate_atr,
    calculate_cluster_tolerance_pct,
    create_display_zone,
)
from signals import generate_signal
from vcp import apply_vcp_to_signal, create_vcp_evidence_chart, detect_vcp
from wills_signal import calculate_wills_exhaustion_signal, calculate_wills_signal


INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
DAILY_CACHE_TTL_SECONDS = 3600
INTRADAY_CACHE_TTL_SECONDS = 600
MAX_INTRADAY_LOOKBACK_DAYS = {
    "1m": 7,
    "2m": 30,
    "5m": 60,
    "15m": 60,
    "30m": 90,
    "60m": 180,
    "90m": 180,
    "1h": 365,
}
LARGE_DATE_RANGE_WARNING_DAYS = 3650
MAX_VCP_ANALYSIS_ROWS = 1500
MAX_BACKTEST_ROWS = 1800
MIN_RELIABLE_VCP_ROWS = 80


def configure_yfinance_cache() -> None:
    """Point yfinance caches to a writable directory inside the workspace."""
    cache_dir = Path(__file__).resolve().parent / ".yfinance-cache"
    cache_dir.mkdir(exist_ok=True)
    yf_cache.set_cache_location(str(cache_dir))
    yf.set_tz_cache_location(str(cache_dir))


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns returned by yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        first_level = df.columns.get_level_values(0)
        if {"Open", "High", "Low", "Close"}.issubset(first_level):
            df.columns = first_level
        else:
            try:
                df = df.xs(df.columns.levels[1][0], axis=1, level=1)
            except Exception:
                df = df.copy()
    return df


def get_series(df: pd.DataFrame, key: str) -> pd.Series:
    """Return a clean series for an OHLC or volume column."""
    if key not in df.columns:
        raise KeyError(key)
    series = df[key]
    if isinstance(series, pd.DataFrame):
        return series.iloc[:, 0]
    return series


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate a simple moving average."""
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate an exponential moving average."""
    return data.ewm(span=period, adjust=False).mean()


def calculate_max_drawdown(close: pd.Series) -> float:
    """Calculate maximum drawdown as a percentage."""
    cumulative_max = close.cummax()
    drawdowns = close / cumulative_max - 1
    return float(drawdowns.min() * 100)


def calculate_annualized_volatility(close: pd.Series, ticker: str) -> float:
    """Calculate the latest 20-bar annualized volatility."""
    daily_vol = close.pct_change().rolling(window=20, min_periods=20).std()
    annual_factor = 365 if "-USD" in ticker.upper() else 252
    latest_daily_vol = daily_vol.iloc[-1] if not daily_vol.empty else np.nan
    if pd.isna(latest_daily_vol):
        return 0.0
    return float(latest_daily_vol * np.sqrt(annual_factor) * 100)


def format_money(value: float | None) -> str:
    """Format a price value for display."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def format_percent(value: float | None, digits: int = 2) -> str:
    """Format a percentage in decimal form."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def format_percent_points(value: float | None, digits: int = 2) -> str:
    """Format a percentage that is already expressed in percentage points."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}%"


def format_level_metric(level: float | None, distance_pct: float | None, distance_atr: float | None):
    """Prepare display strings for support and resistance metrics."""
    if level is None:
        return "N/A", "N/A"

    delta_parts = []
    if distance_pct is not None:
        delta_parts.append(f"{distance_pct * 100:.1f}%")
    if distance_atr is not None:
        delta_parts.append(f"{distance_atr:.2f} ATR")
    return format_money(level), " | ".join(delta_parts) if delta_parts else "N/A"


def is_intraday_interval(interval: str) -> bool:
    """Return True when the selected interval is intraday."""
    return interval in INTRADAY_INTERVALS


def cache_ttl_seconds_for_interval(interval: str) -> int:
    """Pick a cache TTL based on the interval granularity."""
    return INTRADAY_CACHE_TTL_SECONDS if is_intraday_interval(interval) else DAILY_CACHE_TTL_SECONDS


def format_timestamp_label(iso_timestamp: str | None) -> str:
    """Format an ISO timestamp for display."""
    if not iso_timestamp:
        return "N/A"
    try:
        timestamp = datetime.fromisoformat(iso_timestamp)
        return timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_timestamp


def format_age_label(iso_timestamp: str | None) -> str:
    """Show how old a cached artifact is."""
    if not iso_timestamp:
        return "N/A"
    try:
        timestamp = datetime.fromisoformat(iso_timestamp)
        age = datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)
    except Exception:
        return "N/A"

    minutes = max(0, int(age.total_seconds() // 60))
    if minutes < 1:
        return "<1 min"
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    if remaining_minutes == 0:
        return f"{hours}h"
    return f"{hours}h {remaining_minutes}m"


def apply_interval_guardrails(start_date: date, end_date: date, interval: str) -> tuple[date, list[str]]:
    """Clamp extreme intraday ranges and emit user-facing warnings."""
    effective_start_date = start_date
    warnings = []
    selected_days = (end_date - start_date).days

    if is_intraday_interval(interval):
        max_days = MAX_INTRADAY_LOOKBACK_DAYS.get(interval, 30)
        if selected_days > max_days:
            effective_start_date = end_date - timedelta(days=max_days)
            warnings.append(
                f"Intraday range limited to the most recent {max_days} days for interval {interval} to keep calculations responsive."
            )
    elif selected_days > LARGE_DATE_RANGE_WARNING_DAYS:
        warnings.append(
            "Large date range selected. Support/resistance, backtest, and pattern scans may take longer than usual."
        )

    return effective_start_date, warnings


def build_level_table(levels: list[dict]) -> pd.DataFrame:
    """Convert level dictionaries into a display table."""
    rows = []
    for item in levels:
        rows.append({
            "Type": item["type"],
            "Level": format_money(item["level"]),
            "Source": item["source_label"],
            "Level Score": f"{item.get('level_score', 0.0):.2f}",
            "Touches": int(item.get("touches", item.get("hits", 0))),
            "Last Touch": str(item.get("most_recent_touch_date", "N/A")).split(" ")[0],
            "Distance %": format_percent_points(item.get("distance_pct")),
            "Distance ATR": f"{item['distance_atr']:.2f}" if item.get("distance_atr") is not None else "N/A",
            "Dynamic": "Yes" if item.get("dynamic_level") else "No",
        })
    return pd.DataFrame(rows)


def build_wills_signal_table(wills_signal_result: dict, first_column_label: str = "Indicator") -> pd.DataFrame:
    """Convert Will's Signal or Will's Exhaustion Signal rows into a display table."""
    rows = []
    for item in wills_signal_result.get("rows", []):
        rows.append(
            {
                first_column_label: item["category"],
                "Current Value": item["current_reading"],
                "Rule Triggered": item["rule_triggered"],
                "Points": item["points"],
                "Diagnostic": item["diagnostic"],
            }
        )
    return pd.DataFrame(rows)


def get_help_text(key: str | None) -> str | None:
    """Lookup helper text by key."""
    if not key:
        return None
    return TOOLTIPS.get(key)


def metric_with_help(container, label: str, value, delta=None, help_key: str | None = None) -> None:
    """Render a Streamlit metric with centralized help text."""
    kwargs = {}
    if delta is not None:
        kwargs["delta"] = delta
    help_text = get_help_text(help_key)
    if help_text:
        kwargs["help"] = help_text
    container.metric(label, value, **kwargs)


def render_emphasis_card(title: str, value: str, accent: str, help_key: str | None = None) -> None:
    """Render a wider summary card for long text values."""
    help_text = get_help_text(help_key)
    help_badge = ""
    if help_text:
        help_badge = (
            f'<span title="{escape(help_text)}" style="'
            "display:inline-flex;align-items:center;justify-content:center;"
            "width:1.15rem;height:1.15rem;border-radius:999px;border:1px solid rgba(226,232,240,0.55);"
            "font-size:0.75rem;font-weight:700;color:#e2e8f0;cursor:help;"
            '">?</span>'
        )

    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {accent};
            background: rgba(15, 23, 42, 0.78);
            padding: 0.9rem 1rem;
            border-radius: 0.75rem;
            min-height: 92px;
        ">
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:flex-start;
                font-size: 0.78rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #cbd5e1;
                margin-bottom: 0.45rem;
            ">
                <span>{escape(title)}</span>
                {help_badge}
            </div>
            <div style="
                font-size: 1.08rem;
                font-weight: 700;
                line-height: 1.35;
                color: #f8fafc;
                white-space: normal;
                word-break: break-word;
            ">
                {escape(str(value))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_callout_box(title: str, message: str, accent: str) -> None:
    """Render a highlighted interpretation block."""
    st.markdown(
        f"""
        <div style="
            border: 1px solid {accent};
            background: rgba(15, 23, 42, 0.78);
            border-left: 8px solid {accent};
            padding: 1rem 1.1rem;
            border-radius: 0.9rem;
            margin: 0.35rem 0 1rem 0;
        ">
            <div style="
                font-size: 1.0rem;
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 0.4rem;
            ">
                {title}
            </div>
            <div style="
                font-size: 0.96rem;
                line-height: 1.45;
                color: #e2e8f0;
            ">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_price_chart(
    df: pd.DataFrame,
    ticker: str,
    sma_period: int | None = None,
    ema_period: int | None = None,
    actionable_levels=None,
    historical_levels=None,
):
    """Create a candlestick chart with overlays, volume, and support/resistance zones."""
    close_series = get_series(df, "Close")
    open_series = get_series(df, "Open")
    high_series = get_series(df, "High")
    low_series = get_series(df, "Low")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.06,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="Price",
            increasing_line_color="lightgreen",
            decreasing_line_color="lightcoral",
        ),
        row=1,
        col=1,
    )

    if sma_period and sma_period > 0:
        sma = calculate_sma(close_series, sma_period)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sma,
                mode="lines",
                name=f"SMA {sma_period}",
                line=dict(color="orange", width=2),
            ),
            row=1,
            col=1,
        )

    if ema_period and ema_period > 0:
        ema = calculate_ema(close_series, ema_period)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ema,
                mode="lines",
                name=f"EMA {ema_period}",
                line=dict(color="royalblue", width=2),
            ),
            row=1,
            col=1,
        )

    if "Volume" in df.columns:
        volume_series = get_series(df, "Volume")
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=volume_series,
                name="Volume",
                marker_color="gray",
                opacity=0.4,
            ),
            row=2,
            col=1,
        )

    for level in actionable_levels or []:
        zone = create_display_zone(level, atr=0.0)
        fill_color = "rgba(34, 197, 94, 0.18)" if level["type"] == "Support" else "rgba(239, 68, 68, 0.16)"
        line_color = "green" if level["type"] == "Support" else "red"
        fig.add_hrect(
            y0=zone["lower"],
            y1=zone["upper"],
            line_width=0,
            fillcolor=fill_color,
            row=1,
            col=1,
        )
        fig.add_hline(
            y=zone["level"],
            line_dash="solid",
            line_color=line_color,
            line_width=2,
            annotation_text=f"{level['type']} {zone['level']:.2f}",
            row=1,
            col=1,
        )

    for level in historical_levels or []:
        zone = create_display_zone(level, atr=0.0)
        line_color = "rgba(34, 197, 94, 0.45)" if level["type"] == "Support" else "rgba(239, 68, 68, 0.45)"
        fig.add_hline(
            y=zone["level"],
            line_dash="dot",
            line_color=line_color,
            line_width=1,
            row=1,
            col=1,
        )

    fig.update_layout(
        title=f"{ticker} Price Chart",
        template="plotly_dark",
        height=760,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=30, l=40, r=40),
    )

    fig.update_xaxes(type="date", rangeslider_visible=False)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def load_data(ticker: str, start_date, end_date, interval: str) -> pd.DataFrame:
    """Download ticker data and return a normalized DataFrame."""
    configure_yfinance_cache()
    raw_df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    if raw_df.empty:
        return raw_df

    df = normalize_data(raw_df)
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def _load_data_payload(ticker: str, start_date, end_date, interval: str) -> dict:
    """Download data plus basic metadata used by the status panel."""
    configure_yfinance_cache()
    raw_df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )

    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    if raw_df.empty:
        return {
            "data": raw_df,
            "metadata": {
                "fetched_at_utc": fetched_at_utc,
                "raw_rows": 0,
                "clean_rows": 0,
                "completeness_pct": 0.0,
                "interval": interval,
                "cache_ttl_seconds": cache_ttl_seconds_for_interval(interval),
            },
        }

    normalized_df = normalize_data(raw_df)
    raw_rows = int(len(normalized_df))
    clean_df = normalized_df.dropna(subset=["Open", "High", "Low", "Close"])
    clean_rows = int(len(clean_df))
    completeness_pct = float((clean_rows / raw_rows) * 100) if raw_rows else 0.0

    return {
        "data": clean_df,
        "metadata": {
            "fetched_at_utc": fetched_at_utc,
            "raw_rows": raw_rows,
            "clean_rows": clean_rows,
            "completeness_pct": completeness_pct,
            "interval": interval,
            "cache_ttl_seconds": cache_ttl_seconds_for_interval(interval),
        },
    }


@st.cache_data(ttl=DAILY_CACHE_TTL_SECONDS, show_spinner=False)
def load_data_payload_eod_cached(ticker: str, start_date, end_date, interval: str) -> dict:
    """Cache end-of-day downloads for a moderate amount of time."""
    return _load_data_payload(ticker, start_date, end_date, interval)


@st.cache_data(ttl=INTRADAY_CACHE_TTL_SECONDS, show_spinner=False)
def load_data_payload_intraday_cached(ticker: str, start_date, end_date, interval: str) -> dict:
    """Cache intraday downloads for a shorter time because they stale faster."""
    return _load_data_payload(ticker, start_date, end_date, interval)


def load_data_with_metadata(ticker: str, start_date, end_date, interval: str) -> dict:
    """Route data downloads through the appropriate cache policy."""
    if is_intraday_interval(interval):
        return load_data_payload_intraday_cached(ticker, start_date, end_date, interval)
    return load_data_payload_eod_cached(ticker, start_date, end_date, interval)


def _build_market_metrics(df: pd.DataFrame, ticker: str) -> dict:
    """Calculate reusable market statistics from a prepared dataframe."""
    close_series = get_series(df, "Close")
    volume_series = get_series(df, "Volume") if "Volume" in df.columns else None
    latest_close = float(close_series.iloc[-1])
    start_close = float(close_series.iloc[0])
    end_close = float(close_series.iloc[-1])
    bars_used = int(len(close_series))
    period_return = ((end_close / start_close) - 1) * 100 if start_close else 0.0
    volatility = calculate_annualized_volatility(close_series, ticker)
    max_drawdown = calculate_max_drawdown(close_series)
    average_volume = float(volume_series.rolling(window=20, min_periods=1).mean().iloc[-1]) if volume_series is not None else 0.0
    atr = calculate_atr(df, period=14)
    cluster_tolerance_pct = calculate_cluster_tolerance_pct(atr, latest_close)
    actionable_threshold_pct = calculate_actionable_threshold_pct(atr, latest_close)

    return {
        "latest_close": latest_close,
        "start_close": start_close,
        "end_close": end_close,
        "bars_used": bars_used,
        "period_return": period_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "average_volume": average_volume,
        "atr": atr,
        "cluster_tolerance_pct": cluster_tolerance_pct,
        "actionable_threshold_pct": actionable_threshold_pct,
    }


@st.cache_data(ttl=DAILY_CACHE_TTL_SECONDS, show_spinner=False)
def build_market_metrics_eod_cached(df: pd.DataFrame, ticker: str) -> dict:
    return _build_market_metrics(df, ticker)


@st.cache_data(ttl=INTRADAY_CACHE_TTL_SECONDS, show_spinner=False)
def build_market_metrics_intraday_cached(df: pd.DataFrame, ticker: str) -> dict:
    return _build_market_metrics(df, ticker)


def build_market_metrics_cached(df: pd.DataFrame, ticker: str, interval: str) -> dict:
    if is_intraday_interval(interval):
        return build_market_metrics_intraday_cached(df, ticker)
    return build_market_metrics_eod_cached(df, ticker)


def _build_level_snapshot_cached_impl(
    df: pd.DataFrame,
    current_price: float,
    atr: float,
    max_levels: int,
    sensitivity: int,
    include_dynamic_levels: bool,
) -> dict:
    """Compute the full level snapshot once so UI toggles can reuse it."""
    return build_level_snapshot(
        df,
        current_price=current_price,
        atr=atr,
        max_levels=max_levels,
        show_all_historical=True,
        sensitivity=sensitivity,
        include_dynamic_levels=include_dynamic_levels,
    )


@st.cache_data(ttl=DAILY_CACHE_TTL_SECONDS, show_spinner=False)
def build_level_snapshot_eod_cached(
    df: pd.DataFrame,
    current_price: float,
    atr: float,
    max_levels: int,
    sensitivity: int,
    include_dynamic_levels: bool,
) -> dict:
    return _build_level_snapshot_cached_impl(df, current_price, atr, max_levels, sensitivity, include_dynamic_levels)


@st.cache_data(ttl=INTRADAY_CACHE_TTL_SECONDS, show_spinner=False)
def build_level_snapshot_intraday_cached(
    df: pd.DataFrame,
    current_price: float,
    atr: float,
    max_levels: int,
    sensitivity: int,
    include_dynamic_levels: bool,
) -> dict:
    return _build_level_snapshot_cached_impl(df, current_price, atr, max_levels, sensitivity, include_dynamic_levels)


def build_level_snapshot_cached(
    df: pd.DataFrame,
    current_price: float,
    atr: float,
    max_levels: int,
    sensitivity: int,
    include_dynamic_levels: bool,
    interval: str,
) -> dict:
    if is_intraday_interval(interval):
        return build_level_snapshot_intraday_cached(df, current_price, atr, max_levels, sensitivity, include_dynamic_levels)
    return build_level_snapshot_eod_cached(df, current_price, atr, max_levels, sensitivity, include_dynamic_levels)


def _generate_signal_cached_impl(
    df: pd.DataFrame,
    ticker: str,
    support_levels: list[dict],
    resistance_levels: list[dict],
    max_levels: int,
    swing_sensitivity: int,
) -> dict:
    return generate_signal(
        df=df,
        ticker=ticker,
        support_levels=support_levels,
        resistance_levels=resistance_levels,
        max_levels=max_levels,
        swing_sensitivity=swing_sensitivity,
    )


@st.cache_data(ttl=DAILY_CACHE_TTL_SECONDS, show_spinner=False)
def generate_signal_eod_cached(
    df: pd.DataFrame,
    ticker: str,
    support_levels: list[dict],
    resistance_levels: list[dict],
    max_levels: int,
    swing_sensitivity: int,
) -> dict:
    return _generate_signal_cached_impl(df, ticker, support_levels, resistance_levels, max_levels, swing_sensitivity)


@st.cache_data(ttl=INTRADAY_CACHE_TTL_SECONDS, show_spinner=False)
def generate_signal_intraday_cached(
    df: pd.DataFrame,
    ticker: str,
    support_levels: list[dict],
    resistance_levels: list[dict],
    max_levels: int,
    swing_sensitivity: int,
) -> dict:
    return _generate_signal_cached_impl(df, ticker, support_levels, resistance_levels, max_levels, swing_sensitivity)


def generate_signal_cached(
    df: pd.DataFrame,
    ticker: str,
    support_levels: list[dict],
    resistance_levels: list[dict],
    max_levels: int,
    swing_sensitivity: int,
    interval: str,
) -> dict:
    if is_intraday_interval(interval):
        return generate_signal_intraday_cached(df, ticker, support_levels, resistance_levels, max_levels, swing_sensitivity)
    return generate_signal_eod_cached(df, ticker, support_levels, resistance_levels, max_levels, swing_sensitivity)


@st.cache_data(ttl=DAILY_CACHE_TTL_SECONDS, show_spinner=False)
def calculate_daily_signal_bundle_cached(daily_df: pd.DataFrame, ticker: str) -> dict:
    """Cache the two daily-only Will's signal modules together."""
    return {
        "wills_signal": calculate_wills_signal(daily_df, ticker=ticker),
        "wills_exhaustion_signal": calculate_wills_exhaustion_signal(daily_df, ticker=ticker),
    }


def _detect_vcp_cached_impl(
    df: pd.DataFrame,
    ticker: str,
    swing_sensitivity: int,
    min_base_length: int,
    max_base_length: int,
    actionable_resistances: list[dict],
) -> dict:
    return detect_vcp(
        df=df,
        ticker=ticker,
        swing_sensitivity=swing_sensitivity,
        min_base_length=min_base_length,
        max_base_length=max_base_length,
        actionable_resistances=actionable_resistances,
    )


@st.cache_data(ttl=DAILY_CACHE_TTL_SECONDS, show_spinner=False)
def detect_vcp_eod_cached(
    df: pd.DataFrame,
    ticker: str,
    swing_sensitivity: int,
    min_base_length: int,
    max_base_length: int,
    actionable_resistances: list[dict],
) -> dict:
    return _detect_vcp_cached_impl(df, ticker, swing_sensitivity, min_base_length, max_base_length, actionable_resistances)


@st.cache_data(ttl=INTRADAY_CACHE_TTL_SECONDS, show_spinner=False)
def detect_vcp_intraday_cached(
    df: pd.DataFrame,
    ticker: str,
    swing_sensitivity: int,
    min_base_length: int,
    max_base_length: int,
    actionable_resistances: list[dict],
) -> dict:
    return _detect_vcp_cached_impl(df, ticker, swing_sensitivity, min_base_length, max_base_length, actionable_resistances)


def detect_vcp_cached(
    df: pd.DataFrame,
    ticker: str,
    swing_sensitivity: int,
    min_base_length: int,
    max_base_length: int,
    actionable_resistances: list[dict],
    interval: str,
) -> dict:
    if is_intraday_interval(interval):
        return detect_vcp_intraday_cached(df, ticker, swing_sensitivity, min_base_length, max_base_length, actionable_resistances)
    return detect_vcp_eod_cached(df, ticker, swing_sensitivity, min_base_length, max_base_length, actionable_resistances)


def _backtest_cached_impl(
    df: pd.DataFrame,
    ticker: str,
    max_levels: int,
    swing_sensitivity: int,
    enable_vcp_detection: bool,
    vcp_min_base_length: int,
    vcp_max_base_length: int,
) -> dict:
    return backtest_signals_no_lookahead(
        df=df,
        ticker=ticker,
        max_levels=max_levels,
        lookback_periods=(5, 10, 20),
        swing_sensitivity=swing_sensitivity,
        enable_vcp_detection=enable_vcp_detection,
        vcp_min_base_length=vcp_min_base_length,
        vcp_max_base_length=vcp_max_base_length,
    )


@st.cache_data(ttl=DAILY_CACHE_TTL_SECONDS, show_spinner=False)
def backtest_eod_cached(
    df: pd.DataFrame,
    ticker: str,
    max_levels: int,
    swing_sensitivity: int,
    enable_vcp_detection: bool,
    vcp_min_base_length: int,
    vcp_max_base_length: int,
) -> dict:
    return _backtest_cached_impl(df, ticker, max_levels, swing_sensitivity, enable_vcp_detection, vcp_min_base_length, vcp_max_base_length)


@st.cache_data(ttl=INTRADAY_CACHE_TTL_SECONDS, show_spinner=False)
def backtest_intraday_cached(
    df: pd.DataFrame,
    ticker: str,
    max_levels: int,
    swing_sensitivity: int,
    enable_vcp_detection: bool,
    vcp_min_base_length: int,
    vcp_max_base_length: int,
) -> dict:
    return _backtest_cached_impl(df, ticker, max_levels, swing_sensitivity, enable_vcp_detection, vcp_min_base_length, vcp_max_base_length)


def backtest_cached(
    df: pd.DataFrame,
    ticker: str,
    max_levels: int,
    swing_sensitivity: int,
    enable_vcp_detection: bool,
    vcp_min_base_length: int,
    vcp_max_base_length: int,
    interval: str,
) -> dict:
    if is_intraday_interval(interval):
        return backtest_intraday_cached(df, ticker, max_levels, swing_sensitivity, enable_vcp_detection, vcp_min_base_length, vcp_max_base_length)
    return backtest_eod_cached(df, ticker, max_levels, swing_sensitivity, enable_vcp_detection, vcp_min_base_length, vcp_max_base_length)


@st.cache_data(ttl=DAILY_CACHE_TTL_SECONDS, show_spinner=False)
def get_latest_available_market_date_cached(probe_ticker: str = "SPY") -> date:
    """Use a short period query to find a sensible default market date."""
    configure_yfinance_cache()
    try:
        probe_df = yf.download(
            probe_ticker,
            period="10d",
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        if not probe_df.empty:
            probe_df = normalize_data(probe_df)
            return pd.Timestamp(probe_df.index[-1]).date()
    except Exception:
        pass
    return (datetime.now() - timedelta(days=7)).date()


def get_latest_available_market_date(probe_ticker: str = "SPY") -> date:
    return get_latest_available_market_date_cached(probe_ticker)


def main():
    st.set_page_config(page_title="Trading Dashboard v5 - VCP and Adaptive Levels", layout="wide")
    st.title("Trading Dashboard v5: VCP and Adaptive Levels")
    st.markdown("Rule-based and explainable market scoring with ATR-based support/resistance zones and a detailed VCP detector.")
    with st.expander("Dashboard Guide"):
        for title, body in DASHBOARD_GUIDE:
            st.write(f"**{title}:** {body}")

    default_end_date = get_latest_available_market_date()
    default_start_date = default_end_date - timedelta(days=365)

    st.sidebar.header("Market Selection")
    ticker = st.sidebar.text_input(
        "Ticker symbol",
        value="AMD",
        help=get_help_text("ticker_symbol"),
    ).upper()
    start_date = st.sidebar.date_input("Start date", value=default_start_date, help=get_help_text("start_date"))
    end_date = st.sidebar.date_input("End date", value=default_end_date, help=get_help_text("end_date"))
    interval = st.sidebar.selectbox(
        "Interval",
        options=["1d", "1wk", "1mo"],
        index=0,
        help=get_help_text("interval"),
    )

    st.sidebar.divider()
    st.sidebar.header("Chart Overlays")
    sma_enabled = st.sidebar.checkbox("Show SMA", value=True, help=get_help_text("show_sma"))
    sma_period = st.sidebar.slider("SMA period", min_value=5, max_value=200, value=50, disabled=not sma_enabled, help=get_help_text("sma_period")) if sma_enabled else 50
    ema_enabled = st.sidebar.checkbox("Show EMA", value=True, help=get_help_text("show_ema"))
    ema_period = st.sidebar.slider("EMA period", min_value=5, max_value=200, value=20, disabled=not ema_enabled, help=get_help_text("ema_period")) if ema_enabled else 20

    st.sidebar.divider()
    st.sidebar.header("Support / Resistance")
    show_zones = st.sidebar.checkbox("Show actionable zones", value=True, help=get_help_text("show_actionable_zones"))
    show_historical_levels = st.sidebar.checkbox("Show Historical Levels", value=False, help=get_help_text("show_historical_levels"))
    zone_count = st.sidebar.slider("Max actionable levels", min_value=1, max_value=8, value=4, help=get_help_text("max_actionable_levels"))
    swing_sensitivity = st.sidebar.slider("Swing sensitivity (N)", min_value=2, max_value=15, value=5, help=get_help_text("swing_sensitivity"))
    include_dynamic_levels = st.sidebar.checkbox("Include dynamic EMA20 / SMA50 levels", value=True, help=get_help_text("include_dynamic_levels"))

    st.sidebar.divider()
    st.sidebar.header("Signal Engine")
    use_transparent_signals = st.sidebar.checkbox("Use regime-based rule engine", value=True, help=get_help_text("use_regime_engine"))
    show_backtest = st.sidebar.checkbox("Show historical backtest", value=True, help=get_help_text("show_backtest"))

    st.sidebar.divider()
    st.sidebar.header("VCP Detector")
    enable_vcp_detection = st.sidebar.checkbox("Enable VCP Detection", value=True, help=get_help_text("enable_vcp_detection"))
    show_vcp_annotations = st.sidebar.checkbox("Show VCP Annotations", value=True, help=get_help_text("show_vcp_annotations"))
    show_vcp_diagnostics = st.sidebar.checkbox("Show VCP diagnostics even when no VCP is detected", value=False, help=get_help_text("show_vcp_diagnostics"))
    vcp_min_base_length = st.sidebar.slider("VCP base min length", min_value=20, max_value=80, value=20, help=get_help_text("vcp_base_min_length"))
    vcp_max_base_length = st.sidebar.slider("VCP base max length", min_value=40, max_value=180, value=120, help=get_help_text("vcp_base_max_length"))

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    effective_start_date, selection_warnings = apply_interval_guardrails(start_date, end_date, interval)
    performance_timings = []

    def record_timing(step: str, started_at: float) -> None:
        performance_timings.append({"Step": step, "Seconds": perf_counter() - started_at})

    try:
        status_warnings = list(selection_warnings)

        with st.spinner(f"Loading data for {ticker}..."):
            started_at = perf_counter()
            data_payload = load_data_with_metadata(ticker, effective_start_date, end_date, interval)
            record_timing("Data download / cache lookup", started_at)

        df = data_payload["data"]
        data_metadata = data_payload["metadata"]

        if df.empty:
            st.error(f"No data found for {ticker}. Please verify the ticker or date range.")
            return

        if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
            st.error("Downloaded data is missing required OHLC columns.")
            return

        started_at = perf_counter()
        market_metrics = build_market_metrics_cached(df, ticker, interval)
        record_timing("Core indicator statistics", started_at)

        latest_close = market_metrics["latest_close"]
        start_close = market_metrics["start_close"]
        end_close = market_metrics["end_close"]
        bars_used = market_metrics["bars_used"]
        period_return = market_metrics["period_return"]
        volatility = market_metrics["volatility"]
        max_drawdown = market_metrics["max_drawdown"]
        average_volume = market_metrics["average_volume"]
        atr = market_metrics["atr"]
        cluster_tolerance_pct = market_metrics["cluster_tolerance_pct"]
        actionable_threshold_pct = market_metrics["actionable_threshold_pct"]

        daily_end = pd.Timestamp(end_date).to_pydatetime()
        daily_start = min(pd.Timestamp(effective_start_date).to_pydatetime(), daily_end - timedelta(days=450))
        started_at = perf_counter()
        daily_payload = load_data_with_metadata(ticker, daily_start, daily_end, "1d")
        record_timing("Daily helper data download / cache lookup", started_at)
        daily_df = daily_payload["data"]
        daily_metadata = daily_payload["metadata"]

        started_at = perf_counter()
        daily_signal_bundle = calculate_daily_signal_bundle_cached(daily_df, ticker)
        record_timing("Will's daily signals", started_at)
        wills_signal_result = daily_signal_bundle["wills_signal"]
        wills_exhaustion_result = daily_signal_bundle["wills_exhaustion_signal"]

        started_at = perf_counter()
        level_snapshot = build_level_snapshot_cached(
            df,
            current_price=latest_close,
            atr=atr,
            max_levels=zone_count,
            sensitivity=swing_sensitivity,
            include_dynamic_levels=include_dynamic_levels,
            interval=interval,
        )
        record_timing("Support / resistance detection", started_at)
        support_levels = level_snapshot["actionable_supports"]
        resistance_levels = level_snapshot["actionable_resistances"]
        historical_supports = level_snapshot["historical_supports"] if show_historical_levels else []
        historical_resistances = level_snapshot["historical_resistances"] if show_historical_levels else []
        historical_levels = historical_supports + historical_resistances if show_historical_levels else []

        started_at = perf_counter()
        signal_result = generate_signal_cached(
            df,
            ticker,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            max_levels=zone_count,
            swing_sensitivity=swing_sensitivity,
            interval=interval,
        )
        record_timing("Signal scoring", started_at)

        vcp_result = None
        vcp_skip_reason = None
        if enable_vcp_detection:
            reliable_vcp_rows = max(MIN_RELIABLE_VCP_ROWS, vcp_min_base_length)
            if len(df) < vcp_min_base_length:
                vcp_skip_reason = f"VCP detection skipped because only {len(df)} rows are available and at least {vcp_min_base_length} rows are required."
            elif len(df) > MAX_VCP_ANALYSIS_ROWS:
                vcp_skip_reason = (
                    f"VCP detection skipped for performance safety because the dataset has {len(df)} rows, "
                    f"which exceeds the {MAX_VCP_ANALYSIS_ROWS}-row analysis guardrail."
                )
            else:
                if len(df) < reliable_vcp_rows:
                    status_warnings.append(
                        f"Dataset has only {len(df)} rows. VCP detection can run, but results are less reliable below about {reliable_vcp_rows} rows."
                    )
                started_at = perf_counter()
                vcp_result = detect_vcp_cached(
                    df=df,
                    ticker=ticker,
                    swing_sensitivity=swing_sensitivity,
                    min_base_length=vcp_min_base_length,
                    max_base_length=vcp_max_base_length,
                    actionable_resistances=resistance_levels,
                    interval=interval,
                )
                record_timing("VCP detection", started_at)
                signal_result = apply_vcp_to_signal(signal_result, vcp_result)

            if vcp_skip_reason:
                status_warnings.append(vcp_skip_reason)

        volatility_percentile = signal_result["features"]["annual_vol_percentile"]
        volatility_threshold_percentile = signal_result["features"]["volatility_threshold_percentile"]

        if interval != "1d":
            st.info("Current and visual analysis uses the selected visible data range. Daily data is preferred for the signal engine, VCP detector, and historical tests.")

        if signal_result["high_volatility"]:
            st.warning(
                f"High-volatility regime active. Current 20-day annualized volatility is {volatility:.2f}% "
                f"at the {volatility_percentile * 100:.0f}th percentile for the selected period."
            )

        if level_snapshot["price_discovery_mode"]:
            st.info("No overhead actionable resistance detected - price discovery mode.")

        if level_snapshot["price_extended"]:
            st.warning("Price may be extended above short-term trend.")

        if data_metadata["completeness_pct"] < 100:
            status_warnings.append(
                f"Only {data_metadata['completeness_pct']:.1f}% of downloaded rows were complete enough to use after OHLC cleaning."
            )

        st.divider()
        st.subheader("Data / Calculation Status")
        status_cols = st.columns(5)
        metric_with_help(status_cols[0], "Last Data Refresh", format_timestamp_label(data_metadata.get("fetched_at_utc")), help_key="last_data_refresh")
        metric_with_help(status_cols[1], "Rows Downloaded", str(data_metadata.get("clean_rows", len(df))), help_key="rows_downloaded")
        metric_with_help(status_cols[2], "Interval Used", interval, help_key="interval")
        metric_with_help(status_cols[3], "Data Completeness", f"{data_metadata.get('completeness_pct', 0.0):.1f}%", help_key="data_completeness")
        metric_with_help(status_cols[4], "Cache TTL", f"{int(data_metadata.get('cache_ttl_seconds', DAILY_CACHE_TTL_SECONDS) / 60)} min", help_key="cache_ttl")
        st.caption(
            "Cache status: active. Streamlit does not expose exact cache hit/miss per run, "
            f"but the current data artifact is {format_age_label(data_metadata.get('fetched_at_utc'))} old."
        )
        st.caption(
            f"Daily helper dataset for Will's Signal sections: {daily_metadata.get('clean_rows', len(daily_df))} rows, "
            f"last refreshed {format_timestamp_label(daily_metadata.get('fetched_at_utc'))}, "
            f"cache age {format_age_label(daily_metadata.get('fetched_at_utc'))}."
        )

        for warning_message in status_warnings:
            st.warning(warning_message)

        if use_transparent_signals:
            st.divider()
            st.subheader("Signal Summary")

            headline_cols = st.columns([2.15, 2.1, 2.1, 1.2, 1.2, 1.7])
            with headline_cols[0]:
                render_emphasis_card("Regime", signal_result["regime_label"], "#2563eb", help_key="regime")
            with headline_cols[1]:
                render_emphasis_card("Current Bias", signal_result["current_bias"], "#059669", help_key="current_bias")
            with headline_cols[2]:
                render_emphasis_card("Signal Type", signal_result["signal_style"], "#7c3aed", help_key="signal_type")
            headline_cols[3].metric("Trade Trigger", signal_result["trade_signal"], help=get_help_text("trade_trigger"))
            headline_cols[4].metric("Score", f"{signal_result['score']:+.2f}", help=get_help_text("score"))
            with headline_cols[5]:
                render_emphasis_card("Setup Type", signal_result.get("setup_type", "Standard"), "#0f766e", help_key="setup_type")

            st.caption(signal_result["signal_strength_note"])

            stat_cols = st.columns(4)
            metric_with_help(stat_cols[0], "Signal Strength", f"{signal_result['signal_strength']:.1f}%", help_key="signal_strength")
            metric_with_help(stat_cols[1], "Annual Volatility", f"{volatility:.2f}%", help_key="annual_volatility")
            metric_with_help(stat_cols[2], "Volatility Percentile", f"{volatility_percentile * 100:.0f}%" if volatility_percentile is not None else "N/A", help_key="volatility_percentile")
            metric_with_help(stat_cols[3], "High-Vol Threshold", f"{volatility_threshold_percentile * 100:.0f}%", help_key="high_vol_threshold")

            support_value, support_delta = format_level_metric(
                signal_result["nearest_levels"]["support"],
                signal_result["nearest_levels"]["distance_to_support_pct"],
                signal_result["nearest_levels"]["distance_to_support_atr"],
            )
            resistance_value, resistance_delta = format_level_metric(
                signal_result["nearest_levels"]["resistance"],
                signal_result["nearest_levels"]["distance_to_resistance_pct"],
                signal_result["nearest_levels"]["distance_to_resistance_atr"],
            )

            level_cols = st.columns(4)
            metric_with_help(level_cols[0], "Nearest Support", support_value, delta=support_delta, help_key="nearest_support")
            metric_with_help(
                level_cols[1],
                "Nearest Resistance",
                "Price discovery" if level_snapshot["price_discovery_mode"] else resistance_value,
                delta="No overhead actionable resistance" if level_snapshot["price_discovery_mode"] else resistance_delta,
                help_key="nearest_resistance",
            )
            ema20 = signal_result["features"]["ema20"]
            metric_with_help(level_cols[2], "EMA20", format_money(ema20), help_key="ema20")
            metric_with_help(level_cols[3], "Actionable Threshold", f"{actionable_threshold_pct:.2f}%", help_key="actionable_threshold")

            st.markdown("#### Trade Setup Interpretation")
            setup = signal_result["trade_setup"]
            setup_cols = st.columns(4)
            metric_with_help(setup_cols[0], "Market Regime", setup["market_regime"], help_key="market_regime")
            metric_with_help(setup_cols[1], "Current Bias", setup["current_bias"], help_key="current_bias")
            metric_with_help(
                setup_cols[2],
                "Confirmation Level",
                format_money(setup["confirmation_level"]),
                delta=setup["confirmation_label"],
                help_key="confirmation_level",
            )
            metric_with_help(
                setup_cols[3],
                "Invalidation Level",
                format_money(setup["invalidation_level"]),
                delta=setup["invalidation_label"],
                help_key="invalidation_level",
            )

            setup_cols = st.columns(5)
            metric_with_help(setup_cols[0], "Nearest Support", format_money(setup["nearest_support"]), help_key="nearest_support")
            metric_with_help(
                setup_cols[1],
                "Nearest Resistance",
                "Price discovery" if setup["nearest_resistance"] is None else format_money(setup["nearest_resistance"]),
                help_key="nearest_resistance",
            )
            metric_with_help(setup_cols[2], "Potential Upside", format_percent(setup["potential_upside_pct"]), help_key="potential_upside")
            metric_with_help(setup_cols[3], "Downside Risk", format_percent(setup["downside_risk_pct"]), help_key="downside_risk")
            metric_with_help(
                setup_cols[4],
                "Risk / Reward",
                f"{setup['risk_reward_ratio']:.2f}" if setup["risk_reward_ratio"] is not None else "N/A",
                help_key="risk_reward",
            )

            st.info(f"{ticker}: {setup['narrative']}")

            st.markdown("#### Explanation")
            for explanation in signal_result["explanations"]:
                st.write(f"- {explanation}")

            with st.expander("Score Breakdown"):
                for label, value in signal_result["score_components"].items():
                    st.write(f"**{label.replace('_', ' ').title()}:** {value:+.2f}")

        st.divider()
        st.subheader("Will’s Signal")
        st.caption("Will’s Signal uses daily candles only and is separate from the dashboard's other signal engines, support/resistance logic, VCP logic, and aggregate scores.")

        st.markdown("#### Will’s Signal")
        wills_cols = st.columns([1.4, 2.6])
        metric_with_help(wills_cols[0], "Will’s Signal Score", f"{wills_signal_result['score']} / {wills_signal_result['max_score']}", help_key="wills_signal_score")
        metric_with_help(wills_cols[1], "Daily Bars Used", str(wills_signal_result.get("daily_bars_used", 0)), help_key="daily_bars_used")

        interpretation = wills_signal_result["interpretation"]
        render_callout_box(
            f"Interpretation: {interpretation['label']}",
            interpretation["message"],
            interpretation["accent"],
        )

        component_scores = wills_signal_result.get("component_scores", {})
        calculation_detail = (
            f"ADX {component_scores.get('ADX(14)', 0)} + "
            f"DI {component_scores.get('+DI vs -DI', 0)} + "
            f"RSI {component_scores.get('RSI(14)', 0)} + "
            f"MACD {component_scores.get('MACD Daily', 0)} + "
            f"20 EMA Slope {component_scores.get('20 EMA Slope', 0)} + "
            f"Price vs 20 EMA {component_scores.get('Price vs 20 EMA', 0)}"
        )
        st.write(f"**Total Will’s Signal Score = {wills_signal_result['score']} / {wills_signal_result['max_score']}**")
        st.caption(f"Calculation detail: {calculation_detail} = {wills_signal_result['score']} / {wills_signal_result['max_score']}")

        wills_table = build_wills_signal_table(wills_signal_result, first_column_label="Indicator")
        st.table(wills_table)

        st.markdown("#### Non-Scored Warnings")
        st.caption(get_help_text("non_scored_warnings"))
        earnings_warning = wills_signal_result["warnings"]["earnings"]
        atr_extension_warning = wills_signal_result["warnings"]["atr_extension"]
        st.write(f"- {earnings_warning['message']}")
        if atr_extension_warning.get("extension_atr") is not None:
            st.write(
                f"- {atr_extension_warning['message']} "
                f"(Extension from 20 EMA: {atr_extension_warning['extension_atr']:.2f} ATR)"
            )
        else:
            st.write(f"- {atr_extension_warning['message']}")

        st.markdown("#### Will’s Exhaustion Signal")
        exhaustion_cols = st.columns([1.4, 2.6])
        metric_with_help(
            exhaustion_cols[0],
            "Will’s Exhaustion Signal Score",
            f"{wills_exhaustion_result['score']} / {wills_exhaustion_result['max_score']}",
            help_key="wills_exhaustion_signal_score",
        )
        metric_with_help(exhaustion_cols[1], "Daily Bars Used", str(wills_exhaustion_result.get("daily_bars_used", 0)), help_key="daily_bars_used")

        exhaustion_interpretation = wills_exhaustion_result["interpretation"]
        render_callout_box(
            f"Interpretation: {exhaustion_interpretation['label']}",
            exhaustion_interpretation["message"],
            exhaustion_interpretation["accent"],
        )

        exhaustion_component_scores = wills_exhaustion_result.get("component_scores", {})
        exhaustion_calculation_detail = (
            f"ADX Trend Aging {exhaustion_component_scores.get('ADX Trend Aging', 0)} + "
            f"DI Weakening {exhaustion_component_scores.get('+DI / -DI Weakening', 0)} + "
            f"RSI Exhaustion {exhaustion_component_scores.get('RSI Exhaustion / Divergence', 0)} + "
            f"MACD Weakening {exhaustion_component_scores.get('MACD Weakening', 0)} + "
            f"Price vs 20 EMA / Extension {exhaustion_component_scores.get('Price vs 20 EMA / Extension', 0)}"
        )
        st.write(
            f"**Total Will’s Exhaustion Signal Score = {wills_exhaustion_result['score']} / {wills_exhaustion_result['max_score']}**"
        )
        st.caption(
            f"Calculation detail: {exhaustion_calculation_detail} = {wills_exhaustion_result['score']} / {wills_exhaustion_result['max_score']}"
        )

        exhaustion_table = build_wills_signal_table(wills_exhaustion_result, first_column_label="Category")
        st.table(exhaustion_table)

        started_at = perf_counter()
        chart = create_price_chart(
            df,
            ticker,
            sma_period=sma_period if sma_enabled else None,
            ema_period=ema_period if ema_enabled else None,
            actionable_levels=support_levels + resistance_levels if show_zones else None,
            historical_levels=historical_levels,
        )
        record_timing("Chart rendering", started_at)
        st.plotly_chart(chart, use_container_width=True)

        with st.expander("Support/Resistance Methodology"):
            st.write("Levels are automatically detected from swing highs and swing lows.")
            st.write("Nearby swing points are clustered into zones using ATR-based tolerance so volatile stocks get wider grouping than quiet stocks.")
            st.write("Clustering answers whether prices are basically the same zone. Actionable filtering separately answers whether the zone is close enough to matter right now.")
            st.write("Support is only below current price. Resistance is only above current price.")
            st.write("Level strength is based on touches, recency, relative volume near the zone, and a penalty for repeated violations.")
            st.write(f"Current/visual analysis uses swing sensitivity N = {swing_sensitivity}, cluster tolerance {cluster_tolerance_pct:.2f}%, and actionable threshold {actionable_threshold_pct:.2f}%.")

        twenty_bar_return = None
        if len(close_series) > 20:
            prior_close = close_series.shift(20).iloc[-1]
            if pd.notna(prior_close) and prior_close != 0:
                twenty_bar_return = (close_series.iloc[-1] / prior_close - 1) * 100

        st.divider()
        st.subheader("Key Metrics")
        metric_cols = st.columns(4)
        metric_with_help(metric_cols[0], "Latest Close", format_money(latest_close), help_key="latest_close")
        metric_with_help(metric_cols[1], "20-Bar Return", f"{twenty_bar_return:.2f}%" if twenty_bar_return is not None else "N/A", help_key="twenty_bar_return")
        metric_with_help(metric_cols[2], "Annual Volatility", f"{volatility:.2f}%", help_key="annual_volatility")
        metric_with_help(metric_cols[3], "Volatility Percentile", f"{volatility_percentile * 100:.0f}%" if volatility_percentile is not None else "N/A", help_key="volatility_percentile")

        metric_cols = st.columns(4)
        metric_with_help(metric_cols[0], "Period Return", f"{period_return:.2f}%", help_key="period_return")
        metric_with_help(metric_cols[1], "Average Volume (20)", f"{average_volume:,.0f}", help_key="average_volume_20")
        metric_with_help(metric_cols[2], "Cluster Tolerance", f"{cluster_tolerance_pct:.2f}%", help_key="cluster_tolerance")
        metric_with_help(metric_cols[3], "Max Drawdown", f"{max_drawdown:.2f}%", help_key="max_drawdown")

        period_cols = st.columns(3)
        metric_with_help(period_cols[0], "Start Price", format_money(start_close), help_key="start_price")
        metric_with_help(period_cols[1], "End Price", format_money(end_close), help_key="end_price")
        metric_with_help(period_cols[2], "Bars Used", str(bars_used), help_key="bars_used")

        if show_backtest:
            st.divider()
            st.subheader("Historical Backtest")
            st.caption("Historical backtest uses only information available up to each historical date. No look-ahead data is used.")
            st.caption("Exploratory backtest only. It does not include trading costs, slippage, taxes, or execution risk.")
            st.caption("When enabled, VCP is treated as a supporting factor, not a standalone trade instruction.")
            backtest_skip_reason = None
            if len(df) > MAX_BACKTEST_ROWS:
                backtest_skip_reason = (
                    f"Backtest skipped for performance safety because the dataset has {len(df)} rows, "
                    f"which exceeds the {MAX_BACKTEST_ROWS}-row guardrail."
                )
            elif is_intraday_interval(interval):
                backtest_skip_reason = "Backtest is disabled for intraday intervals in this deployment-oriented build to avoid heavy Community Cloud workloads."

            if backtest_skip_reason:
                st.warning(backtest_skip_reason)
            else:
                with st.spinner("Running backtest..."):
                    started_at = perf_counter()
                    backtest_results = backtest_cached(
                        df=df,
                        ticker=ticker,
                        max_levels=zone_count,
                        swing_sensitivity=swing_sensitivity,
                        enable_vcp_detection=enable_vcp_detection and not bool(vcp_skip_reason),
                        vcp_min_base_length=vcp_min_base_length,
                        vcp_max_base_length=vcp_max_base_length,
                        interval=interval,
                    )
                    record_timing("Historical backtest", started_at)

                backtest_cols = st.columns(5)
                metric_with_help(backtest_cols[0], "Total Signals", int(backtest_results["total_signals"]), help_key="total_signals")
                metric_with_help(backtest_cols[1], "BUY Signals", int(backtest_results["buy_signals"]), help_key="buy_signals")
                metric_with_help(backtest_cols[2], "SELL Signals", int(backtest_results["sell_signals"]), help_key="sell_signals")
                metric_with_help(backtest_cols[3], "BUY Win Rate", f"{backtest_results['buy_win_rate']:.1f}%", help_key="buy_win_rate")
                metric_with_help(backtest_cols[4], "SELL Win Rate", f"{backtest_results['sell_win_rate']:.1f}%", help_key="sell_win_rate")

                avg_cols = st.columns(3)
                metric_with_help(avg_cols[0], "Avg Forward Return (5)", f"{backtest_results['average_forward_return_5']:.2f}%", help_key="average_forward_return")
                metric_with_help(avg_cols[1], "Avg Forward Return (10)", f"{backtest_results['average_forward_return_10']:.2f}%", help_key="average_forward_return")
                metric_with_help(avg_cols[2], "Avg Forward Return (20)", f"{backtest_results['average_forward_return_20']:.2f}%", help_key="average_forward_return")

                median_cols = st.columns(3)
                metric_with_help(median_cols[0], "Median Forward Return (5)", f"{backtest_results['median_forward_return_5']:.2f}%", help_key="median_forward_return")
                metric_with_help(median_cols[1], "Median Forward Return (10)", f"{backtest_results['median_forward_return_10']:.2f}%", help_key="median_forward_return")
                metric_with_help(median_cols[2], "Median Forward Return (20)", f"{backtest_results['median_forward_return_20']:.2f}%", help_key="median_forward_return")

                risk_cols = st.columns(2)
                metric_with_help(risk_cols[0], "Signal Equity Max Drawdown", f"{backtest_results['max_drawdown']:.2f}%", help_key="signal_equity_drawdown")
                metric_with_help(risk_cols[1], "Buy-and-Hold Return", f"{backtest_results['buy_and_hold_return']:.2f}%", help_key="buy_and_hold_return")

                consistency_cols = st.columns(3)
                metric_with_help(consistency_cols[0], "Backtest Start Price", format_money(backtest_results["start_price"]), help_key="start_price")
                metric_with_help(consistency_cols[1], "Backtest End Price", format_money(backtest_results["end_price"]), help_key="end_price")
                metric_with_help(consistency_cols[2], "Backtest Bars Used", str(int(backtest_results["bars_used"])), help_key="bars_used")

        st.divider()
        st.subheader("Actionable Support/Resistance Levels")
        actionable_rows = support_levels + resistance_levels
        if actionable_rows:
            st.table(build_level_table(actionable_rows))
        else:
            st.info("No actionable support or resistance levels were detected near current price.")

        st.subheader("Historical Support/Resistance Levels")
        historical_rows = historical_supports + historical_resistances
        if historical_rows:
            st.table(build_level_table(historical_rows))
        else:
            st.caption("Enable 'Show Historical Levels' in the sidebar to inspect distant zones.")

        if enable_vcp_detection:
            st.divider()
            st.subheader("VCP Pattern Detector")
            st.caption("Current/visual analysis uses the selected visible data range. VCP is a setup condition, not a trade recommendation.")
            if vcp_skip_reason:
                st.warning(vcp_skip_reason)
            elif vcp_result is not None:
                if vcp_result["status"] == "No Clear VCP":
                    st.write("No Clear VCP")
                    for reason in vcp_result.get("failed_reasons", []):
                        st.write(f"- {reason}")
                else:
                    st.success(vcp_result["status"])

                vcp_cols = st.columns(5)
                metric_with_help(vcp_cols[0], "VCP Status", vcp_result["status"], help_key="vcp_status")
                metric_with_help(vcp_cols[1], "VCP Score", f"{vcp_result['score']:.2f}", help_key="vcp_score")
                metric_with_help(vcp_cols[2], "Pivot Price", format_money(vcp_result.get("pivot")), help_key="pivot_price")
                metric_with_help(vcp_cols[3], "Current Close", format_money(vcp_result.get("current_close")), help_key="latest_close")
                metric_with_help(vcp_cols[4], "Distance to Pivot %", format_percent(vcp_result.get("distance_to_pivot_pct")), help_key="distance_to_pivot")

                vcp_cols = st.columns(5)
                metric_with_help(vcp_cols[0], "Base Length", str(vcp_result.get("base_length", "N/A")), help_key="vcp_base_length")
                metric_with_help(vcp_cols[1], "Base High", format_money(vcp_result.get("base_high")), help_key="vcp_base_high")
                metric_with_help(vcp_cols[2], "Base Low", format_money(vcp_result.get("base_low")), help_key="vcp_base_low")
                metric_with_help(vcp_cols[3], "Base Depth %", format_percent(vcp_result.get("base_depth_pct")), help_key="vcp_base_depth")
                metric_with_help(vcp_cols[4], "Pullback Sequence", ", ".join(f"{value * 100:.1f}%" for value in vcp_result.get("pullbacks", [])) or "N/A", help_key="vcp_pullback_sequence")

                vcp_cols = st.columns(5)
                metric_with_help(vcp_cols[0], "ATR Contraction Ratio", f"{vcp_result['atr_contraction_ratio']:.2f}" if vcp_result.get("atr_contraction_ratio") is not None else "N/A", help_key="vcp_atr_contraction_ratio")
                metric_with_help(vcp_cols[1], "Volume Contraction Ratio", f"{vcp_result['volume_contraction_ratio']:.2f}" if vcp_result.get("volume_contraction_ratio") is not None else "N/A", help_key="vcp_volume_contraction_ratio")
                metric_with_help(vcp_cols[2], "Breakout Trigger", format_money(vcp_result.get("breakout_trigger_price")), help_key="vcp_breakout_trigger")
                metric_with_help(vcp_cols[3], "Breakout Volume Requirement", f"{vcp_result['breakout_volume_requirement']:,.0f}" if vcp_result.get("breakout_volume_requirement") is not None else "N/A", help_key="vcp_breakout_volume_requirement")
                metric_with_help(vcp_cols[4], "Volume Confirmed", "Yes" if vcp_result.get("breakout_volume_confirmed") else "No", help_key="vcp_volume_confirmed")

                st.markdown("#### VCP Explanation")
                for explanation in vcp_result.get("explanations", []):
                    st.write(f"- {explanation}")
                st.write("- VCP is a setup condition, not a trade recommendation.")

                show_vcp_chart = vcp_result["status"] in {"Strong VCP Candidate", "Possible VCP Candidate", "VCP Breakout Confirmed"} or show_vcp_diagnostics
                if show_vcp_chart:
                    started_at = perf_counter()
                    vcp_chart = create_vcp_evidence_chart(
                        vcp_result,
                        support_levels=support_levels,
                        resistance_levels=resistance_levels,
                        show_annotations=show_vcp_annotations,
                    )
                    record_timing("VCP evidence chart rendering", started_at)
                    st.subheader("VCP Evidence Chart")
                    st.plotly_chart(vcp_chart, use_container_width=True)
                    st.markdown("#### Why this was flagged as VCP")
                    st.write("- Current/visual analysis is based on the selected visible data range.")
                    for explanation in vcp_result.get("explanations", []):
                        st.write(f"- {explanation}")

        with st.expander("Performance diagnostics", expanded=False):
            st.caption("Timings below reflect the current run. Cached results should return much faster than fresh computations.")
            st.caption(get_help_text("performance_diagnostics"))
            if performance_timings:
                diagnostics_df = pd.DataFrame(performance_timings)
                diagnostics_df["Milliseconds"] = diagnostics_df["Seconds"] * 1000
                diagnostics_df["Seconds"] = diagnostics_df["Seconds"].map(lambda value: f"{value:.4f}")
                diagnostics_df["Milliseconds"] = diagnostics_df["Milliseconds"].map(lambda value: f"{value:.1f}")
                st.table(diagnostics_df[["Step", "Seconds", "Milliseconds"]])
            else:
                st.write("No performance timings recorded for this run.")

    except Exception as exc:
        st.error(f"Error loading dashboard: {exc}")
        import traceback
        st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
