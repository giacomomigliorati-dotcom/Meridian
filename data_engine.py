import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from pandas.tseries.offsets import BDay
from pandas_datareader import data as pdr

from config import EQUITY_TICKERS, FRED_SERIES, LOOKBACK_WINDOW, VOL_INDEX_TICKERS

logger = logging.getLogger(__name__)


def _get_date_range(lookback_window: int = LOOKBACK_WINDOW) -> Tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp.today().normalize()
    extra_days = lookback_window + 260
    start = end - BDay(extra_days)
    return start, end


def _init_alpaca_client() -> Optional[StockHistoricalDataClient]:
    api_key = st.secrets.get("ALPACA_API_KEY")
    secret_key = st.secrets.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        logger.warning("Chiavi Alpaca mancanti: verra usato yfinance come fonte principale.")
        return None
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def _fetch_alpaca_series(
    client: StockHistoricalDataClient,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    if client is None:
        raise RuntimeError("Alpaca client non inizializzato")
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start.tz_localize("UTC"),
        end=end.tz_localize("UTC"),
        limit=None,
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.xs(symbol, level="symbol")
        except KeyError:
            df = df.xs(symbol, level=0)
    series = df["close"].copy()
    series.name = symbol
    series.index = series.index.tz_localize(None)
    return series


def _fetch_yf_series(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    data = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise RuntimeError(f"Nessun dato yfinance per {symbol}")
    if "Adj Close" in data.columns:
        series = data["Adj Close"].copy()
    else:
        series = data["Close"].copy()
    series.name = symbol
    return series


def _fetch_price_series(
    client: Optional[StockHistoricalDataClient],
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    try:
        if client is not None:
            return _fetch_alpaca_series(client, symbol, start, end)
        raise RuntimeError("Alpaca client non disponibile")
    except Exception as exc:
        logger.error("Errore Alpaca per %s: %s. Fallback a yfinance.", symbol, exc)
        try:
            return _fetch_yf_series(symbol, start, end)
        except Exception as exc2:
            logger.error("Errore yfinance per %s: %s. NaN.", symbol, exc2)
            idx = pd.date_range(start=start, end=end, freq="B")
            return pd.Series(np.nan, index=idx, name=symbol)


@st.cache_data(ttl=3600, show_spinner=True)
def fetch_equity_data(lookback_window: int = LOOKBACK_WINDOW) -> pd.DataFrame:
    start, end = _get_date_range(lookback_window)
    client = _init_alpaca_client()
    spy_series = _fetch_price_series(client, "SPY", start, end).dropna().sort_index()
    master_index = spy_series.index
    data: Dict[str, pd.Series] = {"SPY": spy_series.reindex(master_index)}
    tickers = [t for t in EQUITY_TICKERS if t != "SPY"]
    with ThreadPoolExecutor(max_workers=min(8, len(tickers))) as executor:
        futures = {
            executor.submit(_fetch_price_series, client, ticker, start, end): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                series = future.result()
            except Exception as exc:
                logger.error("Errore nel fetch per %s: %s", ticker, exc)
                series = pd.Series(np.nan, index=master_index, name=ticker)
            data[ticker] = series.sort_index().reindex(master_index)
    prices = pd.DataFrame(data, index=master_index).ffill()
    return prices


@st.cache_data(ttl=3600, show_spinner=True)
def fetch_macro_data(spy_index: pd.DatetimeIndex) -> pd.DataFrame:
    if len(spy_index) == 0:
        raise ValueError("Indice SPY vuoto")
    start = spy_index[0] - BDay(5)
    end = spy_index[-1] + BDay(1)
    try:
        macro = pdr.DataReader(FRED_SERIES, "fred", start, end)
    except Exception as exc:
        logger.error("Errore FRED: %s.", exc)
        macro = pd.DataFrame(index=pd.date_range(start=start, end=end, freq="B"))
        for s in FRED_SERIES:
            try:
                macro[s] = _fetch_yf_series(s, start, end)
            except Exception as exc2:
                logger.error("Fallback yfinance fallito per %s: %s.", s, exc2)
                macro[s] = np.nan
    macro = macro.sort_index().reindex(spy_index).ffill().shift(1)
    return macro


@st.cache_data(ttl=3600, show_spinner=True)
def fetch_vol_indices(spy_index: pd.DatetimeIndex) -> pd.DataFrame:
    if len(spy_index) == 0:
        raise ValueError("Indice SPY vuoto")
    start = spy_index[0] - BDay(5)
    end = spy_index[-1] + BDay(1)
    try:
        data = yf.download(VOL_INDEX_TICKERS, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
        vol_df = data.copy()
    except Exception as exc:
        logger.error("Errore yfinance volatilita: %s.", exc)
        vol_df = pd.DataFrame(np.nan, index=pd.date_range(start=start, end=end, freq="B"), columns=VOL_INDEX_TICKERS)
    return vol_df.sort_index().reindex(spy_index).ffill()


@st.cache_data(ttl=3600, show_spinner=True)
def load_all_data(lookback_window: int = LOOKBACK_WINDOW) -> Dict[str, pd.DataFrame]:
    prices = fetch_equity_data(lookback_window=lookback_window)
    spy_index = prices.index
    macro = fetch_macro_data(spy_index=spy_index)
    vol = fetch_vol_indices(spy_index=spy_index)
    return {"prices": prices.ffill(), "macro": macro.ffill(), "vol": vol.ffill()}
