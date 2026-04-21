import datetime as dt
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import (
    ALLOCATION_BETA, BASE_BOND_WEIGHT, BASE_EQUITY_WEIGHT,
    COMPOSITE_SCORE_BINS, COMPOSITE_SCORE_BUCKET_LABELS, COMPOSITE_SCORE_WEIGHTS,
    FORWARD_RETURN_HORIZONS, INTERMARKET_RATIOS, N_CLUSTERS,
    ROLLING_Z_WINDOW, VRP_WINDOW, WEAK_SEASONALITY_MONTHS,
)

logger = logging.getLogger(__name__)


def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).rename("log_return")


def rolling_zscore(series: pd.Series, window: int = ROLLING_Z_WINDOW) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std


def rolling_percentile_rank(series: pd.Series, window: int = ROLLING_Z_WINDOW) -> pd.Series:
    def pct_rank(x: pd.Series) -> float:
        return x.rank(pct=True).iloc[-1]
    return series.rolling(window).apply(pct_rank, raw=False)


def compute_net_liquidity(macro: pd.DataFrame) -> pd.DataFrame:
    nl = macro["WALCL"] - macro["WTREGEN"] - macro["RRPONTSYD"]
    nl_yoy = nl.pct_change(252)
    return pd.DataFrame({"net_liquidity": nl, "net_liquidity_yoy": nl_yoy, "z_net_liquidity": rolling_zscore(nl_yoy)})


def compute_term_spread(macro: pd.DataFrame) -> pd.DataFrame:
    spread = macro["T10Y2Y"].copy()
    return pd.DataFrame({"t10y2y": spread, "z_t10y2y": rolling_zscore(spread)})


def compute_erp(spy_prices: pd.Series, macro: pd.DataFrame) -> pd.Series:
    earnings_yield = pd.Series(1.0 / 20.0, index=spy_prices.index)
    dgs10 = macro["DGS10"] / 100.0
    return (earnings_yield - dgs10).rename("erp")


def compute_intermarket_ratios(prices: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name, (num, den) in INTERMARKET_RATIOS.items():
        ratio = prices[num] / prices[den]
        tmp = pd.DataFrame({"value": ratio, "zscore": rolling_zscore(ratio), "percentile": rolling_percentile_rank(ratio)})
        tmp["ratio"] = name
        records.append(tmp)
    ratios = pd.concat(records, axis=0).set_index("ratio", append=True).sort_index()
    return ratios


def compute_tail_risk(vol_indices: pd.DataFrame) -> pd.DataFrame:
    df = vol_indices.copy()
    df["MOVE_VIX_RATIO"] = df["^MOVE"] / df["^VIX"]
    df["VVIX_FLAG"] = df["^VVIX"] > 110.0
    df["SKEW_FLAG"] = df["^SKEW"] > 135.0
    df["VIX_TERM_RATIO"] = df["^VIX3M"] / df["^VIX"]
    df["CRITICAL_STRESS"] = df["VIX_TERM_RATIO"] < 1.0
    return df


def compute_vrp(spy_prices: pd.Series, vix: pd.Series) -> pd.DataFrame:
    log_rets = compute_log_returns(spy_prices)
    realized_vol = log_rets.rolling(VRP_WINDOW).std(ddof=0) * np.sqrt(252.0)
    vrp = vix / 100.0 - realized_vol
    return pd.DataFrame({"realized_vol": realized_vol, "vrp": vrp, "z_vrp": rolling_zscore(vrp)})


def compute_cusum(returns: pd.Series, threshold: float = 5.0, drift: float = 0.0) -> pd.DataFrame:
    rets = returns.dropna()
    h = threshold * rets.std(ddof=0)
    g_pos = np.zeros(len(rets))
    g_neg = np.zeros(len(rets))
    flags = np.zeros(len(rets), dtype=bool)
    for i in range(1, len(rets)):
        g_pos[i] = max(0.0, g_pos[i - 1] + rets.iloc[i] - drift)
        g_neg[i] = min(0.0, g_neg[i - 1] + rets.iloc[i] + drift)
        if g_pos[i] > h or abs(g_neg[i]) > h:
            flags[i] = True
            g_pos[i] = 0.0
            g_neg[i] = 0.0
    return pd.DataFrame({"g_pos": pd.Series(g_pos, index=rets.index), "g_neg": pd.Series(g_neg, index=rets.index), "regime_shift": pd.Series(flags, index=rets.index)})


def compute_kmeans_regimes(features: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> Tuple[pd.Series, int]:
    clean = features.dropna()
    if clean.empty:
        logger.warning("Feature matrix vuota per K-Means.")
        return pd.Series(np.nan, index=features.index, name="cluster"), -1
    scaler = StandardScaler()
    X = scaler.fit_transform(clean.values)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = pd.Series(model.fit_predict(X), index=clean.index, name="cluster", dtype="int64")
    labels = labels.reindex(features.index).ffill()
    return labels, int(labels.iloc[-1])


def compute_seasonality_flags(index: pd.DatetimeIndex) -> Dict[str, bool]:
    if len(index) == 0:
        return {"is_opex_week": False, "is_weak_month": False}
    today = index.max().date()
    month = today.month
    ms = dt.date(today.year, month, 1)
    first_fri = ms + dt.timedelta(days=(4 - ms.weekday()) % 7)
    third_fri = first_fri + dt.timedelta(days=14)
    is_opex = (third_fri - dt.timedelta(days=2)) <= today <= (third_fri + dt.timedelta(days=2))
    return {"is_opex_week": is_opex, "is_weak_month": month in WEAK_SEASONALITY_MONTHS}


def compute_composite_score(ratios: pd.DataFrame, net_liquidity: pd.DataFrame, vrp: pd.DataFrame, vix: pd.Series, term_spread: pd.DataFrame) -> pd.Series:
    z_wide = ratios["zscore"].unstack("ratio").rename(columns={"Copper/Gold": "z_copper_gold", "HYG/LQD": "z_hyg_lqd", "XLY/XLP": "z_xly_xlp", "XLK/XLU": "z_xlk_xlu", "RSP/SPY": "z_rsp_spy"})
    features = pd.concat([z_wide[["z_copper_gold", "z_hyg_lqd", "z_xly_xlp"]], net_liquidity["z_net_liquidity"], vrp["z_vrp"], rolling_zscore(vix).rename("z_vix"), term_spread["z_t10y2y"]], axis=1).dropna(how="all")
    composite = pd.Series(0.0, index=features.index, name="composite_score")
    for key, weight in COMPOSITE_SCORE_WEIGHTS.items():
        if key in features.columns:
            composite += weight * features[key].fillna(0.0)
    return composite


def compute_dynamic_allocation(composite_score: pd.Series, base_equity: float = BASE_EQUITY_WEIGHT, base_bonds: float = BASE_BOND_WEIGHT, beta: float = ALLOCATION_BETA) -> pd.DataFrame:
    score = composite_score.clip(-3.0, 3.0).fillna(0.0)
    eq = (base_equity + beta * score).clip(0.0, 1.0)
    return pd.DataFrame({"equity_weight": eq, "bond_weight": 1.0 - eq})


def compute_forward_return_expectancy(spy_prices: pd.Series, composite_score: pd.Series) -> pd.DataFrame:
    prices = spy_prices.dropna()
    comp = composite_score.reindex(prices.index).dropna()
    bucket = pd.cut(comp, bins=COMPOSITE_SCORE_BINS, labels=COMPOSITE_SCORE_BUCKET_LABELS, include_lowest=True)
    current_bucket = bucket.iloc[-1]
    rows = []
    for h in FORWARD_RETURN_HORIZONS:
        fwd = (prices.shift(-h) / prices - 1.0).reindex(comp.index)
        hist = fwd[bucket == current_bucket].dropna()
        rows.append({"Horizon": f"{h}d", "WinRatePct": (hist > 0).mean() * 100 if len(hist) else np.nan, "AvgReturnPct": hist.mean() * 100 if len(hist) else np.nan, "Bucket": current_bucket})
    return pd.DataFrame(rows).set_index("Horizon")


def backtest_dynamic_allocation(spy_prices: pd.Series, tlt_prices: pd.Series, allocation: pd.DataFrame, base_equity: float = BASE_EQUITY_WEIGHT, base_bonds: float = BASE_BOND_WEIGHT) -> Tuple[pd.DataFrame, float, float]:
    idx = spy_prices.index.intersection(tlt_prices.index).intersection(allocation.index)
    spy = spy_prices.reindex(idx).ffill()
    tlt = tlt_prices.reindex(idx).ffill()
    alloc = allocation.reindex(idx).ffill()
    spy_r = spy.pct_change().fillna(0.0)
    tlt_r = tlt.pct_change().fillna(0.0)
    w_eq = alloc["equity_weight"].shift(1).fillna(base_equity)
    w_bd = alloc["bond_weight"].shift(1).fillna(base_bonds)
    model_curve = (1 + w_eq * spy_r + w_bd * tlt_r).cumprod()
    bench_curve = (1 + base_equity * spy_r + base_bonds * tlt_r).cumprod()
    curves = pd.DataFrame({"Model": model_curve, "Benchmark": bench_curve})
    mdd_model = (model_curve / model_curve.cummax() - 1).min()
    mdd_bench = (bench_curve / bench_curve.cummax() - 1).min()
    return curves, float(mdd_model), float(mdd_bench)


def build_intermarket_heatmap_snapshot(ratios: pd.DataFrame) -> pd.DataFrame:
    if ratios.empty:
        return pd.DataFrame(columns=["value", "zscore", "percentile"])
    last_date = ratios.index.get_level_values(0).max()
    return ratios.xs(last_date, level=0)[["value", "zscore", "percentile"]]


def build_quant_context(prices: pd.DataFrame, macro: pd.DataFrame, vol: pd.DataFrame) -> Dict[str, Any]:
    spy = prices["SPY"].dropna()
    net_liq = compute_net_liquidity(macro)
    term_spread = compute_term_spread(macro)
    erp = compute_erp(spy, macro)
    ratios = compute_intermarket_ratios(prices)
    ratios_snapshot = build_intermarket_heatmap_snapshot(ratios)
    tail_risk = compute_tail_risk(vol)
    vrp = compute_vrp(spy, vol["^VIX"])
    cusum_df = compute_cusum(compute_log_returns(spy))
    features_km = pd.concat([ratios["zscore"].unstack("ratio")[["Copper/Gold", "HYG/LQD", "XLY/XLP"]], net_liq["z_net_liquidity"], vrp["z_vrp"], rolling_zscore(vol["^VIX"]).rename("z_vix"), term_spread["z_t10y2y"]], axis=1)
    features_km.columns = ["z_copper_gold", "z_hyg_lqd", "z_xly_xlp", "z_net_liquidity", "z_vrp", "z_vix", "z_t10y2y"]
    clusters, current_cluster = compute_kmeans_regimes(features_km)
    seasonality = compute_seasonality_flags(spy.index)
    composite = compute_composite_score(ratios=ratios, net_liquidity=net_liq, vrp=vrp, vix=vol["^VIX"], term_spread=term_spread)
    allocation = compute_dynamic_allocation(composite)
    equity_curves, mdd_model, mdd_bench = backtest_dynamic_allocation(spy, prices["TLT"], allocation)
    fwd = compute_forward_return_expectancy(spy, composite)
    return {
        "spy_prices": spy, "net_liquidity": net_liq, "term_spread": term_spread, "erp": erp,
        "ratios_panel": ratios, "ratios_snapshot": ratios_snapshot, "tail_risk": tail_risk,
        "vrp": vrp, "cusum": cusum_df, "clusters": clusters, "current_cluster": current_cluster,
        "seasonality": seasonality,
        "composite": {"series": composite, "current": float(composite.dropna().iloc[-1]) if not composite.dropna().empty else np.nan},
        "allocation": {"series": allocation, "current_equity": float(allocation["equity_weight"].iloc[-1]), "current_bonds": float(allocation["bond_weight"].iloc[-1])},
        "backtest": {"equity_curves": equity_curves, "max_drawdown_model": mdd_model, "max_drawdown_benchmark": mdd_bench},
        "forward_expectancy": fwd,
    }
