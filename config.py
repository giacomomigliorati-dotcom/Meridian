import numpy as np
from typing import Dict, List

LOOKBACK_WINDOW: int = 252
N_CLUSTERS: int = 4
ROLLING_Z_WINDOW: int = 252
VRP_WINDOW: int = 20
BASE_EQUITY_WEIGHT: float = 0.60
BASE_BOND_WEIGHT: float = 0.40
ALLOCATION_BETA: float = 0.15
FORWARD_RETURN_HORIZONS: List[int] = [5, 10, 20]
COMPOSITE_SCORE_BINS = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
COMPOSITE_SCORE_BUCKET_LABELS = ["Very Defensive", "Defensive", "Neutral", "Aggressive", "Very Aggressive"]
WEAK_SEASONALITY_MONTHS: List[int] = [9]
EQUITY_TICKERS: List[str] = ["SPY", "RSP", "TLT", "UUP", "GLD", "USO", "CPER", "HYG", "LQD", "XLY", "XLP", "XLK", "XLU"]
VOL_INDEX_TICKERS: List[str] = ["^VIX", "^VVIX", "^SKEW", "^MOVE", "^VIX3M"]
FRED_SERIES: List[str] = ["WALCL", "WTREGEN", "RRPONTSYD", "T10Y2Y", "DGS10"]
INTERMARKET_RATIOS: Dict[str, tuple] = {
    "Copper/Gold": ("CPER", "GLD"),
    "HYG/LQD": ("HYG", "LQD"),
    "XLY/XLP": ("XLY", "XLP"),
    "XLK/XLU": ("XLK", "XLU"),
    "RSP/SPY": ("RSP", "SPY"),
}
COMPOSITE_SCORE_WEIGHTS: Dict[str, float] = {
    "z_copper_gold": 1.0,
    "z_hyg_lqd": 1.0,
    "z_xly_xlp": 1.0,
    "z_net_liquidity": 1.0,
    "z_vrp": 1.0,
    "z_vix": -1.0,
    "z_t10y2y": -1.0,
}
