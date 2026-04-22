import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import ALLOCATION_BETA, BASE_BOND_WEIGHT, BASE_EQUITY_WEIGHT, LOOKBACK_WINDOW
import data_engine
import quant_features

# ======================================================
# DEPLOYMENT INSTRUCTIONS:
# 1) Push del progetto su GitHub (tutti i file nella root)
# 2) Login su https://share.streamlit.io con il proprio account GitHub
# 3) New app -> seleziona repo -> Main file path: app.py
# 4) Settings -> Secrets -> aggiungere:
#    ALPACA_API_KEY = "la_tua_chiave"
#    ALPACA_SECRET_KEY = "il_tuo_secret"
# ======================================================

st.set_page_config(
    page_title="Systematic Intermarket Robo-Advisor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("robo_advisor")


def _plot_composite_gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Composite Risk Score"},
            gauge={
                "axis": {"range": [-3, 3]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [-3, -1.5], "color": "#8b0000"},
                    {"range": [-1.5, -0.5], "color": "#ff7f0e"},
                    {"range": [-0.5, 0.5], "color": "#aaaaaa"},
                    {"range": [0.5, 1.5], "color": "#2ca02c"},
                    {"range": [1.5, 3], "color": "#006400"},
                ],
            },
        )
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), height=260)
    return fig


def _plot_equity_curves(curves: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curves.index, y=curves["Model"], mode="lines", name="Model", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=curves.index, y=curves["Benchmark"], mode="lines", name="60/40 Benchmark", line=dict(color="#ff7f0e", width=2, dash="dash")))
    fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10), height=400, xaxis_title="Date", yaxis_title="Equity Curve (indexed to 1)")
    return fig


def main() -> None:
    st.title("Systematic Intermarket Robo-Advisor")
    st.caption("Intermarket Z-Scores \u2022 VRP \u2022 Regime Detection \u2022 Dynamic 60/40 Allocation")

    if st.button("\U0001f504 Refresh Dati"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Caricamento dati di mercato e macro..."):
        try:
            data = data_engine.load_all_data(lookback_window=LOOKBACK_WINDOW)
        except Exception as exc:
            logger.exception("Errore nel caricamento dati: %s", exc)
            st.error("Errore nel caricamento dei dati. Verifica log o chiavi API.")
            st.stop()
            return

    with st.spinner("Calcolo feature quantitative..."):
        ctx = quant_features.build_quant_context(data["prices"], data["macro"], data["vol"])

    composite_current = ctx["composite"]["current"]
    composite_series = ctx["composite"]["series"]
    allocation_series = ctx["allocation"]["series"]
    eq = ctx["allocation"]["current_equity"]
    bd = ctx["allocation"]["current_bonds"]
    fwd = ctx["forward_expectancy"]
    curves = ctx["backtest"]["equity_curves"]
    mdd_m = ctx["backtest"]["max_drawdown_model"]
    mdd_b = ctx["backtest"]["max_drawdown_benchmark"]
    ratios_snap = ctx["ratios_snapshot"]
    tail = ctx["tail_risk"]
    net_liq = ctx["net_liquidity"]
    erp = ctx["erp"]
    cusum_df = ctx["cusum"]
    cluster = ctx["current_cluster"]
    season = ctx["seasonality"]
    vrp_df = ctx["vrp"]

    # --- TOP ROW ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        regime = "Risk-On" if composite_current > 0.0 else "Risk-Off"
        color = "#2ca02c" if regime == "Risk-On" else "#d62728"
        st.markdown(f'<div style="padding:.75rem;border-radius:.5rem;background:{color};text-align:center;"><span style="font-size:1.2rem;font-weight:700;color:white;">{regime}</span></div>', unsafe_allow_html=True)
        st.write(f"Cluster K-Means: **{cluster}**")
    with c2:
        st.plotly_chart(_plot_composite_gauge(composite_current), use_container_width=True)
    with c3:
        st.metric("Suggested Equity", f"{eq * 100:.1f}%", delta=f"{(eq - BASE_EQUITY_WEIGHT) * 100:.1f} pts vs 60/40")
        st.metric("Suggested Bonds", f"{bd * 100:.1f}%", delta=f"{(bd - BASE_BOND_WEIGHT) * 100:.1f} pts vs 60/40")
    with c4:
        if season["is_opex_week"]:
            st.warning("\u26a0\ufe0f OPEX Week attiva (3\u00b0 venerd\u00ec del mese).")
        if season["is_weak_month"]:
            st.warning("\u26a0\ufe0f Mese con stagionalit\u00e0 debole (Settembre).")
        st.write(f"Score: **{composite_current:.2f}** | Beta: **{ALLOCATION_BETA:.2f}**")

    st.markdown("---")

    # --- FORWARD RETURNS & EQUITY CURVE ---
    ca, cb = st.columns(2)
    with ca:
        st.subheader("Forward Returns Expectancy")
        st.dataframe(fwd.style.format({"WinRatePct": "{:.1f}%", "AvgReturnPct": "{:.2f}%"}), use_container_width=True)
        shift = bool(cusum_df["regime_shift"].iloc[-1])
        txt = "Regime Shift DETECTED" if shift else "Regime normale"
        col = "#d62728" if shift else "#2ca02c"
        st.markdown(f'<div style="padding:.5rem;border-radius:.5rem;background:{col};text-align:center;"><span style="font-size:1rem;font-weight:600;color:white;">CUSUM: {txt}</span></div>', unsafe_allow_html=True)
    with cb:
        st.subheader("Historical Equity Curve (Model vs 60/40)")
        st.plotly_chart(_plot_equity_curves(curves), use_container_width=True)
        st.write(f"Max Drawdown Model: **{mdd_m * 100:.1f}%** | 60/40: **{mdd_b * 100:.1f}%**")

    st.markdown("---")

    # --- TAIL RISK & LIQUIDITY ---
    st.subheader("Tail Risk & Liquidity")
    last_idx = tail.index.max()
    prev_idx = tail.index[tail.index < last_idx].max()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    nl_l = net_liq.loc[last_idx, "net_liquidity_yoy"]
    nl_p = net_liq.loc[prev_idx, "net_liquidity_yoy"]
    m1.metric("US Net Liquidity YoY", f"{nl_l * 100:.1f}%", delta=f"{(nl_l - nl_p) * 100:.1f} p.p.")

    vr_l = vrp_df.loc[last_idx, "vrp"]
    vr_p = vrp_df.loc[prev_idx, "vrp"]
    m2.metric("VRP", f"{vr_l * 100:.2f}%", delta=f"{(vr_l - vr_p) * 100:.2f} p.p.")

    er_l = erp.loc[last_idx]
    er_p = erp.loc[prev_idx]
    m3.metric("ERP", f"{er_l * 100:.2f}%", delta=f"{(er_l - er_p) * 100:.2f} p.p.")

    vv_l = tail.loc[last_idx, "^VVIX"]
    vv_p = tail.loc[prev_idx, "^VVIX"]
    m4.metric("VVIX", f"{vv_l:.1f}", delta=f"{vv_l - vv_p:.1f}")

    sk_l = tail.loc[last_idx, "^SKEW"]
    sk_p = tail.loc[prev_idx, "^SKEW"]
    m5.metric("SKEW", f"{sk_l:.1f}", delta=f"{sk_l - sk_p:.1f}")

    mv_l = tail.loc[last_idx, "MOVE_VIX_RATIO"]
    mv_p = tail.loc[prev_idx, "MOVE_VIX_RATIO"]
    m6.metric("MOVE / VIX", f"{mv_l:.2f}", delta=f"{mv_l - mv_p:.2f}")

    if bool(tail.loc[last_idx, "CRITICAL_STRESS"]):
        st.error("\U0001f6a8 VIX Term Structure in Backwardation (CRITICAL STRESS).")
    if bool(tail.loc[last_idx, "VVIX_FLAG"]) or bool(tail.loc[last_idx, "SKEW_FLAG"]):
        st.warning("\u26a0\ufe0f Tail Risk flag attivo (VVIX > 110 o SKEW > 135).")

    st.markdown("---")

    # --- QUANT HEATMAP ---
    st.subheader("Quant Heatmap \u2013 Intermarket Ratios")
    if not ratios_snap.empty:
        snap = ratios_snap.copy()
        snap["percentile"] = snap["percentile"] * 100.0
        try:
            styled_snap = snap.style.background_gradient(cmap="RdYlGn", subset=["zscore"]).format(
                {"value": "{:.3f}", "zscore": "{:.2f}", "percentile": "{:.1f}%"}
            )
            st.dataframe(styled_snap, use_container_width=True)
        except ImportError:
            logger.warning("matplotlib non disponibile: visualizzazione heatmap senza gradiente.")
            st.dataframe(
                snap.style.format({"value": "{:.3f}", "zscore": "{:.2f}", "percentile": "{:.1f}%"}),
                use_container_width=True,
            )
    else:
        st.info("Heatmap non disponibile: dati insufficienti.")

    st.markdown("---")

    # --- DIAGNOSTICS ---
    with st.expander("Diagnostics & Raw Series"):
        st.write("Composite Score (ultime 60 osservazioni):")
        st.line_chart(composite_series.tail(60))
        st.write("Dynamic Allocation (equity vs bonds):")
        st.line_chart(allocation_series.tail(60))
        st.write("CUSUM g_pos / g_neg:")
        st.line_chart(cusum_df[["g_pos", "g_neg"]].tail(120))


if __name__ == "__main__":
    main()
