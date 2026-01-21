# app.py
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# requirements.txt precisa ter:
# streamlit
# pandas
# numpy
# matplotlib
# yfinance
# python-bcb
# requests
from bcb import sgs
import yfinance as yf


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="LM Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CSS (WEB3 / DASHBOARD)
# =========================
st.markdown(
    """
<style>
html, body, [data-testid="stApp"] {
  background: radial-gradient(1200px 600px at 50% 0%, rgba(0,229,255,0.08), rgba(2,6,23,1) 60%) !important;
  color: #E5E7EB !important;
}
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(2,6,23,1), rgba(2,6,23,1)) !important;
  border-right: 1px solid rgba(148,163,184,0.15);
}
h1,h2,h3,h4 { color: #E5E7EB !important; }
.small-muted { color: #94A3B8; font-size: 14px; }

.header-wrap{
  text-align:center;
  padding: 24px 0 6px 0;
}
.logo-big{
  display:flex;
  justify-content:center;
  align-items:center;
  margin-top: 4px;
  margin-bottom: 10px;
}
.logo-big svg{
  width: 160px;
  height: 160px;
  filter: drop-shadow(0 0 18px rgba(0,229,255,0.38)) drop-shadow(0 0 26px rgba(124,77,255,0.22));
}
.subline{
  margin-top: 6px;
  margin-bottom: 10px;
  color:#94A3B8;
  font-size: 14px;
}
.social a{
  text-decoration:none;
  margin: 0 10px;
  font-weight: 700;
  color: #38BDF8;
}
.social a:hover{
  color:#A78BFA;
  text-shadow: 0 0 12px rgba(167,139,250,0.25);
}
.hr{
  height:1px;
  background: rgba(148,163,184,0.18);
  margin: 18px 0 16px 0;
}

.neon-btn button{
  border: 1px solid rgba(0,229,255,0.35) !important;
  box-shadow: 0 0 18px rgba(0,229,255,0.10), 0 0 26px rgba(124,77,255,0.08) !important;
}
.neon-btn button:hover{
  border: 1px solid rgba(167,139,250,0.55) !important;
  box-shadow: 0 0 26px rgba(0,229,255,0.16), 0 0 34px rgba(124,77,255,0.14) !important;
}

.metric-card{
  background: rgba(15,23,42,0.55);
  border: 1px solid rgba(148,163,184,0.16);
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.footer-fixed{
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  background: rgba(2,6,23,0.86);
  border-top: 1px solid rgba(148,163,184,0.15);
  color: #94A3B8;
  text-align: center;
  padding: 10px 12px;
  font-size: 12.5px;
  z-index: 999;
  backdrop-filter: blur(8px);
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# LOGO SVG (APENAS ÍCONE)
# =========================
LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 220" fill="none">
  <defs>
    <linearGradient id="gStroke" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#00E5FF"/>
      <stop offset="100%" stop-color="#7C4DFF"/>
    </linearGradient>
    <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="5" result="b"/>
      <feMerge>
        <feMergeNode in="b"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <path d="M110 20 L182 62 V158 L110 200 L38 158 V62 Z"
        stroke="url(#gStroke)" stroke-width="6" filter="url(#glow)" />

  <path d="M110 40 L166 72 V148 L110 180 L54 148 V72 Z"
        stroke="rgba(124,77,255,0.35)" stroke-width="2" />

  <path d="M68 132 L92 112 L112 122 L136 92 L152 78"
        stroke="url(#gStroke)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"
        filter="url(#glow)" />

  <circle cx="68" cy="132" r="5" fill="#00E5FF" filter="url(#glow)"/>
  <circle cx="92" cy="112" r="5" fill="#00E5FF" filter="url(#glow)"/>
  <circle cx="112" cy="122" r="5" fill="#00E5FF" filter="url(#glow)"/>
  <circle cx="136" cy="92" r="5" fill="#7C4DFF" filter="url(#glow)"/>
  <circle cx="152" cy="78" r="5" fill="#7C4DFF" filter="url(#glow)"/>
</svg>
"""


# =========================
# HELPERS
# =========================
@dataclass(frozen=True)
class Params:
    anos_grafico: int
    aporte_mensal: float
    horizonte_min: int
    horizonte_max: int
    sens_base: str  # "periodo_grafico" | "fixo_6anos" | "max_10anos"
    sens_mode: str  # "final" | "janela"


def _today() -> pd.Timestamp:
    return pd.Timestamp(date.today())


def _start_from_years(years: int) -> pd.Timestamp:
    # buffer para meses completos
    return (_today() - pd.DateOffset(years=years)).normalize() - pd.DateOffset(days=7)


def _fmt_brl(x: float) -> str:
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _safe_pct(final: float, total: float) -> float:
    if total <= 0:
        return np.nan
    return (final / total - 1.0) * 100.0


# =========================
# DATA LOAD (CACHE)
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 60)  # 1h
def load_series_daily(start: pd.Timestamp) -> pd.DataFrame:
    """
    DataFrame diário:
    - BTC_USD (Yahoo)
    - USD_BRL (BCB PTAX venda) -> interpolado
    - CDI_DAILY (BCB CDI diário) -> interpolado
    """

    # ---- BTC USD (Yahoo) ----
    btc = yf.download(
        "BTC-USD",
        start=start.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if btc is None or btc.empty:
        raise RuntimeError("Não consegui baixar BTC-USD (Yahoo). Tente novamente mais tarde.")

    # ✅ FIX: yfinance pode retornar colunas MultiIndex no Streamlit Cloud
    if isinstance(btc.columns, pd.MultiIndex):
        # tenta "Close" no nível 0
        if "Close" in btc.columns.get_level_values(0):
            close_df = btc["Close"]  # geralmente vira DataFrame
            btc_close = close_df.iloc[:, 0] if isinstance(close_df, pd.DataFrame) else close_df
        else:
            btc_close = btc.iloc[:, 0]
    else:
        if "Close" in btc.columns:
            btc_close = btc["Close"]
        elif "Adj Close" in btc.columns:
            btc_close = btc["Adj Close"]
        else:
            btc_close = btc.iloc[:, 0]

    btc_usd = pd.Series(btc_close).astype(float)
    btc_usd.name = "BTC_USD"

    # ---- USD/BRL (BCB - PTAX venda) ----
    # PTAX venda USD/BRL (SGS) - em alguns ambientes pode precisar trocar o código
    usd_brl_df = sgs.get({"USD_BRL": 1}, start=start.strftime("%Y-%m-%d"))
    usd_brl = usd_brl_df["USD_BRL"].astype(float)

    # ---- CDI diário (BCB) ----
    cdi_df = sgs.get({"CDI_DAILY": 12}, start=start.strftime("%Y-%m-%d"))
    cdi = cdi_df["CDI_DAILY"].astype(float)

    df = pd.concat([btc_usd, usd_brl, cdi], axis=1).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Interpola USD_BRL e CDI para casar com o BTC (que tem finais de semana)
    df["USD_BRL"] = df["USD_BRL"].interpolate(method="time").ffill().bfill()
    df["CDI_DAILY"] = df["CDI_DAILY"].ffill().bfill()

    return df


def daily_to_monthly_last(df_daily: pd.DataFrame) -> pd.DataFrame:
    # 'M' deprecated -> use 'ME' (month end)
    return df_daily.resample("ME").last()


def compute_dca(df_monthly: pd.DataFrame, aporte_mensal: float) -> pd.DataFrame:
    """
    DCA mensal (aporte no último dia do mês), em BRL:
    - BTC: compra BTC em BRL via BTC_USD * USD_BRL
    - USD: compra USD e marcação em BRL
    - CDI: aporte + capitalização mensal aproximada
    """
    df = df_monthly.copy()

    # BTC em BRL
    df["BTC_BRL"] = df["BTC_USD"] * df["USD_BRL"]

    # Fator mensal do CDI aproximado:
    # CDI_DAILY (% ao dia) -> aproximar dias do mês usando delta do index
    cdi_daily = df["CDI_DAILY"]
    days_in_month = df.index.to_series().diff().dt.days.fillna(30).clip(lower=28, upper=31)
    df["CDI_FACTOR_M"] = (1 + (cdi_daily / 100.0)) ** days_in_month

    # BTC DCA
    btc_units = (aporte_mensal / df["BTC_BRL"]).replace([np.inf, -np.inf], np.nan).fillna(0).cumsum()
    df["DCA_BTC_BRL"] = btc_units * df["BTC_BRL"]

    # USD DCA
    usd_units = (aporte_mensal / df["USD_BRL"]).replace([np.inf, -np.inf], np.nan).fillna(0).cumsum()
    df["DCA_USD_BRL"] = usd_units * df["USD_BRL"]

    # CDI DCA
    cdi_value = []
    acc = 0.0
    for _, row in df.iterrows():
        acc = (acc + aporte_mensal) * float(row["CDI_FACTOR_M"])
        cdi_value.append(acc)
    df["DCA_CDI_BRL"] = cdi_value

    out = df[["DCA_BTC_BRL", "DCA_USD_BRL", "DCA_CDI_BRL"]].copy()
    out.columns = ["BTC", "USD", "CDI"]
    return out


def plot_dca(df_dca: pd.DataFrame, anos: int, aporte: float) -> Tuple[plt.Figure, Dict[str, float]]:
    fig = plt.figure(figsize=(14.5, 6.4), dpi=120)
    ax = fig.add_subplot(111)

    c_btc = "#00E5FF"
    c_usd = "#A78BFA"
    c_cdi = "#34D399"

    ax.plot(df_dca.index, df_dca["BTC"], label=f"Bitcoin (DCA {_fmt_brl(aporte)}/mês)", linewidth=2.6, color=c_btc)
    ax.plot(df_dca.index, df_dca["USD"], label=f"Dólar (DCA {_fmt_brl(aporte)}/mês)", linewidth=2.1, color=c_usd)
    ax.plot(df_dca.index, df_dca["CDI"], label=f"CDI (DCA {_fmt_brl(aporte)}/mês)", linewidth=2.1, color=c_cdi)

    ax.set_title(f"DCA Mensal — BTC vs USD vs CDI ({anos} anos)", fontsize=14, pad=12)
    ax.set_xlabel("Data")
    ax.set_ylabel("Patrimônio acumulado (R$)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    meses = len(df_dca)
    total = aporte * meses
    final_btc = float(df_dca["BTC"].iloc[-1])
    final_usd = float(df_dca["USD"].iloc[-1])
    final_cdi = float(df_dca["CDI"].iloc[-1])

    metrics = {
        "total": total,
        "final_btc": final_btc,
        "final_usd": final_usd,
        "final_cdi": final_cdi,
        "ret_btc": _safe_pct(final_btc, total),
        "ret_usd": _safe_pct(final_usd, total),
        "ret_cdi": _safe_pct(final_cdi, total),
        "meses": meses,
        "start": df_dca.index.min(),
        "end": df_dca.index.max(),
    }
    return fig, metrics


def to_pdf_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def sensitivity_returns(
    df_dca_full: pd.DataFrame,
    aporte: float,
    horizonte_min: int,
    horizonte_max: int,
    mode: str,
) -> pd.DataFrame:
    horizons = list(range(horizonte_min, horizonte_max + 1))
    results = []

    for h in horizons:
        months = h * 12
        if len(df_dca_full) < months + 1:
            results.append({"Horizonte (anos)": h, "BTC (%)": np.nan, "USD (%)": np.nan, "CDI (%)": np.nan})
            continue

        # por enquanto: "janela" = último mês também (estável e sem quebrar)
        df_slice = df_dca_full.iloc[-months:]

        total = aporte * len(df_slice)
        btc_f = float(df_slice["BTC"].iloc[-1])
        usd_f = float(df_slice["USD"].iloc[-1])
        cdi_f = float(df_slice["CDI"].iloc[-1])

        results.append(
            {
                "Horizonte (anos)": h,
                "BTC (%)": _safe_pct(btc_f, total),
                "USD (%)": _safe_pct(usd_f, total),
                "CDI (%)": _safe_pct(cdi_f, total),
            }
        )

    return pd.DataFrame(results)


def plot_sensitivity(df_sens: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(14.5, 4.9), dpi=120)
    ax = fig.add_subplot(111)

    c_btc = "#00E5FF"
    c_usd = "#A78BFA"
    c_cdi = "#34D399"

    x = df_sens["Horizonte (anos)"].values
    ax.plot(x, df_sens["BTC (%)"].values, marker="o", linewidth=2.3, color=c_btc, label="BTC (%)")
    ax.plot(x, df_sens["USD (%)"].values, marker="o", linewidth=2.1, color=c_usd, label="USD (%)")
    ax.plot(x, df_sens["CDI (%)"].values, marker="o", linewidth=2.1, color=c_cdi, label="CDI (%)")

    ax.set_title("Retorno (%) por horizonte (sensibilidade)", fontsize=13, pad=10)
    ax.set_xlabel("Horizonte (anos)")
    ax.set_ylabel("Retorno (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    return fig


# =========================
# HEADER
# =========================
st.markdown(
    f"""
<div class="header-wrap">
  <div class="logo-big">{LOGO_SVG}</div>
  <div class="subline">Research & simulações de investimento • Web3 • Dados reais</div>
  <div class="social">
    <a href="https://www.instagram.com/mikesp18/" target="_blank">Instagram</a> |
    <a href="https://www.linkedin.com/in/leandro-medina-770a64386/" target="_blank">LinkedIn</a>
  </div>
</div>
<div class="hr"></div>
""",
    unsafe_allow_html=True,
)

# =========================
# NAV
# =========================
tab = st.radio(
    "Navegação",
    ["Dashboard", "About / Metodologia"],
    horizontal=True,
    label_visibility="collapsed",
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Parâmetros")

anos_grafico = st.sidebar.slider("Período do gráfico (anos)", 1, 10, 6, 1)
aporte_mensal = st.sidebar.number_input("Aporte mensal (R$)", min_value=50.0, max_value=50000.0, value=500.0, step=50.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Sensibilidade (mutável)")

sens_base = st.sidebar.selectbox(
    "Base da sensibilidade",
    options=["Usar período do gráfico", "Fixar 6 anos", "Fixar 10 anos"],
    index=0,
)

h_min, h_max = st.sidebar.slider("Horizonte (anos) — mínimo e máximo", 1, 10, (1, 10), 1)

sens_mode = st.sidebar.radio(
    "Modo da sensibilidade",
    options=["Fixo no final (últimos N anos)", "Janela móvel (por enquanto: último mês)"],
    index=0,
)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Atualizar automaticamente", value=False)
interval_sec = st.sidebar.number_input("Intervalo (segundos)", min_value=10, max_value=3600, value=60, step=10)
btn_refresh = st.sidebar.button("Atualizar agora")

# Auto refresh simples (rebuild a cada intervalo)
if auto_refresh:
    st.cache_data.clear()
    st.experimental_rerun()

if btn_refresh:
    st.cache_data.clear()
    st.experimental_rerun()

params = Params(
    anos_grafico=int(anos_grafico),
    aporte_mensal=float(aporte_mensal),
    horizonte_min=int(h_min),
    horizonte_max=int(h_max),
    sens_base={"Usar período do gráfico": "periodo_grafico", "Fixar 6 anos": "fixo_6anos", "Fixar 10 anos": "max_10anos"}[sens_base],
    sens_mode={"Fixo no final (últimos N anos)": "final", "Janela móvel (por enquanto: último mês)": "janela"}[sens_mode],
)

# =========================
# PAGES
# =========================
if tab == "About / Metodologia":
    st.header("About / Metodologia")

    st.markdown(
        """
### O que este dashboard faz
Este app simula um **DCA mensal** (aporte fixo todo mês) e compara o patrimônio acumulado em:
- **Bitcoin (BTC)** — em BRL via BTC-USD (Yahoo) × USD/BRL (PTAX BCB)
- **Dólar (USD)** — compra mensal de USD e marcação em BRL pela PTAX
- **CDI** — simulação com capitalização mensal aproximada a partir do CDI diário (BCB)

### Fontes de dados
- **BTC-USD:** Yahoo Finance (yfinance)  
- **USD/BRL (PTAX venda):** Banco Central do Brasil (SGS)  
- **CDI diário:** Banco Central do Brasil (SGS)

### Observações importantes
- Objetivo **educacional** (comparação e simulação).
- Não inclui impostos, taxas, spread de câmbio, IOF ou custos operacionais.
- O CDI é aproximado por fator mensal (boa aproximação visual e comparativa).

**Sem recomendação de investimento.**
"""
    )

else:
    # =========================
    # DASHBOARD
    # =========================
    try:
        start = _start_from_years(params.anos_grafico)
        df_daily = load_series_daily(start)
        df_m = daily_to_monthly_last(df_daily)
        df_dca = compute_dca(df_m, params.aporte_mensal)

        fig_main, metrics = plot_dca(df_dca, params.anos_grafico, params.aporte_mensal)

        # Sensibilidade base
        sens_years = params.anos_grafico if params.sens_base == "periodo_grafico" else (6 if params.sens_base == "fixo_6anos" else 10)
        start_sens = _start_from_years(sens_years)
        df_daily_s = load_series_daily(start_sens)
        df_m_s = daily_to_monthly_last(df_daily_s)
        df_dca_s = compute_dca(df_m_s, params.aporte_mensal)

        df_sens = sensitivity_returns(df_dca_s, params.aporte_mensal, params.horizonte_min, params.horizonte_max, params.sens_mode)
        fig_sens = plot_sensitivity(df_sens)

        col_plot, col_metrics = st.columns([1.65, 1.0], gap="large")

        with col_plot:
            st.pyplot(fig_main, clear_figure=False)

            pdf_bytes = to_pdf_bytes(fig_main)
            st.markdown('<div class="neon-btn">', unsafe_allow_html=True)
            st.download_button(
                "📄 Exportar gráfico em PDF",
                data=pdf_bytes,
                file_name=f"lm_analytics_dca_{params.anos_grafico}anos.pdf",
                mime="application/pdf",
                use_container_width=False,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("#### Sensibilidade — Retorno (%) por horizonte")
            st.pyplot(fig_sens, clear_figure=False)

            with st.expander("Ver tabela da sensibilidade"):
                st.dataframe(df_sens, use_container_width=True)

        with col_metrics:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### Resumo (período selecionado)")

            st.metric("Total aportado", _fmt_brl(metrics["total"]))
            st.metric("Final BTC", _fmt_brl(metrics["final_btc"]), f"{metrics['ret_btc']:.2f}%")
            st.metric("Final USD", _fmt_brl(metrics["final_usd"]), f"{metrics['ret_usd']:.2f}%")
            st.metric("Final CDI", _fmt_brl(metrics["final_cdi"]), f"{metrics['ret_cdi']:.2f}%")

            st.markdown(
                f"""
<div class="small-muted" style="margin-top:10px;">
Meses simulados: <b>{metrics['meses']}</b><br/>
De: <b>{metrics['start'].date()}</b> Até: <b>{metrics['end'].date()}</b><br/>
Obs.: BTC em BRL = BTC-USD × USD/BRL (BCB). CDI é índice diário oficial (BCB) aproximado.
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Ocorreu um erro ao gerar os dados/gráficos.")
        st.exception(e)

# =========================
# FOOTER FIXO
# =========================
st.markdown(
    """
<div class="footer-fixed">
© 2026 • LM Analytics — Dados públicos • Sem recomendação de investimento • v1.0
</div>
""",
    unsafe_allow_html=True,
)
