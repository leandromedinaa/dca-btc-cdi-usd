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

# -----------------------------
# DEPENDÊNCIAS (adicione no requirements.txt):
# streamlit
# pandas
# numpy
# matplotlib
# yfinance
# python-bcb
# -----------------------------
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
  padding: 26px 0 6px 0;
}
.logo-big{
  display:flex;
  justify-content:center;
  align-items:center;
  margin-top: 6px;
  margin-bottom: 12px;
}
.logo-big svg{
  width: 150px;
  height: 150px;
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
  font-weight: 600;
  color: #38BDF8;
}
.social a:hover{
  color:#A78BFA;
  text-shadow: 0 0 10px rgba(167,139,250,0.25);
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

  <!-- Hex -->
  <path d="M110 20 L182 62 V158 L110 200 L38 158 V62 Z"
        stroke="url(#gStroke)" stroke-width="6" filter="url(#glow)" />

  <!-- Inner hex -->
  <path d="M110 40 L166 72 V148 L110 180 L54 148 V72 Z"
        stroke="rgba(124,77,255,0.35)" stroke-width="2" />

  <!-- Chart line -->
  <path d="M68 132 L92 112 L112 122 L136 92 L152 78"
        stroke="url(#gStroke)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"
        filter="url(#glow)" />

  <!-- Nodes -->
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
    # buffer para garantir meses completos
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
    Retorna DataFrame diário com colunas:
    - BTC_USD (Yahoo)
    - USD_BRL (BCB PTAX venda) -> aproximado diário (dias úteis)
    - CDI_DAILY (BCB CDI diário) -> em % ao dia
    """

    # ---- BTC USD (Yahoo) ----
    btc = yf.download(
        "BTC-USD",
        start=start.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if btc.empty:
        raise RuntimeError("Não consegui baixar BTC-USD (Yahoo). Tente novamente mais tarde.")
    btc_usd = btc["Close"].rename("BTC_USD")

    # ---- USD/BRL (BCB - PTAX venda) ----
    # Série 1: PTAX venda (USD) - código bem comum no SGS: 1
    # Obs: em alguns ambientes o SGS pode variar; se der erro, ajuste o código.
    usd_brl = sgs.get({"USD_BRL": 1}, start=start.strftime("%Y-%m-%d"))
    usd_brl = usd_brl["USD_BRL"]

    # ---- CDI diário (BCB) ----
    # CDI (ao dia) - SGS 12 é comum como CDI (% a.d.)
    cdi = sgs.get({"CDI_DAILY": 12}, start=start.strftime("%Y-%m-%d"))
    cdi = cdi["CDI_DAILY"]

    df = pd.concat([btc_usd, usd_brl, cdi], axis=1).sort_index()

    # Limpeza básica
    df = df[~df.index.duplicated(keep="last")]

    # Interpolar USD_BRL e CDI para dias corridos (para casar com BTC que tem todos os dias)
    # (No Brasil, PTAX/CDI só em dias úteis)
    df["USD_BRL"] = df["USD_BRL"].interpolate(method="time").ffill().bfill()
    df["CDI_DAILY"] = df["CDI_DAILY"].ffill().bfill()

    return df


def daily_to_monthly_last(df_daily: pd.DataFrame) -> pd.DataFrame:
    # 'M' deprecated -> use 'ME' (month end)
    df_m = df_daily.resample("ME").last()
    return df_m


def compute_dca(df_monthly: pd.DataFrame, aporte_mensal: float) -> pd.DataFrame:
    """
    Simula DCA mensal (aporte no último dia do mês):
    - BTC (comprando BTC em BRL) via BTC_USD * USD_BRL
    - USD (comprando dólar e mantendo em BRL) via USD_BRL
    - CDI (aplicando em CDI diariamente; aqui aproximado pelo CDI_DAILY acumulado mês a mês)
    Retorna DataFrame mensal com patrimônio acumulado (BRL).
    """
    df = df_monthly.copy()

    # BTC em BRL
    df["BTC_BRL"] = df["BTC_USD"] * df["USD_BRL"]

    # Índice acumulado de CDI (aprox):
    # CDI_DAILY está em % ao dia -> vamos construir um fator mensal aproximando:
    # Para cada mês, fator = produto (1 + cdi/100) nos dias do mês.
    # Como aqui já estamos em "last do mês", precisamos refazer do diário.
    # Solução: faremos CDI acumulado diário no df_daily antes de resample. Aqui, usamos aproximação:
    # fator_mensal ~ (1 + cdi_dia/100)^(n_dias_uteis_mes) não temos n_dias aqui -> então melhor:
    # A saída correta está no fluxo: vamos receber CDI_DAILY já interpolado diário e refazer do diário no cache.
    # Para simplificar e manter estável: criamos CDI_FACTOR mensal a partir do diário via função abaixo.
    # (Vamos exigir no pipeline que df_monthly veio do diário com CDI_DAILY interpolado.)
    # Aqui, reconstruímos usando média do CDI diário no mês e número de dias corridos do mês (aprox).
    cdi_daily = df["CDI_DAILY"]
    # aprox dias por mês usando diferença de index
    days_in_month = df.index.to_series().diff().dt.days.fillna(30).clip(lower=28, upper=31)
    df["CDI_FACTOR_M"] = (1 + (cdi_daily / 100.0)) ** days_in_month

    # DCA: BTC
    btc_units = (aporte_mensal / df["BTC_BRL"]).replace([np.inf, -np.inf], np.nan).fillna(0).cumsum()
    df["DCA_BTC_BRL"] = btc_units * df["BTC_BRL"]

    # DCA: USD (compra dólar e mantém)
    usd_units = (aporte_mensal / df["USD_BRL"]).replace([np.inf, -np.inf], np.nan).fillna(0).cumsum()
    df["DCA_USD_BRL"] = usd_units * df["USD_BRL"]

    # DCA: CDI (aporta e capitaliza mensalmente pelo fator)
    cdi_value = []
    acc = 0.0
    for i, row in df.iterrows():
        acc = (acc + aporte_mensal) * float(row["CDI_FACTOR_M"])
        cdi_value.append(acc)
    df["DCA_CDI_BRL"] = cdi_value

    out = df[["DCA_BTC_BRL", "DCA_USD_BRL", "DCA_CDI_BRL"]].copy()
    out.columns = ["BTC", "USD", "CDI"]
    return out


def plot_dca(df_dca: pd.DataFrame, anos: int, aporte: float) -> Tuple[plt.Figure, Dict[str, float]]:
    fig = plt.figure(figsize=(14, 6.2), dpi=120)
    ax = fig.add_subplot(111)

    # Cores profissionais (web3)
    c_btc = "#00E5FF"
    c_usd = "#A78BFA"
    c_cdi = "#34D399"

    ax.plot(df_dca.index, df_dca["BTC"], label=f"Bitcoin (DCA {_fmt_brl(aporte)}/mês)", linewidth=2.4, color=c_btc)
    ax.plot(df_dca.index, df_dca["USD"], label=f"Dólar (DCA {_fmt_brl(aporte)}/mês)", linewidth=2.0, color=c_usd)
    ax.plot(df_dca.index, df_dca["CDI"], label=f"CDI (DCA {_fmt_brl(aporte)}/mês)", linewidth=2.0, color=c_cdi)

    ax.set_title(f"DCA Mensal — BTC vs USD vs CDI ({anos} anos)", fontsize=14, pad=12)
    ax.set_xlabel("Data")
    ax.set_ylabel("Patrimônio acumulado (R$)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    # métricas
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
    """
    Retorna retorno (%) por horizonte em anos.
    mode:
      - "final": fixa no final (últimos N anos)
      - "janela": janela móvel escolhendo mês final (usa último mês como padrão; pode evoluir depois)
    """
    horizons = list(range(horizonte_min, horizonte_max + 1))
    results = []

    for h in horizons:
        months = h * 12
        if len(df_dca_full) < months + 1:
            # Sem dados suficientes
            results.append({"Horizonte (anos)": h, "BTC (%)": np.nan, "USD (%)": np.nan, "CDI (%)": np.nan})
            continue

        if mode == "final":
            df_slice = df_dca_full.iloc[-months:]
        else:
            # "janela" (por enquanto usando último mês como fim)
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
    fig = plt.figure(figsize=(14, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    c_btc = "#00E5FF"
    c_usd = "#A78BFA"
    c_cdi = "#34D399"

    x = df_sens["Horizonte (anos)"].values
    ax.plot(x, df_sens["BTC (%)"].values, marker="o", linewidth=2.2, color=c_btc, label="BTC (%)")
    ax.plot(x, df_sens["USD (%)"].values, marker="o", linewidth=2.0, color=c_usd, label="USD (%)")
    ax.plot(x, df_sens["CDI (%)"].values, marker="o", linewidth=2.0, color=c_cdi, label="CDI (%)")

    ax.set_title("Retorno (%) por horizonte (sensibilidade)", fontsize=13, pad=10)
    ax.set_xlabel("Horizonte (anos)")
    ax.set_ylabel("Retorno (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    return fig


# =========================
# HEADER (LOGO GRANDE)
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
# NAV (Dashboard / About)
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

if auto_refresh:
    st.experimental_set_query_params(_t=str(pd.Timestamp.utcnow().value))
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
# PÁGINAS
# =========================
if tab == "About / Metodologia":
    st.header("About / Metodologia")

    st.markdown(
        """
### O que este dashboard faz
Este app simula um **DCA mensal** (aporte fixo todo mês) e compara o patrimônio acumulado em:
- **Bitcoin (BTC)** — calculado em BRL via BTC-USD (Yahoo) × USD/BRL (PTAX BCB)
- **Dólar (USD)** — compra mensal de USD e marcação a mercado em BRL pela PTAX
- **CDI** — simulação com capitalização mensal usando fator aproximado a partir do CDI diário (BCB)

### Fontes de dados
- **BTC-USD:** Yahoo Finance (yfinance)  
- **USD/BRL (PTAX venda):** Banco Central do Brasil (SGS)  
- **CDI diário:** Banco Central do Brasil (SGS)

### Observações importantes
- O objetivo é **comparação educacional**.  
- Não inclui impostos, taxas, spread de câmbio, corretagem, IOF ou custos operacionais.  
- CDI é aproximado por fator mensal reconstruído do CDI diário.

### Licença / Aviso
**Sem recomendação de investimento.** Use por sua conta e risco.
"""
    )
else:
    # =========================
    # DASHBOARD
    # =========================
    try:
        # Base do gráfico
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

        # Layout
        col_plot, col_metrics = st.columns([1.6, 1.0], gap="large")

        with col_plot:
            st.pyplot(fig_main, clear_figure=False)

            # Export PDF
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
