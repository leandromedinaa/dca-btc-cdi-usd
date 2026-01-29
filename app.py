import os
import io
import time
import math
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import yfinance as yf
import streamlit as st

# ============================================================
# LM Analytics — Simulações
# Single-file Streamlit app (Dashboard + Carteira + Metodologia)
# - DCA BTC vs CDI vs USD (BRL/USD, nominal/real)
# - CPI/FED via FRED (optional, graceful fallback if no key)
# - Risk metrics (CAGR, Vol, Sharpe, Max DD)
# - Portfolio simulator with rebalance
# - Exports: PDF + CSV + Excel
# ============================================================

APP_TITLE = "LM Analytics — Simulações"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# ---------------------------
# Helpers: formatting
# ---------------------------
def fmt_money(x: float, unit: str) -> str:
    try:
        return f"{x:,.2f} {unit}".replace(" ,", ",")
    except Exception:
        return f"{x} {unit}"

def fmt_pct(x: float) -> str:
    try:
        return f"{x:.2f}%"
    except Exception:
        return f"{x}%"

# ---------------------------
# Theme (simple, stable)
# ---------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 4.0rem; max-width: 1400px; }
small, .stCaption { opacity: .75; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# DATA SOURCES
# ============================================================

@st.cache_data(ttl=60 * 60)
def baixar_btc_usd(days: int = 3650) -> pd.DataFrame:
    df = yf.download("BTC-USD", period=f"{max(1, days//365)}y", interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError("Falha ao baixar BTC-USD do Yahoo Finance.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "BTC_USD"})
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(ttl=24 * 60 * 60)
def sgs_series(serie: int, data_inicial: str, data_final: str) -> pd.DataFrame:
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados"
    params = {"formato": "json", "dataInicial": data_inicial, "dataFinal": data_final}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        raise RuntimeError(f"Falha ao baixar série SGS {serie}.")
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["valor"]).set_index("data")[["valor"]].sort_index()
    return df

@st.cache_data(ttl=24 * 60 * 60)
def baixar_usd_brl(data_inicial: str, data_final: str) -> pd.DataFrame:
    # SGS 1: USD/BRL (PTAX)
    df = sgs_series(1, data_inicial, data_final).rename(columns={"valor": "USD_BRL"})
    return df

@st.cache_data(ttl=24 * 60 * 60)
def baixar_cdi(data_inicial: str, data_final: str) -> pd.DataFrame:
    # SGS 12: CDI diária (% a.d.)
    df = sgs_series(12, data_inicial, data_final).rename(columns={"valor": "cdi_pct"})
    df["cdi_pct"] = df["cdi_pct"] / 100.0
    df["CDI"] = (1.0 + df["cdi_pct"]).cumprod()
    return df[["CDI"]]

@st.cache_data(ttl=24 * 60 * 60)
def baixar_ipca(data_inicial: str, data_final: str) -> pd.DataFrame:
    # SGS 433: IPCA - número índice (dez/1993 = 100)
    df = sgs_series(433, data_inicial, data_final).rename(columns={"valor": "IPCA"})
    return df[["IPCA"]]

# ---------------------------
# FRED (optional)
# ---------------------------
def _get_fred_api_key() -> str | None:
    try:
        return st.secrets.get("FRED_API_KEY", None)
    except Exception:
        return None

def _fred_request_observations(series_id: str, start_iso: str, end_iso: str, api_key: str) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_iso,
        "observation_end": end_iso,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    obs = payload.get("observations", [])
    if not obs:
        return pd.DataFrame(columns=[series_id])
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # '.' -> NaN
    df = df.dropna(subset=["value"]).set_index("date")[["value"]]
    return df.rename(columns={"value": series_id}).sort_index()

@st.cache_data(ttl=24 * 60 * 60)
def baixar_cpi_fred(start_iso: str, end_iso: str) -> pd.DataFrame:
    api_key = _get_fred_api_key() or os.getenv("FRED_API_KEY")
    if not api_key:
        return pd.DataFrame(columns=["CPIAUCSL"])
    return _fred_request_observations("CPIAUCSL", start_iso, end_iso, api_key)

@st.cache_data(ttl=24 * 60 * 60)
def baixar_fedfunds_fred(start_iso: str, end_iso: str) -> pd.DataFrame:
    api_key = _get_fred_api_key() or os.getenv("FRED_API_KEY")
    if not api_key:
        return pd.DataFrame(columns=["FEDFUNDS"])
    return _fred_request_observations("FEDFUNDS", start_iso, end_iso, api_key)

# ---------------------------
# CoinGecko (crypto universe)
# ---------------------------
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Lista "pronta" (ids oficiais do CoinGecko)
COINGECKO_PRESETS = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Solana (SOL)": "solana",
    "Chainlink (LINK)": "chainlink",
    "Avalanche (AVAX)": "avalanche-2",
    "Polygon (MATIC)": "matic-network",
    "Aave (AAVE)": "aave",
    "Lido (LDO)": "lido-dao",
    "Ondo (ONDO)": "ondo-finance",
}

@st.cache_data(ttl=60 * 60)
def cg_ping() -> bool:
    try:
        r = requests.get(f"{COINGECKO_BASE}/ping", timeout=20)
        return r.ok
    except Exception:
        return False

@st.cache_data(ttl=30 * 60)
def cg_global() -> dict:
    r = requests.get(f"{COINGECKO_BASE}/global", timeout=30)
    r.raise_for_status()
    return r.json().get("data", {})

@st.cache_data(ttl=15 * 60)
def cg_markets(vs_currency: str = "usd", per_page: int = 50, page: int = 1) -> pd.DataFrame:
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": "false",
        "price_change_percentage": "24h,7d",
    }
    r = requests.get(f"{COINGECKO_BASE}/coins/markets", params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    keep = [
        "id","symbol","name","current_price","market_cap","total_volume",
        "price_change_percentage_24h","price_change_percentage_7d_in_currency",
    ]
    return df[[c for c in keep if c in df.columns]]

@st.cache_data(ttl=30 * 60)
def cg_search(query: str) -> pd.DataFrame:
    r = requests.get(f"{COINGECKO_BASE}/search", params={"query": query}, timeout=30)
    r.raise_for_status()
    coins = r.json().get("coins", [])
    df = pd.DataFrame(coins)
    if df.empty:
        return df
    keep = ["id","name","symbol","market_cap_rank"]
    return df[[c for c in keep if c in df.columns]]

@st.cache_data(ttl=60 * 60)
def cg_market_chart_range(coin_id: str, vs_currency: str, start_ts: int, end_ts: int) -> pd.Series:
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart/range"
    params = {"vs_currency": vs_currency, "from": start_ts, "to": end_ts}
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    prices = r.json().get("prices", [])
    if not prices:
        return pd.Series(dtype=float)
    s = pd.Series({pd.to_datetime(int(t), unit="ms"): float(v) for t, v in prices}).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s

@st.cache_data(ttl=60 * 60)
def cg_monthly_close_usd(coin_id: str, start_dt: dt.date, end_dt: dt.date) -> pd.Series:
    start_ts = int(dt.datetime.combine(start_dt, dt.time.min).timestamp())
    end_ts = int(dt.datetime.combine(end_dt, dt.time.max).timestamp())
    daily = cg_market_chart_range(coin_id, "usd", start_ts, end_ts)
    if daily.empty:
        return pd.Series(dtype=float, name=coin_id)
    m = daily.resample("ME").last().ffill()
    m.name = coin_id
    return m

@st.cache_data(ttl=15 * 60)
def cg_simple_price(coin_id: str, vs_currencies=("usd","brl")) -> dict:
    params = {"ids": coin_id, "vs_currencies": ",".join(vs_currencies), "include_24hr_change": "true"}
    r = requests.get(f"{COINGECKO_BASE}/simple/price", params=params, timeout=30)
    r.raise_for_status()
    return r.json().get(coin_id, {})


# ============================================================
# TRANSFORMS
# ============================================================

def to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index().resample("ME").last().ffill()
    return out

def build_macro(df_m: pd.DataFrame, br_ini: str, br_fim: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    idx = df_m.index

    ipca = to_month_end(baixar_ipca(br_ini, br_fim)).reindex(idx).ffill().bfill()
    # CPI + FED optional
    cpi_raw = to_month_end(baixar_cpi_fred(start_iso, end_iso)).reindex(idx).ffill().bfill()
    fed_raw = to_month_end(baixar_fedfunds_fred(start_iso, end_iso)).reindex(idx).ffill().bfill()

    # Fallbacks if empty (keeps app alive)
    if ipca.empty or ipca["IPCA"].isna().all():
        raise RuntimeError("IPCA (SGS 433) sem dados no intervalo.")
    if cpi_raw.empty or "CPIAUCSL" not in cpi_raw.columns or cpi_raw["CPIAUCSL"].isna().all():
        cpi_raw = pd.DataFrame(index=idx, data={"CPIAUCSL": 100.0})
    if fed_raw.empty or "FEDFUNDS" not in fed_raw.columns or fed_raw["FEDFUNDS"].isna().all():
        fed_raw = pd.DataFrame(index=idx, data={"FEDFUNDS": 0.0})

    # Indexes base 1
    ipca_lvl = ipca["IPCA"].astype(float)
    ipca_index = (ipca_lvl / float(ipca_lvl.iloc[0])).rename("IPCA_INDEX")

    cpi_lvl = cpi_raw["CPIAUCSL"].astype(float)
    cpi_index = (cpi_lvl / float(cpi_lvl.iloc[0])).rename("CPI_INDEX")

    fed_pct = fed_raw["FEDFUNDS"].astype(float)  # % a.a.
    fed_annual = (fed_pct / 100.0).clip(lower=-0.9999)
    fed_m = (1.0 + fed_annual).pow(1.0 / 12.0) - 1.0
    fed_index = (1.0 + fed_m).cumprod()
    fed_index = (fed_index / float(fed_index.iloc[0])).rename("FED_INDEX")

    df_macro = pd.concat(
        [
            ipca_lvl.rename("IPCA"),
            cpi_lvl.rename("CPIAUCSL"),
            fed_pct.rename("FEDFUNDS"),
            ipca_index,
            cpi_index,
            fed_index,
        ],
        axis=1,
    ).reindex(idx).ffill().bfill()

    return df_macro

def build_prices(df_m: pd.DataFrame, df_macro: pd.DataFrame, base: str, include_fed: bool) -> tuple[pd.DataFrame, str]:
    df = df_m.join(df_macro[["IPCA_INDEX", "CPI_INDEX", "FED_INDEX"]], how="left").ffill().bfill()
    fx = df["USD_BRL"].astype(float)
    ipca = df["IPCA_INDEX"].astype(float)
    cpi = df["CPI_INDEX"].astype(float)
    fed = df["FED_INDEX"].astype(float)

    if base == "BRL nominal":
        prices = pd.DataFrame({"BTC": df["BTC_BRL"], "USD": df["USD_BRL"], "CDI": df["CDI"]}, index=df.index)
        if include_fed:
            prices["FED (USD+juros)"] = fed * fx
        return prices, "R$"

    if base == "BRL real (IPCA)":
        prices = pd.DataFrame(
            {"BTC": df["BTC_BRL"] / ipca, "USD": df["USD_BRL"] / ipca, "CDI": df["CDI"] / ipca},
            index=df.index,
        )
        if include_fed:
            prices["FED (USD+juros)"] = (fed * fx) / ipca
        return prices, "R$ (real)"

    if base == "USD nominal":
        prices = pd.DataFrame({"BTC": df["BTC_USD"], "USD": 1.0, "CDI": df["CDI"] / fx}, index=df.index)
        if include_fed:
            prices["FED (USD+juros)"] = fed
        return prices, "USD"

    if base == "USD real (CPI)":
        prices = pd.DataFrame(
            {"BTC": df["BTC_USD"] / cpi, "USD": 1.0 / cpi, "CDI": (df["CDI"] / fx) / cpi},
            index=df.index,
        )
        if include_fed:
            prices["FED (USD+juros)"] = fed / cpi
        return prices, "USD (real)"

    raise ValueError("Base inválida.")

# ============================================================
# SIMULATION CORE
# ============================================================

def simular_dca(precos: pd.Series, aportes: float | pd.Series, valor_inicial: float = 0.0) -> pd.Series:
    precos = precos.astype(float)
    if isinstance(aportes, pd.Series):
        ap = aportes.reindex(precos.index).fillna(0.0).astype(float)
    else:
        ap = pd.Series(float(aportes), index=precos.index, dtype=float)

    cotas = 0.0
    initialized = False
    valores = []
    last_val = 0.0
    for dt0, p in precos.items():
        p = float(p) if pd.notna(p) else float('nan')
        if (not initialized) and valor_inicial > 0 and p > 0:
            cotas += float(valor_inicial) / p
            initialized = True
        a = float(ap.loc[dt0])
        if p > 0 and a > 0:
            cotas += a / p
        if not np.isfinite(p) or p <= 0:
            valores.append(last_val)
            continue
        last_val = cotas * p
        valores.append(last_val)
    return pd.Series(valores, index=precos.index)

def calc_dca(df_prices: pd.DataFrame, aporte: float | pd.Series, valor_inicial: float = 0.0) -> pd.DataFrame:
    out = pd.DataFrame(index=df_prices.index)
    for c in df_prices.columns:
        out[c] = simular_dca(df_prices[c], aporte, valor_inicial)
    return out

def linha_aportes(index: pd.DatetimeIndex, aporte: float | pd.Series) -> pd.Series:
    if isinstance(aporte, pd.Series):
        s = aporte.reindex(index).fillna(0.0).astype(float)
        return s.cumsum()
    return pd.Series(float(aporte), index=index).cumsum()

def portfolio_dca(
    df_prices: pd.DataFrame,
    aporte: float | pd.Series,
    weights: dict[str, float],
    rebalance: str = "Nunca",
    valor_inicial: float = 0.0,
) -> pd.Series:
    cols = [c for c in df_prices.columns if c in weights and weights[c] > 0]
    if not cols:
        return pd.Series(index=df_prices.index, dtype=float)

    w = np.array([weights[c] for c in cols], dtype=float)
    w = w / w.sum()

    # Shares per asset
    shares = np.zeros(len(cols), dtype=float)
    values = []

    # aplica valor inicial como compra no primeiro ponto (lump sum) proporcional aos pesos
    if float(valor_inicial) > 0:
        p0 = df_prices.loc[df_prices.index[0], cols].astype(float).values
        buy0 = (float(valor_inicial) * w) / p0
        buy0 = np.where(np.isfinite(buy0), buy0, 0.0)
        shares += buy0

    # rebalance schedule
    def should_rebalance(ts: pd.Timestamp) -> bool:
        if rebalance == "Nunca":
            return False
        if rebalance == "Anual":
            return ts.month == 12  # end-year month in ME
        if rebalance == "Semestral":
            return ts.month in (6, 12)
        return False

    # aportes series
    if isinstance(aporte, pd.Series):
        ap = aporte.reindex(df_prices.index).fillna(0.0).astype(float)
    else:
        ap = pd.Series(float(aporte), index=df_prices.index, dtype=float)

    for i, ts in enumerate(df_prices.index):
        p = df_prices.loc[ts, cols].astype(float).values
        total_val = float(np.sum(shares * p))
        a = float(ap.loc[ts])

        # add contributions according to target weights
        if a > 0:
            buy = (a * w) / p
            buy = np.where(np.isfinite(buy), buy, 0.0)
            shares += buy
            total_val = float(np.sum(shares * p))

        # rebalance at schedule (after contributions)
        if should_rebalance(ts) and total_val > 0:
            target_vals = total_val * w
            shares = np.where(p > 0, target_vals / p, shares)

        values.append(float(np.sum(shares * p)))

    return pd.Series(values, index=df_prices.index, name="CARTEIRA")

# ============================================================
# RISK METRICS
# ============================================================

def max_drawdown(series: pd.Series) -> float:
    s = series.astype(float)
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return float(dd.min())

def cagr(series: pd.Series) -> float:
    s = series.dropna().astype(float)
    if len(s) < 2:
        return float("nan")
    years = (s.index[-1] - s.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1.0)

def ann_vol(series: pd.Series) -> float:
    s = series.astype(float).pct_change().dropna()
    if s.empty:
        return float("nan")
    return float(s.std() * math.sqrt(12))  # monthly series

def sharpe(series: pd.Series, rf_annual: float = 0.0) -> float:
    r = series.astype(float).pct_change().dropna()
    if r.empty:
        return float("nan")
    rf_m = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0
    excess = r - rf_m
    if excess.std() == 0 or np.isnan(excess.std()):
        return float("nan")
    return float((excess.mean() / excess.std()) * math.sqrt(12))

def metrics_table(df_dca: pd.DataFrame, unit: str, rf_annual: float = 0.0) -> pd.DataFrame:
    rows = []
    for col in df_dca.columns:
        s = df_dca[col].astype(float)
        rows.append(
            {
                "Ativo": col,
                "Valor Final": s.iloc[-1],
                "CAGR": cagr(s),
                "Vol (a.a.)": ann_vol(s),
                "Sharpe": sharpe(s, rf_annual=rf_annual),
                "Max Drawdown": max_drawdown(s),
            }
        )
    out = pd.DataFrame(rows).set_index("Ativo")
    return out

# ============================================================
# PLOTTING
# ============================================================

HALVINGS = [
    dt.date(2012, 11, 28),
    dt.date(2016, 7, 9),
    dt.date(2020, 5, 11),
    dt.date(2024, 4, 20),
]

def plot_lines(df: pd.DataFrame, title: str, unit: str, show_halvings: bool, show_aportes: bool, aporte_line: pd.Series | None):
    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=140)
    for c in df.columns:
        ax.plot(df.index, df[c], label=c, linewidth=2.2)

    if show_aportes and aporte_line is not None:
        ax.plot(aporte_line.index, aporte_line.values, label="Aportes (sem rendimento)", linewidth=1.8, linestyle="--", alpha=0.9)

    if show_halvings:
        for d in HALVINGS:
            ts = pd.Timestamp(d)
            if df.index.min() <= ts <= df.index.max():
                ax.axvline(ts, linewidth=1.2, alpha=0.35)
                ax.text(ts, ax.get_ylim()[1], "Halving", rotation=90, va="top", ha="right", alpha=0.5)

    ax.set_title(title)
    ax.set_ylabel(f"Patrimônio ({unit})")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig

def export_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Exporta múltiplas abas para Excel.

    No Streamlit Cloud, o pacote `openpyxl` pode não estar instalado se não estiver no requirements.txt.
    Esta função tenta:
      1) openpyxl
      2) xlsxwriter
    Se nenhum estiver disponível, levanta ImportError para a UI tratar com mensagem amigável.
    """
    buf = io.BytesIO()

    engine = None
    try:
        import openpyxl  # noqa: F401
        engine = "openpyxl"
    except Exception:
        try:
            import xlsxwriter  # noqa: F401
            engine = "xlsxwriter"
        except Exception as e:
            raise ImportError(
                "Para exportar Excel, instale 'openpyxl' (recomendado) ou 'xlsxwriter' no requirements.txt."
            ) from e

    with pd.ExcelWriter(buf, engine=engine) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=str(name)[:31])

    buf.seek(0)
    return buf.read()


@st.cache_data(ttl=60*60)
def carregar_base():
    """Carrega e prepara a base mensal completa (independente do período escolhido)."""
    df_btc = baixar_btc_usd(3650)  # ~10y
    br_ini = df_btc.index.min().strftime("%d/%m/%Y")
    br_fim = df_btc.index.max().strftime("%d/%m/%Y")
    start_iso = df_btc.index.min().strftime("%Y-%m-%d")
    end_iso = df_btc.index.max().strftime("%Y-%m-%d")

    df_usd = baixar_usd_brl(br_ini, br_fim)
    df_cdi = baixar_cdi(br_ini, br_fim)

    df_all = df_btc.join(df_usd, how="inner").join(df_cdi, how="inner")
    df_all["BTC_BRL"] = df_all["BTC_USD"] * df_all["USD_BRL"]
    df_all = df_all[["BTC_USD", "BTC_BRL", "USD_BRL", "CDI"]].dropna()

    df_m_full = to_month_end(df_all).dropna()
    if len(df_m_full) < 24:
        raise RuntimeError("Base mensal muito curta. Não há dados suficientes para simular.")

    df_macro_full = build_macro(df_m_full, br_ini, br_fim, start_iso, end_iso)
    return df_m_full, df_macro_full

# ============================================================
# Carrega base completa (antes da sidebar para datas min/max)
# ============================================================
try:
    df_m_full, df_macro_full = carregar_base()
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

# ============================================================
# UI — SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Atalhos")
    preset = st.selectbox("Preset rápido", ["Personalizado", "Conservador", "Moderado", "Agressivo"], index=0)

    st.divider()
    presentation = st.checkbox("Modo apresentação (esconder sidebar)", value=False)
    show_aporte_line = st.checkbox("Mostrar linha de aportes (sem rendimento)", value=True)
    show_halvings = st.checkbox("Marcar halvings do BTC", value=False)

    st.divider()
    st.header("Parâmetros")
    modo_periodo = st.radio("Período da simulação", ["Últimos N anos", "Intervalo personalizado"], index=0)
    anos_plot = st.slider("Últimos N anos", 1, 15, 6, 1, disabled=(modo_periodo!="Últimos N anos"))
    # Campos de data são ajustados depois que a base é carregada (min/max reais)
    data_inicio = st.date_input("Data inicial (personalizado)", value=dt.date.today(), disabled=(modo_periodo!="Intervalo personalizado"))
    data_fim = st.date_input("Data final (personalizado)", value=dt.date.today(), disabled=(modo_periodo!="Intervalo personalizado"))
    aporte = st.number_input("Aporte mensal (base selecionada)", min_value=0.0, value=300.0, step=50.0, format="%.2f")
    valor_inicial = st.number_input("Valor inicial do investimento", min_value=0.0, value=0.0, step=100.0, format="%.2f")
    base = st.selectbox("Base", ["BRL nominal", "BRL real (IPCA)", "USD nominal", "USD real (CPI)"], index=0)

    include_fed = st.checkbox("Mostrar benchmark USD + juros FED", value=True)
    st.caption("Para CPI/FED: defina FRED_API_KEY (Secrets ou env).")
    st.divider()
    st.header("CoinGecko")
    cg_on = st.checkbox("Ativar CoinGecko (altcoins / visão do mercado)", value=True)
    cg_coins = st.multiselect("Criptos para comparar (CoinGecko)", options=list(COINGECKO_PRESETS.keys()), default=[])
    cg_custom = st.text_input("Coin ID custom (opcional, ex: pepe, render-token)", value="")

    st.divider()
    st.subheader("Carteira (pesos)")
    w_btc = st.slider("BTC %", 0, 100, 50, 5)
    w_cdi = st.slider("CDI %", 0, 100, 40, 5)
    w_usd = st.slider("USD %", 0, 100, 10, 5)

    rebalance = st.selectbox("Rebalanceamento", ["Nunca", "Semestral", "Anual"], index=0)

    st.divider()
    auto_refresh = st.checkbox("Atualizar automaticamente", value=False)
    refresh_seconds = st.number_input("Intervalo (segundos)", 10, 3600, 60, 10)
    atualizar_agora = st.button("🔄 Atualizar agora")

if presentation:
    st.markdown(
        """
<style>
[data-testid="stSidebar"] { display: none; }
</style>
""",
        unsafe_allow_html=True,
    )

# Presets
if preset != "Personalizado":
    if preset == "Conservador":
        base = "BRL real (IPCA)"
        include_fed = False
        w_btc, w_cdi, w_usd = 10, 80, 10
    elif preset == "Moderado":
        base = "BRL real (IPCA)"
        include_fed = True
        w_btc, w_cdi, w_usd = 35, 50, 15
    elif preset == "Agressivo":
        base = "USD nominal"
        include_fed = True
        w_btc, w_cdi, w_usd = 70, 20, 10

# Cache controls
if atualizar_agora:
    st.cache_data.clear()
    st.toast("Cache limpo. Recarregando...", icon="🔄")

if auto_refresh:
    time.sleep(float(refresh_seconds))
    st.rerun()

# ============================================================
# MAIN
# ============================================================

st.title(APP_TITLE)
st.caption("Research & simulações de investimento • Web3 • Dados reais")

tab_dash, tab_port, tab_cg, tab_met = st.tabs(["📊 Dashboard", "🧺 Carteira", "🪙 CoinGecko", "ℹ️ Metodologia"])

# ============================================================
# Seleção do período (após sidebar)
# ============================================================
idx_min = df_m_full.index.min().date()
idx_max = df_m_full.index.max().date()

# Ajusta inputs de data (quando modo personalizado)
if modo_periodo == "Intervalo personalizado":
    # Se usuário não escolheu, define defaults
    if data_inicio is None:
        data_inicio = idx_min
    if data_fim is None:
        data_fim = idx_max

    # Normaliza e limita ao intervalo disponível
    if isinstance(data_inicio, (list, tuple)):
        data_inicio = data_inicio[0]
    if isinstance(data_fim, (list, tuple)):
        data_fim = data_fim[0]

    di = max(min(pd.to_datetime(data_inicio).date(), idx_max), idx_min)
    df_ = max(min(pd.to_datetime(data_fim).date(), idx_max), idx_min)

    if df_ < di:
        di, df_ = df_, di

    start_ts = pd.Timestamp(di).to_period("M").to_timestamp("M")
    end_ts = pd.Timestamp(df_).to_period("M").to_timestamp("M")

    df_m = df_m_full.loc[(df_m_full.index >= start_ts) & (df_m_full.index <= end_ts)].copy()
else:
    meses_plot = min(anos_plot * 12, len(df_m_full))
    df_m = df_m_full.tail(meses_plot).copy()

# Alinha macro ao período
df_macro = df_macro_full.reindex(df_m.index).ffill().bfill()

if len(df_m) < 6:
    st.warning("Período escolhido é muito curto. Selecione um intervalo maior para simulação.")


# ============================================================
# Dashboard
# ============================================================
with tab_dash:
    c1, c2 = st.columns([3.2, 1.2], gap="large")

    df_prices, unit = build_prices(df_m, df_macro, base, include_fed=include_fed)
    # --- CoinGecko: adiciona altcoins (mensal) como linhas extras no comparativo ---
    if cg_on and cg_coins:
        try:
            start_d = df_m.index.min().date()
            end_d = df_m.index.max().date()
            for label in cg_coins:
                coin_id = COINGECKO_PRESETS.get(label)
                if not coin_id:
                    continue
                s_usd = cg_monthly_close_usd(coin_id, start_d, end_d).reindex(df_m.index)
                # converte para base escolhida
                name = label.split(" (")[0]
                if base.startswith("BRL"):
                    s = (s_usd * df_m["USD_BRL"]).rename(name)
                    if base == "BRL real (IPCA)":
                        s = s / df_macro["IPCA_INDEX"]
                else:
                    s = s_usd.rename(name)
                    if base == "USD real (CPI)":
                        s = s / df_macro["CPI_INDEX"]
                df_prices[s.name] = s
        except Exception as e:
            st.warning(f"CoinGecko indisponível no momento: {e}")

    if cg_on and cg_custom.strip():
        try:
            start_d = df_m.index.min().date()
            end_d = df_m.index.max().date()
            cid = cg_custom.strip().lower()
            s_usd = cg_monthly_close_usd(cid, start_d, end_d).reindex(df_m.index)
            if not s_usd.empty:
                if base.startswith("BRL"):
                    s = (s_usd * df_m["USD_BRL"]).rename(cid.upper())
                    if base == "BRL real (IPCA)":
                        s = s / df_macro["IPCA_INDEX"]
                else:
                    s = s_usd.rename(cid.upper())
                    if base == "USD real (CPI)":
                        s = s / df_macro["CPI_INDEX"]
                df_prices[s.name] = s
        except Exception as e:
            st.warning(f"CoinGecko (coin id '{cg_custom}') falhou: {e}")

    dca_df = calc_dca(df_prices, float(aporte), float(valor_inicial))
    aporte_line = (linha_aportes(dca_df.index, float(aporte)) + float(valor_inicial)) if show_aporte_line else None

    with c1:
        st.subheader("Curvas de patrimônio (DCA mensal)")
        title = f"DCA — {base} — {anos_plot} anos"
        fig = plot_lines(dca_df, title, unit, show_halvings, show_aporte_line, aporte_line)
        st.pyplot(fig, use_container_width=True)

        # Export PDF
        pdf_buf = io.BytesIO()
        fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
        pdf_buf.seek(0)
        st.download_button("📄 Exportar gráfico em PDF", data=pdf_buf, file_name="lm_analytics_grafico.pdf", mime="application/pdf")

        # Export CSV/Excel
        st.markdown("##### Exportar dados")
        csv_bytes = dca_df.to_csv().encode("utf-8")
        st.download_button("⬇️ CSV (DCA)", data=csv_bytes, file_name="dca.csv", mime="text/csv")
        try:
            xlsx = export_excel({"prices": df_prices, "dca": dca_df, "macro": df_macro})
            st.download_button(
                "⬇️ Excel (prices+dca+macro)",
                data=xlsx,
                file_name="lm_analytics.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except ImportError:
            st.info("Exportação Excel desativada: adicione 'openpyxl' ao requirements.txt (ou 'xlsxwriter').")

    with c2:
        st.subheader("Resumo & Risco")
        total_ap = float(aporte) * len(dca_df) + float(valor_inicial)
        st.metric("Total aportado", fmt_money(total_ap, unit))

        rf = 0.0
        if base.startswith("BRL"):
            # approx: use last 12m CDI change as rf proxy (optional)
            try:
                rf = float((df_m["CDI"].iloc[-1] / df_m["CDI"].iloc[-13]) - 1.0)
            except Exception:
                rf = 0.0
        elif base.startswith("USD") and include_fed:
            try:
                rf = float((df_macro["FED_INDEX"].iloc[-1] / df_macro["FED_INDEX"].iloc[-13]) - 1.0)
            except Exception:
                rf = 0.0

        mt = metrics_table(dca_df, unit, rf_annual=rf).copy()

        # Show key assets first
        for asset in ["BTC", "CDI", "USD", "FED (USD+juros)"]:
            if asset in dca_df.columns:
                s = dca_df[asset].astype(float)
                st.metric(
                    f"Final {asset}",
                    fmt_money(float(s.iloc[-1]), unit),
                    f"CAGR {fmt_pct(100*float(cagr(s)))} • DD {fmt_pct(100*float(max_drawdown(s)))}",
                )

        st.divider()
        st.dataframe(
            mt.assign(
                **{
                    "Valor Final": mt["Valor Final"].map(lambda x: fmt_money(float(x), unit)),
                    "CAGR": mt["CAGR"].map(lambda x: fmt_pct(100*float(x))),
                    "Vol (a.a.)": mt["Vol (a.a.)"].map(lambda x: fmt_pct(100*float(x))),
                    "Sharpe": mt["Sharpe"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "—"),
                    "Max Drawdown": mt["Max Drawdown"].map(lambda x: fmt_pct(100*float(x))),
                }
            ),
            use_container_width=True,
            height=320,
        )

        if (_get_fred_api_key() is None) and (os.getenv("FRED_API_KEY") is None):
            st.info("CPI/FED estão em modo fallback (sem FRED_API_KEY). O app não quebra, mas CPI/FED ficam simplificados.")

# ============================================================
# Portfolio
# ============================================================
with tab_port:
    st.subheader("Simulador de Carteira (DCA + Rebalanceamento)")
    st.caption("Obs.: Altcoins via CoinGecko entram no comparativo do Dashboard. A carteira, por enquanto, usa apenas BTC/CDI/USD.")

    weights = {"BTC": w_btc, "CDI": w_cdi, "USD": w_usd}
    # portfolio uses whatever is in df_prices; ensure USD exists in USD-base too
    # if base is USD nominal/real: USD line is 1/cpi, still valid; CDI is converted.
    df_prices_port = df_prices.copy()
    # Keep only columns present
    weights = {k: float(v) for k, v in weights.items() if k in df_prices_port.columns}
    port_series = portfolio_dca(df_prices_port, float(aporte), weights, rebalance=rebalance, valor_inicial=float(valor_inicial))


    colp1, colp2 = st.columns([3.2, 1.2], gap="large")
    with colp1:
        df_plot = pd.DataFrame(index=df_prices_port.index)
        df_plot["Carteira"] = port_series
        # overlay single assets for comparison
        for k in ["BTC", "CDI", "USD"]:
            if k in dca_df.columns:
                df_plot[k] = dca_df[k]

        figp = plot_lines(df_plot, f"Carteira — {base} — Rebalance: {rebalance}", unit, show_halvings, show_aporte_line, aporte_line)
        st.pyplot(figp, use_container_width=True)

        pdf_buf = io.BytesIO()
        figp.savefig(pdf_buf, format="pdf", bbox_inches="tight")
        pdf_buf.seek(0)
        st.download_button("📄 Exportar gráfico da Carteira (PDF)", data=pdf_buf, file_name="carteira.pdf", mime="application/pdf")

        try:
            xlsx = export_excel({"portfolio": df_plot, "prices": df_prices_port, "macro": df_macro})
            st.download_button(
                "⬇️ Excel (carteira+prices+macro)",
                data=xlsx,
                file_name="carteira.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except ImportError:
            st.info("Exportação Excel desativada: adicione 'openpyxl' ao requirements.txt (ou 'xlsxwriter').")

    with colp2:
        st.subheader("Risco (Carteira)")
        mtp = metrics_table(pd.DataFrame({"Carteira": port_series}), unit).iloc[0]
        st.metric("Valor final", fmt_money(float(port_series.iloc[-1]), unit))
        st.metric("CAGR", fmt_pct(100*float(mtp["CAGR"])))
        st.metric("Vol (a.a.)", fmt_pct(100*float(mtp["Vol (a.a.)"])))
        st.metric("Max Drawdown", fmt_pct(100*float(mtp["Max Drawdown"])))
        st.caption("Rebalanceamento ocorre no mês 6/12 (semestral) ou 12 (anual).")

# ============================================================
# Metodologia
# ============================================================

# ============================================================
# CoinGecko — visão macro do mercado cripto
# ============================================================
with tab_cg:
    st.subheader("CoinGecko — Market Overview")
    if not cg_on:
        st.info("Ative o CoinGecko na sidebar para usar estas funções.")
    else:
        ok = cg_ping()
        if not ok:
            st.warning("CoinGecko não respondeu agora. Tente novamente em instantes.")
        else:
            g = cg_global()
            c1, c2, c3, c4 = st.columns(4)
            try:
                total_mcap_usd = float(g.get("total_market_cap", {}).get("usd", 0.0))
                total_vol_usd = float(g.get("total_volume", {}).get("usd", 0.0))
                btc_dom = float(g.get("market_cap_percentage", {}).get("btc", 0.0))
                eth_dom = float(g.get("market_cap_percentage", {}).get("eth", 0.0))
                c1.metric("Market Cap (USD)", f"${total_mcap_usd:,.0f}")
                c2.metric("Volume 24h (USD)", f"${total_vol_usd:,.0f}")
                c3.metric("Dominância BTC", f"{btc_dom:.2f}%")
                c4.metric("Dominância ETH", f"{eth_dom:.2f}%")
            except Exception:
                st.caption("Dados globais indisponíveis.")

            st.divider()
            left, right = st.columns([1.3, 1], gap="large")

            with left:
                st.markdown("#### Top por Market Cap")
                vs = st.selectbox("Moeda", ["usd", "brl"], index=0, key="cg_vs")
                topn = st.slider("Quantidade", 10, 100, 25, 5, key="cg_topn")
                dfm = cg_markets(vs_currency=vs, per_page=int(topn), page=1)
                if dfm.empty:
                    st.info("Sem dados agora.")
                else:
                    df_show = dfm.copy()
                    df_show["current_price"] = df_show["current_price"].map(lambda x: f"{x:,.6f}")
                    df_show["market_cap"] = df_show["market_cap"].map(lambda x: f"{x:,.0f}")
                    df_show["total_volume"] = df_show["total_volume"].map(lambda x: f"{x:,.0f}")
                    st.dataframe(
                        df_show.rename(
                            columns={
                                "symbol": "ticker",
                                "name": "nome",
                                "current_price": "preço",
                                "market_cap": "market cap",
                                "total_volume": "volume",
                                "price_change_percentage_24h": "24h%",
                                "price_change_percentage_7d_in_currency": "7d%",
                            }
                        ),
                        use_container_width=True,
                        height=520,
                    )

            with right:
                st.markdown("#### Buscar moeda")
                q = st.text_input("Nome/ID/símbolo", value="ethereum", key="cg_query")
                if q.strip():
                    sres = cg_search(q.strip())
                    if sres.empty:
                        st.info("Nada encontrado.")
                    else:
                        pick = st.selectbox("Resultados", sres["id"].head(10).tolist(), key="cg_pick")
                        sp = cg_simple_price(pick, vs_currencies=("usd", "brl"))
                        usd = sp.get("usd", None)
                        brl = sp.get("brl", None)
                        ch = sp.get("usd_24h_change", None)
                        st.metric("Preço USD", f"${usd:,.6f}" if usd is not None else "—", f"{ch:.2f}%" if ch is not None else None)
                        st.metric("Preço BRL", f"R$ {brl:,.6f}" if brl is not None else "—")
                        try:
                            end_d = dt.date.today()
                            start_d = end_d - dt.timedelta(days=30)
                            start_ts = int(dt.datetime.combine(start_d, dt.time.min).timestamp())
                            end_ts = int(dt.datetime.combine(end_d, dt.time.max).timestamp())
                            daily = cg_market_chart_range(pick, "usd", start_ts, end_ts)
                            if not daily.empty:
                                fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=140)
                                ax.plot(daily.index, daily.values)
                                ax.set_title(f"{pick} — últimos 30 dias (USD)")
                                ax.grid(True, alpha=0.25)
                                fig.tight_layout()
                                st.pyplot(fig, use_container_width=True)
                        except Exception:
                            pass

with tab_met:
    st.markdown(
        """
### Fontes de dados
- **BTC-USD**: Yahoo Finance (yfinance), dados diários, reamostrados para fechamento mensal.
- **USD/BRL (SGS 1)**, **CDI (SGS 12)** e **IPCA (SGS 433)**: Banco Central do Brasil (API SGS).
- **CPI (CPIAUCSL)** e **FEDFUNDS**: FRED (St. Louis Fed).  
  - Se não houver `FRED_API_KEY`, o app usa **fallback** (CPI constante e FED=0) para não derrubar.

### Como é calculado
- **DCA mensal**: compra “cotas” do ativo pelo preço mensal e acumula patrimônio.
- **Bases**:
  - **BRL nominal**: valores em R$ correntes.
  - **BRL real (IPCA)**: deflaciona por `IPCA_INDEX` (base 1 no início do período).
  - **USD nominal**: CDI convertido para USD pelo câmbio do mês; USD = 1.
  - **USD real (CPI)**: deflaciona por `CPI_INDEX` (base 1 no início do período).
- **Benchmark FED**: índice acumulado aproximado da taxa anual `FEDFUNDS` convertida para taxa mensal.

### Métricas de risco
- **CAGR**: retorno anual composto do patrimônio.
- **Volatilidade (a.a.)**: desvio-padrão dos retornos mensais anualizado.
- **Sharpe**: retorno excedente / volatilidade (aproximação).
- **Max Drawdown**: pior queda do topo até o fundo.

> Dica: se você quiser CPI/FED reais, configure `FRED_API_KEY` nos **Secrets** do Streamlit Cloud.
"""
    )
