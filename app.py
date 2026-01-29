import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import yfinance as yf
import streamlit as st

# ==========================================================
# LM Analytics — V2 (Dashboard + Carteira + Risco + Export)
# ==========================================================
BRAND_NAME = "LM Analytics"
TAGLINE = "Research & simulações de investimento • Web3 • Dados reais"

COL_BTC = "#F7931A"
COL_USD = "#00C853"
COL_CDI = "#2979FF"
COL_FED = "#FF5252"

BTC_HALVINGS = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-20"),
]

st.set_page_config(page_title=BRAND_NAME, page_icon="📊", layout="wide")

def inject_css(hide_sidebar=False):
    st.markdown(
        """<style>
        .block-container{padding-top:1.2rem;}
        h1,h2,h3{letter-spacing:-0.02em;}
        div[data-testid="stMetricValue"]{font-size:1.6rem;}
        </style>""",
        unsafe_allow_html=True,
    )
    if hide_sidebar:
        st.markdown(
            "<style>[data-testid='stSidebar']{display:none;}</style>",
            unsafe_allow_html=True,
        )

def _plot_invest(df: pd.DataFrame, unidade: str, title: str, aporte_line: pd.Series|None=None, show_halvings=False):
    fig, ax = plt.subplots(figsize=(14,6), dpi=140)
    if aporte_line is not None:
        ax.plot(df.index, aporte_line.reindex(df.index).astype(float), label="Aportes (sem rendimento)", linewidth=1.8, alpha=0.55)

    if "BTC" in df.columns: ax.plot(df.index, df["BTC"], label="BTC (DCA)", color=COL_BTC, linewidth=2.6)
    if "USD" in df.columns: ax.plot(df.index, df["USD"], label="USD (DCA)", color=COL_USD, linewidth=2.2)
    if "CDI" in df.columns: ax.plot(df.index, df["CDI"], label="CDI (DCA)", color=COL_CDI, linewidth=2.2)
    if "FED (USD+juros)" in df.columns: ax.plot(df.index, df["FED (USD+juros)"], label="USD + FED (DCA)", color=COL_FED, linewidth=2.1)

    if show_halvings:
        for d in BTC_HALVINGS:
            if df.index.min() <= d <= df.index.max():
                ax.axvline(d, linewidth=1.2, alpha=0.25)
                ax.text(d, ax.get_ylim()[1], "Halving", rotation=90, va="top", ha="right", fontsize=8, alpha=0.35)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Data")
    ax.set_ylabel(f"Patrimônio ({unidade})")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig

# =========================
# V2 metrics & portfolio
# =========================
def _as_series_aporte(index: pd.DatetimeIndex, aporte_base):
    if isinstance(aporte_base, pd.Series):
        s = aporte_base.reindex(index).astype(float)
        return s.fillna(0.0)
    return pd.Series([float(aporte_base)] * len(index), index=index, dtype=float)

def cumulative_contributions(index: pd.DatetimeIndex, aporte_base) -> pd.Series:
    return _as_series_aporte(index, aporte_base).cumsum()

def annualized_vol(r_m: pd.Series) -> float:
    return float(r_m.std(ddof=0) * np.sqrt(12))

def annualized_return(r_m: pd.Series) -> float:
    g = (1.0 + r_m).prod()
    years = max(len(r_m)/12.0, 1e-9)
    return float(g ** (1.0/years) - 1.0)

def max_drawdown(v: pd.Series) -> float:
    peak = v.cummax()
    dd = (v/peak) - 1.0
    return float(dd.min())

def sharpe_ratio(r_m: pd.Series, rf_m: pd.Series|float=0.0) -> float:
    if isinstance(rf_m, (int,float)):
        rf = pd.Series([float(rf_m)] * len(r_m), index=r_m.index)
    else:
        rf = rf_m.reindex(r_m.index).astype(float).fillna(0.0)
    ex = r_m - rf
    vol = ex.std(ddof=0)
    return 0.0 if vol == 0 else float((ex.mean()/vol) * np.sqrt(12))

def build_risk_table(df_value: pd.DataFrame, aporte_base, rf_proxy: pd.Series|None=None) -> pd.DataFrame:
    out=[]
    aporte_s=_as_series_aporte(df_value.index, aporte_base)
    for col in df_value.columns:
        v=df_value[col].astype(float)
        r=v.pct_change().fillna(0.0)
        out.append({
            "Ativo": col,
            "CAGR": annualized_return(r),
            "Vol": annualized_vol(r),
            "Sharpe": sharpe_ratio(r, rf_proxy if rf_proxy is not None else 0.0),
            "MaxDD": max_drawdown(v),
            "Final": float(v.iloc[-1]),
            "Aportes": float(aporte_s.sum()),
        })
    return pd.DataFrame(out).set_index("Ativo")

def simulate_portfolio_dca_rebalance(prices: pd.DataFrame, weights: dict[str,float], aporte_base, rebalance="Nunca") -> pd.Series:
    df=prices.copy().astype(float).replace([np.inf,-np.inf], np.nan).ffill().bfill()
    idx=df.index
    aportes=_as_series_aporte(idx, aporte_base)

    w={k:float(v) for k,v in weights.items() if k in df.columns and float(v)>0}
    if not w: w={df.columns[0]:1.0}
    s=sum(w.values()); w={k:v/s for k,v in w.items()}

    if rebalance=="Anual":
        reb=set(df.groupby(idx.year).tail(1).index)
    elif rebalance=="Semestral":
        reb=set([d for d in idx if d.month in (6,12)])
    else:
        reb=set()

    holdings={k:0.0 for k in df.columns}
    values=[]
    for dt in idx:
        a=float(aportes.loc[dt])
        if a>0:
            for k,wk in w.items():
                p=float(df.loc[dt,k])
                if p>0: holdings[k]+= (a*wk)/p

        v=sum(holdings[k]*float(df.loc[dt,k]) for k in df.columns)
        values.append(v)

        if dt in reb and v>0:
            # rebalance by selling to cash and rebuying
            for k in df.columns: holdings[k]=0.0
            for k,wk in w.items():
                p=float(df.loc[dt,k])
                if p>0: holdings[k]+= (v*wk)/p

    return pd.Series(values, index=idx, name="CARTEIRA")

def build_excel_export(payload: dict[str,pd.DataFrame]) -> bytes:
    buf=io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in payload.items():
            if df is None: 
                continue
            df.to_excel(w, sheet_name=str(name)[:31])
    buf.seek(0)
    return buf.getvalue()

def baixar_btc_usd(anos: int) -> pd.DataFrame:
    df = yf.download(
        "BTC-USD",
        period=f"{anos}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("Falha ao baixar BTC-USD do Yahoo Finance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": "BTC_USD"})
    df.index = pd.to_datetime(df.index)
    return df

def baixar_usd_brl(data_inicial: str, data_final: str) -> pd.DataFrame:
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados"
    params = {"formato": "json", "dataInicial": data_inicial, "dataFinal": data_final}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    df = pd.DataFrame(r.json())
    if df.empty:
        raise RuntimeError("Falha ao baixar USD/BRL (SGS 1).")

    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["USD_BRL"] = df["valor"].astype(float)
    return df.set_index("data")[["USD_BRL"]]

def baixar_cdi(data_inicial: str, data_final: str) -> pd.DataFrame:
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
    params = {"formato": "json", "dataInicial": data_inicial, "dataFinal": data_final}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    df = pd.DataFrame(r.json())
    if df.empty:
        raise RuntimeError("Falha ao baixar CDI (SGS 12).")

    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["cdi_pct"] = df["valor"].astype(float) / 100.0
    df = df.set_index("data")[["cdi_pct"]]
    df["CDI"] = (1 + df["cdi_pct"]).cumprod()
    return df[["CDI"]]

def baixar_ipca(data_inicial: str, data_final: str) -> pd.DataFrame:
    """
    IPCA - Número índice (dez/1993 = 100) — SGS 433.
    Fonte: Banco Central do Brasil (SGS).
    """
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados"
    params = {"formato": "json", "dataInicial": data_inicial, "dataFinal": data_final}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        raise RuntimeError("Falha ao baixar IPCA (SGS 433).")
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["IPCA"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["IPCA"]).set_index("data")[["IPCA"]]
    return df

def simular_dca(precos: pd.Series, aportes) -> pd.Series:
    """
    DCA mensal:
    - precos: série de 'preço' ou 'índice' do ativo (mesma base do aporte).
    - aportes: float (constante) ou pd.Series (variável por mês).
    """
    precos = precos.astype(float)
    if isinstance(aportes, pd.Series):
        aportes = aportes.reindex(precos.index).astype(float)
    else:
        aportes = pd.Series([float(aportes)] * len(precos), index=precos.index, dtype=float)

    cotas = 0.0
    valores = []
    for dt, p in precos.items():
        a = float(aportes.loc[dt]) if pd.notna(aportes.loc[dt]) else 0.0
        if p <= 0 or a <= 0:
            valores.append(cotas * p)
            continue
        cotas += a / p
        valores.append(cotas * p)
    return pd.Series(valores, index=precos.index)

def resumo_dca(df_dca: pd.DataFrame, aporte_base) -> pd.DataFrame:
    if isinstance(aporte_base, pd.Series):
        total = float(aporte_base.sum())
    else:
        total = float(aporte_base) * len(df_dca.index)

    def ret(v):
        return (v / total - 1) * 100 if total > 0 else 0.0

    rows = []
    for col in df_dca.columns:
        rows.append(
            {
                "Ativo": col,
                "Total Aportado": total,
                "Valor Final": float(df_dca[col].iloc[-1]),
                "Retorno (%)": ret(float(df_dca[col].iloc[-1])),
            }
        )
    return pd.DataFrame(rows).set_index("Ativo")

def _to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Reamostra para mês (ME), pega last e ffill dentro do range disponível."""
    if df.empty:
        return df
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index().resample("ME").last()
    out = out.ffill()
    return out

def build_macro_indices(df_m_full: pd.DataFrame, data_ini_br: str, data_fim_br: str, start_iso: str, end_iso: str):
    """
    Retorna df_macro indexado em ME com:
    - IPCA_INDEX (base 1.0 no início do período do app)
    - CPI_INDEX (base 1.0 no início do período do app)
    - FED_INDEX (base 1.0 no início; acumula taxa efetiva mensal derivada da taxa anual)

    Observação importante:
    - Após reamostragem mensal e reindex para o índice do app, usamos ffill() e, se ainda
      houver NaN no início (muito comum quando a série começa alguns meses depois),
      aplicamos bfill() *apenas dentro do range do app* para definir a base inicial.
      Isso evita o bug clássico: IPCA_INDEX vira tudo NaN (e o ajuste real "não parece funcionar").
    """
    # IPCA (SGS 433) - pode ter granularidade mensal; garantimos ME
    df_ipca_raw = _to_month_end(baixar_ipca(data_ini_br, data_fim_br))
    # FRED - CPI e FedFunds
    df_cpi_raw = _to_month_end(baixar_cpi_fred(start_iso, end_iso))
    df_fed_raw = _to_month_end(baixar_fedfunds_fred(start_iso, end_iso, series_id="FEDFUNDS"))

    # Índice mensal do app (para não inventar meses fora do range)
    idx = df_m_full.index

    def _align_monthly_series(df: pd.DataFrame, col: str) -> pd.Series:
        if df is None or df.empty or col not in df.columns:
            return pd.Series(index=idx, dtype=float)
        s = df[col].copy()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index().reindex(idx)
        # ffill dentro do range, mas pode deixar NaN no início
        s = s.ffill()
        # se ainda tem NaN no começo, traz o primeiro valor válido dentro do range do app
        if s.isna().any():
            s = s.bfill()
        return s.astype(float)

    ipca_lvl = _align_monthly_series(df_ipca_raw, "IPCA")
    cpi_lvl = _align_monthly_series(df_cpi_raw, "CPIAUCSL")
    fed_pct = _align_monthly_series(df_fed_raw, "FEDFUNDS")  # % a.a. (anualizada)

    if ipca_lvl.isna().all():
        raise RuntimeError("IPCA (SGS 433) sem dados no intervalo selecionado.")
    if cpi_lvl.isna().all():
        raise RuntimeError("CPI (CPIAUCSL/FRED) sem dados no intervalo selecionado.")
    if fed_pct.isna().all():
        raise RuntimeError("FEDFUNDS/FRED sem dados no intervalo selecionado.")
    # Índices (base 1.0 no início do período do app)
    # IPCA (SGS) pode vir como:
    # - nível do índice (valores "altos", ex.: 100, 200, 6000...)
    # - ou variação mensal em % (valores "baixos", ex.: 0.2, 1.1, 0.8...)
    # Detectamos e construímos um índice acumulado mensal robusto.
    ipca_med = float(ipca_lvl.dropna().median()) if not ipca_lvl.dropna().empty else 0.0
    if ipca_med >= 20.0:
        # Parece nível do índice: normaliza pelo primeiro valor.
        ipca_index = (ipca_lvl / float(ipca_lvl.iloc[0])).rename("IPCA_INDEX")
    else:
        # Parece variação mensal em %: acumula via cumprod.
        ipca_index = (1.0 + (ipca_lvl / 100.0)).cumprod()
        ipca_index = (ipca_index / float(ipca_index.iloc[0])).rename("IPCA_INDEX")

    cpi_index = (cpi_lvl / float(cpi_lvl.iloc[0])).rename("CPI_INDEX")

    # Fed Funds: taxa anual (%) -> taxa efetiva mensal aproximada -> índice acumulado
    fed_annual = (fed_pct / 100.0).clip(lower=-0.9999)  # segurança numérica
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
    ).reindex(idx)

    return df_macro

def build_prices(
    df_m: pd.DataFrame,
    df_macro: pd.DataFrame,
    base: str,
    include_fed: bool = True,
) -> tuple[pd.DataFrame, str, str]:
    """
    Constrói dataframe de "preços/índices" na moeda base escolhida.
    Retorna: (df_prices, unidade, modo_label)
    """
    df = df_m.copy()
    df = df.join(df_macro[["IPCA_INDEX", "CPI_INDEX", "FED_INDEX"]], how="left")
    df = df.ffill()

    ipca = df["IPCA_INDEX"]
    cpi = df["CPI_INDEX"]
    fed = df["FED_INDEX"]
    fx = df["USD_BRL"]

    if base == "BRL nominal":
        prices = pd.DataFrame(
            {
                "BTC": df["BTC_BRL"],
                "USD": df["USD_BRL"],
                "CDI": df["CDI"],
            },
            index=df.index,
        )
        if include_fed:
            prices["FED (USD+juros)"] = fed * fx
        return prices, "R$", "BRL nominal"

    if base == "BRL real (IPCA)":
        prices = pd.DataFrame(
            {
                "BTC": df["BTC_BRL"] / ipca,
                "USD": df["USD_BRL"] / ipca,
                "CDI": df["CDI"] / ipca,
            },
            index=df.index,
        )
        if include_fed:
            prices["FED (USD+juros)"] = (fed * fx) / ipca
        return prices, "R$ (real)", "BRL real (IPCA)"

    if base == "USD nominal":
        prices = pd.DataFrame(
            {
                "BTC": df["BTC_USD"],
                "USD": 1.0,
                "CDI": df["CDI"] / fx,  # BRL -> USD pelo câmbio do mês
            },
            index=df.index,
        )
        if include_fed:
            prices["FED (USD+juros)"] = fed
        return prices, "USD", "USD nominal"

    if base == "USD real (CPI)":
        prices = pd.DataFrame(
            {
                "BTC": df["BTC_USD"] / cpi,
                "USD": (1.0 / cpi),
                "CDI": (df["CDI"] / fx) / cpi,
            },
            index=df.index,
        )
        if include_fed:
            prices["FED (USD+juros)"] = fed / cpi
        return prices, "USD (real)", "USD real (CPI)"

    raise ValueError(f"Base desconhecida: {base}")

# =========================
# APP
# =========================
st.title(f"{BRAND_NAME} — Simulações")
st.caption(TAGLINE)

with st.sidebar:
    st.subheader("Atalhos")
    preset = st.selectbox("Preset rápido", ["Personalizado","Conservador","Moderado","Agressivo"], index=0)
    presentation = st.checkbox("Modo apresentação (esconder sidebar)", value=False)
    show_contrib_line = st.checkbox("Mostrar linha de aportes (sem rendimento)", value=True)
    show_halvings = st.checkbox("Marcar halvings do BTC", value=False)

    st.divider()
    st.subheader("Parâmetros")
    anos_plot = st.slider("Período do gráfico (anos)", 1, 10, 6, 1)
    aporte = st.number_input("Aporte mensal (base selecionada)", min_value=0.0, value=300.0, step=50.0)

    base = st.selectbox("Base", ["BRL nominal","BRL real (IPCA)","USD nominal","USD real (CPI)"], index=0)
    include_fed = st.checkbox("Mostrar benchmark USD + juros FED", value=True)
    st.caption("Para CPI/FED: defina `FRED_API_KEY` (seu ambiente).")

inject_css(hide_sidebar=presentation)

tab_dash, tab_port, tab_about = st.tabs(["📊 Dashboard","🧺 Carteira","ℹ️ Metodologia"])

# ===== Carrega dados (reuso) =====
anos_download = 10
df_btc = baixar_btc_usd(anos_download)
data_ini = df_btc.index.min().strftime("%d/%m/%Y")
data_fim = df_btc.index.max().strftime("%d/%m/%Y")
start_iso = df_btc.index.min().strftime("%Y-%m-%d")
end_iso = df_btc.index.max().strftime("%Y-%m-%d")

df_usd = baixar_usd_brl(data_ini, data_fim)
df_cdi = baixar_cdi(data_ini, data_fim)

df_all = df_btc.join(df_usd, how="inner").join(df_cdi, how="inner")
df_all["BTC_BRL"] = df_all["BTC_USD"] * df_all["USD_BRL"]
df_all = df_all[["BTC_USD","BTC_BRL","USD_BRL","CDI"]].dropna()

df_m = df_all.resample("ME").last().dropna()
df_macro = build_macro_indices(df_m, data_ini, data_fim, start_iso, end_iso)

meses_plot = min(anos_plot*12, len(df_m))
df_m_plot = df_m.tail(meses_plot)

# build prices in selected base
df_prices, unidade, modo = build_prices(df_m_plot, df_macro, base, include_fed=include_fed)

# simulate DCA per column
df_dca = pd.DataFrame(index=df_prices.index)
for col in df_prices.columns:
    df_dca[col] = simular_dca(df_prices[col], float(aporte))

# rename to standard labels (optional)
ren = {}
if "BTC" in df_dca.columns: ren["BTC"]="BTC"
if "USD" in df_dca.columns: ren["USD"]="USD"
if "CDI" in df_dca.columns: ren["CDI"]="CDI"
if "FED (USD+juros)" in df_dca.columns: ren["FED (USD+juros)"]="FED (USD+juros)"
df_dca = df_dca.rename(columns=ren)

with tab_dash:
    st.markdown(f"### Evolução (DCA) — **{modo}**")
    aporte_line = cumulative_contributions(df_dca.index, float(aporte)) if show_contrib_line else None
    fig = _plot_invest(df_dca, unidade, f"DCA — {modo}", aporte_line=aporte_line, show_halvings=show_halvings)
    st.pyplot(fig, use_container_width=True)

    colA, colB = st.columns([1.1, 1.0])
    with colA:
        st.markdown("#### Resumo")
        res = resumo_dca(df_dca)
        for ativo in res.index:
            st.metric(f"{ativo} — final", f"{res.loc[ativo,'Valor Final']:,.2f} {unidade}", f"{res.loc[ativo,'Retorno (%)']:.2f}%")
    with colB:
        st.markdown("#### Risco")
        rf = None
        if "CDI" in df_prices.columns:
            rf = df_prices["CDI"].pct_change().fillna(0.0)
        risk = build_risk_table(df_dca, float(aporte), rf_proxy=rf)
        st.dataframe(risk.style.format({"CAGR":"{:.2%}","Vol":"{:.2%}","Sharpe":"{:.2f}","MaxDD":"{:.2%}","Final":"{:,.2f}","Aportes":"{:,.2f}"}), use_container_width=True)

    st.markdown("#### Exportar")
    exp = build_excel_export({"prices": df_prices, "dca": df_dca, "macro": df_macro.reindex(df_prices.index)})
    st.download_button("📊 Baixar Excel (prices+dca+macro)", data=exp, file_name="lm_analytics_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("📄 Baixar CSV (DCA)", data=df_dca.to_csv().encode("utf-8"), file_name="lm_dca.csv", mime="text/csv")

with tab_port:
    st.markdown("### 🧺 Simulador de Carteira (DCA + Rebalanceamento)")
    st.caption("Pesos percentuais sobre os mesmos ativos da base selecionada.")

    cols = st.columns(min(4, len(df_prices.columns)))
    weights_pct = {}
    for i, col in enumerate(df_prices.columns):
        with cols[i % len(cols)]:
            default = 50 if col=="BTC" else (50 if col=="CDI" else 0)
            if preset=="Conservador":
                default = 70 if col=="CDI" else (30 if col=="USD" else 0)
            if preset=="Moderado":
                default = 40 if col=="BTC" else (40 if col=="CDI" else (20 if col=="USD" else 0))
            if preset=="Agressivo":
                default = 70 if col=="BTC" else (20 if col=="USD" else (10 if col=="CDI" else 0))
            weights_pct[col] = st.slider(f"{col} (%)", 0, 100, int(default), 5, key=f"w_{col}_{base}")

    if sum(weights_pct.values()) == 0:
        st.info("Defina pelo menos um peso > 0.")
        st.stop()

    rebalance = st.selectbox("Rebalanceamento", ["Nunca","Anual","Semestral"], index=1)
    aporte_port = st.number_input("Aporte mensal (carteira)", min_value=0.0, value=float(aporte), step=50.0)

    weights = {k: v/100.0 for k,v in weights_pct.items()}

    carteira = simulate_portfolio_dca_rebalance(df_prices, weights, float(aporte_port), rebalance=rebalance)
    df_port = pd.DataFrame({"CARTEIRA": carteira}, index=carteira.index)
    aporte_line2 = cumulative_contributions(df_port.index, float(aporte_port)) if show_contrib_line else None

    fig2 = _plot_invest(df_port.rename(columns={"CARTEIRA":"BTC"}), unidade, f"Carteira — {modo} • Rebalance: {rebalance}", aporte_line=aporte_line2, show_halvings=show_halvings)
    st.pyplot(fig2, use_container_width=True)

    r = carteira.pct_change().fillna(0.0)
    st.metric("CAGR", f"{annualized_return(r):.2%}")
    st.metric("Vol anual", f"{annualized_vol(r):.2%}")
    st.metric("Max Drawdown", f"{max_drawdown(carteira):.2%}")

    st.download_button("📄 Baixar CSV (Carteira)", data=df_port.to_csv().encode("utf-8"), file_name="lm_carteira.csv", mime="text/csv")
    st.download_button("📊 Baixar Excel (Carteira)", data=build_excel_export({"portfolio_prices": df_prices, "portfolio_value": df_port}), file_name="lm_carteira.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_about:
    st.markdown("### Metodologia (resumo)")
    st.write("- BTC via Yahoo Finance (BTC-USD)")
    st.write("- USD/BRL e IPCA/CDI via SGS/BCB")
    st.write("- CPI e FEDFUNDS via FRED (opcional, depende de FRED_API_KEY)")
    st.write("- DCA: compra mensal de cotas ao preço do mês (último dia do mês).")
    st.write("- Carteira: compra por pesos e rebalanceamento (se escolhido) no fechamento do período.")