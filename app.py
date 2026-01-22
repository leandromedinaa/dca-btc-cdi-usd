import time
import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import streamlit as st

# =========================
# BRAND / THEME
# =========================
BRAND_NAME = "LM Analytics"
TAGLINE = "Research & simulações de investimento • Web3 • Dados reais"
LOGO_PATH = "assets/logo.svg"

# =========================
# SOCIAL LINKS
# =========================
INSTAGRAM_URL = "https://www.instagram.com/mikesp18/"
LINKEDIN_URL = "https://www.linkedin.com/in/leandro-medina-770a64386/"

# Professional colors
COL_BTC = "#F7931A"   # Bitcoin orange
COL_USD = "#00C853"   # Green
COL_CDI = "#2979FF"   # Institutional blue
COL_FED = "#FF5252"   # Red accent for benchmark
COL_PP  = "#B0BEC5"   # Gray/steel for purchasing power line

NEON_CYAN = "#00E5FF"
NEON_PURPLE = "#7C4DFF"
MUTED = "#8A8F98"

# =========================
# SVG ICONS (FUTURISTIC)
# =========================
SVG_IG = f"""
<svg width="18" height="18" viewBox="0 0 24 24" fill="none"
     xmlns="http://www.w3.org/2000/svg">
  <path d="M7.5 2.5h9A5 5 0 0 1 21.5 7.5v9a5 5 0 0 1-5 5h-9a5 5 0 0 1-5-5v-9a5 5 0 0 1 5-5Z"
        stroke="{NEON_CYAN}" stroke-width="1.7"/>
  <path d="M12 16.2a4.2 4.2 0 1 0 0-8.4 4.2 4.2 0 0 0 0 8.4Z"
        stroke="{NEON_PURPLE}" stroke-width="1.7"/>
  <path d="M17.2 6.8h.01" stroke="{NEON_CYAN}" stroke-width="3" stroke-linecap="round"/>
</svg>
"""

SVG_IN = f"""
<svg width="18" height="18" viewBox="0 0 24 24" fill="none"
     xmlns="http://www.w3.org/2000/svg">
  <path d="M4.5 9.5v10" stroke="{NEON_CYAN}" stroke-width="1.8" stroke-linecap="round"/>
  <path d="M4.5 6.2v.2" stroke="{NEON_CYAN}" stroke-width="3" stroke-linecap="round"/>
  <path d="M9.3 19.5v-6.2c0-1.9 1.3-3.4 3.2-3.4 1.8 0 3.1 1.3 3.1 3.2v6.4"
        stroke="{NEON_PURPLE}" stroke-width="1.8" stroke-linecap="round"/>
  <path d="M9.3 10.2v9.3" stroke="{NEON_CYAN}" stroke-width="1.8" stroke-linecap="round"/>
</svg>
"""

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title=f"{BRAND_NAME} — DCA BTC x CDI x USD",
    page_icon=LOGO_PATH,
    layout="wide",
)

# =========================
# GLOBAL CSS (NEON / BUTTONS / FOOTER FIXED)
# =========================
st.markdown(
    f"""
<style>
/* Layout breathing room for fixed footer */
.main .block-container {{
    padding-bottom: 90px;
}}

/* Neon button glow (Streamlit buttons & download) */
.stButton > button, .stDownloadButton > button {{
    border: 1px solid rgba(0,229,255,0.35) !important;
    background: rgba(0,229,255,0.06) !important;
    color: #E6E6E6 !important;
    border-radius: 12px !important;
    box-shadow: 0 0 0px rgba(0,229,255,0.0);
    transition: all 120ms ease-in-out;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    border: 1px solid rgba(0,229,255,0.7) !important;
    box-shadow: 0 0 18px rgba(0,229,255,0.25), 0 0 32px rgba(124,77,255,0.12);
    transform: translateY(-1px);
}}
.stButton > button:active, .stDownloadButton > button:active {{
    transform: translateY(0px);
    box-shadow: 0 0 12px rgba(0,229,255,0.18);
}}

/* Sidebar link style */
.lm-social a {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: {MUTED};
    text-decoration: none;
    margin-top: 6px;
}}
.lm-social a:hover {{
    color: #E6E6E6;
    text-decoration: none;
}}
.lm-social svg {{
    filter: drop-shadow(0 0 10px rgba(0,229,255,0.20));
}}

/* Fixed footer */
.lm-footer {{
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: rgba(14,17,23,0.80);
    backdrop-filter: blur(10px);
    border-top: 1px solid rgba(255,255,255,0.06);
    z-index: 9999;
}}
.lm-footer-inner {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 12px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    color: {MUTED};
    font-size: 13px;
}}
.lm-footer-inner .left {{
    display: flex;
    align-items: center;
    gap: 10px;
}}
.lm-pill {{
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid rgba(0,229,255,0.20);
    background: rgba(0,229,255,0.05);
    color: #E6E6E6;
}}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# HEADER (LOGO CENTRALIZADO)
# =========================
st.markdown(
    """
    <div style="display:flex; justify-content:center; margin-top:10px; margin-bottom:10px;">
    """,
    unsafe_allow_html=True,
)

try:
    st.image(LOGO_PATH, width=600)  # logo maior
except Exception:
    pass

st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# =========================
# UTIL: FRED API KEY
# =========================
def _get_fred_api_key() -> str | None:
    # Prefer Streamlit secrets, fallback to env var.
    try:
        return st.secrets.get("FRED_API_KEY", None)
    except Exception:
        return None

# =========================
# SIDEBAR (CONTROLES)
# =========================
with st.sidebar:
    st.header("Parâmetros")

    anos_plot = st.slider("Período do gráfico (anos)", 1, 10, 6, 1)

    st.subheader("Ajustes macro")

    brasil_exterior = st.checkbox(
        "Brasil x Exterior",
        value=False,
        help="Mostra duas visões: Brasil (BRL) e Exterior (USD), com ajuste real opcional e benchmark FED no exterior.",
    )

    # toggles pedidos
    toggle_brl_real = st.checkbox("Ajustar valores em BRL real (IPCA)", value=False)
    toggle_usd_real = st.checkbox("Ajustar valores em USD real (CPI)", value=False)
    toggle_fed = st.checkbox("Mostrar benchmark USD + juros FED", value=True)

    st.caption("Fontes macro: IPCA (BCB/SGS 433), CPI e FEDFUNDS (FRED).")

    st.divider()

    if not brasil_exterior:
        moeda_base = st.selectbox(
            "Moeda base do gráfico",
            options=["BRL nominal", "BRL real (IPCA)", "USD nominal", "USD real (CPI)"],
            index=0,
        )

        if moeda_base.startswith("USD"):
            aporte = st.number_input(
                "Aporte mensal (USD)",
                min_value=0.0,
                value=100.0,
                step=10.0,
                format="%.2f",
            )
            conv_aporte_brl_para_usd = st.checkbox(
                "Converter aporte para USD pelo câmbio do mês (entrada em BRL)",
                value=False,
                help="Se ativo, o valor acima é interpretado como BRL e convertido para USD mensalmente pelo USD/BRL.",
            )
        else:
            aporte = st.number_input(
                "Aporte mensal (R$)",
                min_value=0.0,
                value=500.0,
                step=50.0,
                format="%.2f",
            )
            conv_aporte_brl_para_usd = False
    else:
        # Brasil x Exterior: dois aportes independentes
        st.markdown("**Aportes (para DCA)**")
        aporte_brl = st.number_input(
            "Brasil — Aporte mensal (R$)",
            min_value=0.0,
            value=500.0,
            step=50.0,
            format="%.2f",
        )
        aporte_usd = st.number_input(
            "Exterior — Aporte mensal (USD)",
            min_value=0.0,
            value=100.0,
            step=10.0,
            format="%.2f",
        )
        conv_aporte_brl_para_usd = st.checkbox(
            "Exterior — Converter aporte (entrada em BRL) para USD pelo câmbio do mês",
            value=False,
            help="Se ativo, o campo 'Exterior — Aporte mensal (USD)' é interpretado como BRL e convertido mensalmente.",
        )
        moeda_base = "BRL nominal"  # usado só para manter compatibilidade com sensibilidade/resumo
        aporte = aporte_brl

    st.divider()
    st.subheader("Sensibilidade (mutável)")

    base_sens = st.selectbox(
        "Base da sensibilidade",
        options=["Usar período do gráfico", "Usar histórico (10 anos)"],
        index=0,
        help="Se você quer a sensibilidade mudar ao alterar o período do gráfico, use 'Usar período do gráfico'.",
    )

    anos_minmax = st.slider(
        "Horizonte (anos) — mínimo e máximo",
        min_value=1,
        max_value=10,
        value=(1, 10),
        step=1,
    )
    anos_min, anos_max = anos_minmax

    modo_sens = st.radio(
        "Modo da sensibilidade",
        options=["Fixo no final (últimos N anos)", "Janela móvel (escolher mês final)"],
        index=0,
        help="No modo janela móvel você escolhe um mês final e a sensibilidade é calculada terminando nele.",
    )

    st.divider()
    st.subheader("Atualização")
    auto_refresh = st.checkbox("Atualizar automaticamente", value=False)
    refresh_seconds = st.number_input("Intervalo (segundos)", 10, 3600, 60, 10)
    atualizar_agora = st.button("🔄 Atualizar agora")

    st.divider()
    st.caption("© LM Analytics — Leandro Medina")
    st.caption("Dados: Yahoo Finance • Banco Central do Brasil (SGS) • FRED")

    st.markdown(
        f"""
        <div class="lm-social" style="margin-top:8px;">
            <a href="{INSTAGRAM_URL}" target="_blank">{SVG_IG}<span>Instagram</span></a><br>
            <a href="{LINKEDIN_URL}" target="_blank">{SVG_IN}<span>LinkedIn</span></a>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# FUNÇÕES (CACHE / DADOS)
# =========================
@st.cache_data(ttl=60 * 60)  # 1h
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


@st.cache_data(ttl=24 * 60 * 60)  # 24h
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


@st.cache_data(ttl=24 * 60 * 60)  # 24h
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


@st.cache_data(ttl=24 * 60 * 60)  # 24h
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
        raise RuntimeError(f"Sem observações no FRED para {series_id}.")
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    # value '.' significa missing
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).set_index("date")[["value"]]
    df = df.rename(columns={"value": series_id})
    return df


@st.cache_data(ttl=24 * 60 * 60)  # 24h
def baixar_cpi_fred(start_iso: str, end_iso: str) -> pd.DataFrame:
    """
    CPIAUCSL — Consumer Price Index for All Urban Consumers: All Items (Index 1982-84=100)
    Fonte: FRED (St. Louis Fed). Requer API key.
    """
    api_key = _get_fred_api_key() or os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY ausente. Configure st.secrets['FRED_API_KEY'] ou variável de ambiente FRED_API_KEY.")
    return _fred_request_observations("CPIAUCSL", start_iso, end_iso, api_key)


@st.cache_data(ttl=24 * 60 * 60)  # 24h
def baixar_fedfunds_fred(start_iso: str, end_iso: str, series_id: str = "FEDFUNDS") -> pd.DataFrame:
    """
    FEDFUNDS — Effective Federal Funds Rate (percent, annualized; série mensal).
    Alternativas comuns: EFFR (diária) ou SOFR (diária), mas FEDFUNDS atende ao requisito.
    Fonte: FRED (St. Louis Fed). Requer API key.
    """
    api_key = _get_fred_api_key() or os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY ausente. Configure st.secrets['FRED_API_KEY'] ou variável de ambiente FRED_API_KEY.")
    return _fred_request_observations(series_id, start_iso, end_iso, api_key)


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


def calc_dca(df_prices: pd.DataFrame, aporte_base) -> pd.DataFrame:
    out = pd.DataFrame(index=df_prices.index)
    for col in df_prices.columns:
        out[col] = simular_dca(df_prices[col], aporte_base)
    return out


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
    ipca_index = (ipca_lvl / float(ipca_lvl.iloc[0])).rename("IPCA_INDEX")
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


def cdi_poder_compra_brl(df_m: pd.DataFrame, df_macro: pd.DataFrame, aporte_brl) -> pd.Series:
    """
    Simula aporte mensal em CDI (nominal BRL) e deflaciona pelo IPCA_INDEX,
    aproximando o "poder de compra" do patrimônio ao longo do tempo.
    
    - CDI é tratado como índice/serie acumulada mensal (BRL nominal)
    - Resultado final = (carteira em CDI nominal) / IPCA_INDEX
    """
    if df_m.empty:
        return pd.Series(dtype=float)
    idx = df_m.index
    if "CDI" not in df_m.columns:
        return pd.Series(index=idx, dtype=float)
    cdi = df_m["CDI"].reindex(idx).ffill()
    ipca = df_macro["IPCA_INDEX"].reindex(idx).ffill().bfill()
    carteira_nominal = simular_dca(cdi, aporte_brl)
    carteira_real = carteira_nominal / ipca
    return carteira_real


def _plot_dca_figure(df_dca: pd.DataFrame, unidade: str, titulo: str):
    fig, ax = plt.subplots(figsize=(16, 7), dpi=140)

    if "BTC" in df_dca.columns:
        ax.plot(df_dca.index, df_dca["BTC"], label="BTC (DCA)", color=COL_BTC, linewidth=2.6)
    if "USD" in df_dca.columns:
        ax.plot(df_dca.index, df_dca["USD"], label="USD (DCA)", color=COL_USD, linewidth=2.3)
    if "CDI" in df_dca.columns:
        ax.plot(df_dca.index, df_dca["CDI"], label="CDI (DCA)", color=COL_CDI, linewidth=2.3)
    if "CDI (poder de compra)" in df_dca.columns:
        ax.plot(
            df_dca.index,
            df_dca["CDI (poder de compra)"],
            label="CDI (poder de compra, IPCA)",
            color=COL_PP,
            linewidth=2.1,
            linestyle="--",
        )
    if "FED (USD+juros)" in df_dca.columns:
        ax.plot(df_dca.index, df_dca["FED (USD+juros)"], label="USD + juros FED (DCA)", color=COL_FED, linewidth=2.2)

    ax.set_title(titulo, pad=14)
    ax.set_xlabel("Data")
    ax.set_ylabel(f"Patrimônio acumulado ({unidade})")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# =========================
# ATUALIZAÇÃO
# =========================
if atualizar_agora:
    st.cache_data.clear()
    st.toast("Cache limpo. Recarregando dados...", icon="🔄")

if auto_refresh:
    time.sleep(float(refresh_seconds))
    st.rerun()

# =========================
# NAV: DASHBOARD / ABOUT
# =========================
tab_dash, tab_about = st.tabs(["📊 Dashboard", "ℹ️ About / Metodologia"])

with tab_dash:
    try:
        # Baixa 10 anos para suportar sensibilidade até 10
        df_btc_usd = baixar_btc_usd(10)

        data_ini = df_btc_usd.index.min().strftime("%d/%m/%Y")
        data_fim = df_btc_usd.index.max().strftime("%d/%m/%Y")

        start_iso = df_btc_usd.index.min().strftime("%Y-%m-%d")
        end_iso = df_btc_usd.index.max().strftime("%Y-%m-%d")

        df_usd = baixar_usd_brl(data_ini, data_fim)
        df_cdi = baixar_cdi(data_ini, data_fim)

        df_all = df_btc_usd.join(df_usd, how="inner").join(df_cdi, how="inner")
        df_all["BTC_BRL"] = df_all["BTC_USD"] * df_all["USD_BRL"]
        df_all = df_all[["BTC_USD", "BTC_BRL", "USD_BRL", "CDI"]].dropna()

        df_m_full = df_all.resample("ME").last().dropna()
        if len(df_m_full) < 12:
            st.error("Base mensal muito curta. Não há dados suficientes para simular DCA.")
            st.stop()

        # Macro (IPCA/CPI/FED)
        df_macro = build_macro_indices(df_m_full, data_ini, data_fim, start_iso, end_iso)

        # Período do gráfico
        meses_plot = min(anos_plot * 12, len(df_m_full))
        df_m_plot = df_m_full.tail(meses_plot)

        # ===== Helpers para aporte (constante vs série) =====
        def aporte_series_usd_from_brl(df_m_ref: pd.DataFrame, aporte_brl_like: float) -> pd.Series:
            # Converte mensalmente: BRL -> USD via USD/BRL do mês.
            fx = df_m_ref["USD_BRL"].astype(float)
            return pd.Series(float(aporte_brl_like), index=df_m_ref.index) / fx

        # ===== Layout principal: Brasil x Exterior ou modo único =====
        if brasil_exterior:
            tab_br, tab_ex = st.tabs(["🇧🇷 Brasil (BRL)", "🌎 Exterior (USD)"])

            # --- Brasil ---
            with tab_br:
                base_br = "BRL real (IPCA)" if toggle_brl_real else "BRL nominal"
                df_prices_br, unidade_br, modo_br = build_prices(df_m_plot, df_macro, base_br, include_fed=False)
                df_dca_br = calc_dca(df_prices_br, float(aporte_brl))
                # Linha extra: CDI (aporte + juros) deflacionado pelo IPCA (poder de compra)
                if base_br.startswith("BRL"):
                    df_dca_br["CDI (poder de compra)"] = cdi_poder_compra_brl(df_m_plot, df_macro, float(aporte_brl))

                col1, col2 = st.columns([3, 1.2], gap="large")
                with col1:
                    titulo = f"DCA Mensal — Brasil ({modo_br}) • BTC vs CDI vs USD ({anos_plot} anos)"
                    fig = _plot_dca_figure(df_dca_br, unidade_br, titulo)
                    st.pyplot(fig, use_container_width=True)

                    pdf_buffer = io.BytesIO()
                    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
                    pdf_buffer.seek(0)
                    st.download_button(
                        label="📄 Exportar gráfico (Brasil) em PDF",
                        data=pdf_buffer,
                        file_name=f"dca_brasil_{modo_br.replace(' ', '_').lower()}_{anos_plot}anos.pdf",
                        mime="application/pdf",
                    )

                with col2:
                    st.subheader("Resumo\n(Brasil)")
                    res_br = resumo_dca(df_dca_br, float(aporte_brl))
                    total_ap = float(aporte_brl) * len(df_m_plot)
                    st.metric("Total aportado", f"R$ {total_ap:,.2f}")
                    for ativo in ["BTC", "USD", "CDI"]:
                        if ativo in res_br.index:
                            st.metric(
                                f"Final {ativo}",
                                f"{res_br.loc[ativo,'Valor Final']:,.2f} {unidade_br}".replace(" ,", ","),
                                f"{res_br.loc[ativo,'Retorno (%)']:.2f}%",
                            )
                    st.caption("BRL real = deflacionado por IPCA (SGS 433).")

            # --- Exterior ---
            with tab_ex:
                base_ex = "USD real (CPI)" if toggle_usd_real else "USD nominal"
                df_prices_ex, unidade_ex, modo_ex = build_prices(df_m_plot, df_macro, base_ex, include_fed=toggle_fed)

                if conv_aporte_brl_para_usd:
                    aporte_ex_series = aporte_series_usd_from_brl(df_m_plot, float(aporte_usd))
                    aporte_ex_base = aporte_ex_series
                else:
                    aporte_ex_base = float(aporte_usd)

                df_dca_ex = calc_dca(df_prices_ex, aporte_ex_base)

                col1, col2 = st.columns([3, 1.2], gap="large")
                with col1:
                    titulo = f"DCA Mensal — Exterior ({modo_ex}) • BTC vs CDI(USD) vs USD ({anos_plot} anos)"
                    if toggle_fed:
                        titulo += " + benchmark FED"
                    fig = _plot_dca_figure(df_dca_ex, unidade_ex, titulo)
                    st.pyplot(fig, use_container_width=True)

                    pdf_buffer = io.BytesIO()
                    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
                    pdf_buffer.seek(0)
                    st.download_button(
                        label="📄 Exportar gráfico (Exterior) em PDF",
                        data=pdf_buffer,
                        file_name=f"dca_exterior_{modo_ex.replace(' ', '_').lower()}_{anos_plot}anos.pdf",
                        mime="application/pdf",
                    )

                with col2:
                    st.subheader("Resumo\n(Exterior)")
                    res_ex = resumo_dca(df_dca_ex, aporte_ex_base)
                    total_ap = float(aporte_ex_base.sum()) if isinstance(aporte_ex_base, pd.Series) else float(aporte_ex_base) * len(df_m_plot)
                    st.metric("Total aportado", f"{total_ap:,.2f} {unidade_ex}")
                    for ativo in ["BTC", "USD", "CDI", "FED (USD+juros)"]:
                        if ativo in res_ex.index:
                            st.metric(
                                f"Final {ativo.split(' ')[0]}",
                                f"{res_ex.loc[ativo,'Valor Final']:,.2f} {unidade_ex}".replace(" ,", ","),
                                f"{res_ex.loc[ativo,'Retorno (%)']:.2f}%",
                            )
                    st.caption("USD real = deflacionado por CPI (CPIAUCSL/FRED). FED = FEDFUNDS acumulado (aprox.).")

            # Sensibilidade: para não confundir, usa a visão Brasil (a mais próxima do app original).
            st.divider()
            st.subheader("Sensibilidade — Retorno (%) por horizonte (DCA mensal) [Brasil]")
            df_sens_prices_base, _, _ = build_prices(
                df_m_plot if base_sens == "Usar período do gráfico" else df_m_full,
                df_macro.reindex(df_m_full.index).ffill(),
                "BRL real (IPCA)" if toggle_brl_real else "BRL nominal",
                include_fed=False,
            )
            aporte_sens_base = float(aporte_brl)

        else:
            # --- Modo único (usa moeda_base) ---
            df_prices_plot, unidade_plot, modo_plot = build_prices(df_m_plot, df_macro, moeda_base, include_fed=toggle_fed)

            if moeda_base.startswith("USD") and conv_aporte_brl_para_usd:
                aporte_plot_base = aporte_series_usd_from_brl(df_m_plot, float(aporte))
            else:
                aporte_plot_base = float(aporte)

            df_dca_plot = calc_dca(df_prices_plot, aporte_plot_base)
            # Linha extra: CDI (aporte + juros) deflacionado pelo IPCA (poder de compra) — somente para bases BRL
            if moeda_base.startswith("BRL"):
                df_dca_plot["CDI (poder de compra)"] = cdi_poder_compra_brl(df_m_plot, df_macro, float(aporte))
            resultado_plot = resumo_dca(df_dca_plot, aporte_plot_base)

            # Layout principal
            col1, col2 = st.columns([3, 1.2], gap="large")

            with col1:
                titulo = f"DCA Mensal — BTC vs CDI vs USD ({modo_plot}) • {anos_plot} anos"
                if toggle_fed:
                    titulo += " + benchmark FED"
                fig = _plot_dca_figure(df_dca_plot, unidade_plot, titulo)
                st.pyplot(fig, use_container_width=True)

                pdf_buffer = io.BytesIO()
                fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
                pdf_buffer.seek(0)
                st.download_button(
                    label="📄 Exportar gráfico em PDF",
                    data=pdf_buffer,
                    file_name=f"dca_btc_cdi_usd_{modo_plot.replace(' ', '_').lower()}_{anos_plot}anos.pdf",
                    mime="application/pdf",
                )

            with col2:
                st.subheader("Resumo\n(período selecionado)")
                total_aportado = float(aporte_plot_base.sum()) if isinstance(aporte_plot_base, pd.Series) else float(aporte_plot_base) * len(df_m_plot)

                st.metric("Total aportado", f"{total_aportado:,.2f} {unidade_plot}")
                for ativo in ["BTC", "USD", "CDI", "FED (USD+juros)"]:
                    if ativo in resultado_plot.index:
                        st.metric(
                            f"Final {ativo.split(' ')[0]}",
                            f"{resultado_plot.loc[ativo,'Valor Final']:,.2f} {unidade_plot}".replace(" ,", ","),
                            f"{resultado_plot.loc[ativo,'Retorno (%)']:.2f}%",
                        )

                st.caption("BTC em BRL = BTC-USD × USD/BRL (BCB). IPCA (SGS 433). CPI/FEDFUNDS via FRED.")
                st.divider()
                st.write("Meses simulados:", len(df_m_plot))
                st.write("De:", df_m_plot.index.min().date(), "Até:", df_m_plot.index.max().date())

            # Sensibilidade usa o mesmo modo/valores do gráfico (modo único)
            st.divider()
            st.subheader("Sensibilidade — Retorno (%) por horizonte (DCA mensal)")
            df_sens_prices_base = None
            aporte_sens_base = None

            df_base_sens = df_m_plot.copy() if base_sens == "Usar período do gráfico" else df_m_full.copy()

            if modo_sens == "Janela móvel (escolher mês final)":
                fim = st.sidebar.selectbox(
                    "Mês final da sensibilidade",
                    options=list(df_base_sens.index),
                    index=len(df_base_sens.index) - 1,
                    format_func=lambda d: d.strftime("%Y-%m"),
                )
                df_base_sens = df_base_sens.loc[:fim].copy()

            df_sens_prices_base, _, _ = build_prices(
                df_base_sens,
                df_macro.reindex(df_base_sens.index).ffill(),
                moeda_base,
                include_fed=toggle_fed,
            )

            if moeda_base.startswith("USD") and conv_aporte_brl_para_usd:
                aporte_sens_base = aporte_series_usd_from_brl(df_base_sens, float(aporte))
            else:
                aporte_sens_base = float(aporte)

        # ===== Sensibilidade (com df_sens_prices_base + aporte_sens_base) =====
        sens_rows, anos_invalidos = [], []
        if df_sens_prices_base is None:
            df_sens_prices_base = df_m_plot[["BTC_BRL", "USD_BRL", "CDI"]].copy()
            aporte_sens_base = float(aporte)

        for y in range(anos_min, anos_max + 1):
            n = y * 12
            if len(df_sens_prices_base) < n:
                anos_invalidos.append(y)
                continue
            df_slice_prices = df_sens_prices_base.tail(n)
            if len(df_slice_prices) < 2:
                anos_invalidos.append(y)
                continue

            # aporte pode ser série; corta junto
            if isinstance(aporte_sens_base, pd.Series):
                aporte_slice = aporte_sens_base.reindex(df_slice_prices.index)
            else:
                aporte_slice = float(aporte_sens_base)

            df_dca_tmp = calc_dca(df_slice_prices, aporte_slice)
            res_tmp = resumo_dca(df_dca_tmp, aporte_slice)

            def get_ret(name):
                return float(res_tmp.loc[name, "Retorno (%)"]) if name in res_tmp.index else float("nan")

            def get_final(name):
                return float(df_dca_tmp[name].iloc[-1]) if name in df_dca_tmp.columns else float("nan")

            total_ap = float(aporte_slice.sum()) if isinstance(aporte_slice, pd.Series) else float(aporte_slice) * len(df_slice_prices)

            sens_rows.append(
                {
                    "Anos": y,
                    "Meses": len(df_slice_prices),
                    "Total Aportado": total_ap,
                    "Retorno BTC (%)": get_ret("BTC"),
                    "Retorno USD (%)": get_ret("USD"),
                    "Retorno CDI (%)": get_ret("CDI"),
                    "Retorno FED (%)": get_ret("FED (USD+juros)"),
                    "Final BTC": get_final("BTC"),
                    "Final USD": get_final("USD"),
                    "Final CDI": get_final("CDI"),
                    "Final FED": get_final("FED (USD+juros)"),
                }
            )

        df_sens = pd.DataFrame(sens_rows)
        if not df_sens.empty:
            df_sens = df_sens.set_index("Anos").sort_index()

        if anos_invalidos:
            st.warning(
                f"Horizontes ignorados por falta de meses na base escolhida: {anos_invalidos}. "
                f"Dica: use 'Histórico (10 anos)' ou reduza o horizonte."
            )

        if df_sens.empty:
            st.error("Sem dados suficientes para calcular sensibilidade com os parâmetros atuais.")
            st.stop()

        # Tabela
        fmt = {
            "Total Aportado": "{:,.2f}",
            "Final BTC": "{:,.2f}",
            "Final USD": "{:,.2f}",
            "Final CDI": "{:,.2f}",
            "Final FED": "{:,.2f}",
            "Retorno BTC (%)": "{:.2f}%",
            "Retorno USD (%)": "{:.2f}%",
            "Retorno CDI (%)": "{:.2f}%",
            "Retorno FED (%)": "{:.2f}%",
        }

        cols_show = [
            "Meses", "Total Aportado",
            "Retorno BTC (%)", "Retorno USD (%)", "Retorno CDI (%)",
            "Final BTC", "Final USD", "Final CDI",
        ]
        if "Retorno FED (%)" in df_sens.columns and df_sens["Retorno FED (%)"].notna().any():
            cols_show.insert(5, "Retorno FED (%)")
            cols_show.extend(["Final FED"])

        st.dataframe(
            df_sens[cols_show].style.format(fmt),
            use_container_width=True,
        )

        # Chart sens
        fig2, ax2 = plt.subplots(figsize=(16, 5.2), dpi=140)
        if "Retorno BTC (%)" in df_sens.columns:
            ax2.plot(df_sens.index, df_sens["Retorno BTC (%)"], label="BTC", color=COL_BTC, linewidth=2.6)
        if "Retorno USD (%)" in df_sens.columns:
            ax2.plot(df_sens.index, df_sens["Retorno USD (%)"], label="USD", color=COL_USD, linewidth=2.3)
        if "Retorno CDI (%)" in df_sens.columns:
            ax2.plot(df_sens.index, df_sens["Retorno CDI (%)"], label="CDI", color=COL_CDI, linewidth=2.3)
        if "Retorno FED (%)" in df_sens.columns and df_sens["Retorno FED (%)"].notna().any():
            ax2.plot(df_sens.index, df_sens["Retorno FED (%)"], label="FED", color=COL_FED, linewidth=2.0)

        ax2.set_title(f"Retorno (%) do DCA por horizonte ({df_sens.index.min()}–{df_sens.index.max()} anos)", pad=12)
        ax2.set_xlabel("Anos")
        ax2.set_ylabel("Retorno (%)")
        ax2.grid(True, alpha=0.22)
        ax2.legend(loc="best")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Erro no app: {e}")
        st.info(
            "Se o erro for sobre FRED_API_KEY, crie um arquivo .streamlit/secrets.toml com:\n\n"
            "FRED_API_KEY = \"SUA_API_KEY_AQUI\"\n\n"
            "Ou defina a variável de ambiente FRED_API_KEY."
        )
        st.stop()

with tab_about:
    st.subheader("About / Metodologia")

    st.markdown(
        """
### O que este app faz
Este dashboard compara a evolução de um **DCA (Dollar-Cost Averaging)** mensal em:
- **BTC**
- **USD**
- **CDI** (índice acumulado com taxa diária oficial)

E agora permite:
- Selecionar **moeda base** do gráfico (**BRL/USD**, nominal/real),
- Ajustar **BRL real** por **IPCA** (SGS 433),
- Ajustar **USD real** por **CPI** (CPIAUCSL/FRED),
- Comparar **Brasil x Exterior** e, no exterior, mostrar um benchmark de **USD + juros FED** (FEDFUNDS/FRED).

---

### Definições
**DCA (aportes mensais):**  
A cada mês, você compra uma fração do ativo usando o aporte mensal fixo (ou aporte convertido pelo câmbio do mês, se habilitado).  
O patrimônio acumulado é o valor das “cotas” adquiridas vezes o preço do mês.

**Deflação (valores reais):**  
- BRL real = BRL nominal / IPCA_INDEX  
- USD real = USD nominal / CPI_INDEX

**FED benchmark (aproximação):**  
- Usa a série **FEDFUNDS** (taxa anual em %)  
- Converte para taxa efetiva mensal aproximada: (1 + taxa_anual)^(1/12) − 1  
- Acumula um índice base 1.0 (FED_INDEX)

---

### Fontes de dados
- **BTC-USD:** Yahoo Finance (`BTC-USD`)
- **USD/BRL:** Banco Central do Brasil (SGS **1**)
- **CDI diário:** Banco Central do Brasil (SGS **12**)
- **IPCA (índice):** Banco Central do Brasil (SGS **433**)
- **CPI (EUA):** FRED (`CPIAUCSL`) — requer API key
- **Fed Funds Rate:** FRED (`FEDFUNDS`) — requer API key

---

### Limitações importantes
- **Não inclui**: impostos, spreads, corretagem, slippage, IOF, taxas de câmbio, custos operacionais.
- **Deflação e FED benchmark** são aproximações (úteis para análise macro, não para precificação perfeita).
- **Uso educacional/analítico** (não é recomendação de investimento).
        """
    )

# =========================
# FIXED FOOTER (always visible)
# =========================
st.markdown(
    f"""
<div class="lm-footer">
  <div class="lm-footer-inner">
    <div class="left">
      <span class="lm-pill">LM Analytics</span>
      <span>© {pd.Timestamp.now().year} • Dados: Yahoo Finance + BCB + FRED</span>
    </div>
    <div class="lm-social">
      <a href="{INSTAGRAM_URL}" target="_blank">{SVG_IG}<span>Instagram</span></a>
      <a href="{LINKEDIN_URL}" target="_blank" style="margin-left:14px;">{SVG_IN}<span>LinkedIn</span></a>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
