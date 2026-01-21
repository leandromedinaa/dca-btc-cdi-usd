import time
import io
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
    st.image(LOGO_PATH, width=180)  # logo maior
except Exception:
    pass

st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# =========================
# SIDEBAR (CONTROLES)
# =========================
with st.sidebar:
    st.header("Parâmetros")

    anos_plot = st.slider("Período do gráfico (anos)", 1, 10, 6, 1)

    aporte = st.number_input(
        "Aporte mensal (R$)",
        min_value=0.0,
        value=500.0,
        step=50.0,
        format="%.2f",
    )

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
    st.caption("Dados: Yahoo Finance • Banco Central do Brasil (SGS)")

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


def simular_dca(precos: pd.Series, aporte_mensal: float) -> pd.Series:
    cotas = 0.0
    valores = []
    for p in precos:
        if p <= 0 or aporte_mensal <= 0:
            valores.append(cotas * p)
            continue
        cotas += aporte_mensal / p
        valores.append(cotas * p)
    return pd.Series(valores, index=precos.index)


def calc_dca(df_m: pd.DataFrame, aporte_mensal: float) -> pd.DataFrame:
    out = pd.DataFrame(index=df_m.index)
    out["BTC"] = simular_dca(df_m["BTC_BRL"], aporte_mensal)
    out["USD"] = simular_dca(df_m["USD_BRL"], aporte_mensal)
    out["CDI"] = simular_dca(df_m["CDI"], aporte_mensal)
    return out


def resumo_dca(df_dca: pd.DataFrame, aporte_mensal: float) -> pd.DataFrame:
    total = float(aporte_mensal) * len(df_dca.index)

    def ret(v):
        return (v / total - 1) * 100 if total > 0 else 0.0

    return pd.DataFrame(
        {
            "Total Aportado (R$)": [total, total, total],
            "Valor Final (R$)": [df_dca["BTC"].iloc[-1], df_dca["USD"].iloc[-1], df_dca["CDI"].iloc[-1]],
            "Retorno (%)": [ret(df_dca["BTC"].iloc[-1]), ret(df_dca["USD"].iloc[-1]), ret(df_dca["CDI"].iloc[-1])],
        },
        index=["BTC", "USD", "CDI"],
    )


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

        df_usd = baixar_usd_brl(data_ini, data_fim)
        df_cdi = baixar_cdi(data_ini, data_fim)

        df_all = df_btc_usd.join(df_usd, how="inner").join(df_cdi, how="inner")
        df_all["BTC_BRL"] = df_all["BTC_USD"] * df_all["USD_BRL"]
        df_all = df_all[["BTC_BRL", "USD_BRL", "CDI"]].dropna()

        df_m_full = df_all.resample("ME").last().dropna()
        if len(df_m_full) < 12:
            st.error("Base mensal muito curta. Não há dados suficientes para simular DCA.")
            st.stop()

        # Período do gráfico
        meses_plot = min(anos_plot * 12, len(df_m_full))
        df_m_plot = df_m_full.tail(meses_plot)

        df_dca_plot = calc_dca(df_m_plot, aporte)
        resultado_plot = resumo_dca(df_dca_plot, aporte)

        # Base sensibilidade
        df_base_sens = df_m_plot.copy() if base_sens == "Usar período do gráfico" else df_m_full.copy()

        if modo_sens == "Janela móvel (escolher mês final)":
            fim = st.sidebar.selectbox(
                "Mês final da sensibilidade",
                options=list(df_base_sens.index),
                index=len(df_base_sens.index) - 1,
                format_func=lambda d: d.strftime("%Y-%m"),
            )
            df_base_sens = df_base_sens.loc[:fim].copy()

        # Sensibilidade
        sens_rows, anos_invalidos = [], []
        for y in range(anos_min, anos_max + 1):
            n = y * 12
            if len(df_base_sens) < n:
                anos_invalidos.append(y)
                continue
            df_slice = df_base_sens.tail(n)
            if len(df_slice) < 2:
                anos_invalidos.append(y)
                continue

            df_dca_tmp = calc_dca(df_slice, aporte)
            res_tmp = resumo_dca(df_dca_tmp, aporte)

            sens_rows.append(
                {
                    "Anos": y,
                    "Meses": len(df_slice),
                    "Total Aportado (R$)": float(aporte) * len(df_slice),
                    "Retorno BTC (%)": float(res_tmp.loc["BTC", "Retorno (%)"]),
                    "Retorno USD (%)": float(res_tmp.loc["USD", "Retorno (%)"]),
                    "Retorno CDI (%)": float(res_tmp.loc["CDI", "Retorno (%)"]),
                    "Final BTC (R$)": float(df_dca_tmp["BTC"].iloc[-1]),
                    "Final USD (R$)": float(df_dca_tmp["USD"].iloc[-1]),
                    "Final CDI (R$)": float(df_dca_tmp["CDI"].iloc[-1]),
                }
            )

        df_sens = pd.DataFrame(sens_rows)
        if not df_sens.empty:
            df_sens = df_sens.set_index("Anos").sort_index()

        # Layout principal
        col1, col2 = st.columns([3, 1.2], gap="large")

        with col1:
            fig, ax = plt.subplots(figsize=(16, 7), dpi=140)

            ax.plot(df_dca_plot.index, df_dca_plot["BTC"], label=f"BTC (DCA R${aporte:,.0f}/mês)", color=COL_BTC, linewidth=2.6)
            ax.plot(df_dca_plot.index, df_dca_plot["USD"], label=f"USD (DCA R${aporte:,.0f}/mês)", color=COL_USD, linewidth=2.3)
            ax.plot(df_dca_plot.index, df_dca_plot["CDI"], label=f"CDI (DCA R${aporte:,.0f}/mês)", color=COL_CDI, linewidth=2.3)

            ax.set_title(f"DCA Mensal — BTC vs CDI vs USD ({anos_plot} anos)", pad=14)
            ax.set_xlabel("Data")
            ax.set_ylabel("Patrimônio acumulado (R$)")
            ax.grid(True, alpha=0.22)
            ax.legend(loc="upper left")
            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)

            pdf_buffer = io.BytesIO()
            fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
            pdf_buffer.seek(0)
            st.download_button(
                label="📄 Exportar gráfico em PDF",
                data=pdf_buffer,
                file_name=f"dca_btc_cdi_usd_{anos_plot}anos.pdf",
                mime="application/pdf",
            )

        with col2:
            st.subheader("Resumo\n(período selecionado)")
            total_aportado = float(aporte) * len(df_m_plot)

            st.metric("Total aportado", f"R$ {total_aportado:,.2f}")
            st.metric("Final BTC", f"R$ {resultado_plot.loc['BTC','Valor Final (R$)']:,.2f}",
                      f"{resultado_plot.loc['BTC','Retorno (%)']:.2f}%")
            st.metric("Final USD", f"R$ {resultado_plot.loc['USD','Valor Final (R$)']:,.2f}",
                      f"{resultado_plot.loc['USD','Retorno (%)']:.2f}%")
            st.metric("Final CDI", f"R$ {resultado_plot.loc['CDI','Valor Final (R$)']:,.2f}",
                      f"{resultado_plot.loc['CDI','Retorno (%)']:.2f}%")

            st.caption("BTC em BRL = BTC-USD × USD/BRL (BCB). USD é USD/BRL (BCB). CDI é índice diário oficial (BCB).")
            st.divider()
            st.write("Meses simulados:", len(df_m_plot))
            st.write("De:", df_m_plot.index.min().date(), "Até:", df_m_plot.index.max().date())

        st.divider()
        st.subheader("Sensibilidade — Retorno (%) por horizonte (DCA mensal)")

        if anos_invalidos:
            st.warning(
                f"Horizontes ignorados por falta de meses na base escolhida: {anos_invalidos}. "
                f"Dica: use 'Histórico (10 anos)' ou reduza o horizonte."
            )

        if df_sens.empty:
            st.error("Sem dados suficientes para calcular sensibilidade com os parâmetros atuais.")
            st.stop()

        st.dataframe(
            df_sens[[
                "Meses", "Total Aportado (R$)",
                "Retorno BTC (%)", "Retorno USD (%)", "Retorno CDI (%)",
                "Final BTC (R$)", "Final USD (R$)", "Final CDI (R$)"
            ]].style.format(
                {
                    "Total Aportado (R$)": "R$ {:,.2f}",
                    "Final BTC (R$)": "R$ {:,.2f}",
                    "Final USD (R$)": "R$ {:,.2f}",
                    "Final CDI (R$)": "R$ {:,.2f}",
                    "Retorno BTC (%)": "{:.2f}%",
                    "Retorno USD (%)": "{:.2f}%",
                    "Retorno CDI (%)": "{:.2f}%",
                }
            ),
            use_container_width=True,
        )

        fig2, ax2 = plt.subplots(figsize=(16, 5.2), dpi=140)
        ax2.plot(df_sens.index, df_sens["Retorno BTC (%)"], label="BTC", color=COL_BTC, linewidth=2.6)
        ax2.plot(df_sens.index, df_sens["Retorno USD (%)"], label="USD", color=COL_USD, linewidth=2.3)
        ax2.plot(df_sens.index, df_sens["Retorno CDI (%)"], label="CDI", color=COL_CDI, linewidth=2.3)
        ax2.set_title(f"Retorno (%) do DCA por horizonte ({df_sens.index.min()}–{df_sens.index.max()} anos)", pad=12)
        ax2.set_xlabel("Anos")
        ax2.set_ylabel("Retorno (%)")
        ax2.grid(True, alpha=0.22)
        ax2.legend(loc="best")
        fig2.tight_layout()

        st.pyplot(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Erro no app: {e}")
        st.stop()

with tab_about:
    st.subheader("About / Metodologia")

    st.markdown(
        """
### O que este app faz
Este dashboard compara a evolução de um **DCA (Dollar-Cost Averaging)** mensal em:
- **BTC** (convertido para BRL)
- **USD** (USD/BRL)
- **CDI** (índice acumulado com taxa diária oficial)

Você escolhe:
- o **período do gráfico** (1 a 10 anos),
- o **aporte mensal**,
- e o modo de **sensibilidade** (retornos por horizonte, ano a ano).

---

### Definições
**DCA (aportes mensais):**  
A cada mês, você compra uma fração do ativo usando o aporte mensal fixo.  
O patrimônio acumulado é o valor das “cotas” adquiridas vezes o preço do mês.

---

### Fontes de dados
- **BTC-USD:** Yahoo Finance (`BTC-USD`)
- **USD/BRL:** Banco Central do Brasil (SGS **1**)
- **CDI diário:** Banco Central do Brasil (SGS **12**)

---

### Como o BTC em BRL é calculado
**BTC_BRL = BTC_USD × USD_BRL**

---

### Sensibilidade (retorno por horizonte)
O app calcula o retorno do DCA para horizontes de **N anos** (1…10):
- **Fixo no final:** usa os últimos N anos até o mês mais recente disponível.
- **Janela móvel:** você escolhe um mês final e calcula os últimos N anos terminando nele.

---

### Limitações importantes
- **Não inclui**: impostos, spreads, corretagem, slippage, IOF, taxas de câmbio, custos operacionais.
- **Dados históricos** não garantem retornos futuros.
- **Uso educacional/analítico** (não é recomendação de investimento).

---
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
      <span>© {pd.Timestamp.now().year} • Dados: Yahoo Finance + BCB</span>
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
