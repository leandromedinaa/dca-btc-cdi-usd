import time
import io
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import streamlit as st

st.set_page_config(page_title="LM Analytics — DCA BTC x CDI x USD",
    page_icon="assets/logo.svg",
    layout="wide", # ====== BRAND HEADER ======
colA, colB = st.columns([1, 7], vertical_alignment="center")

with colA:
    st.image("assets/logo.svg", width=90)

with colB:
    st.markdown(
        """
        <div style="line-height:1.15">
          <h1 style="margin-bottom:0;">
            LM Analytics
          </h1>
          <p style="margin-top:6px; color:#8A8F98;">
            Research & simulações de investimento • Web3 • Dados reais
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
)

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
        help="Se você quer a sensibilidade mudar quando altera o período do gráfico, use 'Usar período do gráfico'.",
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

# =========================
# FUNÇÕES (CACHE / DADOS)
# =========================
@st.cache_data(ttl=60 * 60)  # 1h (Yahoo)
def baixar_btc_usd(anos: int) -> pd.DataFrame:
    df = yf.download(
        "BTC-USD",
        period=f"{anos}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("Falha ao baixar BTC-USD do Yahoo Finance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": "BTC_USD"})
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=24 * 60 * 60)  # 24h (BCB)
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


@st.cache_data(ttl=24 * 60 * 60)  # 24h (BCB)
def baixar_cdi(data_inicial: str, data_final: str) -> pd.DataFrame:
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
    params = {"formato": "json", "dataInicial": data_inicial, "dataFinal": data_final}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    df = pd.DataFrame(r.json())
    if df.empty:
        raise RuntimeError("Falha ao baixar CDI (SGS 12).")

    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["cdi_pct"] = df["valor"].astype(float) / 100
    df = df.set_index("data")[["cdi_pct"]]
    df["CDI"] = (1 + df["cdi_pct"]).cumprod()
    return df[["CDI"]]


def simular_dca(precos: pd.Series, aporte_mensal: float) -> pd.Series:
    cotas = 0.0
    valores = []
    for p in precos:
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
    res = pd.DataFrame(
        {
            "Total Aportado (R$)": [total, total, total],
            "Valor Final (R$)": [df_dca["BTC"].iloc[-1], df_dca["USD"].iloc[-1], df_dca["CDI"].iloc[-1]],
            "Retorno (%)": [
                (df_dca["BTC"].iloc[-1] / total - 1) * 100 if total > 0 else 0.0,
                (df_dca["USD"].iloc[-1] / total - 1) * 100 if total > 0 else 0.0,
                (df_dca["CDI"].iloc[-1] / total - 1) * 100 if total > 0 else 0.0,
            ],
        },
        index=["BTC", "USD", "CDI"],
    )
    return res


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
# PIPELINE PRINCIPAL
# =========================
try:
    anos_max = 10  # sempre baixa 10 anos
    df_btc_usd = baixar_btc_usd(anosshos_max:=anos_max)

    data_ini = df_btc_usd.index.min().strftime("%d/%m/%Y")
    data_fim = df_btc_usd.index.max().strftime("%d/%m/%Y")

    df_usd = baixar_usd_brl(data_ini, data_fim)
    df_cdi = baixar_cdi(data_ini, data_fim)

    df_all = df_btc_usd.join(df_usd, how="inner").join(df_cdi, how="inner")
    df_all["BTC_BRL"] = df_all["BTC_USD"] * df_all["USD_BRL"]
    df_all = df_all[["BTC_BRL", "USD_BRL", "CDI"]].dropna()

    # mensal fim do mês
    df_m_full = df_all.resample("ME").last().dropna()
    if len(df_m_full) < 6:
        st.error("Base mensal muito curta. Não há dados suficientes para simular DCA.")
        st.stop()

    # =========================
    # Período do gráfico principal
    # =========================
    meses_plot = min(anos_plot * 12, len(df_m_full))
    df_m_plot = df_m_full.tail(meses_plot)
    df_dca_plot = calc_dca(df_m_plot, aporte)
    resultado_plot = resumo_dca(df_dca_plot, aporte)

    # =========================
    # Base da sensibilidade
    # =========================
    if base_sens == "Usar período do gráfico":
        df_base_sens = df_m_plot.copy()
    else:
        df_base_sens = df_m_full.copy()

    # Se modo janela móvel, escolher mês final dentro da base escolhida
    if modo_sens == "Janela móvel (escolher mês final)":
        fim = st.sidebar.selectbox(
            "Mês final da sensibilidade",
            options=list(df_base_sens.index),
            index=len(df_base_sens.index) - 1,
            format_func=lambda d: d.strftime("%Y-%m"),
        )
        df_base_sens = df_base_sens.loc[:fim].copy()

    # =========================
    # Sensibilidade (mutável) — só calcula horizontes que cabem
    # =========================
    sens_rows = []
    anos_invalidos = []

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

    # =========================
    # LAYOUT
    # =========================
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_dca_plot.index, df_dca_plot["BTC"], label=f"Bitcoin (DCA R${aporte:,.0f}/mês)")
        ax.plot(df_dca_plot.index, df_dca_plot["USD"], label=f"Dólar (DCA R${aporte:,.0f}/mês)")
        ax.plot(df_dca_plot.index, df_dca_plot["CDI"], label=f"CDI (DCA R${aporte:,.0f}/mês)")
        ax.set_title(f"DCA Mensal — BTC vs CDI vs USD ({anos_plot} anos)")
        ax.set_xlabel("Data")
        ax.set_ylabel("Patrimônio acumulado (R$)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # PDF do gráfico principal
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
        st.subheader("Resumo (período selecionado)")
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
    st.subheader("Sensibilidade do período (DCA mensal) — Retorno (%) por horizonte")

    # Avisos de horizonte que não cabe
    if anos_invalidos:
        st.warning(
            f"Alguns horizontes foram ignorados por falta de meses na base escolhida: {anos_invalidos}. "
            f"Dica: aumente 'Base da sensibilidade' para 'Histórico (10 anos)' ou reduza o horizonte."
        )

    # Tabela + gráfico só se houver dados
    if df_sens.empty:
        st.error(
            "Não há dados suficientes para calcular a sensibilidade com os parâmetros atuais. "
            "Reduza o horizonte (anos) ou aumente a base da sensibilidade."
        )
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

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df_sens.index, df_sens["Retorno BTC (%)"], label="BTC")
    ax2.plot(df_sens.index, df_sens["Retorno USD (%)"], label="USD")
    ax2.plot(df_sens.index, df_sens["Retorno CDI (%)"], label="CDI")
    ax2.set_title(f"Retorno (%) do DCA por horizonte ({df_sens.index.min()}–{df_sens.index.max()} anos)")
    ax2.set_xlabel("Anos")
    ax2.set_ylabel("Retorno (%)")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Erro no app: {e}")
    st.stop()
st.sidebar.divider()
st.sidebar.caption("© LM Analytics — Leandro Medina")
st.sidebar.caption("Dados: Yahoo Finance • Banco Central do Brasil")
