import streamlit as st
from pathlib import Path

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="LM Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS CUSTOM (UI WEB3)
# =========================
st.markdown("""
<style>
/* Fundo geral */
html, body, [data-testid="stApp"] {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
}

/* Header central */
.header {
    text-align: center;
    margin-top: 30px;
    margin-bottom: 40px;
}

/* Logo grande */
.logo img {
    width: 260px;
    margin-bottom: 18px;
    filter: drop-shadow(0 0 18px rgba(0,229,255,0.45));
}

/* Subtítulo */
.subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 12px;
}

/* Links sociais */
.social a {
    margin: 0 12px;
    font-size: 16px;
    text-decoration: none;
    color: #38bdf8;
}
.social a:hover {
    color: #a78bfa;
}

/* Footer fixo */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: rgba(2,6,23,0.9);
    border-top: 1px solid rgba(148,163,184,0.15);
    text-align: center;
    padding: 10px;
    font-size: 13px;
    color: #94a3b8;
    z-index: 100;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER (LOGO GRANDE)
# =========================
logo_path = Path("assets/logo.svg")

st.markdown('<div class="header">', unsafe_allow_html=True)

if logo_path.exists():
    st.markdown(
        f"""
        <div class="logo">
            <img src="data:image/svg+xml;utf8,{logo_path.read_text()}">
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("""
<div class="subtitle">
Research & simulações de investimento • Web3 • Dados reais
</div>

<div class="social">
    <a href="https://www.instagram.com/mikesp18/" target="_blank">Instagram</a> |
    <a href="https://www.linkedin.com/in/leandro-medina-770a64386/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# =========================
# SIDEBAR (PARÂMETROS)
# =========================
st.sidebar.title("Parâmetros")

anos = st.sidebar.slider(
    "Período do gráfico (anos)",
    min_value=1,
    max_value=10,
    value=4
)

aporte = st.sidebar.number_input(
    "Aporte mensal (R$)",
    min_value=100,
    max_value=10000,
    value=500,
    step=100
)

# =========================
# CONTEÚDO PRINCIPAL
# =========================
st.subheader("Dashboard")

st.info(
    f"""
    🔹 **Período:** {anos} anos  
    🔹 **Aporte mensal:** R$ {aporte:,.2f}  

    *(Aqui entra sua lógica de BTC x USD x CDI — mantive o layout focado no branding agora)*
    """
)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
© 2026 • LM Analytics — Research independente • Dados públicos • Sem recomendação de investimento
</div>
""", unsafe_allow_html=True)
