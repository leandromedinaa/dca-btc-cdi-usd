# 📈 DCA BTC vs CDI vs USD — Dashboard Macroeconômico

Aplicação interativa em **Streamlit** para simular estratégias de **DCA (Dollar-Cost Averaging / Aporte Mensal)** e comparar o desempenho de:

- 🟠 Bitcoin (BTC)
- 🇧🇷 CDI (Brasil)
- 💵 USD (comparação Brasil x Exterior)

com **ajustes macroeconômicos**, levando em conta inflação e juros.

---

## 🚀 Funcionalidades

- Simulação de DCA mensal
- Comparação:
  - BTC vs CDI vs USD
  - Brasil 🇧🇷 x Estados Unidos 🇺🇸
- Ajuste por inflação:
  - IPCA (Brasil)
  - CPI (EUA)
- Benchmark de ativo livre de risco:
  - Fed Funds Rate (acumulado)
- Comparação de retornos:
  - Nominal x Real
- Seleção de moeda base:
  - BRL ou USD
- Exportação de gráficos e resultados em PDF

---

## 📊 Fontes de Dados

- **IPCA (Brasil)** — Banco Central do Brasil (SGS)
- **CPI (EUA)** — FRED (CPIAUCSL)
- **Fed Funds Rate** — FRED (FEDFUNDS)
- **BTC e Câmbio** — APIs de mercado

---

## 🔐 Configuração do Ambiente

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/leandromedinaa/dca-btc-cdi-usd.git
cd dca-btc-cdi-usd
