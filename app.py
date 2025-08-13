# app.py — Credit Portfolio Risk & Return Analyzer — UX pro
import io
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

# =============== Config =================
st.set_page_config(
    page_title="Credit Portfolio Risk & Return Analyzer",
    layout="wide"
)

# =============== Logo em base64 (evita file://) =================
LOGO_PATH = Path("logoEnf.jpg")  # seu arquivo
def load_logo_base64(p: Path):
    try:
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

logo_b64 = load_logo_base64(LOGO_PATH)
ext = LOGO_PATH.suffix.lower()
if ext == ".svg":
    logo_mime = "image/svg+xml"
elif ext == ".png":
    logo_mime = "image/png"
else:
    logo_mime = "image/jpeg"  # jpg, jpeg, etc.

# =============== Tema + UI =================
# IMPORTANTE: não use f-string aqui -> evita problemas com { } no CSS
st.markdown("""
<style>
:root {
  --safra-blue:#002855;
  --safra-blue-deep:#001F3F;
  --safra-gold:#C6A664;
  --safra-gold-weak: rgba(198,166,100,0.45);
}

/* Fundo azul */
.stApp {
  background-color: var(--safra-blue);
}

/* REMOVIDA a marca-d'água central */
.stApp::after {
  content: none !important;
  display: none !important;
}

/* Sidebar corporativa */
section[data-testid="stSidebar"] {
  background: var(--safra-blue-deep) !important;
  padding-top: 4px !important;   /* cola o topo */
}

/* Box do logo compacto no topo da sidebar */
#brandBox {
  display:flex; align-items:center; justify-content:center;
  padding: 6px 8px 10px;
  margin: 2px 10px 12px;         /* bem próximo do topo */
  border-bottom: 1px solid var(--safra-gold-weak);
}
#brandBox img {
  max-width: 108px;              /* ajuste aqui p/ mais/menos */
  width: 100%; height: auto; display:block;
}

/* Painel central sem “caixa” */
.block-container {
  background: transparent !important;
  padding: 1.2rem 1.6rem;
}

/* Paleta: NÃO pinte div/span globalmente para não afetar inputs */
h1,h2,h3,h4,h5,h6 { color: var(--safra-gold) !important; }
.stMarkdown, p, label { color: #F0E8D6 !important; }

/* Separadores dourados */
hr, .stMarkdown hr { border: 0; height: 1px; background: var(--safra-gold); opacity: .6; }

/* Botões */
.stButton>button, .stDownloadButton>button, button {
  color:#000 !important; background:#F7F7F7 !important;
  border:1px solid var(--safra-gold) !important; border-radius:10px !important;
}

/* ================== INPUTS ================== */
/* Caixas BRANCAS + texto PRETO (select, number/text e uploader) */
div[data-baseweb="input"]>div,
.stNumberInput>div>div,
div[data-baseweb="select"]>div,
div[data-testid="stFileUploader"]>div {
  background:#FFFFFF !important;
  color:#000 !important;
  border:1px solid var(--safra-gold-weak) !important;
  border-radius:10px !important;
}

/* Texto dos campos internos */
div[data-baseweb="input"] input,
.stNumberInput input {
  background:#FFFFFF !important;
  color:#000 !important;
}
.stNumberInput button { color:#000 !important; }

/* Selectbox (valor e itens do menu) */
div[data-baseweb="select"] *,
div[data-baseweb="menu"] * { color:#000 !important; }

/* Placeholders */
input::placeholder, textarea::placeholder { color:#000 !important; opacity:.7; }

/* Sliders dourados */
.stSlider [role="slider"] {
  background: var(--safra-gold) !important; border: 2px solid #8f6f2e !important;
}
.stSlider > div > div > div { background: rgba(198,166,100,0.35) !important; }
.stSlider > div > div > div > div { background: var(--safra-gold) !important; }
</style>
""", unsafe_allow_html=True)

# ==== Logo compacto no topo da sidebar ====
if logo_b64:
    st.sidebar.markdown(
        f"""
        <div id="brandBox">
            <img src="data:{logo_mime};base64,{logo_b64}" alt="logo">
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.write("")  # mantém espaçamento se o logo não carregar

# =============== Cabeçalho =================
st.markdown("""
# Credit Portfolio Risk & Return Analyzer  
**Financial Engineering for Bankers**

Este aplicativo analisa a carteira de crédito sob a ótica de risco e retorno,
calcula o *required spread* (EL + custo de capital + funding + opex),
mede *mispricing* e *risk contribution*, e simula rebalanceamentos
(vender piores / aumentar melhores) para otimizar o desempenho ajustado ao risco.

_App by **Walter C Neto**_
""")
st.markdown("<hr/>", unsafe_allow_html=True)

# =============== Helpers (cálculo) =================
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "ead": ["ead","exposure","exposicao","exposição","exposure_amount","exposureamount","exposure_mn","e.a.d"],
        "pd": ["pd","prob_default","probabilidade_default","probabilidade de default","prob","p.d."],
        "lgd": ["lgd","loss_given_default","perda dado default","l.g.d."],
        "spread_bp": ["spread_bp","spread","taxa_spread","taxa","cupom_spread","cupom","bps","spreads"],
        "maturity_years": ["maturity_years","maturity","mad","duration","prazo","tenor"],
        "rho": ["rho","correlation","correlacao","correlação","rô"]
    }
    cols = {c:str(c).strip().lower().replace(" ","_") for c in df.columns}
    df = df.rename(columns=cols)
    for target, aliases in mapping.items():
        for a in aliases:
            if a in df.columns and target not in df.columns:
                df = df.rename(columns={a: target})
                break
    return df

def parse_numbers(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = (df[c].astype(str)
                     .str.replace(".", "", regex=False)
                     .str.replace(",", ".", regex=False)
                     .str.replace("%","", regex=False))
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    return df

def ensure_cols(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Colunas faltando no arquivo: {miss}")

def asrf_capital_K(pd_, lgd_, rho, maturity_years=None, apply_maturity=True):
    pd_  = np.clip(pd_,  1e-6, 0.999999)
    lgd_ = np.clip(lgd_, 1e-6, 0.999999)
    rho  = np.clip(rho,  1e-6, 0.999999)
    num = norm.ppf(pd_) + np.sqrt(rho)*norm.ppf(0.999)
    den = np.sqrt(1 - rho)
    cond_pd = norm.cdf(num/den)
    K = lgd_*cond_pd - pd_*lgd_
    if apply_maturity and maturity_years is not None:
        b = (0.11852 - 0.05478*np.log(pd_))**2
        M = np.maximum(1.0, maturity_years)
        MA = (1 + (M-2.5)*b) / (1 - 1.5*b)
        K = K * MA
    return np.maximum(K, 0.0)

def required_spread_bps(pd_, lgd_, K, hurdle, funding_bp, opex_bp):
    EL = pd_*lgd_
    return 10000.0*(EL + hurdle*K) + funding_bp + opex_bp

def portfolio_table(df):
    total_ead = df["ead"].sum()
    w = df["ead"] / np.maximum(total_ead, 1e-9)
    tot_spread = (w * df["spread_bp"]).sum()          # observado
    exp_spread = (w * df["required_bp"]).sum()        # esperado
    ul_money   = (df["ead"] * df["K"]).sum()          # UL em $$
    rc_bp      = (w * (df["K"]*10000)).sum()
    sharpe_like = (exp_spread) / rc_bp if rc_bp > 0 else np.nan
    return {
        "Exposure Amount - $MM": total_ead/1e6,
        "Number of Exposures": int(len(df)),
        "Total Spread (bps)": tot_spread,
        "Expected Spread (bps)": exp_spread,
        "Unexpected Loss ($MM)": ul_money/1e6,
        "Sharpe Ratio": 100*sharpe_like
    }

def plot_scatter(df, y_col, ylabel):
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=120)
    x = df["risk_contrib_bp"]; y = df[y_col]
    ax.scatter(x, y, alpha=0.9)

    ratio = (y / np.maximum(x,1e-9))
    mask = np.isfinite(ratio)
    if mask.any():
        thr = np.nanpercentile(ratio[mask], 95)
        slope = np.nanmedian(ratio[(ratio<thr) & mask])
        if np.isfinite(slope) and slope > 0:
            xs = np.linspace(0, max(1, x.max())*1.05, 24)
            ax.plot(xs, slope*xs, linewidth=2, color="#C6A664", label="Sharpe Line")  # dourada
            ax.legend(loc="best")

    ax.set_xlabel("Risk Contribution (bp)")
    ax.set_ylabel(ylabel)
    ax.set_title("Exposure | % Mispricing – Risk Contribution")
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("#FFFFFF")  # fundo branco para legibilidade
    st.pyplot(fig)

def performance_box(orig, sold, boosted):
    def fmt(row):
        return {
            "Exposure Amount - $MM": f"{row['Exposure Amount - $MM']:.3f}",
            "Number of Exposures": f"{row['Number of Exposures']}",
            "Total Spread (bps)": f"{row['Total Spread (bps)']:.1f}",
            "Expected Spread (bps)": f"{row['Expected Spread (bps)']:.1f}",
            "Unexpected Loss ($MM)": f"{row['Unexpected Loss ($MM)']:.1f}",
            "Sharpe Ratio": f"{row['Sharpe Ratio']:.1f}%"
        }
    data = pd.DataFrame({
        "Metric": list(fmt(orig).keys()),
        "Original Portfolio": list(fmt(orig).values()),
        "Sold 10 Under Performers": list(fmt(sold).values()),
        "Increased 10 Top Performers": list(fmt(boosted).values()),
    })
    st.dataframe(data, use_container_width=True)

# =============== Sidebar – Parâmetros =================
st.sidebar.header("Parâmetros do Modelo")
mode = st.sidebar.selectbox("Modo de Cálculo", ["Basel/ASRF", "Rápido (heurístico)"], index=0)
use_file_rho = st.sidebar.checkbox("Usar ρ do arquivo (se existir)", value=True)
default_rho  = st.sidebar.slider("Correlação ρ (default)", 0.01, 0.50, 0.12, 0.01)
hurdle       = st.sidebar.slider("Hurdle (custo do capital, a.a.)", 0.00, 0.30, 0.15, 0.01)
funding_bp   = st.sidebar.number_input("Funding (bps)", value=60.0, step=5.0)
opex_bp      = st.sidebar.number_input("Opex/Outros (bps)", value=10.0, step=5.0)

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
st.sidebar.header("Rebalanceamentos")
sell_k       = st.sidebar.number_input("Vender piores (N exposições)", value=10, step=1, min_value=0)
boost_k      = st.sidebar.number_input("Aumentar melhores (N exposições)", value=10, step=1, min_value=0)
boost_factor = st.sidebar.slider("Fator de aumento de EAD dos melhores", 1.0, 3.0, 1.5, 0.1)

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
y_choice = st.sidebar.selectbox(
    "Eixo Y do gráfico",
    ["Expected spread (required_bp)", "Observed spread (spread_bp)"],
    index=0
)
y_col  = "required_bp" if y_choice.startswith("Expected") else "spread_bp"
ylabel = "Expected Spread (bp)" if y_col=="required_bp" else "Observed Spread (bp)"

# =============== Upload & Cálculo =================
uploaded = st.file_uploader(
    "Carregue sua base (CSV ou XLSX). Colunas: ead, pd, lgd, spread_bp, [maturity_years], [rho].",
    type=["csv","xlsx"]
)
if uploaded is None:
    st.info("Faça upload da sua base de crédito.")
    st.stop()

if uploaded.name.lower().endswith(".csv"):
    raw = uploaded.read()
    df = pd.read_csv(io.BytesIO(raw))
else:
    df = pd.read_excel(uploaded)

df = normalize_headers(df)
df = parse_numbers(df)
ensure_cols(df, ["ead","pd","lgd","spread_bp"])
if "maturity_years" not in df.columns: df["maturity_years"] = np.nan
if "rho" not in df.columns:            df["rho"] = np.nan

df["pd_f"]  = np.clip(df["pd"].astype(float)/100.0, 1e-6, 0.999)
df["lgd_f"] = np.clip(df["lgd"].astype(float)/100.0, 1e-6, 0.999)

if "rho" in df.columns and use_file_rho:
    rho_used = df["rho"].where(df["rho"].notna(), default_rho).astype(float)
else:
    rho_used = pd.Series(default_rho, index=df.index, dtype=float)
rho_used = np.clip(rho_used, 1e-4, 0.999)

if mode.startswith("Basel"):
    K = asrf_capital_K(df["pd_f"], df["lgd_f"], rho_used, df["maturity_years"], apply_maturity=True)
else:
    K = df["lgd_f"] * np.sqrt(df["pd_f"]*(1-df["pd_f"])) * np.sqrt(1+rho_used)

df["K"] = np.maximum(K, 0.0)
df["required_bp"] = required_spread_bps(df["pd_f"], df["lgd_f"], df["K"],
                                        hurdle=hurdle, funding_bp=funding_bp, opex_bp=opex_bp)
df["mispricing_pct"] = (df["spread_bp"] - df["required_bp"]) / np.maximum(df["required_bp"],1e-9)
df["risk_contrib_bp"] = df["K"]*10000

# =============== Layout & Outputs =================
left, right = st.columns([0.60, 0.40], gap="large")
with left:
    plot_scatter(df, y_col=y_col, ylabel=ylabel)

with right:
    st.subheader("Portfolio Performance Table")
    orig = portfolio_table(df)

    # vender piores
    if sell_k > 0 and sell_k < len(df):
        worst_idx = df.sort_values("mispricing_pct").index[:sell_k]
        sold_df = df.drop(worst_idx)
    else:
        sold_df = df.copy()
    sold = portfolio_table(sold_df)

    # aumentar melhores
    if boost_k > 0:
        boosted_df = df.copy()
        best_idx = boosted_df.sort_values("mispricing_pct", ascending=False).index[:boost_k]
        boosted_df.loc[best_idx, "ead"] = boosted_df.loc[best_idx, "ead"] * boost_factor
    else:
        boosted_df = df.copy()
    boosted = portfolio_table(boosted_df)

    performance_box(orig, sold, boosted)

with st.expander("Ver dados calculados"):
    show_cols = ["id","ead","pd","lgd","spread_bp","maturity_years","rho",
                 "required_bp","mispricing_pct","risk_contrib_bp","K"]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols].copy(), use_container_width=True)

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("**App by Walter C Neto** – Todos os direitos reservados.")
