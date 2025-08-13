# ================================================================
# Exposure | % Mispricing – Risk Contribution (Sharpe-line)
# Versão robusta (pt-BR): normaliza headers e números com vírgula.
# Usa a MESMA base do Risk Frontier.
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import norm
from pathlib import Path

# ---------------- Config ----------------
INPUT_XLSX = r"C:\Users\Lenovo\Desktop\Desktop\Mestrado FGV\RiskFrontier\exemplo_input_risk_frontier.xlsx"
OUTPUT_DIR = Path(r"C:\Users\Lenovo\Desktop\Desktop\Mestrado FGV\RiskFrontier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUTPUT_DIR / "exposure_mispricing_risk_contribution.png"

SEED = 20250812
np.random.seed(SEED)

RHO_PADRAO = 0.12
PD_FLOOR, PD_CAP = 0.001, 0.30
LGD_FLOOR, LGD_CAP = 0.05, 0.50
ESTADOS_ATIVOS = {"Em dia", "Em atraso", "Na Safra"}
CONF_LEVEL = 0.999
Z_ALPHA = norm.ppf(CONF_LEVEL)

# ---------------- Helpers ----------------
def canonize_cols(df: pd.DataFrame) -> pd.DataFrame:
    canon = (df.columns.astype(str)
             .str.strip().str.lower()
             .str.replace(r'[^a-z0-9]+', '_', regex=True)
             .str.replace(r'_+', '_', regex=True)
             .str.strip('_'))
    return df.rename(columns=dict(zip(df.columns, canon)))

def to_num_pt(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({'': np.nan, 'nan': np.nan})
    s = s.str.replace(r'\.', '', regex=True).str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')

def pmt_price(valor, i_mensal, n_meses):
    if i_mensal == 0: return valor / max(n_meses, 1)
    return valor * (i_mensal * (1 + i_mensal) ** n_meses) / ((1 + i_mensal) ** n_meses - 1)

def saldo_price(valor, i_mensal, n_total, n_pag):
    if i_mensal == 0:
        return float(max(valor - (valor / max(n_total, 1)) * n_pag, 0.0))
    pmt = pmt_price(valor, i_mensal, n_total)
    saldo = valor * (1 + i_mensal) ** n_pag - pmt * ((1 + i_mensal) ** n_pag - 1) / i_mensal
    return float(max(saldo, 0.0))

def pd_dinamico(pd_base, overs, is_delayed, alpha_overs=0.7, alpha_delay=0.3):
    fator = 1 + alpha_overs * overs + alpha_delay * is_delayed
    return float(np.clip(pd_base * fator, PD_FLOOR, PD_CAP))

def vasicek_sigma_loss(pd, lgd, ead, rho, z=Z_ALPHA):
    inv_pd  = norm.ppf(pd)
    cond_pd = norm.cdf((inv_pd + np.sqrt(rho) * z) / np.sqrt(1 - rho))
    ul_z = lgd * ead * cond_pd
    el   = pd  * lgd * ead
    sigma = max(ul_z - el, 0.0) / max(z, 1e-9)
    return float(sigma)

# ---------------- 1) Ler arquivos ----------------
xls = pd.ExcelFile(INPUT_XLSX)
contratos = canonize_cols(pd.read_excel(xls, "contratos"))
ghr_param = canonize_cols(pd.read_excel(xls, "ghr_param"))
try:
    params = canonize_cols(pd.read_excel(xls, "params"))
except Exception:
    params = None

# Renomear para nomes oficiais usados no código
REN = {
    'contratoid':'ContratoID', 'ghr':'GHR', 'estado':'Estado',
    'valor_liberado':'Valor_Liberado', 'prazo_meses':'Prazo_Meses',
    'meses_pagos':'Meses_Pagos', 'meses_restantes':'Meses_Restantes',
    'saldo_atual':'Saldo_Atual', 'spread_aa':'Spread_aa',
    'juros_aa':'Juros_aa', 'juros_am':'Juros_am', 'pmt':'PMT',
    'dpd':'DPD', 'bucket_atraso':'Bucket_Atraso',
    'overs_flag':'Overs_Flag', 'isdelayed_flag':'IsDelayed_Flag',
    'pd_base':'PD_Base', 'lgd_base':'LGD_Base'
}
contratos.rename(columns={k:v for k,v in REN.items() if k in contratos.columns}, inplace=True)
ghr_param.rename(columns={k:v for k,v in REN.items() if k in ghr_param.columns}, inplace=True)
if params is not None:
    params.rename(columns={'parametro':'Parametro','valor':'Valor'}, inplace=True)

# Merge de parâmetros por GHR (PD/LGD/Spread) se faltarem
need_cols = {"PD_Base","LGD_Base","Spread_aa"}
if not need_cols.issubset(contratos.columns):
    bring = ["GHR"] + [c for c in ["PD_Base","LGD_Base","Spread_aa"] if c in ghr_param.columns]
    contratos = contratos.merge(ghr_param[bring].drop_duplicates("GHR"),
                                on="GHR", how="left", validate="many_to_one")

# ---------------- 2) Conversão numérica (pt-BR) ----------------
num_cols = ["Valor_Liberado","Prazo_Meses","Meses_Pagos","Meses_Restantes",
            "Spread_aa","Juros_aa","Juros_am","PMT","Saldo_Atual","DPD",
            "PD_Base","LGD_Base","Overs_Flag","IsDelayed_Flag"]
for c in num_cols:
    if c in contratos.columns:
        contratos[c] = to_num_pt(contratos[c])
for c in ["PD_Base","LGD_Base","Spread_aa"]:
    if c in ghr_param.columns:
        ghr_param[c] = to_num_pt(ghr_param[c])

# Funding default ou vindo de params
FUNDING_AA = 0.10
if params is not None and {"Parametro","Valor"}.issubset(params.columns):
    try:
        FUNDING_AA = float(params.set_index("Parametro")["Valor"].get("Funding_aa", FUNDING_AA))
    except Exception:
        pass

# ---------------- 3) Pré-processamento ----------------
contratos = contratos[contratos["Estado"].isin(ESTADOS_ATIVOS)].copy()

# “Na Safra” => 0 meses pagos
if "Meses_Pagos" in contratos.columns and "Prazo_Meses" in contratos.columns:
    contratos["Meses_Pagos"] = contratos.apply(
        lambda r: 0 if r["Estado"]=="Na Safra" else min(max(int(r["Meses_Pagos"] or 0), 0), int(r["Prazo_Meses"])-1),
        axis=1
    )
# Meses restantes
if "Meses_Restantes" not in contratos.columns or contratos["Meses_Restantes"].isna().any():
    contratos["Meses_Restantes"] = contratos["Prazo_Meses"] - contratos["Meses_Pagos"]
contratos = contratos[contratos["Meses_Restantes"] > 0].copy()

# Juros
if "Juros_aa" not in contratos.columns or contratos["Juros_aa"].isna().any():
    contratos["Juros_aa"] = FUNDING_AA + contratos["Spread_aa"]
if "Juros_am" not in contratos.columns or contratos["Juros_am"].isna().any():
    contratos["Juros_am"] = (1 + contratos["Juros_aa"])**(1/12) - 1

# EAD
if "Saldo_Atual" in contratos.columns and contratos["Saldo_Atual"].notna().any():
    contratos["EAD"] = contratos["Saldo_Atual"].fillna(0.0).astype(float)
else:
    contratos["EAD"] = [
        saldo_price(v, i, n, p)
        for v,i,n,p in zip(contratos["Valor_Liberado"], contratos["Juros_am"],
                           contratos["Prazo_Meses"], contratos["Meses_Pagos"])
    ]

# Flags comportamentais
for c in ["Overs_Flag","IsDelayed_Flag"]:
    if c in contratos.columns:
        contratos[c] = contratos[c].fillna(0).astype(int).astype(bool)
    else:
        rng = np.random.default_rng(SEED)
        contratos[c] = rng.random(len(contratos)) < 0.05

# ===== 4) Métricas por contrato (contínuo + UL-based RC) =====

# --- 4.1 PD forward por bucket usando roll-rates (12m)
BUCKETS = ["current","30","60","90","wo"]

# default roll matrix (se a planilha não tiver)
ROLL_DEFAULT = pd.DataFrame({
    "current":[0.92, 0.06, 0.01, 0.00, 0.01],
    "30"     :[0.35, 0.45, 0.15, 0.02, 0.03],
    "60"     :[0.10, 0.25, 0.45, 0.15, 0.05],
    "90"     :[0.02, 0.05, 0.28, 0.45, 0.20],
    "wo"     :[0.00, 0.00, 0.00, 0.00, 1.00],
}, index=BUCKETS)

def _lower_cols(df): 
    df.columns = df.columns.str.strip().str.lower(); 
    return df

def _load_roll(xls):
    try:
        df = pd.read_excel(xls, "roll_rates")
        df = _lower_cols(df)
        # tenta identificar uma coluna de índice (bucket)
        idxcol = None
        for c in df.columns:
            if str(c).strip().lower() in ["bucket","bkt","estado","faixa","bucket_atraso"]:
                idxcol = c; break
        if idxcol: df = df.set_index(idxcol)
        df.index = df.index.map(lambda x: str(x).strip().lower())
        # se estiver transposto
        if set(BUCKETS).issubset(df.columns):
            df = df[BUCKETS]
        elif set(BUCKETS).issubset(df.index):
            df = df.loc[BUCKETS].T
        df = df.reindex(index=BUCKETS, columns=BUCKETS).fillna(0.0)
        # renormaliza linhas; 'wo' absorvente
        for r in df.index:
            if r == "wo":
                df.loc[r] = 0.0; df.loc[r,"wo"] = 1.0
            else:
                s = df.loc[r].sum()
                df.loc[r] = (df.loc[r]/s) if s>0 else 0.0
                if s==0: df.loc[r,r] = 1.0
        return df
    except Exception:
        return ROLL_DEFAULT.copy()

roll_df = _load_roll(xls)
P = roll_df.values
idx = {b:i for i,b in enumerate(BUCKETS)}
wo_col = roll_df.columns.get_loc("wo")

def pd_forward_from_bucket(bucket: str, horizon_m: int) -> float:
    """Prob. de cair em WO em até 'horizon_m' meses, iniciando em 'bucket'."""
    b = str(bucket).strip().lower()
    if b not in idx: b = "current"
    state = np.zeros(len(BUCKETS)); state[idx[b]] = 1.0
    p_def = 0.0
    for _ in range(horizon_m):
        p_def += state @ P[:, wo_col]
        state = state @ P
        state[idx["wo"]] = 0.0  # absorvente não acumula mais
    return float(np.clip(p_def, 0.0, 1.0))

# --- 4.2 Parâmetros por contrato
ead = contratos["EAD"].values.astype(float)
lgd = np.clip(contratos["LGD_Base"].values.astype(float), LGD_FLOOR, LGD_CAP)

# pega spread mesmo que a planilha tenha 'Spread_aa' ou 'spread_aa'
spread_col = "Spread_aa" if "Spread_aa" in contratos.columns else "spread_aa"
spread_aa = contratos[spread_col].values.astype(float)

# bucket inferido ou vindo da planilha (normaliza para minúsculo)
if "Bucket_Atraso" in contratos.columns:
    bucket = contratos["Bucket_Atraso"].astype(str).str.strip().str.lower().replace({"0":"current","cur":"current"})
else:
    # fallback simples: Em dia/Na Safra -> current; Em atraso -> "30"
    bucket = np.where(contratos["Estado"].isin(["Em dia","Na Safra"]), "current", "30")

# horizonte de 12m ou o que couber até a maturidade
H = np.minimum(12, contratos["Meses_Restantes"].astype(int).clip(lower=1)).values

# PD forward por contrato
pd_fwd = np.array([pd_forward_from_bucket(b, h) for b, h in zip(bucket, H)])

# EL forward 12m (em moeda) e sua versão "taxa anual"
el_forward_money = pd_fwd * lgd * ead
el_forward_rate  = np.divide(el_forward_money, ead, out=np.zeros_like(ead), where=ead>0) / np.maximum(H/12.0, 1e-9)

# Expected Spread anualizado (bp): spread líquido de EL_forward_rate
exp_spread_bp = (spread_aa - el_forward_rate) * 10000.0

# --- 4.3 UL(99.9%) e Risk Contribution (em bp) ---
# UL por contrato com Vasicek (usa PD_base anual como fallback para UL)
pd_base = contratos["PD_Base"].values.astype(float)
from scipy.stats import norm
Z = Z_ALPHA

def ul_99(pd, lgd_i, ead_i, rho=RHO_PADRAO):
    invp = norm.ppf(pd)
    cond = norm.cdf((invp + np.sqrt(rho)*Z)/np.sqrt(1-rho))
    return lgd_i*ead_i*cond

ul_i = np.array([ul_99(pd, l, e) for pd, l, e in zip(pd_base, lgd, ead)])
el_i = pd_base * lgd * ead
ec_i = np.maximum(ul_i - el_i, 0.0)  # EC por contrato (moeda)

# decomposição de risco (RC) usando a mesma fórmula de correlação constante:
# Var(EC) ≈ (1-ρ)*sum(ec_i^2) + ρ*(sum ec_i)^2  -> RC_i proporcional ao termo marginal:
rho = RHO_PADRAO
sum_ec = ec_i.sum()
den = np.sqrt(max((1-rho)*np.sum(ec_i**2) + rho*(sum_ec**2), 1e-12))
rc_abs = ((1 - rho) * (ec_i**2) + rho * ec_i * sum_ec) / max(den, 1e-12)  # moeda
total_ead = np.sum(ead)
rc_bp = (rc_abs / max(total_ead, 1e-12)) * 10000.0  # bp sobre EAD total

# Sharpe (slope) da carteira em bp
port_exp_spread_bp = float(np.average(exp_spread_bp, weights=ead))
port_risk_bp = float((den / max(total_ead, 1e-12)) * 10000.0)
sharpe_port = port_exp_spread_bp / max(port_risk_bp, 1e-12)

excess_bp = exp_spread_bp - sharpe_port * rc_bp

# ===== 5) Cenários e 6) Gráfico permanecem IGUAIS, apenas
#      trocam 'sigma' por 'den' já feito acima. =====


# ---------------- 5) Cenários ----------------
def _portfolio_metrics(mask=None, bump_top=False, bump_pct=0.10):
    if mask is None:
        w_ead = ead.copy()
    else:
        w_ead = ead.copy(); w_ead[~mask] = 0.0
    if bump_top:
        w_ead[mask] *= (1.0 + bump_pct)

    scale = np.divide(w_ead, ead, out=np.zeros_like(w_ead), where=ead>0)
    sigma_i_new = sigma_i * scale

    sum_sigma_new = sigma_i_new.sum()
    var_p_new = (1 - rho) * np.sum(sigma_i_new**2) + rho * (sum_sigma_new**2)
    sigma_p_new = np.sqrt(max(var_p_new, 1e-12))

    total_ead_new = w_ead.sum()
    exp_spread_bp_new = np.average(exp_spread_bp, weights=w_ead) if total_ead_new>0 else 0.0
    risk_bp_new = (sigma_p_new / max(total_ead_new, 1e-12)) * 10000.0
    sharpe_new = exp_spread_bp_new / max(risk_bp_new, 1e-12)
    return total_ead_new, exp_spread_bp_new, risk_bp_new, sharpe_new

idx_sorted = np.argsort(excess_bp)
under_idx = idx_sorted[:10]
top_idx   = idx_sorted[-10:]

mask_keep_without_under = np.ones_like(ead, dtype=bool); mask_keep_without_under[under_idx] = False
mask_top = np.zeros_like(ead, dtype=bool); mask_top[top_idx] = True

tot_ead0, exp0, risk0, shr0 = _portfolio_metrics()
tot_ead1, exp1, risk1, shr1 = _portfolio_metrics(mask_keep_without_under)
tot_ead2, exp2, risk2, shr2 = _portfolio_metrics(mask_top, bump_top=True, bump_pct=0.10)

# >>>>> CORREÇÃO AQUI: pesos SEM fatiar o vetor a; usa 0/1 nos pesos <<<<<
weights_orig   = ead
weights_under  = ead * mask_keep_without_under.astype(float)
weights_top10b = ead * (1.0 + 0.10 * mask_top.astype(float))

def mm(x): return x / 1e6
tab = pd.DataFrame({
    " ": ["Exposure Amount - $MM", "Number of Exposures", "Total Spread (bps)",
          "Expected Spread", "Unexpected Loss", "Sharpe Ratio"],
    "Original Portfolio": [mm(tot_ead0), len(ead), np.average(spread_aa, weights=weights_orig)*10000.0, exp0, risk0, 100.0*shr0],
    "Sold 10 Under Performers": [mm(tot_ead1), len(ead)-10, np.average(spread_aa, weights=weights_under)*10000.0, exp1, risk1, 100.0*shr1],
    "Increased 10 Top Performers": [mm(tot_ead2), len(ead), np.average(spread_aa, weights=weights_top10b)*10000.0, exp2, risk2, 100.0*shr2],
})

# ---------------- 6) Gráfico ----------------
plt.figure(figsize=(12,8))
ax = plt.gca()

ax.scatter(rc_bp, exp_spread_bp, s=18, alpha=0.45, marker="o", label="Exposures")
ax.scatter(rc_bp[top_idx],   exp_spread_bp[top_idx],   marker="^", s=70, color="green", label="Top performers")
ax.scatter(rc_bp[under_idx], exp_spread_bp[under_idx], marker="v", s=70, color="red",   label="Underperformers")

x_line = np.linspace(0, max(rc_bp.max()*1.05, 1.0), 200)
y_line = sharpe_port * x_line
ax.plot(x_line, y_line, color="firebrick", linewidth=2, label=f"Sharpe Ratio (slope = {sharpe_port:.3f})")
ax.text(x_line[-1]*0.55, y_line[-1]*0.55, "Sharpe Ratio", color="firebrick",
        fontsize=10, rotation=np.degrees(np.arctan2(y_line[-1], x_line[-1])))

ax.set_xlabel("Risk Contribution (bp)")
ax.set_ylabel("Expected Spread - Annualized (bp)")
ax.set_title("Exposure | % Mispricing – Risk Contribution")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.35)

ax.text(rc_bp[top_idx].mean()*0.9,  exp_spread_bp[top_idx].mean()*0.9,
        "Increase exposure to\n top performers", color="navy",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
ax.text(rc_bp[under_idx].mean()*1.05, max(1.0, exp_spread_bp[under_idx].mean()*0.5),
        "Sell underperforming exposures", color="navy",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Tabela embutida
tab_fmt = tab.copy()
tab_fmt.iloc[0, 1:] = tab_fmt.iloc[0, 1:].map(lambda v: f"{v:,.0f}")
tab_fmt.iloc[1, 1:] = tab_fmt.iloc[1, 1:].map(lambda v: f"{int(v):,d}")
for i in [2,3,4]: tab_fmt.iloc[i, 1:] = tab_fmt.iloc[i, 1:].map(lambda v: f"{v:,.1f}")
tab_fmt.iloc[5, 1:] = tab_fmt.iloc[5, 1:].map(lambda v: f"{v:,.1f}%")

cell_text = [[tab_fmt.iloc[i,0], tab_fmt.iloc[i,1], tab_fmt.iloc[i,2], tab_fmt.iloc[i,3]] for i in range(len(tab_fmt))]
col_labels = list(tab_fmt.columns)
table = plt.table(cellText=cell_text, colLabels=col_labels, cellLoc='right', colLoc='center',
                  bbox=[0.58, 0.05, 0.39, 0.38])
table.auto_set_font_size(False); table.set_fontsize(8)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=220)
plt.show()

print(f"Figura salva em: {OUT_PNG.resolve()}")
