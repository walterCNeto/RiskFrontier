# ================================================================
# Gerador de Base de Input – Risk Frontier (Consignado INSS)
# Agora com performado + a performar:
# - Valor_Liberado (originação) e Saldo_Atual (via Price)
# - PMT, Juros_aa, Juros_am
# - Na Safra = 0 meses pagos
# - Bucket_Atraso + DPD + flags comportamentais
# Saída: exemplo_input_risk_frontier.xlsx
# Abas: contratos, ghr_param, params, roll_rates, prov_reg, lgd_bucket,
#       opex_cobranca, rho_mult_bucket
# Requisitos: numpy, pandas, xlsxwriter
# ================================================================

import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------------
# Configurações
# -------------------------------
SEED = 20250812
np.random.seed(SEED)

OUT_PATH = Path(r"C:\Users\Lenovo\Desktop\Desktop\Mestrado FGV\RiskFrontier\exemplo_input_risk_frontier.xlsx")

# GHRs (5 grupos homogêneos de risco)
ghr_labels = [f"GHR_{i+1}" for i in range(5)]

# Parâmetros por GHR (coerentes com consignado INSS)
PD_Base   = np.array([0.015, 0.020, 0.030, 0.050, 0.080])  # PD a.a.
LGD_Base  = np.array([0.18, 0.22, 0.25, 0.28, 0.30])       # LGD base
Spread_aa = np.array([0.14, 0.16, 0.18, 0.20, 0.22])       # spread sobre funding a.a.

# Parâmetros globais (aba params)
Funding_aa  = 0.10   # custo de funding a.a.
H_fwd_meses = 12     # horizonte forward p/ ECL+carry (usado pelo script de análise)

# Tamanho da amostra
n_contratos = 3000

# Estados ativos
estados = ["Em dia", "Em atraso", "Na Safra"]
prob_estado = [0.78, 0.12, 0.10]

# Prazos consignado (meses)
prazo_vals  = [24, 36, 48, 60, 72, 84]
prazo_probs = [0.10, 0.18, 0.24, 0.22, 0.18, 0.08]

# -------------------------------
# Helpers: Price
# -------------------------------
def pmt_price(valor, i_mensal, n_meses):
    if i_mensal == 0:
        return valor / max(n_meses, 1)
    return valor * (i_mensal * (1 + i_mensal)**n_meses) / ((1 + i_mensal)**n_meses - 1)

def saldo_price(valor, i_mensal, n_total, n_pag):
    if i_mensal == 0:
        return float(max(valor - (valor / max(n_total, 1)) * n_pag, 0.0))
    pmt = pmt_price(valor, i_mensal, n_total)
    saldo = valor * (1 + i_mensal)**n_pag - pmt * ((1 + i_mensal)**n_pag - 1) / i_mensal
    return float(max(saldo, 0.0))

# -------------------------------
# Gerar aba "contratos"
# -------------------------------
df = pd.DataFrame({
    "ContratoID": np.arange(1, n_contratos + 1),
    "GHR": np.random.choice(ghr_labels, size=n_contratos, p=[0.22, 0.25, 0.22, 0.18, 0.13]),
    "Estado": np.random.choice(estados, size=n_contratos, p=prob_estado),
    "Valor_Liberado": np.random.uniform(1000, 40000, size=n_contratos).round(2),
    "Prazo_Meses": np.random.choice(prazo_vals, size=n_contratos, p=prazo_probs)
})

# Meses_Pagos:
# - "Na Safra" => 0 (sempre)
# - demais => Beta limitada [0, Prazo-1]
meses_base = (df["Prazo_Meses"] * np.random.beta(2, 5, size=n_contratos)).astype(int)
meses_base = np.clip(meses_base, 0, df["Prazo_Meses"].values - 1)
df["Meses_Pagos"] = meses_base
mask_safra = df["Estado"].eq("Na Safra")
df.loc[mask_safra, "Meses_Pagos"] = 0

df["Meses_Restantes"] = df["Prazo_Meses"] - df["Meses_Pagos"]

# Atribuir Spread por contrato (conveniente para calcular saldo atual)
spread_map = dict(zip(ghr_labels, Spread_aa))
df["Spread_aa"] = df["GHR"].map(spread_map)
df["Juros_aa"]  = Funding_aa + df["Spread_aa"]
df["Juros_am"]  = (1 + df["Juros_aa"])**(1/12) - 1

# PMT e Saldo_Atual via Price
df["PMT"] = [
    pmt_price(vl, jm, n)
    for vl, jm, n in zip(df["Valor_Liberado"], df["Juros_am"], df["Prazo_Meses"])
]
df["Saldo_Atual"] = [
    saldo_price(vl, jm, n, p)
    for vl, jm, n, p in zip(df["Valor_Liberado"], df["Juros_am"], df["Prazo_Meses"], df["Meses_Pagos"])
]

# Bucket + DPD coerentes com Estado
rng = np.random.default_rng(SEED)
# Para Em atraso, sorteia DPD: 30/60/90 com pesos 60/30/10
rand_u = rng.random(n_contratos)
dpd = np.where(
    df["Estado"].eq("Em dia"), 0,
    np.where(df["Estado"].eq("Na Safra"), 0,
             np.where(rand_u < 0.6, rng.integers(1, 31, size=n_contratos),
                      np.where(rand_u < 0.9, rng.integers(31, 61, size=n_contratos),
                               rng.integers(61, 91, size=n_contratos)))))
df["DPD"] = dpd

def dpd_to_bucket(x):
    if x <= 0: return "current"
    if x <= 30: return "30"
    if x <= 60: return "60"
    if x <= 90: return "90"
    return "WO"

df["Bucket_Atraso"] = df["DPD"].apply(dpd_to_bucket)

# Flags comportamentais
overs_prob = np.where(df["Estado"].eq("Em atraso"), rng.uniform(0.20, 0.50, n_contratos),
             np.where(df["Estado"].eq("Na Safra"), rng.uniform(0.05, 0.15, n_contratos),
                      rng.uniform(0.00, 0.05, n_contratos)))
isdel_prob = np.where(df["Estado"].eq("Em atraso"), rng.uniform(0.10, 0.30, n_contratos),
             np.where(df["Estado"].eq("Na Safra"), rng.uniform(0.05, 0.20, n_contratos),
                      rng.uniform(0.01, 0.08, n_contratos)))
df["Overs_Flag"]     = (rng.random(n_contratos) < overs_prob).astype(int)
df["IsDelayed_Flag"] = (rng.random(n_contratos) < isdel_prob).astype(int)

# -------------------------------
# Aba "ghr_param"
# -------------------------------
ghr_param = pd.DataFrame({
    "GHR": ghr_labels,
    "PD_Base": PD_Base,
    "LGD_Base": LGD_Base,
    "Spread_aa": Spread_aa
})

# -------------------------------
# Aba "params"
# -------------------------------
params = pd.DataFrame({
    "Parametro": ["Funding_aa", "SEED", "H_fwd_meses"],
    "Valor": [Funding_aa, SEED, H_fwd_meses]
})

# -------------------------------
# Abas de delinquency – defaults (para o analítico usar se não vier outra fonte)
# -------------------------------
BUCKETS = ["current","30","60","90","WO"]

roll_rates = pd.DataFrame({
    "current":[0.92, 0.06, 0.01, 0.00, 0.01],
    "30"     :[0.35, 0.45, 0.15, 0.02, 0.03],
    "60"     :[0.10, 0.25, 0.45, 0.15, 0.05],
    "90"     :[0.02, 0.05, 0.28, 0.45, 0.20],
    "WO"     :[0.00, 0.00, 0.00, 0.00, 1.00],
}, index=BUCKETS)

prov_reg = pd.DataFrame({
    "bucket": BUCKETS,
    "pct":    [0.02, 0.10, 0.30, 0.50, 1.00]
})

lgd_bucket = pd.DataFrame({
    "bucket": BUCKETS,
    "lgd":    [0.25, 0.30, 0.35, 0.40, 0.45]
})

opex_cobranca = pd.DataFrame({
    "bucket": BUCKETS,
    "opex":   [0.0, 15.0, 22.0, 35.0, 80.0]   # R$ por contrato/mês
})

rho_mult_bucket = pd.DataFrame({
    "bucket":   BUCKETS,
    "rho_mult": [1.00, 1.05, 1.10, 1.15, 1.20]
})

# -------------------------------
# Salvar Excel
# -------------------------------
with pd.ExcelWriter(OUT_PATH, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="contratos", index=False)
    ghr_param.to_excel(writer, sheet_name="ghr_param", index=False)
    params.to_excel(writer, sheet_name="params", index=False)
    roll_rates.to_excel(writer, sheet_name="roll_rates", index=True)
    prov_reg.to_excel(writer, sheet_name="prov_reg", index=False)
    lgd_bucket.to_excel(writer, sheet_name="lgd_bucket", index=False)
    opex_cobranca.to_excel(writer, sheet_name="opex_cobranca", index=False)
    rho_mult_bucket.to_excel(writer, sheet_name="rho_mult_bucket", index=False)

print(f"Base de input gerada em: {OUT_PATH.resolve()}")
print("Abas: contratos, ghr_param, params, roll_rates, prov_reg, lgd_bucket, opex_cobranca, rho_mult_bucket")
print("Regras: Na Safra=0 meses pagos; DPD/Bucket coerentes; Saldo_Atual via Price (performado + a performar).")
