# Credit Portfolio Risk & Return Analyzer

Aplicativo em **Streamlit** para análise de carteiras de crédito sob a ótica de **risco e retorno**.  
Faz upload de uma base (CSV/XLSX), calcula **required spread**, **mispricing**, **risk contribution** e simula **rebalanceamentos** (vender piores / reforçar melhores) para otimizar o desempenho ajustado ao risco.

**App principal:** `app.py`  
**Autor:** Walter C Neto

---

## Recursos

- Upload de **CSV** ou **Excel** (normaliza cabeçalhos e números como “10,5%”, “1.234,56”).
- Modos de cálculo:
  - Basel/ASRF (com ajuste de maturidade)
  - Rápido (heurístico)
- Controles na sidebar: ρ default, usar ρ do arquivo, hurdle (% a.a.), funding (bps) e opex (bps).
- Gráfico: *Exposure | % Mispricing – Risk Contribution* com “Sharpe Line”.
- Tabela de desempenho do portfólio: Original, vendendo piores N e aumentando melhores N.
- Visual azul/dourado e **logo na sidebar** (`logoEnf.jpg` na raiz do projeto).

---

## Formato da base

**Obrigatórias:**
- `ead` (exposição monetária)
- `pd` (%)
- `lgd` (%)
- `spread_bp` (bps)

**Opcionais:**
- `maturity_years` (anos)
- `rho` (correlação)

O app aceita sinônimos comuns (ex.: “exposição”, “prob_default”, “taxa_spread”, “duration”, “correlação”) e converte percentuais/textos.

**Exemplo (CSV):**
```csv
ead,pd,lgd,spread_bp,maturity_years,rho
1000000,2.5,45,380,3,0.12
