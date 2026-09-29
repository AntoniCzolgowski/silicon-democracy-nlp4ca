# Feasibility results

coded items: 1450, parse errors: 0

## Coder validation (Polish Q4, n=200)
| domain | gold rate | coder rate | F1 | kappa coder-gold | kappa human-human | kappa PL vs EN coding |
|---|---|---|---|---|---|---|
| governance | 0.51 | 0.55 | 0.89 | 0.76 | 0.74 | 0.84 |
| values | 0.56 | 0.59 | 0.91 | 0.79 | 0.83 | 0.85 |
| outcome | 0.22 | 0.17 | 0.63 | 0.54 | 0.60 | 0.79 |
| nonresponse | 0.09 | 0.04 | 0.64 | 0.62 | 0.86 | 0.93 |
macro-F1 0.77; valence acc 0.88, kappa 0.76 (human-human 0.60)

## Corpora
| corpus | n | words mean | median | <=3 words | gov | val | out | nonresp | positive | negative |
|---|---|---|---|---|---|---|---|---|---|---|
| real_pl | 150 | 7.09 | 5.00 | 0.40 | 0.51 | 0.57 | 0.21 | 0.05 | 0.61 | 0.11 |
| real_pl_all300 | 300 | 7.18 | 5.00 | 0.38 | 0.56 | 0.53 | 0.18 | 0.04 | 0.61 | 0.09 |
| silicon_bielik_en | 150 | 67.59 | 66.00 | 0.00 | 0.96 | 0.79 | 0.89 | 0.01 | 0.37 | 0.00 |
| silicon_bielik_pl | 150 | 64.90 | 65.00 | 0.00 | 1.00 | 0.83 | 0.95 | 0.00 | 0.31 | 0.03 |
| silicon_gemma_en | 150 | 73.81 | 74.00 | 0.00 | 0.96 | 0.71 | 0.93 | 0.00 | 0.16 | 0.15 |
| silicon_gemma_pl | 150 | 54.10 | 52.00 | 0.00 | 0.90 | 0.86 | 0.89 | 0.01 | 0.27 | 0.07 |
| real_us | 150 | 8.75 | 5.00 | 0.37 | 0.51 | 0.49 | 0.27 | 0.09 | 0.57 | 0.11 |

split-half JSD within real PL: mean 0.0064, 95th pct 0.0149

## JSD of domain profiles (bootstrap 95% CI)
- silicon_bielik_en vs_real_pl: 0.0443 [0.0261, 0.0744]
- silicon_bielik_en vs_real_us: 0.0457 [0.0242, 0.0768]
- silicon_bielik_pl vs_real_pl: 0.0548 [0.0345, 0.0819]
- silicon_bielik_pl vs_real_us: 0.0579 [0.0359, 0.0863]
- silicon_gemma_en vs_real_pl: 0.0622 [0.0399, 0.0929]
- silicon_gemma_en vs_real_us: 0.0627 [0.0396, 0.0885]
- silicon_gemma_pl vs_real_pl: 0.0457 [0.0266, 0.0736]
- silicon_gemma_pl vs_real_us: 0.0497 [0.0278, 0.0773]
- real_us vs_real_pl: 0.0075 [0.0026, 0.0335]

## H2 language effect: JSD(EN prompt) - JSD(PL prompt), paired bootstrap
- bielik_real_pl: -0.0104 [-0.0200, +0.0028] (n=150)
- bielik_real_us: -0.0122 [-0.0255, +0.0014] (n=150)
- gemma_real_pl: +0.0165 [+0.0050, +0.0287] (n=150)
- gemma_real_us: +0.0130 [+0.0026, +0.0227] (n=150)

## Education gap (high - low)
- real_pl: governance -0.04, values +0.08, outcome -0.01, nonresponse -0.03, other -0.01
- real_pl_all300: governance -0.08, values +0.10, outcome +0.06, nonresponse -0.01, other -0.01
- silicon_bielik_en: governance -0.04, values +0.13, outcome -0.05, nonresponse -0.02, other -0.02
- silicon_bielik_pl: governance +0.00, values +0.25, outcome -0.05, nonresponse +0.00, other +0.00
- silicon_gemma_en: governance +0.07, values +0.33, outcome +0.03, nonresponse +0.00, other +0.00
- silicon_gemma_pl: governance -0.08, values +0.27, outcome -0.12, nonresponse -0.02, other +0.00

## Individual-level agreement with the same person (kappa)
- silicon_bielik_en: governance acc 0.53 k 0.06, values acc 0.53 k -0.01, outcome acc 0.26 k -0.03, nonresponse acc 0.95 k -0.01
- silicon_bielik_pl: governance acc 0.51 k 0.00, values acc 0.57 k 0.06, outcome acc 0.25 k 0.01, nonresponse acc 0.95 k 0.00
- silicon_gemma_en: governance acc 0.51 k 0.00, values acc 0.55 k 0.04, outcome acc 0.25 k 0.00, nonresponse acc 0.95 k 0.00
- silicon_gemma_pl: governance acc 0.53 k 0.04, values acc 0.55 k -0.00, outcome acc 0.27 k -0.01, nonresponse acc 0.95 k -0.01

## Lexical markers (share of answers)
- real_pl: elections/voting 0.23, freedom 0.37, equality 0.12, law/rights 0.24, majority/people 0.29, corruption/negative 0.01
- real_pl_all300: elections/voting 0.25, freedom 0.33, equality 0.13, law/rights 0.25, majority/people 0.32, corruption/negative 0.01
- silicon_bielik_en: elections/voting 0.40, freedom 0.45, equality 0.33, law/rights 0.57, majority/people 0.65, corruption/negative 0.01
- silicon_bielik_pl: elections/voting 0.97, freedom 0.66, equality 0.40, law/rights 0.92, majority/people 0.95, corruption/negative 0.03
- silicon_gemma_en: elections/voting 0.49, freedom 0.20, equality 0.00, law/rights 0.25, majority/people 0.54, corruption/negative 0.01
- silicon_gemma_pl: elections/voting 0.70, freedom 0.61, equality 0.05, law/rights 0.59, majority/people 0.69, corruption/negative 0.03
- real_us: elections/voting 0.17, freedom 0.23, equality 0.06, law/rights 0.09, majority/people 0.21, corruption/negative 0.01
- real_pl_machine_translated_en: elections/voting 0.13, freedom 0.38, equality 0.16, law/rights 0.16, majority/people 0.29, corruption/negative 0.01

## Type diversity at equal token budget
- real_pl: 485.0 types / 1071 tokens
- silicon_bielik_pl: 360.7 types / 1071 tokens
- silicon_gemma_pl: 422.9 types / 1071 tokens
- silicon_bielik_en: 355.6 types / 1328 tokens
- silicon_gemma_en: 331.8 types / 1328 tokens
- real_us: 405.0 types / 1328 tokens

## Embedding homogeneity (mean pairwise cosine; higher = more uniform)
- real_pl: 0.462 (n=150)
- real_pl_all300: 0.455 (n=300)
- silicon_bielik_en: 0.696 (n=150)
- silicon_bielik_pl: 0.827 (n=150)
- silicon_gemma_en: 0.698 (n=150)
- silicon_gemma_pl: 0.703 (n=150)
- real_us: 0.429 (n=150)