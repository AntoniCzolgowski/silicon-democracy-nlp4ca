"""Analysis for the feasibility test. Reads data/inputs/ and outputs/ (silicon.jsonl, coded.jsonl, embeddings.jsonl);
writes outputs/results.json and outputs/results.md. Run from anywhere: python src/analyze.py"""
import json, os, re, math, random, collections, statistics as st
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN, OUT = os.path.join(ROOT, 'data', 'inputs'), os.path.join(ROOT, 'outputs')
DOMS = ['governance', 'values', 'outcome', 'nonresponse']
PROFILE = ['governance', 'values', 'outcome', 'nonresponse', 'other']
VALS = ['positive', 'negative', 'neutral', 'mixed', 'none']
rng = np.random.default_rng(7)
B = 1000

def rj(p): return [json.loads(l) for l in open(p, encoding='utf-8')] if os.path.exists(p) else []

val = rj(f'{IN}/validation.jsonl'); real_pl = rj(f'{IN}/real_pl.jsonl'); real_us = rj(f'{IN}/real_us.jsonl')
personas = {p['respondent_id']: p for p in rj(f'{IN}/personas.jsonl')}
sil = rj(f'{OUT}/silicon.jsonl')
coded = {}
for r in rj(f'{OUT}/coded.jsonl'):
    lab = r['labels']
    if lab.get('parse_error'): continue
    coded[r['item_id']] = {**{d: int(bool(lab.get(d, 0))) for d in PROFILE},
                           'valence': str(lab.get('valence', 'none')).lower()}
parse_errors = sum(1 for r in rj(f'{OUT}/coded.jsonl') if r['labels'].get('parse_error'))

def kappa(a, b):
    a, b = list(a), list(b); n = len(a)
    if n == 0: return float('nan')
    cats = sorted(set(a) | set(b)); po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float('nan')

def f1(gold, pred):
    tp = sum(g and p for g, p in zip(gold, pred)); fp = sum((not g) and p for g, p in zip(gold, pred)); fn = sum(g and (not p) for g, p in zip(gold, pred))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float('nan')

R = {'n_coded': len(coded), 'parse_errors': parse_errors}

# ---------------- 1. coder validation ----------------
v_rows = [v for v in val if f"val_pl_{v['ann_id']}" in coded]
vd = {}
for d in DOMS:
    gold = [int(d in v['gold_domains']) for v in v_rows]
    pred = [coded[f"val_pl_{v['ann_id']}"][d] for v in v_rows]
    a1 = [int(d in v['ann1_domains']) for v in v_rows]; a2 = [int(d in v['ann2_domains']) for v in v_rows]
    en = [coded[f"val_en_{v['ann_id']}"][d] for v in v_rows if f"val_en_{v['ann_id']}" in coded]
    pl_for_en = [coded[f"val_pl_{v['ann_id']}"][d] for v in v_rows if f"val_en_{v['ann_id']}" in coded]
    vd[d] = dict(gold_rate=np.mean(gold), coder_rate=np.mean(pred), f1=f1(gold, pred), kappa_coder_gold=kappa(gold, pred),
                 kappa_human_human=kappa(a1, a2), kappa_pl_vs_en_coding=kappa(pl_for_en, en))
gv = [v['gold_valence'] for v in v_rows]; pv = [coded[f"val_pl_{v['ann_id']}"]['valence'] for v in v_rows]
R['validation'] = dict(n=len(v_rows), domains=vd, macro_f1=float(np.nanmean([vd[d]['f1'] for d in DOMS])),
                       valence_acc=float(np.mean([a == b for a, b in zip(gv, pv)])) if v_rows else None,
                       valence_kappa=kappa(gv, pv), valence_kappa_human_human=kappa([v['ann1_valence'] for v in v_rows], [v['ann2_valence'] for v in v_rows]))

# ---------------- 2. corpora ----------------
def words(t): return len(t.split())
corpora = collections.OrderedDict()
corpora['real_pl'] = [r for r in real_pl if r['persona_source']]
corpora['real_pl_all300'] = real_pl
for name in sorted({s['corpus'] for s in sil}):
    corpora[name] = [s for s in sil if s['corpus'] == name]
corpora['real_us'] = real_us

def profile(items):
    c = np.array([[coded[i['item_id']][d] for d in PROFILE] for i in items if i['item_id'] in coded], float)
    if len(c) == 0: return None
    tot = c.sum(0); return tot / tot.sum() if tot.sum() else tot

def jsd(p, q):
    p, q = np.asarray(p, float) + 1e-12, np.asarray(q, float) + 1e-12; p, q = p / p.sum(), q / q.sum(); m = (p + q) / 2
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))

def boot_jsd(a, b):
    a = [i for i in a if i['item_id'] in coded]; b = [i for i in b if i['item_id'] in coded]
    if not a or not b: return None
    est = jsd(profile(a), profile(b)); bs = []
    for _ in range(B):
        aa = [a[k] for k in rng.integers(0, len(a), len(a))]; bb = [b[k] for k in rng.integers(0, len(b), len(b))]
        bs.append(jsd(profile(aa), profile(bb)))
    return dict(jsd=est, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])

desc = {}
for name, items in corpora.items():
    ws = [words(i['text']) for i in items]
    cc = [coded[i['item_id']] for i in items if i['item_id'] in coded]
    desc[name] = dict(n=len(items), n_coded=len(cc), words_mean=float(np.mean(ws)) if ws else None, words_median=float(np.median(ws)) if ws else None,
                      share_le3_words=float(np.mean([w <= 3 for w in ws])) if ws else None,
                      **{f'share_{d}': float(np.mean([c[d] for c in cc])) if cc else None for d in PROFILE},
                      **{f'val_{v}': float(np.mean([c['valence'] == v for c in cc])) if cc else None for v in VALS})
R['descriptives'] = desc

# split-half baseline within real Polish answers (150 vs 150 from the 300)
rp = [i for i in real_pl if i['item_id'] in coded]; sh = []
for _ in range(B):
    idx = rng.permutation(len(rp)); h = len(rp) // 2
    sh.append(jsd(profile([rp[k] for k in idx[:h]]), profile([rp[k] for k in idx[h:]])))
R['split_half_real_pl'] = dict(mean=float(np.mean(sh)), p95=float(np.percentile(sh, 95))) if rp else None

comp = {}
for name in corpora:
    if name.startswith('silicon'):
        comp[name] = dict(vs_real_pl=boot_jsd(corpora[name], corpora['real_pl']), vs_real_us=boot_jsd(corpora[name], corpora['real_us']))
comp['real_us'] = dict(vs_real_pl=boot_jsd(corpora['real_us'], corpora['real_pl']))
R['jsd'] = comp

# H2: paired bootstrap of the language effect (same personas): JSD(EN, realPL) - JSD(PL, realPL)
h2 = {}
for gen in sorted({s['corpus'].split('_')[1] for s in sil}):
    en, pl_ = corpora.get(f'silicon_{gen}_en', []), corpora.get(f'silicon_{gen}_pl', [])
    ids = sorted(set(i['respondent_id'] for i in en) & set(i['respondent_id'] for i in pl_) & set(personas))
    E = {i['respondent_id']: i for i in en}; P = {i['respondent_id']: i for i in pl_}; RL = {i['respondent_id']: i for i in corpora['real_pl']}
    ids = [i for i in ids if i in RL and E[i]['item_id'] in coded and P[i]['item_id'] in coded and RL[i]['item_id'] in coded]
    if not ids: continue
    def diff(sample, ref):
        return jsd(profile([E[i] for i in sample]), profile([RL[i] for i in sample] if ref == 'pl' else ref)) - \
               jsd(profile([P[i] for i in sample]), profile([RL[i] for i in sample] if ref == 'pl' else ref))
    us_items = [i for i in corpora['real_us'] if i['item_id'] in coded]
    for ref_name, ref in [('real_pl', 'pl'), ('real_us', us_items)]:
        est = diff(ids, ref); bs = [diff([ids[k] for k in rng.integers(0, len(ids), len(ids))], ref) for _ in range(B)]
        h2[f'{gen}_{ref_name}'] = dict(n=len(ids), jsd_en_minus_jsd_pl=est, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
R['h2_language_effect'] = h2

# education gap in outcome / values / nonresponse: high minus low
def gap(items, d):
    hi = [coded[i['item_id']][d] for i in items if i['item_id'] in coded and i.get('edu') == 'high']
    lo = [coded[i['item_id']][d] for i in items if i['item_id'] in coded and i.get('edu') == 'low']
    return float(np.mean(hi) - np.mean(lo)) if hi and lo else None
R['edu_gap_high_minus_low'] = {name: {d: gap(items, d) for d in PROFILE} for name, items in corpora.items() if name != 'real_us'}

# individual-level agreement: silicon label vs the same person's real label
ind = {}
RL = {i['respondent_id']: i for i in corpora['real_pl']}
for name, items in corpora.items():
    if not name.startswith('silicon'): continue
    pairs = [(coded[RL[i['respondent_id']]['item_id']], coded[i['item_id']]) for i in items
             if i['respondent_id'] in RL and i['item_id'] in coded and RL[i['respondent_id']]['item_id'] in coded]
    ind[name] = {d: dict(acc=float(np.mean([a[d] == b[d] for a, b in pairs])), kappa=kappa([a[d] for a, b in pairs], [b[d] for a, b in pairs])) for d in DOMS} if pairs else None
R['individual_agreement'] = ind

# ---------------- 3. lexical markers ----------------
PL_MARK = {'elections/voting': r'wyb[oó]r|wybory|g[łl]os', 'freedom': r'wolno[śs][ćc]|wolny|wolne|swobod', 'equality': r'r[óo]wno',
           'law/rights': r'praw', 'majority/people': r'wi[ęe]kszo|lud|nar[óo]d|obywatel', 'corruption/negative': r'korup|z[łl]odz|kradn|fikcj|iluzj|oszust|k[łl]am'}
EN_MARK = {'elections/voting': r'elect|vot|ballot', 'freedom': r'free|libert', 'equality': r'equal', 'law/rights': r'\blaw|right',
           'majority/people': r'majorit|people|citizen', 'corruption/negative': r'corrupt|\blie|\bliar|fake|sham|illusion|steal'}
def marker_shares(items, pats, field='text'):
    return {k: float(np.mean([bool(re.search(p, (i.get(field) or '').lower())) for i in items])) if items else None for k, p in pats.items()}
lex = {}
for name, items in corpora.items():
    if name.endswith('_pl') or name == 'real_pl_all300':
        lex[name] = marker_shares(items, PL_MARK)
    if name.endswith('_en') or name == 'real_us':
        lex[name] = marker_shares(items, EN_MARK)
real_pl_en = [dict(text=personas[i['respondent_id']]['real_q4_en']) for i in corpora['real_pl'] if i['respondent_id'] in personas]
lex['real_pl_machine_translated_en'] = marker_shares(real_pl_en, EN_MARK)
R['lexical_markers'] = lex

# type diversity at equal token budget, within language
def toks(t): return re.findall(r'\w+', t.lower())
def types_at(items, budget, reps=200):
    alltok = [toks(i['text']) for i in items]; res = []
    for _ in range(reps):
        order = rng.permutation(len(alltok)); bag = []
        for k in order:
            bag += alltok[k]
            if len(bag) >= budget: break
        res.append(len(set(bag[:budget])))
    return float(np.mean(res))
pl_c = [n for n in corpora if n.endswith('_pl') and n != 'real_pl_all300'] + ['real_pl']
en_c = [n for n in corpora if n.endswith('_en')] + ['real_us']
div = {}
for group in (pl_c, en_c):
    group = [g for g in dict.fromkeys(group) if corpora.get(g)]
    if not group: continue
    budget = min(sum(len(toks(i['text'])) for i in corpora[g]) for g in group)
    for g in group: div[g] = dict(token_budget=budget, types=types_at(corpora[g], budget))
R['type_diversity_equal_budget'] = div

# ---------------- 4. embedding homogeneity ----------------
emb = collections.defaultdict(dict)
for r in rj(f'{OUT}/embeddings.jsonl'):
    v = np.array(r['vec'], float); emb[r['corpus']][r['item_id']] = v / (np.linalg.norm(v) + 1e-12)
hom = {}
for name, items in corpora.items():
    base = 'real_pl' if name.startswith('real_pl') else name
    V = np.array([emb[base][i['item_id']] for i in items if i['item_id'] in emb.get(base, {})])
    if len(V) > 2:
        S = V @ V.T; iu = np.triu_indices(len(V), 1); hom[name] = dict(n=len(V), mean_pairwise_cosine=float(S[iu].mean()))
R['embedding_homogeneity'] = hom

# ---------------- 5. examples ----------------
ex = []
for rid, p in list(personas.items())[:12]:
    row = dict(respondent_id=rid, edu=p['edu'], gender=p['gender'], age=p['age'], real=p['real_q4'], real_en=p['real_q4_en'])
    for s in sil:
        if s['respondent_id'] == rid: row[s['corpus']] = s['text']
    ex.append(row)
R['examples'] = ex

json.dump(R, open(f'{OUT}/results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)

# ---------------- markdown summary ----------------
L = ['# Feasibility results', '', f"coded items: {R['n_coded']}, parse errors: {parse_errors}", '', '## Coder validation (Polish Q4, n=%d)' % R['validation']['n'],
     '| domain | gold rate | coder rate | F1 | kappa coder-gold | kappa human-human | kappa PL vs EN coding |', '|---|---|---|---|---|---|---|']
for d in DOMS:
    x = vd[d]; L.append(f"| {d} | {x['gold_rate']:.2f} | {x['coder_rate']:.2f} | {x['f1']:.2f} | {x['kappa_coder_gold']:.2f} | {x['kappa_human_human']:.2f} | {x['kappa_pl_vs_en_coding']:.2f} |")
L += [f"macro-F1 {R['validation']['macro_f1']:.2f}; valence acc {R['validation']['valence_acc']:.2f}, kappa {R['validation']['valence_kappa']:.2f} (human-human {R['validation']['valence_kappa_human_human']:.2f})", '',
      '## Corpora', '| corpus | n | words mean | median | <=3 words | gov | val | out | nonresp | positive | negative |', '|---|---|---|---|---|---|---|---|---|---|---|']
for n, x in desc.items():
    f = lambda k: f"{x[k]:.2f}" if x.get(k) is not None else '-'
    L.append(f"| {n} | {x['n']} | {f('words_mean')} | {f('words_median')} | {f('share_le3_words')} | {f('share_governance')} | {f('share_values')} | {f('share_outcome')} | {f('share_nonresponse')} | {f('val_positive')} | {f('val_negative')} |")
L += ['', f"split-half JSD within real PL: mean {R['split_half_real_pl']['mean']:.4f}, 95th pct {R['split_half_real_pl']['p95']:.4f}" if R['split_half_real_pl'] else '', '', '## JSD of domain profiles (bootstrap 95% CI)']
for n, x in comp.items():
    for k, v in x.items():
        if v: L.append(f"- {n} {k}: {v['jsd']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]")
L += ['', '## H2 language effect: JSD(EN prompt) - JSD(PL prompt), paired bootstrap']
for k, v in h2.items(): L.append(f"- {k}: {v['jsd_en_minus_jsd_pl']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] (n={v['n']})")
L += ['', '## Education gap (high - low)']
for n, x in R['edu_gap_high_minus_low'].items(): L.append(f"- {n}: " + ', '.join(f"{d} {v:+.2f}" for d, v in x.items() if v is not None))
L += ['', '## Individual-level agreement with the same person (kappa)']
for n, x in ind.items():
    if x: L.append(f"- {n}: " + ', '.join(f"{d} acc {v['acc']:.2f} k {v['kappa']:.2f}" for d, v in x.items()))
L += ['', '## Lexical markers (share of answers)']
for n, x in lex.items(): L.append(f"- {n}: " + ', '.join(f"{k} {v:.2f}" for k, v in x.items() if v is not None))
L += ['', '## Type diversity at equal token budget', *[f"- {n}: {x['types']:.1f} types / {x['token_budget']} tokens" for n, x in div.items()],
      '', '## Embedding homogeneity (mean pairwise cosine; higher = more uniform)', *[f"- {n}: {x['mean_pairwise_cosine']:.3f} (n={x['n']})" for n, x in hom.items()]]
open(f'{OUT}/results.md', 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
