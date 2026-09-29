"""Build inputs for the NLP4CA proposal 1 feasibility test (run locally).

Inputs : data/raw/All_Countries_Merged.xlsm, data/raw/polish.jsonl (download from the Dataverse package, see README)  (LES Democracy Survey 2020, CC0,
         Dahlberg et al. 2026 replication package, doi:10.7910/DVN/XOWF9C)
Outputs: data/inputs/personas.jsonl    150 real Polish respondents (demographics + real Q4 answer) -> persona prompts
         data/inputs/real_pl.jsonl     300 real Polish Q4 answers (the 150 persona sources + 150 more) for coding
         data/inputs/real_us.jsonl     150 real US Q4 answers (English-language anchor) for coding
         data/inputs/validation.jsonl  200 human-annotated Polish Q4 answers (+ English translation) with gold labels
"""
import json, os, random, re
import pandas as pd

SEED = 42
rng = random.Random(SEED)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = ROOT + '/data/raw/'
INP = ROOT + '/data/inputs/'

df = pd.read_excel(RAW + 'All_Countries_Merged.xlsm', engine='openpyxl')

def clean(x):
    if pd.isna(x): return ''
    return re.sub(r'\s+', ' ', str(x)).strip()

def usable(sub, col):
    t = sub[col].map(clean)
    noise = sub['q4_remove_noise'].astype(str).str.strip() == '1'
    return sub[(t != '') & (t.str.lower() != 'nan') & ~noise].assign(q4=t)

# ---------- Polish respondents ----------
pl = usable(df[df.country == 'Poland'], 'q4_pl_pl')
for c in ['q5', 'q6', 'q7', 'q8', 'q13']:
    pl = pl[pl[c].astype(str).str.strip() != '.']
pl = pl.copy()
pl['age'] = pl['q2'].astype(int) + 17          # codebook: 1 = 18 ... 48 = 65
pl['gender'] = pl['q1'].map({1: 'female', 2: 'male'})
pl['edu'] = pl['q3_education_quota'].map({1: 'low', 2: 'high'})
print('usable Polish respondents:', len(pl))

# stratified sample by education x gender, proportional
def strat_sample(frame, n):
    out = []
    groups = list(frame.groupby(['edu', 'gender']))
    for (e, g), sub in groups:
        k = round(n * len(sub) / len(frame))
        ids = list(sub.index); rng.shuffle(ids); out += ids[:k]
    rng.shuffle(out)
    return out[:n]

idx300 = strat_sample(pl, 300)
idx150 = idx300[:150]

def person(r):
    return dict(respondent_id=int(r.respondent_id), gender=r.gender, age=int(r.age), edu=r.edu,
                employment=int(r.q13), pol_interest=int(r.q8),
                imm_economy=int(r.q5), imm_culture=int(r.q6), imm_crime=int(r.q7),
                real_q4=r.q4, real_q4_en=clean(r.q4_translated))

with open(INP + 'personas.jsonl', 'w', encoding='utf-8') as f:
    for i in idx150: f.write(json.dumps(person(pl.loc[i]), ensure_ascii=False) + '\n')
with open(INP + 'real_pl.jsonl', 'w', encoding='utf-8') as f:
    for k, i in enumerate(idx300):
        r = pl.loc[i]
        f.write(json.dumps(dict(item_id=f'realpl_{int(r.respondent_id)}', corpus='real_pl', lang='pl',
                                respondent_id=int(r.respondent_id), edu=r.edu, gender=r.gender, age=int(r.age),
                                persona_source=k < 150, text=r.q4), ensure_ascii=False) + '\n')

# ---------- US anchor ----------
us = usable(df[df.country == 'United States'], 'q4_en_us')
us_ids = list(us.index); rng.shuffle(us_ids)
with open(INP + 'real_us.jsonl', 'w', encoding='utf-8') as f:
    for i in us_ids[:150]:
        r = us.loc[i]
        f.write(json.dumps(dict(item_id=f'realus_{int(r.respondent_id)}', corpus='real_us', lang='en',
                                respondent_id=int(r.respondent_id), text=r.q4), ensure_ascii=False) + '\n')

# ---------- validation set from human annotations ----------
DOMAIN = {'1': 'governance', '2': 'values', '3': 'outcome', '4': 'other', '5': 'nonresponse', '6': 'other'}

def q1_parts(text):
    # "Q1: <pl>\n \nQ2: <pl>\n \n[TRANSLATION IN ENGLISH BELOW]\n \nQ1: <en>\n \nQ2: <en>"
    pl_part, _, en_part = text.partition('[TRANSLATION IN ENGLISH BELOW]')
    m = re.search(r'Q1:(.*?)(?:\n\s*\nQ2:|$)', pl_part, re.S)
    q1_start = pl_part.find('Q1:')
    q2_start = pl_part.find('Q2:')
    m_en = re.search(r'Q1:(.*?)(?:\n\s*\nQ2:|$)', en_part, re.S)
    return (m.group(1).strip() if m else '', m_en.group(1).strip() if m_en else '', q1_start, q2_start)

def labels_for_q1(spans, q2_start):
    doms, vals = set(), []
    for s, e, lab in spans or []:
        if q2_start != -1 and s >= q2_start: continue
        if lab.startswith('Valance'): vals.append(lab.split()[-1].lower()); continue
        d = DOMAIN.get(lab.strip()[0])
        if d: doms.add(d)
    v = set(vals)
    if not doms and not v: val = 'none'
    elif {'positive', 'negative'} <= v: val = 'mixed'
    elif 'negative' in v: val = 'negative'
    elif 'positive' in v: val = 'positive'
    elif 'neutral' in v: val = 'neutral'
    else: val = 'none'
    if 'nonresponse' in doms: val = 'none'
    return sorted(doms), val

rows = [json.loads(l.replace(': NaN', ': null')) for l in open(RAW + 'polish.jsonl', encoding='utf-8')]
vrows = []
for r in rows:
    q1_pl, q1_en, q1s, q2s = q1_parts(r['text'])
    if not q1_pl: continue
    gold = r.get('label_train') or r.get('label_val') or r.get('label_test') or r.get('label_alemel+joint')
    g_d, g_v = labels_for_q1(gold, q2s)
    a_d, a_v = labels_for_q1(r.get('label_alemel'), q2s)
    b_d, b_v = labels_for_q1(r.get('label_aleprz'), q2s)
    vrows.append(dict(ann_id=r['id'], text_pl=q1_pl, text_en=q1_en, gold_domains=g_d, gold_valence=g_v,
                      ann1_domains=a_d, ann1_valence=a_v, ann2_domains=b_d, ann2_valence=b_v))
rng.shuffle(vrows)
with open(INP + 'validation.jsonl', 'w', encoding='utf-8') as f:
    for v in vrows[:200]: f.write(json.dumps(v, ensure_ascii=False) + '\n')
print('validation docs available:', len(vrows), '-> using 200')
print('personas: 150, real_pl: 300, real_us: 150')
