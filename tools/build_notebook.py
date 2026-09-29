"""Builds notebooks/feasibility_alpine.ipynb (keeps the notebook source reviewable in git)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Silicon Poles on democracy: feasibility test (NLP4CA proposal 1)

Do LLM personas built from real Polish survey respondents describe **democracy** the way those respondents did,
and does the **prompt language** (Polish vs English) change that?

This notebook runs everything on CU Alpine through Open OnDemand, no terminal needed:
starts its own Ollama server on the GPU, downloads the models, generates silicon answers,
codes all answers with the Democracy Tree scheme, embeds them, and runs the analysis.

**Data:** LES Democracy Survey 2020, CC0 replication package of Dahlberg et al. (2026, *Political Analysis*),
Dürlich et al. (2026) Harvard Dataverse doi:10.7910/DVN/XOWF9C. Derived inputs are in `data/inputs/` (built by `src/prep.py`).

### How to launch on Alpine (Open OnDemand)
1. https://ondemand.rc.colorado.edu → **Interactive Apps → Jupyter Session**
2. Configuration type **Custom configuration**: Cluster `alpine`, Account `ucb-general`, Partition `aa100`,
   Account: your allocation (`ucb757_asc1`) or `ucb-general`. Any of these GPUs works (the A100 pool is often the busiest,
   so queue two or three at once and delete the others as soon as one starts; never Run All in two sessions at the same time):
   - Partition `aa100`, QoS `gpu-normal`, gres `gpu:a100-40gb:1`, Time `2`, cores `8`
   - Partition `artxpro6000`, QoS `gpu-normal`, gres `gpu:rtx_pro_6000_2g.48gb:1` (or `gpu:rtx_pro_6000_1g.24gb:1`), Time `2`, cores `8`
   - Partition `ah200`, QoS `gpu-normal`, gres `gpu:h200_2g.35gb:1`, Time `2`, cores `8`
   - Partition `al40`, QoS `gpu-normal`, gres `gpu:l40:1`, Time `2`, cores `8`
   - testing slice (usually starts at once, 1 h max, the notebook resumes across sessions): Partition `aa100`, QoS `gpu-testing`, gres `gpu:a100_3g.20gb:1`, Time `1`, cores `10`
3. **Connect to Jupyter**, then File → Open from Path → `/projects/<you>`; clone this repo there (cell below) and open this notebook.
4. Run with `SMOKE = True` first (3 items per stage), then set `SMOKE = False` and **Run All** again.
   Every stage checkpoints to `outputs/`: if the session ends, just run again and it continues.
""")

code(r"""
# (only once) clone the repo into /projects/$USER, then open notebooks/feasibility_alpine.ipynb from the file browser
# !cd /projects/$USER && git clone https://github.com/AntoniCzolgowski/silicon-democracy-nlp4ca.git
# update to the latest version later (keeps outputs/; resets this notebook, so set SMOKE again afterwards):
# !cd /projects/$USER/silicon-democracy-nlp4ca && git fetch -q && git reset --hard origin/main
""")

md("## 1. Settings")
code(r"""
SMOKE = True            # True: 3 items per stage (end-to-end check). Then set False and Run All.

GEN_MODELS = {'gemma': 'gemma3:12b',                                   # global open model
              'bielik': 'SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M'}  # Polish-native open model
CODER_MODEL = 'gemma3:27b'       # larger model as the Democracy Tree coder (validated against human codes)
EMBED_MODEL = 'embeddinggemma'   # multilingual embeddings for the homogeneity measure
TEMPERATURE = 1.0                # sampling temperature for the silicon respondents

import os, sys, json, time, socket, shutil, hashlib, subprocess, threading, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path.cwd()
while not (ROOT / 'data' / 'inputs').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
assert (ROOT / 'data' / 'inputs').exists(), 'Open this notebook from inside the cloned repo'
IN, OUT = ROOT / 'data' / 'inputs', ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'src'))
USER = os.environ.get('USER', 'me')
OLLAMA_MODELS = os.environ.get('OLLAMA_MODELS_DIR', f'/projects/{USER}/ollama_models')
LIMIT = 3 if SMOKE else None
print('repo:', ROOT, '| smoke test:', SMOKE, '| models dir:', OLLAMA_MODELS)
""")

md("## 2. GPU and Ollama server\nThe server is started from this notebook with CURC's `ollama` module and our own model folder in `/projects` (~33 GB).")
code(r"""
try:
    gpu = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
except FileNotFoundError:
    gpu = 'NO GPU'
    if not os.environ.get('OLLAMA_URL'):
        raise RuntimeError('No GPU in this session: relaunch the Jupyter Session on a GPU partition with a gres (see top of notebook)')
print(gpu)
import re
mig = re.findall(r'MIG\s+\d+g\.(\d+)gb', gpu)                   # e.g. 'MIG 3g.20gb' -> 20 GB slice
if mig:
    VRAM = int(mig[0])
else:                                                          # full GPU: ask nvidia-smi for its memory
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                           stdout=subprocess.PIPE, universal_newlines=True).stdout.split()
        VRAM = int(int(q[0]) / 1024)
    except Exception:
        VRAM = 40
# the 27B coder needs ~17 GB + ~2 GB per parallel request
PAR, GEN_WORKERS, CODE_WORKERS = (1, 2, 1) if VRAM <= 20 else (2, 3, 2) if VRAM <= 30 else (4, 4, 3)
print(f'GPU memory for us: ~{VRAM} GB ({"MIG slice" if mig else "full GPU"}) | parallel requests: {PAR}')
""")
code(r"""
def free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close(); return p

def module_env():
    # environment that CURC's `module load ollama` sets up (PATH, LD_LIBRARY_PATH, ...)
    cmd = f'export OLLAMA_MODELS={OLLAMA_MODELS}; module load ollama >/dev/null 2>&1; env -0'
    raw = subprocess.run(['bash', '-lc', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180).stdout
    return dict(kv.split('=', 1) for kv in raw.decode(errors='ignore').split('\0') if '=' in kv and not kv.startswith('BASH_FUNC'))

def api(path, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

if os.environ.get('OLLAMA_URL'):                     # use an already running server (testing)
    BASE, server = os.environ['OLLAMA_URL'], None
else:
    env = module_env()
    exe = shutil.which('ollama', path=env.get('PATH', ''))
    assert exe, 'CURC ollama module not found: run this on an Alpine compute node (Jupyter Session on a GPU)'
    subprocess.run(['pkill', '-u', USER, '-f', 'ollama serve'])     # stop servers left by the module or earlier runs
    time.sleep(2)
    port = free_port()
    env.update(OLLAMA_HOST=f'127.0.0.1:{port}', OLLAMA_MODELS=OLLAMA_MODELS, OLLAMA_NUM_PARALLEL=str(PAR),
               OLLAMA_MAX_LOADED_MODELS='1', OLLAMA_KEEP_ALIVE='30m')
    Path(OLLAMA_MODELS).mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([exe, 'serve'], env=env, stdout=open(OUT / 'ollama_server.log', 'a'), stderr=subprocess.STDOUT)
    BASE = f'http://127.0.0.1:{port}'
for _ in range(60):
    try:
        print('Ollama', api('/api/version')['version'], 'at', BASE); break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError('Ollama did not start; see outputs/ollama_server.log')
""")

md("## 3. Download models (first time only, 5-15 min)")
code(r"""
def pull(model):
    have = {m['name'] for m in api('/api/tags').get('models', [])}
    if model in have or f'{model}:latest' in have:
        print('have', model); return
    req = urllib.request.Request(BASE + '/api/pull', data=json.dumps({'model': model, 'stream': True}).encode(),
                                 headers={'Content-Type': 'application/json'})
    last = -10
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            ev = json.loads(line)
            if 'error' in ev: raise RuntimeError(f"{model}: {ev['error']}")
            if ev.get('total') and ev.get('completed'):
                pct = 100 * ev['completed'] / ev['total']
                if pct - last >= 10: print(f'  {model}: {pct:.0f}%'); last = pct
    print('pulled', model)

for m in [*GEN_MODELS.values(), CODER_MODEL, EMBED_MODEL]:
    pull(m)
print([m['name'] for m in api('/api/tags')['models']])
""")

md("## 4. Inputs and prompts\nPersona prompts carry the respondent's gender, age, education, employment, political interest and three immigration attitudes; the question is the exact LES 2020 wording. No instruction about answer length is given, so length is an outcome.")
code(r"""
from personas import persona_messages, CODER_SYSTEM

def read_jsonl(p):
    p = Path(p); return [json.loads(l) for l in open(p, encoding='utf-8')] if p.exists() else []

personas = read_jsonl(IN / 'personas.jsonl')
print(len(personas), 'personas;', len(read_jsonl(IN / 'real_pl.jsonl')), 'real PL;', len(read_jsonl(IN / 'real_us.jsonl')), 'real US;',
      len(read_jsonl(IN / 'validation.jsonl')), 'human-coded validation answers')
for lang in ('pl', 'en'):
    m = persona_messages(personas[0], lang)
    print(f'\n[{lang}] SYSTEM: {m[0]["content"]}\n[{lang}] USER:   {m[1]["content"]}')
print('\nreal answer of this person:', personas[0]['real_q4'])
""")

md("## 5. Helpers (parallel, checkpointed)")
code(r"""
LOCK = threading.Lock()
def append(path, obj):
    with LOCK, open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def run_pool(jobs, fn, workers, label):
    t0, done, errors = time.time(), 0, 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fn, j) for j in jobs]
        for f in as_completed(futs):
            done += 1
            try: f.result()
            except Exception as e:
                errors += 1; print('ERROR', repr(e)[:200])
            if done % 25 == 0 or done == len(jobs):
                print(f'[{label}] {done}/{len(jobs)}  {time.time() - t0:.0f}s  errors={errors}', flush=True)

def seed_for(*parts):
    return int(hashlib.md5('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)
""")

md("## 6. Generate silicon answers (2 models × 2 prompt languages × 150 personas)")
code(r"""
SIL = OUT / 'silicon.jsonl'
for short, model in GEN_MODELS.items():
    done = {(r['model'], r['lang'], r['respondent_id']) for r in read_jsonl(SIL)}
    jobs = [(p, lang) for p in personas[:LIMIT] for lang in ('pl', 'en') if (model, lang, p['respondent_id']) not in done]
    print(model, '->', len(jobs), 'to generate')

    def one(job, model=model, short=short):
        p, lang = job
        msgs = persona_messages(p, lang); t = time.time()
        r = api('/api/chat', {'model': model, 'messages': msgs, 'stream': False,
                              'options': {'temperature': TEMPERATURE, 'top_p': 0.95, 'num_predict': 400,
                                          'seed': seed_for(model, lang, p['respondent_id'])}})
        append(SIL, {'item_id': f"sil_{short}_{lang}_{p['respondent_id']}", 'corpus': f'silicon_{short}_{lang}',
                     'model': model, 'lang': lang, 'respondent_id': p['respondent_id'], 'edu': p['edu'],
                     'gender': p['gender'], 'age': p['age'], 'text': r['message']['content'].strip(),
                     'temperature': TEMPERATURE, 'seconds': round(time.time() - t, 2)})
    run_pool(jobs, one, GEN_WORKERS, short)

for r in read_jsonl(SIL)[:4]:
    print(f"\n{r['corpus']} | {r['gender']} {r['age']} {r['edu']}:\n  {r['text'][:400]}")
""")

md("## 7. Code every answer with the Democracy Tree (validation set in Polish and English, real PL, real US, all silicon)")
code(r"""
CODED = OUT / 'coded.jsonl'

def parse_json(s):
    s = s.strip()
    i, j = s.find('{'), s.rfind('}')
    return json.loads(s[i:j + 1])

items = []
for v in read_jsonl(IN / 'validation.jsonl'):
    items.append({'item_id': f"val_pl_{v['ann_id']}", 'corpus': 'validation_pl', 'text': v['text_pl']})
    items.append({'item_id': f"val_en_{v['ann_id']}", 'corpus': 'validation_en', 'text': v['text_en']})
items += read_jsonl(IN / 'real_pl.jsonl') + read_jsonl(IN / 'real_us.jsonl') + read_jsonl(SIL)
if LIMIT:
    items = items[:LIMIT]
done = {(r['item_id'], r['coder']) for r in read_jsonl(CODED)}
jobs = [it for it in items if (it['item_id'], CODER_MODEL) not in done]
print(len(jobs), 'to code with', CODER_MODEL)

def code_one(it):
    r = api('/api/chat', {'model': CODER_MODEL, 'stream': False, 'format': 'json',
                          'messages': [{'role': 'system', 'content': CODER_SYSTEM},
                                       {'role': 'user', 'content': 'Answer to code:\n<answer>' + it['text'] + '</answer>'}],
                          'options': {'temperature': 0, 'num_predict': 120, 'num_ctx': 4096}})
    raw = r['message']['content']
    try: lab = parse_json(raw)
    except Exception: lab = {'parse_error': True}
    append(CODED, {'item_id': it['item_id'], 'corpus': it['corpus'], 'coder': CODER_MODEL, 'labels': lab, 'raw': raw})

run_pool(jobs, code_one, CODE_WORKERS, 'code')
print(read_jsonl(CODED)[:2])
""")

md("## 8. Embeddings (homogeneity measure)")
code(r"""
EMB = OUT / 'embeddings.jsonl'
emb_items = read_jsonl(IN / 'real_pl.jsonl') + read_jsonl(IN / 'real_us.jsonl') + read_jsonl(SIL)
if LIMIT: emb_items = emb_items[:LIMIT]
done = {r['item_id'] for r in read_jsonl(EMB)}
todo = [it for it in emb_items if it['item_id'] not in done]
print(len(todo), 'to embed')
for k in range(0, len(todo), 32):
    batch = todo[k:k + 32]
    r = api('/api/embed', {'model': EMBED_MODEL, 'input': [b['text'] for b in batch]})
    for b, e in zip(batch, r['embeddings']):
        append(EMB, {'item_id': b['item_id'], 'corpus': b['corpus'], 'model': EMBED_MODEL, 'vec': [round(x, 5) for x in e]})
print('embeddings:', len(read_jsonl(EMB)))
""")

md("## 9. Run manifest (for reproducibility)")
code(r"""
manifest = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'smoke': SMOKE, 'gpu': gpu.strip(),
            'ollama_version': api('/api/version')['version'], 'temperature': TEMPERATURE,
            'models': [{**{k: m.get(k) for k in ('name', 'digest', 'size')}, 'details': m.get('details')} for m in api('/api/tags')['models']],
            'counts': {p.name: sum(1 for _ in open(p, encoding='utf-8')) for p in OUT.glob('*.jsonl')}}
json.dump(manifest, open(OUT / 'manifest.json', 'w'), indent=1)
manifest['counts']
""")

md("## 10. Analysis\nCoder validation against the human codes, corpus descriptives, JSD of Democracy Tree profiles with a split-half baseline, the paired prompt-language effect (H2), education gap, individual-level agreement, lexical markers, type diversity and embedding homogeneity.")
code(r"""
from IPython.display import Markdown, display
if SMOKE:
    print('Smoke test finished. Set SMOKE = False and Run All for the full run; the analysis runs on the full data.')
else:
    subprocess.run([sys.executable, str(ROOT / 'src' / 'analyze.py')], check=True, stdout=subprocess.DEVNULL)
    display(Markdown((OUT / 'results.md').read_text(encoding='utf-8')))
""")

md("## 11. Stop the server and save results to GitHub\nOptional: commit `outputs/` from here (needs your git credentials on Alpine), or download the folder from the Jupyter file browser.")
code(r"""
if server is not None:
    server.terminate()
    print('Ollama server stopped')
# !cd {ROOT} && git add outputs && git commit -m "Feasibility run outputs" && git push
""")

nb['cells'] = C
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                  'language_info': {'name': 'python'}}
nbf.write(nb, 'notebooks/feasibility_alpine.ipynb')
print('written', len(C), 'cells')
