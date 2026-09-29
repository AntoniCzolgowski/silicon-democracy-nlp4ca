# Silicon Poles on democracy

Do LLM personas built from real Polish survey respondents describe **democracy** the way those respondents did, and does the **prompt language** (Polish vs English) change that?

Feasibility test for a research proposal in *NLP for Cultural Analytics* (CSCI 7000, CU Boulder, Fall 2026).

- **H1 (flattening):** silicon answers are longer, more homogeneous and almost never non-substantive, and their mix of Democracy Tree domains differs from the real mix by more than two random halves of the real sample.
- **H2 (prompt language):** the same personas prompted in English drift away from real Poles and towards real Americans, relative to the Polish prompt.

## Data
LES Democracy Survey 2020 (10 countries, open-ended question "When you hear the word *democracy*, what are you thinking about?"), released under **CC0** as the replication package of Dahlberg, Dürlich, Axelsson, Zhao & Nivre (2026), *Political Analysis*, doi:10.1017/pan.2026.10043; data: Dürlich et al. (2026), Harvard Dataverse, doi:10.7910/DVN/XOWF9C. Human annotations follow the Democracy Tree (Dahlberg & Mörkenstam 2024, *Democratization* 31(8), doi:10.1080/13510347.2024.2342485).

`data/inputs/` holds the derived samples (built by `src/prep.py` from `All_Countries_Merged.xlsm` and `polish.jsonl`, which go in `data/raw/`, not committed):
- `personas.jsonl`: 150 real Polish respondents (stratified by education × gender) with demographics and their real answer
- `real_pl.jsonl` (300), `real_us.jsonl` (150): real answers to code
- `validation.jsonl`: 200 human-annotated Polish answers (+ English translation) with gold and both annotators' labels

## Run (CU Alpine, Open OnDemand, no terminal)
Open `notebooks/feasibility_alpine.ipynb` in a **Jupyter Session** on a GPU (instructions at the top of the notebook). It starts its own Ollama server, downloads the models, generates and codes the answers, embeds them and runs `src/analyze.py`. Every stage is checkpointed in `outputs/`.

Models (Ollama, 4-bit): Gemma 3 12B and Bielik 11B v3 as silicon respondents, Gemma 3 27B as Democracy Tree coder, embeddinggemma for embeddings.

## Layout
```
notebooks/feasibility_alpine.ipynb   generation, coding, embeddings, analysis
src/personas.py                      persona and question wording (PL/EN) and the coder prompt
src/prep.py                          builds data/inputs from the raw replication data
src/analyze.py                       validation, JSD + bootstrap, H2 paired test, lexical and embedding measures
tools/build_notebook.py              regenerates the notebook from source
outputs/                             results (silicon.jsonl, coded.jsonl, embeddings.jsonl, results.md, manifest.json)
```
