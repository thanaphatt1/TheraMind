# TheraMind — CPsyCounR Baseline Progress

## Objective

Reproduce TheraMind's client simulation pipeline on a stratified sample of **CPsyCounR** cases, to serve as a baseline for benchmarking against the proposed `adaptive_plannerv2` framework.

---

## Pipeline Overview

```
CPsyCounR.json (3,134 Chinese cases)
        │
        ▼  sample_cpsy.py
classified_cpsy.json          ← all 3,134 cases LLM-classified into 10 categories
        │
        ▼  sample_cpsy.py
cpsy_sample_100.json          ← 100 cases stratified-sampled (10 per category), Chinese
        │
        ▼  translate_cpsy.py
cpsy_sample_100_en.json       ← 100 cases fully translated to English
        │
        ▼  prepare_cpsy_data.py   [NOT YET RUN]
cpsy_client_records.json      ← medical records per case (TheraMind format)
cpsy_guidance.json            ← 6-session guidance per case (TheraMind format)
        │
        ▼  run_theramind_simulation.py  [NOT YET RUN]
experiment_results/theramind_sim/
```

---

## Step 1 — LLM Classification

**Script**: `thesis-framework/sample_cpsy.py`
**Output**: `classified_cpsy.json`
**Model**: `gemini-3-flash-preview` (temperature=0.1, zero-shot)

Classified all 3,134 cases into 10 clinical categories:

| Category | Count |
|---|---|
| Love | 995 |
| Youth | 488 |
| Emotion | 447 |
| Self-growth | 278 |
| Family | 270 |
| Anxiety | 216 |
| Stress | 191 |
| Social | 118 |
| Rare | 71 |
| Addiction | 60 |
| **Total** | **3,134** |

---

## Step 2 — Stratified Sampling

**Script**: `thesis-framework/sample_cpsy.py`
**Output**: `cpsy_sample_100.json`

- 10 items per category → **100 total**
- `random.seed(42)` for reproducibility
- Each item assigned a `sample_id` (e.g., `love_01`, `emotion_03`) used as `patient_id` downstream

---

## Step 3 — English Translation

**Script**: `thesis-framework/translate_cpsy.py`
**Output**: `cpsy_sample_100_en.json`
**Model**: `deepseek-chat` (DeepSeek-V3.2, temperature=0.1)

Translated 6 fields per case in a single API call:

| Field | Description |
|---|---|
| `title` | Case title |
| `original_category_zh` | Original Chinese category tags |
| `techniques` | Therapeutic techniques used |
| `summary` | Case background and client description |
| `process` | Full counseling process narrative |
| `insights` | Counselor reflections |

Fields `category` and `sample_id` were already English and preserved as-is.

**Status**: All 100 items confirmed fully translated (0 remaining Chinese characters).

---

## Step 4 — Generate Medical Records + Session Guidance

**Script**: `thesis-framework/prepare_cpsy_data.py`
**Outputs**: `cpsy_client_records.json`, `cpsy_guidance.json`
**Model**: `deepseek-chat` (DeepSeek-V3.2)
**Status**: Not yet run

Replicates two original TheraMind preprocessing scripts:

### Medical Records (replicates `case_produce.py`)
Generates a structured patient medical record from each case:
- `patient pseudonym` — anonymised English name
- `patient age` — inferred from case context
- `mental health history`
- `physical health history`
- `current problems and symptoms`

Output format: `{ "love_01": { "medical information": { ... } } }`

### Session Guidance (replicates `data_produce.py`)
Generates 6-session counseling guidance from the patient's perspective, derived from the consultation process narrative.

Output format: `{ "love_01": { "Conversation guidance": { "session_1": "...", ..., "session_6": "..." } } }`

**Run command**:
```bash
cd thesis-framework
python prepare_cpsy_data.py --workers 8 --sessions 6
```

---

## Step 5 — Run Simulation

**Script**: `thesis-framework/run_theramind_simulation.py`
**Status**: Not yet run

```bash
python run_theramind_simulation.py \
  --guidance ../theramind/cpsy_guidance.json \
  --records  ../theramind/cpsy_client_records.json \
  --patient  all \
  --sessions 6 \
  --rounds   8
```

> Note: `run_theramind_simulation.py` currently hardcodes `GUIDANCE_PATH` and `RECORDS_PATH`.
> `--guidance` and `--records` CLI arguments need to be added before running on CPsyCounR data.

---

## Cost Summary

All costs tracked in `thesis-framework/total_costs.csv`.

| Step | Script | Model | Items | Input Tokens | Output Tokens | Cost (USD) |
|---|---|---|---|---|---|---|
| Classification | `sample_cpsy.py` | gemini-3-flash-preview | 3,134 | 1,246,726 | 3,145,898 | $10.06 |
| Translation (initial) | `translate_cpsy.py` | deepseek-chat | 100 | 104,854 | 118,064 | $0.079 |
| Translation (retry ×4) | `translate_cpsy.py` | deepseek-chat | 4 | 17,773 | 26,475 | $0.016 |
| **Total so far** | | | | | | **$10.15** |
| Records + Guidance | `prepare_cpsy_data.py` | deepseek-chat | 100 | — | — | Pending |
| Simulation | `run_theramind_simulation.py` | deepseek-chat / gemini | 100 × 6 sessions | — | — | Pending |

> Classification cost is high due to Gemini's thinking tokens (3.1M thought tokens at thinking=high).
> Translation is cheap on DeepSeek-V3.2 at $0.28/$0.42 per 1M input/output tokens (cache miss rate).

---

## File Index

| File | Status | Description |
|---|---|---|
| `CPsyCounR.json` | Source | 3,134 original Chinese cases |
| `classified_cpsy.json` | Done | All 3,134 cases with LLM-assigned category |
| `cpsy_sample_100.json` | Done | 100-item stratified sample (Chinese) |
| `cpsy_sample_100_en.json` | Done | 100-item sample translated to English |
| `cpsy_client_records_checkpoint.json` | — | Intermediate checkpoint for Step 4 |
| `cpsy_guidance_checkpoint.json` | — | Intermediate checkpoint for Step 4 |
| `cpsy_client_records.json` | Pending | Medical records (TheraMind format) |
| `cpsy_guidance.json` | Pending | 6-session guidance (TheraMind format) |
