# Transaction Anomaly Detection & Diagnostic Assistant

A GenAI-powered POC for card-issuer payment operations teams.
Detects anomalies in transaction metrics, generates plain-language
diagnostic narratives, and supports conversational follow-up Q&A —
all grounded in statistical detection outputs, never hallucinated.

---

## Architecture overview

```
Synthetic data  →  Feature engineering  →  4 detectors  →  Root cause attribution
                                                                      ↓
                                              Streamlit UI  ←  LLM grounding layer
                                                    ↕
                                            Conversational Q&A
```

**Detection layer** — four independent statistical detectors:
- Volume detector (STL decomposition — catches absolute drops)
- Rate detector (conditional Z-score per slice — catches rate anomalies)
- Reason code detector (chi-squared test — catches why-declines-shifted)
- Fraud concentration detector (KL divergence — catches attack concentration)

**LLM layer** — llama-3.3-70b-versatile via Groq API:
- The LLM explains anomalies. It never detects them.
- Every number cited is traceable to the detection layer output.
- Post-generation verification flags hallucination risk.

---

## Injected anomalies (what the system should find)

| ID | Scenario | Dimensions | Key signal |
|----|----------|------------|------------|
| A1 | Issuer processor BIN outage | BIN 4531xx, all channels | RC 96 → 82.7%, approval 10.4% |
| A2 | 3DS ACS cascade failure | DE/NL, ecom, 3DS | RC 65 → 57.1%, approval 57.1% |
| A3 | GB grocery fraud attack | GB, grocery, ecom, CNP | Fraud 5.26×, ticket £12 |
| A4 | Cross-border silent erosion | Cross-border, ecom | RC 91 trending, −4.2pp over 6d |
| A5 | Weekend contactless rule change | GB/FR, contactless, weekend | RC 61 4×, weekend-only pattern |

---

## Quick start — Google Colab

### Step 1 — Clone and install

```python
# Cell 1
!git clone https://github.com/YOUR_USERNAME/anomaly-diagnostic-assistant.git
%cd anomaly-diagnostic-assistant
!pip install -r requirements.txt
```

### Step 2 — Set your Groq API key

Get your free API key at https://console.groq.com

```python
# Cell 2
import os
# Option A: type directly (for quick testing — do not share the notebook)
os.environ["GROQ_API_KEY"] = "gsk_..."

# Option B: use Colab secrets (recommended)
# from google.colab import userdata
# os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
```

### Step 3 — Run the data pipeline

```python
# Cell 3 — generates synthetic data + runs all detectors (~2–3 minutes)
!python run_pipeline.py
```

Expected output:
```
[1/4] Generating synthetic transaction data...
      Rows: 327,471 | Txns: 3,421,968 | Anomalies injected: 5
[2/4] Engineering features...
      Feature store: 327,471 rows | Slices: 192
[3/4] Running anomaly detectors...
      Volume detector:    3 anomalies
      Rate detector:      8 anomalies
      Reason code detector: 5 anomalies
      Fraud detector:     2 anomalies
      Total unique:       12 anomalies flagged
[4/4] Root cause attribution...
      Enriched: 12 anomaly briefs written to data/anomaly_objects.json
Pipeline complete.
```

### Step 4 — Launch Streamlit in Colab

```python
# Cell 4
!pip install streamlit pyngrok -q
from pyngrok import ngrok
import subprocess, threading, time

def run_streamlit():
    subprocess.run(["streamlit", "run", "app.py",
                    "--server.port=8501",
                    "--server.headless=true"])

threading.Thread(target=run_streamlit, daemon=True).start()
time.sleep(5)

# Create public tunnel
public_url = ngrok.connect(8501)
print(f"\n  Open your app: {public_url}\n")
```

---

## Local development

### Prerequisites
- Python 3.10 or 3.11
- A Groq API key
  Get your free key at https://console.groq.com

### Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/anomaly-diagnostic-assistant.git
cd anomaly-diagnostic-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and add your key

# 5. Generate data and run detectors
python run_pipeline.py

# 6. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploying to Streamlit Community Cloud

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — anomaly diagnostic assistant POC"
git remote add origin https://github.com/YOUR_USERNAME/anomaly-diagnostic-assistant.git
git push -u origin main
```

**Important:** verify `.gitignore` is working before pushing:
```bash
git status   # data/*.csv and .streamlit/secrets.toml must NOT appear
```

### Step 2 — Create the Streamlit app

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **New app**
4. Repository: `YOUR_USERNAME/anomaly-diagnostic-assistant`
5. Branch: `main`
6. Main file path: `app.py`
7. Click **Advanced settings**

### Step 3 — Add secrets

In Advanced settings → Secrets, paste:
```toml
GROQ_API_KEY = "gsk_..."
```

Click **Deploy**.

### Step 4 — First-run data generation

On first deploy, the app detects that `data/anomaly_objects.json` does not exist
and automatically runs the pipeline. This takes 2–3 minutes. A spinner is shown.
Subsequent loads use the cached data.

---

## Project structure

```
anomaly-diagnostic-assistant/
├── .streamlit/
│   └── secrets.toml.example     # copy → secrets.toml, never commit
├── data/                        # gitignored — generated at runtime
│   └── .gitkeep
├── pipeline/
│   ├── data_generator.py        # synthetic 90-day hourly data + 5 anomalies
│   ├── feature_engineer.py      # STL, conditional baselines, RC distributions
│   └── root_cause.py            # failure class attribution + brief enrichment
├── detectors/
│   ├── base_detector.py         # abstract base class
│   ├── volume_detector.py       # STL residual anomaly
│   ├── rate_detector.py         # conditional Z-score
│   ├── reason_code_detector.py  # chi-squared distribution shift
│   └── fraud_concentration.py   # KL divergence concentration
├── llm/
│   ├── prompt_templates.py      # all prompts versioned here
│   ├── context_builder.py       # assembles grounded LLM context
│   ├── output_verifier.py       # post-generation hallucination check
│   ├── llm_client.py            # Groq API client + conversation state
│   └── narrative_generator.py  # 8-section structured diagnostic narrative
├── ui/
│   ├── anomaly_feed.py          # left panel — anomaly list
│   ├── diagnostic_panel.py      # top-right — LLM narrative
│   ├── chart_panel.py           # bottom-left — time series chart
│   └── chat_panel.py            # bottom-right — conversational Q&A
├── app.py                       # Streamlit entry point
├── run_pipeline.py              # CLI runner: generate → detect → enrich
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Design principles

**The LLM never detects anomalies.** Detection is entirely statistical.
The LLM's role is translation: structured facts → analyst language.

**Every cited number is traceable.** The `OutputVerifier` checks that all
statistics in the LLM narrative exist in the anomaly object's `supporting_stats`.
The UI shows a hallucination risk badge on every response.

**Explainability over accuracy.** Isolation Forest and Prophet are excluded.
All detectors produce named, typed outputs with σ deviations and evidence lists
that an analyst can reproduce in a spreadsheet.

**Colab-first, Streamlit-ready.** No pyarrow, no background threads,
no async — everything works in both environments without modification.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: groq` | `pip install groq>=0.9.0` |
| `GROQ_API_KEY not set` | Set in `.streamlit/secrets.toml` or `os.environ` |
| `data/anomaly_objects.json not found` | Run `python run_pipeline.py` |
| Streamlit shows blank page | Check browser console; usually a missing import |
| Colab tunnel disconnects | Re-run the ngrok cell |
| `statsmodels` STL error | `pip install statsmodels>=0.14.0` |

---

## License

MIT — see LICENSE file.
