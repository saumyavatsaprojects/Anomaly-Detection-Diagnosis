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


## Setup and Installation

**Prerequisites**
Python 3.9 or higher
A Groq API key (free tier, high rate limits — used instead of OpenAI/Anthropic due to volume of testing required)

**Git**

1. Clone the repository
bash
git clone https://github.com/saumyavatsaprojects/Anomaly-Detection-Diagnosis/.git
cd Anomaly-Detection-Diagnosis

3. Create a virtual environment
bash
python -m venv venv
source venv/bin/activate    	  macOS/Linux
venv\Scripts\activate       	  Windows

5. Install dependencies
bash
pip install -r requirements.txt

7. Set your API key
Create a .env file in the project root:
GROQ_API_KEY=your_groq_api_key_here
Or export it directly in your terminal:
bash
export GROQ_API_KEY=your_groq_api_key_here

**Running the App**

Option A: Full pipeline (generate data, detect, launch UI)
bash
python run_pipeline.py
streamlit run app.py

Option B: Step by step
bash
  1. Generate synthetic data
python data_generator.py
 
  2. Engineer features
python pipeline/feature_engineer.py
 
  3. Run detectors
python pipeline/run_detectors.py
 
  4. Root cause attribution
python pipeline/root_cause.py
 
  5. Launch UI
streamlit run app.py
Streamlit Community Cloud
The app is deployed at:
Add your API key in the Streamlit Cloud app under Settings > Secrets:
toml
GROQ_API_KEY = "your_groq_api_key_here"

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
