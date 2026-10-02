# DHAGA FITLENS

**AI-powered Return Reason Intelligence** for the Dhaga & Co. FDE Academy case. This is an internal, business-focused Streamlit MVP—not a return chatbot.

> **Demo data is synthetic.** The project is not connected to real Dhaga systems. It does not claim to reduce the company's 31% overall return rate. The business hypothesis is that product and vendor insight can help the Category team investigate fit problems and potentially improve outcomes later.

## Business problem

Dhaga & Co. has a 31% return rate and 44% of reasons are recorded as `Other`. Neha, the Category Head, suspects many free-text `Other` comments concern fit, but cannot manually read more than a few hundred at a time across roughly 14,000 SKUs. FITLENS turns structured dropdown values and unstructured customer text (including Hinglish, vernacular phrasing, informal language and spelling mistakes) into reviewable category and fit signals.

## What the MVP does

1. Validates, cleans, de-duplicates, and previews a return-record CSV with Pandas.
2. Uses a LangChain classifier to label every usable comment `FIT`, `QUALITY`, `PRODUCT`, `DELIVERY`, or `OTHER`.
3. Uses a LangChain `RunnableBranch` to route only `FIT` returns through detailed fit analysis.
4. Uses a second, stronger LangChain model to evaluate every proposed classification.
5. Sends low-confidence, rejected, missing-text, failed, and ambiguous records to a searchable human-review queue.
6. Uses Pandas—not an LLM—to calculate distributions, SKU/vendor fit-return rates, and cautious investigation recommendations.

## Architecture

```text
CSV upload → Pandas validation/cleaning → classifier model (LangChain)
                                          ↓
                              RunnableBranch: FIT?
                                  ├─ yes → FIT detail model chain
                                  └─ no  → final category
                                          ↓
                           evaluator model (LangChain)
                                          ↓
                    Pandas analytics → Streamlit dashboard/review queue
```

### Required AI patterns

- **Routing:** `chains/router.py` uses LangChain `RunnableBranch`; only `FIT` classifications invoke the fit-detail chain.
- **Evaluator–optimizer:** the proposed label, confidence, explanation, and original text go to an independent evaluator. A rejection or uncertainty is made visible and requires human review; it is never silently accepted.

### Two models and cost awareness

`CLASSIFIER_MODEL` is intended as the cheaper/faster bulk model; `EVALUATOR_MODEL` is intended as the stronger independent evaluator. They must have different names in live mode. Both are centrally configured via environment variables; no API key is committed.

The dashboard reports processed records and classifier, fit-chain, evaluator, and total calls. If all four optional per-million-token prices are supplied, it presents a clearly labelled planning estimate; it never invents provider prices. Without pricing configuration it shows: **“Cost estimate unavailable until model pricing is configured.”**

## Setup

Requires Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

For live analysis through OpenRouter, edit `.env`:

```dotenv
LLM_API_KEY=your_openrouter_key_here
LLM_PROVIDER=openrouter
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_APP_NAME=DHAGA FITLENS
CLASSIFIER_MODEL=openai/gpt-4.1-mini
EVALUATOR_MODEL=openai/gpt-4.1
```

The live integration uses `langchain-openai`, invoked exclusively through LangChain `ChatPromptTemplate | ChatOpenAI.with_structured_output(...)`; no raw HTTP or direct API calls are used. OpenRouter provides an OpenAI-compatible endpoint, configured centrally with `OPENROUTER_BASE_URL` and provider-prefixed model IDs. Pydantic validates every LLM boundary. Official OpenAI API keys remain supported with `LLM_PROVIDER=openai` and unprefixed OpenAI model names.

## Generate and run the synthetic demo

```powershell
python data/generate_demo_data.py
streamlit run app.py
```

The generator creates `data/demo_returns.csv` with 650 clearly labelled `SYNTHETIC DEMO DATA` records, multiple SKUs/vendors/sizes, an intentionally high share of `Other` dropdown values, mixed language, weak text, and elevated fit signals for selected SKUs. Upload it or choose **Load synthetic demo data**.

### Offline demonstration mode

Without `LLM_API_KEY`, the app deliberately displays **Deterministic demo (no LLM key)** and uses visible rule-based output so the end-to-end interface can be demonstrated locally. This is not represented as LLM analysis. Set the API key for the required live LangChain two-model workflow.

## Expected CSV columns

Required: `order_id`, `sku`, `product_name`, `vendor`, `size`, `return_reason`, `return_reason_detail`.

Additional columns are retained. Blank values are cleaned; duplicate records based on all required fields are removed; blank text is surfaced and routed to review. Invalid encodings, empty files, missing columns, LLM/API errors, timeouts, invalid structured responses, low confidence, and evaluator disagreement are displayed rather than hidden.

## Example workflow

1. Open **Upload Data**, load the synthetic dataset, inspect validation results, and click **Analyze Returns**.
2. The app classifies comments such as `kurti bahut tight hai`, routes FIT comments, and evaluates the result.
3. Review **Dashboard** for category/fit distributions plus SKU and vendor rates.
4. Use **Return Analysis** to search/filter results.
5. Resolve ambiguous items on **Human Review**; e.g. `size feels weird` should not be silently accepted.

## Testing

```powershell
pytest -q
```

Tests cover CSV validation/missing columns, Pydantic validation, `RunnableBranch` routing, and Pandas analytics. A practical end-to-end test is: generate `data/demo_returns.csv`, run Streamlit, upload/load it, analyze, then inspect the dashboard and human-review queue. With credentials present this exercises the live classifier → router → fit chain → evaluator flow; without credentials it exercises the explicitly labelled deterministic UI walkthrough.

## Limitations and future work

- This is an MVP using synthetic data, not production data or a measured intervention.
- LLM labels should be reviewed, especially low-confidence and vernacular edge cases.
- SKU fit-return rate uses returns in the uploaded file as its denominator; it is not a sales/units-shipped rate.
- Future work: integrate returns/catalog/units-shipped sources, add an approval-feedback loop, batch/concurrency controls, privacy controls, monitoring, role-based access, and calibrated evaluation against human-labelled production samples.

## Deployment

Containerize the app or deploy it to a managed internal Streamlit-compatible runtime. Store `LLM_API_KEY` and optional price configuration in the platform's secret manager (not in source control), restrict access to Category-team users, and connect approved data sources only after privacy/security review. Configure logs and alerting for API and structured-output failures before using it with real data.
