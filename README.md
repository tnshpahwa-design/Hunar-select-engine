# Hunar-select-engine
Hunar Select: AI-Powered Recruitment Intelligence Engine

An automated GTM engineering prototype built for Hunar.AI (Hunar Select) to dynamically prospect high-fit Indian enterprises, detect real-time frontline hiring signals, map recruitment decision-makers, and prioritize outreach using deterministic scoring.

1. System Architecture

The engine runs a 5-stage automated research pipeline triggered dynamically by Target Industry and Company Size (1,000–5,000, 5,000–20,000, 20,000+ employees) with Geography defaulting to India.

[User Inputs: Industry + Company Size]
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 1: Company Discovery                               │
│ • Multi-segment Serper (Google Search) queries           │
│ • Gemini Flash extraction + strict URL verification      │
│ • Deduplication & initial size-band filtering (up to 50) │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 2: Headcount & Footprint Enrichment                │
│ • Targeted headcount & footprint queries (+ second-look) │
│ • Source quality filtering & verified band matching      │
│ • Industry-specific frontline workforce estimation (Cap 25)│
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 3: Hiring & Recruitment Signal Intelligence        │
│ • Google News RSS (en-IN) + Serper hiring queries        │
│ • Classifies S1–S8 signal types & roles/locations/volume │
│ • Enforces Confirmed vs. Observed vs. Inferred guardrails│
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 4: Decision-Maker Identification & Contact Lookup  │
│ • Targeted LinkedIn X-ray search (CHRO, TA, HR Ops, Ops) │
│ • Persona prioritization + public careers contact scrape │
│ • Optional Hunter.io & Apollo.io email/phone enrichment  │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 5: Deterministic Scoring & Prioritization UI       │
│ • Weighted 0–100 Priority Score (P1–P4) & ICP Fit Score  │
│ • Interactive Streamlit Dashboard, Drill-downs & CSVs    │
└──────────────────────────────────────────────────────────┘


2. Data Sources & Tech Stack

Layer

Technology / Source

Role in Pipeline

Frontend & State

Streamlit (app.py)

Interactive UI, stage-by-stage progress logging, checkpointed JSON persistence (saved_runs/), and CSV exports.

Web & LinkedIn Search

Serper API (google.serper.dev)

India-localized (gl: "in") organic search and Knowledge Graph extraction for discovery, headcount verification, public HR contacts, and LinkedIn profiles.

Real-Time News

Google News RSS (news.google.com/rss)

Free, high-recency Indian business and expansion news feed (hl=en-IN&gl=IN) with exact publication timestamps.

LLM Extraction

Google Gemini Flash API

Structured JSON extraction (responseMimeType: application/json) across batched evidence blocks with automatic multi-model fallback and quota rotation.

Contact Enrichment

Hunter.io & Apollo.io APIs (Optional)

Domain-level HR email discovery (Hunter) and LinkedIn-to-work-email/phone matching (Apollo) for top-ranked accounts.

3. Automation Logic & Engineering Guardrails

To ensure reliability and eliminate LLM hallucinations, the pipeline enforces strict programmatic guardrails:

Zero-Hallucination Source Verification:

Every company source URL, headcount figure, hiring signal, and LinkedIn profile must match a real URL or evidence ID (E1, P1) returned during the live search step. Unsourced figures are discarded or flagged as Unverified.

Code-Enforced Evidence Classification (check_level):

Confirmed (10 pts): Explicit hiring mandate with a stated volume, target, or named recruitment drive.

Observed (6 pts): Active job openings or recruitment campaigns without a quantified volume.

Inferred (3 pts): Business expansion (e.g., new dark stores, branches, warehouses, funding) that implies frontline hiring. Regex guardrails automatically downgrade signals lacking explicit hiring keywords to Inferred.

Recency Decay & Urgency Weighting:

Signals are weighted by age: <= 30 days (1.0x, High Urgency), 31–90 days (0.7x), 91–180 days (0.4x), 181–365 days (0.2x), and > 365 days (0.0x).

Resilience & Checkpointing:

Search responses are cached (@st.cache_data) to conserve API credits.

Each completed stage saves a JSON checkpoint to saved_runs/, allowing interrupted runs to resume mid-pipeline (Resume unfinished run) or load instantaneously for demos.

4. Prioritization Scoring Model

All rankings are calculated deterministically in Python (score_company)—the LLM extracts facts, while code computes scores:

Priority Score (0–100):

Hiring Intensity (20%): Sum of recency-weighted signal points.

Open Positions / Hiring Volume (15%): Quantified role counts extracted from reliable sources.

Frontline Workforce Scale (15%): Total headcount multiplied by industry frontline ratio (78% Retail/Logistics, 70% Quick Commerce, 58% Healthcare, 50% BFSI).

Urgency of Hiring (15%): Recency of the latest Confirmed or Observed signal + seasonal peak bonus.

Outsourcing Potential (15%): Industry baseline adjusted for attrition signals (+10), third-party staffing mentions (+10), multi-city footprint (+10), and large in-house TA teams (-10).

Geographic Expansion (10%): Count of active expansion/launch signals (S2, S3) and distinct operating cities.

Decision-Maker Accessibility (10%): Presence of prioritized buyer personas (Head of TA > CHRO > Head of HR Ops > Regional HR > Business/Ops Head) and verified contact info.

Action Tiers: P1 (75+ — Approach immediately), P2 (60–74), P3 (45–59), P4 (<45).

5. Local Setup & Running the App

Prerequisites

Python 3.9+

API Keys for Google Gemini and Serper (plus optional Hunter.io and Apollo.io keys).

Installation

git clone https://github.com/tnshpahwa-design/Hunar-select-engine.git
cd Hunar-select-engine
pip install -r requirements.txt


Configure Secrets

Create .streamlit/secrets.toml in the project root:

GEMINI_API_KEY = "your-gemini-api-key"
SERPER_API_KEY = "your-serper-api-key"

# Optional contact enrichment keys
HUNTER_API_KEY = "your-hunter-api-key"
APOLLO_API_KEY = "your-apollo-api-key"


Launch

streamlit run app.py
