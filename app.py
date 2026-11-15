import html as _html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Hunar Select Recruitment Intelligence Engine", layout="wide")

SIZE_BANDS = ["1,000 to 5,000", "5,000 to 20,000", "20,000+"]
INDUSTRIES = ["Quick Commerce", "Retail", "Logistics", "BFSI", "Healthcare"]
INDUSTRY_SEGMENTS = {
    "Quick Commerce": ("quick commerce and instant delivery",
                       ["e-grocery and online supermarkets", "hyperlocal delivery",
                        "fresh food, dairy and meat delivery", "e-commerce fulfilment and last-mile delivery"]),
    "Retail": ("retail chains",
               ["fashion and apparel retail", "grocery and supermarket chains",
                "electronics, home and lifestyle retail", "pharmacy and specialty retail"]),
    "Logistics": ("logistics and supply chain",
                  ["express parcel and courier", "warehousing and third-party logistics",
                   "trucking and freight", "last-mile delivery and fleet operators"]),
    "BFSI": ("banks NBFC and insurance",
             ["microfinance and housing finance", "life and general insurance",
              "private and small finance banks", "broking and wealth management"]),
    "Healthcare": ("hospital chains and healthcare",
                   ["diagnostic labs and pathology", "pharmacy chains",
                    "clinics and dialysis centres", "healthcare services and home care"]),
}

MACRO_SECTOR_BENCHMARKS = [
    {
        "Sector": "Quick Commerce",
        "Est. Frontline Share": "70%",
        "Core Frontline Roles": "Dark store pickers/packers, delivery riders, store managers, hub supervisors",
        "Primary Hiring Triggers": "Dark store city launches, 10-min delivery expansion, festive/sale surges",
        "Peak Hiring Cycles": "Oct–Nov (Festive), Mar–May (Summer), Major Sale Events",
        "Outsourcing Propensity": "High (70/100 base)",
    },
    {
        "Sector": "Retail",
        "Est. Frontline Share": "78%",
        "Core Frontline Roles": "Store associates, cashiers, sales executives, visual merchandisers, store managers",
        "Primary Hiring Triggers": "New mall/high-street store rollouts, Tier-2/3 expansion, seasonal replacement",
        "Peak Hiring Cycles": "Sep–Jan (Festive & Wedding), Jun–Jul (EOSS)",
        "Outsourcing Propensity": "High (70/100 base)",
    },
    {
        "Sector": "Logistics",
        "Est. Frontline Share": "78%",
        "Core Frontline Roles": "Last-mile delivery executives, line-haul drivers, warehouse sorters, loaders",
        "Primary Hiring Triggers": "Fulfilment center commissioning, e-commerce volume spikes, fleet scaling",
        "Peak Hiring Cycles": "Aug–Nov (E-commerce Pre-Festive Ramp), Q4",
        "Outsourcing Propensity": "Very High (75/100 base)",
    },
    {
        "Sector": "BFSI",
        "Est. Frontline Share": "50%",
        "Core Frontline Roles": "Field sales executives, relationship officers, collections agents, telecallers",
        "Primary Hiring Triggers": "Branch network expansion, microfinance/lending push, credit card/CASA drives",
        "Peak Hiring Cycles": "Jan–Mar (Financial Year-End), Oct–Dec",
        "Outsourcing Propensity": "Moderate-High (60/100 base)",
    },
    {
        "Sector": "Healthcare",
        "Est. Frontline Share": "58%",
        "Core Frontline Roles": "Nurses, phlebotomists, lab technicians, ward staff, front-desk coordinators",
        "Primary Hiring Triggers": "New hospital wings, diagnostic collection center rollouts, home-care scaling",
        "Peak Hiring Cycles": "Year-round steady + Monsoon/Winter diagnostic surges",
        "Outsourcing Propensity": "Moderate (55/100 base)",
    },
]


# ---------------- Shared helpers ----------------
def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return None


SKIP_WORDS = ["preview", "exp", "image", "tts", "live", "audio", "thinking", "robotics", "computer", "omni"]


def list_gemini_models():
    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not found in Secrets.")
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                     headers={"x-goog-api-key": key}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Could not list models: {r.status_code} {r.text[:200]}")
    return [m["name"] for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])]


def order_models(names, prefer_lite=False, pinned=None):
    """Decide which models to try, in order."""
    def usable(n):
        return "flash" in n and not any(w in n for w in SKIP_WORDS) and "latest" not in n

    full = sorted([n for n in names if usable(n) and "lite" not in n], reverse=True)
    lite = sorted([n for n in names if usable(n) and "lite" in n], reverse=True)
    order = (lite + full) if prefer_lite else (full + lite)
    aliases = ["models/gemini-flash-lite-latest", "models/gemini-flash-latest"] if prefer_lite \
        else ["models/gemini-flash-latest", "models/gemini-flash-lite-latest"]
    for alias in aliases:
        if alias in names and alias not in order:
            order.append(alias)
    if pinned:
        pinned = pinned if pinned.startswith("models/") else "models/" + pinned
        if pinned in order:
            order.remove(pinned)
        order.insert(0, pinned)
    return order


def gemini_generate(prompt, json_mode=False, timeout=40, max_models=6, prefer_lite=False):
    """Ask Gemini. Tries several Flash models, skips quickly on quota errors, retries when busy."""
    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not found in Secrets.")
    headers = {"x-goog-api-key": key}
    names = list_gemini_models()
    order = order_models(names, prefer_lite, get_secret("GEMINI_MODEL"))
    if not order:
        raise RuntimeError("No Gemini Flash model available for this key.")

    memory_key = "gemini_model_lite" if prefer_lite else "gemini_model_full"
    last_good = st.session_state.get(memory_key)
    if last_good in order:
        order.remove(last_good)
        order.insert(0, last_good)

    gen_config = {"temperature": 0.25, "maxOutputTokens": 12000}
    if json_mode:
        gen_config["responseMimeType"] = "application/json"
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen_config}

    attempts = []
    for model in order[:max_models]:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"
        for attempt in range(2):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            except requests.exceptions.RequestException:
                attempts.append(f"{model}: timed out")
                break
            if resp.status_code == 200:
                try:
                    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    attempts.append(f"{model}: empty reply")
                    break
                st.session_state[memory_key] = model
                return text.strip(), model
            attempts.append(f"{model}: {resp.status_code}")
            if resp.status_code in (401, 403):
                raise RuntimeError(f"Key rejected. {resp.text[:200]}")
            if resp.status_code in (500, 503) and attempt == 0:
                time.sleep(3)
                continue
            break
    raise RuntimeError("All Gemini models were busy or failed. Tried: " + "; ".join(attempts))


@st.cache_data(ttl=86400, show_spinner=False)
def serper_raw(query, num=10):
    """Google search through Serper (India results). Cached for 24 hours."""
    key = get_secret("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY not found in Secrets.")
    r = requests.post("https://google.serper.dev/search",
                      headers={"X-API-KEY": key, "Content-Type": "application/json"},
                      json={"q": query, "gl": "in", "num": num}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Serper failed: {r.status_code} {r.text[:200]}")
    return r.json()


def serper_search(query, num=10):
    return serper_raw(query, num).get("organic", [])


# ---------------- Stage 1: company discovery ----------------
def build_queries(industry, size):
    primary, alts = INDUSTRY_SEGMENTS[industry]
    queries = [
        f"top {primary} companies in India",
        f"largest {primary} companies in India by number of employees",
        f"{primary} companies India {size} employees",
    ]
    for alt in alts[:4]:
        queries.append(f"major {alt} companies in India number of employees")
    return queries


def parse_json_list(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        data = json.loads(t)
    except Exception:
        s, e = t.find("["), t.rfind("]")
        if s == -1 or e <= s:
            raise RuntimeError("The AI reply was not valid JSON. Try Launch Research again.")
        data = json.loads(t[s:e + 1])
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        data = lists[0] if lists else []
    return data


def parse_json_obj(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        data = json.loads(t)
    except Exception:
        s, e = t.find("{"), t.rfind("}")
        if s == -1 or e <= s:
            raise RuntimeError("The AI reply was not valid JSON object.")
        data = json.loads(t[s:e + 1])
    return data if isinstance(data, dict) else {}


def with_retry(fn, *args, tries=2, wait=15):
    last = None
    for i in range(tries):
        try:
            return fn(*args)
        except Exception as ex:
            last = ex
            if i < tries - 1:
                time.sleep(wait)
    raise last


def normalise_url(u):
    u = (u or "").strip()
    if not u:
        return ""
    if not u.startswith("http"):
        u = "https://" + u
    return u


def process_companies(data, valid_urls, size, cap=50):
    seen, rows = set(), []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("company", "")).strip()
        if not name:
            continue
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in seen:
            continue
        seen.add(key)
        src = str(item.get("source_url", "")).strip()
        if src not in valid_urls:
            src = ""
        band = str(item.get("size_band", "Unknown")).strip()
        if band not in SIZE_BANDS:
            band = "Unknown"
        if band == "Unknown":
            status = "Size unverified"
        elif band != size:
            status = "Size to be checked"
        elif not src:
            status = "Needs verification"
        else:
            status = "Sourced"
        rows.append({
            "Company": name,
            "Business profile": str(item.get("business_profile", "")).strip(),
            "Business model": str(item.get("business_model", "")).strip(),
            "HQ": str(item.get("headquarters", "")).strip(),
            "Website": normalise_url(item.get("website", "")),
            "Size band (est.)": band,
            "Size evidence": str(item.get("size_evidence", "")).strip(),
            "Status": status,
            "Source": src,
        })
    matched = [r for r in rows if r["Size band (est.)"] == size]
    unknown = [r for r in rows if r["Size band (est.)"] == "Unknown"]
    pos = {b: i for i, b in enumerate(SIZE_BANDS)}
    adjacent = [r for r in rows if r["Size band (est.)"] in pos and r["Size band (est.)"] != size
                and abs(pos[r["Size band (est.)"]] - pos[size]) == 1]
    excluded = [r for r in rows if r["Size band (est.)"] in pos and r["Size band (est.)"] != size
                and abs(pos[r["Size band (est.)"]] - pos[size]) > 1]
    matched.sort(key=lambda r: 0 if r["Source"] else 1)
    kept = (matched + unknown + adjacent)[:cap]
    return kept, excluded


def discovery_prompt(industry, size, evidence, already=None):
    primary, alts = INDUSTRY_SEGMENTS[industry]
    segments = "; ".join([primary] + alts)
    count = "up to 45"
    extra = ""
    if already:
        count = "up to 30 additional"
        extra = ("\nALREADY LISTED (do not repeat any of these): " + ", ".join(already[:80]) +
                 "\nList additional companies only. Cover every segment in scope, and include mid-sized and "
                 "regional chains, not only the best-known names.\n")
    return f"""You are a research assistant building a prospect list for an Indian frontline hiring and recruitment-outsourcing company.

TASK: List companies that operate in India in this industry: {industry}. Segments in scope: {segments}. Prefer companies with roughly {size} employees (global or India headcount), but also include companies of nearby sizes when you are unsure, because headcount will be checked in a later step.
{extra}
EVIDENCE: numbered web search results below. Prefer companies named in them. You may add well-known companies from your own knowledge only if you are confident they belong in this industry and size band; for those, set source_url to an empty string.

RULES:
- Return {count} distinct companies. No duplicates.
- Use the name people use in the market. If it is a brand of a larger parent, mention the parent in business_profile.
- size_band must be exactly one of: "1,000 to 5,000", "5,000 to 20,000", "20,000+", "Unknown".
- If a headcount appears in the results, put that figure in size_evidence and use it to choose size_band. Otherwise give your best estimate and set size_evidence to "Estimate, to be verified". Use "Unknown" if you cannot tell.
- source_url must be copied exactly from one result link above that mentions the company, or be an empty string.
- website: only if shown in the results or you are certain; otherwise an empty string.
- business_profile: maximum 25 words. business_model: maximum 15 words. headquarters: city.
- Exclude companies with no India presence and government bodies. Do not drop a company only because its size is uncertain: give your best estimate band, or Unknown.

OUTPUT: a JSON array only. Each item has exactly these keys: company, business_profile, business_model, headquarters, website, size_band, size_evidence, source_url.

SEARCH RESULTS:
{evidence}
"""


def discover_companies(industry, size, results):
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title'][:120]} | {r['snippet'][:200]} | {r['link']}")
    evidence = "\n".join(lines)[:18000]
    valid_urls = {r["link"] for r in results}
    text, model = gemini_generate(discovery_prompt(industry, size, evidence), json_mode=True,
                                  timeout=100, max_models=4, prefer_lite=True)
    data = parse_json_list(text)
    kept, excluded = process_companies(data, valid_urls, size)
    if len(kept) < 40:
        try:
            already = [c["Company"] for c in kept] + [e["Company"] for e in excluded]
            text2, _ = gemini_generate(discovery_prompt(industry, size, evidence, already), json_mode=True,
                                       timeout=100, max_models=4, prefer_lite=True)
            data2 = parse_json_list(text2)
            kept, excluded = process_companies(data + data2, valid_urls, size)
        except Exception:
            pass
    return kept, excluded, model


def run_discovery(industry, size, log):
    queries = build_queries(industry, size)
    log("Step 1 of 2: searching the web...")
    results, seen = [], set()
    for q in queries:
        for item in serper_search(q):
            link = item.get("link", "")
            if link and link not in seen:
                seen.add(link)
                results.append({"query": q, "title": item.get("title", ""),
                                "snippet": item.get("snippet", ""), "link": link})
    log(f"Collected {len(results)} search results from {len(queries)} searches.")
    if not results:
        raise RuntimeError("The searches returned nothing. Check the Serper key and try again.")
    log("Step 2 of 2: the AI is reading the results and listing companies...")
    kept, excluded, model = discover_companies(industry, size, results)
    return {"run_id": time.time(), "industry": industry, "size": size, "companies": kept,
            "excluded": excluded, "results": results, "queries": queries, "model": model}


# ---------------- Stage 2: company details ----------------
FRONTLINE_SHARE = {"Quick Commerce": 0.70, "Retail": 0.78, "Logistics": 0.78, "BFSI": 0.50, "Healthcare": 0.58}
ROLE_HINTS = {
    "Quick Commerce": "delivery partners, dark store pickers and packers, dark store managers, rider supervisors",
    "Retail": "store associates, cashiers, sales executives, visual merchandisers, store managers",
    "Logistics": "drivers, delivery executives, warehouse associates, sorters, hub staff, fleet supervisors",
    "BFSI": "sales executives, relationship officers, collections agents, branch staff, telecallers",
    "Healthcare": "nurses, ward staff, paramedics, front desk staff, lab technicians, hospital support staff",
}

LOW_QUALITY_SOURCES = ("linkedin.com", "ambitionbox", "glassdoor", "indeed.com", "naukri.com", "builtin",
                       "zoominfo", "rocketreach", "growjo", "owler", "craft.co", "facebook.com", "revelio", "leadiq",
                       "apollo.io", "signalhire", "datanyze", "comparably")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def band_for(n):
    if n < 1000:
        return "Below 1,000"
    if n <= 5000:
        return "1,000 to 5,000"
    if n <= 20000:
        return "5,000 to 20,000"
    return "20,000+"


def build_evidence(raw):
    parts, urls = [], []
    kg = raw.get("knowledgeGraph") or {}
    if kg:
        attrs = kg.get("attributes") or {}
        attr_text = "; ".join(f"{k}: {v}" for k, v in attrs.items())
        site = kg.get("website", "")
        parts.append(f"Knowledge panel: {kg.get('title', '')} - {kg.get('description', '')} "
                     f"{attr_text} (source: {site})")
        if site:
            urls.append(site)
    for item in (raw.get("organic") or [])[:6]:
        link = item.get("link", "")
        if link:
            urls.append(link)
        parts.append(f"- {item.get('title', '')[:120]} | {item.get('snippet', '')[:260]} | {link}")
    return "\n".join(parts)[:2200], urls


def enrich_batch(industry, batch):
    blocks = [f"### COMPANY {i}: {b['name']}\n{b['evidence']}" for i, b in enumerate(batch, 1)]
    prompt = f"""You are extracting facts about companies for a sales prospect database. Industry: {industry}.

For each company below, use ONLY the evidence text given for that company when stating employees, locations and footprint. Do not use outside knowledge for those three. If the evidence does not say, use null, an empty list or an empty string.

Return a JSON array with exactly one object per company, in the same order, with these keys:
- company: the exact company name given
- employees: integer total headcount if a figure appears in the evidence, else null. If several figures appear, use the most recent and most reliable one.
- employees_note: short note such as "FY24 global headcount", "India only", "gig partners counted separately". Empty if employees is null.
- employees_source_url: a link copied exactly from that company's evidence that supports the figure, else ""
- employees_confidence: "High" if the figure is recent and from the company, an annual report or reputable news; "Medium" if from an aggregator or dated; "Low" if vague or approximate; "" if employees is null
- locations: array of up to 12 cities, states or regions where the evidence shows stores, outlets or operations. Do not list only the head-office city unless that is the only evidence. Leave empty if the evidence names no places.
- footprint: short text such as "300+ dark stores in 25 cities" only if stated in the evidence, else ""
- frontline_roles: array of 3 to 6 frontline job categories typical for this business. You may use general industry knowledge here. Typical roles for this industry: {ROLE_HINTS[industry]}
- recruitment_needs: one sentence on the likely recruitment requirement, based on the business model and evidence

COMPANIES AND EVIDENCE:
{chr(10).join(blocks)}
"""
    text, model = gemini_generate(prompt, json_mode=True, timeout=100, max_models=4, prefer_lite=True)
    return parse_json_list(text), model


def as_list(v, limit):
    if isinstance(v, str):
        v = [v] if v.strip() else []
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if str(x).strip()][:limit]


def clean_enrichment(item, urls):
    n = item.get("employees")
    try:
        n = int(float(str(n).replace(",", ""))) if n not in (None, "", False) else None
    except Exception:
        n = None
    if n is not None and n <= 0:
        n = None
    src = str(item.get("employees_source_url", "")).strip()
    if n is not None and src not in urls:
        n, src = None, ""
    conf = str(item.get("employees_confidence", "")).strip().title() if n else ""
    if conf not in ("High", "Medium", "Low"):
        conf = "Low" if n else ""
    if n and any(d in src.lower() for d in LOW_QUALITY_SOURCES):
        conf = "Low"
    return {
        "employees": n,
        "employees_note": str(item.get("employees_note", "")).strip() if n else "",
        "employees_source_url": src if n else "",
        "employees_confidence": conf,
        "locations": as_list(item.get("locations"), 12),
        "footprint": str(item.get("footprint", "")).strip(),
        "frontline_roles": as_list(item.get("frontline_roles"), 6),
        "recruitment_needs": str(item.get("recruitment_needs", "")).strip(),
    }


def assemble_final(candidates, enrichments, industry, size, cap=25):
    share = FRONTLINE_SHARE[industry]
    verified, unverified, excluded = [], [], []
    for c in candidates:
        e = enrichments.get(c["Company"]) or {}
        n = e.get("employees")
        row = {
            "Company": c["Company"],
            "Business profile": c["Business profile"],
            "Business model": c["Business model"],
            "HQ": c["HQ"],
            "Employees": n,
            "Confidence": e.get("employees_confidence", ""),
            "Headcount note": e.get("employees_note", ""),
            "Employee source": e.get("employees_source_url", ""),
            "Est. frontline (estimate)": int(round(n * share, -2)) if n else None,
            "Frontline share assumed": f"{int(share * 100)}%" if n else "",
            "Operating locations": ", ".join(e.get("locations", [])),
            "Footprint (as stated)": e.get("footprint", ""),
            "Frontline roles (inferred)": ", ".join(e.get("frontline_roles", [])),
            "Potential recruitment needs (inferred)": e.get("recruitment_needs", ""),
            "Size status": "",
            "Website": c["Website"],
            "Stage 3 source": c["Source"],
        }
        if n:
            band = band_for(n)
            if band == size:
                row["Size status"] = "Verified"
                verified.append(row)
            else:
                row["Size status"] = f"Headcount {n:,} is {band}"
                excluded.append(row)
        else:
            row["Size status"] = "Size unverified"
            unverified.append(row)
    rank = {"High": 0, "Medium": 1, "Low": 2}
    verified.sort(key=lambda r: rank.get(r["Confidence"], 3))
    return (verified + unverified)[:cap], excluded


def enrich_items(industry, items, log, warnings):
    out, model = {}, ""
    for start in range(0, len(items), 5):
        batch = items[start:start + 5]
        log(f"AI is reading the evidence: companies {start + 1} to {start + len(batch)}...")
        try:
            data, model = with_retry(enrich_batch, industry, batch)
            by_name = {norm(d.get("company", "")): d for d in data if isinstance(d, dict)}
            for j, b in enumerate(batch):
                d = by_name.get(norm(b["name"]))
                if d is None and len(data) == len(batch) and isinstance(data[j], dict):
                    d = data[j]
                if d:
                    out[b["name"]] = clean_enrichment(d, b["urls"])
        except Exception as ex:
            warnings.append(f"AI step failed for companies {start + 1} to {start + len(batch)}: {ex}")
        time.sleep(1)
    return out, model


def run_enrichment(disc, log):
    industry, size = disc["industry"], disc["size"]
    cands = disc["companies"]
    items, evidence_log, warnings = [], {}, []
    for i, c in enumerate(cands, 1):
        name = c["Company"]
        log(f"Searching facts: {name} ({i} of {len(cands)})")
        q = f"{name} number of employees headcount stores warehouses cities India"
        try:
            raw = serper_raw(q, 8)
        except Exception as ex:
            raw = {}
            warnings.append(f"Search failed for {name}: {ex}")
        evidence, urls = build_evidence(raw)
        items.append({"name": name, "evidence": evidence or "No search results.", "urls": set(urls)})
        evidence_log[name] = {"query": q, "links": urls}

    enrichments, model = enrich_items(industry, items, log, warnings)

    missing = [it for it in items if not (enrichments.get(it["name"]) or {}).get("employees")][:12]
    if missing:
        items2 = []
        for i, it in enumerate(missing, 1):
            log(f"Second look for headcount: {it['name']} ({i} of {len(missing)})")
            q2 = f"{it['name']} company size number of employees workforce India"
            try:
                raw = serper_raw(q2, 8)
            except Exception:
                raw = {}
            evidence, urls = build_evidence(raw)
            items2.append({"name": it["name"], "evidence": evidence or "No search results.", "urls": set(urls)})
            evidence_log[it["name"]]["links"] = evidence_log[it["name"]]["links"] + urls
            evidence_log[it["name"]]["query"] += f" | {q2}"
        second, m2 = enrich_items(industry, items2, log, warnings)
        model = m2 or model
        for name, e2 in second.items():
            if e2.get("employees"):
                merged = dict(enrichments.get(name) or e2)
                for k in ("employees", "employees_note", "employees_source_url", "employees_confidence"):
                    merged[k] = e2[k]
                for k in ("locations", "footprint", "frontline_roles", "recruitment_needs"):
                    if not merged.get(k):
                        merged[k] = e2[k]
                enrichments[name] = merged

    final, excluded = assemble_final(cands, enrichments, industry, size)
    return {"run_id": disc["run_id"], "industry": industry, "size": size, "rows": final,
            "excluded": excluded, "evidence": evidence_log, "warnings": warnings, "model": model}


# ---------------- Stage 3: hiring signals ----------------
SIGNAL_TYPES = {
    "S1": "Large-scale job openings",
    "S2": "New facility launch",
    "S3": "Geographic or business expansion",
    "S4": "Hiring drive or campaign",
    "S5": "Leadership or organisational expansion",
    "S6": "Seasonal workforce need",
    "S7": "High attrition or replacement hiring",
    "S8": "Industry development",
}
EVIDENCE_POINTS = {"Confirmed": 10, "Observed": 6, "Inferred": 3}
HIRING_WORDS = re.compile(
    r"\b(hiring|hires?|hired|recruit\w*|jobs?|vacanc\w*|walk-?in|openings?|careers?|job creation|employment)\b", re.I)
LOW_QUALITY_SIGNAL_DOMAINS = ("linkedin.com", "facebook.com", "naukri.com", "indeed.com", "glassdoor", "ambitionbox",
                              "instahyre", "workindia", "hirist", "jobskul", "quikr", "shine.com", "timesjobs",
                              "reddit.com", "youtube.com", "instagram.com")


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_news(query):
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        raise RuntimeError(f"Google News returned {r.status_code}")
    root = ET.fromstring(r.content)
    items = []
    for it in root.iter("item"):
        pub = (it.findtext("pubDate") or "").strip()
        try:
            d = parsedate_to_datetime(pub).date().isoformat()
        except Exception:
            d = ""
        items.append({"title": (it.findtext("title") or "").strip(),
                      "link": (it.findtext("link") or "").strip(),
                      "date": d, "source": (it.findtext("source") or "").strip()})
    return items


def get_news(name):
    queries = [
        f'"{name}" India (expansion OR "new store" OR "new stores" OR warehouse OR "dark store" OR branches OR launch)',
        f'"{name}" India (hiring OR recruitment OR jobs OR "walk-in" OR vacancies OR "job creation")',
    ]
    merged, seen, errors = [], set(), []
    for q in queries:
        try:
            items = fetch_news(q)
        except Exception as ex:
            errors.append(str(ex))
            continue
        for it in items:
            if it["link"] and it["link"] not in seen:
                seen.add(it["link"])
                merged.append(it)
    return merged, errors


def parse_date(s):
    try:
        return datetime.strptime(str(s).strip()[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def recency_info(date_str, level, today):
    d = parse_date(date_str)
    if d is None:
        return None, 0.2, "Unknown"
    age = (today - d).days
    if age <= 30:
        return age, 1.0, "High"
    if age <= 90:
        return age, 0.7, "High" if level in ("Confirmed", "Observed") else "Medium"
    if age <= 180:
        return age, 0.4, "Medium"
    if age <= 365:
        return age, 0.2, "Low"
    return age, 0.0, "None"


def compute_freshness(latest_date_str, has_signals=False, has_verified_headcount=False):
    d = parse_date(latest_date_str)
    if d is not None:
        age = (datetime.now(timezone.utc).date() - d).days
        if age <= 45:
            return "Fresh"
        if age <= 180:
            return "Needs verification"
        return "Stale"
    if has_signals or has_verified_headcount:
        return "Needs verification"
    return "Stale"


def build_signal_evidence(news, web, today):
    ev_map, lines, display = {}, [], []
    cutoff = today - timedelta(days=365)
    recent = []
    for n in news:
        d = parse_date(n["date"])
        if d is None or d >= cutoff:
            recent.append(n)
    recent.sort(key=lambda n: n["date"], reverse=True)
    k = 0
    for n in recent[:12]:
        k += 1
        eid = f"E{k}"
        ev_map[eid] = {"kind": "NEWS", "date": n["date"], "title": n["title"], "snippet": "",
                       "source": n["source"], "link": n["link"]}
        line = f"{eid} | NEWS | {n['date'] or 'no date'} | {n['title'][:150]} | source: {n['source']}"
        lines.append(line)
        display.append(f"{line} | {n['link']}")
    for w in (web or [])[:6]:
        link = w.get("link", "")
        if not link:
            continue
        k += 1
        eid = f"E{k}"
        ev_map[eid] = {"kind": "WEB", "date": "", "title": w.get("title", ""), "snippet": w.get("snippet", ""),
                       "source": "", "link": link}
        line = f"{eid} | WEB | no date | {w.get('title', '')[:120]} | {w.get('snippet', '')[:240]}"
        lines.append(line)
        display.append(f"{line} | {link}")
    return "\n".join(lines)[:5200], ev_map, display


def extract_signals_batch(industry, batch):
    blocks = [f"### COMPANY {i}: {b['name']}\n{b['evidence'] or 'No results found.'}" for i, b in enumerate(batch, 1)]
    prompt = f"""You are a sales-intelligence analyst for Hunar Select, an Indian recruitment business that supplies frontline and volume hiring (sourcing, screening and onboarding of large numbers of workers across cities) to enterprises. Industry: {industry}.

For each company below, find hiring signals using ONLY the numbered evidence items given for that company (NEWS items have a date and headline; WEB items are search results). Report every item that shows store, branch, warehouse or facility openings, expansion plans, hiring, job creation, recruitment drives, leadership expansion or seasonal demand. Be generous: if an item is relevant, report it.

Ignore items that are about a different organisation with a similar name (for example a recreation centre, a mall, a foreign company or a city), items about share prices or results with no expansion detail, and generic job-board pages that are not about this company.

Signal types (use the code):
S1 Large-scale job openings across locations
S2 New store, warehouse, dark store, branch, hospital or facility launch
S3 Geographic or business expansion (new cities, states, verticals, formats)
S4 Hiring drive or recruitment campaign
S5 Leadership or organisational expansion
S6 Seasonal workforce requirement (festive season, sale events, peaks)
S7 High attrition or replacement hiring
S8 Industry development creating workforce demand

Evidence levels (choose exactly one):
- Confirmed: the item explicitly mentions hiring, jobs or recruitment AND gives a number, a target or a named drive (for example "creates 7,000 jobs" or "recruitment drive on 23 May").
- Observed: the item shows actual open roles or a recruitment campaign, without a stated number or mandate.
- Inferred: the item shows a business change that normally needs frontline staff (new stores, expansion targets, funding to scale, peak season) but does not mention hiring or jobs. Opening or planning stores is ALWAYS Inferred.

Return a JSON array. Each item is one signal with exactly these keys:
- company: exact company name given
- evidence_id: the ID of the evidence item that supports it, for example "E3"
- signal_type: one of S1 to S8
- evidence_level: Confirmed, Observed or Inferred
- summary: one sentence describing the development
- roles: array of frontline roles involved (empty array if unknown)
- locations: array of cities or states involved (empty array if unknown)
- volume: a number or range ONLY if stated in the evidence item or directly derivable from it; if derived, say how (for example "about 3,600 (300 stores x about 12 staff)"); otherwise "Unknown". Never guess. Totals shown on job-board pages (for example "7,000+ open roles") are search totals and must NOT be used as volume.
- why_opportunity: one or two sentences on why this is a recruitment opportunity
- how_hunar_can_help: one or two sentences tying Hunar Select's service to these roles and locations

Return at most 6 signals per company, strongest first. Do not repeat the same development twice.

COMPANIES AND EVIDENCE:
{chr(10).join(blocks)}
"""
    text, model = gemini_generate(prompt, json_mode=True, timeout=100, max_models=4, prefer_lite=True)
    return parse_json_list(text), model


def check_level(level, ev):
    text = f"{ev['title']} {ev['snippet']}"
    has_hiring = bool(HIRING_WORDS.search(text))
    if level in ("Confirmed", "Observed") and not has_hiring:
        return "Inferred"
    if level == "Confirmed" and ev["kind"] == "WEB" and not re.search(r"\d", text):
        return "Observed"
    return level


def clean_signal(item, ev_map, today, diag):
    code = str(item.get("signal_type", "")).strip().upper()
    if code not in SIGNAL_TYPES:
        diag["rejected_bad_type"] += 1
        return None
    ev = ev_map.get(str(item.get("evidence_id", "")).strip().upper())
    if ev is None:
        diag["rejected_bad_evidence_id"] += 1
        return None
    level = str(item.get("evidence_level", "")).strip().title()
    if level not in EVIDENCE_POINTS:
        level = "Inferred"
    checked = check_level(level, ev)
    if checked != level:
        diag["levels_downgraded"] += 1
    level = checked
    src = ev["link"]
    host = urlparse(src).netloc.lower()
    quality = "Low" if any(d in host for d in LOW_QUALITY_SIGNAL_DOMAINS) else "Normal"
    date_str = ev["date"]
    age, weight, urgency = recency_info(date_str, level, today)
    freshness = "Fresh" if (age is not None and age <= 45) else "Stale" if (age is not None and age > 180) else "Needs verification"
    volume = str(item.get("volume", "")).strip() or "Unknown"
    if quality == "Low":
        volume = "Unknown"
    points = EVIDENCE_POINTS[level] * weight * (0.5 if quality == "Low" else 1.0)
    return {
        "Signal type": f"{code} {SIGNAL_TYPES[code]}",
        "Evidence level": level,
        "Date": date_str or "Date unverified",
        "Freshness": freshness,
        "Age (days)": age,
        "Urgency": urgency,
        "Summary": str(item.get("summary", "")).strip(),
        "Roles": ", ".join(as_list(item.get("roles"), 8)),
        "Locations": ", ".join(as_list(item.get("locations"), 10)),
        "Est. volume": volume,
        "Why it is an opportunity": str(item.get("why_opportunity", "")).strip(),
        "How Hunar Select can help": str(item.get("how_hunar_can_help", "")).strip(),
        "Source": src,
        "Source name": ev["source"] or host.replace("www.", ""),
        "Source quality": quality,
        "Points": round(points, 1),
    }


def summarise_signals(companies, signals):
    rows = []
    for name in companies:
        mine = [s for s in signals if s["Company"] == name]
        counts = {lvl: sum(1 for s in mine if s["Evidence level"] == lvl) for lvl in EVIDENCE_POINTS}
        dated = [s["Date"] for s in mine if parse_date(s["Date"])]
        top = max(mine, key=lambda s: s["Points"]) if mine else None
        rows.append({
            "Company": name,
            "Confirmed": counts["Confirmed"], "Observed": counts["Observed"], "Inferred": counts["Inferred"],
            "Signal points": round(sum(s["Points"] for s in mine), 1),
            "Latest signal": max(dated) if dated else "",
            "Top signal": top["Summary"] if top else "No signals found",
        })
    rows.sort(key=lambda r: r["Signal points"], reverse=True)
    return rows


def run_signals(enr, log):
    industry = enr["industry"]
    companies = [r["Company"] for r in enr["rows"]]
    today = datetime.now(timezone.utc).date()
    items, evidence_log, warnings = [], {}, []
    for i, name in enumerate(companies, 1):
        log(f"Collecting news and hiring pages: {name} ({i} of {len(companies)})")
        news, web = [], []
        news, errs = get_news(name)
        for e in errs:
            warnings.append(f"News search problem for {name}: {e}")
        try:
            web = serper_raw(f'"{name}" hiring recruitment drive openings India', 8).get("organic", [])
        except Exception as ex:
            warnings.append(f"Web search failed for {name}: {ex}")
        text, ev_map, display = build_signal_evidence(news, web, today)
        items.append({"name": name, "evidence": text, "ev_map": ev_map})
        evidence_log[name] = {"lines": display}

    diag = {"ai_returned": 0, "rejected_bad_type": 0, "rejected_bad_evidence_id": 0,
            "rejected_unknown_company": 0, "levels_downgraded": 0, "duplicates_removed": 0}
    signals, model = [], ""
    for start in range(0, len(items), 2):
        batch = items[start:start + 2]
        log(f"AI is reading the evidence: companies {start + 1} to {start + len(batch)}...")
        try:
            data, model = with_retry(extract_signals_batch, industry, batch)
            maps = {norm(b["name"]): b["ev_map"] for b in batch}
            names = {norm(b["name"]): b["name"] for b in batch}
            for item in data:
                if not isinstance(item, dict):
                    continue
                diag["ai_returned"] += 1
                key = norm(item.get("company", ""))
                if key not in maps:
                    diag["rejected_unknown_company"] += 1
                    continue
                sig = clean_signal(item, maps[key], today, diag)
                if sig:
                    sig["Company"] = names[key]
                    signals.append(sig)
        except Exception as ex:
            warnings.append(f"AI step failed for companies {start + 1} to {start + len(batch)}: {ex}")
        time.sleep(1)

    seen, unique = set(), []
    for s in signals:
        k = (s["Company"], s["Source"], norm(s["Summary"])[:60])
        if k not in seen:
            seen.add(k)
            unique.append(s)
        else:
            diag["duplicates_removed"] += 1
    unique.sort(key=lambda s: s["Points"], reverse=True)
    return {"run_id": enr["run_id"], "signals": unique, "summary": summarise_signals(companies, unique),
            "evidence": evidence_log, "warnings": warnings, "model": model, "diagnostics": diag}


# ---------------- Stage 4: decision-makers and contacts ----------------
LINKEDIN_PROFILE = re.compile(r"linkedin\.com/(in|pub)/", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)")
IMG_TLDS = {"png", "jpg", "jpeg", "gif", "svg", "webp", "css", "js"}
FREE_MAIL = ("gmail.", "yahoo.", "outlook.", "hotmail.", "rediffmail.", "live.", "icloud.", "proton")
GENERIC_NAME_WORDS = {"the", "and", "ltd", "limited", "india", "pvt", "private", "company", "retail", "stores",
                      "store", "group", "international", "services", "enterprises"}
SENIOR_RE = re.compile(r"\b(chief|chro|cpo|head|vp|svp|avp|evp|vice president|director|gm|general manager|"
                       r"president|coo|lead|manager)\b", re.I)
TOP_RE = re.compile(r"\b(chief|head|vp|svp|avp|evp|vice president|director|gm|general manager|president|coo)\b", re.I)
JUNIOR_RE = re.compile(r"\b(intern|trainee|associate|executive|recruiter|coordinator|analyst|assistant|specialist)\b", re.I)


def classify_persona(designation):
    d = (designation or "").lower().strip()
    if not d:
        return None
    if JUNIOR_RE.search(d) and not TOP_RE.search(d):
        return None
    if re.search(r"\bchro\b|\bcpo\b|chief human resource|chief people|chief hr|chief talent", d):
        return ("CHRO / Chief People Officer", 2)
    if re.search(r"talent acquisition|recruitment|recruiting|\bta\b", d):
        return ("Head of Talent Acquisition / Recruitment", 1) if SENIOR_RE.search(d) else None
    if re.search(r"hr operations|hr ops|hr shared service|people operations|human resources operations|hr services", d):
        return ("Head of HR Operations", 3)
    if re.search(r"\b(regional|zonal|cluster|area|state|south|north|east|west|city)\b", d) and \
            re.search(r"\bhr\b|human resource|\bpeople\b", d):
        return ("Regional HR Head", 4)
    if re.search(r"\bhr\b|human resource|\bpeople\b|\btalent\b", d) and SENIOR_RE.search(d):
        return ("HR leader (other)", 6)
    if re.search(r"operations|business head|retail|stores|supply chain|logistics|region|sales|fulfil|warehouse|branch", d) \
            and TOP_RE.search(d):
        return ("Business / Operations Head", 5)
    return None


def company_tokens(name):
    toks = re.findall(r"[a-z0-9]+", name.lower())
    keep = [t for t in toks if len(t) > 2 and t not in GENERIC_NAME_WORDS]
    return keep or [t for t in toks if len(t) > 1]


def mentions_company(name, text):
    t = (text or "").lower()
    return any(tok in t for tok in company_tokens(name))


def domain_of(url):
    host = urlparse(normalise_url(url)).netloc.lower()
    return re.sub(r"^www\d*\.", "", host)


def clean_linkedin(url):
    return str(url or "").split("?")[0].split("#")[0].strip()


def people_queries(name):
    return [
        f'site:linkedin.com/in ("Head of Talent Acquisition" OR "Head of Recruitment" OR "Talent Acquisition Head" '
        f'OR "Chief People Officer" OR CHRO OR "Chief Human Resources Officer" OR "VP HR") "{name}"',
        f'site:linkedin.com/in ("HR Operations" OR "Regional HR" OR "HR Head" OR "Head of HR" OR "Director HR" '
        f'OR "Operations Head" OR "Business Head") "{name}"',
    ]


def fallback_people_query(name):
    return f'site:linkedin.com/in "{name}" ("Talent Acquisition" OR "Human Resources" OR "HR" OR "Recruitment" OR "People")'


def contact_query(name):
    return f'"{name}" careers recruitment HR contact email phone'


def extract_company_contacts(name, results):
    emails, phones, seen_e, seen_p = [], [], set(), set()
    for r in results:
        link = r.get("link", "")
        text = f"{r.get('title', '')} {r.get('snippet', '')}"
        if not mentions_company(name, f"{text} {link}"):
            continue
        for m in EMAIL_RE.findall(text):
            low = m.lower().rstrip(".")
            domain = low.split("@")[1]
            if low in seen_e or domain.rsplit(".", 1)[-1] in IMG_TLDS or any(f in domain for f in FREE_MAIL):
                continue
            seen_e.add(low)
            emails.append({"value": low, "source": link})
        for m in PHONE_RE.findall(text):
            digits = re.sub(r"\D", "", m)[-10:]
            if digits not in seen_p:
                seen_p.add(digits)
                phones.append({"value": "+91 " + digits, "source": link})
    return emails[:3], phones[:2]


def build_people_evidence(results):
    ev, lines, seen, k = {}, [], set(), 0
    for r in results:
        link = r.get("link", "")
        if not LINKEDIN_PROFILE.search(link) or link in seen:
            continue
        seen.add(link)
        k += 1
        pid = f"P{k}"
        ev[pid] = {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "link": clean_linkedin(link)}
        lines.append(f"{pid} | {r.get('title', '')[:140]} | {r.get('snippet', '')[:220]}")
        if k >= 18:
            break
    return ev, "\n".join(lines)


def extract_people_batch(batch):
    blocks = [f"### COMPANY {i}: {b['name']}\nCONTEXT: {b['context']}\n{b['evidence']}" for i, b in enumerate(batch, 1)]
    prompt = f"""You are building a list of recruitment decision-makers for a sales team at Hunar Select, an Indian company that supplies outsourced frontline and volume recruitment to enterprises.

For each company below, read the numbered LinkedIn search results (a title and a snippet each) and list the people who currently work at THAT company in HR, talent acquisition, recruitment, or senior operations and business leadership.

Rules:
- Use only the evidence given. Never invent names, titles or companies.
- name: the person's name exactly as written in the result title.
- designation: the job title exactly as shown for THIS company.
- works_at_company: true only if the title or snippet shows the person holds that role at this company now. If the result shows a former role, a different company, or only mentions the company in passing, set it to false.
- relevance: one sentence on why this person matters for the company's hiring opportunity, using the CONTEXT line (hiring signals, regions).
- Return an empty list for a company with no valid people.

Return a JSON array. Each item has exactly these keys: company, evidence_id, name, designation, works_at_company, relevance.

COMPANIES AND EVIDENCE:
{chr(10).join(blocks)}
"""
    text, model = gemini_generate(prompt, json_mode=True, timeout=100, max_models=4, prefer_lite=True)
    return parse_json_list(text), model


def clean_person(item, company, ev_map, diag):
    ev = ev_map.get(str(item.get("evidence_id", "")).strip().upper())
    if ev is None:
        diag["rejected_bad_evidence_id"] += 1
        return None
    name = re.sub(r"\s+", " ", str(item.get("name", "")).strip())
    if len(name.split()) < 2 or name.split()[0].lower() not in ev["title"].lower():
        diag["rejected_name_not_in_result"] += 1
        return None
    designation = str(item.get("designation", "")).strip()
    persona = classify_persona(designation)
    if persona is None:
        diag["rejected_not_relevant_role"] += 1
        return None
    works = item.get("works_at_company")
    if not (works is True or str(works).lower() == "true"):
        diag["rejected_not_current"] += 1
        return None
    if not mentions_company(company, f"{ev['title']} {ev['snippet']}"):
        diag["rejected_company_mismatch"] += 1
        return None
    return {"Company": company, "Name": name, "Designation": designation, "Persona": persona[0],
            "Priority": persona[1], "LinkedIn": ev["link"], "Email": "", "Phone": "",
            "Contact status": "LinkedIn profile", "Contact source": "",
            "Relevance": str(item.get("relevance", "")).strip(),
            "Source note": "Public LinkedIn search result. Verify the current role before outreach."}


def hunter_people(domain):
    key = get_secret("HUNTER_API_KEY")
    r = requests.get("https://api.hunter.io/v2/domain-search",
                     params={"domain": domain, "department": "hr", "limit": 10, "api_key": key}, timeout=30)
    if r.status_code != 200:
        return [], f"Hunter returned {r.status_code} for {domain}: {r.text[:120]}"
    out = []
    for e in (r.json().get("data") or {}).get("emails") or []:
        if not e.get("value"):
            continue
        src = next((s.get("uri") for s in (e.get("sources") or []) if s.get("uri")), "")
        out.append({"email": e["value"], "first": e.get("first_name") or "", "last": e.get("last_name") or "",
                    "position": e.get("position") or "", "confidence": e.get("confidence"),
                    "linkedin": e.get("linkedin_url") or e.get("linkedin") or "", "phone": e.get("phone_number") or "",
                    "source": src, "type": e.get("type") or ""})
    return out, ""


def apollo_match(name, linkedin_url, domain):
    key = get_secret("APOLLO_API_KEY")
    r = requests.post("https://api.apollo.io/api/v1/people/match",
                      headers={"X-Api-Key": key, "Content-Type": "application/json", "Cache-Control": "no-cache"},
                      json={"name": name, "linkedin_url": linkedin_url, "domain": domain,
                            "reveal_personal_emails": False}, timeout=30)
    if r.status_code != 200:
        return None, f"Apollo returned {r.status_code}: {r.text[:120]}"
    person = r.json().get("person") or {}
    email = person.get("email") or ""
    if not email or "not_unlocked" in email or "@" not in email:
        email = ""
    phones = person.get("phone_numbers") or []
    phone = next((p.get("sanitized_number") for p in phones if p.get("sanitized_number")), "")
    return {"email": email, "status": person.get("email_status") or "", "phone": phone}, ""


def run_decision_makers(enr, sig, log):
    rows = {r["Company"]: r for r in enr["rows"]}
    companies = list(rows)
    top_signal = {}
    for s in sig["signals"]:
        top_signal.setdefault(s["Company"], s)
    diag = {"searches": 0, "linkedin_results": 0, "ai_returned": 0, "rejected_bad_evidence_id": 0,
            "rejected_name_not_in_result": 0, "rejected_not_relevant_role": 0, "rejected_not_current": 0,
            "rejected_company_mismatch": 0, "hunter_calls": 0, "hunter_people_added": 0,
            "apollo_calls": 0, "apollo_emails_found": 0}
    warnings, items, evidence, contacts = [], [], {}, {}
    for i, name in enumerate(companies, 1):
        log(f"Searching LinkedIn decision-makers: {name} ({i} of {len(companies)})")
        results = []
        used_queries = people_queries(name)
        for q in used_queries:
            diag["searches"] += 1
            try:
                results += serper_raw(q, 10).get("organic", [])
            except Exception as ex:
                warnings.append(f"People search failed for {name}: {ex}")

        ev_map, text = build_people_evidence(results)
        if len(ev_map) < 3:
            fq = fallback_people_query(name)
            used_queries.append(fq)
            diag["searches"] += 1
            try:
                results += serper_raw(fq, 10).get("organic", [])
                ev_map, text = build_people_evidence(results)
            except Exception:
                pass

        diag["searches"] += 1
        try:
            cres = serper_raw(contact_query(name), 8).get("organic", [])
        except Exception as ex:
            cres = []
            warnings.append(f"Contact search failed for {name}: {ex}")
        emails, phones = extract_company_contacts(name, cres)
        contacts[name] = {"emails": emails, "phones": phones}
        diag["linkedin_results"] += len(ev_map)
        ts = top_signal.get(name)
        context = (f"top hiring signal: {ts['Summary']} (locations: {ts['Locations'] or 'not stated'}). "
                   if ts else "no hiring signal found. ") + f"Operating locations: {rows[name].get('Operating locations') or 'not stated'}."
        evidence[name] = {"queries": used_queries + [contact_query(name)],
                          "links": [v["link"] for v in ev_map.values()]}
        if ev_map:
            items.append({"name": name, "evidence": text, "ev_map": ev_map, "context": context})

    people_by, ta_counts, model = {}, {}, ""
    for start in range(0, len(items), 3):
        batch = items[start:start + 3]
        log(f"AI is verifying LinkedIn profiles: companies {start + 1} to {start + len(batch)} of {len(items)}...")
        try:
            data, model = with_retry(extract_people_batch, batch)
            maps = {norm(b["name"]): b for b in batch}
            for item in data:
                if not isinstance(item, dict):
                    continue
                diag["ai_returned"] += 1
                b = maps.get(norm(item.get("company", "")))
                if b is None:
                    continue
                p = clean_person(item, b["name"], b["ev_map"], diag)
                if p:
                    people_by.setdefault(b["name"], []).append(p)
        except Exception as ex:
            warnings.append(f"AI step failed for people batch starting at company {start + 1}: {ex}")
        time.sleep(1)
    for name, plist in people_by.items():
        uniq, seen = [], set()
        for p in sorted(plist, key=lambda p: p["Priority"]):
            if p["LinkedIn"] not in seen and norm(p["Name"]) not in seen:
                seen.add(p["LinkedIn"])
                seen.add(norm(p["Name"]))
                uniq.append(p)
        ta_counts[name] = sum(1 for p in uniq if p["Priority"] == 1)
        people_by[name] = uniq[:6]

    order = [s["Company"] for s in sig["summary"]]
    try:
        hunter_n = int(get_secret("HUNTER_MAX_COMPANIES") or 10)
        apollo_n = int(get_secret("APOLLO_MAX_CONTACTS") or 6)
    except Exception:
        hunter_n, apollo_n = 10, 6
    if get_secret("HUNTER_API_KEY"):
        for name in order[:hunter_n]:
            domain = domain_of(rows[name].get("Website") or "")
            if not domain:
                continue
            log(f"Looking up public work emails (Hunter): {name}")
            diag["hunter_calls"] += 1
            found, err = hunter_people(domain)
            if err:
                warnings.append(err)
                if any(c in err for c in ("401", "403", "429")):
                    break
                continue
            plist = people_by.setdefault(name, [])
            for h in found:
                full = f"{h['first']} {h['last']}".strip()
                source = f"Hunter, public web source: {h['source']}" if h["source"] else "Hunter"
                if h["confidence"] is not None:
                    source += f" (confidence {h['confidence']}%)"
                if not full:
                    if h["email"] not in [e["value"] for e in contacts[name]["emails"]]:
                        contacts[name]["emails"].append({"value": h["email"], "source": source})
                    continue
                match = next((p for p in plist if norm(p["Name"]) == norm(full)), None)
                if match:
                    match.update({"Email": h["email"], "Phone": match["Phone"] or h["phone"],
                                  "Contact status": "Email found (Hunter)", "Contact source": source})
                    continue
                persona = classify_persona(h["position"]) or ("HR contact (position not listed)", 7)
                if persona[1] == 7 and sum(1 for p in plist if p["Priority"] == 7) >= 2:
                    continue
                plist.append({"Company": name, "Name": full, "Designation": h["position"] or "HR team (Hunter)",
                              "Persona": persona[0], "Priority": persona[1],
                              "LinkedIn": clean_linkedin(h["linkedin"]), "Email": h["email"], "Phone": h["phone"],
                              "Contact status": "Email found (Hunter)", "Contact source": source,
                              "Relevance": "Listed as an HR contact on public web sources.",
                              "Source note": "From Hunter. Verify the current role before outreach."})
                diag["hunter_people_added"] += 1
    if get_secret("APOLLO_API_KEY"):
        used, stop = 0, False
        for name in order[:hunter_n]:
            domain = domain_of(rows[name].get("Website") or "")
            for p in sorted(people_by.get(name, []), key=lambda p: p["Priority"]):
                if stop or used >= apollo_n:
                    break
                if p["Email"] or not p["LinkedIn"] or p["Priority"] > 3:
                    continue
                log(f"Looking up work email (Apollo): {p['Name']}")
                used += 1
                diag["apollo_calls"] += 1
                res, err = apollo_match(p["Name"], p["LinkedIn"], domain)
                if err:
                    warnings.append(err)
                    stop = True
                    break
                if res and (res["email"] or res["phone"]):
                    if res["email"]:
                        diag["apollo_emails_found"] += 1
                        p["Email"] = res["email"]
                        p["Contact status"] = "Email found (Apollo)"
                        p["Contact source"] = f"Apollo, email status: {res['status'] or 'unknown'}"
                    if res["phone"]:
                        p["Phone"] = res["phone"]

    people = []
    for plist in people_by.values():
        people += sorted(plist, key=lambda p: p["Priority"])
    return {"run_id": enr["run_id"], "people": people, "company_contacts": contacts, "ta_counts": ta_counts,
            "evidence": evidence, "warnings": warnings, "model": model, "diagnostics": diag}


# ---------------- Stage 5: scoring (25 / 20 / 15 / 15 / 10 / 10 / 5) ----------------
WEIGHTS = [
    ("Hiring intensity", 0.25),
    ("Open positions / hiring volume", 0.20),
    ("Frontline workforce", 0.15),
    ("Hiring urgency", 0.15),
    ("Outsourcing potential", 0.10),
    ("Geographic expansion", 0.10),
    ("Decision-maker accessibility", 0.05),
]
OUTSOURCING_BASE = {"Logistics": 75, "Quick Commerce": 70, "Retail": 70, "BFSI": 60, "Healthcare": 55}
INDUSTRY_FIT = {"Quick Commerce": 100, "Retail": 100, "Logistics": 100, "BFSI": 80, "Healthcare": 75}
OPEN_STEPS = [(1, 20), (50, 40), (200, 60), (500, 80), (2000, 100)]
FRONTLINE_STEPS = [(1, 20), (1000, 40), (5000, 60), (15000, 80), (50000, 100)]
LOCATION_STEPS = [(1, 20), (5, 50), (15, 80), (50, 100)]
PARTNER_RE = re.compile(r"staffing|outsourc|recruitment partner|third-party|contract(ual)? (staff|worker)|agency", re.I)


def step_score(value, steps):
    score = 0
    for minimum, s in steps:
        if value >= minimum:
            score = s
    return score


def parse_volume(text):
    t = str(text or "")
    if not t or t.lower().startswith("unknown"):
        return 0
    nums = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", t.split("(")[0]) if n.replace(",", "").isdigit()]
    return max(nums) if nums else 0


def count_locations(row):
    locs = [x for x in (row.get("Operating locations") or "").split(", ") if x.strip()]
    cities = [int(n.replace(",", "")) for n in re.findall(r"(\d[\d,]*)\s*(?:\+|-plus)?\s*(?:cities|towns)",
                                                        row.get("Footprint (as stated)") or "", re.I)
              if n.replace(",", "").isdigit()]
    return max([len(locs)] + cities)


def tier_for(score):
    return "P1" if score >= 75 else "P2" if score >= 60 else "P3" if score >= 45 else "P4"


def score_company(row, sigs, people, contacts, industry, ta_count):
    pts = round(sum(s["Points"] for s in sigs), 1)
    s1 = min(100, pts * 2.5)
    n1 = f"{len(sigs)} signals worth {pts} points (Confirmed 10, Observed 6, Inferred 3, scaled by recency)"

    vol, seen = 0, set()
    for s in sigs:
        if s["Source quality"] == "Normal" and s["Source"] not in seen:
            v = parse_volume(s["Est. volume"])
            if v:
                vol += v
                seen.add(s["Source"])
    s2 = step_score(vol, OPEN_STEPS)
    n2 = f"About {vol:,} roles stated or derived in sourced signals" if vol else "No sourced hiring volume found"

    fl = row.get("Est. frontline (estimate)") or 0
    s3 = step_score(fl, FRONTLINE_STEPS)
    n3 = f"Estimated frontline workforce {fl:,}" if fl else "No sourced headcount, so no frontline estimate"

    exp = [s for s in sigs if s["Signal type"][:2] in ("S2", "S3") and (s["Age (days)"] is None or s["Age (days)"] <= 365)]
    exp_locs = {x.strip() for s in exp for x in s["Locations"].split(",") if x.strip()}
    s4 = 0 if not exp else 40 if len(exp) == 1 else 60 if len(exp) == 2 else 80
    if len(exp) >= 5 or len(exp_locs) >= 5:
        s4 = 100
    n4 = f"{len(exp)} expansion signals in the last 12 months, {len(exp_locs)} named locations"

    strong = [s for s in sigs if s["Evidence level"] in ("Confirmed", "Observed")]
    best = 0
    for s in strong:
        age = s["Age (days)"]
        v = 40 if age is None else 100 if age <= 30 else 70 if age <= 90 else 40 if age <= 180 else 20
        if s["Source quality"] == "Low":
            v = min(v, 40)
        best = max(best, v)
    if not strong:
        best = 20 if sigs else 0
    seasonal = any(s["Signal type"].startswith("S6") and s["Age (days)"] is not None and s["Age (days)"] <= 90 for s in sigs)
    s5 = min(100, best + (10 if seasonal else 0))
    n5 = ("Most recent strong signal drives this score" if strong else "Only inferred signals" if sigs else "No signals") \
        + (", plus a seasonal-peak bonus" if seasonal else "")

    n_loc = max(count_locations(row), len({x.strip() for s in sigs for x in s["Locations"].split(",") if x.strip()}))
    s6 = OUTSOURCING_BASE.get(industry, 60)
    adj = [f"{industry} base {s6}"]
    if any(s["Signal type"].startswith("S7") for s in sigs):
        s6 += 10
        adj.append("+10 attrition signal")
    if any(PARTNER_RE.search(s["Summary"]) for s in sigs):
        s6 += 10
        adj.append("+10 uses staffing partners")
    if n_loc >= 10:
        s6 += 10
        adj.append("+10 hiring across 10+ locations")
    if ta_count >= 5:
        s6 -= 10
        adj.append("-10 large in-house talent team found")
    s6 = max(0, min(100, s6))
    n6 = ", ".join(adj)

    pri = {p["Priority"] for p in people if p["LinkedIn"]}
    s7 = (35 if 1 in pri else 0) + (20 if (3 in pri or 4 in pri) else 0) + (15 if 2 in pri else 0) + (15 if 5 in pri else 0)
    person_contact = any(p["Email"] or p["Phone"] for p in people)
    company_contact = bool(contacts.get("emails") or contacts.get("phones"))
    s7 += 15 if person_contact else 8 if company_contact else 0
    s7 = min(100, s7)
    n7 = f"{len(people)} decision-makers; " + ("a sourced person-level contact found" if person_contact
                                                else "a public company recruitment contact found" if company_contact
                                                else "no sourced email or phone")

    # Exact order matching WEIGHTS: 25 / 20 / 15 / 15 / 10 / 10 / 5
    subs = [(s1, n1), (s2, n2), (s3, n3), (s5, n5), (s6, n6), (s4, n4), (s7, n7)]
    breakdown, total = [], 0.0
    for (label, w), (score, note) in zip(WEIGHTS, subs):
        total += score * w
        breakdown.append({"Criterion": label, "Sub-score (0-100)": round(score), "Weight": f"{int(w * 100)}%",
                          "Contribution": round(score * w, 1), "Basis": note})
    loc_score = step_score(n_loc, LOCATION_STEPS)
    role_fit = round(FRONTLINE_SHARE.get(industry, 0.6) * 100)
    icp = round(0.40 * s3 + 0.25 * loc_score + 0.20 * INDUSTRY_FIT.get(industry, 80) + 0.15 * role_fit)
    return {"priority": round(total, 1), "icp": icp, "breakdown": breakdown, "vol": vol, "n_loc": n_loc,
            "urgency_score": s5}


def outreach_angle(row, sigs, people):
    best = max(sigs, key=lambda s: s["Points"]) if sigs else None
    if best:
        roles = best["Roles"] or "frontline roles"
        where = best["Locations"] or "its key cities"
        text = f"Open with: {best['Summary']} ({best['Date']}). Offer volume hiring for {roles} in {where}."
    else:
        text = f"No fresh trigger found. Lead with: {row.get('Potential recruitment needs (inferred)') or 'frontline volume hiring support'}"
    if people:
        p = sorted(people, key=lambda p: (p["Priority"], 0 if (p["Email"] or p["Phone"]) else 1))[0]
        text += f" Approach {p['Name']} ({p['Designation']})."
    else:
        text += " Search LinkedIn for the Head of Talent Acquisition."
    return text


def run_scoring(enr, sig, dm):
    industry = enr["industry"]
    out, breakdowns = [], {}
    for r in enr["rows"]:
        name = r["Company"]
        sigs = [s for s in sig["signals"] if s["Company"] == name]
        people = [p for p in dm["people"] if p["Company"] == name]
        contacts = dm["company_contacts"].get(name, {})
        res = score_company(r, sigs, people, contacts, industry, dm["ta_counts"].get(name, 0))
        breakdowns[name] = res["breakdown"]
        counts = {lvl: sum(1 for s in sigs if s["Evidence level"] == lvl) for lvl in EVIDENCE_POINTS}
        dated = [s["Date"] for s in sigs if parse_date(s["Date"])]
        latest_sig_date = max(dated) if dated else ""
        freshness = compute_freshness(
            latest_sig_date,
            has_signals=bool(sigs),
            has_verified_headcount=(r.get("Size status") == "Verified"),
        )
        top = max(sigs, key=lambda s: s["Points"]) if sigs else None
        best = sorted(people, key=lambda p: (p["Priority"], 0 if (p["Email"] or p["Phone"]) else 1))[:1]
        contact_text = "Not found"
        if best:
            b = best[0]
            contact_text = f"{b['Name']} ({b['Designation']})" + (f" | {b['Email']}" if b["Email"] else "")
        elif contacts.get("emails"):
            contact_text = f"Company HR: {contacts['emails'][0]['value']}"
        all_dms_text = " | ".join(
            f"{p['Name']} ({p['Designation']} - {p['Persona']}) [{p['LinkedIn']}]"
            + (f" <{p['Email']}>" if p.get("Email") else "")
            for p in people
        ) or "None found"
        urgency = "High" if res["urgency_score"] >= 70 else "Medium" if res["urgency_score"] >= 40 else "Low" \
            if res["urgency_score"] > 0 else "None"
        out.append({
            "Company": name,
            "Tier": tier_for(res["priority"]),
            "Priority score": res["priority"],
            "ICP fit": res["icp"],
            "Freshness": freshness,
            "Employees": r.get("Employees"),
            "Est. frontline (estimate)": r.get("Est. frontline (estimate)"),
            "Size status": r.get("Size status"),
            "Signals (C/O/I)": f"{counts['Confirmed']}/{counts['Observed']}/{counts['Inferred']}",
            "Hiring volume (sourced)": res["vol"] or None,
            "Urgency": urgency,
            "Latest signal": latest_sig_date,
            "Top signal": top["Summary"] if top else "No signals found",
            "Top signal source": top["Source"] if top else "",
            "Decision-makers": len(people),
            "Best contact": contact_text,
            "All decision-makers": all_dms_text,
            "Outreach angle": outreach_angle(r, sigs, people),
        })
    out.sort(key=lambda r: (r["Priority score"], r["ICP fit"]), reverse=True)
    for i, r in enumerate(out, 1):
        r["Rank"] = i
    return {"run_id": enr["run_id"], "rows": out, "breakdowns": breakdowns}


# ---------------- Assisted Outreach Draft Generator ----------------
def build_default_outreach(company, enr_row, sigs, person, industry):
    best = max(sigs, key=lambda s: s["Points"]) if sigs else None
    recipient_name = person["Name"].split()[0] if person else "there"
    recipient_title = person["Designation"] if person else "Head of Talent Acquisition"
    roles = (best["Roles"] if best and best.get("Roles") else enr_row.get("Frontline roles (inferred)")) or ROLE_HINTS[industry]
    locations = (best["Locations"] if best and best.get("Locations") else enr_row.get("Operating locations")) or "key Indian cities"
    trigger = best["Summary"] if best else f"your ongoing {industry.lower()} workforce expansion across India"
    trigger_date = f" ({best['Date']})" if best and best.get("Date") and best["Date"] != "Date unverified" else ""

    email_subject = f"Frontline hiring support for {company}'s expansion ({roles.split(',')[0].strip()})"
    email_body = (
        f"Hi {recipient_name},\n\n"
        f"Noticed the recent update on {company} — {trigger}{trigger_date}. "
        f"Scaling {roles} across {locations} typically puts intense turnaround pressure on regional TA teams.\n\n"
        f"At Hunar.AI (Hunar Select), we help {industry} enterprises source, pre-screen, and onboard verified frontline talent "
        f"at speed with AI-led screening and dedicated field fulfilment.\n\n"
        f"Worth a brief 15-minute conversation next week to share how we're helping similar {industry} leaders cut time-to-hire?\n\n"
        f"Best regards,\n"
        f"Enterprise Partnerships | Hunar.AI"
    )
    li_note = (
        f"Hi {recipient_name}, saw {company}'s update on {trigger[:95]}. "
        f"We help {industry} leaders scale {roles.split(',')[0].strip()} hiring across {locations.split(',')[0].strip()}. "
        f"Would love to connect!"
    )[:295]
    li_message = (
        f"Hi {recipient_name},\n\n"
        f"Thanks for connecting! Given your leadership as {recipient_title} at {company}, I wanted to reach out regarding "
        f"{trigger}{trigger_date}.\n\n"
        f"We partner with high-growth {industry} teams to deliver pre-vetted {roles} across {locations} with high show-up and retention rates. "
        f"Open to a quick 10-min chat this week to exchange notes on your upcoming hiring targets?"
    )
    call_points = (
        f"1. Trigger Hook: Reference '{trigger}{trigger_date}' and ask which cities/clusters have the sharpest ramp-up.\n"
        f"2. Volume & Bottleneck Probe: Ask how they currently source and screen {roles} in {locations}, and where drop-offs happen (sourcing vs. interview show-up vs. Day-1 joining).\n"
        f"3. Hunar Select Value Prop: Highlight Hunar.AI's vernacular AI voice screening, hyperlocal candidate pool, and SLA-backed volume onboarding for {industry} teams.\n"
        f"4. Low-Friction Pilot Ask: Propose starting with one high-priority city or role mandate as a 2-week turnaround benchmark."
    )
    return {
        "email_subject": email_subject,
        "email_body": email_body,
        "linkedin_note": li_note,
        "linkedin_message": li_message,
        "call_points": call_points,
        "mode": "Signal-Grounded Template",
    }


def generate_ai_outreach(company, enr_row, sigs, person, industry):
    best = max(sigs, key=lambda s: s["Points"]) if sigs else None
    recipient = f"{person['Name']} ({person['Designation']} - {person['Persona']})" if person else "Head of Talent Acquisition / HR"
    trigger_text = f"{best['Summary']} (Date: {best['Date']}, Roles: {best['Roles']}, Locations: {best['Locations']})" if best else "General frontline expansion"
    prompt = f"""You are an enterprise B2B sales strategist at Hunar.AI (Hunar Select), an Indian recruitment platform specializing in high-volume frontline hiring (sourcing, AI voice screening, and rapid onboarding across Indian cities).

Write a personalized multi-channel outreach pack for:
- Target Account: {company} (Industry: {industry}, HQ: {enr_row.get('HQ', 'India')})
- Decision-Maker: {recipient}
- Captured Hiring Signal: {trigger_text}
- Typical Frontline Roles: {enr_row.get('Frontline roles (inferred)', ROLE_HINTS[industry])}
- Operating Locations: {enr_row.get('Operating locations', 'India')}

RULES:
- Ground every message strictly in the captured hiring signal and roles above. Never invent facts or metrics.
- linkedin_note MUST be under 280 characters.
- Keep tone crisp, consultative, and tailored to Indian enterprise TA/HR leaders.

Return ONLY a JSON object with exact keys:
email_subject, email_body, linkedin_note, linkedin_message, call_points
"""
    text, _ = gemini_generate(prompt, json_mode=True, timeout=60, max_models=4, prefer_lite=True)
    obj = parse_json_obj(text)
    if not obj.get("email_body"):
        raise RuntimeError("AI returned an incomplete outreach draft.")
    obj["linkedin_note"] = str(obj.get("linkedin_note", ""))[:298]
    obj["mode"] = "AI-Personalized"
    return obj


# ---------------- Macro Industry Intelligence Generator ----------------
def generate_macro_brief(sector):
    query = f'"{sector}" India hiring OR jobs OR expansion OR workforce 2026'
    news = fetch_news(query)[:12]
    headlines = "\n".join(f"- [{n['date']}] {n['title']} ({n['source']})" for n in news) or "No recent RSS items."
    prompt = f"""You are a labor-market analyst for Hunar.AI, an Indian frontline recruitment company.
Analyze these recent Indian news headlines for the **{sector}** sector and produce a concise macro hiring brief.

HEADLINES:
{headlines}

Return ONLY a JSON object with exact keys:
- demand_outlook: 2-sentence summary of current frontline hiring momentum in India for {sector}
- key_drivers: array of 3 concise bullet strings explaining what is driving workforce demand
- hotspot_cities: array of 4 to 6 Indian cities/regions seeing hiring activity
- hunar_pitch_theme: 1-sentence strategic positioning angle for Hunar Select sales reps targeting {sector} this quarter
"""
    text, _ = gemini_generate(prompt, json_mode=True, timeout=60, max_models=4, prefer_lite=True)
    obj = parse_json_obj(text)
    obj["headlines"] = news[:6]
    return obj


# ---------------- Stage 6: pipeline & saved runs ----------------
RUN_KEYS = ("disc", "enr", "sig", "dm", "score", "review_states", "outreach_drafts")
SAVE_DIR = "saved_runs"


def save_run():
    S = st.session_state
    if "disc" not in S:
        return
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        d = S["disc"]
        fname = f"{d['industry']}__{d['size']}".replace(" ", "_").replace(",", "").replace("+", "plus") + ".json"
        with open(os.path.join(SAVE_DIR, fname), "w") as f:
            json.dump({k: S[k] for k in RUN_KEYS if k in S}, f, default=str)
    except Exception:
        pass


def load_run_data(data):
    S = st.session_state
    for k in RUN_KEYS:
        S.pop(k, None)
    for k in RUN_KEYS:
        if k in data:
            S[k] = data[k]
    S.setdefault("review_states", {})
    S.setdefault("outreach_drafts", {})


def get_review_state(company):
    return st.session_state.setdefault("review_states", {}).get(company, "Unreviewed")


def set_review_state(company, state):
    st.session_state.setdefault("review_states", {})[company] = state
    save_run()


def run_all(industry, size, log, resume=False):
    S = st.session_state
    if resume and "disc" in S:
        industry, size = S["disc"]["industry"], S["disc"]["size"]
    else:
        for k in RUN_KEYS:
            S.pop(k, None)
        S["review_states"] = {}
        S["outreach_drafts"] = {}
    if "disc" not in S:
        log("STEP 1 of 5: finding candidate companies")
        S["disc"] = run_discovery(industry, size, log)
        save_run()
    if "enr" not in S:
        log("STEP 2 of 5: headcount, locations and frontline estimates")
        S["enr"] = run_enrichment(S["disc"], log)
        save_run()
    if "sig" not in S:
        log("STEP 3 of 5: hiring signals")
        S["sig"] = run_signals(S["enr"], log)
        save_run()
    if "dm" not in S:
        log("STEP 4 of 5: LinkedIn decision-makers and contacts")
        S["dm"] = run_decision_makers(S["enr"], S["sig"], log)
        save_run()
    if "score" not in S:
        log("STEP 5 of 5: scoring and ranking")
        S["score"] = run_scoring(S["enr"], S["sig"], S["dm"])
        save_run()


# ---------------- Clean B2B SaaS UI ----------------
def esc(val):
    return _html.escape(str(val or ""))


def render_html(raw_html):
    """Strip line indentation so Streamlit Markdown never renders HTML as a code block."""
    clean = "\n".join(line.strip() for line in raw_html.strip().splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0F172A;
}
.stApp {
    background-color: #F8FAFC;
}
h1, h2, h3 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.02em;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1240px !important;
}
.saas-hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 60%, #312E81 100%);
    border-radius: 16px;
    padding: 24px 28px;
    color: #FFFFFF;
    margin-bottom: 18px;
    box-shadow: 0 10px 25px -8px rgba(15, 23, 42, 0.25);
}
.saas-hero h1 {
    color: #FFFFFF !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    margin: 0 0 6px 0 !important;
}
.saas-hero p {
    color: #CBD5E1;
    font-size: 14px;
    margin: 0;
}
.bento-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 14px 0 20px 0;
}
.bento-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 14px 18px;
}
.bento-label {
    font-size: 11.5px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}
.bento-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 24px;
    font-weight: 800;
    color: #0F172A;
}
.prospect-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
}
.card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}
.rank-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    background: #F1F5F9;
    color: #334155;
    padding: 3px 8px;
    border-radius: 6px;
    margin-right: 6px;
}
.company-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #0F172A;
}
.badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    margin-left: 6px;
}
.tier-P1 { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
.tier-P2 { background: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; }
.tier-P3 { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
.tier-P4 { background: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }
.urg-High { background: #FFF1F2; color: #BE123C; }
.urg-Medium { background: #FFFBEB; color: #B45309; }
.urg-Low, .urg-None { background: #F1F5F9; color: #64748B; }
.fresh-Fresh { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
.fresh-Needs { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
.fresh-Stale { background: #F1F5F9; color: #64748B; border: 1px solid #CBD5E1; }
.rev-Verified { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
.rev-Disputed { background: #FFF1F2; color: #BE123C; border: 1px solid #FECDD3; }
.rev-Unreviewed { background: #F8FAFC; color: #64748B; border: 1px solid #E2E8F0; }
.ev-Confirmed { background: #ECFDF5; color: #047857; }
.ev-Observed { background: #EFF6FF; color: #1D4ED8; }
.ev-Inferred { background: #F8FAFC; color: #475569; border: 1px solid #E2E8F0; }
.score-num {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 20px;
    font-weight: 800;
    color: #4F46E5;
    text-align: right;
}
.metrics-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    background: #F8FAFC;
    border: 1px solid #F1F5F9;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 12px;
}
.m-item span {
    display: block;
    font-size: 11px;
    color: #64748B;
}
.m-item strong {
    font-size: 13px;
    color: #0F172A;
    font-weight: 600;
}
.signal-callout {
    background: #FAFAFF;
    border-left: 3px solid #4F46E5;
    padding: 9px 12px;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
    color: #334155;
    margin-bottom: 12px;
}
.dm-header {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748B;
    margin-bottom: 6px;
}
.dm-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 8px;
    margin-bottom: 12px;
}
.dm-chip {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 8px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
}
.dm-persona {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    color: #4338CA;
    background: #EEF2FF;
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 2px;
}
.dm-name {
    font-size: 13px;
    font-weight: 700;
    color: #0F172A;
}
.dm-role {
    font-size: 11.5px;
    color: #475569;
}
.dm-mail {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    color: #047857;
    margin-top: 2px;
}
.li-btn {
    font-size: 11px;
    font-weight: 600;
    color: #0A66C2 !important;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    padding: 4px 8px;
    border-radius: 6px;
    text-decoration: none !important;
    white-space: nowrap;
}
.card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    padding-top: 10px;
    border-top: 1px solid #F1F5F9;
    font-size: 12.5px;
    color: #475569;
}
.action-link {
    padding: 4px 10px;
    border-radius: 6px;
    background: #EEF2FF;
    color: #4F46E5 !important;
    font-weight: 600;
    font-size: 12px;
    text-decoration: none !important;
    margin-left: 6px;
}
div.stButton > button[kind="primary"] {
    background: #4F46E5 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
</style>
""")

LINK_COLUMNS = {
    "Website": st.column_config.LinkColumn("Website"),
    "Source": st.column_config.LinkColumn("Source"),
    "Employee source": st.column_config.LinkColumn("Employee source"),
    "LinkedIn": st.column_config.LinkColumn("LinkedIn"),
    "Employees": st.column_config.NumberColumn("Employees", format="%d"),
    "Est. frontline (estimate)": st.column_config.NumberColumn("Est. frontline (estimate)", format="%d"),
    "Hiring volume (sourced)": st.column_config.NumberColumn("Hiring volume (sourced)", format="%d"),
}
S = st.session_state

render_html("""
<div class="saas-hero">
<h1>Hunar Select: Recruitment Intelligence Engine</h1>
<p>Discover target Indian enterprises, verify frontline workforce scale, detect live hiring signals, map LinkedIn decision-makers, and generate tailored outreach.</p>
</div>
""")

with st.container(border=True):
    col1, col2, col3, col4 = st.columns([2.2, 2.2, 1.3, 2.3])
    industry = col1.selectbox("Industry", INDUSTRIES)
    size = col2.selectbox("Company Size (Employees)", SIZE_BANDS)
    col3.text_input("Geography", value="India", disabled=True)
    with col4:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        run_clicked = st.button("Launch Research", type="primary", use_container_width=True)

    resume_clicked = st.button("Resume Unfinished Run", use_container_width=True) if ("disc" in S and "score" not in S) else False

saved = sorted(os.listdir(SAVE_DIR)) if os.path.isdir(SAVE_DIR) else []
if saved:
    with st.expander("Load Saved Run Snapshot"):
        sc1, sc2 = st.columns([3, 1])
        pick = sc1.selectbox("Saved runs", saved, label_visibility="collapsed")
        if sc2.button("Load Run", use_container_width=True):
            with open(os.path.join(SAVE_DIR, pick)) as f:
                load_run_data(json.load(f))
            st.rerun()

if run_clicked or resume_clicked:
    with st.status("Running 5-stage research pipeline...", expanded=True) as status:
        try:
            run_all(industry, size, st.write, resume=resume_clicked)
            status.update(label="Research complete.", state="complete")
            st.rerun()
        except Exception as e:
            status.update(label="Run paused. Click Resume Unfinished Run to continue.", state="error")
            st.error(str(e))

# Always show Macro Industry Intelligence alongside Account Research
top_tabs = st.tabs([
    "Account Intelligence & Outreach",
    "Macro Industry Intelligence",
])

# ==================== MACRO INDUSTRY INTELLIGENCE VIEW ====================
with top_tabs[1]:
    st.subheader("India Frontline Workforce & Outsourcing Benchmark Matrix")
    st.caption("Cross-sector baseline intelligence used by Hunar Select to size frontline pools and evaluate outsourcing propensity.")
    st.dataframe(pd.DataFrame(MACRO_SECTOR_BENCHMARKS), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Live Sector Hiring Pulse")
    mc1, mc2 = st.columns([2.5, 1.5])
    macro_sector = mc1.selectbox("Select Sector for Live Macro Brief", INDUSTRIES, index=INDUSTRIES.index(industry), key="macro_sector_pick")
    with mc2:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        gen_macro = st.button("Fetch Live Macro Hiring Trends", type="primary", use_container_width=True)

    if gen_macro:
        with st.spinner(f"Analyzing live Indian hiring and expansion news for {macro_sector}..."):
            try:
                S[f"macro_brief_{macro_sector}"] = generate_macro_brief(macro_sector)
            except Exception as ex:
                st.error(f"Could not fetch macro brief: {ex}")

    brief = S.get(f"macro_brief_{macro_sector}")
    if brief:
        with st.container(border=True):
            st.markdown(f"#### {macro_sector} — Current Demand Outlook")
            st.write(brief.get("demand_outlook", ""))
            b1, b2 = st.columns(2)
            with b1:
                st.markdown("**Key Workforce Demand Drivers**")
                for drv in brief.get("key_drivers", []):
                    st.markdown(f"- {drv}")
            with b2:
                st.markdown("**Active Hiring Geographies**")
                st.write(", ".join(brief.get("hotspot_cities", [])))
                st.info(f"**Hunar Select Pitch Angle:** {brief.get('hunar_pitch_theme', '')}")

            if brief.get("headlines"):
                with st.expander("Source News Headlines"):
                    for h in brief["headlines"]:
                        st.markdown(f"- **{h['date']}** — [{h['title']}]({h['link']}) *({h['source']})*")

    if "sig" in S and "score" in S:
        st.markdown("---")
        st.subheader(f"Active Run Sector Analytics ({S['disc']['industry']} • {S['disc']['size']})")
        sigs_all = S["sig"]["signals"]
        rows_all = S["score"]["rows"]
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Detected Signal Distribution**")
            if sigs_all:
                sig_counts = pd.Series([s["Signal type"] for s in sigs_all]).value_counts().reset_index()
                sig_counts.columns = ["Signal Type", "Count"]
                st.dataframe(sig_counts, hide_index=True, use_container_width=True)
            else:
                st.caption("No signals captured in current run.")
        with a2:
            st.markdown("**Account Tier Breakdown**")
            tier_counts = pd.Series([r["Tier"] for r in rows_all]).value_counts().reset_index()
            tier_counts.columns = ["Tier", "Accounts"]
            st.dataframe(tier_counts, hide_index=True, use_container_width=True)


# ==================== ACCOUNT INTELLIGENCE & OUTREACH ====================
with top_tabs[0]:
    if "disc" not in S:
        st.info("Select an Industry and Company Size above and click **Launch Research** (or load a saved run) to view ranked accounts, score attribution, and outreach drafts.")
    else:
        d = S["disc"]
        if (d["industry"], d["size"]) != (industry, size):
            st.info(f"Showing **{d['industry']} ({d['size']} employees)**. Click **Launch Research** to run for {industry} ({size}).")

        tabs = st.tabs([
            "Ranked Prospects & Outreach Studio",
            "Company Profiles",
            "Hiring Signals",
            "Decision-Makers",
        ])

        # ----- TAB 1: Ranked Prospects + Score Attribution + Outreach Generator -----
        with tabs[0]:
            sc = S.get("score")
            if not sc:
                st.info("Ranked prospects will appear once all stages complete.")
            else:
                rows = sc["rows"]
                for r in rows:
                    r["Review status"] = get_review_state(r["Company"])
                    r.setdefault("Freshness", compute_freshness(r.get("Latest signal", ""), bool(r.get("Latest signal")), r.get("Size status") == "Verified"))

                p1_count = sum(1 for r in rows if r["Tier"] == "P1")
                total_fl = sum(r["Est. frontline (estimate)"] or 0 for r in rows)
                total_dms = len(S.get("dm", {}).get("people", []))

                render_html(f"""
                <div class="bento-grid">
                <div class="bento-card"><div class="bento-label">Target Companies</div><div class="bento-value">{len(rows)}</div></div>
                <div class="bento-card"><div class="bento-label">Priority P1 Accounts</div><div class="bento-value">{p1_count}</div></div>
                <div class="bento-card"><div class="bento-label">Est. Frontline Pool</div><div class="bento-value">{total_fl:,}</div></div>
                <div class="bento-card"><div class="bento-label">LinkedIn Decision-Makers</div><div class="bento-value">{total_dms}</div></div>
                </div>
                """)

                f1, f2, f3, f4, f5 = st.columns([1.8, 1.8, 1.8, 1.5, 1.3])
                tiers = f1.multiselect("Tier", ["P1", "P2", "P3", "P4"], default=["P1", "P2", "P3", "P4"])
                fresh_opts = f2.multiselect("Freshness", ["Fresh", "Needs verification", "Stale"],
                                            default=["Fresh", "Needs verification", "Stale"])
                rev_opts = f3.multiselect("Human Review", ["Unreviewed", "Verified", "Disputed"],
                                          default=["Unreviewed", "Verified", "Disputed"])
                min_score = f4.slider("Min Score", 0, 100, 0)
                view_mode = f5.radio("View", ["Cards", "Table"], horizontal=True)

                shown = [
                    r for r in rows
                    if r["Tier"] in tiers
                    and r["Freshness"] in fresh_opts
                    and r["Review status"] in rev_opts
                    and r["Priority score"] >= min_score
                ]

                col_cap, col_dl = st.columns([4, 1])
                col_cap.caption(f"Showing **{len(shown)} of {len(rows)}** companies in **{d['industry']}** ({d['size']})")
                if shown:
                    cols = ["Rank", "Company", "Tier", "Priority score", "ICP fit", "Freshness", "Review status",
                            "Employees", "Est. frontline (estimate)", "Signals (C/O/I)", "Hiring volume (sourced)",
                            "Urgency", "Latest signal", "Top signal", "Decision-makers", "All decision-makers", "Outreach angle"]
                    ddf = pd.DataFrame(shown)
                    col_dl.download_button("Export CSV", ddf[cols].to_csv(index=False).encode("utf-8"),
                                           file_name="ranked_companies.csv", mime="text/csv", use_container_width=True)

                if not shown:
                    st.warning("No companies match the selected filters.")
                elif view_mode == "Table":
                    st.dataframe(ddf[cols], hide_index=True, column_config=LINK_COLUMNS, use_container_width=True)
                else:
                    enr_map = {r["Company"]: r for r in S.get("enr", {}).get("rows", [])}
                    all_people = S.get("dm", {}).get("people", [])
                    for r in shown:
                        cname = r["Company"]
                        er = enr_map.get(cname, {})
                        c_people = [p for p in all_people if p["Company"] == cname]
                        emp_txt = f"{r['Employees']:,}" if r["Employees"] else "Unverified"
                        fl_txt = f"{r['Est. frontline (estimate)']:,}" if r["Est. frontline (estimate)"] else "—"
                        vol_txt = f"{r['Hiring volume (sourced)']:,} roles" if r["Hiring volume (sourced)"] else "Inferred"
                        fresh_cls = "fresh-Needs" if r["Freshness"] == "Needs verification" else f"fresh-{r['Freshness']}"
                        rev_cls = f"rev-{r['Review status']}"

                        links_html = []
                        if er.get("Website"):
                            links_html.append(f'<a class="action-link" href="{esc(er["Website"])}" target="_blank">Website &#8599;</a>')
                        if er.get("Employee source"):
                            links_html.append(f'<a class="action-link" href="{esc(er["Employee source"])}" target="_blank">Headcount Source &#8599;</a>')
                        if r.get("Top signal source"):
                            links_html.append(f'<a class="action-link" href="{esc(r["Top signal source"])}" target="_blank">Signal Source &#8599;</a>')

                        if c_people:
                            chips = []
                            for p in c_people:
                                mail_div = f'<div class="dm-mail">{esc(p["Email"])}</div>' if p.get("Email") else ""
                                li_btn = f'<a class="li-btn" href="{esc(p["LinkedIn"])}" target="_blank">LinkedIn &#8599;</a>' if p.get("LinkedIn") else ""
                                chips.append(
                                    f'<div class="dm-chip">'
                                    f'<div><span class="dm-persona">{esc(p["Persona"])}</span>'
                                    f'<div class="dm-name">{esc(p["Name"])}</div>'
                                    f'<div class="dm-role">{esc(p["Designation"])}</div>{mail_div}</div>'
                                    f'<div>{li_btn}</div></div>'
                                )
                            dm_block = f'<div class="dm-header">Points of Contact ({len(c_people)})</div><div class="dm-grid">{"".join(chips)}</div>'
                        else:
                            li_search = f"https://www.linkedin.com/search/results/people/?keywords={quote_plus(cname + ' Talent Acquisition OR HR OR CHRO')}"
                            dm_block = (
                                f'<div class="dm-header">Points of Contact</div>'
                                f'<div class="dm-chip"><span>No public LinkedIn profile matched in search snippets.</span>'
                                f'<a class="li-btn" href="{esc(li_search)}" target="_blank">Search LinkedIn &#8599;</a></div>'
                            )

                        render_html(f"""
                        <div class="prospect-card">
                        <div class="card-top">
                        <div>
                        <span class="rank-pill">#{r['Rank']}</span>
                        <span class="company-name">{esc(cname)}</span>
                        <span class="badge tier-{esc(r['Tier'])}">{esc(r['Tier'])}</span>
                        <span class="badge urg-{esc(r['Urgency'])}">Urgency: {esc(r['Urgency'])}</span>
                        <span class="badge {fresh_cls}">{esc(r['Freshness'])}</span>
                        <span class="badge {rev_cls}">Review: {esc(r['Review status'])}</span>
                        <div style="font-size:13px; color:#64748B; margin-top:4px;">{esc(er.get('Business profile', ''))} &bull; HQ: <strong>{esc(er.get('HQ', 'India'))}</strong> &bull; ICP Fit: <strong>{r['ICP fit']}/100</strong></div>
                        </div>
                        <div class="score-num">{r['Priority score']} <span style="font-size:12px; color:#64748B; font-weight:500;">/ 100</span></div>
                        </div>
                        <div class="metrics-strip">
                        <div class="m-item"><span>Employees ({esc(er.get('Confidence') or 'Est.')})</span><strong>{emp_txt}</strong></div>
                        <div class="m-item"><span>Est. Frontline</span><strong>{fl_txt}</strong></div>
                        <div class="m-item"><span>Signals (C / O / I)</span><strong>{esc(r['Signals (C/O/I)'])} ({vol_txt})</strong></div>
                        <div class="m-item"><span>Operating Footprint</span><strong>{esc(er.get('Operating locations') or er.get('HQ') or 'India')}</strong></div>
                        </div>
                        <div class="signal-callout"><strong>Top Hiring Signal ({esc(r['Latest signal'] or 'Recent')}):</strong> {esc(r['Top signal'])}</div>
                        {dm_block}
                        <div class="card-footer">
                        <div><strong>Outreach Angle:</strong> {esc(r['Outreach angle'])}</div>
                        <div>{"".join(links_html)}</div>
                        </div>
                        </div>
                        """)

                # ----- Interactive Account Deep-Dive, Score Attribution & Assisted Outreach Studio -----
                if shown:
                    st.markdown("---")
                    st.subheader("Account Deep-Dive: Score Attribution & Assisted Outreach Studio")
                    sel_col1, sel_col2 = st.columns([2.5, 1.5])
                    selected_company = sel_col1.selectbox(
                        "Select Account to Inspect Score Breakdown & Generate Outreach",
                        [r["Company"] for r in shown],
                        key="studio_company_select",
                    )
                    cur_rev = get_review_state(selected_company)
                    new_rev = sel_col2.selectbox(
                        "Human Verification Status",
                        ["Unreviewed", "Verified", "Disputed"],
                        index=["Unreviewed", "Verified", "Disputed"].index(cur_rev),
                        key=f"rev_select_{selected_company}",
                    )
                    if new_rev != cur_rev:
                        set_review_state(selected_company, new_rev)
                        st.rerun()

                    sel_row = next(r for r in shown if r["Company"] == selected_company)
                    sel_enr = next((r for r in S["enr"]["rows"] if r["Company"] == selected_company), {})
                    sel_sigs = [s for s in S["sig"]["signals"] if s["Company"] == selected_company]
                    sel_people = [p for p in S["dm"]["people"] if p["Company"] == selected_company]

                    with st.container(border=True):
                        s_left, s_right = st.columns([1.15, 1])
                        with s_left:
                            st.markdown(f"#### Score Attribution — {selected_company} ({sel_row['Priority score']}/100 • {sel_row['Tier']})")
                            st.caption("Deterministic breakdown across the 7 weighted criteria (25 / 20 / 15 / 15 / 10 / 10 / 5):")
                            st.dataframe(
                                pd.DataFrame(sc["breakdowns"][selected_company]),
                                hide_index=True,
                                use_container_width=True,
                            )
                        with s_right:
                            st.markdown("#### Source Verification & Evidence")
                            st.write(f"**Headcount Status:** {sel_enr.get('Size status', 'Unverified')} ({sel_enr.get('Headcount note') or 'No note'})")
                            if sel_enr.get("Employee source"):
                                st.markdown(f"- **Headcount Source:** [{sel_enr['Employee source']}]({sel_enr['Employee source']})")
                            if sel_enr.get("Website"):
                                st.markdown(f"- **Company Website:** [{sel_enr['Website']}]({sel_enr['Website']})")
                            if sel_sigs:
                                st.markdown("**Verified Signal Sources:**")
                                for idx_s, sg in enumerate(sel_sigs[:4], 1):
                                    st.markdown(f"{idx_s}. `[{sg['Evidence level']}]` {sg['Summary']} — [Source ({sg['Date']})]({sg['Source']})")
                            else:
                                st.caption("No external hiring news links captured for this account.")

                    # Assisted Outreach Draft Generator
                    with st.container(border=True):
                        st.markdown(f"#### Assisted Outreach Draft Generator — {selected_company}")
                        oc1, oc2 = st.columns([2.5, 1.5])
                        person_labels = [f"{p['Name']} ({p['Designation']} — {p['Persona']})" for p in sel_people]
                        if not person_labels:
                            person_labels = ["Default Buyer: Head of Talent Acquisition / HR"]
                        chosen_label = oc1.selectbox("Target Decision-Maker", person_labels, key=f"dm_pick_{selected_company}")
                        chosen_person = sel_people[person_labels.index(chosen_label)] if sel_people else None

                        draft_key = f"{selected_company}__{chosen_label}"
                        drafts_store = S.setdefault("outreach_drafts", {})
                        if draft_key not in drafts_store:
                            drafts_store[draft_key] = build_default_outreach(
                                selected_company, sel_enr, sel_sigs, chosen_person, d["industry"]
                            )

                        with oc2:
                            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
                            if st.button("Personalize with Gemini AI", type="primary", use_container_width=True, key=f"ai_outreach_{draft_key}"):
                                with st.spinner("Generating tailored multi-channel outreach pack..."):
                                    try:
                                        drafts_store[draft_key] = generate_ai_outreach(
                                            selected_company, sel_enr, sel_sigs, chosen_person, d["industry"]
                                        )
                                        save_run()
                                    except Exception as ex:
                                        st.error(f"AI personalization failed: {ex}")

                        cur_draft = drafts_store[draft_key]
                        st.caption(f"Draft Mode: **{cur_draft.get('mode', 'Signal-Grounded Template')}** • All fields below are editable before copying or exporting.")

                        d_tab1, d_tab2, d_tab3, d_tab4 = st.tabs([
                            "Cold Email",
                            "LinkedIn Connection Note",
                            "LinkedIn Message",
                            "Discovery Call Talking Points",
                        ])
                        with d_tab1:
                            subj = st.text_input("Email Subject", value=cur_draft["email_subject"], key=f"subj_{draft_key}")
                            body = st.text_area("Email Body", value=cur_draft["email_body"], height=210, key=f"body_{draft_key}")
                            cur_draft["email_subject"] = subj
                            cur_draft["email_body"] = body
                        with d_tab2:
                            lnote = st.text_area("LinkedIn Connection Request Note (Max 300 chars)", value=cur_draft["linkedin_note"], height=100, key=f"lnote_{draft_key}")
                            cur_draft["linkedin_note"] = lnote
                            st.caption(f"Character count: **{len(lnote)} / 300**")
                        with d_tab3:
                            lmsg = st.text_area("LinkedIn InMail / Follow-Up Message", value=cur_draft["linkedin_message"], height=180, key=f"lmsg_{draft_key}")
                            cur_draft["linkedin_message"] = lmsg
                        with d_tab4:
                            cpts = st.text_area("Discovery Call Talking Points", value=cur_draft["call_points"], height=180, key=f"cpts_{draft_key}")
                            cur_draft["call_points"] = cpts

                        full_pack_txt = (
                            f"HUNAR SELECT OUTREACH PACK — {selected_company}\n"
                            f"Recipient: {chosen_label}\n"
                            f"{'='*60}\n\n"
                            f"1. COLD EMAIL\nSubject: {cur_draft['email_subject']}\n\n{cur_draft['email_body']}\n\n"
                            f"{'='*60}\n\n"
                            f"2. LINKEDIN CONNECTION NOTE ({len(cur_draft['linkedin_note'])} chars)\n{cur_draft['linkedin_note']}\n\n"
                            f"{'='*60}\n\n"
                            f"3. LINKEDIN MESSAGE\n{cur_draft['linkedin_message']}\n\n"
                            f"{'='*60}\n\n"
                            f"4. DISCOVERY CALL TALKING POINTS\n{cur_draft['call_points']}\n"
                        )
                        st.download_button(
                            "Download Full Outreach Pack (.txt)",
                            full_pack_txt.encode("utf-8"),
                            file_name=f"outreach_{selected_company.lower().replace(' ', '_')}.txt",
                            mime="text/plain",
                        )

        # ----- TAB 2: Company Profiles -----
        with tabs[1]:
            enr = S.get("enr")
            if not enr:
                st.info("Company profiles appear after Stage 2.")
            else:
                rows = enr["rows"]
                df = pd.DataFrame(rows)
                df.insert(0, "#", range(1, len(df) + 1))
                df["Review status"] = [get_review_state(c) for c in df["Company"]]
                show_cols = ["#", "Company", "Business profile", "HQ", "Employees", "Confidence",
                             "Est. frontline (estimate)", "Operating locations", "Frontline roles (inferred)",
                             "Size status", "Review status", "Employee source", "Website"]
                st.dataframe(df[show_cols], hide_index=True, column_config=LINK_COLUMNS, use_container_width=True)
                st.download_button("Download Company Profiles (CSV)", df[show_cols].to_csv(index=False).encode("utf-8"),
                                   file_name="companies.csv", mime="text/csv")

        # ----- TAB 3: Hiring Signals -----
        with tabs[2]:
            sig = S.get("sig")
            if not sig:
                st.info("Hiring signals appear after Stage 3.")
            else:
                levels = st.multiselect("Evidence Level", list(EVIDENCE_POINTS), default=list(EVIDENCE_POINTS), key="level_filter")
                shown_sigs = [x for x in sig["signals"] if x["Evidence level"] in levels]
                if shown_sigs:
                    sdf = pd.DataFrame(shown_sigs)
                    if "Freshness" not in sdf.columns:
                        sdf["Freshness"] = [compute_freshness(d_str, True) for d_str in sdf["Date"]]
                    cols = ["Company", "Signal type", "Evidence level", "Date", "Freshness", "Urgency", "Summary",
                            "Roles", "Locations", "Est. volume", "Why it is an opportunity",
                            "How Hunar Select can help", "Source"]
                    st.dataframe(sdf[cols], hide_index=True, column_config=LINK_COLUMNS, use_container_width=True)
                    st.download_button("Download Hiring Signals (CSV)", sdf[cols].to_csv(index=False).encode("utf-8"),
                                       file_name="signals.csv", mime="text/csv")

        # ----- TAB 4: Decision-Makers -----
        with tabs[3]:
            dm = S.get("dm")
            if not dm:
                st.info("Decision-makers appear after Stage 4.")
            else:
                if dm["people"]:
                    pdf = pd.DataFrame(dm["people"])[["Company", "Name", "Designation", "Persona", "LinkedIn", "Email", "Phone", "Relevance"]]
                    st.dataframe(pdf, hide_index=True, column_config=LINK_COLUMNS, use_container_width=True)
                    st.download_button("Download Decision-Makers (CSV)", pdf.to_csv(index=False).encode("utf-8"),
                                       file_name="decision_makers.csv", mime="text/csv")
                else:
                    st.info("No LinkedIn decision-makers were extracted for this run.")
