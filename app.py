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
    """Ask Gemini. Tries several Flash models, skips quickly on quota errors, retries when busy.
    Returns (text, model_used). Raises RuntimeError listing every attempt if all fail."""
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

    gen_config = {"temperature": 0.2, "maxOutputTokens": 12000}
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
            break  # 429 (quota) and other errors: go to the next model straight away
    raise RuntimeError("All Gemini models were busy or failed. Tried: " + "; ".join(attempts))


@st.cache_data(ttl=86400, show_spinner=False)
def serper_raw(query, num=10):
    """Google search through Serper (India results). Cached for 24 hours to save free credits."""
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


# ---------------- Stage 3: company discovery ----------------
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
            raise RuntimeError("The AI reply was not valid JSON. Try Run research again.")
        data = json.loads(t[s:e + 1])
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        data = lists[0] if lists else []
    return data


def with_retry(fn, *args, tries=2, wait=15):
    """Run an AI step again after a short pause if every model was busy or the reply was unusable."""
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
    """Clean the AI output, drop duplicates, apply the size filter. Returns (kept, excluded)."""
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
            src = ""  # a source must be a real search result, never invented
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
    matched.sort(key=lambda r: 0 if r["Source"] else 1)  # sourced companies first
    # Neighbouring bands are kept because the Step 1 size is only a guess; Step 2 checks real headcount
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
    if len(kept) < 40:  # second pass to widen the pool
        try:
            already = [c["Company"] for c in kept] + [e["Company"] for e in excluded]
            text2, _ = gemini_generate(discovery_prompt(industry, size, evidence, already), json_mode=True,
                                       timeout=100, max_models=4, prefer_lite=True)
            data2 = parse_json_list(text2)
            kept, excluded = process_companies(data + data2, valid_urls, size)
        except Exception:
            pass  # keep the first-pass result
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


# ---------------- Stage 4: company details ----------------
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
    """Turn one Serper response into short evidence text plus the list of links it contains."""
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
        n, src = None, ""  # no valid source link, so the figure is not accepted
    conf = str(item.get("employees_confidence", "")).strip().title() if n else ""
    if conf not in ("High", "Medium", "Low"):
        conf = "Low" if n else ""
    if n and any(d in src.lower() for d in LOW_QUALITY_SOURCES):
        conf = "Low"  # profile counts and review sites are not reliable headcounts
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

    # Second look: companies with no sourced headcount get one more search with different wording
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
                        merged[k] = e2.get(k)
                enrichments[name] = merged

    final, excluded = assemble_final(cands, enrichments, industry, size)
    return {"run_id": disc["run_id"], "industry": industry, "size": size, "rows": final,
            "excluded": excluded, "evidence": evidence_log, "warnings": warnings, "model": model}


# ---------------- Stage 5: hiring signals ----------------
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
    """Google News RSS (free, no key). Returns recent items with title, link, date and source."""
    url = f"[https://news.google.com/rss/search?q=](https://news.google.com/rss/search?q=){quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
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
    """Two news searches per company: one on expansion, one on hiring. Company name is an exact phrase."""
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
    """Returns (age_days, recency_weight, urgency) using the rules in the Product Definition."""
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


def build_signal_evidence(news, web, today):
    """Number every evidence item (E1, E2...) so the AI cites an ID instead of copying long links."""
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
    """Code-side guardrail: hiring levels need hiring language in the source item."""
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
    volume = str(item.get("volume", "")).strip() or "Unknown"
    if quality == "Low":
        volume = "Unknown"  # job-board and social counts are not reliable
    points = EVIDENCE_POINTS[level] * weight * (0.5 if quality == "Low" else 1.0)
    return {
        "Signal type": f"{code} {SIGNAL_TYPES[code]}",
        "Evidence level": level,
        "Date": date_str or "Date unverified",
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
    for s in signals:  # drop duplicate developments
        k = (s["Company"], s["Source"], norm(s["Summary"])[:60])
        if k not in seen:
            seen.add(k)
            unique.append(s)
        else:
            diag["duplicates_removed"] += 1
    unique.sort(key=lambda s: s["Points"], reverse=True)
    return {"run_id": enr["run_id"], "signals": unique, "summary": summarise_signals(companies, unique),
            "evidence": evidence_log, "warnings": warnings, "model": model, "diagnostics": diag}


# ---------------- Stage 6: decision-makers and contacts ----------------
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
    """Maps a job title to (persona label, priority). Priority 1 is best. None means not relevant."""
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
        f'site:[linkedin.com/in](https://linkedin.com/in) ("Head of Talent Acquisition" OR "Head of Recruitment" OR "Talent Acquisition Head" '
        f'OR "Chief People Officer" OR CHRO OR "Chief Human Resources Officer") "{name}"',
        f'site:[linkedin.com/in](https://linkedin.com/in) ("HR Operations" OR "Regional HR" OR "HR Head" OR "Head of HR" OR "Operations Head" '
        f'OR "Business Head" OR "Regional Head") "{name}"',
    ]


def contact_query(name):
    return f'"{name}" careers recruitment HR contact email phone'


def extract_company_contacts(name, results):
    """Public company-level recruitment emails and phone numbers found in search results, each with its source."""
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
        if k >= 16:
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
            "Contact status": "LinkedIn only", "Contact source": "",
            "Relevance": str(item.get("relevance", "")).strip(),
            "Source note": "Public LinkedIn search result. Verify the current role before outreach."}


def hunter_people(domain):
    key = get_secret("HUNTER_API_KEY")
    r = requests.get("[https://api.hunter.io/v2/domain-search](https://api.hunter.io/v2/domain-search)",
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
    r = requests.post("[https://api.apollo.io/api/v1/people/match](https://api.apollo.io/api/v1/people/match)",
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
        log(f"Finding decision-makers: {name} ({i} of {len(companies)})")
        results = []
        for q in people_queries(name):
            diag["searches"] += 1
            try:
                results += serper_raw(q, 10).get("organic", [])
            except Exception as ex:
                warnings.append(f"People search failed for {name}: {ex}")
        diag["searches"] += 1
        try:
            cres = serper_raw(contact_query(name), 8).get("organic", [])
        except Exception as ex:
            cres = []
            warnings.append(f"Contact search failed for {name}: {ex}")
        emails, phones = extract_company_contacts(name, cres)
        contacts[name] = {"emails": emails, "phones": phones}
        ev_map, text = build_people_evidence(results)
        diag["linkedin_results"] += len(ev_map)
        ts = top_signal.get(name)
        context = (f"top hiring signal: {ts['Summary']} (locations: {ts['Locations'] or 'not stated'}). "
                   if ts else "no hiring signal found. ") + f"Operating locations: {rows[name].get('Operating locations') or 'not stated'}."
        evidence[name] = {"queries": people_queries(name) + [contact_query(name)],
                          "links": [v["link"] for v in ev_map.values()]}
        if ev_map:
            items.append({"name": name, "evidence": text, "ev_map": ev_map, "context": context})

    people_by, ta_counts, model = {}, {}, ""
    for start in range(0, len(items), 3):
        batch = items[start:start + 3]
        log(f"AI is reading LinkedIn results: companies {start + 1} to {start + len(batch)} of {len(items)}...")
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

    # Optional contact lookups for the strongest companies only (free-plan limits)
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


# ---------------- Stage 7: scoring ----------------
WEIGHTS = [("Hiring intensity", 0.20), ("Open positions / hiring volume", 0.15), ("Frontline workforce", 0.15),
           ("Geographic expansion", 0.10), ("Urgency of hiring", 0.15), ("Outsourcing potential", 0.15),
           ("Decision-maker accessibility", 0.10)]
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

    subs = [(s1, n1), (s2, n2), (s3, n3), (s4, n4), (s5, n5), (s6, n6), (s7, n7)]
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
        text += " No decision-maker found yet: search LinkedIn for the Head of Talent Acquisition."
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
        top = max(sigs, key=lambda s: s["Points"]) if sigs else None
        best = sorted(people, key=lambda p: (p["Priority"], 0 if (p["Email"] or p["Phone"]) else 1))[:1]
        contact_text = "Not found"
        if best:
            b = best[0]
            contact_text = f"{b['Name']} ({b['Designation']})" + (f" | {b['Email']}" if b["Email"] else "")
        elif contacts.get("emails"):
            contact_text = f"Company recruitment email: {contacts['emails'][0]['value']}"
        urgency = "High" if res["urgency_score"] >= 70 else "Medium" if res["urgency_score"] >= 40 else "Low" \
            if res["urgency_score"] > 0 else "None"
        out.append({
            "Company": name, "Tier": tier_for(res["priority"]), "Priority score": res["priority"], "ICP fit": res["icp"],
            "Employees": r.get("Employees"), "Est. frontline (estimate)": r.get("Est. frontline (estimate)"),
            "Size status": r.get("Size status"),
            "Signals (C/O/I)": f"{counts['Confirmed']}/{counts['Observed']}/{counts['Inferred']}",
            "Hiring volume (sourced)": res["vol"] or None, "Urgency": urgency,
            "Latest signal": max(dated) if dated else "",
            "Top signal": top["Summary"] if top else "No signals found",
            "Decision-makers": len(people), "Best contact": contact_text,
            "Outreach angle": outreach_angle(r, sigs, people)})
    out.sort(key=lambda r: (r["Priority score"], r["ICP fit"]), reverse=True)
    for i, r in enumerate(out, 1):
        r["Rank"] = i
    return {"run_id": enr["run_id"], "rows": out, "breakdowns": breakdowns}


# ---------------- Stage 8: one-click pipeline and saved runs ----------------
RUN_KEYS = ("disc", "enr", "sig", "dm", "score")
SAVE_DIR = "saved_runs"


def save_run():
    S = st.session_state
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        d = S["disc"]
        fname = f"{d['industry']}__{d['size']}".replace(" ", "_").replace(",", "").replace("+", "plus") + ".json"
        with open(os.path.join(SAVE_DIR, fname), "w") as f:
            json.dump({k: S[k] for k in RUN_KEYS if k in S}, f, default=str)
    except Exception:
        pass  # saving is a convenience; never stop the run for it


def load_run_data(data):
    S = st.session_state
    for k in RUN_KEYS:
        S.pop(k, None)
    for k in RUN_KEYS:
        if k in data:
            S[k] = data[k]


def run_all(industry, size, log, resume=False):
    S = st.session_state
    if resume and "disc" in S:
        industry, size = S["disc"]["industry"], S["disc"]["size"]
    else:
        for k in RUN_KEYS:
            S.pop(k, None)
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
        log("STEP 4 of 5: decision-makers and contacts")
        S["dm"] = run_decision_makers(S["enr"], S["sig"], log)
        save_run()
    if "score" not in S:
        log("STEP 5 of 5: scoring and ranking")
        S["score"] = run_scoring(S["enr"], S["sig"], S["dm"])
        save_run()


# ---------------- Page (Modern B2B SaaS UI Layer) ----------------
def esc(val):
    return _html.escape(str(val or ""))


def inject_saas_theme():
    st.markdown("""
    <style>
    @import url('[https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap)');

    :root {
        --bg-canvas: #F8FAFC;
        --bg-card: #FFFFFF;
        --border-subtle: #E2E8F0;
        --border-hover: #CBD5E1;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
        --accent-indigo: #4F46E5;
        --accent-indigo-soft: #EEF2FF;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background-color: var(--bg-canvas);
    }

    h1, h2, h3, .saas-title {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: -0.025em;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1280px !important;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .saas-hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 60%, #312E81 100%);
        border-radius: 18px;
        padding: 28px 32px;
        color: #FFFFFF;
        margin-bottom: 20px;
        box-shadow: 0 12px 30px -8px rgba(15, 23, 42, 0.25);
        animation: fadeInUp 0.4s ease-out;
        position: relative;
        overflow: hidden;
    }
    .saas-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #C7D2FE;
        margin-bottom: 12px;
    }
    .saas-hero h1 {
        color: #FFFFFF !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        margin: 0 0 6px 0 !important;
        line-height: 1.2 !important;
    }
    .saas-hero p {
        color: #CBD5E1;
        font-size: 14.5px;
        margin: 0;
        max-width: 720px;
        line-height: 1.5;
    }

    .bento-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 16px 0 22px 0;
        animation: fadeInUp 0.45s ease-out;
    }
    .bento-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        transition: all 0.2s ease;
    }
    .bento-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -6px rgba(15, 23, 42, 0.08);
    }
    .bento-label {
        font-size: 12px;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .bento-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 26px;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.1;
    }
    .bento-sub {
        font-size: 12px;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    .prospect-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
        animation: fadeInUp 0.35s ease-out both;
    }
    .prospect-card:hover {
        border-color: #A5B4FC;
        transform: translateY(-2px);
        box-shadow: 0 12px 24px -8px rgba(79, 70, 229, 0.12);
    }
    .card-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 14px;
    }
    .rank-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        background: #F1F5F9;
        color: #334155;
        padding: 3px 9px;
        border-radius: 8px;
        margin-right: 8px;
    }
    .company-name {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
        text-decoration: none;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11.5px;
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

    .ev-Confirmed { background: #ECFDF5; color: #047857; }
    .ev-Observed { background: #EFF6FF; color: #1D4ED8; }
    .ev-Inferred { background: #F8FAFC; color: #475569; border: 1px solid #E2E8F0; }

    .score-box {
        text-align: right;
        min-width: 150px;
    }
    .score-num {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 20px;
        font-weight: 800;
        color: var(--accent-indigo);
    }
    .score-bar-bg {
        width: 140px;
        height: 6px;
        background: #E2E8F0;
        border-radius: 999px;
        overflow: hidden;
        margin-left: auto;
        margin-top: 4px;
    }
    .score-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        border-radius: 999px;
    }

    .metrics-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        background: #F8FAFC;
        border: 1px solid #F1F5F9;
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }
    .m-item span {
        display: block;
        font-size: 11px;
        color: var(--text-muted);
        font-weight: 500;
    }
    .m-item strong {
        font-size: 13.5px;
        color: var(--text-primary);
        font-weight: 600;
    }

    .signal-callout {
        background: #FAFAFF;
        border-left: 3px solid var(--accent-indigo);
        padding: 10px 14px;
        border-radius: 0 10px 10px 0;
        font-size: 13px;
        color: var(--text-secondary);
        margin-bottom: 10px;
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
        color: var(--text-secondary);
    }
    .action-link {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 5px 12px;
        border-radius: 8px;
        background: var(--accent-indigo-soft);
        color: var(--accent-indigo) !important;
        font-weight: 600;
        font-size: 12px;
        text-decoration: none !important;
        transition: background 0.15s ease;
    }
    .action-link:hover {
        background: #E0E7FF;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.25rem !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #FFFFFF;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 13.5px;
    }
    </style>
    """, unsafe_allow_html=True)


inject_saas_theme()

LINK_COLUMNS = {
    "Website": st.column_config.LinkColumn("Website"),
    "Source": st.column_config.LinkColumn("Source"),
    "Employee source": st.column_config.LinkColumn("Employee source"),
    "Stage 3 source": st.column_config.LinkColumn("Stage 3 source"),
    "LinkedIn": st.column_config.LinkColumn("LinkedIn"),
    "Employees": st.column_config.NumberColumn("Employees", format="%d"),
    "Est. frontline (estimate)": st.column_config.NumberColumn("Est. frontline (estimate)", format="%d"),
    "Hiring volume (sourced)": st.column_config.NumberColumn("Hiring volume (sourced)", format="%d"),
}
S = st.session_state

# Hero Header
st.markdown("""
<div class="saas-hero">
    <div class="saas-hero-badge">Hunar.AI &bull; GTM Intelligence Engine</div>
    <h1>Hunar Select Prospecting &amp; Hiring Radar</h1>
    <p>Select a target sector and workforce scale to dynamically discover high-intent Indian enterprises, verify frontline headcount, capture live hiring triggers, and map key Talent Acquisition &amp; Operations buyers.</p>
</div>
""", unsafe_allow_html=True)

# Control Bar Card
with st.container(border=True):
    col1, col2, col3, col4 = st.columns([2.2, 2.2, 1.4, 2.2])
    industry = col1.selectbox("Target Industry", INDUSTRIES)
    size = col2.selectbox("Company Size (Employees)", SIZE_BANDS)
    col3.text_input("Geography", value="India", disabled=True)
    with col4:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        run_clicked = st.button("Launch Live Research", type="primary", use_container_width=True)

    if "disc" in S and "score" not in S:
        if st.button("Resume Unfinished Run", use_container_width=True):
            resume_clicked = True
        else:
            resume_clicked = False
    else:
        resume_clicked = False

if run_clicked or resume_clicked:
    with st.status("Executing 5-Stage Intelligence Pipeline (~10–15 mins for full live discovery)...", expanded=True) as status:
        try:
            run_all(industry, size, st.write, resume=resume_clicked)
            status.update(label="Research pipeline complete.", state="complete")
            st.rerun()
        except Exception as e:
            status.update(label="Pipeline paused. Completed stages are checkpointed — click Resume.", state="error")
            st.error(str(e))

with st.expander("Workspace Snapshots: Instant Load / Export Runs"):
    sc1, sc2 = st.columns([2, 2])
    saved = sorted(os.listdir(SAVE_DIR)) if os.path.isdir(SAVE_DIR) else []
    with sc1:
        if saved:
            pick = st.selectbox("Saved workspace runs", saved)
            if st.button("Load Snapshot", use_container_width=True):
                with open(os.path.join(SAVE_DIR, pick)) as f:
                    load_run_data(json.load(f))
                st.rerun()
        else:
            st.caption("No server snapshots yet. Completed runs save automatically.")
    with sc2:
        up = st.file_uploader("Upload run snapshot (.json)", type="json")
        if up is not None and S.get("_loaded") != (up.name, up.size):
            load_run_data(json.load(up))
            S["_loaded"] = (up.name, up.size)
            st.rerun()
        if "disc" in S:
            st.download_button(
                "Export Current Run (.json)",
                json.dumps({k: S[k] for k in RUN_KEYS if k in S}, default=str).encode("utf-8"),
                file_name=f"hunar_{S['disc']['industry'].lower().replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True,
            )

if "disc" in S:
    d = S["disc"]
    if (d["industry"], d["size"]) != (industry, size):
        st.info(f"Currently viewing **{d['industry']} ({d['size']} employees)**. Click **Launch Live Research** to run for {industry} ({size}).")

    tabs = st.tabs([
        "Priority Feed & Accounts",
        "Company Directory",
        "Hiring Signals",
        "Decision-Makers",
        "Audit & Methodology",
    ])

    # ----- TAB 1: Ranked SaaS Feed -----
    with tabs[0]:
        sc = S.get("score")
        if not sc:
            st.info("Priority rankings appear once all 5 stages complete.")
        else:
            rows = sc["rows"]
            p1_count = sum(1 for r in rows if r["Tier"] == "P1")
            total_fl = sum(r["Est. frontline (estimate)"] or 0 for r in rows)
            total_sigs = len(S.get("sig", {}).get("signals", []))
            total_dms = len(S.get("dm", {}).get("people", []))

            st.markdown(f"""
            <div class="bento-grid">
                <div class="bento-card">
                    <div class="bento-label">Qualified Accounts</div>
                    <div class="bento-value">{len(rows)}</div>
                    <div class="bento-sub">{p1_count} Priority P1 accounts ready now</div>
                </div>
                <div class="bento-card">
                    <div class="bento-label">Est. Frontline Pool</div>
                    <div class="bento-value">{total_fl:,}</div>
                    <div class="bento-sub">Addressable frontline workforce</div>
                </div>
                <div class="bento-card">
                    <div class="bento-label">Verified Hiring Signals</div>
                    <div class="bento-value">{total_sigs}</div>
                    <div class="bento-sub">News &amp; web triggers (365d window)</div>
                </div>
                <div class="bento-card">
                    <div class="bento-label">Mapped Decision-Makers</div>
                    <div class="bento-value">{total_dms}</div>
                    <div class="bento-sub">TA, CHRO, HR Ops &amp; Ops leaders</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Filter Toolbar
            with st.container(border=True):
                f1, f2, f3, f4 = st.columns([2, 2, 1.8, 1.5])
                tiers = f1.multiselect("Action Tier", ["P1", "P2", "P3", "P4"], default=["P1", "P2", "P3", "P4"])
                min_score = f2.slider("Min Priority Score", 0, 100, 0)
                only_dm = f3.checkbox("Has Decision-Maker")
                view_mode = f4.radio("Layout", ["SaaS Cards", "Table"], horizontal=True)

            shown = [r for r in rows if r["Tier"] in tiers and r["Priority score"] >= min_score
                     and (not only_dm or r["Decision-makers"] > 0)]

            col_cap, col_dl = st.columns([4, 1])
            col_cap.caption(f"Showing **{len(shown)} of {len(rows)}** ranked accounts for **{d['industry']}** ({d['size']})")
            if shown:
                cols = ["Rank", "Company", "Tier", "Priority score", "ICP fit", "Employees", "Est. frontline (estimate)",
                        "Signals (C/O/I)", "Hiring volume (sourced)", "Urgency", "Latest signal", "Top signal",
                        "Decision-makers", "Best contact", "Outreach angle", "Size status"]
                ddf = pd.DataFrame(shown)
                col_dl.download_button("Export CSV", ddf[cols].to_csv(index=False).encode("utf-8"),
                                       file_name="ranked_companies.csv", mime="text/csv", use_container_width=True)

            if not shown:
                st.warning("No companies match your active filters.")
            elif view_mode == "Table":
                st.dataframe(ddf[cols], hide_index=True, column_config=LINK_COLUMNS, use_container_width=True)
            else:
                enr_map = {r["Company"]: r for r in S.get("enr", {}).get("rows", [])}
                for r in shown:
                    cname = r["Company"]
                    er = enr_map.get(cname, {})
                    emp_txt = f"{r['Employees']:,}" if r["Employees"] else "Unverified"
                    fl_txt = f"{r['Est. frontline (estimate)']:,}" if r["Est. frontline (estimate)"] else "—"
                    vol_txt = f"{r['Hiring volume (sourced)']:,} roles" if r["Hiring volume (sourced)"] else "Inferred scale"
                    site = er.get("Website", "")
                    site_btn = f'<a class="action-link" href="{esc(site)}" target="_blank">Website &#8599;</a>' if site else ""

                    st.markdown(f"""
                    <div class="prospect-card">
                        <div class="card-top">
                            <div>
                                <span class="rank-pill">#{r['Rank']}</span>
                                <span class="company-name">{esc(cname)}</span>
                                <span class="badge tier-{esc(r['Tier'])}">{esc(r['Tier'])}</span>
                                <span class="badge urg-{esc(r['Urgency'])}">Urgency: {esc(r['Urgency'])}</span>
                                <span class="badge ev-Inferred">ICP Fit: {r['ICP fit']}/100</span>
                                <div style="font-size:13px; color:#64748B; margin-top:4px;">
                                    {esc(er.get('Business profile', ''))} &bull; HQ: <strong>{esc(er.get('HQ', 'India'))}</strong>
                                </div>
                            </div>
                            <div class="score-box">
                                <div style="font-size:11px; color:#64748B; font-weight:600; text-transform:uppercase;">Priority Score</div>
                                <div class="score-num">{r['Priority score']} <span style="font-size:13px; color:#94A3B8; font-weight:500;">/ 100</span></div>
                                <div class="score-bar-bg"><div class="score-bar-fill" style="width:{min(100, int(r['Priority score']))}%;"></div></div>
                            </div>
                        </div>
                        <div class="metrics-strip">
                            <div class="m-item"><span>Total Workforce</span><strong>{emp_txt}</strong></div>
                            <div class="m-item"><span>Est. Frontline</span><strong>{fl_txt}</strong></div>
                            <div class="m-item"><span>Signals (C / O / I)</span><strong>{esc(r['Signals (C/O/I)'])} ({vol_txt})</strong></div>
                            <div class="m-item"><span>Primary Contact</span><strong>{esc(r['Best contact'])}</strong></div>
                        </div>
                        <div class="signal-callout">
                            <strong>Latest Trigger ({esc(r['Latest signal'] or 'Recent')}):</strong> {esc(r['Top signal'])}
                        </div>
                        <div class="card-footer">
                            <div><strong>Recommended Pitch:</strong> {esc(r['Outreach angle'])}</div>
                            <div>{site_btn}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Deep-Dive Inspector Drawer
            if shown:
                st.markdown("---")
                st.subheader("Account Deep-Dive Inspector")
                choice = st.selectbox("Select an account to inspect score math, signals, and buyers",
                                      [r["Company"] for r in shown])
                row = next(r for r in shown if r["Company"] == choice)
                enr_row = next((r for r in S["enr"]["rows"] if r["Company"] == choice), {})

                with st.container(border=True):
                    d1, d2 = st.columns([1.2, 1])
                    with d1:
                        st.markdown(f"#### {choice} ({row['Tier']} &bull; Score {row['Priority score']})")
                        st.write(f"**Business Model:** {enr_row.get('Business model') or 'N/A'}")
                        st.write(f"**Operating Footprint:** {enr_row.get('Operating locations') or 'Not stated'}")
                        st.write(f"**Target Roles:** {enr_row.get('Frontline roles (inferred)') or 'N/A'}")
                        st.info(f"**Sales Hook:** {row['Outreach angle']}")
                        if enr_row.get("Employee source"):
                            st.markdown(f"[Verify Headcount Source &#8599;]({enr_row['Employee source']})")
                    with d2:
                        st.markdown("**Score Attribution Breakdown**")
                        st.dataframe(pd.DataFrame(sc["breakdowns"][choice])[["Criterion", "Sub-score (0-100)", "Weight", "Contribution"]],
                                     hide_index=True, use_container_width=True)

                    mine = [s for s in S["sig"]["signals"] if s["Company"] == choice]
                    if mine:
                        st.markdown("**Captured Hiring Signals**")
                        st.dataframe(pd.DataFrame(mine)[["Signal type", "Evidence level", "Date", "Summary", "Est. volume", "How Hunar Select can help", "Source"]],
                                     hide_index=True, column_config=LINK_COLUMNS, use_container_width=True)

                    ppl = [p for p in S["dm"]["people"] if p["Company"] == choice]
                    if ppl:
                        st.markdown("**Mapped Decision-Makers**")
                        st.dataframe(pd.DataFrame(ppl)[["Name", "Designation", "Persona", "LinkedIn", "Email", "Phone", "Relevance"]],
                                     hide_index=True, column_config=LINK_COLUMNS, use_container_width=True)

    # ----- TAB 2: Companies -----
    with tabs[1]:
        enr = S.get("enr")
        if not enr:
            st.info("Company details appear after Step 2.")
        else:
            rows = enr["rows"]
            n_ver = sum(1 for r in rows if r["Size status"] == "Verified")
            st.caption(f"{len(rows)} target enterprises discovered ({n_ver} headcount-verified within {d['size']}).")
            df = pd.DataFrame(rows)
            df.insert(0, "#", range(1, len(df) + 1))
            st.dataframe(df, hide_index=True, column_config=LINK_COLUMNS, use_container_width=True)
            st.download_button("Download Company Directory (CSV)", df.to_csv(index=False).encode("utf-8"),
                               file_name="companies.csv", mime="text/csv")

    # ----- TAB 3: Hiring Signals -----
    with tabs[2]:
        sig = S.get("sig")
        if not sig:
            st.info("Hiring signals appear after Step 3.")
        else:
            sc1, sc2 = st.columns([2, 1])
            levels = sc1.multiselect("Filter by Evidence Level", list(EVIDENCE_POINTS), default=list(EVIDENCE_POINTS), key="level_filter")
            sig_view = sc2.radio("Signal View", ["Signal Cards", "Table"], horizontal=True)
            shown_sigs = [x for x in sig["signals"] if x["Evidence level"] in levels]

            if shown_sigs:
                sdf = pd.DataFrame(shown_sigs)
                st.download_button("Export Hiring Signals (CSV)", sdf.to_csv(index=False).encode("utf-8"),
                                   file_name="signals.csv", mime="text/csv")
                if sig_view == "Table":
                    first = ["Company", "Signal type", "Evidence level", "Date", "Urgency", "Summary", "Roles", "Locations",
                             "Est. volume", "Why it is an opportunity", "How Hunar Select can help", "Source"]
                    st.dataframe(sdf[first], hide_index=True, column_config=LINK_COLUMNS, use_container_width=True)
                else:
                    for s in shown_sigs[:40]:
                        st.markdown(f"""
                        <div class="prospect-card" style="padding:16px 20px;">
                            <div class="card-top" style="margin-bottom:8px;">
                                <div>
                                    <span class="company-name" style="font-size:16px;">{esc(s['Company'])}</span>
                                    <span class="badge ev-{esc(s['Evidence level'])}">{esc(s['Evidence level'])}</span>
                                    <span class="badge urg-{esc(s['Urgency'])}">{esc(s['Signal type'])}</span>
                                </div>
                                <div style="font-size:12px; color:#64748B;">
                                    {esc(s['Date'])} &bull; <strong>{esc(s['Source name'])}</strong>
                                </div>
                            </div>
                            <div style="font-size:14px; font-weight:600; color:#0F172A; margin-bottom:6px;">{esc(s['Summary'])}</div>
                            <div style="font-size:13px; color:#475569; margin-bottom:8px;">
                                <strong>Hunar Select Angle:</strong> {esc(s['How Hunar Select can help'])}
                            </div>
                            <div class="card-footer" style="padding-top:8px;">
                                <span><strong>Roles:</strong> {esc(s['Roles'] or 'Frontline staff')} &bull; <strong>Locations:</strong> {esc(s['Locations'] or 'Pan-India')} &bull; <strong>Est. Volume:</strong> {esc(s['Est. volume'])}</span>
                                <a class="action-link" href="{esc(s['Source'])}" target="_blank">Read Source &#8599;</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # ----- TAB 4: Decision-Makers -----
    with tabs[3]:
        dm = S.get("dm")
        if not dm:
            st.info("Decision-makers appear after Step 4.")
        else:
            n_with = len({p["Company"] for p in dm["people"]})
            n_mail = sum(1 for p in dm["people"] if p["Email"])
            st.caption(f"{len(dm['people'])} verified buyer profiles across {n_with} companies ({n_mail} enriched with direct work emails).")
            if dm["people"]:
                pdf = pd.DataFrame(dm["people"])[["Company", "Name", "Designation", "Persona", "LinkedIn", "Email", "Phone",
                                                   "Contact status", "Contact source", "Relevance"]]
                st.download_button("Export Decision-Makers (CSV)", pdf.to_csv(index=False).encode("utf-8"),
                                   file_name="decision_makers.csv", mime="text/csv")

                cols_dm = st.columns(2)
                for idx, p in enumerate(dm["people"]):
                    with cols_dm[idx % 2]:
                        mail_html = f'<span class="badge ev-Confirmed">{esc(p["Email"])}</span>' if p["Email"] else '<span class="badge ev-Inferred">LinkedIn Outreach</span>'
                        li_btn = f'<a class="action-link" href="{esc(p["LinkedIn"])}" target="_blank">LinkedIn Profile &#8599;</a>' if p["LinkedIn"] else ""
                        st.markdown(f"""
                        <div class="prospect-card" style="padding:16px 18px;">
                            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                                <div>
                                    <div style="font-size:11.5px; font-weight:700; color:#4F46E5; text-transform:uppercase;">{esc(p['Company'])}</div>
                                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:16px; font-weight:700; color:#0F172A;">{esc(p['Name'])}</div>
                                    <div style="font-size:13px; color:#475569;">{esc(p['Designation'])}</div>
                                </div>
                                <div>{mail_html}</div>
                            </div>
                            <div style="font-size:12.5px; color:#64748B; margin:10px 0;">{esc(p['Relevance'])}</div>
                            <div class="card-footer" style="padding-top:8px;">
                                <span style="font-size:11.5px;">{esc(p['Persona'])}</span>
                                {li_btn}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # ----- TAB 5: Sources & Audit -----
    with tabs[4]:
        st.markdown("**Deterministic Scoring Formula:** Priority Score = Hiring Intensity (20%) + Open Volume (15%) + Frontline Workforce (15%) + Urgency (15%) + Outsourcing Potential (15%) + Geographic Expansion (10%) + Decision-Maker Accessibility (10%).")
        enr, sig, dm = S.get("enr"), S.get("sig"), S.get("dm")
        with st.expander("Stage 1: Discovery Queries & Raw Search Links"):
            st.write("Queries executed: " + " | ".join(d["queries"]))
            st.dataframe(pd.DataFrame(d["results"])[["title", "link", "query"]], hide_index=True,
                         column_config={"link": st.column_config.LinkColumn("link")}, use_container_width=True)
        if enr:
            with st.expander("Stage 2: Headcount Verification Logs"):
                ev = [{"Company": k, "Search": v["query"], "Links": "  ".join(v["links"][:4])} for k, v in enr["evidence"].items()]
                st.dataframe(pd.DataFrame(ev), hide_index=True, use_container_width=True)
        if sig:
            with st.expander("Stage 3: Raw Signal Headlines & Guardrail Diagnostics"):
                st.json(sig["diagnostics"])
                for name, ev in sig["evidence"].items():
                    st.markdown(f"**{name}**")
                    st.text("\n".join(ev["lines"]) if ev["lines"] else "No evidence found.")
        if dm:
            with st.expander("Stage 4: Decision-Maker Search Audit & Diagnostics"):
                st.json(dm["diagnostics"])

# ---------------- API Health Check ----------------
with st.expander("System Diagnostics & API Health Check"):
    def test_gemini():
        text, model = gemini_generate("Reply with the single word OK")
        return True, f"Connected ({model})"

    def test_serper():
        res = serper_search("quick commerce companies India", 3)
        return True, f"Connected ({len(res)} live results)"

    def test_news():
        url = "[https://news.google.com/rss/search?q=quick+commerce+hiring&hl=en-IN&gl=IN&ceid=IN:en](https://news.google.com/rss/search?q=quick+commerce+hiring&hl=en-IN&gl=IN&ceid=IN:en)"
        r = requests.get(url, timeout=30)
        return (r.status_code == 200), f"Connected ({r.text.count('<item>')} RSS items)"

    if st.button("Verify Live API Connections"):
        for label, fn in [("Gemini LLM", test_gemini), ("Serper Search", test_serper), ("Google News RSS", test_news)]:
            try:
                ok, msg = fn()
                st.success(f"{label}: {msg}") if ok else st.error(f"{label}: {msg}")
            except Exception as e:
                st.error(f"{label}: {e}")

# --- end of app.py ---
