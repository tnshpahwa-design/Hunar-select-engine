import json
import re
import time

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


def normalise_url(u):
    u = (u or "").strip()
    if not u:
        return ""
    if not u.startswith("http"):
        u = "https://" + u
    return u


def process_companies(data, valid_urls, size, cap=45):
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


def discover_companies(industry, size, results):
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title'][:120]} | {r['snippet'][:200]} | {r['link']}")
    evidence = "\n".join(lines)[:18000]
    primary, alts = INDUSTRY_SEGMENTS[industry]
    segments = "; ".join([primary] + alts)
    prompt = f"""You are a research assistant building a prospect list for an Indian frontline hiring and recruitment-outsourcing company.

TASK: List companies that operate in India in this industry: {industry}. Segments in scope: {segments}. Prefer companies with roughly {size} employees (global or India headcount), but also include companies of nearby sizes when you are unsure, because headcount will be checked in a later step.

EVIDENCE: numbered web search results below. Prefer companies named in them. You may add well-known companies from your own knowledge only if you are confident they belong in this industry and size band; for those, set source_url to an empty string.

RULES:
- Return up to 45 distinct companies. No duplicates.
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
    text, model = gemini_generate(prompt, json_mode=True, timeout=100, max_models=4, prefer_lite=True)
    data = parse_json_list(text)
    valid_urls = {r["link"] for r in results}
    kept, excluded = process_companies(data, valid_urls, size)
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
- locations: array of up to 12 cities, states or regions where the evidence shows the company operates
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

    enrichments, model = {}, ""
    for start in range(0, len(items), 5):
        batch = items[start:start + 5]
        log(f"AI is reading the evidence: companies {start + 1} to {start + len(batch)}...")
        try:
            data, model = enrich_batch(industry, batch)
            by_name = {norm(d.get("company", "")): d for d in data if isinstance(d, dict)}
            for j, b in enumerate(batch):
                d = by_name.get(norm(b["name"]))
                if d is None and len(data) == len(batch) and isinstance(data[j], dict):
                    d = data[j]
                if d:
                    enrichments[b["name"]] = clean_enrichment(d, b["urls"])
        except Exception as ex:
            warnings.append(f"AI step failed for companies {start + 1} to {start + len(batch)}: {ex}")
        time.sleep(1)

    final, excluded = assemble_final(cands, enrichments, industry, size)
    return {"run_id": disc["run_id"], "industry": industry, "size": size, "rows": final,
            "excluded": excluded, "evidence": evidence_log, "warnings": warnings, "model": model}


# ---------------- Page ----------------
LINK_COLUMNS = {
    "Website": st.column_config.LinkColumn("Website"),
    "Source": st.column_config.LinkColumn("Source"),
    "Employee source": st.column_config.LinkColumn("Employee source"),
    "Stage 3 source": st.column_config.LinkColumn("Stage 3 source"),
    "Employees": st.column_config.NumberColumn("Employees", format="%d"),
    "Est. frontline (estimate)": st.column_config.NumberColumn("Est. frontline (estimate)", format="%d"),
}

st.title("Hunar Select: Recruitment Intelligence Engine")
st.caption("Pick an industry and company size. The system finds target companies, hiring signals and decision-makers.")

col1, col2, col3 = st.columns(3)
industry = col1.selectbox("Target industry", INDUSTRIES)
size = col2.selectbox("Company size (employees)", SIZE_BANDS)
col3.text_input("Geography", value="India", disabled=True)

if st.button("Run research", type="primary"):
    st.session_state.pop("enriched", None)
    with st.status("Researching...", expanded=True) as status:
        try:
            out = run_discovery(industry, size, st.write)
            st.session_state["discovery"] = out
            status.update(label=f"Done. Found {len(out['companies'])} candidate companies.", state="complete")
        except Exception as e:
            status.update(label="Research failed", state="error")
            st.error(str(e))

disc = st.session_state.get("discovery")
enr = st.session_state.get("enriched")
if enr and disc and enr["run_id"] != disc["run_id"]:
    enr = None

if disc:
    if (disc["industry"], disc["size"]) != (industry, size):
        st.warning(f"The tables below are for {disc['industry']}, {disc['size']}. "
                   "Click Run research to generate the new selection.")

    if not enr:
        st.info("Step 1 is done. Click the button below to add company details (headcount, locations, "
                "frontline estimate). This takes about 3 to 6 minutes.")
        if st.button("Step 2: Add company details", type="primary"):
            with st.status("Adding company details...", expanded=True) as status2:
                try:
                    enr = run_enrichment(disc, st.write)
                    st.session_state["enriched"] = enr
                    status2.update(label="Details added.", state="complete")
                    st.rerun()
                except Exception as e:
                    status2.update(label="Adding details failed", state="error")
                    st.error(str(e))

    if enr:
        st.subheader(f"Prospect companies: {enr['industry']}, {enr['size']} employees, India")
        rows = enr["rows"]
        n_ver = sum(1 for r in rows if r["Size status"] == "Verified")
        n_unv = sum(1 for r in rows if r["Size status"] == "Size unverified")
        msg = f"{len(rows)} companies listed. {n_ver} have a sourced headcount inside this size band."
        if n_unv:
            msg += f" {n_unv} have no sourced headcount yet (size unverified)."
        if len(rows) < 25:
            msg += " Fewer than 25 qualified, so this is everything found."
        st.caption(msg + " Blank means Not found. Frontline numbers are estimates.")
        for w in enr["warnings"]:
            st.warning(w)
        df = pd.DataFrame(rows)
        df.insert(0, "#", range(1, len(df) + 1))
        st.dataframe(df, hide_index=True, column_config=LINK_COLUMNS)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="prospects.csv", mime="text/csv")
        with st.expander("Companies left out after headcount check"):
            if enr["excluded"]:
                st.dataframe(pd.DataFrame(enr["excluded"])[["Company", "Employees", "Size status", "Employee source"]],
                             hide_index=True, column_config=LINK_COLUMNS)
            else:
                st.write("None.")
        with st.expander("Evidence used for each company"):
            ev = [{"Company": k, "Search": v["query"], "Links": "  ".join(v["links"][:4])}
                  for k, v in enr["evidence"].items()]
            st.dataframe(pd.DataFrame(ev), hide_index=True)
            st.caption(f"AI model used: {enr['model']}")

    companies = disc["companies"]
    with st.expander("Step 1 results: candidate companies", expanded=not enr):
        if not companies:
            st.warning("No companies matched. Try another size band or run again.")
        else:
            n_unv = sum(1 for c in companies if c["Status"] == "Size unverified")
            n_src = sum(1 for c in companies if c["Source"])
            msg = f"{len(companies)} candidates. {n_src} have a source link."
            if n_unv:
                msg += f" {n_unv} have an unverified size. Others may be in a neighbouring size band; Step 2 checks real headcount."
            st.caption(msg)
            cdf = pd.DataFrame(companies)
            cdf.insert(0, "#", range(1, len(cdf) + 1))
            st.dataframe(cdf, hide_index=True, column_config=LINK_COLUMNS)
        if disc["excluded"]:
            st.caption("Left out in Step 1 (estimated size band did not match): "
                       + ", ".join(e["Company"] for e in disc["excluded"]))
    with st.expander("Step 1 sources: searches and links"):
        st.write("Searches run: " + " | ".join(disc["queries"]))
        st.caption(f"AI model used: {disc['model']}")
        st.dataframe(pd.DataFrame(disc["results"])[["title", "link", "query"]], hide_index=True,
                     column_config={"link": st.column_config.LinkColumn("link")})

# ---------------- Connection test ----------------
st.divider()
with st.expander("Connection test"):
    st.caption("Checks that every service the engine needs is reachable. Keys are never shown.")

    def test_gemini():
        text, model = gemini_generate("Reply with the single word OK")
        return True, f"Working. Model used: {model}. Reply: {text}"

    def test_serper():
        res = serper_search("quick commerce companies India", 3)
        return True, f"Working. Got {len(res)} results."

    def test_news():
        url = "https://news.google.com/rss/search?q=quick+commerce+hiring&hl=en-IN&gl=IN&ceid=IN:en"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return False, f"Failed: {r.status_code}"
        return True, f"Working. Got {r.text.count('<item>')} news items."

    def test_jina():
        r = requests.get("https://r.jina.ai/https://example.com", timeout=45)
        if r.status_code != 200:
            return False, f"Failed: {r.status_code}"
        return True, f"Working. Read {len(r.text)} characters from a test page."

    if st.button("Show available AI models"):
        try:
            st.write(list_gemini_models())
        except Exception as e:
            st.error(str(e))

    if st.button("Run connection test"):
        tests = [("Gemini (AI)", test_gemini), ("Serper (Google search)", test_serper),
                 ("Google News feed", test_news), ("Jina Reader (web pages)", test_jina)]
        for label, fn in tests:
            with st.spinner(f"Testing {label}..."):
                try:
                    ok, msg = fn()
                except Exception as e:
                    ok, msg = False, f"Error: {e}"
            if ok:
                st.success(f"{label}: {msg}")
            else:
                st.error(f"{label}: {msg}")
