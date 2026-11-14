import json
import re
import time

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Hunar Select Recruitment Intelligence Engine", layout="wide")

SIZE_BANDS = ["1,000 to 5,000", "5,000 to 20,000", "20,000+"]
INDUSTRIES = ["Quick Commerce", "Retail", "Logistics", "BFSI", "Healthcare"]
INDUSTRY_TERMS = {
    "Quick Commerce": "quick commerce",
    "Retail": "retail chains",
    "Logistics": "logistics and supply chain",
    "BFSI": "banks NBFC and insurance",
    "Healthcare": "hospital chains and healthcare",
}


# ---------------- Shared helpers ----------------
def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return None


SKIP_WORDS = ["preview", "exp", "image", "tts", "live", "audio", "thinking", "robotics", "computer"]


def gemini_generate(prompt, json_mode=False, timeout=40, max_models=6):
    """Ask Gemini. Tries several Flash models, retries when busy, moves on after a timeout.
    Returns (text, model_used). Raises RuntimeError listing every attempt if all fail."""
    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not found in Secrets.")
    headers = {"x-goog-api-key": key}

    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                     headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Could not list models: {r.status_code} {r.text[:200]}")
    names = [m["name"] for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]

    def usable(n):
        return "flash" in n and not any(w in n for w in SKIP_WORDS) and "latest" not in n

    full = sorted([n for n in names if usable(n) and "lite" not in n], reverse=True)
    lite = sorted([n for n in names if usable(n) and "lite" in n], reverse=True)
    order = full + lite
    for alias in ("models/gemini-flash-latest", "models/gemini-flash-lite-latest"):
        if alias in names and alias not in order:
            order.append(alias)
    if not order:
        raise RuntimeError("No Gemini Flash model available for this key.")

    last_good = st.session_state.get("gemini_model")
    if last_good in order:
        order.remove(last_good)
        order.insert(0, last_good)

    gen_config = {"temperature": 0.2, "maxOutputTokens": 8192}
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
                st.session_state["gemini_model"] = model
                return text.strip(), model
            attempts.append(f"{model}: {resp.status_code}")
            if resp.status_code in (401, 403):
                raise RuntimeError(f"Key rejected. {resp.text[:200]}")
            if resp.status_code in (429, 500, 503):
                time.sleep(2 * (attempt + 1))
                continue
            break
    raise RuntimeError("All Gemini models were busy or failed. Tried: " + "; ".join(attempts))


@st.cache_data(ttl=86400, show_spinner=False)
def serper_search(query, num=10):
    """Google search through Serper (India results). Cached for 24 hours to save free credits."""
    key = get_secret("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY not found in Secrets.")
    r = requests.post("https://google.serper.dev/search",
                      headers={"X-API-KEY": key, "Content-Type": "application/json"},
                      json={"q": query, "gl": "in", "num": num}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Serper failed: {r.status_code} {r.text[:200]}")
    return r.json().get("organic", [])


# ---------------- Stage 3: company discovery ----------------
def build_queries(industry, size):
    term = INDUSTRY_TERMS[industry]
    return [
        f"top {term} companies in India",
        f"largest {term} companies in India by number of employees",
        f"{term} companies India {size} employees",
        f"list of major {term} companies India frontline workforce",
        f"biggest employers {term} India stores warehouses branches",
    ]


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


def process_companies(data, valid_urls, size, cap=25):
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
    excluded = [r for r in rows if r["Size band (est.)"] not in (size, "Unknown")]
    matched.sort(key=lambda r: 0 if r["Source"] else 1)  # sourced companies first
    kept = (matched + unknown)[:cap]
    return kept, excluded


def discover_companies(industry, size, results):
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']} | {r['snippet']} | {r['link']}")
    evidence = "\n".join(lines)[:24000]
    prompt = f"""You are a research assistant building a prospect list for an Indian frontline hiring and recruitment-outsourcing company.

TASK: List companies that operate in India in this industry: {industry}, and that have roughly {size} employees (global or India headcount).

EVIDENCE: numbered web search results below. Prefer companies named in them. You may add well-known companies from your own knowledge only if you are confident they belong in this industry and size band; for those, set source_url to an empty string.

RULES:
- Return up to 35 distinct companies. No duplicates.
- Use the name people use in the market. If it is a brand of a larger parent, mention the parent in business_profile.
- size_band must be exactly one of: "1,000 to 5,000", "5,000 to 20,000", "20,000+", "Unknown".
- If a headcount appears in the results, put that figure in size_evidence and use it to choose size_band. Otherwise give your best estimate and set size_evidence to "Estimate, to be verified". Use "Unknown" if you cannot tell.
- source_url must be copied exactly from one result link above that mentions the company, or be an empty string.
- website: only if shown in the results or you are certain; otherwise an empty string.
- business_profile: maximum 25 words. business_model: maximum 15 words. headquarters: city.
- Exclude companies with no India presence, government bodies, and companies clearly outside the size range.

OUTPUT: a JSON array only. Each item has exactly these keys: company, business_profile, business_model, headquarters, website, size_band, size_evidence, source_url.

SEARCH RESULTS:
{evidence}
"""
    text, model = gemini_generate(prompt, json_mode=True, timeout=60, max_models=4)
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
    return {"industry": industry, "size": size, "companies": kept, "excluded": excluded,
            "results": results, "queries": queries, "model": model}


# ---------------- Page ----------------
st.title("Hunar Select: Recruitment Intelligence Engine")
st.caption("Pick an industry and company size. The system finds target companies, hiring signals and decision-makers.")

col1, col2, col3 = st.columns(3)
industry = col1.selectbox("Target industry", INDUSTRIES)
size = col2.selectbox("Company size (employees)", SIZE_BANDS)
col3.text_input("Geography", value="India", disabled=True)

if st.button("Run research", type="primary"):
    with st.status("Researching...", expanded=True) as status:
        try:
            out = run_discovery(industry, size, st.write)
            st.session_state["discovery"] = out
            status.update(label=f"Done. Found {len(out['companies'])} companies.", state="complete")
        except Exception as e:
            status.update(label="Research failed", state="error")
            st.error(str(e))

disc = st.session_state.get("discovery")
if disc:
    if (disc["industry"], disc["size"]) != (industry, size):
        st.warning(f"The table below is for {disc['industry']}, {disc['size']}. "
                   "Click Run research to generate the new selection.")
    st.subheader(f"Prospect companies: {disc['industry']}, {disc['size']} employees, India")
    companies = disc["companies"]
    if not companies:
        st.warning("No companies matched. Try another size band or run again.")
    else:
        n_unv = sum(1 for c in companies if c["Status"] == "Size unverified")
        n_src = sum(1 for c in companies if c["Source"])
        msg = f"{len(companies)} companies listed. {n_src} have a source link."
        if n_unv:
            msg += f" {n_unv} have an unverified size (Stage 4 will check headcount)."
        if len(companies) < 25:
            msg += f" Fewer than 25 qualified, so this is everything found."
        st.caption(msg)
        df = pd.DataFrame(companies)
        df.insert(0, "#", range(1, len(df) + 1))
        st.dataframe(
            df,
            hide_index=True,
            column_config={
                "Website": st.column_config.LinkColumn("Website"),
                "Source": st.column_config.LinkColumn("Source"),
            },
        )
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="companies.csv", mime="text/csv")

    with st.expander("Companies left out (size band did not match)"):
        if disc["excluded"]:
            st.dataframe(pd.DataFrame(disc["excluded"])[["Company", "Size band (est.)", "Size evidence"]],
                         hide_index=True)
        else:
            st.write("None.")
    with st.expander("Sources used for this research"):
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
