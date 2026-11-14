import streamlit as st
import requests

st.set_page_config(page_title="Hunar Select Recruitment Intelligence Engine", layout="wide")

st.title("Hunar Select: Recruitment Intelligence Engine")
st.caption("Pick an industry and company size. The system finds target companies, hiring signals and decision-makers.")

col1, col2, col3 = st.columns(3)
industry = col1.selectbox("Target industry", ["Quick Commerce", "Retail", "Logistics", "BFSI", "Healthcare"])
size = col2.selectbox("Company size (employees)", ["1,000 to 5,000", "5,000 to 20,000", "20,000+"])
col3.text_input("Geography", value="India", disabled=True)

if st.button("Run research", type="primary"):
    st.info(f"Research for {industry}, {size}, India will run here from Stage 3.")

# ---------------- Stage 2: Connection test ----------------
st.divider()
st.subheader("Connection test")
st.caption("Checks that every service the engine needs is reachable. Keys are never shown.")


def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return None


def test_gemini():
    key = get_secret("GEMINI_API_KEY")
    if not key:
        return False, "GEMINI_API_KEY not found in Secrets."
    headers = {"x-goog-api-key": key}
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                     headers=headers, timeout=30)
    if r.status_code != 200:
        return False, f"Could not list models: {r.status_code} {r.text[:200]}"
    names = [m["name"] for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    if "models/gemini-flash-latest" in names:
        model = "models/gemini-flash-latest"
    else:
        flash = sorted([n for n in names if "flash" in n], reverse=True)
        if not flash:
            return False, "No Gemini Flash model available for this key."
        model = flash[0]
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"
    body = {"contents": [{"parts": [{"text": "Reply with the single word OK"}]}]}
    r2 = requests.post(url, headers=headers, json=body, timeout=60)
    if r2.status_code != 200:
        return False, f"Model {model} failed: {r2.status_code} {r2.text[:200]}"
    text = r2.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    return True, f"Working. Model used: {model}. Reply: {text}"


def test_serper():
    key = get_secret("SERPER_API_KEY")
    if not key:
        return False, "SERPER_API_KEY not found in Secrets."
    r = requests.post("https://google.serper.dev/search",
                      headers={"X-API-KEY": key, "Content-Type": "application/json"},
                      json={"q": "quick commerce companies India", "gl": "in", "num": 3},
                      timeout=30)
    if r.status_code != 200:
        return False, f"Failed: {r.status_code} {r.text[:200]}"
    results = r.json().get("organic", [])
    return True, f"Working. Got {len(results)} results. First: {results[0]['title'] if results else 'none'}"


def test_news():
    url = "https://news.google.com/rss/search?q=quick+commerce+hiring&hl=en-IN&gl=IN&ceid=IN:en"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return False, f"Failed: {r.status_code}"
    count = r.text.count("<item>")
    return True, f"Working. Got {count} news items."


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
