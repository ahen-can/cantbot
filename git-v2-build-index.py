#trying to embed content and not just page titles
import os
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
import openai
import numpy as np
import time
import pickle
import faiss
from openai import RateLimitError, OpenAIError

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

# ── CONFIG ─────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")
SITEMAP_URL    = "https://canarabank.com/sitemap.xml"
FAISS_FILE     = "index_v2.faiss"
META_FILE      = "index_meta_v2.pkl"
SNIPPET_LEN    = 500   # adjust snippet length to balance relevance vs cost

# ── UTILITIES ──────────────────────────────────────────
def parse_sitemap(url: str) -> list[str]:
    resp = requests.get(url, timeout=10); resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [loc.text for loc in root.findall(".//ns:loc", ns)]

def fetch_snippet(url: str, timeout: int = 5) -> str:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        return text[:SNIPPET_LEN]
    except Exception:
        return ""

def get_embedding(text: str, model="text-embedding-ada-002", retries=3) -> np.ndarray:
    for i in range(retries):
        try:
            resp = openai.embeddings.create(model=model, input=[text])
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except RateLimitError:
            wait = 2**i; print(f"Rate limited, retry {i+1} in {wait}s…"); time.sleep(wait)
        except OpenAIError as e:
            print("Embedding error:", e); raise
    raise RuntimeError("Embedding failed after retries")

# ── BUILD & SAVE INDEX ─────────────────────────────────
def build_index():
    print("📥 Parsing sitemap…")
    urls = parse_sitemap(SITEMAP_URL)

    print(f"🔨 Building index for {len(urls)} pages…")
    entries, vectors = [], []
    for url in urls:
        snippet = fetch_snippet(url) or url
        emb     = get_embedding(snippet)
        entries.append({"url": url, "snippet": snippet})
        vectors.append(emb)
        print(f" • Indexed: {url}")

    arr = np.stack(vectors)
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    faiss.write_index(index, FAISS_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(entries, f)

    print("✅ Index and metadata saved.")

if __name__ == "__main__":
    if os.path.exists(FAISS_FILE) and os.path.exists(META_FILE):
        print("✅ Existing v2 index found—skipping rebuild.")
    else:
        build_index()
