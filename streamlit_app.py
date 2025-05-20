# streamlit_app.py

import os
import pickle
import faiss
import numpy as np
import openai
import requests
from bs4 import BeautifulSoup
import time
from openai import RateLimitError, OpenAIError
import streamlit as st

# ── CONFIG ─────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("Missing OPENAI_API_KEY. Please set it in your environment or Streamlit secrets.")
    st.stop()

FAISS_FILE = "index_v2.faiss"
META_FILE  = "index_meta_v2.pkl"

# ── LOAD & CACHE RESOURCES ─────────────────────────────
@st.cache_resource
def load_index():
    idx = faiss.read_index(FAISS_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    return idx, meta

faiss_index, index_meta = load_index()

# ── EMBEDDING & RETRIEVAL ────────────────────────────
def get_embedding(text: str, model="text-embedding-ada-002", retries=3):
    for i in range(retries):
        try:
            resp = openai.embeddings.create(model=model, input=[text])
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except RateLimitError:
            time.sleep(2**i)
        except OpenAIError as e:
            st.error(f"Embedding error: {e}")
            return None
    st.error("Embedding failed after retries.")
    return None

def retrieve_top_k(query: str, k=3):
    q_emb = get_embedding(query)
    if q_emb is None:
        return []
    D, I = faiss_index.search(q_emb.reshape(1,-1), k)
    return [
        {
            "url":     index_meta[i]["url"],
            "snippet": index_meta[i]["snippet"],
            "score":   1 - d
        }
        for d, i in zip(D[0], I[0])
    ]

def ask_gpt_with_context(query: str, contexts):
    system = "You are a helpful assistant that uses provided snippets to answer."
    combined = "\n\n".join(f"URL: {c['url']}\nSnippet: {c['snippet']}" for c in contexts)
    prompt = f"{combined}\n\nQuestion: {query}\nAnswer concisely."
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

# ── STREAMLIT PAGE LAYOUT ────────────────────────────
st.set_page_config(page_title="CanaraBot", page_icon="🤖")
st.title("🤖 Canara Bank Chatbot")

query = st.text_input("Ask me anything about Canara Bank:")
if st.button("Submit") and query:
    with st.spinner("Thinking…"):
        tops = retrieve_top_k(query, k=3)
        if not tops:
            st.warning("Couldn't retrieve any context.")
        else:
            st.markdown("**Top candidates:**")
            for i, c in enumerate(tops, 1):
                st.markdown(f"{i}. [{c['score']:.2f}] [{c['url']}]({c['url']})")
            answer = ask_gpt_with_context(query, tops)
            st.markdown("---")
            st.subheader("💡 Answer")
            st.write(answer)
