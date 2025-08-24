from __future__ import annotations

import os
import re
import time
import pickle
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


# =========================
# Retrieval / models
# =========================
import faiss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from sentence_transformers import SentenceTransformer

# Optional notebook rendering
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup


# =========================
# Configuration
# =========================
FAISS_DIR_DEFAULT = "./nestle_qna.index"
BM25_PKL_DEFAULT = "./bm25_index.pkl"
FINETUNED_DIR_DEFAULT = "./finetuned_nestle_gpt2"


# =========================
# Guardrails
# =========================
@dataclass(frozen=True)
class GuardrailHit:
    """Structured result for a blocked input."""
    category: str
    reason: str
    message: str


class NestleInputGuard:
    """Input-side guardrails tailored for Nestlé finance Q&A."""
    def __init__(self) -> None:
        self._rules = {
            "illegal_violent": {
                "patterns": [r"\bbomb\b", r"\battack\b", r"\bkill\b", r"\bshoot\b",
                             r"\bterror\b", r"\bmurder\b", r"\bfraud\b", r"\blaunder\s+money\b"],
                "reason": "Illegal/violent content.",
                "message": "I can’t assist with harmful, illegal, or violent requests.",
            },
            "pii": {
                "patterns": [r"\bsocial\s*security\b", r"\bcredit\s*card\b", r"\bpassword\b",
                             r"\bprivate\s*key\b", r"\bhome\s*address\b", r"\bphone\s*number\b"],
                "reason": "Solicitation of personal/sensitive information.",
                "message": "I can’t assist with requests for personal or sensitive data.",
            },
            "insider": {
                "patterns": [r"\binsider\b", r"\bnon[-\s]?public\b", r"\bconfidential\b",
                             r"\bleak\b", r"\bunreleased\b", r"\bearnings\s*(leak|before\s+release)\b"],
                "reason": "Requests for material non-public information.",
                "message": "I can’t help with material non-public or confidential information. I can use public figures.",
            },
            "forward_looking": {
                "patterns": [r"\bforecast\b", r"\bprojection(s)?\b", r"\bguidance\b",
                             r"\bnext\s+(quarter|year|q[1-4]|fy)\b",
                             r"\b(price|stock|share)\s*(target|prediction|will go)\b",
                             r"\bshould\s+(i|we)\s+(buy|sell|hold)\b",
                             r"\bexpected\s+(sales|revenue|eps|margin)\b"],
                "reason": "Forward-looking/investment advice not allowed.",
                "message": "I can’t provide forward-looking guidance or investment advice. I can summarize published figures.",
            },
            "off_topic": {
                "patterns": [r"\b(movie|recipe|weather|sports|celebrity)\b",
                             r"\bhow\s+to\s+cook\b", r"\btravel\s+tips\b"],
                "reason": "Out of Nestlé-finance scope.",
                "message": "This assistant focuses on Nestlé financials (sales, margins, cash flow, segments, etc.).",
            },
        }
        self._compiled = {
            k: [re.compile(p, re.IGNORECASE) for p in v["patterns"]]
            for k, v in self._rules.items()
        }

    def assess(self, query: str) -> Tuple[bool, Optional[GuardrailHit]]:
        """Return (allowed, hit) where hit is populated if blocked."""
        for key, rx_list in self._compiled.items():
            for rx in rx_list:
                if rx.search(query):
                    data = self._rules[key]
                    return False, GuardrailHit(category=key, reason=data["reason"], message=data["message"])
        return True, None


class OutputGuard:
    """Light post-processing: remove template echoes; detect 'not specified' noise."""
    _uncertain = re.compile(r"\b(maybe|perhaps|not\s+sure|i\s*think)\b", re.IGNORECASE)
    _echo = re.compile(r"(?i)^context:.*?\n\n", flags=re.DOTALL)

    def clean(self, text: str) -> str:
        """Trim artifacts and normalize spacing."""
        text = self._echo.sub("", text)
        text = re.sub(r"(?:^|\n)\s*(Q:.*?\n)?\s*A:\s*", "", text)
        text = re.sub(r"\s+\n", "\n", text).strip()
        return text

    def flag_uncertain(self, text: str) -> bool:
        return bool(self._uncertain.search(text))

    def squash_not_specified(self, text: str) -> str:
        """If model prepends 'Not specified.' but provides facts later, keep facts only."""
        if text.lower().startswith("not specified"):
            tail = text.split(".", 1)
            if len(tail) > 1 and tail[1].strip():
                return tail[1].strip()
        return text


# =========================
# Utilities
# =========================
def preprocess_query(q: str) -> List[str]:
    """Lowercase, strip punctuation, and split."""
    q = q.lower().translate(str.maketrans("", "", string.punctuation))
    return q.split()


# =========================
# Cached loaders
# =========================
@st.cache_resource(show_spinner=False)
def load_faiss(faiss_index_path: str, docstore_path: str):
    if not os.path.exists(faiss_index_path) or not os.path.exists(docstore_path):
        raise RuntimeError("FAISS index or docstore not found.")
    index = faiss.read_index(faiss_index_path)
    with open(docstore_path, "rb") as f:
        docstore = pickle.load(f)
        
    # Debug info about the loaded docstore
    if isinstance(docstore, list) and len(docstore) > 0:
        st.info(f"Loaded {len(docstore)} documents")
        sample = docstore[0]
        st.info(f"Document format: {sample.keys() if isinstance(sample, dict) else type(sample)}")
    else:
        st.warning("Docstore is empty or not a list")
        
    return index, docstore

@st.cache_resource(show_spinner=False)
def load_bm25(bm25_pkl: str):
    if not os.path.exists(bm25_pkl):
        return None
    with open(bm25_pkl, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    tok = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    model.eval()
    return tok, model


@st.cache_resource(show_spinner=False)
def load_finetuned(finetuned_dir: str):
    tok = AutoTokenizer.from_pretrained(finetuned_dir)
    model = AutoModelForCausalLM.from_pretrained(finetuned_dir)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


# =========================
# Retrieval (Dense via FAISS)
# =========================
def retrieve_dense_faiss(index, docstore, query: str, encoder, k: int = 6) -> List[Dict]:
    """Embed query, search FAISS, return docs with scores."""
    q_emb = encoder.encode([query])
    q_emb = np.array(q_emb).astype("float32")

    D, I = index.search(q_emb, k)
    docs = []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0 or idx >= len(docstore):
            continue
        # Handle different document formats
        doc = docstore[idx]
        if isinstance(doc, dict):
            text = doc.get("text") or doc.get("document") or doc.get("Question", "") + " " + doc.get("Answer", "")
        else:
            text = str(doc)
            
        docs.append({
            "text": text,
            "metadata": doc if isinstance(doc, dict) else {},
            "dense_score": float(1.0 / (1.0 + dist))
        })
    return docs


def rerank_cross_encoder(query: str, docs: List[Dict], ce_tok, ce_model, k: int = 3) -> List[Dict]:
    if not docs:
        return []
    pairs = ([query] * len(docs), [d["text"] for d in docs])
    enc = ce_tok(*pairs, padding=True, truncation=True, return_tensors="pt")

    with st.spinner("Re-ranking…"):
        scores = ce_model(**enc).logits.squeeze()

    for d, s in zip(docs, scores):
        d["rerank_score"] = float(s)

    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:k]



# =========================
# Generation (Fine-tuned)
# =========================
def generate_finetuned_answer(query: str, tok, model, max_new_tokens: int = 120):
    """Return (answer, avg_confidence, latency)."""
    prompt = f"Question: {query}\nAnswer:"
    inputs = tok(prompt, return_tensors="pt")
    start = time.time()
    
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )
    
    latency = time.time() - start
    trans = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    conf = float(np.exp(trans[0]).mean())
    full = tok.decode(out.sequences[0], skip_special_tokens=True)
    ans = full.split("Answer:", 1)[-1].strip()
    return ans, conf, latency


def generate_rag_answer(query: str, collection, ce_tok, ce_model, max_new_tokens: int = 120) -> Tuple[str, float, float, List[str]]:
    """Return (answer, confidence, latency, contexts)."""
    # Retrieve + rerank
    t0 = time.time()
    embed_model = load_sentence_transformer()  # Get the same model used for index creation
    dense = retrieve_dense_faiss(faiss_index, docstore, query, embed_model, k=6)
    top_docs = rerank_cross_encoder(query, dense, ce_tok, ce_model, k=3)
    ctx = [d["text"] for d in top_docs]

    # Compose strict prompt to reduce repetition
    context_block = "\n".join(ctx)
    prompt = (
        "You are a precise Nestlé finance assistant.\n"
        "Use ONLY the context facts below. Answer in ONE concise sentence. "
        "If the context lacks the answer, say 'Not specified.'\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    # Load generator model
    try:
        gen_tok, gen_model = load_finetuned(FINETUNED_DIR_DEFAULT)
    except Exception:
        gen_tok = AutoTokenizer.from_pretrained("distilgpt2")
        gen_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        if gen_tok.pad_token_id is None:
            gen_tok.pad_token = gen_tok.eos_token
        gen_model.eval()

    # Tokenize and move to device
    inputs = gen_tok(prompt, return_tensors="pt", truncation=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate answer
    with st.spinner("Generating answer…"):
        with torch.no_grad():
            out = gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=gen_tok.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

    latency = time.time() - t0

    # Confidence
    trans = gen_model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    conf = float(np.exp(trans[0]).mean())

    # Extract answer
    full = gen_tok.decode(out.sequences[0], skip_special_tokens=True)
    ans = full.split("Answer:", 1)[-1].strip()

    return ans, conf, latency, ctx



# =========================
# UI Styling
# =========================
APP_CSS = """
<style>
/* App-wide tweaks */
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
.stMetric {background: #111; border-radius: 12px; padding: 10px;}
.answer-box {background: #0e1117; border: 1px solid #2b2f3a; border-radius: 12px; padding: 16px;}
.context-box {background: #0b0d13; border: 1px dashed #394055; border-radius: 10px; padding: 12px;}
.badge {display:inline-block; padding: 2px 10px; border-radius: 999px; background:#1f6feb; color:white; font-size:12px; margin-left:8px;}
.note {color:#9aa4b2; font-size:13px;}
footer {visibility: hidden;}
</style>
"""


# =========================
# Notebook rendering
# =========================
def render_notebook(ipynb_path: str):
    """Display an ipynb as HTML inside Streamlit."""
    if not os.path.exists(ipynb_path):
        st.warning("Notebook path not found.")
        return
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    html_exporter = HTMLExporter()
    (body, _resources) = html_exporter.from_notebook_node(nb)
    # Limit overly bright backgrounds
    soup = BeautifulSoup(body, "html.parser")
    st.components.v1.html(str(soup), height=800, scrolling=True)


# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="Nestlé Finance QA", page_icon="📊", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.title("📊 Nestlé Financial Q&A Assistant")
    st.caption("Hybrid RAG & Fine-tuned model • Input/Output guardrails • Notebook viewer")

    # Sidebar: configuration
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("Mode", ["RAG", "Fine-tuned"], index=0)
    chroma_dir = st.sidebar.text_input("Chroma store path", FAISS_DIR_DEFAULT)
    bm25_path = st.sidebar.text_input("BM25 index (optional)", BM25_PKL_DEFAULT)
    finetuned_dir = st.sidebar.text_input("Fine-tuned model dir", FINETUNED_DIR_DEFAULT)
    st.sidebar.divider()

    with st.sidebar.expander("Notebook Viewer"):
        notebook_options = {
            "Part I": "./notebooks/Conversation_AI_Financial_Statements_Assignment_part_I.ipynb",
            "Part II": "./notebooks/Conversation_AI_Financial_Statements_Assignment_part_II.ipynb",
            "Part III": "./notebooks/Conversation_AI_Financial_Statements_Assignment_part_III.ipynb"
        }
        selected_notebook = st.selectbox("Select Notebook", list(notebook_options.keys()))
        nb_path = notebook_options[selected_notebook]
        
        if st.button("Open Notebook"):
            st.session_state["show_nb"] = True
            st.session_state["current_nb"] = nb_path
        else:
            st.session_state["show_nb"] = st.session_state.get("show_nb", False)
            st.session_state["current_nb"] = st.session_state.get("current_nb", nb_path)

    # Tabs
    tab_qa, tab_nb = st.tabs(["Assistant", "Notebook"])

    # Preload assets
    guard_in = NestleInputGuard()
    guard_out = OutputGuard()

    # Load indexes/models only when needed
    if mode == "RAG":
        try:
            # Initialize global variables for components
            global faiss_index, docstore
            
            # Get the directory containing the files
            index_dir = os.path.dirname(os.path.abspath(chroma_dir))
            
            # Construct paths for both files
            faiss_path = os.path.join(index_dir, "nestle_qna.index")
            meta_path = os.path.join(index_dir, "metadata.pkl")
            
            # Print paths for debugging
            st.info(f"Looking for FAISS index at: {faiss_path}")
            st.info(f"Looking for metadata at: {meta_path}")
            
            # Load FAISS components
            faiss_index, docstore = load_faiss(faiss_path, meta_path)
            st.success("Successfully loaded FAISS index and metadata")
            
            # Load models
            ce_tok, ce_model = load_cross_encoder()
            embed_model = load_sentence_transformer()
            
        except Exception as e:
            st.error(f"Failed to initialize RAG components: {str(e)}")
            st.info("Please check that nestle_qna.index and metadata.pkl exist in the same directory")
            return

    if mode == "Fine-tuned":
        try:
            ft_tok, ft_model = load_finetuned(finetuned_dir)
        except Exception as e:
            st.error(f"Failed to load fine-tuned model at '{finetuned_dir}': {e}")
            return

    # Assistant tab
    with tab_qa:
        st.subheader("Ask a question")
        q = st.text_input("Your question", placeholder="e.g., What were Nestlé’s sales in 2023?")
        go = st.button("Run")

        if go and q.strip():
            allowed, hit = guard_in.assess(q)
            if not allowed and hit:
                st.warning(f"⛔ {hit.message}")
                st.caption(f"Reason: {hit.reason}")
            else:
                if mode == "RAG":
                    ans, conf, t_taken, ctx = generate_rag_answer(q, None, ce_tok, ce_model)
                    ans = guard_out.clean(guard_out.squash_not_specified(ans))
                    uncertain = guard_out.flag_uncertain(ans)

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Method", "RAG")
                    col2.metric("Confidence", f"{conf:.3f}")
                    col3.metric("Latency", f"{t_taken:.2f}s")

                    # Answer
                    st.markdown("#### Answer")
                    st.markdown(f"<div class='answer-box'>{ans}</div>", unsafe_allow_html=True)
                    if uncertain:
                        st.markdown("<div class='note'>⚠️ This answer may be uncertain. Try rephrasing or narrowing the question.</div>", unsafe_allow_html=True)

                    # Context panel
                    if ctx:
                        st.markdown("#### Context Used")
                        for i, c in enumerate(ctx, 1):
                            st.markdown(f"<div class='context-box'><b>Passage {i}</b><br>{c}</div>", unsafe_allow_html=True)

                else:
                    # Fine-tuned mode
                    ans, conf, t_taken = generate_finetuned_answer(q, ft_tok, ft_model)
                    ans = guard_out.clean(guard_out.squash_not_specified(ans))
                    uncertain = guard_out.flag_uncertain(ans)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Method", "Fine-tuned")
                    col2.metric("Confidence", f"{conf:.3f}")
                    col3.metric("Latency", f"{t_taken:.2f}s")

                    st.markdown("#### Answer")
                    st.markdown(f"<div class='answer-box'>{ans}</div>", unsafe_allow_html=True)
                    if uncertain:
                        st.markdown("<div class='note'>⚠️ This answer may be uncertain. Consider clarifying the question.</div>", unsafe_allow_html=True)

    # Notebook tab
    with tab_nb:
        st.subheader("Notebook Viewer")
        if st.session_state.get("show_nb"):
            current_nb = st.session_state.get("current_nb", nb_path)
            if os.path.exists(current_nb):
                render_notebook(current_nb)
            else:
                st.warning(f"Notebook not found at path: {current_nb}")
        else:
            st.info("Select a notebook and click **Open Notebook** in the sidebar to view it.")


if __name__ == "__main__":
    main()
