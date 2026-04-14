# app_streamlit_docgen.py
# Single-file Streamlit app for integrated documentation generation
# Requirements: streamlit, torch, numpy, pandas, sklearn, tqdm

import os, sys, pickle, math, time, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import streamlit as st
import ast
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# -------------- Configuration (edit paths if needed) --------------
# Use model_artifacts/ in repository (Git LFS) for Streamlit Cloud deployment
BASE_PATH = "model_artifacts"    # place tokenizers, pkls and model files here
DATA_SAMPLE_PKL = os.path.join(BASE_PATH, "tokenized_sample.pkl")
BPE_CODE_PKL = os.path.join(BASE_PATH, "bpe_code_tokenizer.pkl")
BPE_DOC_PKL  = os.path.join(BASE_PATH, "bpe_doc_tokenizer.pkl")
W2V_CODE_PKL = os.path.join(BASE_PATH, "word2vec_code.pkl")
W2V_DOC_PKL  = os.path.join(BASE_PATH, "word2vec_doc.pkl")
MODEL_STATE_PATH = os.path.join(BASE_PATH, "seq2seq_attention_state.pt")   # state_dict (preferred)
FULL_MODEL_PATH  = os.path.join(BASE_PATH, "seq2seq_attention_full.pt")    # optional full pickle (NOT used)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------- Utilities: load / safe helpers --------------------
def safe_load_pickle(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def load_bpe_info(path):
    # tries to return token_to_id, id_to_token, decode_fn, encode_fn (encode may be None)
    data = safe_load_pickle(path)
    if data is None: return None, None, None, None
    token_to_id = getattr(data, "token_to_id", None) or (data.get("token_to_id") if isinstance(data, dict) else None)
    id_to_token = getattr(data, "id_to_token", None) or (data.get("id_to_token") if isinstance(data, dict) else None)
    decode_fn = getattr(data, "decode", None) or (data.get("decode") if isinstance(data, dict) else None)
    encode_fn = getattr(data, "encode", None) or (data.get("encode") if isinstance(data, dict) else None)
    # normalize id_to_token keys
    if isinstance(id_to_token, dict):
        try:
            id_to_token = {int(k): v for k,v in id_to_token.items()}
        except Exception:
            pass
    # fallback decode
    if decode_fn is None and isinstance(id_to_token, dict):
        def simple_decode(ids):
            toks = [id_to_token.get(int(i), "<UNK>") for i in ids]
            s = ''.join([t for t in toks if t not in ['<PAD>']])
            s = s.replace('</w>', ' ')
            return ' '.join(s.split())
        decode_fn = simple_decode
    return (token_to_id or {}), (id_to_token or {}), decode_fn, encode_fn

# Fallback encoding: whitespace tokens -> ids (best-effort)
def fallback_encode(text, token_to_id, max_len=None, unk_token="<UNK>"):
    toks = text.strip().split()
    ids = [ token_to_id.get(t, token_to_id.get(unk_token, 0)) for t in toks ]
    if max_len:
        ids = ids[:max_len]
    return ids

# -------------- Model classes (must match training definitions) --------------
class Seq2SeqAttention(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, emb_dim, hid_dim, emb_enc=None, emb_dec=None, dropout=0.1, num_layers=1, PAD_ENC=0, PAD_DEC=0):
        super().__init__()
        self.enc_emb = nn.Embedding(enc_vocab, emb_dim, padding_idx=PAD_ENC)
        if emb_enc is not None:
            self.enc_emb.weight.data.copy_(torch.tensor(emb_enc, dtype=torch.float32))
        self.dec_emb = nn.Embedding(dec_vocab, emb_dim, padding_idx=PAD_DEC)
        if emb_dec is not None:
            self.dec_emb.weight.data.copy_(torch.tensor(emb_dec, dtype=torch.float32))
        self.encoder = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0, bidirectional=False)
        self.decoder_cell = nn.LSTMCell(emb_dim + hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_proj = nn.Linear(hid_dim + hid_dim, hid_dim)
        self.out = nn.Linear(hid_dim, dec_vocab)
    def forward(self, enc_input, dec_input=None, max_len=50):
        B = enc_input.size(0)
        enc_emb = self.enc_emb(enc_input)
        enc_out, (h_n, c_n) = self.encoder(enc_emb)
        dec_h = h_n[-1]
        dec_c = c_n[-1]
        outputs = []
        if dec_input is None:
            bos_ids = torch.zeros((B,), dtype=torch.long, device=enc_input.device)
            emb_prev = self.dec_emb(bos_ids)
        seq_len = (dec_input.size(1) if dec_input is not None else max_len)
        for t in range(seq_len):
            if dec_input is not None:
                emb_t = self.dec_emb(dec_input[:, t])
            else:
                emb_t = emb_prev
            scores = torch.bmm(enc_out, dec_h.unsqueeze(2)).squeeze(2)
            attn_weights = torch.softmax(scores, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_out).squeeze(1)
            cell_in = torch.cat([emb_t, context], dim=1)
            dec_h, dec_c = self.decoder_cell(cell_in, (dec_h, dec_c))
            proj = torch.tanh(self.attn_proj(torch.cat([dec_h, context], dim=1)))
            proj = self.dropout(proj)
            logits_t = self.out(proj)
            outputs.append(logits_t.unsqueeze(1))
            if dec_input is None:
                top1 = logits_t.argmax(dim=1)
                emb_prev = self.dec_emb(top1)
        logits = torch.cat(outputs, dim=1)
        return logits

# ---------------------------
# Loading everything (tokenizers, w2v, model, dataset index)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    code_tok, code_idtok, code_decode, code_encode = load_bpe_info(BPE_CODE_PKL)
    doc_tok, doc_idtok, doc_decode, doc_encode = load_bpe_info(BPE_DOC_PKL)
    w2v_code = safe_load_pickle(W2V_CODE_PKL)
    w2v_doc  = safe_load_pickle(W2V_DOC_PKL)
    tokenized_df = None
    if os.path.exists(DATA_SAMPLE_PKL):
        try:
            tokenized_df = pd.read_pickle(DATA_SAMPLE_PKL)
        except Exception:
            tokenized_df = None
    model = None
    model_meta = {}
    if model is None and os.path.exists(MODEL_STATE_PATH):
        enc_vocab = (max(code_tok.values())+1) if code_tok else 3000
        dec_vocab = (max(doc_tok.values())+1) if doc_tok else 3000
        emb_enc = None; emb_dec = None
        if w2v_code and "W_in" in w2v_code:
            W_in = np.array(w2v_code["W_in"])
            emb_enc = np.random.normal(scale=0.01, size=(enc_vocab, W_in.shape[1])).astype(np.float32)
        if w2v_doc and "W_in" in w2v_doc:
            W_in2 = np.array(w2v_doc["W_in"])
            emb_dec = np.random.normal(scale=0.01, size=(dec_vocab, W_in2.shape[1])).astype(np.float32)
        model = Seq2SeqAttention(enc_vocab, dec_vocab, emb_dim=emb_enc.shape[1] if emb_enc is not None else 128,
                                 hid_dim=256, emb_enc=emb_enc, emb_dec=emb_dec, dropout=0.2,
                                 num_layers=1, PAD_ENC=code_tok.get("<PAD>",0) if code_tok else 0,
                                 PAD_DEC=doc_tok.get("<PAD>",0) if doc_tok else 0)
        try:
            state = torch.load(MODEL_STATE_PATH, map_location=DEVICE)
            model.load_state_dict(state)
            model.to(DEVICE)
            model_meta['loaded_state_dict'] = True
        except Exception as e:
            model = None
            model_meta['state_load_error'] = str(e)
    else:
        model_meta['loaded_state_dict'] = False
    try:
        model_meta['base_files'] = os.listdir(BASE_PATH)
    except Exception:
        model_meta['base_files'] = None
    return {
        "code_tok": code_tok, "code_idtok": code_idtok, "code_decode": code_decode, "code_encode": code_encode,
        "doc_tok": doc_tok,   "doc_idtok": doc_idtok,   "doc_decode": doc_decode,   "doc_encode": doc_encode,
        "w2v_code": w2v_code, "w2v_doc": w2v_doc,
        "model": model, "model_meta": model_meta, "tokenized_df": tokenized_df
    }

def average_w2v_for_tokens(token_ids, w2v):
    if w2v is None or "W_in" not in w2v: return None
    W = np.array(w2v["W_in"])
    map_word2id = w2v.get("word_to_id") or w2v.get("id_to_word") or {}
    vecs = []
    for t in token_ids:
        if isinstance(map_word2id, dict) and t in map_word2id:
            idx = map_word2id[t]
            if 0 <= int(idx) < W.shape[0]:
                vecs.append(W[int(idx)])
        else:
            try:
                if isinstance(t, int) and t < W.shape[0]:
                    vecs.append(W[int(t)])
            except Exception:
                pass
    if not vecs:
        return None
    return np.mean(vecs, axis=0)

def retrieve_similar_examples(avg_vec, tokenized_df, w2v, top_k=3):
    if avg_vec is None or tokenized_df is None or w2v is None: return []
    sample_embs = []
    for i, row in tokenized_df.iterrows():
        code_ids = row.get("code_token_ids") or row.get("code_tokens") or []
        emb = average_w2v_for_tokens(code_ids, w2v)
        sample_embs.append(emb)
    sample_embs_arr = np.array([e for e in sample_embs if e is not None])
    if sample_embs_arr.size == 0: return []
    sims = []
    for e in sample_embs:
        if e is None: sims.append(-1.0)
        else:
            sims.append(float(np.dot(avg_vec, e) / (np.linalg.norm(avg_vec) * (np.linalg.norm(e) + 1e-9) + 1e-9)))
    top_idx = np.argsort(sims)[-top_k:][::-1]
    results = []
    for idx in top_idx:
        row = tokenized_df.iloc[int(idx)]
        results.append({
            "func_name": row.get("func_name"),
            "repo": row.get("repo"),
            "docstring": row.get("docstring"),
            "summary": row.get("summary"),
            "similarity": sims[int(idx)]
        })
    return results

def encode_input_text(text, artifacts, max_len=256):
    code_tok = artifacts["code_tok"]
    code_encode = artifacts["code_encode"]
    if code_encode is not None:
        try:
            ids = code_encode(text)
            return ids[:max_len]
        except Exception:
            pass
    return fallback_encode(text, code_tok, max_len=max_len)

def greedy_seq2seq_generate_local(model, enc_ids, artifacts, max_len=256):
    model.eval()
    code_decode = artifacts["code_decode"]
    doc_decode = artifacts["doc_decode"]
    enc = torch.tensor([enc_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(enc, dec_input=None, max_len=max_len)  # (1,L,V)
        ids = torch.argmax(logits, dim=2).cpu().numpy().tolist()[0]
    EOS = artifacts["doc_tok"].get("<EOS>") if artifacts["doc_tok"] else None
    if EOS is not None:
        if EOS in ids:
            ids = ids[:ids.index(EOS)]
    if doc_decode:
        return doc_decode(ids)
    else:
        return " ".join(map(str, ids))

# -------------- UI Design CSS Variables --------------
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        /* Base Styling */
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif !important;
        }

        /* Headers */
        h1 {
            background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
            font-size: 3rem !important;
            letter-spacing: -1px;
            margin-bottom: 0.5rem;
            text-align: center;
        }

        /* Glassmorphism Containers */
        .glass-container {
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 1rem;
            padding: 2rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            margin-bottom: 2rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4);
        }

        /* Buttons */
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
            color: white !important;
            border: none;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            border-radius: 9999px; /* full rounded */
            box-shadow: 0 4px 14px 0 rgba(129, 140, 248, 0.39);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(129, 140, 248, 0.5);
            background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 100%);
        }
        </style>
        """, unsafe_allow_html=True)

# -------------- Main UI Function --------------
def main():
    st.set_page_config(page_title="AutoDocGen AI", layout="wide")
    inject_custom_css()

    st.markdown("<h1>✨ AutoDocGen AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Automated Code Documentation using BPE, Word2Vec, and Seq2Seq with Self-Attention</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h3 style='color: white;'>⚙️ Generation Settings</h3>", unsafe_allow_html=True)
        gen_type = st.selectbox("Documentation Type", ["Short Summary", "Full Docstring"])
        max_len = st.slider("Max Length (tokens)", min_value=30, max_value=400, value=128)
        use_context = st.checkbox("Enable Context Retrieval (Word2Vec)", value=True)
        top_k_context = st.slider("Nearest Neighbors", 1, 5, 2)
        st.markdown("---")
        st.markdown("<p style='font-size: 0.8rem; color: #64748b; text-align: center;'>Powered by Streamlit</p>", unsafe_allow_html=True)

    with st.spinner("Loading AI Models & Context Spaces..."):
        artifacts = load_artifacts()
    
    model = artifacts.get("model")
    if model is None:
        st.error("⚠️ AI Models not found or failed to load. Check model artifacts paths.")
        st.stop()

    left_col, right_col = st.columns([1.2, 1])

    # Variables for snippet tracking
    func_choice = None
    code_input = ""

    with left_col:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.markdown("### 💻 Source Code", unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Upload Python File (.py)", type=["py"])
        
        if uploaded:
            code_input = uploaded.read().decode("utf8")
        else:
            code_input = st.text_area("Or copy and paste your function code below:", height=250, placeholder="def calculate_total_amount(cart):\n    return sum(item.price for item in cart)")

        # Auto-extract logic
        funcs = []
        if code_input:
            try:
                parsed = ast.parse(code_input)
                for node in parsed.body:
                    if isinstance(node, ast.FunctionDef):
                        src = ast.get_source_segment(code_input, node) or ast.unparse(node)
                        funcs.append((node.name, src))
            except Exception:
                pass

        if funcs:
            names = [f[0] for f in funcs]
            idx = st.selectbox("Select function to analyze:", range(len(names)), format_func=lambda i: names[i])
            func_choice = funcs[idx][1]
            st.code(func_choice, language="python")

        col1, col2 = st.columns(2)
        with col1:
            generate_btn = st.button("🚀 Generate Docstring")
        with col2:
            clear_btn = st.button("🗑️ Clear Input")
            if clear_btn:
                st.query_params.clear()

        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.markdown("### 📄 Generated Output", unsafe_allow_html=True)
        
        output_placeholder = st.empty()
        
        if generate_btn:
            snippet = func_choice or code_input
            if not snippet.strip():
                st.warning("Please provide some code to generate documentation for.")
            else:
                with st.spinner("Synthesizing documentation..."):
                    enc_ids = encode_input_text(snippet, artifacts, max_len=256)
                    context_docstrings = []
                    
                    if use_context and artifacts.get("w2v_code") is not None and artifacts.get("tokenized_df") is not None:
                        avg_vec = average_w2v_for_tokens(enc_ids, artifacts["w2v_code"])
                        sims = retrieve_similar_examples(avg_vec, artifacts["tokenized_df"], artifacts["w2v_code"], top_k=top_k_context)
                        for s in sims:
                            context_docstrings.append(s.get("docstring") or s.get("summary") or "")

                    context_concat = "\n\n".join([snippet] + context_docstrings) if context_docstrings else snippet
                    enc_ids_final = encode_input_text(context_concat, artifacts, max_len=400)

                    t0 = time.time()
                    out_text = greedy_seq2seq_generate_local(model, enc_ids_final, artifacts, max_len=max_len)
                    t1 = time.time()

                    st.success(f"Generation completed in {t1-t0:.2f} seconds!")
                    st.code(out_text, language="python")
                    st.download_button("📥 Download Result", out_text, file_name="generated_docstring.txt", mime="text/plain")
                    
                    if context_docstrings:
                        with st.expander("🔍 Show Context Used (Semantic Neighbors)"):
                            for idx, c in enumerate(context_docstrings):
                                st.markdown(f"**Neighbor {idx+1}:**")
                                st.code(c, language="python")
        else:
            st.info("Awaiting input... Provide your code on the left and hit 'Generate Docstring'.")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
