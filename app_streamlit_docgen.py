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
# Minimal copy of Seq2SeqAttention used during training
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
        Le = enc_out.size(1)
        if dec_input is None:
            # BOS handling done by caller
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
    # 1) tokenizers
    code_tok, code_idtok, code_decode, code_encode = load_bpe_info(BPE_CODE_PKL)
    doc_tok, doc_idtok, doc_decode, doc_encode = load_bpe_info(BPE_DOC_PKL)
    # 2) w2v
    w2v_code = safe_load_pickle(W2V_CODE_PKL)
    w2v_doc  = safe_load_pickle(W2V_DOC_PKL)
    # 3) dataset index (tokenized_sample.pkl) for context retrieval
    tokenized_df = None
    if os.path.exists(DATA_SAMPLE_PKL):
        try:
            tokenized_df = pd.read_pickle(DATA_SAMPLE_PKL)
        except Exception:
            tokenized_df = None
    # 4) Model: DO NOT attempt to load full model pickle; build model & load state_dict
    model = None
    model_meta = {}
    # NOTE: intentionally skip loading FULL_MODEL_PATH to avoid torch unpickling of custom classes
    if model is None and os.path.exists(MODEL_STATE_PATH):
        # reconstruct architecture using tokenizers sizes
        enc_vocab = (max(code_tok.values())+1) if code_tok else 3000
        dec_vocab = (max(doc_tok.values())+1) if doc_tok else 3000
        emb_enc = None; emb_dec = None
        # If w2v available, build embedding matrices (best-effort)
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
            # if state_dict load fails, mark and leave model None
            model = None
            model_meta['state_load_error'] = str(e)
    else:
        model_meta['loaded_state_dict'] = False
    # include a listing of files in BASE_PATH for diagnostics
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

# -------------- Context retrieval helpers using W2V --------------
def average_w2v_for_tokens(token_ids, w2v):
    # w2v expected structure: {"W_in": array, "word_to_id": {...}} or similar
    if w2v is None or "W_in" not in w2v: return None
    W = np.array(w2v["W_in"])
    map_word2id = w2v.get("word_to_id") or w2v.get("id_to_word") or {}
    vecs = []
    for t in token_ids:
        # try several lookups
        if isinstance(map_word2id, dict) and t in map_word2id:
            idx = map_word2id[t]
            if 0 <= int(idx) < W.shape[0]:
                vecs.append(W[int(idx)])
        else:
            # fallback: if token is int and maps directly:
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
    # precompute sample embeddings (cacheable in production)
    sample_embs = []
    for i, row in tokenized_df.iterrows():
        code_ids = row.get("code_token_ids") or row.get("code_tokens") or []
        emb = average_w2v_for_tokens(code_ids, w2v)
        if emb is None:
            sample_embs.append(None)
        else:
            sample_embs.append(emb)
    sample_embs_arr = np.array([e for e in sample_embs if e is not None])
    if sample_embs_arr.size == 0: return []
    # compute cosine similarities — this is simplified; in practice you'd pre-index
    sims = []
    for i,e in enumerate(sample_embs):
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

# -------------- Generation functions -------------------------------
def encode_input_text(text, artifacts, max_len=256):
    code_tok = artifacts["code_tok"]
    code_encode = artifacts["code_encode"]
    if code_encode is not None:
        try:
            ids = code_encode(text)
            return ids[:max_len]
        except Exception:
            pass
    # fallback: whitespace -> token ids
    return fallback_encode(text, code_tok, max_len=max_len)

# removed @st.cache_data decorator to avoid UnhashableParamError from Streamlit caching
def greedy_seq2seq_generate_local(model, enc_ids, artifacts, max_len=256):
    model.eval()
    code_decode = artifacts["code_decode"]
    doc_decode = artifacts["doc_decode"]
    PAD_DEC = artifacts["doc_tok"].get("<PAD>", 0) if artifacts["doc_tok"] else 0
    BOS_DEC = artifacts["doc_tok"].get("<BOS>", None) if artifacts["doc_tok"] else None
    enc = torch.tensor([enc_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        # if model is Seq2SeqAttention we can call .forward with dec_input=None
        logits = model(enc, dec_input=None, max_len=max_len)  # (1,L,V)
        ids = torch.argmax(logits, dim=2).cpu().numpy().tolist()[0]
    # trim at EOS if present
    EOS = artifacts["doc_tok"].get("<EOS>") if artifacts["doc_tok"] else None
    if EOS is not None:
        if EOS in ids:
            ids = ids[:ids.index(EOS)]
    if doc_decode:
        return doc_decode(ids)
    else:
        return " ".join(map(str, ids))

# -------------- Streamlit UI (aesthetic layout, no functionality change) ---------------------------
def main():
    st.set_page_config(page_title="DocGen - Integrated System", layout="wide")
    # tiny CSS to reduce vertical spacing a bit for a denser layout
    st.markdown(
        """
        <style>
          .stButton>button { padding: .375rem .75rem; }
          .css-1d391kg {padding-top: .5rem;} /* smaller top padding for header area */
          .block-container { padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Integrated Documentation Generation System (BPE + Word2Vec + Seq2Seq)")

    # --- Sidebar: generation options (keeps main page compact) ---
    with st.sidebar:
        st.header("Generation options")
        gen_type = st.selectbox("What to generate", ["Short summary", "Full docstring"])
        max_len = st.slider("Max generated length (tokens)", min_value=30, max_value=400, value=128)
        # Beam search removed — always use Greedy
        use_context = st.checkbox("Use context retrieval (Word2Vec)", value=True)
        top_k_context = st.slider("Nearest neighbors", 1, 5, 2)
        st.markdown("---")

    # load artifacts
    with st.spinner("Loading models & tokenizers..."):
        artifacts = load_artifacts()
    model = artifacts.get("model")
    # If model failed to load, show diagnostic info (helps debug on Streamlit Cloud)
    if model is None:
        st.error("Model not found or failed to load. Check MODEL_STATE_PATH or FULL_MODEL_PATH.")
        meta = artifacts.get("model_meta", {})
        if meta:
            st.write("model_meta:", {k: v for k, v in meta.items() if k != 'state_load_error'})
            if meta.get("state_load_error"):
                st.write("State load error:", meta.get("state_load_error"))
        try:
            files = os.listdir(BASE_PATH)
            st.write("Files in", BASE_PATH, ":", files)
        except Exception as e:
            st.write("Could not list BASE_PATH:", str(e))
        return

    # Main layout: left = input, right = output & context (compact)
    left, right = st.columns([2, 1])

    # LEFT: input & controls
    with left:
        st.subheader("Input function")
        code_input = st.text_area("Paste function code (or upload .py)", height=200, key="code_input")
        uploaded = st.file_uploader("Upload a .py file (optional)", type=["py"])
        if uploaded and not code_input:
            try:
                raw = uploaded.read().decode("utf8")
                code_input = raw
                st.session_state["code_input"] = raw
            except Exception:
                pass

        # Try to auto-extract functions
        funcs = []
        if code_input:
            try:
                parsed = ast.parse(code_input)
                for node in parsed.body:
                    if isinstance(node, ast.FunctionDef):
                        src = ast.get_source_segment(code_input, node) or ast.unparse(node)
                        funcs.append((node.name, src))
            except Exception:
                funcs = []

        func_choice = None
        if funcs:
            names = [f[0] for f in funcs]
            idx = st.selectbox("Select function (extracted)", range(len(names)), format_func=lambda i: names[i])
            func_choice = funcs[idx][1]
            st.code(func_choice, language="python")
        else:
            st.info("No function automatically extracted — paste a single function or upload a .py file.")

        # Generate button (keeps everything above fold)
        gen_col1, gen_col2 = st.columns([1, 1])
        with gen_col1:
            generate_btn = st.button("Generate documentation")
        with gen_col2:
            clear_btn = st.button("Clear input")
            if clear_btn:
                st.session_state["code_input"] = ""
                code_input = ""

    # RIGHT: output + context (collapsed sections)
    with right:
        out_placeholder = st.empty()

        # show a small header & download area (will be populated on generation)
        out_placeholder.info("Generated documentation will appear here.")

        with st.expander("Context / nearest neighbors", expanded=False):
            st.write("Nearest neighbor docstrings (if context retrieval is enabled):")
            st.write("(Will populate after generation)")

        with st.expander("Quick diagnostics", expanded=False):
            meta = artifacts.get("model_meta", {})
            st.write("Files in model_artifacts:", meta.get("base_files"))
            st.write("State dict loaded:", meta.get("loaded_state_dict", False))

    # Handle generation when button clicked
    if generate_btn:
        snippet = func_choice or code_input
        if not snippet:
            st.error("No code provided.")
        else:
            # Encode
            enc_ids = encode_input_text(snippet, artifacts, max_len=256)

            # Context retrieval
            context_docstrings = []
            if use_context and artifacts.get("w2v_code") is not None and artifacts.get("tokenized_df") is not None:
                avg_vec = average_w2v_for_tokens(enc_ids, artifacts["w2v_code"])
                sims = retrieve_similar_examples(avg_vec, artifacts["tokenized_df"], artifacts["w2v_code"], top_k=top_k_context)
                for s in sims:
                    context_docstrings.append(s.get("docstring") or s.get("summary") or "")

            context_concat = "\n\n".join([snippet] + context_docstrings) if context_docstrings else snippet
            enc_ids_final = encode_input_text(context_concat, artifacts, max_len=400)

            # Run generation (always Greedy now)
            st.info("Running generation on model (this may take a few seconds)...")
            t0 = time.time()
            out_text = greedy_seq2seq_generate_local(model, enc_ids_final, artifacts, max_len=max_len)
            t1 = time.time()

            # Update right column with results (compact)
            with right:
                st.subheader("Generated Documentation")
                st.code(out_text, language=None)
                dl_col1, dl_col2 = st.columns([1, 3])
                with dl_col1:
                    st.download_button("Download .txt", out_text, file_name="generated_docstring.txt")
                with dl_col2:
                    st.success(f"Generated in {t1-t0:.2f}s")

                if context_docstrings:
                    with st.expander("Context used (nearest neighbors)", expanded=False):
                        for c in context_docstrings:
                            st.write(c)

    # Validation examples hidden by default to save space
    st.markdown("---")
    with st.expander("Show validation examples (generate for samples)", expanded=False):
        if artifacts.get("tokenized_df") is None:
            st.info("No tokenized_sample found in DATA_SAMPLE_PKL.")
        else:
            sample_df = artifacts["tokenized_df"].sample(min(6, len(artifacts["tokenized_df"])))
            for i, r in sample_df.iterrows():
                st.markdown(f"**Function:** {r.get('func_name', 'unknown')}")
                st.code(r.get("code", "")[:400], language="python")
                if st.button(f"Generate for sample {i}", key=f"gen_{i}"):
                    enc_ids = r.get("code_token_ids") or r.get("code_tokens") or encode_input_text(r.get("code",""), artifacts)
                    out_text = greedy_seq2seq_generate_local(model, enc_ids, artifacts)
                    st.code(out_text)

if __name__ == "__main__":
    main()
